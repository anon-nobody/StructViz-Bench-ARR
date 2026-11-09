"""Resumable, shardable evaluation of LOCAL models over pre-rendered image suites.

Reads a manifest of pre-rendered images (one row per (question_id, viz_type)),
joins it with the benchmark JSONL for question/answer/source/difficulty, runs
the local model, scores with ``compute_metrics`` exactly like
``scripts/run_fullscale_eval.py::compute_row``, and appends result rows.

Output rows carry the ``results/full_qwen.jsonl`` schema plus provenance
fields (raw_response, model_id, model_revision, image_path, image_sha256,
image_w, image_h, prompt_sha256, suite, timestamp_utc, latency_s).

Examples:
    # One shard per GPU (4 GPUs):
    python scripts/eval_local_suite.py --model qwen --manifest suites/v2/manifest.jsonl \
        --output results/suite_v2/qwen.shard0.jsonl --gpu 0 --shard 0/4
    # No-image (question text only) baseline:
    python scripts/eval_local_suite.py --model qwen --no-image \
        --output results/noimage_qwen.jsonl --gpu 1
    # Raw-data text baseline (serialized data + question, no image):
    python scripts/eval_local_suite.py --model qwen --raw-text \
        --output results/v2/rawtext_qwen.jsonl --gpu 1
    # Merge:
    python scripts/merge_shards.py results/suite_v2/qwen.shard*.jsonl \
        --output results/suite_v2/qwen.jsonl
"""

from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

import argparse
import hashlib
import io
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NO_IMAGE_VIZ = "none"
RAW_TEXT_VIZ = "raw_text"

# Pinned HF revisions (commit hashes of the locally cached snapshots).
PINNED_REVISIONS: dict[str, str] = {
    "qwen32b-hf": "7cfb30d71a1f4f49a57592323337a4a4727301da",
    "internvl": "e9e4c0dc1db56bfab10458671519b7fa3dd29463",
    "qwen3vl8b": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    "internvl2-4b": "6eb21ec278fbdb03b751a3d48d8dfc9724e05608",
    "internvl2-8b": "6fb9ad6924f69424e57fab2ab061d707688f0296",
}

MODEL_IDS: dict[str, str] = {
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen32b": "Qwen/Qwen2.5-VL-32B-Instruct",
    "qwen32b-hf": "Qwen/Qwen2.5-VL-32B-Instruct",
    "internvl": "OpenGVLab/InternVL2_5-8B",
    "qwen3vl8b": "Qwen/Qwen3-VL-8B-Instruct",
    "internvl2-4b": "OpenGVLab/InternVL2-4B",
    "internvl2-8b": "OpenGVLab/InternVL2-8B",
}
INTERNVL_KEYS = ("internvl", "internvl2-4b", "internvl2-8b")
# Qwen-family keys that share the QwenVLModel prompting / chat-template path.
QWEN_KEYS = ("qwen", "qwen32b", "qwen32b-hf", "qwen3vl8b")
# Keys that use the Qwen2.5-VL (transformers 4.49) vision tower.
QWEN25_KEYS = ("qwen", "qwen32b", "qwen32b-hf")

# --cot system prompts. "final_answer": the revision CoT prompt (same answer-format rules as
# the default prompt). "ablation_v1": the verbatim CoT prompt of the original prompt-ablation
# (configs/ablation_prompt_sensitivity.yaml), for auditing that run's answer extraction.
COT_SYSTEM_PROMPTS: dict[str, str] = {
    "final_answer": (
        "You are a precise visual data analyst. "
        "Look at the image and answer the question. "
        "Think step by step, then end with a line `Final answer: <answer>`.\n"
        "\n"
        "Answer format rules for <answer>:\n"
        "- For yes/no questions: answer exactly 'yes' or 'no'\n"
        "- For numeric questions: answer with just the number (e.g. 48.17)\n"
        "- For name/label questions: answer with just the name (e.g. east_grid)\n"
        "- For direction questions: answer with just the word (e.g. increasing, higher, "
        "first half)\n"
        "- If no clear answer exists: answer 'none'\n"
        "- No units, no sentence, no explanation after 'Final answer:'\n"
    ),
    "ablation_v1": (
        "Look at the visualization and think step by step. After your reasoning, provide the "
        "final answer on a new line starting with 'Answer:'"
    ),
}
COT_USER_SUFFIXES: dict[str, str] = {
    "final_answer": "\n\nThink step by step, then end with a line `Final answer: <answer>`.",
    "ablation_v1": ("\n\nThink step by step. After your reasoning, provide the final answer on "
                    "a new line starting with 'Answer:'"),
}
COT_MARKERS: dict[str, str] = {
    "final_answer": r"final\s+answer\s*[:\uff1a]",
    "ablation_v1": r"(?:final\s+)?answer\s*[:\uff1a]",
}

# --transcribe-then-answer stage-1 prompts (per modality) and system prompt.
TRANSCRIBE_SYSTEM_PROMPT = "You are a precise data transcriber."
TRANSCRIBE_PROMPTS: dict[str, str] = {
    "tabular": (
        "Transcribe ALL data shown in this image as plain text: a CSV table with a header row, "
        "then one line per data row, including every row and every column shown. Output only "
        "the CSV, no commentary."
    ),
    "timeseries": (
        "Transcribe ALL data shown in this image as plain text: one `index,value` line per data "
        "point, in order, for every point of the series. Output only these lines, no commentary."
    ),
    "graph": (
        "Transcribe ALL data shown in this image as plain text: first one line `nodes: ` "
        "followed by a comma-separated list of every node, then one `u,v` line per edge for "
        "every edge shown. Output only these lines, no commentary."
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, choices=sorted(MODEL_IDS))
    p.add_argument("--manifest", type=Path, default=None,
                   help="Manifest JSONL (question_id, modality, task, viz_type, image_path).")
    p.add_argument("--benchmark", type=Path, default=Path("benchmark/realworld_test.jsonl"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=str, default=None,
                   help="Physical GPU index or comma list, e.g. 0,2 (sets CUDA_VISIBLE_DEVICES "
                        "before torch import; a list spreads the model with device_map=auto).")
    p.add_argument("--headroom-gb", type=float, default=4.0,
                   help="qwen32b-hf: max_memory per GPU = free memory at load time minus this.")
    p.add_argument("--shard", type=str, default="0/1", help="i/n: keep rows with index %% n == i.")
    p.add_argument("--no-image", action="store_true",
                   help="One row per question_id with viz_type='none'; question text only.")
    p.add_argument("--raw-text", action="store_true",
                   help="One row per question_id with viz_type='raw_text'; no image, the user "
                        "turn is the serialized raw data followed by the question.")
    p.add_argument("--suite", type=str, default=None,
                   help="Suite label (default: manifest row 'suite', else manifest parent dir).")
    p.add_argument("--limit", type=int, default=None, help="Process at most N pending rows.")
    p.add_argument("--checkpoint-every", type=int, default=50,
                   help="Flush+fsync output every N rows.")
    p.add_argument("--log", type=Path, default=None, help="Also append log lines to this file.")
    p.add_argument("--no-vision-mask-patch", action="store_true",
                   help="Disable the (bit-identical) Qwen2.5-VL vision mask sync fix.")
    p.add_argument("--vision-attn-chunk", type=int, default=1024,
                   help="Query-chunk size for Qwen vision attention (0=off). Caps memory for "
                        "large images (e.g. 1024x2048); not guaranteed bit-identical.")
    p.add_argument("--resume", action="store_true",
                   help="No-op: runs always resume from --output (skip done keys).")
    p.add_argument("--allow-offload", action="store_true",
                   help="Do not abort if device_map='auto' offloads layers to CPU/disk.")
    p.add_argument("--max-retries", type=int, default=2,
                   help="Retries per row on exceptions (e.g. transient CUDA OOM).")
    p.add_argument("--internvl-max-tiles", type=int, default=None,
                   help="InternVL: official dynamic 448px tiling with up to N tiles (+ thumbnail "
                        "when N>1). Default: the existing single 448x448 resize.")
    p.add_argument("--internvl-quant", choices=["nf4", "bf16"], default="nf4",
                   help="InternVL weights: nf4 (existing 4-bit path, default) or bf16.")
    p.add_argument("--transcribe-then-answer", action="store_true",
                   help="Qwen: stage 1 transcribes the image to plain text; stage 2 answers from "
                        "the transcript alone (no image).")
    p.add_argument("--transcribe-max-new-tokens", type=int, default=4096,
                   help="Stage-1 generation budget for --transcribe-then-answer.")
    p.add_argument("--cot", action="store_true",
                   help="Qwen: chain-of-thought prompt; prediction = text after the last answer "
                        "marker; also stores the legacy extraction.")
    p.add_argument("--cot-prompt", choices=sorted(COT_SYSTEM_PROMPTS), default="final_answer",
                   help="CoT system prompt: revision prompt or the original ablation wording.")
    p.add_argument("--cot-user-suffix", action="store_true",
                   help="--cot: also append the CoT instruction to the user turn (after the "
                        "question); improves compliance of models that ignore system prompts.")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="Answer generation budget (default: model default 256; 1024 with --cot).")
    return p.parse_args()


_LOG_FH: Any = None


def log(msg: str) -> None:
    """Timestamped log to stdout (and --log file)."""
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL, skipping blank and truncated (unparseable) lines."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                log(f"WARNING: skipping unparseable line {n} in {path}")
    return rows


def parse_shard(spec: str) -> tuple[int, int]:
    """Parse 'i/n' into (i, n)."""
    i_str, n_str = spec.split("/")
    i, n = int(i_str), int(n_str)
    if not (n >= 1 and 0 <= i < n):
        raise ValueError(f"Bad --shard {spec!r}")
    return i, n


def _fmt_cell(v: Any) -> str:
    """Cell text for the raw-text CSV: values as stored (None -> empty)."""
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def serialize_raw_data(item: dict[str, Any]) -> str:
    """Serialize a benchmark item's ``data`` field as plain text (no truncation).

    tabular (list of row dicts) -> CSV with header and a leading ``row`` index column;
    timeseries (list of floats) -> one ``t=<i>: <value>`` line per point;
    graph (node_link dict) -> ``nodes: 0..N-1`` then one ``u -- v`` line per edge in
    stored order (node/edge attributes are not included).
    """
    import csv

    modality = str(item["modality"])
    data = item["data"]
    if modality == "tabular":
        cols: list[str] = []
        for r in data:
            for c in r:
                if c not in cols:
                    cols.append(c)
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["row", *cols])
        for i, r in enumerate(data):
            w.writerow([i, *(_fmt_cell(r.get(c)) for c in cols)])
        return buf.getvalue().rstrip("\n")
    if modality == "timeseries":
        return "\n".join(f"t={i}: {v}" for i, v in enumerate(data))
    if modality == "graph":
        ids = sorted(int(n["id"]) for n in data["nodes"])
        if ids != list(range(len(ids))):
            raise ValueError(f"graph node ids are not 0..N-1: {item['question_id']}")
        lines = [f"nodes: 0..{len(ids) - 1}"]
        lines += [f"{e['source']} -- {e['target']}" for e in data["edges"]]
        return "\n".join(lines)
    raise ValueError(f"unknown modality {modality!r}")


def raw_text_message(item: dict[str, Any]) -> str:
    """User-turn text for --raw-text: serialized data, blank line, question."""
    return f"{serialize_raw_data(item)}\n\nQuestion: {item['question']}"


def resolve_image_path(raw: str, manifest_path: Path | None) -> Path:
    """Resolve manifest image_path: absolute, project-relative, or manifest-relative."""
    p = Path(raw)
    if p.is_absolute():
        return p
    if (PROJECT_ROOT / p).exists():
        return PROJECT_ROOT / p
    if manifest_path is not None and (manifest_path.parent / p).exists():
        return manifest_path.parent / p
    return PROJECT_ROOT / p


def hf_revision(repo_id: str) -> str | None:
    """Snapshot hash of the locally cached HF repo, if resolvable."""
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(repo_id, "config.json")
        if isinstance(cached, str):
            parts = Path(cached).parts
            if "snapshots" in parts:
                return parts[parts.index("snapshots") + 1]
    except Exception:  # noqa: BLE001
        pass
    return None


def free_memory_caps(headroom_gb: float) -> dict[int, int]:
    """max_memory per visible GPU: current free bytes minus headroom (no CPU entry)."""
    import torch

    caps: dict[int, int] = {}
    for i in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(i)
        caps[i] = max(0, int(free - headroom_gb * 2**30))
    return caps


def build_qwen_hf_sharded(checkpoint: str, revision: str, max_memory: dict[Any, int]) -> Any:
    """QwenVLModel with an HF bf16 load across several GPUs (device_map=auto + max_memory).

    Only ``_load`` differs from QwenVLModel; prompting, greedy decoding, parsing and the
    raw-text path are inherited unchanged.
    """
    from dataclasses import dataclass, field
    import importlib

    from src.models.local_models import QwenVLModel

    @dataclass(slots=True)
    class QwenVLModelSharded(QwenVLModel):
        revision: str | None = None
        max_memory: dict[Any, int] = field(default_factory=dict)

        def _load(self) -> None:
            tf = importlib.import_module("transformers")
            torch = importlib.import_module("torch")
            self._processor = tf.AutoProcessor.from_pretrained(
                self.checkpoint, revision=self.revision)
            self._model = tf.Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.checkpoint,
                revision=self.revision,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory=self.max_memory or None,
            )
            self._model.eval()

    return QwenVLModelSharded(
        name=checkpoint.split("/")[-1], checkpoint=checkpoint, device="cuda",
        revision=revision, max_memory=max_memory,
    )


def build_qwen3vl(checkpoint: str, revision: str) -> Any:
    """QwenVLModel subclass for Qwen3-VL (needs transformers>=4.57, e.g. PYTHONPATH=.tf_new).

    Prompting (system prompt + [image, text] user turn, chat template) and ``parse_answer`` are
    inherited from QwenVLModel; only loading and the HF generate call differ: the image goes
    straight to the Qwen3-VL processor (its own smart-resize, patch 16, merge 2), greedy decoding.
    """
    from dataclasses import dataclass
    import importlib

    from src.models.local_models import QwenVLModel

    @dataclass(slots=True)
    class Qwen3VLModel(QwenVLModel):
        revision: str | None = None
        last_geometry: dict[str, int] | None = None

        def _load(self) -> None:
            tf = importlib.import_module("transformers")
            torch = importlib.import_module("torch")
            self._processor = tf.AutoProcessor.from_pretrained(
                self.checkpoint, revision=self.revision)
            self._model = tf.Qwen3VLForConditionalGeneration.from_pretrained(
                self.checkpoint, revision=self.revision, torch_dtype=torch.bfloat16,
                device_map="auto", attn_implementation="sdpa",
            )
            self._model.eval()

        def _generate_raw(self, question: str, image: object | None) -> str:
            if self._model is None:
                self._load()
            torch = importlib.import_module("torch")
            messages = self.build_messages(question, image)
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            kwargs: dict[str, Any] = {"text": [text], "padding": True, "return_tensors": "pt"}
            if image is not None:
                kwargs["images"] = [image.convert("RGB")]  # type: ignore[attr-defined]
            inputs = self._processor(**kwargs).to(self._model.device)
            self.last_geometry = None
            if image is not None and "image_grid_thw" in inputs:
                _t, gh, gw = (int(x) for x in inputs["image_grid_thw"][0].tolist())
                patch = int(self._processor.image_processor.patch_size)
                merge = int(self._processor.image_processor.merge_size)
                self.last_geometry = {"model_input_w": gw * patch, "model_input_h": gh * patch,
                                      "n_visual_tokens": (gh * gw) // (merge * merge)}
            with torch.inference_mode():
                generated = self._model.generate(
                    **inputs, max_new_tokens=self.max_tokens, do_sample=False,
                    temperature=None, top_p=None, top_k=None,
                )
            trimmed = generated[:, inputs["input_ids"].shape[1]:]
            return str(self._processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

    return Qwen3VLModel(name=checkpoint.split("/")[-1], checkpoint=checkpoint, device="cuda",
                        revision=revision)


def _internvl_build_transform(input_size: int) -> Any:
    """Official InternVL preprocessing transform (RGB, bicubic resize, ImageNet norm)."""
    import torchvision.transforms as T  # noqa: N812
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def _internvl_closest_ratio(aspect_ratio: float, target_ratios: list[tuple[int, int]],
                            width: int, height: int, image_size: int) -> tuple[int, int]:
    """``find_closest_aspect_ratio`` from the official InternVL model card."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def internvl_dynamic_preprocess(image: Any, max_num: int, image_size: int = 448,
                                use_thumbnail: bool = True, min_num: int = 1) -> list[Any]:
    """``dynamic_preprocess`` from the official InternVL model card (tiles + thumbnail)."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1)
         for j in range(1, n + 1) if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1])
    ratio = _internvl_closest_ratio(aspect_ratio, target_ratios, orig_width, orig_height,
                                    image_size)
    target_width, target_height = image_size * ratio[0], image_size * ratio[1]
    blocks = ratio[0] * ratio[1]
    resized = image.resize((target_width, target_height))
    tiles = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def internvl_tiled_answer(model: Any, question: str, image: Any, task: str,
                          max_tiles: int) -> tuple[str, str, dict[str, Any]]:
    """InternVL chat with official dynamic tiling. Same prompt string and generation config as
    InternVLModel.answer; only the pixel preprocessing differs. Returns (raw, parsed, meta)."""
    import torch

    from src.models.response_parser import parse_answer

    if model._model is None:  # noqa: SLF001
        model._load()  # noqa: SLF001
    rgb = image.convert("RGB") if image.mode != "RGB" else image
    tiles = internvl_dynamic_preprocess(rgb, max_num=max_tiles, image_size=448,
                                        use_thumbnail=True)
    transform = _internvl_build_transform(448)
    hf = model._model  # noqa: SLF001
    pixel_values = torch.stack([transform(t) for t in tiles]).to(
        device=hf.device, dtype=torch.bfloat16)
    generation_config = {
        "max_new_tokens": model.max_tokens,
        "do_sample": model.temperature > 0.0,
        "temperature": model.temperature,
    }
    try:
        out = hf.chat(model._tokenizer, pixel_values, internvl_prompt(model, question),  # noqa: SLF001
                      generation_config)
        raw = str(out[0]) if isinstance(out, tuple) else str(out)
        n_tok = getattr(hf, "num_image_token", None)
        meta = {"model_input_w": 448, "model_input_h": 448, "n_tiles": len(tiles),
                "n_visual_tokens": (n_tok * len(tiles)) if n_tok else None}
        return raw, parse_answer(raw_response=raw, task=task), meta
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_model(model_key: str, internvl_quant: str = "nf4") -> Any:
    """Instantiate the model exactly as run_fullscale_eval.build_model does.

    CUDA_VISIBLE_DEVICES must already be set by the caller.
    """
    if model_key == "qwen":
        from src.models.local_models import QwenVLModel

        return QwenVLModel(name="Qwen2.5-VL-7B-Instruct", device="cuda")
    if model_key == "qwen32b":
        # NOTE: mirrors run_fullscale_eval (vLLM, gpu_memory_utilization=0.92):
        # this takes a whole 80 GB GPU; it does NOT fit a 25 GB/GPU budget.
        from src.models.local_models import QwenVLModel

        return QwenVLModel(
            name="Qwen2.5-VL-32B-Instruct",
            checkpoint="Qwen/Qwen2.5-VL-32B-Instruct",
            device="cuda",
            dtype="bfloat16",
            use_vllm=True,
        )
    if model_key == "internvl":
        from src.models.local_models import InternVLModel

        # Existing v1 path (4-bit NF4 via bitsandbytes, single 448x448 tile, its own short
        # prompt). Only the checkpoint is pinned: load the local snapshot of the revision.
        from huggingface_hub import snapshot_download

        ckpt = snapshot_download("OpenGVLab/InternVL2_5-8B",
                                 revision=PINNED_REVISIONS["internvl"], local_files_only=True)
        return _internvl_with_quant(
            InternVLModel(name="OpenGVLab/InternVL2_5-8B", checkpoint=ckpt, device="cuda"),
            internvl_quant)
    if model_key in ("internvl2-4b", "internvl2-8b"):
        from huggingface_hub import snapshot_download

        from src.models.local_models import InternVLModel

        ckpt = snapshot_download(MODEL_IDS[model_key], revision=PINNED_REVISIONS[model_key],
                                 local_files_only=True)
        return _internvl_with_quant(
            InternVLModel(name=MODEL_IDS[model_key], checkpoint=ckpt, device="cuda"),
            internvl_quant)
    if model_key == "qwen3vl8b":
        import transformers

        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        if (major, minor) < (4, 57):
            raise SystemExit(f"qwen3vl8b needs transformers>=4.57 (have "
                             f"{transformers.__version__}); run with PYTHONPATH=.tf_new")
        from huggingface_hub import snapshot_download

        ckpt = snapshot_download(MODEL_IDS[model_key], revision=PINNED_REVISIONS[model_key],
                                 local_files_only=True)
        return build_qwen3vl(ckpt, PINNED_REVISIONS[model_key])
    raise ValueError(model_key)


def _internvl_with_quant(model: Any, quant: str) -> Any:
    """nf4: unchanged InternVLModel (4-bit NF4 load). bf16: same wrapper, plain bf16 load on GPU."""
    if quant == "nf4":
        return model
    import importlib

    import torch

    tf = importlib.import_module("transformers")
    model._processor = tf.AutoProcessor.from_pretrained(  # noqa: SLF001
        model.checkpoint, trust_remote_code=True)
    model._tokenizer = tf.AutoTokenizer.from_pretrained(  # noqa: SLF001
        model.checkpoint, trust_remote_code=True, use_fast=False)
    model._model = tf.AutoModel.from_pretrained(  # noqa: SLF001
        model.checkpoint, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True).eval().cuda()
    return model


def patch_qwen_vision_mask(query_chunk: int = 0) -> bool:
    """Remove per-window GPU syncs in Qwen2.5-VL vision SDPA attention (transformers 4.49).

    The stock forward builds the block-diagonal mask with
    ``attention_mask[..., cu_seqlens[i-1]:cu_seqlens[i], ...] = True`` where
    ``cu_seqlens`` is a CUDA tensor, so every slice bound triggers ``.item()``
    (a cudaStreamSynchronize): ~10k syncs per image. On a time-shared GPU each
    sync costs ~2 ms (~17 s/image). This replacement builds the *identical*
    boolean mask from one host-side ``tolist()`` and caches it per distinct
    ``cu_seqlens`` (window vs. full-attention layers), then makes the same
    ``F.scaled_dot_product_attention`` call. Outputs are bit-identical.

    ``query_chunk > 0`` additionally splits the attention over query rows in
    chunks of that size (memory O(chunk*N) instead of O(N^2) in the fp32 math
    SDPA path). Rows are independent, so this is mathematically exact but may
    differ from the unchunked kernel by fp32 rounding; use only when large
    images would otherwise exceed the per-GPU memory budget.
    """
    try:
        import torch
        import torch.nn.functional as F  # noqa: N812
        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as mq
    except Exception:  # noqa: BLE001
        return False
    cls = getattr(mq, "Qwen2_5_VLVisionSdpaAttention", None)
    if cls is None or getattr(cls, "_structviz_patched", False):
        return cls is not None
    cache: dict[tuple[Any, ...], Any] = {}

    def forward(self: Any, hidden_states: Any, cu_seqlens: Any,
                rotary_pos_emb: Any = None, position_embeddings: Any = None) -> Any:
        seq_length = hidden_states.shape[0]
        q, k, v = (self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1)
                   .permute(1, 0, 2, 3).unbind(0))
        if position_embeddings is None:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
        q, k = mq.apply_rotary_pos_emb_vision(q, k, cos, sin)

        bounds = tuple(int(x) for x in cu_seqlens.tolist())
        key = (bounds, seq_length, str(q.device))
        attention_mask = cache.get(key)
        if attention_mask is None:
            if len(cache) >= 4:
                cache.clear()
            attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device,
                                         dtype=torch.bool)
            for a, b in zip(bounds[:-1], bounds[1:]):
                attention_mask[..., a:b, a:b] = True
            cache[key] = attention_mask
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        if query_chunk > 0 and seq_length > query_chunk:
            attn_output = torch.cat(
                [
                    F.scaled_dot_product_attention(
                        q[:, s:s + query_chunk], k, v,
                        attention_mask[:, s:s + query_chunk], dropout_p=0.0,
                    )
                    for s in range(0, seq_length, query_chunk)
                ],
                dim=1,
            )
        else:
            attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        return self.proj(attn_output)

    cls._structviz_orig_forward = cls.forward
    cls.forward = forward
    cls._structviz_patched = True
    return True


def run_model(model: Any, question: str, image: Any, task: str) -> tuple[str | None, str]:
    """Return (raw_response or None if unavailable, parsed prediction)."""
    meta = {"task": task}
    if hasattr(model, "answer_with_raw"):
        raw, parsed = model.answer_with_raw(question=question, image=image, metadata=meta)
        return raw, parsed
    if image is None:
        if type(model).__name__ == "InternVLModel":
            return internvl_text_only(model, question, task)
        raise NotImplementedError(f"{type(model).__name__} has no text-only path")
    return None, str(model.answer(question=question, image=image, metadata=meta))


def _clean_final(text: str) -> str:
    """First non-empty line after an answer marker, stripped of markdown and edge punctuation."""
    for line in text.splitlines():
        line = line.replace("**", "").replace("`", "").strip()
        if line:
            return line.strip(" \t\"'.,;:!?$")
    return ""


def cot_extract(raw: str, task: str, style: str) -> dict[str, Any]:
    """CoT answer extraction.

    ``prediction``: text after the LAST answer marker ('Final answer:'; for the ablation_v1
    prompt also 'Answer:'), first line, normalized by ``parse_answer`` exactly like every
    non-CoT prediction. Fallback when no marker: ``parse_answer(extract_answer(raw))``.
    ``prediction_legacy``: ``parse_answer(raw)`` -- what the original pipeline (QwenVLModel.answer)
    stored for a CoT response. ``prediction_extract_answer``: ``extract_answer(raw)``.
    """
    import re

    from src.evaluation.answer_extractor import extract_answer
    from src.models.response_parser import parse_answer

    text = raw.replace("**", "")
    matches = list(re.finditer(COT_MARKERS[style], text, flags=re.IGNORECASE))
    if matches:
        final_text = _clean_final(text[matches[-1].end():])
        pred = parse_answer(raw_response=final_text, task=task) if final_text else ""
    else:
        final_text = None
        pred = parse_answer(raw_response=extract_answer(raw), task=task)
    return {
        "prediction": pred,
        "cot_marker_found": bool(matches),
        "cot_final_answer_text": final_text,
        "prediction_legacy": parse_answer(raw_response=raw, task=task),
        "prediction_extract_answer": extract_answer(raw),
    }


def qwen_generate(model: Any, question: str, image: Any, system_prompt: str,
                  max_tokens: int) -> str:
    """One greedy QwenVLModel/Qwen3VLModel generation with a temporary system prompt and
    generation budget (both restored afterwards). Returns the raw decoded text."""
    import torch

    old_sp, old_mt = model.system_prompt, model.max_tokens
    model.system_prompt, model.max_tokens = system_prompt, max_tokens
    try:
        return str(model._generate_raw(question, image))  # noqa: SLF001
    finally:
        model.system_prompt, model.max_tokens = old_sp, old_mt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def count_tokens(tokenizer: Any, text: str) -> int | None:
    """Token count of generated text (re-tokenized; equals the generated count for BPE
    round-trips up to rare merge differences)."""
    if tokenizer is None:
        return None
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def truncate_tokens(tokenizer: Any, text: str, n: int) -> str:
    """First ``n`` tokens of ``text`` (re-tokenized), decoded."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return str(tokenizer.decode(ids[:n], skip_special_tokens=True))


def no_image_system_prompt(system_prompt: str) -> str:
    """The default system prompt with only the image phrase removed (as in --raw-text)."""
    return (system_prompt
            .replace("You are a precise visual data analyst. ", "You are a precise data analyst. ")
            .replace("Look at the image and answer the question. ", "Answer the question. "))


def internvl_prompt(model: Any, question: str) -> str:
    """The exact prompt string InternVLModel.answer passes to model.chat."""
    return f"{model.system_prompt}\nQuestion: {question}"


def internvl_text_only(model: Any, question: str, task: str) -> tuple[str, str]:
    """No-image InternVL call: same prompt and generation config as InternVLModel.answer,
    but ``pixel_values=None`` (InternVL chat's pure-text branch). Returns (raw, parsed)."""
    import torch

    from src.models.response_parser import parse_answer

    if model._model is None:  # noqa: SLF001
        model._load()  # noqa: SLF001
    generation_config = {
        "max_new_tokens": model.max_tokens,
        "do_sample": model.temperature > 0.0,
        "temperature": model.temperature,
    }
    try:
        out = model._model.chat(  # noqa: SLF001
            model._tokenizer, None, internvl_prompt(model, question), generation_config)  # noqa: SLF001
        raw = str(out[0]) if isinstance(out, tuple) else str(out)
        return raw, parse_answer(raw_response=raw, task=task)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def prompt_hash(model: Any, question: str, has_image: bool) -> str | None:
    """sha256 of the chat-templated prompt (system + user turn + gen prompt)."""
    if type(model).__name__ == "InternVLModel":
        # Pre-template prompt string (InternVL's chat() applies its own template).
        return hashlib.sha256(internvl_prompt(model, question).encode("utf-8")).hexdigest()
    if not hasattr(model, "build_prompt_text"):
        return None
    text = model.build_prompt_text(question, has_image=has_image)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def visual_geometry(width: int, height: int) -> dict[str, int]:
    """Model-input size and visual-token count exactly as qwen_vl_utils.fetch_image resizes.

    Uses the default min/max pixels (4..16384 merged tokens at factor 28).
    """
    from qwen_vl_utils.vision_process import (
        IMAGE_MAX_TOKEN_NUM, IMAGE_MIN_TOKEN_NUM, SPATIAL_MERGE_SIZE, smart_resize,
    )

    factor = 14 * SPATIAL_MERGE_SIZE
    rh, rw = smart_resize(height, width, factor=factor,
                          min_pixels=IMAGE_MIN_TOKEN_NUM * factor**2,
                          max_pixels=IMAGE_MAX_TOKEN_NUM * factor**2)
    return {"model_input_w": rw, "model_input_h": rh,
            "n_visual_tokens": (rh // factor) * (rw // factor)}


def img_meta_sha(data: bytes) -> str:
    """sha256 hex digest of the image file bytes."""
    return hashlib.sha256(data).hexdigest()


def load_done(output: Path) -> tuple[set[tuple[str, str]], int]:
    """Done keys = (question_id, viz_type) with a non-[ERROR] prediction."""
    done: set[tuple[str, str]] = set()
    n_err = 0
    if not output.exists():
        return done, 0
    for row in read_jsonl(output):
        key = (str(row.get("question_id", "")), str(row.get("viz_type", "")))
        if str(row.get("prediction", "")) == "[ERROR]":
            n_err += 1
            continue
        done.add(key)
    return done, n_err


def build_tasks(args: argparse.Namespace, bench: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the ordered list of work rows (before sharding)."""
    if args.no_image or args.raw_text:
        if args.manifest is not None:
            qids: list[str] = []
            seen: set[str] = set()
            for r in read_jsonl(args.manifest):
                q = str(r["question_id"])
                if q not in seen:
                    seen.add(q)
                    qids.append(q)
        else:
            qids = list(bench)
        suite = args.suite or ("raw_text" if args.raw_text else "no_image")
        viz = RAW_TEXT_VIZ if args.raw_text else NO_IMAGE_VIZ
        return [
            {"question_id": q, "viz_type": viz, "image_path": None, "suite": suite}
            for q in qids
        ]
    if args.manifest is None:
        raise SystemExit("--manifest is required unless --no-image")
    default_suite = args.suite or args.manifest.resolve().parent.name
    tasks = []
    n_skipped = 0
    for r in read_jsonl(args.manifest):
        if not r.get("viz_type") or not r.get("image_path"):
            n_skipped += 1  # render-failure rows (render_suite.py writes 'error' instead)
            continue
        tasks.append(
            {
                "question_id": str(r["question_id"]),
                "viz_type": str(r["viz_type"]),
                "image_path": str(r["image_path"]),
                "suite": args.suite or str(r.get("suite") or default_suite),
                "manifest_modality": r.get("modality"),
                "manifest_task": r.get("task"),
                "manifest_sha256": r.get("sha256"),
            }
        )
    if n_skipped:
        log(f"WARNING: skipped {n_skipped} manifest rows without viz_type/image_path "
            f"(render failures)")
    return tasks


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Entry point."""
    global _LOG_FH
    args = parse_args()
    if args.no_image and args.raw_text:
        raise SystemExit("--no-image and --raw-text are mutually exclusive")
    if args.cot and args.transcribe_then_answer:
        raise SystemExit("--cot and --transcribe-then-answer are mutually exclusive")
    if (args.cot or args.transcribe_then_answer) and (args.no_image or args.raw_text):
        raise SystemExit("--cot/--transcribe-then-answer need images (not --no-image/--raw-text)")
    if (args.cot or args.transcribe_then_answer) and args.model not in QWEN_KEYS:
        raise SystemExit("--cot/--transcribe-then-answer are implemented for Qwen models only")
    if args.internvl_max_tiles is not None and args.model not in INTERNVL_KEYS:
        raise SystemExit("--internvl-max-tiles applies to InternVL models only")

    # Must happen before torch is imported anywhere.
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cwd = Path.cwd()

    def _abs(p: Path | None) -> Path | None:
        if p is None or p.is_absolute():
            return p
        return (cwd / p) if (cwd / p).exists() or not (PROJECT_ROOT / p).exists() else PROJECT_ROOT / p

    args.output, args.manifest, args.benchmark, args.log = (
        _abs(args.output), _abs(args.manifest), _abs(args.benchmark), _abs(args.log))

    if args.log is not None:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        _LOG_FH = open(args.log, "a", encoding="utf-8")  # noqa: SIM115

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))
    from PIL import Image

    from src.evaluation.metrics import compute_metrics

    bench = {str(r["question_id"]): r for r in read_jsonl(args.benchmark)}
    all_tasks = build_tasks(args, bench)
    shard_i, shard_n = parse_shard(args.shard)
    # Shard and order by a hash of the question id, so (a) every format of a question lands in
    # the same shard and (b) any prefix of a shard is a random sample of complete questions.
    import hashlib as _hl

    def _qkey(t: dict) -> str:
        return _hl.md5(str(t["question_id"]).encode()).hexdigest()

    all_tasks = sorted(all_tasks, key=lambda t: (_qkey(t), str(t.get("viz_type"))))
    tasks = [t for t in all_tasks if int(_qkey(t), 16) % shard_n == shard_i]

    missing_q = [t["question_id"] for t in tasks if t["question_id"] not in bench]
    if missing_q:
        raise SystemExit(f"{len(missing_q)} manifest question_ids not in benchmark, "
                         f"e.g. {missing_q[:3]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done, n_prev_err = load_done(args.output)
    pending = [t for t in tasks if (t["question_id"], t["viz_type"]) not in done]
    if args.limit is not None:
        pending = pending[: args.limit]
    n_done = sum((t["question_id"], t["viz_type"]) in done for t in tasks)
    log(f"model={args.model} shard={shard_i}/{shard_n} total_manifest={len(all_tasks)} "
        f"shard_rows={len(tasks)} already_done={n_done} "
        f"prev_error_rows(retrying)={n_prev_err} pending_this_run={len(pending)} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    if not pending:
        log("Nothing to do.")
        return

    model_id = MODEL_IDS[args.model]
    revision = PINNED_REVISIONS.get(args.model) or hf_revision(model_id)
    if args.model in QWEN25_KEYS and not args.no_vision_mask_patch:
        ok = patch_qwen_vision_mask(query_chunk=args.vision_attn_chunk)
        log(f"Qwen vision-mask sync patch applied: {ok} "
            f"(vision_attn_chunk={args.vision_attn_chunk})")
    t0 = time.time()
    if args.model == "qwen32b-hf":
        caps = free_memory_caps(args.headroom_gb)
        log("max_memory caps (GiB, visible index -> physical "
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')}): "
            + ", ".join(f"{k}:{v / 2**30:.1f}" for k, v in caps.items()))
        model = build_qwen_hf_sharded(model_id, revision, caps)
    else:
        model = build_model(args.model, args.internvl_quant)
    if (args.no_image or args.raw_text) and hasattr(model, "system_prompt"):
        # The shared system prompt says "Look at the image"; without an image that wording
        # invites refusals ("none"). Keep every answer-format rule, drop only the image phrase.
        model.system_prompt = (
            model.system_prompt
            .replace("You are a precise visual data analyst. ", "You are a precise data analyst. ")
            .replace("Look at the image and answer the question. ", "Answer the question. ")
        )
        log(f"{'raw-text' if args.raw_text else 'no-image'} mode: system prompt image phrase "
            f"removed")
    if args.cot:
        model.system_prompt = COT_SYSTEM_PROMPTS[args.cot_prompt]
        model.max_tokens = args.max_new_tokens or 1024
        log(f"CoT mode: prompt={args.cot_prompt} max_new_tokens={model.max_tokens}")
    elif args.max_new_tokens is not None:
        model.max_tokens = args.max_new_tokens
    if args.transcribe_then_answer:
        # Stage 2 sees no image: same answer-format rules, image phrase removed (as --raw-text).
        model.system_prompt = no_image_system_prompt(model.system_prompt)
        log(f"transcribe-then-answer: stage1 max_new_tokens={args.transcribe_max_new_tokens}, "
            f"stage2 max_new_tokens={model.max_tokens}, stage-2 system prompt image phrase "
            f"removed")
    if hasattr(model, "_load") and getattr(model, "_model", None) is None:
        model._load()  # noqa: SLF001 - eager load so load time is not billed to row 1
    log(f"Model loaded in {time.time() - t0:.1f}s (revision={revision})")
    hf_model = getattr(model, "_model", None)
    dmap = getattr(hf_model, "hf_device_map", None) or {}
    offloaded = sorted({str(d) for d in dmap.values() if str(d) in ("cpu", "disk")})
    if offloaded:
        n_off = sum(str(d) in ("cpu", "disk") for d in dmap.values())
        msg = (f"device_map='auto' offloaded {n_off}/{len(dmap)} modules to {offloaded} "
               f"(GPU too full at load time) -> inference would be ~10x slower")
        if not args.allow_offload:
            raise SystemExit(f"ABORT: {msg}. Free the GPU or pass --allow-offload.")
        log(f"WARNING: {msg}")
    elif dmap:
        log(f"device_map: all {len(dmap)} modules on {sorted({str(d) for d in dmap.values()})}")
    if hf_model is not None and dmap:
        import collections

        import torch as _t

        per_dev = collections.Counter(str(d) for d in dmap.values())
        vis_dev = {str(d) for k, d in dmap.items() if k.startswith("visual")}
        log("placement: modules per device " + str(dict(per_dev)) + f"; visual on {sorted(vis_dev)}; "
            + "allocated GiB " + ", ".join(
                f"{i}:{_t.cuda.memory_allocated(i) / 2**30:.1f}"
                for i in range(_t.cuda.device_count())))
    vis = getattr(hf_model, "visual", None)
    if vis is not None and getattr(vis, "blocks", None):
        attn_cls = type(vis.blocks[0].attn)
        log(f"vision attention: {attn_cls.__name__} x{len(vis.blocks)} blocks, "
            f"patched={getattr(attn_cls, '_structviz_patched', False)}, "
            f"window_size={getattr(vis, 'window_size', None)}, "
            f"fullatt_blocks={getattr(vis, 'fullatt_block_indexes', None)}")

    import torch

    shutdown = False

    def _handle(sig: int, _frame: Any) -> None:
        nonlocal shutdown
        log(f"Signal {sig}: finishing current row, flushing, exiting.")
        shutdown = True

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    out = open(args.output, "a", encoding="utf-8")  # noqa: SIM115
    n_new = n_err = 0
    run_start = time.time()
    prompt_cache: dict[tuple[str, bool], str | None] = {}
    tokenizer = getattr(getattr(model, "_processor", None), "tokenizer", None)
    if args.raw_text and not hasattr(model, "build_prompt_text"):
        raise SystemExit("--raw-text needs a model with build_prompt_text (Qwen)")

    def flush() -> None:
        out.flush()
        os.fsync(out.fileno())

    stage1_hashes: dict[str, str] = {}

    def _stage1_hash(modality: str) -> str:
        old_sp = model.system_prompt
        model.system_prompt = TRANSCRIBE_SYSTEM_PROMPT
        try:
            text = model.build_prompt_text(TRANSCRIBE_PROMPTS[modality], has_image=True)
        finally:
            model.system_prompt = old_sp
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    try:
        for k, t in enumerate(pending, 1):
            if shutdown:
                break
            item = bench[t["question_id"]]
            question, task = str(item["question"]), str(item.get("task", "generic"))
            if args.raw_text:
                question = raw_text_message(item)  # user turn = serialized data + question
            if args.cot and args.cot_user_suffix:
                question = question + COT_USER_SUFFIXES[args.cot_prompt]
            image = None
            img_meta: dict[str, Any] = {"image_path": None, "image_sha256": None,
                                        "image_w": None, "image_h": None}
            if t["image_path"] is not None:
                ipath = resolve_image_path(t["image_path"], args.manifest)
                data = ipath.read_bytes()
                image = Image.open(io.BytesIO(data))
                image.load()
                if t.get("manifest_sha256") and t["manifest_sha256"] != img_meta_sha(data):
                    log(f"WARNING: sha256 mismatch vs manifest for {ipath}")
                img_meta = {
                    "image_path": t["image_path"],
                    "image_sha256": img_meta_sha(data),
                    "image_w": image.width,
                    "image_h": image.height,
                }
                if args.model in QWEN25_KEYS:
                    img_meta.update(visual_geometry(image.width, image.height))
                elif args.model in INTERNVL_KEYS and args.internvl_max_tiles is None:
                    # InternVLModel.answer resizes every image to one 448x448 tile.
                    n_tok = getattr(getattr(model, "_model", None), "num_image_token", None)
                    img_meta.update({"model_input_w": 448, "model_input_h": 448,
                                     "n_visual_tokens": n_tok})
            has_image = image is not None
            pkey = (question, has_image)
            if pkey not in prompt_cache:
                prompt_cache[pkey] = prompt_hash(model, question, has_image)
            n_prompt_tokens: int | None = None
            if args.raw_text and tokenizer is not None:
                n_prompt_tokens = len(tokenizer(
                    model.build_prompt_text(question, has_image=False),
                    add_special_tokens=False)["input_ids"])

            raw: str | None = None
            prediction = "[ERROR]"
            error_msg: str | None = None
            extra: dict[str, Any] = {}
            t_row = time.time()
            for attempt in range(args.max_retries + 1):
                try:
                    extra = {}
                    if args.transcribe_then_answer:
                        modality = str(item["modality"])
                        t1 = time.time()
                        transcript = qwen_generate(
                            model, TRANSCRIBE_PROMPTS[modality], image,
                            TRANSCRIBE_SYSTEM_PROMPT, args.transcribe_max_new_tokens)
                        lat1 = time.time() - t1
                        q2 = f"{transcript}\n\nQuestion: {item['question']}"
                        t2 = time.time()
                        raw, prediction = run_model(model, q2, None, task)
                        n_tr = count_tokens(tokenizer, transcript)
                        extra = {
                            "mode": "transcribe_then_answer",
                            "transcript": transcript,
                            "transcript_n_tokens": n_tr,
                            "transcript_truncated": (n_tr is not None
                                                     and n_tr >= args.transcribe_max_new_tokens),
                            "latency_stage1_s": round(lat1, 3),
                            "latency_stage2_s": round(time.time() - t2, 3),
                            "prompt_sha256_stage1": stage1_hashes.setdefault(
                                modality, _stage1_hash(modality)),
                            "prompt_sha256_stage2": hashlib.sha256(model.build_prompt_text(
                                q2, has_image=False).encode("utf-8")).hexdigest(),
                            "max_new_tokens_stage1": args.transcribe_max_new_tokens,
                            "prompt_sha256": None,  # set below to the stage-2 hash
                            "max_new_tokens": model.max_tokens,
                        }
                    elif args.internvl_max_tiles is not None and image is not None:
                        raw, prediction, tmeta = internvl_tiled_answer(
                            model, question, image, task, args.internvl_max_tiles)
                        img_meta.update(tmeta)
                    else:
                        raw, prediction = run_model(model, question, image, task)
                    if args.cot:
                        assert raw is not None
                        ce = cot_extract(raw, task, args.cot_prompt)
                        prediction = ce.pop("prediction")
                        n_gen = count_tokens(tokenizer, raw)
                        extra = {"mode": "cot", "cot_prompt": args.cot_prompt,
                                 "cot_user_suffix": bool(args.cot_user_suffix), **ce,
                                 "n_gen_tokens": n_gen,
                                 "hit_max_new_tokens": (n_gen is not None
                                                        and n_gen >= model.max_tokens),
                                 "max_new_tokens": model.max_tokens}
                        if tokenizer is not None and n_gen is not None and n_gen > 256:
                            from src.models.response_parser import parse_answer
                            extra["prediction_legacy_trunc256"] = parse_answer(
                                raw_response=truncate_tokens(tokenizer, raw, 256), task=task)
                        else:
                            extra["prediction_legacy_trunc256"] = ce["prediction_legacy"]
                    if args.model == "qwen3vl8b" and getattr(model, "last_geometry", None) \
                            and not args.transcribe_then_answer:
                        img_meta.update(model.last_geometry)
                    error_msg = None
                    break
                except Exception as exc:  # noqa: BLE001
                    error_msg = f"{type(exc).__name__}: {str(exc)[:500]}"
                    log(f"  row error ({t['question_id']}, {t['viz_type']}) "
                        f"attempt {attempt + 1}: {error_msg[:200]}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    time.sleep(2.0 * (attempt + 1))
            latency = time.time() - t_row

            bundle = compute_metrics(prediction=prediction, answer=str(item["answer"]))
            row: dict[str, Any] = {
                "question_id": item["question_id"],
                "question": item["question"],
                "answer": item["answer"],
                "prediction": prediction,
                "modality": item["modality"],
                "source": str(item.get("source", "synthetic")),
                "viz_type": t["viz_type"],
                "difficulty": item["difficulty"],
                "task": item["task"],
                "exact_match": bundle.exact,
                "f1": bundle.f1,
                "numeric_accuracy": bundle.numeric,
                "raw_response": raw,
                "model_id": model_id,
                "model_revision": revision,
                **img_meta,
                "prompt_sha256": prompt_cache[pkey],
                "suite": t["suite"],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "latency_s": round(latency, 3),
            }
            if args.raw_text:
                row["n_prompt_tokens"] = n_prompt_tokens
            if extra:
                for k_legacy in ("prediction_legacy", "prediction_legacy_trunc256",
                                 "prediction_extract_answer"):
                    if k_legacy in extra:
                        extra["exact_match_" + k_legacy.removeprefix("prediction_")] = (
                            compute_metrics(prediction=str(extra[k_legacy]),
                                            answer=str(item["answer"])).exact)
                row.update(extra)
                if extra.get("mode") == "transcribe_then_answer":
                    row["prompt_sha256"] = extra["prompt_sha256_stage2"]
            if args.model in INTERNVL_KEYS and (args.internvl_max_tiles is not None
                                                or args.internvl_quant != "nf4"
                                                or args.model != "internvl"):
                row["internvl_max_tiles"] = args.internvl_max_tiles
                row["internvl_quant"] = args.internvl_quant
            if args.model == "qwen3vl8b":
                import transformers as _tfm
                row["transformers_version"] = _tfm.__version__
            if error_msg is not None:
                row["error"] = error_msg
                n_err += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_new += 1

            if k % args.checkpoint_every == 0:
                flush()
                el = time.time() - run_start
                rate = n_new / el * 60 if el > 0 else 0.0
                eta_h = (len(pending) - k) / (n_new / el) / 3600 if n_new else 0.0
                mem = (torch.cuda.max_memory_allocated() / 2**30
                       if torch.cuda.is_available() else 0.0)
                log(f"{k}/{len(pending)} new={n_new} err={n_err} rate={rate:.1f} rows/min "
                    f"ETA={eta_h:.2f}h peak_alloc={mem:.1f}GiB")
    finally:
        flush()
        out.close()
        el = time.time() - run_start
        mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        log(f"{'STOPPED' if shutdown else 'DONE'}: new={n_new} err={n_err} "
            f"elapsed={el:.1f}s rate={n_new / el * 60 if el else 0:.1f} rows/min "
            f"peak_alloc={mem:.1f}GiB peak_reserved="
            f"{(torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else 0):.1f}GiB")


if __name__ == "__main__":
    main()
