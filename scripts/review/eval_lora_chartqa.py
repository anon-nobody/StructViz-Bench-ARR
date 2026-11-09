"""Base / LoRA Qwen2.5-VL-7B on the ChartQA manifests, exactly as scripts/mitigation/eval_lora.py.

Same prompt (question + " Answer concisely. Give only the final answer."), same processor
min/max_pixels, do_resize=False, greedy decoding, 16 new tokens -- i.e. the protocol of the
paper's LoRA table (App. "Mitigation training details"). Differences from eval_lora.py are I/O
only: it reads an eval_local_suite-style manifest (one row per (question_id, viz_type)) joined
with the ChartQA benchmark JSONL, writes one resumable prediction row per image, and scores
with src.evaluation.metrics.compute_metrics (exact_match) for the chartqa_score.py reader.

    python scripts/review/eval_lora_chartqa.py --manifest results/review/chartqa/manifest_chartqa_original.jsonl \
        --adapter checkpoints/consistency_lora_mm --setting lambda1 \
        --output results/review/chartqa/pred_lora_lambda1_original.jsonl --gpu 0
    add --dry_run to parse args / join manifest without importing torch.
"""

from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PROMPT_SUFFIX = " Answer concisely. Give only the final answer."


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--benchmark", type=Path,
                    default=ROOT / "results/review/chartqa/chartqa_items.jsonl")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir; omit for base model")
    ap.add_argument("--setting", required=True, choices=["base", "lambda0", "lambda1"])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max_pixels", type=int, default=200704, help="LoRA training resolution")
    ap.add_argument("--min_pixels", type=int, default=50176, help="eval_lora.py default")
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--gpu", type=str, default=None, help="sets CUDA_VISIBLE_DEVICES")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    if (a.setting == "base") != (a.adapter is None):
        ap.error("--setting base requires no --adapter; lambda0/lambda1 require --adapter")
    if a.adapter and not (Path(a.adapter) / "adapter_config.json").exists() and \
            not (ROOT / a.adapter / "adapter_config.json").exists():
        ap.error(f"adapter not found: {a.adapter}")
    return a


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def hf_revision(repo_id: str) -> str | None:
    """Snapshot hash of the locally cached HF repo (as eval_local_suite.hf_revision)."""
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


def adapter_sha256(adapter: str | None) -> str | None:
    """sha256 of adapter_model.safetensors (identifies the exact LoRA weights)."""
    if not adapter:
        return None
    path = Path(adapter) if Path(adapter).is_absolute() else ROOT / adapter
    digest = hashlib.sha256()
    with (path / "adapter_model.safetensors").open("rb") as fh:
        for chunk in iter(lambda: fh.read(2**24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:  # noqa: PLR0915
    a = parse_args()
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    bench = {r["question_id"]: r for r in read_jsonl(a.benchmark)}
    rows = [r for r in read_jsonl(a.manifest) if r.get("image_path")]
    missing = [r["question_id"] for r in rows if r["question_id"] not in bench]
    if missing:
        raise SystemExit(f"{len(missing)} manifest qids not in benchmark, e.g. {missing[:3]}")
    done = set()
    if a.output.exists():
        done = {(r["question_id"], r["viz_type"]) for r in read_jsonl(a.output)
                if r.get("prediction") != "[ERROR]"}
    pending = [r for r in rows if (r["question_id"], r["viz_type"]) not in done]
    if a.limit:
        pending = pending[: a.limit]
    print(f"setting={a.setting} adapter={a.adapter} manifest={a.manifest} rows={len(rows)} "
          f"done={len(done)} pending={len(pending)} max_pixels={a.max_pixels} "
          f"min_pixels={a.min_pixels} max_new_tokens={a.max_new_tokens} -> {a.output}",
          flush=True)
    if a.dry_run or not pending:
        return

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
    except ImportError:
        from transformers import AutoModelForImageTextToText as VLModel

    from src.evaluation.metrics import compute_metrics

    revision = hf_revision(a.model_id)
    ad_sha = adapter_sha256(a.adapter)
    print(f"model_revision={revision} adapter_sha256={ad_sha}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(a.model_id, max_pixels=a.max_pixels,
                                              min_pixels=a.min_pixels)
    model = VLModel.from_pretrained(a.model_id, torch_dtype=torch.bfloat16).to(device).eval()
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).eval()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with a.output.open("a", encoding="utf-8") as out:
        for k, r in enumerate(pending, 1):
            item = bench[r["question_id"]]
            path = str(ROOT / r["image_path"])
            data = Path(path).read_bytes()
            q = item["question"] + PROMPT_SUFFIX
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": path, "max_pixels": a.max_pixels,
                 "min_pixels": a.min_pixels},
                {"type": "text", "text": q}]}]
            t_row = time.time()
            err = None
            try:
                text = processor.apply_chat_template(msgs, tokenize=False,
                                                     add_generation_prompt=True)
                imgs, vids = process_vision_info(msgs)
                inp = processor(text=[text], images=imgs, videos=vids, do_resize=False,
                                return_tensors="pt").to(device)
                with torch.no_grad():
                    gen = model.generate(**inp, max_new_tokens=a.max_new_tokens,
                                         do_sample=False)
                pred = processor.batch_decode([gen[0][inp["input_ids"].shape[1]:]],
                                              skip_special_tokens=True)[0].strip()
            except Exception as exc:  # noqa: BLE001
                pred, err = "[ERROR]", f"{type(exc).__name__}: {str(exc)[:300]}"
            bundle = compute_metrics(prediction=pred, answer=str(item["answer"]))
            row = {"question_id": r["question_id"], "question": item["question"],
                   "answer": item["answer"], "prediction": pred, "raw_response": pred,
                   "modality": "tabular", "source": item.get("source"),
                   "viz_type": r["viz_type"], "difficulty": item.get("difficulty"),
                   "task": item["task"], "exact_match": bundle.exact,
                   "numeric_accuracy": bundle.numeric, "f1": bundle.f1,
                   "model_id": a.model_id, "model_revision": revision,
                   "adapter_sha256": ad_sha, "setting": a.setting, "adapter": a.adapter,
                   "max_pixels": a.max_pixels, "min_pixels": a.min_pixels,
                   "max_new_tokens": a.max_new_tokens, "image_path": r["image_path"],
                   "image_sha256": hashlib.sha256(data).hexdigest(), "suite": "chartqa",
                   "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "latency_s": round(time.time() - t_row, 3)}
            if err:
                row["error"] = err
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            if k % 50 == 0:
                out.flush()
                print(f"{k}/{len(pending)} {(time.time() - t0) / k:.2f}s/row", flush=True)
    print(f"DONE {len(pending)} rows in {time.time() - t0:.0f}s -> {a.output}")


if __name__ == "__main__":
    main()
