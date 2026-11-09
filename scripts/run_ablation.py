"""Run StructViz-Bench ablation studies.

Supports:
1) Visualization-type leave-one-out ablation (offline recomputation).
2) Prompt sensitivity ablation over a stratified benchmark sample.
"""

from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import argparse
import csv
import os
import random
import signal
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import yaml

matplotlib.use("Agg")

from src.evaluation.metrics import compute_metrics
from src.models.api_models import ClaudeModel, GPT4oModel, GeminiModel
from src.rendering.render_pipeline import RenderPipeline
from src.utils.io_utils import BenchmarkItem, read_jsonl, write_jsonl

os.environ["PYTHONUNBUFFERED"] = "1"

# ── API Keys (read from environment) ─────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "gpt4o": {
        "name": "gpt-4o",
        "rpm": 50,
        "output": "full_gpt4o.jsonl",
    },
    "gemini": {
        "name": "gemini-2.0-flash",
        "rpm": 30,
        "output": "full_gemini.jsonl",
    },
    "claude": {
        "name": "claude-sonnet-4-20250514",
        "rpm": 40,
        "output": "full_claude.jsonl",
    },
    "qwen": {
        "name": "Qwen2.5-VL-7B-Instruct",
        "rpm": 999,
        "output": "full_qwen.jsonl",
    },
    "qwen72b": {
        "name": "Qwen2.5-VL-72B-Instruct",
        "rpm": 999,
        "output": "full_qwen72b.jsonl",
    },
    "internvl": {
        "name": "OpenGVLab/InternVL2_5-8B",
        "rpm": 999,
        "output": "full_internvl.jsonl",
    },
    "internvl3": {
        "name": "OpenGVLab/InternVL3-8B",
        "rpm": 999,
        "output": "full_internvl3.jsonl",
    },
    "llava": {
        "name": "llava-hf/llava-v1.6-mistral-7b-hf",
        "rpm": 999,
        "output": "full_llava.jsonl",
    },
}


@dataclass(slots=True)
class ShutdownState:
    """Track graceful shutdown intent from signal handlers."""

    requested: bool = False


def log(msg: str) -> None:
    """Log with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments with subcommands."""
    parser = argparse.ArgumentParser(
        description="Run StructViz-Bench ablation studies."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    viz_parser = subparsers.add_parser(
        "viz-removal",
        help="Leave-one-out visualization-type ablation from existing full results.",
    )
    viz_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ablation_viz_removal.yaml"),
        help="Visualization-removal ablation config path.",
    )
    viz_parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing full_*.jsonl files.",
    )
    viz_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ablation"),
        help="Directory for ablation outputs.",
    )

    prompt_parser = subparsers.add_parser(
        "prompt-sensitivity",
        help="Prompt sensitivity ablation on a stratified sample.",
    )
    prompt_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ablation_prompt_sensitivity.yaml"),
        help="Prompt-sensitivity ablation config path.",
    )
    prompt_parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("benchmark/realworld_test.jsonl"),
        help="Benchmark JSONL path.",
    )
    prompt_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ablation"),
        help="Directory for ablation outputs.",
    )
    prompt_parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_SPECS.keys()),
        help="Model key to evaluate.",
    )
    prompt_parser.add_argument(
        "--gpu",
        type=int,
        default=3,
        help="GPU index for local models (default: 3).",
    )
    prompt_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing prompt_{model}_{variant}.jsonl files.",
    )
    prompt_parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Save checkpoint every N processed items (default: 5).",
    )
    prompt_parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries per model call (default: 5).",
    )
    return parser.parse_args()


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read YAML config as a dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return dict(loaded)


def _install_signal_handlers(shutdown: ShutdownState) -> None:
    """Install SIGINT/SIGTERM handlers for graceful shutdown."""

    def _handle_signal(sig: int, frame: Any) -> None:
        del frame
        log(f"Signal {sig} received - finishing current item, then exiting...")
        shutdown.requested = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)


def build_model(model_key: str, gpu: int = 3) -> Any:
    """Build model instance with API keys configured."""
    if model_key == "gpt4o":
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        return GPT4oModel(
            name="gpt-4o",
            requests_per_minute=MODEL_SPECS["gpt4o"]["rpm"],
        )

    if model_key == "gemini":
        os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
        return GeminiModel(
            name="gemini-2.0-flash",
            requests_per_minute=MODEL_SPECS["gemini"]["rpm"],
        )

    if model_key == "claude":
        os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
        return ClaudeModel(
            name="claude-sonnet-4-20250514",
            requests_per_minute=MODEL_SPECS["claude"]["rpm"],
        )

    if model_key == "qwen":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        from src.models.local_models import QwenVLModel

        return QwenVLModel(name="Qwen2.5-VL-7B-Instruct", device="cuda")

    if model_key == "qwen72b":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        from src.models.local_models import QwenVLModel

        return QwenVLModel(
            name="Qwen2.5-VL-72B-Instruct",
            checkpoint="Qwen/Qwen2.5-VL-72B-Instruct",
            device="cuda",
            use_vllm=True,
        )

    if model_key == "internvl3":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        from src.models.local_models import InternVLModel

        return InternVLModel(
            name="InternVL3-8B",
            checkpoint="OpenGVLab/InternVL3-8B",
            device="cuda",
        )

    if model_key == "internvl":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        from src.models.local_models import InternVLModel

        return InternVLModel(name="OpenGVLab/InternVL2_5-8B", device="cuda")

    if model_key == "llava":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        from src.models.local_models import LlavaModel

        return LlavaModel(
            checkpoint="llava-hf/llava-v1.6-mistral-7b-hf",
            name="llava-v1.6-mistral-7b",
            device="cuda",
        )

    raise ValueError(f"Unknown model key: {model_key}")


def call_with_retries(
    model: Any,
    question: str,
    image: Any,
    metadata: dict[str, Any],
    max_retries: int = 5,
) -> str:
    """Call model.answer() with retry and exponential backoff."""
    for attempt in range(max_retries):
        try:
            return str(model.answer(question=question, image=image, metadata=metadata))
        except Exception as error:  # noqa: BLE001
            message = str(error)
            is_rate_limited = (
                "429" in message
                or "rate limit" in message.lower()
                or "quota" in message.lower()
                or "resource_exhausted" in message.lower()
            )
            if attempt == max_retries - 1:
                log(f"  API FAILED after {max_retries} retries: {message[:200]}")
                return "[ERROR]"

            wait: float | None = None
            response = getattr(error, "response", None)
            headers = getattr(response, "headers", None)
            if headers:
                retry_after: Any = None
                if isinstance(headers, dict):
                    retry_after = headers.get("retry-after") or headers.get(
                        "Retry-After"
                    )
                elif hasattr(headers, "get"):
                    retry_after = headers.get("retry-after") or headers.get(
                        "Retry-After"
                    )
                if retry_after:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = None

            if wait is None:
                if is_rate_limited:
                    wait = min(3.0**attempt, 120.0)
                else:
                    wait = min(2.0**attempt, 30.0)

            log(
                f"  Retry {attempt + 1}/{max_retries}: "
                f"{'RATE_LIMIT' if is_rate_limited else 'ERROR'}: "
                f"{message[:120]}; sleeping {wait:.1f}s",
            )
            time.sleep(wait)
    return "[ERROR]"


def compute_row(
    item: BenchmarkItem,
    viz_type: str,
    prediction: str,
    prompt_variant: str | None = None,
) -> dict[str, Any]:
    """Build one result row with metrics."""
    bundle = compute_metrics(prediction=prediction, answer=item.answer)
    row: dict[str, Any] = {
        "question_id": item.question_id,
        "question": item.question,
        "answer": item.answer,
        "prediction": prediction,
        "modality": item.modality,
        "source": item.source,
        "viz_type": viz_type,
        "difficulty": item.difficulty,
        "task": item.task,
        "exact_match": bundle.exact,
        "f1": bundle.f1,
        "numeric_accuracy": bundle.numeric,
    }
    if prompt_variant is not None:
        row["prompt_variant"] = prompt_variant
    return row


def recompute_metric_means(rows: list[dict[str, Any]]) -> tuple[float, float, float]:
    """Recompute mean EM/F1/numeric from prediction-answer rows."""
    if not rows:
        return 0.0, 0.0, 0.0
    em_total = 0.0
    f1_total = 0.0
    numeric_total = 0.0
    for row in rows:
        bundle = compute_metrics(
            prediction=str(row.get("prediction", "")),
            answer=str(row.get("answer", "")),
        )
        em_total += bundle.exact
        f1_total += bundle.f1
        numeric_total += bundle.numeric
    size = float(len(rows))
    return em_total / size, f1_total / size, numeric_total / size


def _latex_escape(text: str) -> str:
    """Escape underscore for LaTeX table values."""
    return text.replace("_", "\\_")


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write a CSV file from dictionaries with fixed column ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_viz_removal(args: argparse.Namespace) -> None:
    """Run leave-one-out viz-type ablation by filtering existing full results."""
    config = _read_yaml(args.config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    viz_map: dict[str, list[str]] = {
        "tabular": [str(v) for v in config.get("tabular_viz_types", [])],
        "timeseries": [str(v) for v in config.get("timeseries_viz_types", [])],
        "graph": [str(v) for v in config.get("graph_viz_types", [])],
    }
    if not all(viz_map.values()):
        raise ValueError(
            "Ablation config must define tabular/timeseries/graph viz lists"
        )

    model_keys = ["gpt4o", "gemini", "claude", "qwen"]
    summary_rows: list[dict[str, Any]] = []

    for model_key in model_keys:
        full_path = args.results_dir / f"full_{model_key}.jsonl"
        rows = read_jsonl(full_path)
        if not rows:
            raise FileNotFoundError(f"Missing or empty full result file: {full_path}")
        log(f"Loaded {len(rows)} rows for model={model_key} from {full_path}")

        baseline_by_modality: dict[str, tuple[float, float, float]] = {}
        for modality in ("tabular", "timeseries", "graph"):
            modality_rows = [
                row for row in rows if str(row.get("modality", "")) == modality
            ]
            baseline_by_modality[modality] = recompute_metric_means(modality_rows)

        for modality in ("tabular", "timeseries", "graph"):
            for excluded_viz in viz_map[modality]:
                filtered = [
                    row
                    for row in rows
                    if not (
                        str(row.get("modality", "")) == modality
                        and str(row.get("viz_type", "")) == excluded_viz
                    )
                ]

                condition_path = (
                    output_dir
                    / f"viz_removal_{model_key}_{modality}_{excluded_viz}.jsonl"
                )
                write_jsonl(condition_path, filtered)

                ablated_modality_rows = [
                    row for row in filtered if str(row.get("modality", "")) == modality
                ]
                baseline_em, baseline_f1, baseline_numeric = baseline_by_modality[
                    modality
                ]
                ablated_em, ablated_f1, ablated_numeric = recompute_metric_means(
                    ablated_modality_rows
                )

                summary_rows.append(
                    {
                        "model": model_key,
                        "modality": modality,
                        "excluded_viz": excluded_viz,
                        "baseline_em": baseline_em,
                        "ablated_em": ablated_em,
                        "delta_em": ablated_em - baseline_em,
                        "baseline_f1": baseline_f1,
                        "ablated_f1": ablated_f1,
                        "delta_f1": ablated_f1 - baseline_f1,
                        "baseline_numeric": baseline_numeric,
                        "ablated_numeric": ablated_numeric,
                        "delta_numeric": ablated_numeric - baseline_numeric,
                        "remaining_rows": len(filtered),
                    }
                )

                log(
                    f"viz-removal | model={model_key} modality={modality} exclude={excluded_viz} "
                    f"delta_em={ablated_em - baseline_em:+.4f}",
                )

    summary_csv = output_dir / "viz_removal_summary.csv"
    _write_csv(
        summary_csv,
        summary_rows,
        [
            "model",
            "modality",
            "excluded_viz",
            "baseline_em",
            "ablated_em",
            "delta_em",
            "baseline_f1",
            "ablated_f1",
            "delta_f1",
            "baseline_numeric",
            "ablated_numeric",
            "delta_numeric",
            "remaining_rows",
        ],
    )

    latex_lines = [
        "\\begin{tabular}{lllrrr}",
        "\\toprule",
        "Model & Modality & Excluded Viz & Baseline EM (\\%) & Ablated EM (\\%) & Delta (pp) \\\\",
        "\\midrule",
    ]
    for row in sorted(
        summary_rows,
        key=lambda x: (str(x["model"]), str(x["modality"]), str(x["excluded_viz"])),
    ):
        latex_lines.append(
            "{} & {} & {} & {:.2f} & {:.2f} & {:+.2f} \\\\".format(
                _latex_escape(str(row["model"])),
                _latex_escape(str(row["modality"])),
                _latex_escape(str(row["excluded_viz"])),
                float(row["baseline_em"]) * 100.0,
                float(row["ablated_em"]) * 100.0,
                float(row["delta_em"]) * 100.0,
            )
        )
    latex_lines.extend(["\\bottomrule", "\\end{tabular}"])

    table_path = output_dir / "viz_removal_table.tex"
    table_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")
    log(f"Wrote viz-removal summary CSV: {summary_csv}")
    log(f"Wrote viz-removal LaTeX table: {table_path}")


def _build_stratified_sample(
    benchmark_path: Path,
    sample_size: int,
    seed: int,
) -> list[BenchmarkItem]:
    """Sample benchmark items stratified by (modality, difficulty)."""
    rows = read_jsonl(benchmark_path)
    all_items = [BenchmarkItem.from_dict(row) for row in rows]
    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")
    if len(all_items) < sample_size:
        raise ValueError(
            f"Requested sample_size={sample_size} but only {len(all_items)} items available"
        )

    by_stratum: dict[tuple[str, str], list[BenchmarkItem]] = defaultdict(list)
    for item in all_items:
        key = (item.modality, item.difficulty)
        by_stratum[key].append(item)

    total_items = len(all_items)
    allocations: dict[tuple[str, str], int] = {}
    fractions: list[tuple[float, tuple[str, str]]] = []
    allocated = 0

    for key, items in by_stratum.items():
        exact_target = sample_size * len(items) / total_items
        floor_target = int(exact_target)
        floor_target = min(floor_target, len(items))
        allocations[key] = floor_target
        allocated += floor_target
        fractions.append((exact_target - floor_target, key))

    remainder = sample_size - allocated
    for _, key in sorted(fractions, key=lambda x: (-x[0], x[1][0], x[1][1])):
        if remainder <= 0:
            break
        if allocations[key] < len(by_stratum[key]):
            allocations[key] += 1
            remainder -= 1

    if remainder > 0:
        capacity_keys = sorted(by_stratum.keys())
        idx = 0
        while remainder > 0 and capacity_keys:
            key = capacity_keys[idx % len(capacity_keys)]
            if allocations[key] < len(by_stratum[key]):
                allocations[key] += 1
                remainder -= 1
            idx += 1
            if idx > sample_size * 20:
                break

    if sum(allocations.values()) != sample_size:
        raise RuntimeError(
            f"Could not allocate exact sample size: target={sample_size}, got={sum(allocations.values())}"
        )

    rng = random.Random(seed)
    sampled: list[BenchmarkItem] = []
    for key in sorted(by_stratum.keys()):
        group = by_stratum[key]
        take = allocations[key]
        if take > 0:
            sampled.extend(rng.sample(group, take))
    rng.shuffle(sampled)
    return sampled


def _load_existing_prompt_rows(
    output_path: Path,
) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    """Load existing prompt-ablation rows and done key set."""
    if not output_path.exists():
        return [], set()
    rows = read_jsonl(output_path)
    retained_rows: list[dict[str, Any]] = []
    done: set[tuple[str, str]] = set()
    retryable_rows = 0
    for row in rows:
        if str(row.get("prediction", "")) == "[ERROR]":
            retryable_rows += 1
            continue
        retained_rows.append(row)
        done.add((str(row.get("question_id", "")), str(row.get("viz_type", ""))))
    if retryable_rows:
        log(
            f"Dropped {retryable_rows} retryable [ERROR] rows from {output_path} before resume"
        )
    return retained_rows, done


def _run_prompt_variant(
    *,
    model: Any,
    model_key: str,
    variant_name: str,
    system_prompt: str,
    sampled_items: list[BenchmarkItem],
    output_path: Path,
    checkpoint_every: int,
    max_retries: int,
    resume: bool,
    shutdown: ShutdownState,
) -> list[dict[str, Any]]:
    """Run one prompt variant and write variant-specific JSONL output."""
    existing_rows: list[dict[str, Any]]
    done_keys: set[tuple[str, str]]
    if resume:
        existing_rows, done_keys = _load_existing_prompt_rows(output_path)
        if existing_rows:
            log(
                f"[{variant_name}] resume loaded {len(existing_rows)} rows from {output_path}",
            )
    else:
        existing_rows, done_keys = [], set()

    setattr(model, "system_prompt", system_prompt)

    rpm = int(MODEL_SPECS[model_key]["rpm"])
    min_interval = 60.0 / float(rpm)
    last_call_ts = 0.0

    pipeline = RenderPipeline()
    results = list(existing_rows)
    errors = 0
    new_count = 0
    items_processed = 0
    start_time = time.time()

    for idx, item in enumerate(sampled_items, start=1):
        if shutdown.requested:
            break

        undone = [
            vt for vt in item.viz_methods if (item.question_id, vt) not in done_keys
        ]
        if not undone:
            continue

        items_processed += 1

        try:
            render_input = {
                "modality": item.modality,
                "data": item.data,
                "data_meta": item.metadata.get("data_meta") if item.metadata else None,
                "viz_titles": item.metadata.get("viz_titles", {})
                if item.metadata
                else {},
            }
            rendered = pipeline.render_all(render_input)
        except Exception as error:  # noqa: BLE001
            log(f"[{variant_name}] RENDER ERROR item {idx} {item.question_id}: {error}")
            for vt in undone:
                results.append(
                    compute_row(
                        item=item,
                        viz_type=vt,
                        prediction="[ERROR]",
                        prompt_variant=variant_name,
                    )
                )
                done_keys.add((item.question_id, vt))
                errors += 1
            continue

        for vt in undone:
            if shutdown.requested:
                break
            if vt not in rendered:
                continue

            now = time.time()
            elapsed = now - last_call_ts
            if last_call_ts > 0 and elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            prediction = call_with_retries(
                model=model,
                question=item.question,
                image=rendered[vt],
                metadata={"task": item.task},
                max_retries=max_retries,
            )
            last_call_ts = time.time()

            if prediction == "[ERROR]":
                errors += 1

            results.append(
                compute_row(
                    item=item,
                    viz_type=vt,
                    prediction=prediction,
                    prompt_variant=variant_name,
                )
            )
            done_keys.add((item.question_id, vt))
            new_count += 1

        if items_processed % checkpoint_every == 0 or shutdown.requested:
            write_jsonl(output_path, results)
            elapsed_total = time.time() - start_time
            rate = new_count / elapsed_total * 3600.0 if elapsed_total > 0 else 0.0
            log(
                f"[{variant_name}] item {idx}/{len(sampled_items)} | rows={len(results)} "
                f"new={new_count} err={errors} rate={rate:.0f}/h",
            )

    write_jsonl(output_path, results)
    elapsed_total = time.time() - start_time
    log(
        f"[{variant_name}] {'SHUTDOWN' if shutdown.requested else 'DONE'}: "
        f"rows={len(results)} new={new_count} err={errors} "
        f"elapsed={elapsed_total / 3600.0:.2f}h",
    )
    return results


def run_prompt_sensitivity(args: argparse.Namespace) -> None:
    """Run prompt sensitivity ablation with stratified sampling and checkpointing."""
    config = _read_yaml(args.config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_size = int(config.get("sample_size", 500))
    seed = int(config.get("seed", 42))

    prompt_variants_raw = config.get("prompt_variants", {})
    if not isinstance(prompt_variants_raw, dict) or not prompt_variants_raw:
        raise ValueError("prompt_variants must be a non-empty mapping")

    prompt_variants: dict[str, str] = {}
    for variant_name, variant_config in prompt_variants_raw.items():
        if not isinstance(variant_config, dict):
            raise ValueError(f"Prompt variant config must be mapping: {variant_name}")
        system_prompt = str(variant_config.get("system_prompt", "")).strip()
        if not system_prompt:
            raise ValueError(f"Missing system_prompt in variant: {variant_name}")
        prompt_variants[str(variant_name)] = system_prompt

    sampled_items = _build_stratified_sample(
        benchmark_path=args.benchmark,
        sample_size=sample_size,
        seed=seed,
    )
    log(
        f"Sampled {len(sampled_items)} items (stratified by modality+difficulty, seed={seed})"
    )

    shutdown = ShutdownState()
    _install_signal_handlers(shutdown)

    log(f"Building model: {args.model}...")
    model = build_model(args.model, gpu=args.gpu)
    log("Model ready.")

    summary_rows: list[dict[str, Any]] = []
    for variant_name, system_prompt in prompt_variants.items():
        if shutdown.requested:
            break

        output_path = output_dir / f"prompt_{args.model}_{variant_name}.jsonl"
        log(f"Running prompt variant: {variant_name} -> {output_path}")
        rows = _run_prompt_variant(
            model=model,
            model_key=args.model,
            variant_name=variant_name,
            system_prompt=system_prompt,
            sampled_items=sampled_items,
            output_path=output_path,
            checkpoint_every=args.checkpoint_every,
            max_retries=args.max_retries,
            resume=args.resume,
            shutdown=shutdown,
        )

        em, f1, numeric = recompute_metric_means(rows)
        errors = sum(1 for row in rows if str(row.get("prediction", "")) == "[ERROR]")
        summary_rows.append(
            {
                "model": args.model,
                "variant": variant_name,
                "rows": len(rows),
                "errors": errors,
                "em": em,
                "f1": f1,
                "numeric_accuracy": numeric,
            }
        )

    summary_csv = output_dir / "prompt_sensitivity_summary.csv"
    _write_csv(
        summary_csv,
        summary_rows,
        ["model", "variant", "rows", "errors", "em", "f1", "numeric_accuracy"],
    )
    log(f"Wrote prompt-sensitivity summary CSV: {summary_csv}")


def main() -> None:
    """Entry point for ablation command dispatch."""
    args = parse_args()
    if args.command == "viz-removal":
        run_viz_removal(args)
        return
    if args.command == "prompt-sensitivity":
        run_prompt_sensitivity(args)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
