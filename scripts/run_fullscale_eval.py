"""Full-scale StructViz-Bench evaluation for a single model.

Processes ALL benchmark items × all visualization methods.
Designed for long-running (5-15h) evaluation with robust checkpointing.

Usage (launch each in a separate tmux session):
    python scripts/run_fullscale_eval.py --model gpt4o --resume
    python scripts/run_fullscale_eval.py --model gemini --resume
    python scripts/run_fullscale_eval.py --model claude --resume
    python scripts/run_fullscale_eval.py --model qwen --gpu 3 --resume
"""

from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["PYTHONUNBUFFERED"] = "1"

import matplotlib

matplotlib.use("Agg")

from src.evaluation.metrics import compute_metrics
from src.models.api_models import GPT4oModel, GeminiModel, ClaudeModel
from src.rendering.render_pipeline import RenderPipeline
from src.utils.io_utils import BenchmarkItem, read_jsonl, write_jsonl

# ── API Keys (read from environment) ─────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Model Specs ──────────────────────────────────────────────────────────────
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
    "gemini25": {
        "name": "gemini-2.5-flash",
        "rpm": 30,
        "output": "full_gemini25.jsonl",
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
    "qwen32b": {
        "name": "Qwen/Qwen2.5-VL-32B-Instruct",
        "rpm": 999,
        "output": "full_qwen32b.jsonl",
    },
    "qwen72b": {
        "name": "Qwen/Qwen2.5-VL-72B-Instruct",
        "rpm": 999,
        "output": "full_qwen72b.jsonl",
    },
    "phi35v": {
        "name": "microsoft/Phi-3.5-vision-instruct",
        "rpm": 999,
        "output": "full_phi35v.jsonl",
    },
    "internvl": {
        "name": "OpenGVLab/InternVL2_5-8B",
        "rpm": 999,
        "output": "full_internvl.jsonl",
    },
    "llava": {
        "name": "llava-hf/llava-v1.6-mistral-7b-hf",
        "rpm": 999,
        "output": "full_llava.jsonl",
    },
}


def log(msg: str) -> None:
    """Log with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Full-scale StructViz-Bench evaluation for a single model.",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("benchmark/realworld_test.jsonl"),
        help="Benchmark JSONL path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for result JSONL files.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_SPECS.keys()),
        help="Model to evaluate.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=3,
        help="GPU index for local models (default: 3).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output JSONL.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Save checkpoint every N processed items (default: 5).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max API call retries per request (default: 5).",
    )
    return parser.parse_args()


def load_all_items(benchmark_path: Path) -> list[BenchmarkItem]:
    """Load ALL benchmark items without sampling."""
    rows = read_jsonl(benchmark_path)
    items = [BenchmarkItem.from_dict(row) for row in rows]
    log(f"Loaded {len(items)} benchmark items from {benchmark_path}")
    return items


def load_existing(
    output_path: Path,
) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    """Load existing results and build done-key set for resume."""
    if not output_path.exists():
        return [], set()
    rows: list[dict[str, Any]] = []
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    retained_rows: list[dict[str, Any]] = []
    done: set[tuple[str, str]] = set()
    retryable_rows = 0
    for row in rows:
        if str(row.get("prediction", "")) == "[ERROR]":
            retryable_rows += 1
            continue
        retained_rows.append(row)
        done.add((str(row.get("question_id", "")), str(row.get("viz_type", ""))))
    log(
        f"Resumed {len(retained_rows)} retained rows, {len(done)} done (q_id, viz_type) pairs"
        + (
            f"; dropped {retryable_rows} retryable [ERROR] rows"
            if retryable_rows
            else ""
        )
    )
    return retained_rows, done


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

    if model_key == "gemini25":
        os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
        return GeminiModel(
            name="gemini-2.5-flash",
            requests_per_minute=MODEL_SPECS["gemini25"]["rpm"],
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

    if model_key == "qwen32b":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        from src.models.local_models import QwenVLModel

        return QwenVLModel(
            name="Qwen2.5-VL-32B-Instruct",
            checkpoint="Qwen/Qwen2.5-VL-32B-Instruct",
            device="cuda",
            dtype="bfloat16",
            use_vllm=True,
        )

    if model_key == "qwen72b":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        from src.models.local_models import QwenVLModel

        return QwenVLModel(
            name="Qwen2.5-VL-72B-Instruct",
            checkpoint="Qwen/Qwen2.5-VL-72B-Instruct",
            device="cuda",
            dtype="bfloat16",
            use_vllm=True,
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
    """Call model.answer() with retry logic and exponential backoff."""
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

            # Parse Retry-After header if available
            wait: float | None = None
            response = getattr(error, "response", None)
            headers = getattr(response, "headers", None)
            if headers:
                retry_after = None
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
                        pass

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
) -> dict[str, Any]:
    """Build a result row with computed metrics."""
    bundle = compute_metrics(prediction=prediction, answer=item.answer)
    return {
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


def main() -> None:
    """Run full-scale evaluation."""
    args = parse_args()
    spec = MODEL_SPECS[args.model]
    output_path = args.output_dir / spec["output"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ALL benchmark items (no sampling) ───────────────────────────
    all_items = load_all_items(args.benchmark)

    # Count total expected rows (items × viz_methods)
    total_rows = sum(len(item.viz_methods) for item in all_items)

    # ── Resume support ───────────────────────────────────────────────────
    results: list[dict[str, Any]]
    done_keys: set[tuple[str, str]]
    if args.resume:
        results, done_keys = load_existing(output_path)
    else:
        results, done_keys = [], set()

    remaining = total_rows - len(done_keys)
    log(
        f"Model: {spec['name']} | Items: {len(all_items)} | "
        f"Total rows: {total_rows} | Done: {len(done_keys)} | "
        f"Remaining: {remaining}",
    )

    if remaining <= 0:
        log("All rows already complete. Nothing to do.")
        return

    # ── Build model ──────────────────────────────────────────────────────
    log(f"Building model: {args.model}...")
    model = build_model(args.model, gpu=args.gpu)
    log("Model ready.")

    # ── Rendering pipeline ───────────────────────────────────────────────
    pipeline = RenderPipeline()

    # ── Rate limiting ────────────────────────────────────────────────────
    rpm = int(spec["rpm"])
    min_interval = 60.0 / float(rpm)
    last_call_ts = 0.0

    # ── Progress tracking ────────────────────────────────────────────────
    new_count = 0
    errors = 0
    start_time = time.time()
    items_processed = 0

    # ── Graceful shutdown ────────────────────────────────────────────────
    shutdown = False

    def handle_signal(sig: int, frame: Any) -> None:
        nonlocal shutdown
        log(
            f"Signal {sig} received — finishing current item, "
            "then saving and exiting...",
        )
        shutdown = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # ── Main evaluation loop ─────────────────────────────────────────────
    for idx, item in enumerate(all_items, start=1):
        if shutdown:
            break

        # Which viz types still need processing?
        undone = [
            vt for vt in item.viz_methods if (item.question_id, vt) not in done_keys
        ]
        if not undone:
            continue

        items_processed += 1

        # Render all viz types for this item
        try:
            render_input = {
                "modality": item.modality,
                "data": item.data,
                "data_meta": (
                    item.metadata.get("data_meta") if item.metadata else None
                ),
                "viz_titles": (
                    item.metadata.get("viz_titles", {}) if item.metadata else {}
                ),
            }
            rendered = pipeline.render_all(render_input)
        except Exception as e:  # noqa: BLE001
            log(f"RENDER ERROR item {idx} {item.question_id}: {e}")
            for vt in undone:
                results.append(compute_row(item, vt, "[ERROR]"))
                done_keys.add((item.question_id, vt))
                errors += 1
            continue

        for vt in undone:
            if shutdown:
                break
            if vt not in rendered:
                continue

            # Rate limiting
            now = time.time()
            elapsed = now - last_call_ts
            if last_call_ts > 0 and elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            # API call with retries
            prediction = call_with_retries(
                model=model,
                question=item.question,
                image=rendered[vt],
                metadata={"task": item.task},
                max_retries=args.max_retries,
            )
            last_call_ts = time.time()

            if prediction == "[ERROR]":
                errors += 1

            results.append(compute_row(item, vt, prediction))
            done_keys.add((item.question_id, vt))
            new_count += 1

        # Checkpoint + progress logging
        if items_processed % args.checkpoint_every == 0 or shutdown:
            write_jsonl(output_path, results)

            elapsed_total = time.time() - start_time
            rate = new_count / elapsed_total * 3600 if elapsed_total > 0 else 0
            pct = len(done_keys) / total_rows * 100
            remaining_now = total_rows - len(done_keys)
            if new_count > 0 and elapsed_total > 0:
                eta_seconds = remaining_now / (new_count / elapsed_total)
                eta_h = eta_seconds / 3600
            else:
                eta_h = 0.0

            log(
                f"Item {idx}/{len(all_items)} | "
                f"rows={len(results)}/{total_rows} ({pct:.1f}%) | "
                f"new={new_count} err={errors} | "
                f"rate={rate:.0f}/h | "
                f"ETA={eta_h:.1f}h",
            )

    # ── Final save ───────────────────────────────────────────────────────
    write_jsonl(output_path, results)
    elapsed_total = time.time() - start_time
    log(
        f"{'SHUTDOWN' if shutdown else 'DONE'}: "
        f"{len(results)} total rows, {new_count} new, {errors} errors, "
        f"{elapsed_total / 3600:.1f}h elapsed",
    )


if __name__ == "__main__":
    main()
