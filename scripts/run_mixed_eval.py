from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import argparse
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

from src.evaluation.metrics import compute_metrics
from src.models.api_models import ClaudeModel, GPT4oModel, GeminiModel
from src.rendering.mixed_renderer import MixedRenderer
from src.utils.io_utils import BenchmarkItem, read_jsonl, write_jsonl

# ── API Keys (read from environment) ──────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "gpt4o": {
        "name": "gpt-4o",
        "rpm": 50,
        "output": "mixed_gpt4o.jsonl",
    },
    "gemini": {
        "name": "gemini-2.0-flash",
        "rpm": 30,
        "output": "mixed_gemini.jsonl",
    },
    "claude": {
        "name": "claude-sonnet-4-20250514",
        "rpm": 40,
        "output": "mixed_claude.jsonl",
    },
    "qwen": {
        "name": "Qwen2.5-VL-7B-Instruct",
        "rpm": 999,
        "output": "mixed_qwen.jsonl",
    },
    "internvl": {
        "name": "OpenGVLab/InternVL2_5-8B",
        "rpm": 999,
        "output": "mixed_internvl.jsonl",
    },
    "llava": {
        "name": "llava-hf/llava-v1.6-mistral-7b-hf",
        "rpm": 999,
        "output": "mixed_llava.jsonl",
    },
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mixed-type StructViz-Bench evaluation for a single model.",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("benchmark/mixed_items.jsonl"),
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
    rows = read_jsonl(benchmark_path)
    items = [BenchmarkItem.from_dict(row) for row in rows]
    log(f"Loaded {len(items)} benchmark items from {benchmark_path}")
    return items


def load_existing(
    output_path: Path,
) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
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
    if model_key == "gpt4o":
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        return GPT4oModel(
            name="gpt-4o", requests_per_minute=MODEL_SPECS["gpt4o"]["rpm"]
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
                wait = (
                    min(3.0**attempt, 120.0)
                    if is_rate_limited
                    else min(2.0**attempt, 30.0)
                )

            log(
                f"  Retry {attempt + 1}/{max_retries}: "
                f"{'RATE_LIMIT' if is_rate_limited else 'ERROR'}: "
                f"{message[:120]}; sleeping {wait:.1f}s",
            )
            time.sleep(wait)
    return "[ERROR]"


def compute_row(item: BenchmarkItem, viz_type: str, prediction: str) -> dict[str, Any]:
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
    args = parse_args()
    spec = MODEL_SPECS[args.model]
    output_path = args.output_dir / spec["output"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_items = load_all_items(args.benchmark)
    total_rows = sum(len(item.viz_methods) for item in all_items)

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

    log(f"Building model: {args.model}...")
    model = build_model(args.model, gpu=args.gpu)
    log("Model ready.")

    mixed_renderer = MixedRenderer()

    rpm = int(spec["rpm"])
    min_interval = 60.0 / float(rpm)
    last_call_ts = 0.0

    new_count = 0
    errors = 0
    start_time = time.time()
    items_processed = 0
    shutdown = False

    def handle_signal(sig: int, frame: Any) -> None:
        nonlocal shutdown
        log(
            f"Signal {sig} received — finishing current item, then saving and exiting..."
        )
        shutdown = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    for idx, item in enumerate(all_items, start=1):
        if shutdown:
            break

        undone = [
            vt for vt in item.viz_methods if (item.question_id, vt) not in done_keys
        ]
        if not undone:
            continue

        items_processed += 1

        for vt in undone:
            if shutdown:
                break

            try:
                image = mixed_renderer.render_mixed_item(
                    {
                        "modality": item.modality,
                        "data": item.data,
                        "metadata": item.metadata,
                    },
                    vt,
                )
            except Exception as error:  # noqa: BLE001
                log(f"RENDER ERROR item {idx} {item.question_id} {vt}: {error}")
                results.append(compute_row(item, vt, "[ERROR]"))
                done_keys.add((item.question_id, vt))
                new_count += 1
                errors += 1
                continue

            now = time.time()
            elapsed = now - last_call_ts
            if last_call_ts > 0 and elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            prediction = call_with_retries(
                model=model,
                question=item.question,
                image=image,
                metadata={"task": item.task},
                max_retries=args.max_retries,
            )
            last_call_ts = time.time()

            if prediction == "[ERROR]":
                errors += 1

            results.append(compute_row(item, vt, prediction))
            done_keys.add((item.question_id, vt))
            new_count += 1

        if items_processed % args.checkpoint_every == 0 or shutdown:
            write_jsonl(output_path, results)
            elapsed_total = time.time() - start_time
            rate = new_count / elapsed_total * 3600 if elapsed_total > 0 else 0
            pct = len(done_keys) / total_rows * 100 if total_rows > 0 else 0.0
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

    write_jsonl(output_path, results)
    elapsed_total = time.time() - start_time
    log(
        f"{'SHUTDOWN' if shutdown else 'DONE'}: "
        f"{len(results)} total rows, {new_count} new, {errors} errors, "
        f"{elapsed_total / 3600:.1f}h elapsed",
    )


if __name__ == "__main__":
    main()
