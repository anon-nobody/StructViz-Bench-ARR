from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false
# pyright: reportUnusedCallResult=false

import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import signal
import time
from pathlib import Path
from typing import Any

from src.evaluation.metrics import compute_metrics
from src.models.api_models import ClaudeModel
from src.rendering.render_pipeline import RenderPipeline
from src.utils.io_utils import BenchmarkItem, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry Claude [ERROR] rows only.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("benchmark/realworld_test.jsonl"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/full_claude.jsonl"),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="claude-sonnet-4-20250514",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=30,
        help="API requests per minute.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="If >0, only process this many error rows (for testing).",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create .bak backup for results file.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=90,
        help="Per-request timeout in seconds.",
    )
    return parser.parse_args()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def call_with_retries(
    model: ClaudeModel,
    question: str,
    image: Any,
    metadata: dict[str, Any],
    max_retries: int,
    request_timeout: int,
) -> str:
    for attempt in range(max_retries):
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    model.answer,
                    question=question,
                    image=image,
                    metadata=metadata,
                )
                result = future.result(timeout=request_timeout)
            return str(result)
        except FutureTimeoutError:
            if attempt == max_retries - 1:
                log(
                    f"API FAILED after {max_retries} retries: timeout>{request_timeout}s"
                )
                return "[ERROR]"
            wait = min(2.0**attempt, 30.0)
            log(
                f"Retry {attempt + 1}/{max_retries}: timeout>{request_timeout}s; sleeping {wait:.1f}s",
            )
            time.sleep(wait)
        except Exception as error:  # noqa: BLE001
            if attempt == max_retries - 1:
                log(f"API FAILED after {max_retries} retries: {str(error)[:160]}")
                return "[ERROR]"
            wait = min(2.0**attempt, 30.0)
            log(
                f"Retry {attempt + 1}/{max_retries}: {str(error)[:120]}; sleeping {wait:.1f}s",
            )
            time.sleep(wait)
    return "[ERROR]"


def main() -> None:
    args = parse_args()

    if args.backup:
        backup_path = args.results.with_suffix(args.results.suffix + ".bak")
        backup_path.write_text(
            args.results.read_text(encoding="utf-8"), encoding="utf-8"
        )
        log(f"Backup created: {backup_path}")

    all_items = [BenchmarkItem.from_dict(r) for r in read_jsonl(args.benchmark)]
    rows = read_jsonl(args.results)

    row_idx_by_key: dict[tuple[str, str], int] = {}
    error_keys: set[tuple[str, str]] = set()
    for i, row in enumerate(rows):
        qid = str(row.get("question_id", ""))
        viz = str(row.get("viz_type", ""))
        key = (qid, viz)
        row_idx_by_key[key] = i
        if str(row.get("prediction", "")) == "[ERROR]":
            error_keys.add(key)

    if args.max_items > 0:
        limited: list[tuple[str, str]] = list(error_keys)[: args.max_items]
        error_keys = set(limited)

    total_errors = len(error_keys)
    log(f"Loaded rows={len(rows)}, target_error_rows={total_errors}")
    if total_errors == 0:
        log("No [ERROR] rows to retry.")
        return

    model = ClaudeModel(name=args.model_name, requests_per_minute=args.rpm)
    pipeline = RenderPipeline()

    min_interval = 60.0 / float(args.rpm)
    last_call_ts = 0.0
    retried = 0
    recovered = 0
    still_error = 0

    shutdown = False

    def handle_signal(sig: int, frame: Any) -> None:
        del frame
        nonlocal shutdown
        shutdown = True
        log(f"Signal {sig} received. Saving and exiting soon.")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    for item in all_items:
        if shutdown:
            break
        pending_viz = [
            vt for vt in item.viz_methods if (item.question_id, vt) in error_keys
        ]
        if not pending_viz:
            continue

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
        except Exception as e:  # noqa: BLE001
            log(f"RENDER ERROR {item.question_id}: {e}")
            for vt in pending_viz:
                retried += 1
                still_error += 1
                error_keys.discard((item.question_id, vt))
            continue

        for vt in pending_viz:
            if shutdown:
                break
            img = rendered.get(vt)
            if img is None:
                retried += 1
                still_error += 1
                error_keys.discard((item.question_id, vt))
                continue

            now = time.time()
            elapsed = now - last_call_ts
            if last_call_ts > 0 and elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            pred = call_with_retries(
                model=model,
                question=item.question,
                image=img,
                metadata={"task": item.task},
                max_retries=args.max_retries,
                request_timeout=args.request_timeout,
            )
            last_call_ts = time.time()

            key = (item.question_id, vt)
            idx = row_idx_by_key[key]
            m = compute_metrics(prediction=pred, answer=item.answer)
            rows[idx]["prediction"] = pred
            rows[idx]["exact_match"] = m.exact
            rows[idx]["f1"] = m.f1
            rows[idx]["numeric_accuracy"] = m.numeric

            retried += 1
            if pred == "[ERROR]":
                still_error += 1
            else:
                recovered += 1
            error_keys.discard(key)

            if retried % args.checkpoint_every == 0:
                write_jsonl(args.results, rows)
                log(
                    f"progress {retried}/{total_errors} | recovered={recovered} | "
                    f"still_error={still_error} | remaining={len(error_keys)}",
                )

    write_jsonl(args.results, rows)
    log(
        f"DONE retry_error_rows={retried}/{total_errors} | recovered={recovered} | "
        f"still_error={still_error}",
    )


if __name__ == "__main__":
    main()
