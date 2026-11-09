"""Continue Claude pilot evaluation with verbose progress logging."""

from __future__ import annotations

import json
import os
import random
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

from src.evaluation.metrics import compute_metrics
from src.models.api_models import ClaudeModel
from src.rendering.render_pipeline import RenderPipeline
from src.utils.io_utils import BenchmarkItem, read_jsonl, write_jsonl

# ── Config ──────────────────────────────────────────────────────────────────
BENCHMARK = Path("benchmark/realworld_test.jsonl")
OUTPUT = Path("results/pilot_claude.jsonl")
NUM_ITEMS = 200
SEED = 42
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
RPM = 40
MIN_INTERVAL = 60.0 / RPM  # 1.5s between calls

if not API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY environment variable must be set.")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sample_items() -> list[BenchmarkItem]:
    """Sample items identically to run_pilot_eval.py."""
    rows = read_jsonl(BENCHMARK)
    all_items = [BenchmarkItem.from_dict(r) for r in rows]
    log(f"Loaded {len(all_items)} benchmark items")

    by_mod: dict[str, list[BenchmarkItem]] = defaultdict(list)
    for it in all_items:
        by_mod[it.modality].append(it)

    rng = random.Random(SEED)
    base = NUM_ITEMS // 3
    rem = NUM_ITEMS % 3
    targets = {"tabular": base, "timeseries": base, "graph": base}
    if rem >= 1:
        targets["tabular"] += 1
    if rem >= 2:
        targets["timeseries"] += 1

    sampled: list[BenchmarkItem] = []
    for mod in ("tabular", "timeseries", "graph"):
        pool = by_mod[mod]
        syn = [i for i in pool if i.source == "synthetic"]
        real = [i for i in pool if i.source != "synthetic"]
        t = targets[mod]
        ds = t // 2
        dr = t - ds
        ts_cnt = min(len(syn), ds)
        tr_cnt = min(len(real), dr)
        remainder = t - ts_cnt - tr_cnt
        while remainder > 0:
            if len(real) - tr_cnt > len(syn) - ts_cnt and len(real) - tr_cnt > 0:
                tr_cnt += 1
            elif len(syn) - ts_cnt > 0:
                ts_cnt += 1
            elif len(real) - tr_cnt > 0:
                tr_cnt += 1
            remainder -= 1
        if ts_cnt > 0:
            sampled.extend(rng.sample(syn, ts_cnt))
        if tr_cnt > 0:
            sampled.extend(rng.sample(real, tr_cnt))

    rng.shuffle(sampled)
    log(f"Sampled {len(sampled)} items ({targets})")
    return sampled


def load_existing() -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    """Load existing results and build done-key set."""
    if not OUTPUT.exists():
        return [], set()
    rows = [json.loads(line) for line in open(OUTPUT)]
    done = set()
    for r in rows:
        done.add((str(r.get("question_id", "")), str(r.get("viz_type", ""))))
    log(f"Resumed {len(rows)} existing rows, {len(done)} done pairs")
    return rows, done


def main() -> None:
    sampled = sample_items()
    results, done_keys = load_existing()

    pipeline = RenderPipeline()
    model = ClaudeModel(name="claude-sonnet-4-20250514", requests_per_minute=RPM)

    new_count = 0
    errors = 0
    last_call_ts = 0.0

    # Graceful shutdown
    shutdown = False

    def handle_signal(sig: int, frame: Any) -> None:
        nonlocal shutdown
        log("Received signal, saving and exiting...")
        shutdown = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    for idx, item in enumerate(sampled, start=1):
        if shutdown:
            break

        undone = [
            vt for vt in item.viz_methods if (item.question_id, vt) not in done_keys
        ]
        if not undone:
            continue

        # Render
        t0 = time.time()
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
        except Exception as e:
            log(f"RENDER ERROR item {idx} {item.question_id}: {e}")
            for vt in undone:
                bundle = compute_metrics(prediction="[ERROR]", answer=item.answer)
                results.append(
                    {
                        "question_id": item.question_id,
                        "question": item.question,
                        "answer": item.answer,
                        "prediction": "[ERROR]",
                        "modality": item.modality,
                        "source": item.source,
                        "viz_type": vt,
                        "difficulty": item.difficulty,
                        "task": item.task,
                        "exact_match": bundle.exact,
                        "f1": bundle.f1,
                        "numeric_accuracy": bundle.numeric,
                    }
                )
                done_keys.add((item.question_id, vt))
                errors += 1
            continue

        for vt in undone:
            if shutdown:
                break
            if vt not in rendered:
                continue

            # Rate limit
            now = time.time()
            elapsed = now - last_call_ts
            if last_call_ts > 0 and elapsed < MIN_INTERVAL:
                time.sleep(MIN_INTERVAL - elapsed)

            # API call with timeout
            t1 = time.time()
            try:
                prediction = str(
                    model.answer(
                        question=item.question,
                        image=rendered[vt],
                        metadata={"task": item.task},
                    )
                )
            except Exception as e:
                prediction = "[ERROR]"
                errors += 1
                log(f"  API ERROR {vt}: {e}")

            last_call_ts = time.time()
            api_time = last_call_ts - t1

            bundle = compute_metrics(prediction=prediction, answer=item.answer)
            results.append(
                {
                    "question_id": item.question_id,
                    "question": item.question,
                    "answer": item.answer,
                    "prediction": prediction,
                    "modality": item.modality,
                    "source": item.source,
                    "viz_type": vt,
                    "difficulty": item.difficulty,
                    "task": item.task,
                    "exact_match": bundle.exact,
                    "f1": bundle.f1,
                    "numeric_accuracy": bundle.numeric,
                }
            )
            done_keys.add((item.question_id, vt))
            new_count += 1

        # Progress log every item
        total_done = len(
            [
                1
                for i2 in sampled
                if all((i2.question_id, v) in done_keys for v in i2.viz_methods)
            ]
        )
        log(
            f"Item {idx}/{len(sampled)} done ({item.modality}) | "
            f"rows={len(results)} new={new_count} errors={errors} | "
            f"items_complete={total_done}/{len(sampled)}"
        )

        # Checkpoint every 10 items
        if idx % 10 == 0 or shutdown:
            write_jsonl(OUTPUT, results)
            log(f"  CHECKPOINT saved {len(results)} rows")

    # Final save
    write_jsonl(OUTPUT, results)
    log(f"DONE: {len(results)} total rows, {new_count} new, {errors} errors")


if __name__ == "__main__":
    main()
