#!/usr/bin/env python3
"""Evaluate the proprietary models on the corrected (v2) rendering suite.

Same system prompt, decoding (temperature 0) and scoring as the March 2026 runs
(scripts/run_fullscale_eval.py); images are read from a v2 manifest instead of being
re-rendered, so every model sees byte-identical inputs (sha256 checked).

    export OPENAI_API_KEY=...   # and/or GEMINI_API_KEY, ANTHROPIC_API_KEY
    python scripts/review/run_api_v2.py --model gpt4o                      # strict subset (5,480 rows)
    python scripts/review/run_api_v2.py --model claude --manifest benchmark/render_v2/manifest.jsonl  # full (18,975)

Output: results/review/v2api_<model>[_full].jsonl, resumable (rows already present are skipped).
Each row keeps the raw response, the exact provider model string and a UTC timestamp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.answer_extractor import extract_answer  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.models.api_models import ClaudeModel, GeminiModel, GPT4oModel  # noqa: E402

MODELS = {
    # key: (class, provider model string, requests per minute)
    "gpt4o": (GPT4oModel, "gpt-4o", 50),
    "gemini": (GeminiModel, "gemini-2.0-flash", 30),
    "gemini25": (GeminiModel, "gemini-2.5-flash", 30),
    "claude": (ClaudeModel, "claude-sonnet-4-20250514", 40),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--model-string", default=None,
                    help="override the provider model string (e.g. a dated snapshot)")
    ap.add_argument("--manifest", type=Path, default=ROOT / "benchmark/render_v2/manifest_subset32b.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="stop after N new rows (0 = all)")
    args = ap.parse_args()
    cls, name, rpm = MODELS[args.model]
    name = args.model_string or name
    model = cls(name=name, requests_per_minute=rpm)
    full = "_full" if args.manifest.name == "manifest.jsonl" else ""
    out = ROOT / "results/review" / f"v2api_{args.model}{full}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open():
            if line.strip():
                r = json.loads(line)
                if r.get("prediction") != "[ERROR]":
                    done.add((r["question_id"], r["viz_type"]))
    base = {}
    for line in (ROOT / "benchmark/realworld_test.jsonl").open():
        r = json.loads(line)
        base[r["question_id"]] = r
    rows = [json.loads(l) for l in args.manifest.open() if l.strip()]
    rows = [r for r in rows if "error" not in r and (r["question_id"], r["viz_type"]) not in done]
    print(f"{len(rows)} rows to query with {name} -> {out}")
    n = 0
    with out.open("a") as fh:
        for r in rows:
            path = ROOT / r["image_path"]
            data = path.read_bytes()
            if r.get("sha256") and hashlib.sha256(data).hexdigest() != r["sha256"]:
                raise SystemExit(f"sha256 mismatch for {path}")
            b = base[r["question_id"]]
            t0 = time.time()
            for attempt in range(5):
                try:
                    pred = str(model.answer(question=b["question"], image=Image.open(path).convert("RGB"),
                                            metadata={"modality": b["modality"], "task": b["task"]}))
                    break
                except Exception as err:  # noqa: BLE001
                    if attempt == 4:
                        print(f"  FAILED {r['question_id']} {r['viz_type']}: {str(err)[:150]}")
                        pred = "[ERROR]"
                    else:
                        time.sleep(min(3.0 ** attempt, 120.0))
            # exact_match: scored on the raw reply exactly as the March 2026 runs were;
            # prediction_extracted / exact: after the post-hoc extractor used for *_extracted.jsonl.
            m = compute_metrics(prediction=pred, answer=b["answer"])
            ext = extract_answer(pred)
            me = compute_metrics(ext, b["answer"], extract=False)
            fh.write(json.dumps({
                "question_id": b["question_id"], "question": b["question"], "answer": b["answer"],
                "prediction": pred, "prediction_extracted": ext, "exact": me.exact, "modality": b["modality"], "source": b.get("source"),
                "viz_type": r["viz_type"], "difficulty": b["difficulty"], "task": b["task"],
                "exact_match": m.exact, "f1": m.f1, "numeric_accuracy": m.numeric,
                "model_id": name, "image_path": r["image_path"], "image_sha256": r.get("sha256"),
                "suite": "v2", "latency_s": round(time.time() - t0, 3),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }) + "\n")
            fh.flush()
            n += 1
            if args.limit and n >= args.limit:
                break
    print(f"wrote {n} rows")


if __name__ == "__main__":
    main()
