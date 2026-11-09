"""Merge shard outputs of eval_local_suite.py, dedupe, and report coverage.

Dedupe key: (question_id, viz_type). Among duplicates a non-[ERROR] row wins
over an [ERROR] row; otherwise the latest row (later file / later line) wins.

Usage:
    python scripts/merge_shards.py results/suite/qwen.shard*.jsonl \
        --output results/suite/qwen.jsonl [--manifest suites/x/manifest.jsonl]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("inputs", nargs="+", type=Path, help="Shard JSONL files.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=None,
                   help="Optional manifest to report missing (question_id, viz_type) keys.")
    return p.parse_args()


def _is_err(row: dict[str, Any]) -> bool:
    return str(row.get("prediction", "")) == "[ERROR]"


def main() -> None:
    """Entry point."""
    args = parse_args()
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    n_in = n_bad = n_dup = 0
    for path in args.inputs:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    n_bad += 1
                    continue
                n_in += 1
                key = (str(row["question_id"]), str(row["viz_type"]))
                prev = merged.get(key)
                if prev is not None:
                    n_dup += 1
                    if _is_err(row) and not _is_err(prev):
                        continue
                merged[key] = row

    rows = sorted(merged.values(), key=lambda r: (r["question_id"], r["viz_type"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"inputs={len(args.inputs)} rows_read={n_in} unparseable={n_bad} "
          f"duplicates={n_dup} merged={len(rows)} -> {args.output}")
    err = [r for r in rows if _is_err(r)]
    print(f"error rows: {len(err)}")
    for r in err[:10]:
        print(f"  {r['question_id']} {r['viz_type']}: {str(r.get('error', ''))[:150]}")

    by_mv = Counter((r["modality"], r["viz_type"]) for r in rows)
    err_mv = Counter((r["modality"], r["viz_type"]) for r in err)
    em: dict[tuple[str, str], float] = Counter()
    for r in rows:
        em[(r["modality"], r["viz_type"])] += float(r.get("exact_match", 0.0))
    print(f"\n{'modality':<11} {'viz_type':<18} {'n':>6} {'err':>5} {'EM':>6}")
    for k in sorted(by_mv):
        print(f"{k[0]:<11} {k[1]:<18} {by_mv[k]:>6} {err_mv[k]:>5} {em[k] / by_mv[k]:>6.3f}")
    by_m = Counter(r["modality"] for r in rows)
    print("per modality:", dict(sorted(by_m.items())))
    suites = Counter(str(r.get("suite")) for r in rows)
    print("suites:", dict(suites))

    if args.manifest is not None:
        want: set[tuple[str, str]] = set()
        with open(args.manifest, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    m = json.loads(line)
                    want.add((str(m["question_id"]), str(m["viz_type"])))
        missing = want - set(merged)
        extra = set(merged) - want
        print(f"manifest keys={len(want)} missing={len(missing)} not_in_manifest={len(extra)}")
        for k in sorted(missing)[:10]:
            print("  missing:", k)


if __name__ == "__main__":
    main()
