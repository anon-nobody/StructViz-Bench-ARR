#!/usr/bin/env python3
"""Merge filled per-annotator sheets into the long-format ratings CSV that
human_eval_aggregate.py consumes. Pure stdlib.

Usage:
  python assemble_ratings.py --sheets ratings_annotator1.csv ratings_annotator2.csv \
      --out ratings_long.csv
Then:
  python human_eval_aggregate.py --ratings ratings_long.csv
"""
from __future__ import annotations
import argparse, csv

VALID = {"correct": "Correct", "ambiguous": "Ambiguous", "incorrect": "Incorrect",
         "c": "Correct", "a": "Ambiguous", "i": "Incorrect"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", nargs="+", required=True)
    ap.add_argument("--out", default="ratings_long.csv")
    a = ap.parse_args()
    rows = []
    n_blank = 0
    for si, path in enumerate(a.sheets, 1):
        ev = f"ann{si}"
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                raw = (r.get("rating") or "").strip().lower()
                if not raw:
                    n_blank += 1
                    continue
                if raw not in VALID:
                    raise SystemExit(f"[{path}] item {r.get('item_id')}: bad rating "
                                     f"{r.get('rating')!r} (use Correct/Ambiguous/Incorrect)")
                rows.append((r["item_id"], ev, VALID[raw]))
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["item_id", "evaluator", "rating"])
        w.writerows(rows)
    print(f"wrote {len(rows)} ratings from {len(a.sheets)} sheets -> {a.out}"
          + (f"  ({n_blank} blank cells skipped)" if n_blank else ""))
    print("next: python human_eval_aggregate.py --ratings " + a.out)


if __name__ == "__main__":
    main()
