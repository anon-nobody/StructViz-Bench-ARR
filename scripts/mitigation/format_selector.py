#!/usr/bin/env python3
"""Learned FORMAT SELECTOR prototype (GPU-free, offline) for StructViz-Bench.

Motivation: the format-ensemble oracle (correct-if-any-format) is 40-67%, far above the
mean single-format EM. A cheap per-(modality,task) selector -- "for this kind of question,
which rendering does this model read best?" -- should capture much of that headroom WITHOUT
touching model weights.

Method: split base questions 50/50 by a deterministic hash of question_id. On TRAIN, compute
the best viz_type per (modality, task) by mean EM. On TEST, for each question pick that viz and
read the model's EM for it. Report:
  mean-single (test)  : expected accuracy of a random format (baseline a user gets)
  selector(mod,task)  : learned lookup table over (modality, task)
  selector(mod)       : coarser lookup over modality only
  oracle (test)       : correct-if-any-format (upper bound)

Usage: python format_selector.py results/full_gpt4o.jsonl [more...]
"""
from __future__ import annotations
import json, sys, hashlib
from collections import defaultdict


def split(qid: str) -> str:
    h = int(hashlib.md5(qid.encode()).hexdigest(), 16)
    return "train" if (h % 2 == 0) else "test"


def load(path):
    # per question_id: {viz: em}; plus (modality,task) tag
    q = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        qid = r["question_id"]
        rec = q.setdefault(qid, {"mod": r.get("modality", ""), "task": r.get("task", ""),
                                 "viz": {}})
        rec["viz"][r.get("viz_type", "")] = float(r.get("exact_match", 0) or 0)
    return q


def best_viz_table(qs, key):
    agg = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))  # k -> viz -> [sum,n]
    for qid, rec in qs.items():
        if split(qid) != "train":
            continue
        k = key(rec)
        for viz, em in rec["viz"].items():
            agg[k][viz][0] += em
            agg[k][viz][1] += 1
    best = {}
    for k, vd in agg.items():
        best[k] = max(vd.items(), key=lambda kv: kv[1][0] / kv[1][1])[0]
    return best


def eval_selector(qs, best, key):
    n = hit = 0
    for qid, rec in qs.items():
        if split(qid) != "test":
            continue
        n += 1
        viz = best.get(key(rec))
        if viz is not None and viz in rec["viz"]:
            hit += rec["viz"][viz]
        else:  # unseen key/format -> fall back to mean over available
            hit += sum(rec["viz"].values()) / len(rec["viz"])
    return 100 * hit / n if n else 0.0


def baseline_and_oracle(qs):
    n = 0; single = 0.0; orc = 0.0
    for qid, rec in qs.items():
        if split(qid) != "test":
            continue
        n += 1
        ems = list(rec["viz"].values())
        single += sum(ems) / len(ems)
        orc += 1.0 if any(e >= 0.5 for e in ems) else 0.0
    return (100 * single / n, 100 * orc / n) if n else (0, 0)


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    print(f"{'model':26} {'meanSingle':>10} {'sel(mod,task)':>13} {'sel(mod)':>9} {'oracle':>7}")
    print("-" * 72)
    for path in sys.argv[1:]:
        qs = load(path)
        bt = best_viz_table(qs, lambda r: (r["mod"], r["task"]))
        bm = best_viz_table(qs, lambda r: (r["mod"],))
        s_mt = eval_selector(qs, bt, lambda r: (r["mod"], r["task"]))
        s_m = eval_selector(qs, bm, lambda r: (r["mod"],))
        single, orc = baseline_and_oracle(qs)
        print(f"{path.split('/')[-1]:26} {single:10.2f} {s_mt:13.2f} {s_m:9.2f} {orc:7.2f}")
    print("-" * 72)
    print("sel(mod,task): learned best-format lookup per (modality,task), train->test.")
    print("Gain over meanSingle = headroom captured with NO weight update. oracle = ceiling.")


if __name__ == "__main__":
    main()
