#!/usr/bin/env python3
"""Learned format selector on the cross-modal (Level-2) questions.

Each L2 question pairs two objects of different modalities and is rendered in five paired
formats. We learn the best paired format per (modality-pair, task) on half of the questions
(deterministic hash split by question id) and evaluate on the other half against three references.

CAVEAT (not used in the paper): the released L2 set is built from ONE (table, series, graph)
triple and 30 unique (pair, question, answer) problems, each repeated 15-25 times under
different item ids. A hash split by item id therefore puts repeats of the same problem on both
sides, and a question->answer lookup already scores 100% on the test half. The numbers this
script prints measure nothing about generalisation and are reported only for completeness.

  random        expected accuracy of an arbitrary paired format
  fixed (pair)  best single paired format per modality pair, learned on train
  oracle        correct if any paired format is correct

Usage: python scripts/mitigation/format_selector_l2.py [results/mixed_<model>_extracted.jsonl ...]
"""
from __future__ import annotations
import hashlib, json, os, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT = [f"results/mixed_{m}_extracted.jsonl" for m in ("gpt4o", "gemini", "qwen", "claude")]


def side(qid: str) -> str:
    return "train" if int(hashlib.md5(qid.encode()).hexdigest(), 16) % 2 == 0 else "test"


def run(path: str):
    byq = defaultdict(dict); tag = {}
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        byq[r["question_id"]][r["viz_type"]] = float(r.get("exact_match", 0))
        tag[r["question_id"]] = (r["modality"], r.get("task", "?"))
    acc_mt = defaultdict(lambda: defaultdict(list)); acc_m = defaultdict(lambda: defaultdict(list))
    for q, fm in byq.items():
        if side(q) != "train":
            continue
        for v, e in fm.items():
            acc_mt[tag[q]][v].append(e); acc_m[tag[q][0]][v].append(e)
    best_mt = {k: max(d, key=lambda v: sum(d[v]) / len(d[v])) for k, d in acc_mt.items()}
    best_m = {k: max(d, key=lambda v: sum(d[v]) / len(d[v])) for k, d in acc_m.items()}
    n = rnd = smt = sm = orc = 0
    for q, fm in byq.items():
        if side(q) != "test":
            continue
        mean = sum(fm.values()) / len(fm)
        n += 1; rnd += mean; orc += max(fm.values())
        mt = best_mt.get(tag[q]) or best_m.get(tag[q][0]); m = best_m.get(tag[q][0])
        smt += fm.get(mt, mean); sm += fm.get(m, mean)
    return n, 100 * rnd / n, 100 * sm / n, 100 * smt / n, 100 * orc / n


if __name__ == "__main__":
    files = sys.argv[1:] or DEFAULT
    print(f"{'file':32s} {'n_test':>6s} {'random':>7s} {'fixed(pair)':>11s} {'sel(pair,task)':>14s} {'oracle':>7s} | {'d vs fixed':>10s}")
    for f in files:
        p = f if os.path.isabs(f) else os.path.join(ROOT, f)
        if not os.path.exists(p):
            print(f"{os.path.basename(f):32s} missing"); continue
        n, r, m, mt, o = run(p)
        print(f"{os.path.basename(f):32s} {n:6d} {r:7.1f} {m:11.1f} {mt:14.1f} {o:7.1f} | {mt-m:+10.1f}")
