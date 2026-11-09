#!/usr/bin/env python3
"""Format selector with a GROUP split by source object (data_id), not by question.

The original selector hashes question_id 50/50, so the ~25 questions drawn from one
underlying table/series/graph land on both sides. This variant assigns every question of a
data_id to the same side, so held-out questions come from unseen source objects. Reports the
same quantities as format_selector.py plus a per-modality fixed-format baseline.
"""
from __future__ import annotations
import json, sys, hashlib, os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
qid2did = {}
for fn in ("benchmark/realworld_test.jsonl", "benchmark/base_items.jsonl"):
    p = os.path.join(ROOT, fn)
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l); qid2did.setdefault(r["question_id"], r.get("data_id"))

def side(did: str) -> str:
    h = int(hashlib.md5(did.encode()).hexdigest(), 16)
    return "train" if h % 2 == 0 else "test"

def run(path):
    byq = defaultdict(dict); tag = {}
    for l in open(path):
        r = json.loads(l); q = r["question_id"]
        byq[q][r["viz_type"]] = float(r.get("exact_match", 0))
        tag[q] = (r["modality"], r.get("task", "?"))
    def sp(q): return side(qid2did.get(q, q))
    # learn best format per (mod,task) and per mod on TRAIN
    acc_mt = defaultdict(lambda: defaultdict(list)); acc_m = defaultdict(lambda: defaultdict(list))
    for q, fm in byq.items():
        if sp(q) != "train": continue
        for v, e in fm.items():
            acc_mt[tag[q]][v].append(e); acc_m[tag[q][0]][v].append(e)
    best_mt = {k: max(d, key=lambda v: sum(d[v]) / len(d[v])) for k, d in acc_mt.items()}
    best_m = {k: max(d, key=lambda v: sum(d[v]) / len(d[v])) for k, d in acc_m.items()}
    # evaluate on TEST
    n = single = s_mt = s_m = orc = 0
    for q, fm in byq.items():
        if sp(q) != "test": continue
        n += 1
        single += sum(fm.values()) / len(fm)
        orc += max(fm.values())
        mt = best_mt.get(tag[q]) or best_m.get(tag[q][0]); m = best_m.get(tag[q][0])
        s_mt += fm.get(mt, sum(fm.values()) / len(fm)); s_m += fm.get(m, sum(fm.values()) / len(fm))
    ntr = sum(1 for q in byq if sp(q) == "train")
    return n, ntr, 100*single/n, 100*s_mt/n, 100*s_m/n, 100*orc/n

if __name__ == "__main__":
    files = sys.argv[1:] or [f"results/full_{m}_extracted.jsonl" for m in ("gpt4o","gemini","qwen","claude")] + \
            ["results/full_qwen32b.jsonl", "results/full_internvl.jsonl"]
    print(f"{'file':28s} {'n_test':>6s} {'n_train':>7s} {'single':>7s} {'sel(m,t)':>9s} {'sel(m)':>7s} {'oracle':>7s} | {'Δ vs single':>11s} {'Δ vs sel(m)':>11s}")
    for f in files:
        p = f if os.path.isabs(f) else os.path.join(ROOT, f)
        if not os.path.exists(p): continue
        n, ntr, s, smt, sm, o = run(p)
        print(f"{os.path.basename(f):28s} {n:6d} {ntr:7d} {s:7.2f} {smt:9.2f} {sm:7.2f} {o:7.2f} | {smt-s:+11.2f} {smt-sm:+11.2f}")
    dids = set(qid2did.values()); print(f"\ndata_id groups: {len(dids)}  (train {sum(1 for d in dids if side(d)=='train')}, test {sum(1 for d in dids if side(d)=='test')})")
