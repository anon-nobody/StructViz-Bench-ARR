#!/usr/bin/env python3
"""Deployable format selector: predict the task from the QUESTION TEXT (no oracle label),
then pick the best rendering for (modality, predicted-task). GPU-free, pure stdlib.

Removes the "selector needs the ground-truth task" caveat: a tiny multinomial Naive Bayes
classifier over question tokens predicts the task, and we show the selector's accuracy with
PREDICTED tasks nearly matches the oracle-task selector.

Split 50/50 by hash(question_id). On train: (a) NB task classifier per modality; (b) best
viz per (modality, task). On test: predict task -> pick viz -> read that viz's EM.

Usage: python selector_with_classifier.py results/full_gpt4o.jsonl [more...]
"""
from __future__ import annotations
import json, sys, hashlib, re, math
from collections import defaultdict, Counter


def split(qid):
    return "train" if int(hashlib.md5(qid.encode()).hexdigest(), 16) % 2 == 0 else "test"


def toks(q):
    return re.findall(r"[a-z]+", str(q).lower())


def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows.append(r)
    return rows


def train_nb(rows):
    # per modality: word|task counts, task priors
    wc = defaultdict(lambda: defaultdict(Counter))   # mod -> task -> Counter(word)
    tc = defaultdict(Counter)                          # mod -> Counter(task)
    vocab = defaultdict(set)
    for r in rows:
        if split(r["question_id"]) != "train":
            continue
        mod, task = r.get("modality", ""), r.get("task", "")
        tc[mod][task] += 1
        for w in toks(r.get("question", "")):
            wc[mod][task][w] += 1
            vocab[mod].add(w)
    return wc, tc, vocab


def predict_task(q, mod, wc, tc, vocab):
    tasks = tc.get(mod)
    if not tasks:
        return None
    V = max(1, len(vocab[mod]))
    words = toks(q)
    best, best_lp = None, -1e18
    total = sum(tasks.values())
    for task, n in tasks.items():
        lp = math.log(n / total)
        denom = sum(wc[mod][task].values()) + V
        for w in words:
            lp += math.log((wc[mod][task][w] + 1) / denom)
        if lp > best_lp:
            best_lp, best = lp, task
    return best


def best_viz(rows):
    agg = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))  # (mod,task)->viz->[s,n]
    for r in rows:
        if split(r["question_id"]) != "train":
            continue
        k = (r.get("modality", ""), r.get("task", ""))
        agg[k][r.get("viz_type", "")][0] += float(r.get("exact_match", 0) or 0)
        agg[k][r.get("viz_type", "")][1] += 1
    return {k: max(v.items(), key=lambda kv: kv[1][0] / kv[1][1])[0] for k, v in agg.items()}


def evaluate(path):
    rows = load(path)
    wc, tc, vocab = train_nb(rows)
    bv = best_viz(rows)
    # group test rows per question: qid -> {viz:em}, mod, task, question
    q = {}
    for r in rows:
        if split(r["question_id"]) != "test":
            continue
        rec = q.setdefault(r["question_id"], {"mod": r.get("modality", ""),
                                              "task": r.get("task", ""),
                                              "qtext": r.get("question", ""), "viz": {}})
        rec["viz"][r.get("viz_type", "")] = float(r.get("exact_match", 0) or 0)
    n = 0; single = 0.0; sel_oracle = 0.0; sel_pred = 0.0; orc = 0.0; task_hit = 0
    for rec in q.values():
        if not rec["viz"]:
            continue
        n += 1
        ems = list(rec["viz"].values())
        single += sum(ems) / len(ems)
        orc += 1.0 if any(e >= 0.5 for e in ems) else 0.0
        # oracle-task selector
        vo = bv.get((rec["mod"], rec["task"]))
        sel_oracle += rec["viz"].get(vo, sum(ems) / len(ems)) if vo else sum(ems) / len(ems)
        # predicted-task selector
        pt = predict_task(rec["qtext"], rec["mod"], wc, tc, vocab)
        task_hit += (pt == rec["task"])
        vp = bv.get((rec["mod"], pt))
        sel_pred += rec["viz"].get(vp, sum(ems) / len(ems)) if vp else sum(ems) / len(ems)
    f = 100.0 / n
    return dict(n=n, single=single * f, sel_oracle=sel_oracle * f, sel_pred=sel_pred * f,
                oracle=orc * f, task_acc=task_hit * f)


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    print(f"{'model':26} {'mean':>6} {'sel(oracle-task)':>16} {'sel(pred-task)':>14} "
          f"{'taskAcc':>8} {'oracle':>7}")
    print("-" * 84)
    for path in sys.argv[1:]:
        r = evaluate(path)
        print(f"{path.split('/')[-1]:26} {r['single']:6.1f} {r['sel_oracle']:16.1f} "
              f"{r['sel_pred']:14.1f} {r['task_acc']:8.1f} {r['oracle']:7.1f}")
    print("-" * 84)
    print("sel(pred-task) uses a stdlib Naive-Bayes task classifier on question text (deployable).")


if __name__ == "__main__":
    main()
