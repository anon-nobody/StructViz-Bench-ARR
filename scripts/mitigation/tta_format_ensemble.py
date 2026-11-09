#!/usr/bin/env python3
"""Training-free FORMAT ENSEMBLE (TTAug) prototype for StructViz-Bench.

Key idea: the benchmark already renders each base question in several visualization
formats and has model predictions for each. A format ensemble aggregates the per-format
predictions for the SAME base question into one answer. This needs NO GPU and NO retraining
-- it runs offline on the existing results/full_<model>.jsonl files -- and directly yields
the paper's "sensitivity X pp -> Y pp" mitigation number.

For each model it reports:
  - single-format mean EM        (baseline: pick one format at random, in expectation)
  - worst / best single-format EM
  - ENSEMBLE EM (majority vote across that question's formats; ties -> highest-confidence
    format if a score is present, else the best-a-priori format order)
  - oracle EM (upper bound: correct if ANY format is correct)
  - Consistency Rate (mean fraction of format pairs that agree per question)

Grouping key = base question id with the "::difficulty=" suffix kept but the viz_type stripped.
Records must have: question_id, viz_type, exact_match (0/1), and prediction (for majority).

Usage:
  python tta_format_ensemble.py results/full_gpt4o.jsonl results/full_qwen.jsonl ...
"""
from __future__ import annotations
import json, sys, re
from collections import defaultdict, Counter


def base_id(qid: str, viz: str) -> str:
    """Strip the viz component so all formats of one question share a key."""
    # question_id looks like "graph_000100::difficulty=1-hop"; the viz is a separate field,
    # so the base key is just question_id. (If viz is embedded, strip it here.)
    return qid


def norm(p: str) -> str:
    p = str(p).strip().lower()
    m = re.search(r"-?\d+(?:\.\d+)?", p.replace(",", ""))
    return m.group() if m else p


def load(path):
    groups = defaultdict(list)  # base_id -> list of (viz, em, pred)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qid = r.get("question_id", "")
            viz = r.get("viz_type", "")
            em = float(r.get("exact_match", 0) or 0)
            pred = norm(r.get("prediction", ""))
            groups[base_id(qid, viz)].append((viz, em, pred))
    return groups


def consistency_rate(preds):
    """Mean pairwise agreement of predictions within a question."""
    n = len(preds)
    if n < 2:
        return 1.0
    agree = tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += 1
            agree += (preds[i] == preds[j])
    return agree / tot if tot else 1.0


def evaluate(groups):
    rows = [g for g in groups.values() if g]
    n = len(rows)
    if not n:
        return None
    per_fmt_sum = 0.0; per_fmt_cnt = 0
    ens = orc = 0.0
    cr_sum = 0.0
    best_by_fmt = defaultdict(lambda: [0.0, 0]);
    for g in rows:
        ems = [em for _v, em, _p in g]
        preds = [p for _v, _em, p in g]
        # per-question mean so the single-format baseline shares the ensemble's denominator
        per_fmt_sum += sum(ems) / len(ems); per_fmt_cnt += 1
        orc += 1.0 if any(e >= 0.5 for e in ems) else 0.0
        cr_sum += consistency_rate(preds)
        # majority vote over predictions; tie -> the prediction whose formats have the
        # highest mean single-format EM globally (approximated here by first-most-common).
        c = Counter(preds)
        top = c.most_common(1)[0][0]
        # ensemble correct iff the majority-voted prediction matches ground truth on any
        # format that produced it AND that format was correct -> use max em among formats
        # that emitted the winning prediction:
        win_em = max((em for _v, em, p in g if p == top), default=0.0)
        ens += 1.0 if win_em >= 0.5 else 0.0
        for v, em, _p in g:
            best_by_fmt[v][0] += em; best_by_fmt[v][1] += 1
    fmt_em = {v: 100 * s / c for v, (s, c) in best_by_fmt.items() if c}
    return {
        "n_questions": n,
        "mean_single_format_EM": 100 * per_fmt_sum / per_fmt_cnt,
        "worst_format_EM": min(fmt_em.values()),
        "best_format_EM": max(fmt_em.values()),
        "ensemble_EM": 100 * ens / n,
        "oracle_EM": 100 * orc / n,
        "consistency_rate": 100 * cr_sum / n,
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    print(f"{'model file':28} {'meanFmt':>8} {'worst':>7} {'best':>7} "
          f"{'ENSEMBLE':>9} {'oracle':>7} {'CR%':>6}")
    print("-" * 80)
    for path in sys.argv[1:]:
        r = evaluate(load(path))
        if not r:
            print(f"{path}: no data"); continue
        name = path.split("/")[-1]
        print(f"{name:28} {r['mean_single_format_EM']:8.2f} {r['worst_format_EM']:7.2f} "
              f"{r['best_format_EM']:7.2f} {r['ensemble_EM']:9.2f} {r['oracle_EM']:7.2f} "
              f"{r['consistency_rate']:6.1f}")
    print("-" * 80)
    print("Read: ENSEMBLE vs mean-single-format = training-free mitigation gain.")
    print("      oracle = headroom if format were chosen perfectly. CR = prediction agreement.")
    print("NOTE: verify field names (question_id/viz_type/exact_match/prediction) match your JSONL;")
    print("      the majority-vote tie rule is a first cut -- see TTAUG_DESIGN.md for the full method.")


if __name__ == "__main__":
    main()
