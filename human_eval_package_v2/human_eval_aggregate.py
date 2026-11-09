#!/usr/bin/env python3
"""StructViz-Bench human-validation aggregator (pure stdlib, no numpy/pandas).

Purpose: turn 3 annotators' Task-A correctness ratings into the exact numbers the
rebuttal / paper needs, plus (optionally) human answer-accuracy vs. ground truth.

--- INPUTS ---------------------------------------------------------------------
(1) RATINGS (Task A). A CSV in LONG format, one row per (item, evaluator):
        item_id,evaluator,rating
        graph_000100::difficulty=1-hop,ann1,Correct
        graph_000100::difficulty=1-hop,ann2,Ambiguous
        ...
    rating in {Correct, Ambiguous, Incorrect} (case-insensitive; C/A/I accepted).
    (WIDE format also works: item_id,ann1,ann2,ann3 with the three ratings.)

(2) OPTIONAL human-answer accuracy: the filled annotation_sheet.csv
    (columns include item_id, human_answer) + task_a_items.jsonl (ground truth).
    Human answers are matched to ground truth with the SAME tolerance the
    benchmark uses (numeric abs 0.05 / rel 0.01; case/space-insensitive).

--- USAGE ----------------------------------------------------------------------
    python human_eval_aggregate.py --ratings ratings.csv
    python human_eval_aggregate.py --ratings ratings.csv \
        --answers results/human_eval/annotation_sheet.csv \
        --truth   results/human_eval/task_a_items.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict

CATS = ["Correct", "Ambiguous", "Incorrect"]
_CANON = {"c": "Correct", "a": "Ambiguous", "i": "Incorrect",
          "correct": "Correct", "ambiguous": "Ambiguous", "incorrect": "Incorrect"}


def canon(r: str) -> str:
    r = (r or "").strip().lower()
    if r not in _CANON:
        raise ValueError(f"Unrecognized rating: {r!r} (use Correct/Ambiguous/Incorrect)")
    return _CANON[r]


def load_ratings(path: str) -> dict[str, list[str]]:
    """Return {item_id: [rating, ...]} accepting long or wide CSV."""
    per_item: dict[str, list[str]] = defaultdict(list)
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return per_item
    header = [h.strip().lower() for h in rows[0]]
    if "evaluator" in header and "rating" in header:  # long
        i_id, i_ev, i_rt = header.index("item_id"), header.index("evaluator"), header.index("rating")
        for row in rows[1:]:
            if len(row) <= max(i_id, i_ev, i_rt) or not row[i_id].strip():
                continue
            per_item[row[i_id].strip()].append(canon(row[i_rt]))
    else:  # wide: item_id, <ann cols...>
        i_id = header.index("item_id")
        rating_cols = [j for j, h in enumerate(header) if j != i_id]
        for row in rows[1:]:
            if len(row) <= i_id or not row[i_id].strip():
                continue
            for j in rating_cols:
                if j < len(row) and row[j].strip():
                    per_item[row[i_id].strip()].append(canon(row[j]))
    return per_item


def fleiss_kappa(per_item: dict[str, list[str]]) -> float | None:
    """Fleiss' kappa over items with the SAME number of raters n (n>=2)."""
    counts = defaultdict(int)
    for ratings in per_item.values():
        counts[len(ratings)] += 1
    if not counts:
        return None
    n = max(counts, key=counts.get)  # modal rater count
    items = [r for r in per_item.values() if len(r) == n]
    if n < 2 or len(items) < 2:
        return None
    N = len(items)
    p_j = {c: 0 for c in CATS}
    P_i_sum = 0.0
    for ratings in items:
        row = {c: 0 for c in CATS}
        for r in ratings:
            row[r] += 1
        for c in CATS:
            p_j[c] += row[c]
        P_i = (sum(v * v for v in row.values()) - n) / (n * (n - 1))
        P_i_sum += P_i
    for c in CATS:
        p_j[c] /= (N * n)
    P_bar = P_i_sum / N
    P_e = sum(v * v for v in p_j.values())
    if abs(1 - P_e) < 1e-12:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def _num(s: str):
    m = re.search(r"-?\d+(?:\.\d+)?", str(s).replace(",", ""))
    return float(m.group()) if m else None


def answer_match(pred: str, gold: str, abs_tol=0.05, rel_tol=0.01) -> bool:
    p, g = str(pred).strip().lower(), str(gold).strip().lower()
    if not p:
        return False
    if p == g:
        return True
    pn, gn = _num(p), _num(g)
    if pn is not None and gn is not None:
        return abs(pn - gn) <= max(abs_tol, rel_tol * abs(gn))
    return g in p  # lenient substring for text


def report(per_item, answers=None, truth=None):
    N = len(per_item)
    if N == 0:
        print("No ratings loaded."); return
    maj = {c: 0 for c in CATS}
    ties = 0
    for ratings in per_item.values():
        tally = {c: ratings.count(c) for c in CATS}
        top = max(tally.values())
        winners = [c for c in CATS if tally[c] == top]
        if len(winners) > 1:
            ties += 1
            # tie -> most conservative (worst) label
            winners.sort(key=lambda c: CATS.index(c))
            maj[winners[-1]] += 1
        else:
            maj[winners[0]] += 1
    pct = {c: 100 * maj[c] / N for c in CATS}
    kappa = fleiss_kappa(per_item)

    print("=" * 64)
    print(f"TASK A — answer-correctness audit  (items = {N}, ties = {ties})")
    print("=" * 64)
    for c in CATS:
        print(f"  {c:10s}: {maj[c]:4d}  ({pct[c]:5.1f}%)")
    ks = f"{kappa:.3f}" if kappa is not None else "n/a (uneven rater counts)"
    print(f"  Fleiss' kappa : {ks}")
    print()
    c_pct, a_pct, i_pct = pct["Correct"], pct["Ambiguous"], pct["Incorrect"]
    print("REBUTTAL SENTENCE (paste-ready):")
    print(f'  "On a stratified sample of {N} items independently rated by our annotators,')
    print(f"   {c_pct:.0f}% were judged correct by majority, {a_pct:.0f}% ambiguous,")
    print(f"   and {i_pct:.0f}% incorrect (Fleiss' kappa = {ks}).\"")
    print()

    if answers and truth:
        gold = {}
        with open(truth) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                gold[r.get("question_id") or r.get("item_id")] = str(r.get("answer", ""))
        n = ok = 0
        with open(answers, newline="") as f:
            for row in csv.DictReader(f):
                iid = (row.get("item_id") or "").strip()
                ha = (row.get("human_answer") or "").strip()
                if not iid or not ha or iid not in gold:
                    continue
                n += 1
                ok += answer_match(ha, gold[iid])
        if n:
            print("HUMAN ANSWER-ACCURACY vs ground truth (upper-bound / ceiling):")
            print(f"  answered {n} items, {100*ok/n:.1f}% match ground truth "
                  f"(numeric tol abs .05/rel .01; substring for text)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True, help="Task-A ratings CSV (long or wide)")
    ap.add_argument("--answers", help="filled annotation_sheet.csv (optional)")
    ap.add_argument("--truth", help="task_a_items.jsonl ground truth (optional)")
    a = ap.parse_args()
    report(load_ratings(a.ratings), a.answers, a.truth)


if __name__ == "__main__":
    main()
