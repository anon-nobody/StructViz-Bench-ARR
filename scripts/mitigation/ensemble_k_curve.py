#!/usr/bin/env python3
"""Cost-accuracy curve for the format ensemble (GPU-free, offline).

For each model, ensemble EM as a function of K = number of formats aggregated per question
(majority vote), averaged over R random K-subsets. Shows diminishing returns: how many
renderings you must pay for. K=1 == mean single-format baseline.

Usage: python ensemble_k_curve.py results/full_gpt4o.jsonl [more...]
"""
from __future__ import annotations
import json, sys, re, random
from collections import defaultdict, Counter

R = 12          # random subsets per K
random.seed(42)


def norm(p):
    p = str(p).strip().lower()
    m = re.search(r"-?\d+(?:\.\d+)?", p.replace(",", ""))
    return m.group() if m else p


def load(path):
    q = defaultdict(list)  # qid -> list of (em, pred)
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        q[r["question_id"]].append((float(r.get("exact_match", 0) or 0),
                                    norm(r.get("prediction", ""))))
    return [v for v in q.values() if v]


def ens_em(question, k):
    # majority vote over a random k-subset (or all if fewer)
    if len(question) <= k:
        sub = question
    else:
        sub = random.sample(question, k)
    preds = [p for _e, p in sub]
    top = Counter(preds).most_common(1)[0][0]
    win = max((e for e, p in sub if p == top), default=0.0)
    return 1.0 if win >= 0.5 else 0.0


def curve(rows, kmax=5):
    out = {}
    for k in range(1, kmax + 1):
        acc = 0.0
        for _ in range(R):
            acc += sum(ens_em(qu, k) for qu in rows) / len(rows)
        out[k] = 100 * acc / R
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    print(f"{'model':26} " + " ".join(f'K={k}' for k in range(1, 6)))
    print("-" * 60)
    for path in sys.argv[1:]:
        c = curve(load(path))
        print(f"{path.split('/')[-1]:26} " + " ".join(f'{c[k]:5.1f}' for k in range(1, 6)))
    print("-" * 60)
    print("K=1 is the single-format baseline; gains typically saturate by K=3-4.")


if __name__ == "__main__":
    main()
