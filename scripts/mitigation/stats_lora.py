#!/usr/bin/env python3
"""Paired bootstrap statistics for the consistency-LoRA before/after comparison.

Consumes the per-question dumps from eval_lora.py (--dump) for BASE and AFTER, which are paired
by question_id. Reports, per modality, with 95% bootstrap CIs (resampling questions):
  - best-worst format gap (Base, After) and Delta with a one-sided bootstrap p-value
  - cross-format std of per-format EM (a smooth sensitivity measure) Base/After
  - Consistency Rate (Base, After) and Delta with p-value
  - mean per-format accuracy (Base, After)
Pure stdlib (no numpy). Deterministic (seeded).

Usage:
  python stats_lora.py --base base_records.jsonl --after after_records.jsonl [--B 2000]
"""
from __future__ import annotations
import argparse, json, random
from collections import defaultdict


def load(path):
    d = {}
    for line in open(path):
        line = line.strip()
        if line:
            r = json.loads(line)
            d[r["qid"]] = r
    return d


def cr_of(preds):
    vals = list(preds.values())
    n = len(vals)
    if n < 2:
        return None
    agree = tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += 1; agree += (vals[i] == vals[j])
    return agree / tot


def per_format_em(recs):
    """recs: list of {'correct': {viz:0/1}} -> {viz: EM%}"""
    hit = defaultdict(lambda: [0, 0])
    for r in recs:
        for v, c in r["correct"].items():
            hit[v][0] += c; hit[v][1] += 1
    return {v: 100 * a / n for v, (a, n) in hit.items() if n}


def gap(em):
    return (max(em.values()) - min(em.values())) if em else 0.0


def std(em):
    xs = list(em.values())
    if not xs:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def mean_cr(recs):
    vs = [cr_of(r["preds"]) for r in recs]
    vs = [100 * v for v in vs if v is not None]
    return sum(vs) / len(vs) if vs else 0.0


def ci(vals):
    s = sorted(vals)
    lo = s[int(0.025 * len(s))]
    hi = s[int(0.975 * len(s)) - 1]
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    base, after = load(a.base), load(a.after)
    qids = [q for q in base if q in after]

    by_mod = defaultdict(list)
    for q in qids:
        by_mod[base[q]["modality"]].append(q)

    print(f"paired questions: {len(qids)}  (B={a.B} bootstrap)\n")
    for mod in ["tabular", "timeseries", "graph"]:
        qs = by_mod.get(mod, [])
        if len(qs) < 5:
            continue
        b_recs = [base[q] for q in qs]; a_recs = [after[q] for q in qs]
        b_gap, a_gap = gap(per_format_em(b_recs)), gap(per_format_em(a_recs))
        b_std, a_std = std(per_format_em(b_recs)), std(per_format_em(a_recs))
        b_cr, a_cr = mean_cr(b_recs), mean_cr(a_recs)
        # bootstrap
        dg, dc, bg_s, ag_s, bc_s, ac_s, bs_s, as_s = [], [], [], [], [], [], [], []
        for _ in range(a.B):
            samp = [qs[rng.randrange(len(qs))] for _ in range(len(qs))]
            br = [base[q] for q in samp]; ar = [after[q] for q in samp]
            beg, aeg = per_format_em(br), per_format_em(ar)
            g0, g1 = gap(beg), gap(aeg); c0, c1 = mean_cr(br), mean_cr(ar)
            bg_s.append(g0); ag_s.append(g1); dg.append(g1 - g0)
            bc_s.append(c0); ac_s.append(c1); dc.append(c1 - c0)
            bs_s.append(std(beg)); as_s.append(std(aeg))
        p_gap = sum(1 for x in dg if x >= 0) / a.B          # want gap to DROP (Delta<0)
        p_cr = sum(1 for x in dc if x <= 0) / a.B           # want CR to RISE (Delta>0)
        print(f"[{mod}]  n={len(qs)}")
        print(f"  gap : {b_gap:5.1f} (95%CI {ci(bg_s)[0]:.1f},{ci(bg_s)[1]:.1f})  ->  "
              f"{a_gap:5.1f} ({ci(ag_s)[0]:.1f},{ci(ag_s)[1]:.1f})  | "
              f"Delta 95%CI [{ci(dg)[0]:+.1f},{ci(dg)[1]:+.1f}]  p(no drop)={p_gap:.3f}")
        print(f"  std : {b_std:5.1f} (95%CI {ci(bs_s)[0]:.1f},{ci(bs_s)[1]:.1f})  ->  "
              f"{a_std:5.1f} ({ci(as_s)[0]:.1f},{ci(as_s)[1]:.1f})")
        print(f"  CR  : {b_cr:5.1f} (95%CI {ci(bc_s)[0]:.1f},{ci(bc_s)[1]:.1f})  ->  "
              f"{a_cr:5.1f} ({ci(ac_s)[0]:.1f},{ci(ac_s)[1]:.1f})  | "
              f"Delta 95%CI [{ci(dc)[0]:+.1f},{ci(dc)[1]:+.1f}]  p(no rise)={p_cr:.3f}")
        print()


if __name__ == "__main__":
    main()
