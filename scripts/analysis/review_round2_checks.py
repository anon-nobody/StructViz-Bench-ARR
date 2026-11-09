#!/usr/bin/env python3
"""Round-2 review checks: rendering completeness, complete-subset gaps, visual-only
gaps, split-half correction for post-hoc best/worst selection, Bonferroni counts.

Stdlib only, no inference. Mirrors scripts/verify_paper_numbers.py (`load`,
`per_format`, `by_question`) and its paired sign-flip permutation test with the
estimator p = (hits + 1) / (B + 1), two-sided on |mean difference|.

Implementation note: exact_match is 0/1, so each paired difference is -1, 0 or +1.
Zero differences never change the permuted statistic, and randomly sign-flipping k
nonzero +-1 differences yields the same distribution as a sum of k independent
random signs. Each permutation is therefore drawn as 2*popcount(getrandbits(k)) - k,
which is the same test as the harness loop, only faster. Every test uses its own
random.Random(0), B = 10,000.

    python scripts/analysis/review_round2_checks.py
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("STRUCTVIZ_ROOT", os.path.dirname(os.path.dirname(HERE)))
RES = os.path.join(ROOT, "results")
BENCH = os.path.join(ROOT, "benchmark", "realworld_test.jsonl")
OUT = os.path.join(HERE, "review_round2_results.json")

CORE = {
    "GPT-4o": "full_gpt4o_extracted.jsonl",
    "Gemini Flash": "full_gemini_extracted.jsonl",
    "Qwen2.5-VL-7B": "full_qwen_extracted.jsonl",
    "Claude Sonnet": "full_claude_extracted.jsonl",
}
MODS = ["tabular", "timeseries", "graph"]
B_PERM = 10000
SEED = 0


def load(fn):
    for cand in (fn, fn.replace("_extracted", "")):
        p = os.path.join(RES, cand)
        if os.path.exists(p):
            return [json.loads(l) for l in open(p) if l.strip()]
    raise FileNotFoundError(fn)


def by_question(recs, modality=None):
    g = defaultdict(dict)
    for r in recs:
        if modality is None or r.get("modality") == modality:
            g[r["question_id"]][r["viz_type"]] = float(r.get("exact_match", 0))
    return g


def fmt_means(q, qids, formats):
    out = {}
    for f in formats:
        v = [q[i][f] for i in qids if f in q[i]]
        out[f] = 100 * sum(v) / len(v) if v else float("nan")
    return out


def perm_test(q, qids, a, b):
    """Paired permutation test of mean(a - b) over qids. Returns (diff_pp, p, n)."""
    diffs = [int(q[i][a] - q[i][b]) for i in qids if a in q[i] and b in q[i]]
    n = len(diffs)
    s_obs = sum(diffs)
    k = sum(1 for d in diffs if d != 0)
    rng = random.Random(SEED)
    hits = 0
    for _ in range(B_PERM):
        tot = 2 * bin(rng.getrandbits(k)).count("1") - k if k else 0
        if abs(tot) >= abs(s_obs):
            hits += 1
    p = (hits + 1) / (B_PERM + 1)
    return 100 * s_obs / n if n else float("nan"), p, n


def fp(p):
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


def best_worst(means):
    b = max(means, key=means.get)
    w = min(means, key=means.get)
    return b, w


def half_of(qid):
    return int(hashlib.md5(qid.encode()).hexdigest(), 16) % 2


def main():
    bench = [json.loads(l) for l in open(BENCH) if l.strip()]
    meta = {b["question_id"]: b for b in bench}
    core = {k: load(v) for k, v in CORE.items()}
    res = {}

    # ---------------- A. rendering completeness ----------------
    tab_rows = {i: len(b["data"]) for i, b in meta.items() if b["modality"] == "tabular"}
    ts_pts = {i: len(b["data"]) for i, b in meta.items() if b["modality"] == "timeseries"}
    g_edges = {i: len(b["data"]["edges"]) for i, b in meta.items() if b["modality"] == "graph"}
    deg_q = {i for i, b in meta.items() if b["modality"] == "graph" and b["task"] == "degree_query"}
    A = {
        "tabular_total": len(tab_rows),
        "tabular_rows_le12": sum(1 for v in tab_rows.values() if v <= 12),
        "tabular_rows_13_14": sum(1 for v in tab_rows.values() if 13 <= v <= 14),
        "tabular_rows_15_20": sum(1 for v in tab_rows.values() if 15 <= v <= 20),
        "tabular_rows_gt20": sum(1 for v in tab_rows.values() if v > 20),
        "timeseries_total": len(ts_pts),
        "timeseries_points_le43": sum(1 for v in ts_pts.values() if v <= 43),
        "timeseries_points_gt43": sum(1 for v in ts_pts.values() if v > 43),
        "graph_total": len(g_edges),
        "graph_edges_le43": sum(1 for v in g_edges.values() if v <= 43),
        "graph_edges_gt43": sum(1 for v in g_edges.values() if v > 43),
        "graph_degree_query": len(deg_q),
    }
    res["A_rendering_completeness"] = A
    print("A. Rendering-completeness counts")
    print("-" * 78)
    for k, v in A.items():
        print(f"  {k:28s} {v:>6d}")

    # ---------------- B. tabular complete subset ----------------
    tab_fmts = ["bar_chart", "heatmap", "table_image", "scatter_plot", "text_only"]
    le12 = sorted(i for i, v in tab_rows.items() if v <= 12)
    gt12 = sorted(i for i, v in tab_rows.items() if v > 12)
    res["B_tabular_subsets"] = {}
    print("\nB. Tabular complete subset (rows<=12) and complement (rows>12)")
    print("-" * 78)
    for label, ids in (("rows<=12", le12), ("rows>12", gt12)):
        print(f"  subset {label}  n={len(ids)}")
        hdr = "  ".join(f"{f[:11]:>11s}" for f in tab_fmts)
        print(f"  {'model':14s} {hdr}  {'best':>12s} {'worst':>12s} {'gap':>6s} {'p':>7s}")
        res["B_tabular_subsets"][label] = {"n": len(ids), "models": {}}
        for name, recs in core.items():
            q = by_question(recs, "tabular")
            m = fmt_means(q, ids, tab_fmts)
            b, w = best_worst(m)
            d, p, n = perm_test(q, ids, b, w)
            row = "  ".join(f"{m[f]:>11.1f}" for f in tab_fmts)
            print(f"  {name:14s} {row}  {b:>12s} {w:>12s} {d:>6.1f} {fp(p):>7s}")
            entry = {"per_format_em": m, "best": b, "worst": w, "gap_pp": d, "p": p, "n": n}
            if label == "rows<=12":
                d2, p2, n2 = perm_test(q, ids, "table_image", "text_only")
                entry["table_image_minus_text_only"] = {"diff_pp": d2, "p": p2, "n": n2}
            res["B_tabular_subsets"][label]["models"][name] = entry
        print()
    print("  table_image - text_only on rows<=12")
    for name, e in res["B_tabular_subsets"]["rows<=12"]["models"].items():
        t = e["table_image_minus_text_only"]
        print(f"  {name:14s} diff={t['diff_pp']:>6.1f}pp  p={fp(t['p'])}  n={t['n']}")

    # ---------------- C. visual-only gaps ----------------
    fmts_by_mod = {
        "tabular": tab_fmts,
        "timeseries": ["line_plot", "gaf", "recurrence_plot", "heatmap", "text_only"],
        "graph": ["node_link", "adjacency_matrix", "circular_layout", "text_only"],
    }
    res["C_visual_only"] = {}
    print("\nC. Visual-only gaps (text_only excluded)")
    print("-" * 78)
    print(f"  {'model':14s} {'modality':10s} {'best':>16s} {'worst':>16s} {'gap':>6s} {'p':>7s} {'n':>5s}")
    for name, recs in core.items():
        res["C_visual_only"][name] = {}
        for md in MODS:
            q = by_question(recs, md)
            ids = sorted(q)
            vis = [f for f in fmts_by_mod[md] if f != "text_only"]
            m = fmt_means(q, ids, vis)
            b, w = best_worst(m)
            d, p, n = perm_test(q, ids, b, w)
            res["C_visual_only"][name][md] = {"per_format_em": m, "best": b, "worst": w,
                                              "gap_pp": d, "p": p, "n": n}
            print(f"  {name:14s} {md:10s} {b:>16s} {w:>16s} {d:>6.1f} {fp(p):>7s} {n:>5d}")

    print("\n  Graph (i): visual-only gap excluding degree_query")
    print(f"  {'model':14s} {'best':>16s} {'worst':>16s} {'gap':>6s} {'p':>7s} {'n':>5s}")
    res["C_graph_extra"] = {}
    gvis = ["node_link", "adjacency_matrix", "circular_layout"]
    for name, recs in core.items():
        q = by_question(recs, "graph")
        nondeg = sorted(i for i in q if i not in deg_q)
        deg = sorted(i for i in q if i in deg_q)
        m = fmt_means(q, nondeg, gvis)
        b, w = best_worst(m)
        d, p, n = perm_test(q, nondeg, b, w)
        e = {"i_visual_gap_excl_degree": {"per_format_em": m, "best": b, "worst": w,
                                          "gap_pp": d, "p": p, "n": n}}
        print(f"  {name:14s} {b:>16s} {w:>16s} {d:>6.1f} {fp(p):>7s} {n:>5d}")
        dd, pd, nd = perm_test(q, deg, "text_only", "adjacency_matrix")
        do, po, no = perm_test(q, nondeg, "text_only", "adjacency_matrix")
        e["ii_text_minus_adj"] = {"degree_query": {"diff_pp": dd, "p": pd, "n": nd},
                                  "other_tasks": {"diff_pp": do, "p": po, "n": no}}
        d3, p3, n3 = perm_test(q, nondeg, "text_only", b)
        e["iii_text_minus_best_visual_nondeg"] = {"best_visual": b, "diff_pp": d3, "p": p3, "n": n3}
        res["C_graph_extra"][name] = e

    print("\n  Graph (ii): text_only - adjacency_matrix, degree_query vs other tasks")
    print(f"  {'model':14s} {'deg diff':>9s} {'p':>7s} {'n':>4s}   {'other diff':>10s} {'p':>7s} {'n':>4s}")
    for name, e in res["C_graph_extra"].items():
        a, o = e["ii_text_minus_adj"]["degree_query"], e["ii_text_minus_adj"]["other_tasks"]
        print(f"  {name:14s} {a['diff_pp']:>9.1f} {fp(a['p']):>7s} {a['n']:>4d}   "
              f"{o['diff_pp']:>10.1f} {fp(o['p']):>7s} {o['n']:>4d}")

    print("\n  Graph (iii): text_only - best visual, non-degree_query questions")
    print(f"  {'model':14s} {'best visual':>16s} {'diff':>6s} {'p':>7s} {'n':>4s}")
    for name, e in res["C_graph_extra"].items():
        t = e["iii_text_minus_best_visual_nondeg"]
        print(f"  {name:14s} {t['best_visual']:>16s} {t['diff_pp']:>6.1f} {fp(t['p']):>7s} {t['n']:>4d}")

    # ---------------- D. split-half ----------------
    print("\nD. Split-half correction (md5 parity; select on one half, test on the other)")
    print("-" * 78)
    print(f"  {'model':14s} {'modality':10s} {'sel':>3s} {'best':>16s} {'worst':>16s} "
          f"{'held gap':>8s} {'p':>7s} {'n':>5s} {'=full?':>6s}")
    res["D_split_half"] = []
    for name, recs in core.items():
        for md in MODS:
            q = by_question(recs, md)
            ids = sorted(q)
            fm = fmts_by_mod[md]
            full_b, full_w = best_worst(fmt_means(q, ids, fm))
            halves = {h: [i for i in ids if half_of(i) == h] for h in (0, 1)}
            for sel, test in ((0, 1), (1, 0)):
                b, w = best_worst(fmt_means(q, halves[sel], fm))
                d, p, n = perm_test(q, halves[test], b, w)
                same = (b, w) == (full_b, full_w)
                res["D_split_half"].append({
                    "model": name, "modality": md, "select_half": sel, "test_half": test,
                    "n_select": len(halves[sel]), "best": b, "worst": w,
                    "full_best": full_b, "full_worst": full_w, "held_out_gap_pp": d,
                    "p": p, "n_test": n, "pair_equals_full": same,
                })
                print(f"  {name:14s} {md:10s} {sel:>3d} {b:>16s} {w:>16s} {d:>8.1f} "
                      f"{fp(p):>7s} {n:>5d} {'yes' if same else 'no':>6s}")

    # ---------------- E. Bonferroni ----------------
    print("\nE. Bonferroni (x12) on held-out contrasts")
    print("-" * 78)
    E = {}
    for sel in (0, 1):
        rows = [r for r in res["D_split_half"] if r["select_half"] == sel]
        E[f"select_half_{sel}"] = {
            "n_contrasts": len(rows),
            "p_x12_lt_0.05": sum(1 for r in rows if r["p"] * 12 < 0.05),
            "p_x12_lt_0.01": sum(1 for r in rows if r["p"] * 12 < 0.01),
        }
        e = E[f"select_half_{sel}"]
        print(f"  select on half {sel}, test on half {1 - sel}: "
              f"p*12<0.05: {e['p_x12_lt_0.05']}/{e['n_contrasts']}   "
              f"p*12<0.01: {e['p_x12_lt_0.01']}/{e['n_contrasts']}")
    E["p_floor"] = 1 / (B_PERM + 1)
    res["E_bonferroni"] = E
    res["settings"] = {"B": B_PERM, "seed": SEED, "estimator": "(hits+1)/(B+1)",
                       "split": "int(md5(question_id).hexdigest(),16) % 2"}

    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
