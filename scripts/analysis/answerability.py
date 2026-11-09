#!/usr/bin/env python3
"""Answerability audit: is each (question, format) answer recoverable from the rendered image?

Stdlib only. Classifies every (question, format) pair as
    V = visible, D = derivable, A = absent/truncated, C = constant/unanswerable-by-design
using fixed renderer facts (see answerability_matrix.md), then recomputes per-format exact
match and best-worst gaps on answerable pairs for the four core models.

    python scripts/analysis/answerability.py

Writes scripts/analysis/answerability_matrix.md and scripts/analysis/answerability_results.json.
Permutation test mirrors scripts/verify_paper_numbers.py: paired sign-flip, B=10,000,
p = (hits+1)/(B+1), two-sided on |mean diff|; here each test uses its own random.Random(0).
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

ROOT = os.environ.get(
    "STRUCTVIZ_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
RES = os.path.join(ROOT, "results")
BENCH = os.path.join(ROOT, "benchmark", "realworld_test.jsonl")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

CORE = {
    "GPT-4o": "full_gpt4o_extracted.jsonl",
    "Gemini Flash": "full_gemini_extracted.jsonl",
    "Qwen2.5-VL-7B": "full_qwen_extracted.jsonl",
    "Claude Sonnet": "full_claude_extracted.jsonl",
}
MODS = ["tabular", "timeseries", "graph"]
FORMATS = {
    "tabular": ["table_image", "text_only", "heatmap", "bar_chart", "scatter_plot"],
    "timeseries": ["line_plot", "heatmap", "gaf", "recurrence_plot", "text_only"],
    "graph": ["node_link", "circular_layout", "adjacency_matrix", "text_only"],
}
CATS = ["V", "D", "A", "C"]
B_PERM = 10000
N_CELLS = 12

# Renderer limits (renderer facts supplied by the lead; not re-derived here).
TAB_TABLE_ROWS, TAB_TEXT_ROWS, TAB_HEAT_ROWS = 12, 14, 20
TS_TEXT_VISIBLE = 43
G_LABEL_NODES_LT = 30          # node_link / circular labels iff nodes < 30
G_ADJ_TICKS_LE = 24            # adjacency tick labels for every node iff nodes <= 24
G_TEXT_VISIBLE_EDGES = 43

# Branch counters, printed so every regex/rule branch is auditable.
BRANCH: Counter = Counter()


# --------------------------------------------------------------------------- helpers
def load(fn):
    p = os.path.join(RES, fn)
    return [json.loads(l) for l in open(p) if l.strip()]


def em_of(vals):
    return 100 * sum(vals) / len(vals) if vals else float("nan")


def fmt_p(p):
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


def f1(x):
    return "  nan" if x != x else f"{x:5.1f}"


def perm_p(paired):
    """Paired sign-flip permutation, B=10,000, estimator (hits+1)/(B+1), seed 0.

    Zero differences contribute nothing to any permuted sum, so only nonzero pairs are
    resampled. For 0/1 exact match every nonzero |diff| is 1, so a permuted sum equals
    2*popcount(random k bits) - k; the test statistic |sum| >= |obs sum| is identical to
    |mean| >= |obs mean| for fixed n.
    """
    diffs = [x - y for x, y in paired]
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan")
    obs = sum(diffs)
    nz = [d for d in diffs if d != 0]
    k = len(nz)
    rng = random.Random(0)
    hits = 0
    if all(abs(d) == 1 for d in nz):
        a_obs = abs(obs)
        for _ in range(B_PERM):
            s = 2 * rng.getrandbits(k).bit_count() - k if k else 0
            if abs(s) >= a_obs:
                hits += 1
    else:  # general fallback
        a_obs = abs(obs) - 1e-12
        for _ in range(B_PERM):
            s = sum(d if rng.random() < 0.5 else -d for d in nz)
            if abs(s) >= a_obs:
                hits += 1
    return 100 * obs / n, (hits + 1) / (B_PERM + 1)


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def numeric_cols(rows):
    """Columns pandas select_dtypes('number') would keep: every value int/float."""
    cols = list(rows[0].keys())
    return [c for c in cols if all(is_num(r.get(c)) for r in rows)]


def quoted(q):
    return re.findall(r"'([^']*)'", q)


def nx_edge_order(data):
    """Reproduce networkx Graph.edges() order for a graph built by node_link_graph.

    Nodes are added in the stored node order, edges in the stored edge order; each node's
    adjacency is ordered by first insertion; edges() walks nodes in order and yields
    (u, v) for each neighbour v not yet visited (self-loops once).
    """
    adj = {n["id"]: {} for n in data["nodes"]}
    key = "edges" if "edges" in data else "links"
    for e in data[key]:
        u, v = e["source"], e["target"]
        adj.setdefault(u, {})
        adj.setdefault(v, {})
        adj[u].setdefault(v, None)
        adj[v].setdefault(u, None)
    seen, out = set(), []
    for u, nbrs in adj.items():
        for v in nbrs:
            if v not in seen:
                out.append((u, v))
        seen.add(u)
    return out


# --------------------------------------------------------------------------- rules
def classify_tabular(r):
    task, q, rows = r["task"], r["question"], r["data"]
    nrows = len(rows)
    num = set(numeric_cols(rows))
    qc = quoted(q)
    fit_t, fit_x, fit_h = nrows <= TAB_TABLE_ROWS, nrows <= TAB_TEXT_ROWS, nrows <= TAB_HEAT_ROWS
    D_or_A = lambda ok: "D" if ok else "A"
    # "average" outside quoted column names (a column is named 'average_daily_rate_usd')
    has_avg = "average" in re.sub(r"'[^']*'", "", q).lower()
    out = {}

    if task == "value_extraction":
        m = re.search(r"value of '([^']*)' in row (\d+)", q)
        col, ridx = m.group(1), int(m.group(2))
        BRANCH[f"tab.value_extraction row<12={ridx < 12}"] += 1
        BRANCH[f"tab.value_extraction col_numeric={col in num}"] += 1
        out = {"table_image": "V", "text_only": "V", "heatmap": "V" if col in num else "A",
               "bar_chart": "A", "scatter_plot": "A"}
    elif task == "comparison" or (task not in ("aggregation", "counterfactual") and has_avg):
        a, b = qc[0], qc[1]
        both = a in num and b in num
        BRANCH[f"tab.comparison-rule[{task}] both_numeric={both}"] += 1
        BRANCH[f"tab.comparison-rule[{task}] template={'counterfactual%' if 'increased' in q else 'which-higher'}"] += 1
        out = {"bar_chart": "V", "heatmap": D_or_A(fit_h and both), "table_image": D_or_A(fit_t),
               "text_only": D_or_A(fit_x), "scatter_plot": "A"}
    elif task == "aggregation":
        col = qc[0]
        BRANCH[f"tab.aggregation average={has_avg}"] += 1
        BRANCH[f"tab.aggregation col_numeric={col in num}"] += 1
        out = {"bar_chart": "V" if has_avg else "A", "table_image": D_or_A(fit_t),
               "text_only": D_or_A(fit_x), "heatmap": D_or_A(fit_h and col in num),
               "scatter_plot": "A"}
    elif task in ("filtering", "ranking", "outlier_detection", "trend_analysis"):
        if task in ("ranking", "outlier_detection"):
            label_col, val_col = qc[0], qc[1]
            label_cat = label_col not in num
            BRANCH[f"tab.{task} label_col_categorical={label_cat}"] += 1
            BRANCH[f"tab.{task} value_col_numeric={val_col in num}"] += 1
            heat_ok = fit_h and (not label_cat) and val_col in num
        else:
            col = qc[0]
            BRANCH[f"tab.{task} col_numeric={col in num}"] += 1
            heat_ok = fit_h and col in num
        out = {"table_image": D_or_A(fit_t), "text_only": D_or_A(fit_x),
               "heatmap": D_or_A(heat_ok), "bar_chart": "A", "scatter_plot": "A"}
    elif task == "correlation":
        a, b = qc[0], qc[1]
        ordered = numeric_cols(rows)
        first2 = set(ordered[:2]) if len(ordered) >= 2 else set()
        vis = {a, b} == first2
        BRANCH[f"tab.correlation pair==first_two_numeric={vis}"] += 1
        out = {"scatter_plot": "V" if vis else "A", "table_image": D_or_A(fit_t),
               "text_only": D_or_A(fit_x), "heatmap": D_or_A(fit_h and a in num and b in num),
               "bar_chart": "A"}
    elif task == "counterfactual":
        cols = qc[:2] if len(qc) >= 2 else qc
        cn = all(c in num for c in cols)
        BRANCH[f"tab.counterfactual average={has_avg}"] += 1
        BRANCH[f"tab.counterfactual cols_numeric={cn}"] += 1
        out = {"bar_chart": "D" if has_avg else "A", "table_image": D_or_A(fit_t),
               "text_only": D_or_A(fit_x), "heatmap": D_or_A(fit_h and cn), "scatter_plot": "A"}
    else:
        raise ValueError(task)
    return out


TS_CONST = {"pattern_label_lookup", "counterfactual_scale_peak", "counterfactual_sign_flip",
            "pattern_classification"}
TS_LINE_V = {"value_lookup", "range_query", "mean_shift_magnitude", "forecasting",
             "threshold_count", "median_mean_relation", "normalized_amplitude",
             "local_peak_median_relation", "oscillation_count"}
TS_LINE_D = {"peak_identification", "seasonality_detection", "change_point", "volatility",
             "mean_half_comparison", "endpoint_comparison", "change_volatility",
             "anomaly_detection", "split_trend_consistency", "peak_trough_order",
             "half_change_intensity", "counterfactual_half_shift"}
TS_HEAT_D = {"peak_identification", "seasonality_detection", "change_point", "volatility",
             "mean_half_comparison", "endpoint_comparison", "anomaly_detection",
             "split_trend_consistency", "peak_trough_order", "oscillation_count",
             "half_change_intensity", "change_volatility"}
TS_HEAT_A = {"value_lookup", "range_query", "mean_shift_magnitude", "forecasting",
             "threshold_count", "median_mean_relation", "normalized_amplitude",
             "local_peak_median_relation", "counterfactual_half_shift"}
TS_GAF_RP_D = {"seasonality_detection", "change_point"}


def classify_timeseries(r):
    task, q, n = r["task"], r["question"], len(r["data"])
    if task in TS_CONST:
        BRANCH[f"ts.C {task}"] += 1
        return {f: "C" for f in FORMATS["timeseries"]}
    assert task in TS_LINE_V | TS_LINE_D, task
    assert task in TS_HEAT_D | TS_HEAT_A, task
    line = "V" if task in TS_LINE_V else "D"
    out = {"line_plot": line, "heatmap": "D" if task in TS_HEAT_D else "A"}
    out["gaf"] = out["recurrence_plot"] = "D" if task in TS_GAF_RP_D else "A"
    if n <= TS_TEXT_VISIBLE:
        BRANCH["ts.text_only len<=43 (same as line_plot)"] += 1
        out["text_only"] = line
    elif task == "value_lookup":
        first = re.search(r"first", q) is not None
        BRANCH[f"ts.text_only len>43 value_lookup first={first}"] += 1
        out["text_only"] = "V" if first else "A"
    else:
        BRANCH["ts.text_only len>43 other task -> A"] += 1
        out["text_only"] = "A"
    return out


G_NODE_TASKS = {"degree_query", "connectivity", "shortest_path", "centrality", "counterfactual"}
G_GLOBAL_TASKS = {"edge_count", "clustering", "bipartite_check", "cycle_detection", "community",
                  "diameter"}
RE_CF_ADD = re.compile(r"If an edge were added between node \S+ and node \S+, would they be "
                       r"directly connected\?")


def classify_graph(r):
    task, q, data = r["task"], r["question"], r["data"]
    nn = len(data["nodes"])
    key = "edges" if "edges" in data else "links"
    ne = len(data[key])
    if task == "connectivity":
        BRANCH["g.C connectivity"] += 1
        return {f: "C" for f in FORMATS["graph"]}
    if task == "counterfactual" and RE_CF_ADD.fullmatch(q.strip()):
        BRANCH["g.C counterfactual add-edge-directly-connected"] += 1
        return {f: "C" for f in FORMATS["graph"]}
    node_ref = task in G_NODE_TASKS
    assert node_ref or task in G_GLOBAL_TASKS, task
    out = {}
    for f in ("node_link", "circular_layout"):
        out[f] = ("D" if nn < G_LABEL_NODES_LT else "A") if node_ref else "D"
    out["adjacency_matrix"] = ("D" if nn <= G_ADJ_TICKS_LE else "A") if node_ref else "D"
    if ne <= G_TEXT_VISIBLE_EDGES:
        BRANCH["g.text_only edges<=43"] += 1
        out["text_only"] = "V" if task == "degree_query" else "D"
    elif task == "degree_query":
        visible = {str(x) for e in nx_edge_order(data)[:G_TEXT_VISIBLE_EDGES] for x in e}
        qnodes = re.findall(r"node (\d+)", q)
        ok = bool(qnodes) and all(x in visible for x in qnodes)
        tmpl = ("no-node-referenced" if not qnodes else
                f"{len(qnodes)}-node(s)")
        BRANCH[f"g.text_only edges>43 degree_query [{tmpl}] visible={ok}"] += 1
        out["text_only"] = "V" if ok else "A"
    else:
        BRANCH["g.text_only edges>43 other task -> A"] += 1
        out["text_only"] = "A"
    return out


CLASSIFY = {"tabular": classify_tabular, "timeseries": classify_timeseries,
            "graph": classify_graph}


# --------------------------------------------------------------------------- rule table (md)
RULE_TABLE = """\
# Answerability matrix (StructViz-Bench, realworld_test)

Categories: **V** visible, **D** derivable (all information present, must be computed/counted),
**A** absent or truncated away, **C** constant / unanswerable-by-design.
Generated by `scripts/analysis/answerability.py` from renderer facts only (no inference).
"Numeric column" = every value is int/float (pandas `select_dtypes('number')`; columns that mix
strings such as `[EMPTY]` or `-` with floats are object dtype and are dropped by heatmap/bar/scatter).
"rows fit" = table_image rows<=12, text_only rows<=14, heatmap rows<=20.

## Tabular

| task | table_image | text_only | heatmap | bar_chart | scatter_plot |
|---|---|---|---|---|---|
| value_extraction (row r, all r<12) | V | V | V if column numeric else A | A | A |
| comparison (and any other question containing "average", excl. aggregation/counterfactual) | D if rows<=12 else A | D if rows<=14 else A | D if rows<=20 and both columns numeric else A | V | A |
| aggregation | D if fit | D if fit | D if fit and column numeric | V if "average" in question else A | A |
| filtering, trend_analysis | D if fit | D if fit | D if rows<=20 and column numeric | A | A |
| ranking, outlier_detection | D if fit | D if fit | A when answer is a categorical-column label (D only if label column numeric, rows<=20, value column numeric) | A | A |
| correlation | D if fit | D if fit | D if rows<=20 and both numeric | A | V if {a,b} == first two numeric columns else A |
| counterfactual | D if fit | D if fit | D if rows<=20 and columns numeric | D if "average" in question else A | A |

## Time series

| task | line_plot | heatmap (colour strip) | gaf | recurrence_plot | text_only |
|---|---|---|---|---|---|
| pattern_label_lookup, counterfactual_scale_peak, counterfactual_sign_flip, pattern_classification | C | C | C | C | C |
| value_lookup, range_query, mean_shift_magnitude, forecasting, threshold_count, median_mean_relation, normalized_amplitude, local_peak_median_relation | V | A | A | A | len<=43: V; else value_lookup "first" V, others A |
| oscillation_count | V | D | A | A | len<=43: V; else A |
| seasonality_detection, change_point | D | D | D | D | len<=43: D; else A |
| peak_identification, volatility, mean_half_comparison, endpoint_comparison, change_volatility, anomaly_detection, split_trend_consistency, peak_trough_order, half_change_intensity | D | D | A | A | len<=43: D; else A |
| counterfactual_half_shift | D | A | A | A | len<=43: D; else A |

## Graph

Node-referencing tasks: degree_query, connectivity, shortest_path, centrality, counterfactual.
Global tasks: edge_count, clustering, bipartite_check, cycle_detection, community, diameter.

| task group | node_link | circular_layout | adjacency_matrix | text_only |
|---|---|---|---|---|
| connectivity (answer always "yes"); counterfactual "if an edge were added between u and v, would they be directly connected" | C | C | C | C |
| node-referencing (other) | D if nodes<30 else A | D if nodes<30 else A | D if nodes<=24 else A | edges<=43: D (degree_query V); else degree_query V iff every queried node appears in the first 43 edges (networkx edge order), else A; other tasks A |
| global | D | D | D | edges<=43: D; else A |
"""


# --------------------------------------------------------------------------- main
def main() -> int:
    bench = [json.loads(l) for l in open(BENCH) if l.strip()]
    meta = {r["question_id"]: r for r in bench}
    cat = {}  # qid -> {fmt: cat}
    for r in bench:
        c = CLASSIFY[r["modality"]](r)
        assert set(c) == set(FORMATS[r["modality"]]), (r["question_id"], c)
        cat[r["question_id"]] = c

    print("StructViz-Bench answerability audit")
    print(f"root: {ROOT}")
    print("\n[rule branches]  (question counts per regex/rule branch)")
    for k in sorted(BRANCH):
        print(f"  {BRANCH[k]:5d}  {k}")
    # Informational: answer distribution of the other graph counterfactual templates.
    cf = defaultdict(Counter)
    for r in bench:
        if r["modality"] == "graph" and r["task"] == "counterfactual":
            t = re.sub(r"node \d+", "node N", r["question"])
            cf[t][str(r["answer"]).lower()] += 1
    print("  INFO graph counterfactual answer distribution by template:")
    for t, c in cf.items():
        print(f"    {dict(c)}  {t}")
    print("  INFO graph text_only edge order reproduced from networkx Graph.edges() semantics "
          "(node order, then adjacency insertion order).")

    # ---- (1) matrix counts
    counts = defaultdict(Counter)  # (mod, task, fmt) -> Counter(cat)
    for qid, c in cat.items():
        r = meta[qid]
        for f, k in c.items():
            counts[(r["modality"], r["task"], f)][k] += 1
    print("\n(1) task x format: question counts per category (V/D/A/C)")
    md_counts = ["\n## Counts per (modality, task, format)  [V/D/A/C]\n"]
    for md in MODS:
        fmts = FORMATS[md]
        tasks = sorted({t for (m, t, _) in counts if m == md})
        hdr = f"  {md:10s} {'task':28s}" + "".join(f"{f[:16]:>18s}" for f in fmts)
        print(hdr)
        md_counts.append(f"\n### {md}\n\n| task | n | " + " | ".join(fmts) + " |\n|---|---|"
                         + "---|" * len(fmts))
        for t in tasks:
            n = sum(counts[(md, t, fmts[0])].values())
            cells = []
            for f in fmts:
                c = counts[(md, t, f)]
                cells.append("/".join(str(c[k]) for k in CATS))
            print(f"  {'':10s} {t[:22]:22s} n={n:<4d}" + "".join(f"{x:>18s}" for x in cells))
            md_counts.append(f"| {t} | {n} | " + " | ".join(cells) + " |")
    with open(os.path.join(OUT_DIR, "answerability_matrix.md"), "w") as fh:
        fh.write(RULE_TABLE + "\n".join(md_counts) + "\n")

    results = {"branches": dict(BRANCH), "matrix_counts": {
        f"{m}|{t}|{f}": dict(c) for (m, t, f), c in counts.items()}}

    # ---- (7) category fractions
    print("\n(7) fraction of (question, format) pairs per category, by modality")
    results["category_fractions"] = {}
    for md in MODS:
        c = Counter()
        for qid, cc in cat.items():
            if meta[qid]["modality"] == md:
                c.update(cc.values())
        tot = sum(c.values())
        fr = {k: 100 * c[k] / tot for k in CATS}
        results["category_fractions"][md] = {"n_pairs": tot, **fr}
        print(f"  {md:10s} pairs={tot:5d}  " + "  ".join(f"{k}={fr[k]:5.1f}%" for k in CATS))

    core = {k: load(v) for k, v in CORE.items()}
    # em[model][qid][fmt]
    EM = {}
    for name, recs in core.items():
        d = defaultdict(dict)
        for r in recs:
            d[r["question_id"]][r["viz_type"]] = float(r.get("exact_match", 0))
        EM[name] = d
        missing = sum(1 for qid in cat if set(d.get(qid, {})) != set(cat[qid]))
        if missing:
            print(f"  WARN {name}: {missing} questions with format set mismatch")

    # ---- (2) EM by category
    print("\n(2) per-format EM on V∪D, A, and C pairs  [EM (n)]")
    results["em_by_category"] = {}
    for name in CORE:
        for md in MODS:
            fmts = FORMATS[md]
            line = f"  {name:14s} {md:10s}"
            rec = {}
            for f in fmts:
                g = {"VD": [], "A": [], "C": []}
                for qid, cc in cat.items():
                    if meta[qid]["modality"] != md:
                        continue
                    k = cc[f]
                    g["VD" if k in "VD" else k].append(EM[name][qid][f])
                rec[f] = {k: {"em": em_of(v), "n": len(v)} for k, v in g.items()}
            results["em_by_category"][f"{name}|{md}"] = rec
            print(line)
            for f in fmts:
                s = "   ".join(f"{k:>2s} {f1(rec[f][k]['em'])} ({rec[f][k]['n']:4d})"
                               for k in ("VD", "A", "C"))
                print(f"      {f:18s} {s}")

    # ---- (3) fully answerable subset
    print("\n(3) fully answerable subset (every format V or D): per-format EM, gap, paired p")
    results["fully_answerable"] = {}
    full_q = {md: [q for q, cc in cat.items() if meta[q]["modality"] == md
                   and all(k in "VD" for k in cc.values())] for md in MODS}
    nsig3 = 0
    for md in MODS:
        fmts = FORMATS[md]
        print(f"  {md}: n={len(full_q[md])}   formats: " + ", ".join(fmts))
        for name in CORE:
            qs = full_q[md]
            pf = {f: em_of([EM[name][q][f] for q in qs]) for f in fmts}
            if not qs:
                print(f"    {name:14s} (empty)")
                continue
            b, w = max(pf, key=pf.get), min(pf, key=pf.get)
            obs, p = perm_p([(EM[name][q][b], EM[name][q][w]) for q in qs])
            sig = p * N_CELLS < 0.01
            nsig3 += sig
            results["fully_answerable"][f"{name}|{md}"] = {
                "n": len(qs), "per_format": pf, "best": b, "worst": w, "gap": pf[b] - pf[w],
                "p": p, "p_bonf12": min(1.0, p * N_CELLS), "sig_bonf12_p<0.01": sig}
            print(f"    {name:14s} " + " ".join(f"{f1(pf[f])}" for f in fmts)
                  + f"  best={b} worst={w} gap={pf[b]-pf[w]:5.1f} p={fmt_p(p)}"
                  f" bonf12={fmt_p(min(1.0, p*N_CELLS))} {'SIG' if sig else 'ns'}")
    print(f"  cells significant after Bonferroni x12 (p_bonf<0.01): {nsig3}/12")
    results["fully_answerable_nsig"] = nsig3

    # ---- (4) best & worst full-sample formats both answerable
    print("\n(4) subset where the full-sample best and worst formats are both V/D")
    results["best_worst_answerable"] = {}
    nsig4 = 0
    for md in MODS:
        fmts = FORMATS[md]
        for name in CORE:
            allq = [q for q in cat if meta[q]["modality"] == md]
            full = {f: em_of([EM[name][q][f] for q in allq]) for f in fmts}
            b, w = max(full, key=full.get), min(full, key=full.get)
            qs = [q for q in allq if cat[q][b] in "VD" and cat[q][w] in "VD"]
            pf = {f: em_of([EM[name][q][f] for q in qs]) for f in fmts}
            if qs:
                obs, p = perm_p([(EM[name][q][b], EM[name][q][w]) for q in qs])
            else:
                obs, p = float("nan"), float("nan")
            sig = bool(qs) and p * N_CELLS < 0.01
            nsig4 += sig
            results["best_worst_answerable"][f"{name}|{md}"] = {
                "best_full": b, "worst_full": w, "gap_full": full[b] - full[w], "n": len(qs),
                "per_format": pf, "gap_subset": pf[b] - pf[w] if qs else None,
                "p": p, "p_bonf12": min(1.0, p * N_CELLS) if qs else None,
                "sig_bonf12_p<0.01": sig}
            print(f"  {name:14s} {md:10s} best={b:16s} worst={w:16s} full_gap={full[b]-full[w]:5.1f}"
                  f"  n={len(qs):4d}  EM_best={f1(pf[b])} EM_worst={f1(pf[w])}"
                  f" gap={f1(pf[b]-pf[w]) if qs else '  nan'} p={fmt_p(p) if qs else 'nan'}"
                  f" bonf12={fmt_p(min(1.0, p*N_CELLS)) if qs else 'nan'} {'SIG' if sig else 'ns'}")
    print(f"  cells significant after Bonferroni x12 (p_bonf<0.01): {nsig4}/12")
    results["best_worst_answerable_nsig"] = nsig4

    # ---- (5) flip rate on fully answerable subset
    print("\n(5) flip rate on the fully answerable subset (% questions whose EM differs across formats)")
    results["flip_rate_fully_answerable"] = {}
    for name in CORE:
        parts = []
        for md in MODS:
            qs = full_q[md]
            fl = sum(1 for q in qs if len(set(EM[name][q].values())) > 1)
            fr = 100 * fl / len(qs) if qs else float("nan")
            allq = [q for q in cat if meta[q]["modality"] == md]
            fr_all = 100 * sum(1 for q in allq if len(set(EM[name][q].values())) > 1) / len(allq)
            results["flip_rate_fully_answerable"][f"{name}|{md}"] = {
                "n": len(qs), "flip_rate": fr, "flip_rate_full_sample": fr_all}
            parts.append(f"{md}={f1(fr)} (n={len(qs)}; full sample {fr_all:.1f})")
        print(f"  {name:14s} " + "   ".join(parts))

    # ---- (6) tabular bar/scatter on their V questions vs table_image
    print("\n(6) tabular: bar_chart / scatter_plot on their V questions vs table_image on the same questions")
    results["tabular_visible_vs_table"] = {}
    for f in ("bar_chart", "scatter_plot"):
        qs = [q for q, cc in cat.items() if meta[q]["modality"] == "tabular" and cc[f] == "V"]
        tasks = Counter(meta[q]["task"] for q in qs)
        print(f"  {f}: n={len(qs)}  tasks={dict(tasks)}")
        for name in CORE:
            a = em_of([EM[name][q][f] for q in qs])
            t = em_of([EM[name][q]["table_image"] for q in qs])
            tcat = Counter(cat[q]["table_image"] for q in qs)
            results["tabular_visible_vs_table"][f"{name}|{f}"] = {
                "n": len(qs), "em_format": a, "em_table_image": t,
                "table_image_categories": dict(tcat)}
            print(f"    {name:14s} {f}={f1(a)}  table_image={f1(t)}  diff={f1(a - t)}"
                  f"   (table_image cats on these q: {dict(tcat)})")

    with open(os.path.join(OUT_DIR, "answerability_results.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    print("\nwrote scripts/analysis/answerability_matrix.md, scripts/analysis/answerability_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
