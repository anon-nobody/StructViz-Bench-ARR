#!/usr/bin/env python3
"""v1 vs v2 ("answer-complete" rendering) analysis for Qwen2.5-VL-7B.

Stdlib only, no inference. Consumes the v2 evaluation outputs written by
scripts/eval_local_suite.py and compares them with the released v1 predictions.

    python scripts/analysis/v2_analysis.py            # full run (all v2 files required)
    python scripts/analysis/v2_analysis.py --smoke    # whatever partial shards exist

Inputs (paths relative to the project root, overridable by flags):
    benchmark/realworld_test.jsonl           questions, answers, data, data_id (cluster unit)
    results/full_qwen_extracted.jsonl        v1 predictions, one row per (question_id, viz_type)
    results/v2/v2_qwen_shard*.jsonl          v2 predictions (merged; deduped by key)
    results/v2/noimage_qwen.jsonl            question-text-only baseline (viz_type "none")
    results/v2/assist_qwen.jsonl             tabular calculation-aid 2x2
    benchmark/render_v2/manifest.jsonl       v2 image sizes + renderer constants

Conventions follow scripts/analysis/cluster_cis.py: every statistic is a ratio estimator
sum(d)/sum(n) over data_id clusters (point estimate = plain question-level mean); percentile
95% object-cluster bootstrap CIs (B=5000); two-sided object-level sign-flip permutation p
(B=10000, p=(hits+1)/(B+1)); each resampling call uses a fresh random.Random(0). Best/worst
formats are chosen once on the analysed sample and held fixed inside resamples.

Deduplication: eval_local_suite.py appends, and retries [ERROR] rows by appending again, so
for every (question_id, viz_type) the LAST non-error row wins; keys with only error rows are
dropped and counted. The "main" v2 format set equals the v1 set; the graph-only diagnostic
format text_only_deg is used only in section 7.

Writes scripts/analysis/v2_results.json (smoke: v2_results_smoke.json).

Revision (H1 audit fix, 2026-11-08): three v2 tabular scatter_plot rules were too strict or wrong
(the H1 test found models above the question-only prior on these "absent" pairs):
  * comparison of two columns that are plotted together (main panel = first two numeric
    columns, or one of the extra pair panels) is D: both are drawn with every row on labelled
    axes (src/rendering/v2_renderers.py, tabular_scatter_plot l.398-401, l.423-424, l.431-443);
  * value_extraction of the label column is V when the full key is printed: every point is
    annotated "idx: <first categorical value>" with row_labels(df, max_chars=8), which cuts
    values longer than 8 characters to 7 + "~" (l.382, l.419-422; row_labels l.146-158), so V
    iff the queried column is the first categorical column and len(key) <= 8; strictify_v2 now
    demotes only the position-estimated numeric case (D -> A);
  * tasks on the synthetic __row_index__ column (0..n-1, the only numeric column of its 284
    questions' tables) are D on the scatter: the column is plotted against the row index and
    every point is labelled with that index (l.402-405, l.419-422); ranking / outlier_detection
    additionally need the answer label printed in full (as above);
  * the extra pair panels are chosen as the renderer does (pairs other than (0, 1) sorted by
    descending |Pearson r|, first V2_SCATTER_MAX_PAIR_PANELS; l.384-388), not in column order.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import struct
import sys
from collections import Counter, defaultdict
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import answerability as ans  # noqa: E402  (classify_* -> dict viz_type -> "V"/"D"/"A"/"C")

SEED = 0
B_BOOT = 5000
B_PERM = 10000

MODS = ["tabular", "timeseries", "graph"]
FORMATS = ans.FORMATS  # identical format set in v1 and v2 ("main" formats)
DEG_FMT = "text_only_deg"
ASSIST = ["table_full", "table_full_means", "bar_rows", "bar_rows_means"]
TS_LOSSY = {"gaf", "recurrence_plot"}
CATS = ["V", "D", "A", "C"]
ROW_INDEX = "__row_index__"

# v2 renderer defaults (src/rendering/v2_renderers.py); overridden per question by the
# renderer_constants recorded in the manifest row when present.
V2_DEFAULTS = {"V2_BAR_ANNOTATE_MAX_ROWS": 35, "V2_BAR_MAX_PANELS": None,
               "V2_ROW_LABEL_MAX_CHARS": 12, "V2_SCATTER_MAX_PAIR_PANELS": 4}
SCATTER_LABEL_MAX_CHARS = 8  # row_labels(df, max_chars=8) in tabular_scatter_plot

BRANCH: Counter = Counter()

# --strict-answerable: time-series tasks whose (numeric) answer must be read to exact-match
# precision; any drawing without value annotations (line_plot, heatmap strip, gaf, recurrence
# plot) becomes A for them. median_mean_relation / anomaly_detection count only when the
# answer is numeric (every anomaly_detection answer is a timestep; median_mean_relation
# answers are all non-numeric in realworld_test).
TS_VALUE_PRECISE = {"value_lookup", "range_query", "threshold_count", "mean_shift_magnitude",
                    "forecasting"}
TS_VALUE_PRECISE_IF_NUMERIC = {"median_mean_relation", "anomaly_detection"}
TS_DRAWINGS = ["line_plot", "heatmap", "gaf", "recurrence_plot"]
TS_NONLOSSY = ["line_plot", "heatmap", "text_only"]

# --model: file names per evaluated model (qwen = default, full v2 suite in 4 shards;
# internvl = v2 subset of default-fully-answerable questions, single file)
MODEL_FILES = {
    "qwen": {"name": "Qwen2.5-VL-7B", "v1": "full_qwen_extracted.jsonl",
             "v2_glob": "v2_qwen_shard*.jsonl", "noimg": "noimage_qwen.jsonl",
             "assist": "assist_qwen.jsonl", "manifest": "manifest.jsonl"},
    "internvl": {"name": "InternVL2.5-8B", "v1": "full_internvl.jsonl",
                 "v2_glob": "v2_internvl_subset.jsonl", "noimg": "noimage_internvl.jsonl",
                 "assist": "assist_internvl.jsonl", "manifest": "manifest_subset32b.jsonl"},
}

# section 9 (hand-rule selector) inputs: v1 predictions of the 7 models used in cluster_cis.py
SELECTOR_MODELS = {
    "GPT-4o": "full_gpt4o_extracted.jsonl",
    "Gemini Flash": "full_gemini_extracted.jsonl",
    "Qwen2.5-VL-7B": "full_qwen_extracted.jsonl",
    "Claude Sonnet": "full_claude_extracted.jsonl",
    "Qwen2.5-VL-32B": "full_qwen32b.jsonl",
    "InternVL2.5-8B": "full_internvl.jsonl",
    "Gemini-2.5": "full_gemini25.jsonl",
}


# ---------------------------------------------------------------- loading helpers
def read_jsonl(path):
    """Read JSONL, skipping blank and truncated lines (a running writer may leave one)."""
    out, bad = [], 0
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"  WARN {os.path.relpath(path, ROOT)}: skipped {bad} unparseable line(s)")
    return out


def is_error(r):
    return str(r.get("prediction", "")) == "[ERROR]" or bool(r.get("error"))


def dedupe(rows):
    """Last non-error row per (question_id, viz_type); returns (dict, stats)."""
    good, err_keys = {}, set()
    n_dup = 0
    for r in rows:
        k = (r["question_id"], r["viz_type"])
        if is_error(r):
            err_keys.add(k)
            continue
        if k in good:
            n_dup += 1
        good[k] = r
    only_err = err_keys - set(good)
    return good, {"rows_read": len(rows), "keys_ok": len(good), "duplicates_replaced": n_dup,
                  "keys_error_only": len(only_err), "error_rows_total":
                  sum(1 for r in rows if is_error(r))}


def em_table(keyed):
    """{(q, f): row} -> EM[q][f], PRED[q][f]."""
    em, pred = defaultdict(dict), defaultdict(dict)
    for (q, f), r in keyed.items():
        em[q][f] = float(r.get("exact_match", 0) or 0)
        pred[q][f] = r.get("prediction", "")
    return em, pred


def is_numeric_str(s):
    return re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*", str(s)) is not None


def numerical_accuracy(pred, answer, abs_tol=0.05, rel_tol=0.01):
    """Fallback copy of src.evaluation.metrics.numerical_accuracy (no answer extraction)."""
    try:
        pv, av = float(str(pred).strip()), float(str(answer).strip())
    except ValueError:
        return 0.0
    d = abs(pv - av)
    if d <= abs_tol:
        return 1.0
    den = max(abs(av), abs(pv))
    return 1.0 if den > 0 and d / den <= rel_tol else 0.0


def tol_table(keyed, meta):
    """Tolerant score: numeric answers count as correct within abs 0.05 / rel 1% (the rows'
    numeric_accuracy field when present), otherwise exact match."""
    out = defaultdict(dict)
    for (q, f), r in keyed.items():
        em = float(r.get("exact_match", 0) or 0)
        ansr = meta[q]["answer"] if q in meta else r.get("answer")
        if is_numeric_str(ansr):
            na = r.get("numeric_accuracy")
            na = float(na) if na is not None else numerical_accuracy(r.get("prediction", ""),
                                                                    ansr)
            em = max(em, na)
        out[q][f] = em
    return out


def load_near_constant():
    """(modality, task) cells with one answer >= 90% (prior_and_subsets_results.json 1(c))."""
    p = os.path.join(HERE, "prior_and_subsets_results.json")
    d = json.load(open(p))
    return {(c["modality"], c["task"]) for c in d["near_constant_tasks"]}


def png_size(path):
    with open(path, "rb") as fh:
        head = fh.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", head[16:24])


# ---------------------------------------------------------------- resampling core
# (copied from scripts/analysis/cluster_cis.py; empty-input guards added)
def clusters_from(items, q2d):
    """items: iterable of (question_id, value). Returns list of (sum, n) per data_id."""
    agg = defaultdict(lambda: [0.0, 0])
    for q, v in items:
        a = agg[q2d[q]]
        a[0] += v
        a[1] += 1
    return [tuple(v) for v in agg.values()]


def point(cl):
    n = sum(n for _, n in cl)
    return 100 * sum(s for s, _ in cl) / n if n else float("nan")


def pct(xs, lo=2.5, hi=97.5):
    xs = sorted(xs)
    k = len(xs) - 1

    def q(p):
        f = p / 100 * k
        i = int(f)
        j = min(i + 1, k)
        return xs[i] + (xs[j] - xs[i]) * (f - i)

    return q(lo), q(hi)


def cluster_boot(cl, B=None):
    B = B or B_BOOT
    if not cl:
        return {"lo": float("nan"), "hi": float("nan"), "hw": float("nan")}
    rng = random.Random(SEED)
    m = len(cl)
    stats = []
    for _ in range(B):
        s = n = 0.0
        for _ in range(m):
            a, b = cl[int(rng.random() * m)]
            s += a
            n += b
        stats.append(100 * s / n)
    lo, hi = pct(stats)
    return {"lo": lo, "hi": hi, "hw": (hi - lo) / 2}


def cluster_perm(cl, B=None):
    """Two-sided sign-flip test: flip all paired differences of one object together."""
    B = B or B_PERM
    if not cl:
        return float("nan"), None
    rng = random.Random(SEED)
    sums = [s for s, _ in cl]
    N = sum(n for _, n in cl)
    obs = abs(sum(sums) / N)
    hits = 0
    for _ in range(B):
        t = 0.0
        for s in sums:
            t += s if rng.random() < 0.5 else -s
        if abs(t / N) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (B + 1), hits


def summarize(cl, perm=False):
    out = {"est": point(cl), "n_obj": len(cl), "n_units": sum(n for _, n in cl)}
    out["obj_ci"] = cluster_boot(cl)
    if perm:
        out["obj_perm_p"], out["obj_perm_hits"] = cluster_perm(cl)
    return out


def contrast(items, q2d, perm=False):
    return summarize(clusters_from(items, q2d), perm=perm)


# ---------------------------------------------------------------- formatting
def pp(x):
    if x != x:
        return "nan"
    return f"{x:+.1f}" if x < 0 or x > 0 else f"{x:.1f}"


def f1(x):
    return "nan" if x is None or x != x else f"{x:.1f}"


def fp(p, hits=None):
    if p != p:
        return "nan"
    if hits == 0 or p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"


def ci_str(c):
    if c["lo"] != c["lo"]:
        return "[nan]"
    return f"[{c['lo']:+.1f}, {c['hi']:+.1f}]"


def section(t):
    print(f"\n{t}\n" + "-" * 100)


def mean100(vals):
    return 100 * sum(vals) / len(vals) if vals else float("nan")


def norm_pred(p):
    return str(p).strip().lower()


# ---------------------------------------------------------------- v2 answerability
def _first_categorical(rows, num):
    for c in rows[0]:
        if c not in num:
            return c
    return None


def _label_visible(r, label_col, num, max_chars, numeric_ok):
    """Is the answer (a label_col value) readable from row tick labels ``idx: <first cat>``?"""
    if label_col == ROW_INDEX:
        return True
    if label_col in num:
        return numeric_ok
    if label_col != _first_categorical(r["data"], num):
        return False
    return len(str(r["answer"])) <= max_chars


def scatter_pair_sets(rows, num_list, k):
    """Column pairs drawn in the v2 scatter's extra panels, exactly as
    v2_renderers.tabular_scatter_plot chooses them (l.384-388): every pair except (0, 1), sorted
    by descending |Pearson r| (NaN -> 0, ties by pair index), first k. (Revision fix: the
    earlier audit took the first k pairs in column order.)"""
    import math

    cols = [[float(rw[c]) for rw in rows] for c in num_list]

    def corr(x, y):
        n = len(x)
        if n < 2:
            return 0.0
        mx, my = sum(x) / n, sum(y) / n
        sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
        sxx = sum((a - mx) ** 2 for a in x)
        syy = sum((b - my) ** 2 for b in y)
        if sxx <= 0 or syy <= 0:
            return 0.0
        v = abs(sxy / math.sqrt(sxx * syy))
        return 0.0 if v != v else v
    pairs = [p for p in combinations(range(len(num_list)), 2) if p != (0, 1)]
    pairs.sort(key=lambda p: (-corr(cols[p[0]], cols[p[1]]), p))
    return [{num_list[i], num_list[j]} for i, j in pairs[:k]]


def classify_v2_tabular(r, const):
    task, q, rows = r["task"], r["question"], r["data"]
    nrows = len(rows)
    num_list = ans.numeric_cols(rows)
    num = set(num_list)
    qc = ans.quoted(q)
    has_avg = "average" in re.sub(r"'[^']*'", "", q).lower()
    lab_max = const["V2_ROW_LABEL_MAX_CHARS"]
    bar_cap = const["V2_BAR_MAX_PANELS"]
    bar_cols = set(num_list[:bar_cap] if bar_cap else num_list)
    annotated = nrows <= const["V2_BAR_ANNOTATE_MAX_ROWS"]
    first2 = set(num_list[:2]) if len(num_list) >= 2 else set()
    pair_sets = scatter_pair_sets(rows, num_list, const["V2_SCATTER_MAX_PAIR_PANELS"])

    def heat_vis(c):  # heatmap shows numeric columns (annotated) and the index
        return c in num or c == ROW_INDEX

    def bar_vis(c, need_values):
        if c == ROW_INDEX:
            return True
        return c in bar_cols and (annotated or not need_values)

    out = {"table_image": "D", "text_only": "D"}
    if task == "value_extraction":
        col = re.search(r"value of '([^']*)' in row (\d+)", q).group(1)
        out["table_image"] = out["text_only"] = "V"
        lab = _label_visible(r, col, num, lab_max, True)
        out["heatmap"] = "V" if (col in num or lab) else "A"
        out["bar_chart"] = "V" if (bar_vis(col, True) or (col not in num and lab)) else "A"
        # scatter: every point is annotated with row_labels(df, max_chars=8), i.e. "idx: <first
        # categorical value>" cut to 7 chars + "~" when longer (v2_renderers.tabular_scatter_plot
        # l.382 / l.419-422, row_labels l.146-158): the full key is printed iff the queried column
        # is the first categorical column and the key has <= 8 characters -> V. A numeric column
        # of the main panel can only be estimated from the point position -> D (strict: A).
        lab_s = _label_visible(r, col, num, SCATTER_LABEL_MAX_CHARS, False)
        out["scatter_plot"] = "V" if lab_s else "D" if col in first2 else "A"
        BRANCH[f"v2tab.value_extraction col_numeric={col in num} label_visible={lab} "
               f"scatter_label={lab_s}"] += 1
    elif task == "comparison" or (task not in ("aggregation", "counterfactual") and has_avg):
        a, b = qc[0], qc[1]
        out["heatmap"] = "D" if heat_vis(a) and heat_vis(b) else "A"
        # Lead-specified: bar chart mean comparison = V (row bars; means must be estimated).
        out["bar_chart"] = "V" if bar_vis(a, False) and bar_vis(b, False) else "A"
        # scatter: the two plotted columns of the main panel (first two numeric columns, axis
        # labels + ticks, v2_renderers.tabular_scatter_plot l.398-401, l.423-424) or of a pair
        # panel (l.431-443) are drawn with every row, so which of the two has the higher mean is
        # derivable from the plotted ranges -> D; any other pair is not drawn together -> A.
        both_plotted = {a, b} == first2 or {a, b} in pair_sets
        out["scatter_plot"] = "D" if both_plotted else "A"
        BRANCH[f"v2tab.comparison-rule[{task}] both_numeric={a in num and b in num} "
               f"scatter_plotted={both_plotted}"] += 1
    elif task in ("aggregation", "filtering", "trend_analysis"):
        col = qc[0]
        need = task != "trend_analysis"
        out["heatmap"] = "D" if heat_vis(col) else "A"
        out["bar_chart"] = "D" if bar_vis(col, need) else "A"
        # scatter: the synthetic __row_index__ column (0..n-1, always the only numeric column of
        # its table) is plotted against the row index and every point is labelled with its index
        # (tabular_scatter_plot l.402-405, l.419-422; row_labels l.146-158), so its values are
        # printed exactly -> D. Any other column can only be read from point positions -> A.
        out["scatter_plot"] = "D" if col == ROW_INDEX else "A"
        BRANCH[f"v2tab.{task} col_numeric={col in num} row_index={col == ROW_INDEX} "
               f"bar_annotated={annotated}"] += 1
    elif task in ("ranking", "outlier_detection"):
        label_col, val_col = qc[0], qc[1]
        lab_h = _label_visible(r, label_col, num, lab_max, True)
        lab_s = _label_visible(r, label_col, num, SCATTER_LABEL_MAX_CHARS, False)
        out["heatmap"] = "D" if heat_vis(val_col) and lab_h else "A"
        out["bar_chart"] = ("D" if bar_vis(val_col, task == "outlier_detection") and lab_h
                            else "A")
        # scatter: the value column must be plotted (main panel; __row_index__ is plotted on the
        # y axis and printed as the point index) and the answer label printed in full.
        out["scatter_plot"] = ("D" if (val_col in first2 or val_col == ROW_INDEX) and lab_s
                               else "A")
        BRANCH[f"v2tab.{task} value_numeric={val_col in num or val_col == ROW_INDEX} "
               f"label_visible={lab_h}"] += 1
    elif task == "correlation":
        a, b = qc[0], qc[1]
        out["heatmap"] = "D" if heat_vis(a) and heat_vis(b) else "A"
        out["bar_chart"] = "D" if bar_vis(a, False) and bar_vis(b, False) else "A"
        vis = {a, b} == first2 or {a, b} in pair_sets
        out["scatter_plot"] = "V" if vis else "A"
        in_panel = {a, b} in pair_sets
        BRANCH[f"v2tab.correlation first_pair={first2 == {a, b}} pair_panel={in_panel}"] += 1
    elif task == "counterfactual":
        cols = qc[:2] if len(qc) >= 2 else qc
        out["heatmap"] = "D" if all(heat_vis(c) for c in cols) else "A"
        out["bar_chart"] = "D" if all(bar_vis(c, False) for c in cols) else "A"
        out["scatter_plot"] = "A"
        BRANCH[f"v2tab.counterfactual cols_numeric={all(c in num for c in cols)}"] += 1
    else:
        raise ValueError(task)
    return out


def classify_v2_timeseries(r):
    out = ans.classify_timeseries(r)
    out["text_only"] = out["line_plot"]  # v2 text view lists every point
    return out


def classify_v2_graph(r):
    task, q = r["task"], r["question"]
    fmts = FORMATS["graph"] + [DEG_FMT]
    if task == "connectivity" or (task == "counterfactual"
                                  and ans.RE_CF_ADD.fullmatch(q.strip())):
        return {f: "C" for f in fmts}
    out = {f: "D" for f in fmts}
    if task == "degree_query":
        out[DEG_FMT] = "V"
    return out


def classify_v2(r, const):
    md = r["modality"]
    if md == "tabular":
        return classify_v2_tabular(r, const)
    if md == "timeseries":
        return classify_v2_timeseries(r)
    return classify_v2_graph(r)


def strictify_v2(r, cats):
    """--strict-answerable: V/D only if readable to exact-match precision.

    tabular: scatter_plot shows row labels but no value annotations -> value_extraction of a
      numeric column (D, a position estimate) becomes A; value_extraction of the label column
      whose full value is printed next to each point stays V (revision fix, see
      classify_v2_tabular); ranking / outlier_detection keep D when the label is readable;
      correlation keeps V; comparison of two plotted columns keeps D;
      bar_chart value_extraction is already V only when bars are annotated (<=35 rows).
    timeseries: value-precise tasks (TS_VALUE_PRECISE, plus TS_VALUE_PRECISE_IF_NUMERIC with
      a numeric answer) -> A in every drawing without annotations; text_only unchanged.
    graph: unchanged (degree, shortest_path, diameter, edge_count, clustering are counting,
      kept D).
    """
    out = dict(cats)
    md, task = r["modality"], r["task"]
    if md == "tabular" and task == "value_extraction" and out["scatter_plot"] == "D":
        out["scatter_plot"] = "A"
        BRANCH["strict.tab scatter value_extraction (position estimate) -> A"] += 1
    elif md == "timeseries":
        vp = task in TS_VALUE_PRECISE or (task in TS_VALUE_PRECISE_IF_NUMERIC
                                          and is_numeric_str(r["answer"]))
        if vp:
            changed = [f for f in TS_DRAWINGS if out[f] in "VD"]
            for f in changed:
                out[f] = "A"
            BRANCH[f"strict.ts {task} drawings->A changed={','.join(changed) or '-'}"] += 1
    return out


# ---------------------------------------------------------------- per-question tables
def complete(EM, qs, fmts):
    return [q for q in qs if q in EM and all(f in EM[q] for f in fmts)]


def gap_block(EM, qs, fmts, q2d):
    """Best-worst gap on questions with every format; best/worst fixed on this sample."""
    qs = complete(EM, qs, fmts)
    if not qs:
        return {"n_q": 0}
    pf = {f: mean100([EM[q][f] for q in qs]) for f in fmts}
    b, w = max(pf, key=pf.get), min(pf, key=pf.get)
    r = contrast([(q, EM[q][b] - EM[q][w]) for q in qs], q2d, perm=True)
    r.update(n_q=len(qs), best=b, worst=w, per_format=pf)
    return r


def flip_cr(EM, PRED, qs, fmts):
    qs = complete(EM, qs, fmts)
    flips, crs = [], []
    for q in qs:
        flips.append(float(len({EM[q][f] for f in fmts}) > 1))
        pr = [norm_pred(PRED[q][f]) for f in fmts]
        k = len(pr)
        agree = sum(1 for i in range(k) for j in range(i + 1, k) if pr[i] == pr[j])
        crs.append(agree / (k * (k - 1) / 2) if k > 1 else 1.0)
    return qs, flips, crs


def gap_line(tag, g):
    if not g.get("n_q"):
        return f"  {tag:34s} (no complete questions)"
    return (f"  {tag:34s} {g['best']:>16s} {g['worst']:>16s} {f1(g['est']):>6s} "
            f"{ci_str(g['obj_ci']):>15s} {fp(g['obj_perm_p'], g['obj_perm_hits']):>7s} "
            f"{g['n_q']:5d} {g['n_obj']:4d}")


# ---------------------------------------------------------------- sections
def s1_categories(meta, cat1, cat2, qs_by_mod):
    section("1. Answerability categories per modality: share of (question, format) pairs, "
            "v1 vs v2 (main formats)")
    print(f"  {'modality':10s} {'suite':5s} {'pairs':>6s} " + " ".join(f"{k:>6s}" for k in CATS))
    out = {}
    for md in MODS:
        rec = {}
        for tag, cat in (("v1", cat1), ("v2", cat2)):
            c = Counter()
            for q in qs_by_mod[md]:
                c.update(cat[q][f] for f in FORMATS[md])
            tot = sum(c.values())
            rec[tag] = {"n_pairs": tot, **{k: 100 * c[k] / tot for k in CATS}}
            print(f"  {md:10s} {tag:5s} {tot:6d} "
                  + " ".join(f"{f1(rec[tag][k]):>6s}" for k in CATS))
        rec["A_share_change"] = rec["v2"]["A"] - rec["v1"]["A"]
        fa1 = sum(all(cat1[q][f] in "VD" for f in FORMATS[md]) for q in qs_by_mod[md])
        fa2 = sum(all(cat2[q][f] in "VD" for f in FORMATS[md]) for q in qs_by_mod[md])
        rec["fully_answerable_q"] = {"v1": fa1, "v2": fa2, "n": len(qs_by_mod[md])}
        print(f"  {md:10s} A share change v2-v1 {pp(rec['A_share_change'])} pp;  fully "
              f"answerable questions v1 {fa1}/{len(qs_by_mod[md])}, v2 {fa2}/"
              f"{len(qs_by_mod[md])}")
        out[md] = rec
    return out


def s2_per_format(EM1, EM2, qs_by_mod, q2d):
    section("2. Per-format EM v1 vs v2 (each on all available questions) and paired v2-v1 "
            "change (object-cluster 95% CI)")
    print(f"  {'modality':10s} {'format':16s} {'v1':>6s} {'n1':>5s} {'v2':>6s} {'n2':>5s} "
          f"{'d(v2-v1)':>8s} {'obj CI':>15s} {'n_pair':>6s} {'n_obj':>5s}")
    out = {}
    for md in MODS:
        for f in FORMATS[md]:
            q1 = [q for q in qs_by_mod[md] if f in EM1.get(q, {})]
            q2 = [q for q in qs_by_mod[md] if f in EM2.get(q, {})]
            both = [q for q in q2 if f in EM1.get(q, {})]
            r = contrast([(q, EM2[q][f] - EM1[q][f]) for q in both], q2d)
            rec = {"v1": mean100([EM1[q][f] for q in q1]), "n_v1": len(q1),
                   "v2": mean100([EM2[q][f] for q in q2]), "n_v2": len(q2),
                   "v1_on_paired": mean100([EM1[q][f] for q in both]),
                   "v2_on_paired": mean100([EM2[q][f] for q in both]), "paired": r}
            out[f"{md}|{f}"] = rec
            print(f"  {md:10s} {f:16s} {f1(rec['v1']):>6s} {len(q1):5d} {f1(rec['v2']):>6s} "
                  f"{len(q2):5d} {pp(r['est']):>8s} {ci_str(r['obj_ci']):>15s} "
                  f"{r['n_units']:6d} {r['n_obj']:5d}")
    return out


def _gap_rows(EM1, EM2, cat1, cat2, md, allq, q2d):
    fm = FORMATS[md]
    if True:
        rec = {"v1_all": gap_block(EM1, allq, fm, q2d),
               "v2_all": gap_block(EM2, allq, fm, q2d)}
        # same questions for both suites (complete in v1 and v2)
        common = complete(EM2, complete(EM1, allq, fm), fm)
        rec["v1_on_v2_complete"] = gap_block(EM1, common, fm, q2d)
        if md == "timeseries":
            fm_nl = [f for f in fm if f not in TS_LOSSY]
            rec["v2_excl_gaf_rp"] = gap_block(EM2, allq, fm_nl, q2d)
            rec["v1_excl_gaf_rp"] = gap_block(EM1, allq, fm_nl, q2d)
        fa2 = [q for q in allq if all(cat2[q][f] in "VD" for f in fm)]
        fa1 = [q for q in allq if all(cat1[q][f] in "VD" for f in fm)]
        rec["v2_fully_answerable"] = gap_block(EM2, fa2, fm, q2d)
        rec["v1_on_v2_fully_answerable"] = gap_block(EM1, fa2, fm, q2d)
        rec["v1_fully_answerable_v1cats"] = gap_block(EM1, fa1, fm, q2d)
        rec["n_fully_answerable_v2"] = len(fa2)
    return rec


def s3_gaps(EM1, EM2, cat1, cat2, qs_by_mod, q2d, nc=None, meta=None):
    section("3. Best-worst gap per modality (object sign-flip p, object-cluster CI)")
    print(f"  {'sample':34s} {'best':>16s} {'worst':>16s} {'gap':>6s} {'obj CI':>15s} "
          f"{'p_obj':>7s} {'n_q':>5s} {'nobj':>4s}")
    out = {}
    for md in MODS:
        allq = qs_by_mod[md]
        rec = _gap_rows(EM1, EM2, cat1, cat2, md, allq, q2d)
        print(f" {md}")
        for k, g in rec.items():
            if isinstance(g, dict):
                print(gap_line(k, g))
        if nc is not None:
            allx = [q for q in allq if (md, meta[q]["task"]) not in nc]
            recx = _gap_rows(EM1, EM2, cat1, cat2, md, allx, q2d)
            print(f" {md}  [near-constant cells excluded: n_q {len(allx)}/{len(allq)}]")
            for k, g in recx.items():
                if isinstance(g, dict):
                    print(gap_line(k, g))
            rec["excl_near_constant"] = recx
        out[md] = rec
    print("\n  HEADLINE (same information, different rendering) = v2_fully_answerable rows: "
          "every main v2 format V or D, C questions excluded.")
    return out


def fa_counts(meta, cats_by_label, qs_by_mod, nc):
    """All-answerable (every main format V/D) question counts per category set."""
    print(f"  {'modality':10s} {'categories':10s} {'all q':>7s} {'excl NC':>8s} {'n_q':>5s} "
          f"{'n_q excl NC':>11s}")
    out = {}
    for md in MODS:
        allq = qs_by_mod[md]
        allx = [q for q in allq if (md, meta[q]["task"]) not in nc]
        for lab, cats in cats_by_label.items():
            fa = [q for q in allq if all(cats[q][f] in "VD" for f in FORMATS[md])]
            fax = [q for q in fa if (md, meta[q]["task"]) not in nc]
            out[f"{md}|{lab}"] = {"all": len(fa), "excl_nc": len(fax), "n_q": len(allq),
                                  "n_q_excl_nc": len(allx),
                                  "tasks_excl_nc": dict(Counter(meta[q]["task"] for q in fax))}
            print(f"  {md:10s} {lab:10s} {len(fa):7d} {len(fax):8d} {len(allq):5d} "
                  f"{len(allx):11d}   tasks(excl NC) {out[f'{md}|{lab}']['tasks_excl_nc']}")
    return out


def s3b_identification(EM1, EM2, EMt1, EMt2, cat2, meta, qs_by_mod, q2d, nc, cat_label):
    section(f"3b. Identification checks (categories: {cat_label}; NC = near-constant "
            f"(modality, task) cells, one answer >= 90%)")
    print(f"  {'sample':34s} {'best':>16s} {'worst':>16s} {'gap':>6s} {'obj CI':>15s} "
          f"{'p_obj':>7s} {'n_q':>5s} {'nobj':>4s}")
    out = {}
    for md in MODS:
        fm = FORMATS[md]
        allq = qs_by_mod[md]
        allx = [q for q in allq if (md, meta[q]["task"]) not in nc]
        fa = [q for q in allq if all(cat2[q][f] in "VD" for f in fm)]
        fax = [q for q in fa if (md, meta[q]["task"]) not in nc]
        rec = {"v2_all_answerable": gap_block(EM2, fa, fm, q2d),
               "v2_all_answerable_exclNC": gap_block(EM2, fax, fm, q2d),
               "v1_on_v2_all_answerable_exclNC": gap_block(EM1, fax, fm, q2d)}
        if md == "timeseries":
            nl_fa = [q for q in allx if all(cat2[q][f] in "VD" for f in TS_NONLOSSY)]
            rec["v2_nonlossy3_all_exclNC"] = gap_block(EM2, allx, TS_NONLOSSY, q2d)
            rec["v2_nonlossy3_answerable_exclNC"] = gap_block(EM2, nl_fa, TS_NONLOSSY, q2d)
            rec["v1_nonlossy3_all_exclNC"] = gap_block(EM1, allx, TS_NONLOSSY, q2d)
            rec["v1_nonlossy3_answerable_exclNC"] = gap_block(EM1, nl_fa, TS_NONLOSSY, q2d)
        if md == "tabular":
            nve = [q for q in allx if meta[q]["task"] != "value_extraction"]
            fax_nve = [q for q in fax if meta[q]["task"] != "value_extraction"]
            rec["v2_all_exclNC_excl_value_extraction"] = gap_block(EM2, nve, fm, q2d)
            rec["v2_all_answerable_exclNC_excl_value_extraction"] = gap_block(
                EM2, fax_nve, fm, q2d)
            rec["v2_tol1pct_all_exclNC"] = gap_block(EMt2, allx, fm, q2d)
            rec["v2_tol1pct_all_answerable_exclNC"] = gap_block(EMt2, fax, fm, q2d)
            rec["v2_tol1pct_all_exclNC_excl_value_extraction"] = gap_block(EMt2, nve, fm, q2d)
            rec["v1_tol1pct_all_exclNC"] = gap_block(EMt1, allx, fm, q2d)
        print(f" {md}  (n_q all {len(allq)}, excl NC {len(allx)}; all-answerable {len(fa)}, "
              f"excl NC {len(fax)})")
        for k, g in rec.items():
            print(gap_line(k, g))
        if md == "tabular":
            for k in ("v2_tol1pct_all_exclNC", "v2_tol1pct_all_answerable_exclNC"):
                g = rec[k]
                if g.get("n_q"):
                    print(f"    {k} per-format: " + "  ".join(
                        f"{f} {f1(v)}" for f, v in g["per_format"].items()))
        out[md] = rec
    return out


def s4_flip(EM1, P1, EM2, P2, qs_by_mod, q2d, nc=None, meta=None):
    section("4. Flip rate (% questions whose EM differs across formats) and Consistency Rate "
            "(mean share of agreeing format pairs), main formats")
    print(f"  {'modality':10s} {'sample':18s} {'flip v1':>7s} {'flip v2':>7s} {'d':>6s} "
          f"{'obj CI':>15s} {'CR v1':>6s} {'CR v2':>6s} {'d':>6s} {'obj CI':>15s} {'n_q':>5s}")
    out = {}
    samples = [("", None)] + ([("excl-NC ", nc)] if nc is not None else [])
    for md in MODS:
        for tag, ncs in samples:
            qs = [q for q in qs_by_mod[md] if ncs is None or (md, meta[q]["task"]) not in ncs]
            rec = _flip_block(EM1, P1, EM2, P2, md, qs, q2d, tag)
            if ncs is None:
                out[md] = rec
            else:
                out[md]["excl_near_constant"] = rec
    return out


def _flip_block(EM1, P1, EM2, P2, md, qs, q2d, tag):
    fm = FORMATS[md]
    if True:
        rec = {}
        q1, fl1, cr1 = flip_cr(EM1, P1, qs, fm)
        q2, fl2, cr2 = flip_cr(EM2, P2, qs, fm)
        rec["v1_all"] = {"n": len(q1), "flip": mean100(fl1), "cr": mean100(cr1)}
        rec["v2_all"] = {"n": len(q2), "flip": mean100(fl2), "cr": mean100(cr2)}
        common = sorted(set(q1) & set(q2))
        d1 = dict(zip(q1, zip(fl1, cr1)))
        d2 = dict(zip(q2, zip(fl2, cr2)))
        df = contrast([(q, d2[q][0] - d1[q][0]) for q in common], q2d)
        dc = contrast([(q, d2[q][1] - d1[q][1]) for q in common], q2d)
        rec["paired"] = {"n": len(common),
                         "flip_v1": mean100([d1[q][0] for q in common]),
                         "flip_v2": mean100([d2[q][0] for q in common]),
                         "cr_v1": mean100([d1[q][1] for q in common]),
                         "cr_v2": mean100([d2[q][1] for q in common]),
                         "flip_diff": df, "cr_diff": dc}
        p = rec["paired"]
        print(f"  {md:10s} {tag + 'all (unpaired)':18s} {f1(rec['v1_all']['flip']):>7s} "
              f"{f1(rec['v2_all']['flip']):>7s} {'':>6s} {'':>15s} {f1(rec['v1_all']['cr']):>6s} "
              f"{f1(rec['v2_all']['cr']):>6s} {'':>6s} {'':>15s} "
              f"{rec['v1_all']['n']:5d}/{rec['v2_all']['n']}")
        print(f"  {md:10s} {tag + 'paired':18s} {f1(p['flip_v1']):>7s} {f1(p['flip_v2']):>7s} "
              f"{pp(df['est']):>6s} {ci_str(df['obj_ci']):>15s} {f1(p['cr_v1']):>6s} "
              f"{f1(p['cr_v2']):>6s} {pp(dc['est']):>6s} {ci_str(dc['obj_ci']):>15s} "
              f"{len(common):5d}")
    return rec


def load_prior():
    p = os.path.join(HERE, "prior_and_subsets_results.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {md: {"in_sample": d.get("prior_in_sample", {}).get(md, {}).get("acc"),
                 "cv_mean": d.get("prior_cv", {}).get(md, {}).get("mean")} for md in MODS}


def s5_noimage(EM0, EM2, meta, qs_by_mod, q2d):
    section("5. No-image baseline (question text only)")
    out = {"prior_file": None, "by_modality": {}, "by_task": {}}
    prior = load_prior()
    out["prior_file"] = prior
    # in-sample task-majority prior (predict the most frequent answer of the task)
    by_task = defaultdict(list)
    analysed = {q for md in MODS for q in qs_by_mod[md]}
    for q, r in meta.items():
        if q in analysed:
            by_task[(r["modality"], r["task"])].append(norm_pred(r["answer"]))
    task_prior = {k: 100 * Counter(v).most_common(1)[0][1] / len(v) for k, v in by_task.items()}

    print(f"  {'modality':10s} {'noimg':>6s} {'n':>5s} {'prior_in':>8s} {'prior_cv':>8s} "
          f"{'taskMaj':>7s} | formats beating no-image (obj CI lo>0) / total")
    for md in MODS:
        qs = [q for q in qs_by_mod[md] if q in EM0]
        em0 = mean100([EM0[q] for q in qs])
        tm = mean100([task_prior[(md, meta[q]["task"])] / 100 for q in qs])
        fmt_rows = {}
        n_beat = 0
        for f in FORMATS[md]:
            qq = [q for q in qs if f in EM2.get(q, {})]
            r = contrast([(q, EM2[q][f] - EM0[q]) for q in qq], q2d)
            beat = r["obj_ci"]["lo"] > 0
            n_beat += beat
            fmt_rows[f] = {"v2_em": mean100([EM2[q][f] for q in qq]),
                           "noimg_em": mean100([EM0[q] for q in qq]), "diff": r, "beats": beat}
        # questions the no-image model gets right: does every v2 format also get them?
        comp = complete(EM2, qs, FORMATS[md])
        right0 = [q for q in comp if EM0[q] >= 0.5]
        all_right = sum(all(EM2[q][f] >= 0.5 for f in FORMATS[md]) for q in right0)
        pr = prior[md] if prior else {"in_sample": None, "cv_mean": None}
        out["by_modality"][md] = {
            "noimg_em": em0, "n": len(qs), "prior_in_sample": pr["in_sample"],
            "prior_cv_mean": pr["cv_mean"], "task_majority_prior": tm, "formats": fmt_rows,
            "n_formats_beating": n_beat, "n_formats": len(FORMATS[md]),
            "noimg_correct_n": len(right0), "noimg_correct_all_v2_correct": all_right,
            "noimg_correct_all_v2_correct_pct": mean100([1.0] * all_right
                                                        + [0.0] * (len(right0) - all_right))}
        o = out["by_modality"][md]
        print(f"  {md:10s} {f1(em0):>6s} {len(qs):5d} {f1(pr['in_sample']):>8s} "
              f"{f1(pr['cv_mean']):>8s} {f1(tm):>7s} | {n_beat}/{len(FORMATS[md])}")
        for f, fr in fmt_rows.items():
            d = fr["diff"]
            print(f"      {f:16s} v2 {f1(fr['v2_em']):>5s} vs noimg {f1(fr['noimg_em']):>5s}  "
                  f"d {pp(d['est']):>6s} {ci_str(d['obj_ci']):>15s} n {d['n_units']:5d} "
                  f"{'BEATS' if fr['beats'] else ''}")
        print(f"      no-image correct on {len(right0)} complete questions; every v2 format "
              f"also correct on {all_right} ({f1(o['noimg_correct_all_v2_correct_pct'])}%)")
    print(f"\n  per task: {'modality':10s} {'task':28s} {'noimg':>6s} {'taskMaj':>7s} {'n':>4s}")
    for (md, t) in sorted(by_task):
        qs = [q for q in qs_by_mod[md] if q in EM0 and meta[q]["task"] == t]
        if not qs:
            continue
        e = mean100([EM0[q] for q in qs])
        out["by_task"][f"{md}|{t}"] = {"noimg_em": e, "n": len(qs),
                                       "task_majority_prior": task_prior[(md, t)]}
        print(f"            {md:10s} {t[:28]:28s} {f1(e):>6s} {f1(task_prior[(md, t)]):>7s} "
              f"{len(qs):4d}")
    return out


def subset_task_prior(meta, qs):
    """In-sample majority prior on the question set qs: per (modality, task) predict the most
    frequent (normalised) answer among qs; returns EM % over qs."""
    by = defaultdict(list)
    for q in qs:
        by[meta[q]["task"]].append(norm_pred(meta[q]["answer"]))
    hit = sum(Counter(v).most_common(1)[0][1] for v in by.values())
    return 100 * hit / len(qs) if qs else float("nan")


def s5b_cv_prior(EM2, qs_by_mod, q2d, meta=None, subset=False):
    section("5b. v2 format EM vs the CROSS-VALIDATED majority prior "
            "(prior_and_subsets_results.json prior_cv mean; prior treated as a constant)")
    prior = load_prior()
    out = {}
    print(f"  {'modality':10s} {'format':16s} {'EM':>6s} {'EM obj CI':>15s} {'prior':>6s} "
          f"{'EM-prior':>8s} {'CI of diff':>15s} {'n_q':>5s} {'nobj':>4s}  exceeds")
    for md in MODS:
        pv = prior[md]["cv_mean"] if prior else None
        n_ex = 0
        for f in FORMATS[md]:
            qq = [q for q in qs_by_mod[md] if f in EM2.get(q, {})]
            r = contrast([(q, EM2[q][f]) for q in qq], q2d)
            c = r["obj_ci"]
            ex = pv is not None and c["lo"] > pv
            n_ex += ex
            dci = ({"lo": c["lo"] - pv, "hi": c["hi"] - pv, "hw": c["hw"]} if pv is not None
                   else c)
            out[f"{md}|{f}"] = {"em": r["est"], "em_ci": c, "prior_cv": pv,
                                "diff": r["est"] - pv if pv is not None else None,
                                "diff_ci": dci, "n_q": r["n_units"], "n_obj": r["n_obj"],
                                "exceeds": ex}
            print(f"  {md:10s} {f:16s} {f1(r['est']):>6s} {ci_str(c):>15s} {f1(pv):>6s} "
                  f"{pp(r['est'] - pv) if pv is not None else 'nan':>8s} {ci_str(dci):>15s} "
                  f"{r['n_units']:5d} {r['n_obj']:4d}  {'YES' if ex else 'no'}")
        out[f"{md}|n_exceeding"] = n_ex
        print(f"  {md:10s} formats whose EM CI lies above the prior: {n_ex}/{len(FORMATS[md])}")
        if subset:
            sp = subset_task_prior(meta, qs_by_mod[md])
            n_ex2 = 0
            for f in FORMATS[md]:
                o = out[f"{md}|{f}"]
                ex2 = o["em_ci"]["lo"] > sp
                n_ex2 += ex2
                o.update(prior_subset_in_sample=sp, diff_subset_prior=o["em"] - sp,
                         exceeds_subset_prior=ex2)
                print(f"      vs in-sample majority prior on these {len(qs_by_mod[md])} subset "
                      f"questions ({f1(sp)}): {f:16s} EM-prior {pp(o['em'] - sp):>6s} "
                      f"CI [{o['em_ci']['lo'] - sp:+.1f}, {o['em_ci']['hi'] - sp:+.1f}]  "
                      f"{'YES' if ex2 else 'no'}")
            out[f"{md}|prior_subset_in_sample"] = sp
            out[f"{md}|n_exceeding_subset_prior"] = n_ex2
            print(f"  {md:10s} formats whose EM CI lies above the subset in-sample prior: "
                  f"{n_ex2}/{len(FORMATS[md])}")
    return out


def _selector_eval(recs, q2d, meta, nc):
    """Held-out comparison of random / fixed(mod) / hand rule / learned (mod, task) selector.

    Split and learned selector as scripts/mitigation/format_selector_grouped.py (md5(data_id)
    parity; best format per (modality, task), falling back to per-modality, learned on train;
    a question lacking the chosen format scores its mean over formats). nc: excluded
    (modality, task) cells, dropped from train and test.
    """
    byq = defaultdict(dict)
    tag = {}
    for r in recs:
        q = r["question_id"]
        byq[q][r["viz_type"]] = float(r.get("exact_match", 0) or 0)
        tag[q] = (r["modality"], r.get("task", "?"))
    if nc:
        byq = {q: v for q, v in byq.items() if tag[q] not in nc}

    def side(q):
        did = q2d.get(q, q)
        return "train" if int(hashlib.md5(did.encode()).hexdigest(), 16) % 2 == 0 else "test"

    acc_mt = defaultdict(lambda: defaultdict(list))
    acc_m = defaultdict(lambda: defaultdict(list))
    for q, fm in byq.items():
        if side(q) != "train":
            continue
        for v, e in fm.items():
            acc_mt[tag[q]][v].append(e)
            acc_m[tag[q][0]][v].append(e)
    best_mt = {k: max(d, key=lambda v: sum(d[v]) / len(d[v])) for k, d in acc_mt.items()}
    best_m = {k: max(d, key=lambda v: sum(d[v]) / len(d[v])) for k, d in acc_m.items()}

    def rule(q):
        md, task = tag[q]
        if md == "tabular":
            avg = "average" in re.sub(r"'[^']*'", "", meta[q]["question"]).lower()
            if task == "comparison" or (task == "aggregation" and avg):
                return "bar_chart"
            return "table_image"
        return "text_only"

    vals = defaultdict(list)
    d_sel_rule, d_rule_fix, d_rule_rand = [], [], []
    n_rule_missing = 0
    for q, fm in byq.items():
        if side(q) != "test":
            continue
        mean = sum(fm.values()) / len(fm)
        mt = best_mt.get(tag[q]) or best_m.get(tag[q][0])
        s_sel = fm.get(mt, mean)
        s_fix = fm.get(best_m.get(tag[q][0]), mean)
        rf = rule(q)
        n_rule_missing += rf not in fm
        s_rule = fm.get(rf, mean)
        vals["random"].append(mean)
        vals["fixed_mod"].append(s_fix)
        vals["rule"].append(s_rule)
        vals["selector"].append(s_sel)
        d_sel_rule.append((q, s_sel - s_rule))
        d_rule_fix.append((q, s_rule - s_fix))
        d_rule_rand.append((q, s_rule - mean))
    n = len(vals["random"])
    return {"n_test": n, **{k: mean100(v) for k, v in vals.items()},
            "selector_minus_rule": contrast(d_sel_rule, q2d),
            "rule_minus_fixed": contrast(d_rule_fix, q2d),
            "rule_minus_random": contrast(d_rule_rand, q2d),
            "fixed_formats_train": best_m, "n_rule_format_missing": n_rule_missing}


def s9_rule_selector(q2d, meta, nc):
    section("9. Hand-rule selector vs learned selector, held-out objects (md5(data_id) parity), "
            "v1 predictions")
    print("  rule: tabular comparison / aggregation-with-'average' -> bar_chart, other tabular "
          "-> table_image; timeseries, graph -> text_only")
    out = {}
    for tag, ncs in (("all cells", None), ("near-constant cells excluded", nc)):
        print(f"\n  [{tag}]")
        print(f"  {'model':15s} {'n_test':>6s} {'nobj':>4s} {'random':>7s} {'fixed':>6s} "
              f"{'rule':>6s} {'select':>7s} {'sel-rule':>8s} {'obj CI':>15s} {'rule-fix':>8s} "
              f"{'obj CI':>15s}")
        rec = {}
        for name, fn in SELECTOR_MODELS.items():
            path = os.path.join(ROOT, "results", fn)
            if not os.path.exists(path):
                print(f"  {name:15s} missing {fn}")
                continue
            r = _selector_eval(read_jsonl(path), q2d, meta, ncs)
            rec[name] = r
            sr, rf = r["selector_minus_rule"], r["rule_minus_fixed"]
            print(f"  {name:15s} {r['n_test']:6d} {sr['n_obj']:4d} {f1(r['random']):>7s} "
                  f"{f1(r['fixed_mod']):>6s} {f1(r['rule']):>6s} {f1(r['selector']):>7s} "
                  f"{pp(sr['est']):>8s} {ci_str(sr['obj_ci']):>15s} {pp(rf['est']):>8s} "
                  f"{ci_str(rf['obj_ci']):>15s}")
        out["all" if ncs is None else "excl_near_constant"] = rec
    return out


def s6_assist(EMa, meta, q2d):
    section("6. Assist 2x2 (tabular): {table, bar} x {means absent, present}")
    qs_all = [q for q in EMa if all(c in EMa[q] for c in ASSIST)]

    def avg_q(q):
        return "average" in re.sub(r"'[^']*'", "", meta[q]["question"]).lower()

    subsets = {
        "all": qs_all,
        "comparison": [q for q in qs_all if meta[q]["task"] == "comparison"],
        "aggregation_mean": [q for q in qs_all if meta[q]["task"] == "aggregation"
                             and avg_q(q)],
    }
    out = {"task_counts": dict(Counter(meta[q]["task"] for q in qs_all)),
           "n_questions_with_any": len(EMa)}
    print(f"  questions with all 4 conditions: {len(qs_all)} (any condition: {len(EMa)}); "
          f"tasks {out['task_counts']}")
    effects = {
        "means_effect_table": ("table_full_means", "table_full", None, None),
        "means_effect_bar": ("bar_rows_means", "bar_rows", None, None),
        "format_effect_no_means(table-bar)": ("table_full", "bar_rows", None, None),
        "format_effect_with_means(table-bar)": ("table_full_means", "bar_rows_means", None, None),
        "interaction(table_means_eff - bar_means_eff)":
            ("table_full_means", "table_full", "bar_rows_means", "bar_rows"),
    }
    for name, qs in subsets.items():
        rec = {"n": len(qs), "means": {c: mean100([EMa[q][c] for q in qs]) for c in ASSIST}}
        print(f"\n  [{name}] n={len(qs)}  " + "  ".join(f"{c} {f1(rec['means'][c])}"
                                                    for c in ASSIST))
        for ename, (a, b, c, d) in effects.items():
            if c is None:
                items = [(q, EMa[q][a] - EMa[q][b]) for q in qs]
            else:
                items = [(q, (EMa[q][a] - EMa[q][b]) - (EMa[q][c] - EMa[q][d])) for q in qs]
            r = contrast(items, q2d, perm=True)
            rec[ename] = r
            print(f"    {ename:46s} {pp(r['est']):>6s} {ci_str(r['obj_ci']):>15s} "
                  f"p_obj {fp(r['obj_perm_p'], r['obj_perm_hits']):>7s} n {r['n_units']} "
                  f"n_obj {r['n_obj']}")
        out[name] = rec
    return out


def s7_degree(EM2, meta, qs_by_mod, q2d):
    section("7. Graph degree confound: v2 text_only_deg minus text_only (paired)")
    qs = [q for q in qs_by_mod["graph"] if DEG_FMT in EM2.get(q, {})
          and "text_only" in EM2.get(q, {})]
    out = {}
    for name, sel in (("degree_query", lambda q: meta[q]["task"] == "degree_query"),
                      ("other_tasks", lambda q: meta[q]["task"] != "degree_query")):
        qq = [q for q in qs if sel(q)]
        r = contrast([(q, EM2[q][DEG_FMT] - EM2[q]["text_only"]) for q in qq], q2d, perm=True)
        r.update(em_text_only=mean100([EM2[q]["text_only"] for q in qq]),
                 em_text_only_deg=mean100([EM2[q][DEG_FMT] for q in qq]))
        out[name] = r
        print(f"  {name:13s} text_only {f1(r['em_text_only']):>5s}  text_only_deg "
              f"{f1(r['em_text_only_deg']):>5s}  d {pp(r['est']):>6s} {ci_str(r['obj_ci']):>15s} "
              f"p_obj {fp(r['obj_perm_p'], r['obj_perm_hits'])}  n {r['n_units']} "
              f"n_obj {r['n_obj']}")
    return out


def v1_image_sizes():
    """Mean pixels per (modality, format) over the v1 PNGs present in benchmark/rendered."""
    acc = defaultdict(list)
    known = sorted({f for fs in FORMATS.values() for f in fs}, key=len, reverse=True)
    for p in glob.glob(os.path.join(ROOT, "benchmark", "rendered", "**", "*.png"),
                       recursive=True):
        base = os.path.basename(p)[:-4]
        md = base.split("_")[0]
        f = next((k for k in known if base.endswith("_" + k)), None)
        sz = png_size(p)
        if f and sz and md in FORMATS:
            acc[(md, f)].append(sz[0] * sz[1])
    return acc


def s8b_model_input(v2rows):
    section("8b. Model input size from the v2 rows (image as fed to the model)")
    acc = defaultdict(lambda: defaultdict(list))
    for (q, f), r in v2rows.items():
        for k in ("image_w", "image_h", "model_input_w", "model_input_h", "n_visual_tokens"):
            if r.get(k) is not None:
                acc[(r["modality"], f)][k].append(float(r[k]))
    out = {}
    print(f"  {'modality':10s} {'format':16s} {'img w':>6s} {'img h':>6s} {'in w':>6s} "
          f"{'in h':>6s} {'vis tok':>7s} {'n':>5s}")
    for md in MODS:
        for f in FORMATS[md] + ([DEG_FMT] if md == "graph" else []):
            a = acc.get((md, f))
            if not a:
                continue
            m = {k: sum(v) / len(v) for k, v in a.items()}
            n = max(len(v) for v in a.values())
            out[f"{md}|{f}"] = {**m, "n": n}
            print(f"  {md:10s} {f:16s} "
                  + " ".join(f"{m.get(k, float('nan')):6.0f}" for k in
                             ("image_w", "image_h", "model_input_w", "model_input_h"))
                  + f" {m.get('n_visual_tokens', float('nan')):7.0f} {n:5d}")
    return out


def s8_cost(v2rows, man, noimg_rows):
    section("8. Cost: mean image pixels (Mpx) and mean latency (s) per format")
    v1px = v1_image_sizes()
    px2 = defaultdict(list)
    for (q, f), m in man.items():
        if m.get("width") and m.get("height"):
            px2[(m["modality"], f)].append(m["width"] * m["height"])
    lat2, rowpx = defaultdict(list), defaultdict(list)
    for (q, f), r in v2rows.items():
        if r.get("latency_s") is not None:
            lat2[(r["modality"], f)].append(float(r["latency_s"]))
        if r.get("image_w") and r.get("image_h"):
            rowpx[(r["modality"], f)].append(r["image_w"] * r["image_h"])
    print(f"  {'modality':10s} {'format':16s} {'v1 Mpx':>7s} {'n':>5s} {'v2 Mpx':>7s} {'n':>5s} "
          f"{'v2row Mpx':>9s} {'v2 lat':>7s} {'n':>5s}   (v1 latency not recorded)")
    out = {}

    def mpx(v):
        return sum(v) / len(v) / 1e6 if v else float("nan")

    for md in MODS:
        for f in FORMATS[md] + ([DEG_FMT] if md == "graph" else []):
            k = (md, f)
            rec = {"v1_mpx": mpx(v1px.get(k, [])), "n_v1_png": len(v1px.get(k, [])),
                   "v2_mpx_manifest": mpx(px2.get(k, [])), "n_v2_manifest": len(px2.get(k, [])),
                   "v2_mpx_rows": mpx(rowpx.get(k, [])),
                   "v2_latency_s": (sum(lat2[k]) / len(lat2[k])) if lat2.get(k) else None,
                   "n_v2_latency": len(lat2.get(k, []))}
            out[f"{md}|{f}"] = rec
            lat = rec["v2_latency_s"]
            lat_s = f"{lat:.2f}" if lat is not None else "n/a"
            print(f"  {md:10s} {f:16s} {rec['v1_mpx']:7.2f} {rec['n_v1_png']:5d} "
                  f"{rec['v2_mpx_manifest']:7.2f} {rec['n_v2_manifest']:5d} "
                  f"{rec['v2_mpx_rows']:9.2f} {lat_s:>7s} {rec['n_v2_latency']:5d}")
    lat0 = [float(r["latency_s"]) for r in noimg_rows.values() if r.get("latency_s") is not None]
    out["noimage_latency_s"] = sum(lat0) / len(lat0) if lat0 else None
    print(f"  no-image mean latency: "
          f"{f1(out['noimage_latency_s']) if lat0 else 'n/a'} s (n {len(lat0)}); "
          f"v1 pixels from the {sum(len(v) for v in v1px.values())} v1 PNGs present locally")
    return out


# ---------------------------------------------------------------- main
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--smoke", action="store_true",
                   help="Run on whatever v2 files/shards exist; B_boot=1000, B_perm=2000.")
    p.add_argument("--model", choices=sorted(MODEL_FILES), default="qwen",
                   help="Which model's v1/v2/no-image/assist files to analyse.")
    p.add_argument("--v1", default=None, help="Default: results/<model v1 file>.")
    p.add_argument("--v2-dir", default=os.path.join(ROOT, "results", "v2"))
    p.add_argument("--manifest", default=None,
                   help="Coverage/size manifest (default per model). Renderer constants are "
                        "always read from benchmark/render_v2/manifest.jsonl.")
    p.add_argument("--benchmark", default=os.path.join(ROOT, "benchmark",
                                                       "realworld_test.jsonl"))
    p.add_argument("--out", default=None)
    p.add_argument("--b-boot", type=int, default=None)
    p.add_argument("--b-perm", type=int, default=None)
    p.add_argument("--strict-answerable", action="store_true",
                   help="V/D only if readable to exact-match precision (see strictify_v2).")
    p.add_argument("--exclude-near-constant", action="store_true",
                   help="Also report sections 3/4 and all-answerable subsets with the "
                        "near-constant (modality, task) cells removed.")
    p.add_argument("--dump-fully-answerable", default=None, metavar="PATH",
                   help="Write the question ids answerable (V/D) in every main format under "
                        "the v2 categories (one per line) and exit; needs no predictions.")
    return p.parse_args()


def main() -> int:
    global B_BOOT, B_PERM
    a = parse_args()
    if a.smoke:
        B_BOOT, B_PERM = 1000, 2000
    B_BOOT = a.b_boot or B_BOOT
    B_PERM = a.b_perm or B_PERM
    extended = a.strict_answerable or a.exclude_near_constant
    mf = MODEL_FILES[a.model]
    msuf = "" if a.model == "qwen" else f"_{a.model}"
    out_path = a.out or os.path.join(HERE, f"v2_results{msuf}_smoke.json" if a.smoke
                                     else f"v2_results{msuf}_strict.json" if extended
                                     else f"v2_results{msuf}.json")
    a.v1 = a.v1 or os.path.join(ROOT, "results", mf["v1"])
    const_manifest = os.path.join(ROOT, "benchmark", "render_v2", "manifest.jsonl")
    a.manifest = a.manifest or os.path.join(ROOT, "benchmark", "render_v2", mf["manifest"])

    bench = read_jsonl(a.benchmark)
    meta = {r["question_id"]: r for r in bench}
    q2d = {q: r["data_id"] for q, r in meta.items()}
    qs_by_mod = {md: [q for q, r in meta.items() if r["modality"] == md] for md in MODS}

    shard_paths = sorted(glob.glob(os.path.join(a.v2_dir, mf["v2_glob"])))
    noimg_path = os.path.join(a.v2_dir, mf["noimg"])
    assist_path = os.path.join(a.v2_dir, mf["assist"])
    missing = [p for p in [noimg_path, assist_path] if not os.path.exists(p)]
    if not shard_paths:
        missing.insert(0, os.path.join(a.v2_dir, mf["v2_glob"]))
    if missing and not a.smoke and not a.dump_fully_answerable:
        print("ERROR: missing inputs (use --smoke for partial runs):\n  " + "\n  ".join(missing))
        return 2

    print(f"StructViz-Bench v2 analysis, {mf['name']}  (seed {SEED}, B_boot={B_BOOT}, "
          f"B_perm={B_PERM}{', SMOKE' if a.smoke else ''})")
    print(f"questions {len(meta)}  objects {len(set(q2d.values()))}")

    # ---- manifest (sizes + renderer constants)
    man = {}
    if os.path.exists(a.manifest):
        for r in read_jsonl(a.manifest):
            man[(r["question_id"], r["viz_type"])] = r
    cman = man
    if os.path.abspath(a.manifest) != os.path.abspath(const_manifest) and os.path.exists(
            const_manifest):
        cman = {(r["question_id"], r["viz_type"]): r for r in read_jsonl(const_manifest)}
    const_by_q = {}
    const_src = Counter()
    for q in qs_by_mod["tabular"]:
        c = dict(V2_DEFAULTS)
        rc = {}
        for f in ("bar_chart", "scatter_plot"):
            rc.update(cman.get((q, f), {}).get("renderer_constants") or {})
        for k in c:
            if k in rc:
                c[k] = rc[k]
        const_src[tuple(sorted((k, str(c[k])) for k in c))] += 1
        const_by_q[q] = c

    if a.dump_fully_answerable:
        # Same rule as s1_categories' "fully answerable questions v2" count.
        fa = []
        for md in MODS:
            qs = [q for q in qs_by_mod[md]
                  if all(c in "VD" for c in (
                      classify_v2(meta[q], const_by_q.get(q, V2_DEFAULTS))[f]
                      for f in FORMATS[md]))]
            print(f"fully answerable questions v2 {md}: {len(qs)}/{len(qs_by_mod[md])}")
            fa.extend(qs)
        with open(a.dump_fully_answerable, "w") as fh:
            fh.write("\n".join(fa) + "\n")
        print(f"wrote {len(fa)} question ids -> {a.dump_fully_answerable}")
        return 0

    # ---- predictions
    raw1 = read_jsonl(a.v1)
    v1rows, st1 = dedupe(raw1)
    if st1["keys_error_only"]:
        for r in raw1:
            k = (r["question_id"], r["viz_type"])
            if k not in v1rows:
                v1rows[k] = r  # scored exact_match 0, as in the released per-format EM
        print(f"  NOTE v1: kept {st1['keys_error_only']} error-only keys as EM 0 (released "
              f"scoring); questions affected "
              f"{len({q for (q, f), r in v1rows.items() if is_error(r)})}")
    raw2 = []
    for p in shard_paths:
        raw2.extend(read_jsonl(p))
    v2rows, st2 = dedupe(raw2)
    v2rows = {k: v for k, v in v2rows.items() if k[0] in meta}
    noimg_rows, st0 = dedupe(read_jsonl(noimg_path)) if os.path.exists(noimg_path) else ({}, {})
    assist_rows, sta = (dedupe(read_jsonl(assist_path)) if os.path.exists(assist_path)
                        else ({}, {}))
    EM1, P1 = em_table(v1rows)
    EM2, P2 = em_table(v2rows)
    EM0 = {q: float(r.get("exact_match", 0) or 0) for (q, f), r in noimg_rows.items()}
    EMa, _ = em_table({k: v for k, v in assist_rows.items() if k[1] in ASSIST})

    exp_v2 = Counter(f for (_, f) in man)
    got_v2 = Counter(f for (_, f) in v2rows)
    print(f"v2 shard files: {len(shard_paths)} ({', '.join(os.path.basename(p) for p in shard_paths)})")
    print(f"dedupe v1 {st1}\n       v2 {st2}\n       noimage {st0}\n       assist {sta}")
    print("v2 coverage per format (rows kept / manifest rows): "
          + ", ".join(f"{f} {got_v2[f]}/{exp_v2[f]}" for f in sorted(exp_v2)))
    unknown = sorted(set(got_v2) - set(exp_v2))
    if unknown:
        print(f"  WARN v2 viz types not in manifest: {unknown}")
    v2q = {q for (q, f) in v2rows if f in FORMATS.get(meta[q]["modality"], [])}
    subset = bool(v2q) and len(v2q) < len(meta)
    if subset:
        full_n = {md: len(qs_by_mod[md]) for md in MODS}
        qs_by_mod = {md: [q for q in qs_by_mod[md] if q in v2q] for md in MODS}
        print("v2 covers a SUBSET of questions; every section (incl. v1 comparisons and "
              "all-answerable subsets) is restricted to it: "
              + ", ".join(f"{md} {len(qs_by_mod[md])}/{full_n[md]}" for md in MODS))
    print("tabular renderer constants used (question counts): "
          + "; ".join(f"{dict(k)} x{v}" for k, v in const_src.items()))

    # ---- categories
    cat1, cat2 = {}, {}
    for q, r in meta.items():
        cat1[q] = ans.CLASSIFY[r["modality"]](r)
        cat2[q] = classify_v2(r, const_by_q.get(q, V2_DEFAULTS))
    cat2_default = cat2
    if a.strict_answerable:
        cat2 = {q: strictify_v2(meta[q], c) for q, c in cat2_default.items()}
    nc = load_near_constant() if a.exclude_near_constant else None
    print("\n[v2 rule branches] (question counts)")
    for k in sorted(BRANCH):
        if k.startswith("v2") or (extended and k.startswith("strict")):
            print(f"  {BRANCH[k]:5d}  {k}")
    if extended:
        print(f"flags: strict_answerable={a.strict_answerable} "
              f"exclude_near_constant={a.exclude_near_constant}")
        if nc is not None:
            print("near-constant cells excluded: " + ", ".join(f"{m}/{t}" for m, t in sorted(nc)))

    res = {"config": {"seed": SEED, "B_boot": B_BOOT, "B_perm": B_PERM, "smoke": a.smoke,
                      "v2_shards": [os.path.relpath(p, ROOT) if p.startswith(ROOT) else p
                                    for p in shard_paths],
                      "dedupe": {"v1": st1, "v2": st2, "noimage": st0, "assist": sta},
                      "v2_coverage": {f: [got_v2[f], exp_v2[f]] for f in exp_v2},
                      "v2_branches": {k: v for k, v in BRANCH.items() if k.startswith("v2")}}}
    if a.model != "qwen":
        res["config"].update(model=mf["name"], v1_file=os.path.relpath(a.v1, ROOT),
                             subset=subset, n_questions_analysed={md: len(qs_by_mod[md])
                                                                  for md in MODS})
    res["1_categories"] = s1_categories(meta, cat1, cat2, qs_by_mod)
    res["2_per_format"] = s2_per_format(EM1, EM2, qs_by_mod, q2d)
    if extended:
        res["config"].update(strict_answerable=a.strict_answerable,
                             exclude_near_constant=a.exclude_near_constant,
                             near_constant_cells=sorted(f"{m}|{t}" for m, t in nc) if nc
                             else None)
        section("1b. All-answerable question counts (every main format V or D)")
        labs = {"default": cat2_default}
        if a.strict_answerable:
            labs["strict"] = cat2
        res["1b_all_answerable_counts"] = fa_counts(meta, labs, qs_by_mod,
                                                    nc or load_near_constant())
    res["3_gaps"] = s3_gaps(EM1, EM2, cat1, cat2, qs_by_mod, q2d, nc=nc, meta=meta)
    if extended:
        EMt1 = tol_table(v1rows, meta)
        EMt2 = tol_table(v2rows, meta)
        res["3b_identification"] = s3b_identification(
            EM1, EM2, EMt1, EMt2, cat2, meta, qs_by_mod, q2d, nc or load_near_constant(),
            "strict" if a.strict_answerable else "default")
    res["4_flip_cr"] = s4_flip(EM1, P1, EM2, P2, qs_by_mod, q2d, nc=nc, meta=meta)
    res["5_noimage"] = s5_noimage(EM0, EM2, meta, qs_by_mod, q2d)
    if extended:
        res["5b_cv_prior"] = s5b_cv_prior(EM2, qs_by_mod, q2d, meta=meta, subset=subset)
    res["6_assist"] = s6_assist(EMa, meta, q2d)
    res["7_degree"] = s7_degree(EM2, meta, qs_by_mod, q2d)
    res["8_cost"] = s8_cost(v2rows, man, noimg_rows)
    if a.model != "qwen":
        res["8b_model_input"] = s8b_model_input(v2rows)
    if extended:
        res["9_rule_selector"] = s9_rule_selector(q2d, meta, nc or load_near_constant())

    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"\nwrote {os.path.relpath(out_path, ROOT) if out_path.startswith(ROOT) else out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
