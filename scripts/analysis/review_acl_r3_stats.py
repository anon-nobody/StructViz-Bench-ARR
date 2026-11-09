#!/usr/bin/env python3
"""ACL 2027 mock review, round 3: CPU-only statistical checks on released predictions.

    python scripts/analysis/review_acl_r3_stats.py

No inference; reads the released prediction files and reuses the paper's loaders and scorers
(review_common, scoring.canonical_em via canonical_rescore / v2_multimodel, v2_analysis gap and
cluster helpers, chartqa_score relaxed accuracy). Conventions: canonical_em, cluster = data_id
(ChartQA: chart id), B_boot = 5000 percentile 95% CIs, every resampling stream a fresh
random.Random(0) (numpy only sums the drawn indices).

Sections (keys of review_acl_r3_stats.json)
  s1_selection   Selection-corrected best-worst gaps. "naive" reproduces the paper (best and
                 worst chosen on the full sample, held fixed in the bootstrap). "split_half":
                 2000 random object splits; best/worst chosen on half A and measured on half B
                 and vice versa. Per direction we report the held-out gap; "crossfit" is the
                 question-weighted mean of both directions (every question is scored with the
                 formats chosen on the other half). Percentiles are over splits (split noise,
                 not sampling noise). "crossfit_boot": per resample, a random object split and
                 an object bootstrap inside each half, then the cross-fitted gap (sampling CI
                 for the cross-fitted estimator). "reselect_boot": object bootstrap that
                 re-selects best and worst inside every resample (max-statistic CI of
                 max_f EM_f - min_f EM_f); "bias_corrected" = 2 * naive - mean(resamples).
                 Cells: (a) v1 default suite, four core models x 3 modalities, all questions;
                 (b) v2 expressive suite, the six models of Table tab:v2, tabular and graph on
                 all_answerable_exclNC (v2_multimodel definitions, strict v2 audit, canonical);
                 (c) ChartQA, 616 questions without visual reference, relaxed accuracy, the
                 five re-renderings (Gap) and the four without the scatter plot (Gap4).
  s2_decomposition  v1 gap by audit class. gap = mean_q [c_b(q) - c_w(q)] with the paper's
                 b/w; questions are partitioned by (class of (q, w), class of (q, b)) with the
                 default v1 audit (review_common.v1_cats, non-strict, as canonical_rescore):
                 "w_A_b_VD" (worst absent, best answerable), "both_VD", "both_A", and
                 "other" (any constant pair, or best absent while worst answerable). Share =
                 class sum / total gap sum, object-cluster bootstrap CI (ratio of sums). Per model
                 also pooled over the three modalities (each with its own b/w), and the range
                 over the seven models. The full 3x3 table of (w class, b class) is included.
  s3_template_exclusion  Gaps after dropping templates whose key and wording may disagree
                 (sec_appendix.tex / appendix human check): "core" = tabular counterfactual
                 (ordering-unchanged key vs "remain above" wording), timeseries forecasting
                 (naive x_n + (x_n - x_{n-1}) key for "next expected value"), timeseries
                 change_point (generator change index / "none" for "significant change"),
                 timeseries threshold_count (population SD for "mean plus one standard
                 deviation"); "extended" = core + timeseries volatility (population SD of
                 the halves keys "more volatile") + every other counterfactual template
                 (graph counterfactual, timeseries counterfactual_half_shift /
                 counterfactual_scale_peak / counterfactual_sign_flip). v2: residual tabular /
                 graph gaps and the series three-format gap (definitions as v2_multimodel);
                 v1: every model's gaps on all remaining questions, b/w re-chosen, plus the max.
  s4_graph_sensitivity  v2 graph residual gaps (a) without circular_layout in the format set,
                 (b) without ring-lattice objects, (c) both. Ring lattices are the
                 Watts-Strogatz objects (metadata.topology_type watts_strogatz / small_world);
                 the structural check is the share of edges joining nodes at most k/2 apart on
                 the circle (node order = circular_layout order, k = round(2m/n)). A broader
                 variant drops every object with that share >= 0.5 (adds caveman, grid and the
                 dense Davis projection).
  s5_selector_v2  The paper's lookup selector (canonical_extras.selector_direct /
                 v2_analysis._selector_eval: md5(data_id) parity split, best format per
                 (modality, task) on train with per-modality fallback, a missing chosen format
                 scores the question's mean over formats) on v2 main-format predictions of
                 Qwen2.5-VL-7B / -32B / Qwen3-VL-8B, all questions. Baselines: random format,
                 best fixed format per modality (train), simple rule (table_image for tables,
                 text_only for series and graphs), the paper's hand rule; oracle. Differences
                 with object-cluster CIs over held-out objects. "v1_check" reruns the same code
                 on the v1 files and must reproduce Table tab:selector.

Writes scripts/analysis/review_acl_r3_stats.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import canonical_rescore as cr  # noqa: E402
import chartqa_score as cq  # noqa: E402
import review_common as rc  # noqa: E402
import v2_analysis as v2a  # noqa: E402
import v2_multimodel as mm  # noqa: E402

OUT = os.path.join(HERE, "review_acl_r3_stats.json")
MODS = rc.MODS
FORMATS = rc.FORMATS
B_BOOT = 5000
N_SPLITS = 2000
V2_MODELS = [s for s in mm.MODELS if s["name"] != "InternVL2.5-8B (1 tile)"]
SEL_MODELS = ["Qwen2.5-VL-7B", "Qwen2.5-VL-32B", "Qwen3-VL-8B"]
PAPER = {
    "v1_max_gap": 39.5,
    "v2_tabular": {"Qwen2.5-VL-7B": 6.9, "Qwen2.5-VL-32B": 6.1, "Qwen3-VL-8B": 24.7,
                   "InternVL2-4B": 24.3, "InternVL2-8B": 11.4, "InternVL2.5-8B (6 tiles)": 21.1},
    "v2_graph": {"Qwen2.5-VL-7B": 4.9, "Qwen2.5-VL-32B": 13.1, "Qwen3-VL-8B": 17.8,
                 "InternVL2-4B": 1.9, "InternVL2-8B": 4.5, "InternVL2.5-8B (6 tiles)": 5.9},
    "chartqa": {"qwen": {"gap5": 18.8, "gap4": 8.3}, "qwen3vl8b": {"gap5": 23.9, "gap4": 2.6}},
    "selector_v1": {"GPT-4o": 55.5, "Gemini-2.0-Flash": 57.2, "Gemini-2.5-Flash": 54.5,
                    "Qwen2.5-VL-7B": 50.1, "Claude Sonnet 4": 45.6, "Qwen2.5-VL-32B": 52.3,
                    "InternVL2.5-8B": 31.9},
}
CORE_MISMATCH = [("tabular", "counterfactual"), ("timeseries", "forecasting"),
                 ("timeseries", "change_point"), ("timeseries", "threshold_count")]
EXTENDED_MISMATCH = CORE_MISMATCH + [
    ("timeseries", "volatility"), ("graph", "counterfactual"),
    ("timeseries", "counterfactual_half_shift"), ("timeseries", "counterfactual_scale_peak"),
    ("timeseries", "counterfactual_sign_flip")]
GRAPH_NO_CIRC = ["node_link", "adjacency_matrix", "text_only"]
RING_TOPOLOGIES = {"watts_strogatz", "small_world"}
GROUPS = ("w_A_b_VD", "both_VD", "both_A", "other")


# ---------------------------------------------------------------- resampling
def pct(xs, lo=2.5, hi=97.5):
    return v2a.pct(list(xs), lo, hi)


def obj_matrix(EM, qs, fmts, q2d):
    """Per-object sums of correctness per format (m x F) and question counts (m)."""
    objs = sorted({q2d[q] for q in qs})
    ix = {o: i for i, o in enumerate(objs)}
    S = np.zeros((len(objs), len(fmts)))
    N = np.zeros(len(objs))
    for q in qs:
        i = ix[q2d[q]]
        S[i] += [EM[q][f] for f in fmts]
        N[i] += 1
    return S, N


def pick(s):
    """(best, worst) indices from per-format sums; first max / first min (as max()/min())."""
    return int(np.argmax(s)), int(np.argmin(s))


def gap_of(s, n):
    b, w = pick(s)
    return 100 * (s[b] - s[w]) / n, b, w


def selection_corrected(EM, qs, fmts, q2d):
    """naive / split_half / crossfit_boot / reselect_boot for one cell (see docstring)."""
    S, N = obj_matrix(EM, qs, fmts, q2d)
    m = len(N)
    tot, ntot = S.sum(0), N.sum()
    naive, b0, w0 = gap_of(tot, ntot)
    fixed = v2a.contrast([(q, EM[q][fmts[b0]] - EM[q][fmts[w0]]) for q in qs], q2d)
    out = {"n_q": int(ntot), "n_obj": m, "formats": fmts,
           "naive": {"est": naive, "lo": fixed["obj_ci"]["lo"], "hi": fixed["obj_ci"]["hi"],
                     "best": fmts[b0], "worst": fmts[w0],
                     "per_format": {f: 100 * tot[i] / ntot for i, f in enumerate(fmts)}}}

    def held_out(SA, SB, nB):
        b, w = pick(SA)
        return 100 * (SB[b] - SB[w]) / nB, (b, w) == (b0, w0)

    # split-half
    rng = random.Random(0)
    order = list(range(m))
    dirs, cross, same = [], [], 0
    for _ in range(N_SPLITS):
        rng.shuffle(order)
        A, B = order[: m // 2], order[m // 2:]
        SA, SB, nA, nB = S[A].sum(0), S[B].sum(0), N[A].sum(), N[B].sum()
        gab, s1 = held_out(SA, SB, nB)
        gba, s2 = held_out(SB, SA, nA)
        dirs += [gab, gba]
        same += s1 + s2
        cross.append((gab * nB + gba * nA) / (nA + nB))
    lo, hi = pct(dirs)
    clo, chi = pct(cross)
    out["split_half"] = {
        "n_splits": N_SPLITS,
        "per_direction": {"mean": float(np.mean(dirs)), "p2.5": lo, "p97.5": hi,
                          "frac_le_0": float(np.mean(np.array(dirs) <= 0))},
        "crossfit": {"mean": float(np.mean(cross)), "p2.5": clo, "p97.5": chi,
                     "frac_le_0": float(np.mean(np.array(cross) <= 0))},
        "frac_same_best_worst_as_full": same / (2 * N_SPLITS)}
    # cross-fitted estimator with a sampling CI
    rng = random.Random(0)
    cb = []
    for _ in range(B_BOOT):
        rng.shuffle(order)
        A, B = order[: m // 2], order[m // 2:]
        Ar = [A[int(rng.random() * len(A))] for _ in A]
        Br = [B[int(rng.random() * len(B))] for _ in B]
        SA, SB, nA, nB = S[Ar].sum(0), S[Br].sum(0), N[Ar].sum(), N[Br].sum()
        gab, _ = held_out(SA, SB, nB)
        gba, _ = held_out(SB, SA, nA)
        cb.append((gab * nB + gba * nA) / (nA + nB))
    lo, hi = pct(cb)
    out["crossfit_boot"] = {"mean": float(np.mean(cb)), "lo": lo, "hi": hi,
                            "frac_le_0": float(np.mean(np.array(cb) <= 0))}
    # re-selecting (max-statistic) bootstrap
    rng = random.Random(0)
    rs = []
    for _ in range(B_BOOT):
        idx = [int(rng.random() * m) for _ in range(m)]
        rs.append(gap_of(S[idx].sum(0), N[idx].sum())[0])
    lo, hi = pct(rs)
    out["reselect_boot"] = {"label": "max-statistic bootstrap: best and worst re-selected "
                                     "inside every object resample",
                            "mean": float(np.mean(rs)), "lo": lo, "hi": hi,
                            "bias_corrected": 2 * naive - float(np.mean(rs))}
    return out


def share_ci(rows, q2d):
    """rows (qid, num, den) -> 100 * sum num / sum den, object bootstrap percentile CI."""
    by = defaultdict(lambda: [0.0, 0.0])
    for q, a, b in rows:
        by[q2d[q]][0] += a
        by[q2d[q]][1] += b
    objs = sorted(by)
    M = np.array([by[o] for o in objs])
    tot = M.sum(0)
    rng = random.Random(0)
    m = len(objs)
    bs = []
    for _ in range(B_BOOT):
        s = M[[int(rng.random() * m) for _ in range(m)]].sum(0)
        if s[1]:
            bs.append(100 * s[0] / s[1])
    lo, hi = pct(bs)
    return {"est": 100 * tot[0] / tot[1] if tot[1] else None, "lo": lo, "hi": hi, "n_obj": m}


def gap_ci(EM, qs, fmts, q2d):
    g = v2a.gap_block(EM, qs, fmts, q2d)
    if not g.get("n_q"):
        return {"n_q": 0}
    return {"est": g["est"], "lo": g["obj_ci"]["lo"], "hi": g["obj_ci"]["hi"],
            "p": g["obj_perm_p"], "n_q": g["n_q"], "n_obj": g["n_obj"], "best": g["best"],
            "worst": g["worst"]}


# ---------------------------------------------------------------- data
def load_v1_all():
    return {m: cr.em_tables(os.path.join("results", fn), cr.canon_v1)[0]
            for m, fn in rc.MODELS7.items()}


def load_v2_all():
    mm.SCORER = cr.canon_raw
    out = {}
    for spec in mm.MODELS:
        out[spec["name"]] = mm.load_em(spec["v2"], False)[0]
    mm.SCORER = None
    return out


def v2_samples(EM2, meta, nc, cat2s, drop=(), fmts=None):
    """v2_multimodel.analyse samples: questions present, outside near-constant cells and
    ``drop`` templates; tabular / graph additionally answerable in every format of ``fmts``."""
    out = {}
    for md in MODS:
        fm = (fmts or {}).get(md, FORMATS[md])
        qs = [q for q, r in meta.items() if r["modality"] == md
              and any(f in EM2.get(q, {}) for f in FORMATS[md])]
        allx = [q for q in qs if (md, meta[q]["task"]) not in nc
                and (md, meta[q]["task"]) not in drop]
        out[md] = {"all_exclNC": allx,
                   "answerable": [q for q in allx if all(cat2s[q][f] in "VD" for f in fm)]}
    return out


def load_chartqa():
    d = Path(ROOT) / "results/review/chartqa"
    items = {r["question_id"]: r for r in cq.read_jsonl(d / "chartqa_items.jsonl")}
    q2d = {q: r["data_id"] for q, r in items.items()}
    out = {}
    for name in ("qwen", "qwen3vl8b"):
        by_key = {}
        for r in cq.read_jsonl(d / f"pred_{name}_chartqa.jsonl"):
            if str(r.get("prediction")) == "[ERROR]" or r.get("error"):
                continue
            if r.get("question_id") in items:
                by_key[(r["question_id"], r["viz_type"])] = r
        fmts = sorted({v for _, v in by_key})
        qids = sorted({q for q, _ in by_key})
        complete = [q for q in qids if all((q, v) in by_key for v in fmts)]
        sc = defaultdict(dict)
        for (q, v), r in by_key.items():
            sc[q][v] = int(cq.relaxed_correct(str(r.get("prediction", "")),
                                              str(items[q]["answer"])))
        no_vis = [q for q in complete if not items[q]["metadata"].get("visual_reference")]
        out[name] = (sc, no_vis)
    return out, q2d


# ---------------------------------------------------------------- section 1
def s1(EM1, EM2, meta, q2d, nc, cat2s, cqa, cq2d):
    res = {"v1_core4": {}, "v2": {}, "chartqa": {}}
    for m in rc.CORE4:
        res["v1_core4"][m] = {}
        for md in MODS:
            qs = [q for q in meta if meta[q]["modality"] == md]
            res["v1_core4"][m][md] = selection_corrected(EM1[m], qs, FORMATS[md], q2d)
    for spec in V2_MODELS:
        m = spec["name"]
        smp = v2_samples(EM2[m], meta, nc, cat2s)
        res["v2"][m] = {}
        for md in ("tabular", "graph"):
            qs = v2a.complete(EM2[m], smp[md]["answerable"], FORMATS[md])
            r = selection_corrected(EM2[m], qs, FORMATS[md], q2d)
            r["paper_naive"] = PAPER[f"v2_{md}"][m]
            r["reproduces_paper"] = round(r["naive"]["est"], 1) == PAPER[f"v2_{md}"][m]
            res["v2"][m][md] = r
    for name, (sc, qs) in cqa.items():
        res["chartqa"][name] = {}
        for lab, fm in (("gap5", cq.RERENDER),
                        ("gap4", [f for f in cq.RERENDER if f != "scatter_plot"])):
            r = selection_corrected(sc, qs, fm, cq2d)
            r["paper_naive"] = PAPER["chartqa"][name][lab]
            r["reproduces_paper"] = round(r["naive"]["est"], 1) == PAPER["chartqa"][name][lab]
            res["chartqa"][name][lab] = r
    mx = max(((m, md, r["naive"]["est"]) for m, d in res["v1_core4"].items()
              for md, r in d.items()), key=lambda t: t[2])
    res["v1_max_gap_core4"] = {"model": mx[0], "modality": mx[1], "naive": mx[2],
                               "paper": PAPER["v1_max_gap"],
                               "reproduces_paper": round(mx[2], 1) == PAPER["v1_max_gap"]}
    return res


# ---------------------------------------------------------------- section 2
def cls(c):
    return "VD" if c in "VD" else c


def s2(EM1, meta, q2d, cat1):
    res = {"audit": "review_common.v1_cats(meta) (non-strict default audit, as "
                    "canonical_rescore.py)", "models": {}}
    pooled_rng = []
    for m in rc.MODELS7:
        EM = EM1[m]
        rec, pool = {}, defaultdict(list)
        for md in MODS:
            qs = [q for q in meta if meta[q]["modality"] == md]
            fm = FORMATS[md]
            pf = {f: rc.mean100(EM[q][f] for q in qs) for f in fm}
            b, w = max(pf, key=pf.get), min(pf, key=pf.get)
            d = {q: EM[q][b] - EM[q][w] for q in qs}
            gsum = sum(d.values())
            grid = defaultdict(lambda: [0, 0.0])
            for q in qs:
                k = f"w_{cls(cat1[q][w])}|b_{cls(cat1[q][b])}"
                grid[k][0] += 1
                grid[k][1] += d[q]

            def group(q):
                cw, cb = cls(cat1[q][w]), cls(cat1[q][b])
                if cw == "A" and cb == "VD":
                    return "w_A_b_VD"
                if cw == "VD" and cb == "VD":
                    return "both_VD"
                if cw == "A" and cb == "A":
                    return "both_A"
                return "other"
            r = {"best": b, "worst": w, "gap": 100 * gsum / len(qs), "n_q": len(qs),
                 "shares": {}, "grid_w_by_b": {k: {"n_q": v[0], "gap_pp_contrib":
                                                   100 * v[1] / len(qs),
                                                   "share": 100 * v[1] / gsum if gsum else None}
                                               for k, v in sorted(grid.items())}}
            for g in GROUPS:
                rows = [(q, d[q] if group(q) == g else 0.0, d[q]) for q in qs]
                r["shares"][g] = share_ci(rows, q2d)
                r["shares"][g]["n_q"] = sum(group(q) == g for q in qs)
                pool[g] += rows
            rec[md] = r
        rec["pooled"] = {g: share_ci(rows, q2d) for g, rows in pool.items()}
        pooled_rng.append(rec["pooled"]["w_A_b_VD"]["est"])
        res["models"][m] = rec
    for g in GROUPS:
        v = [res["models"][m]["pooled"][g]["est"] for m in rc.MODELS7]
        res.setdefault("pooled_range_7models", {})[g] = {"min": min(v), "max": max(v)}
        cells = [res["models"][m][md]["shares"][g]["est"] for m in rc.MODELS7 for md in MODS]
        res.setdefault("cell_range_7models", {})[g] = {"min": min(cells), "max": max(cells)}
    return res


# ---------------------------------------------------------------- section 3
def v2_gap_set(EM2, meta, nc, cat2s, drop=(), fmts=None, drop_objs=frozenset(), q2d=None):
    smp = v2_samples(EM2, meta, nc, cat2s, drop, fmts)
    out = {}
    for md in ("tabular", "graph"):
        fm = (fmts or {}).get(md, FORMATS[md])
        qs = [q for q in smp[md]["answerable"] if q2d[q] not in drop_objs]
        out[md] = gap_ci(EM2, qs, fm, q2d)
    qs = [q for q in smp["timeseries"]["all_exclNC"] if q2d[q] not in drop_objs]
    out["timeseries_nonlossy3"] = gap_ci(EM2, qs, v2a.TS_NONLOSSY, q2d)
    return out


def s3(EM1, EM2, meta, q2d, nc, cat2s):
    cnt = defaultdict(int)
    for b in meta.values():
        cnt[(b["modality"], b["task"])] += 1
    res = {"sets": {}, "v2": {}, "v1": {}}
    for lab, drop in (("none", []), ("core", CORE_MISMATCH), ("extended", EXTENDED_MISMATCH)):
        res["sets"][lab] = {"templates": [f"{a}|{t}" for a, t in drop],
                            "n_questions": {f"{a}|{t}": cnt[(a, t)] for a, t in drop},
                            "already_near_constant_in_v2": [f"{a}|{t}" for a, t in drop
                                                            if (a, t) in nc]}
        ds = set(drop)
        res["v2"][lab] = {s["name"]: v2_gap_set(EM2[s["name"]], meta, nc, cat2s, ds, q2d=q2d)
                          for s in V2_MODELS}
        v1 = {}
        for m in rc.MODELS7:
            v1[m] = {}
            for md in MODS:
                qs = [q for q in meta if meta[q]["modality"] == md
                      and (md, meta[q]["task"]) not in ds]
                v1[m][md] = gap_ci(EM1[m], qs, FORMATS[md], q2d)
        mx4 = max(((m, md, v1[m][md]) for m in rc.CORE4 for md in MODS),
                  key=lambda t: t[2]["est"])
        mx7 = max(((m, md, v1[m][md]) for m in rc.MODELS7 for md in MODS),
                  key=lambda t: t[2]["est"])
        res["v1"][lab] = {"models": v1,
                          "max_core4": {"model": mx4[0], "modality": mx4[1], **mx4[2]},
                          "max_all7": {"model": mx7[0], "modality": mx7[1], **mx7[2]}}
    return res


# ---------------------------------------------------------------- section 4
def ring_objects(meta):
    out = {}
    for b in meta.values():
        if b["modality"] != "graph" or b["data_id"] in out:
            continue
        d = b["data"]
        links = d.get("links") or d.get("edges") or []
        ids = [x["id"] for x in d["nodes"]]
        n, pos = len(ids), {v: i for i, v in enumerate(ids)}
        k = max(2, round(2 * len(links) / n))
        dist = [min(abs(pos[e["source"]] - pos[e["target"]]),
                    n - abs(pos[e["source"]] - pos[e["target"]])) for e in links]
        out[b["data_id"]] = {"topology_type": b["metadata"].get("topology_type"), "n_nodes": n,
                             "n_edges": len(links), "k": k,
                             "ring_edge_share": sum(x <= k // 2 for x in dist) / len(links)}
    return out


def s4(EM2, meta, q2d, nc, cat2s):
    objs = ring_objects(meta)
    ws = frozenset(o for o, v in objs.items() if v["topology_type"] in RING_TOPOLOGIES)
    short = frozenset(o for o, v in objs.items() if v["ring_edge_share"] >= 0.5)
    gq = [q for q in meta if meta[q]["modality"] == "graph"]
    res = {"objects": objs, "definitions": {}, "models": {}}
    for lab, s in (("ring_lattice_ws", ws), ("ring_edge_share_ge_0.5", short)):
        res["definitions"][lab] = {"objects": sorted(s), "n_obj": len(s),
                                   "n_questions": sum(q2d[q] in s for q in gq)}
    nocirc = {"graph": GRAPH_NO_CIRC}
    for spec in V2_MODELS:
        m = spec["name"]
        E = EM2[m]

        def g(fm=None, drop=frozenset()):
            return v2_gap_set(E, meta, nc, cat2s, fmts=fm, drop_objs=drop, q2d=q2d)["graph"]
        res["models"][m] = {"baseline": g(), "no_circular_layout": g(nocirc),
                            "no_ring_lattice_ws": g(drop=ws),
                            "no_ring_lattice_ws_and_no_circular": g(nocirc, ws),
                            "no_ring_edge_share_ge_0.5": g(drop=short),
                            "no_ring_edge_share_ge_0.5_and_no_circular": g(nocirc, short)}
    return res


# ---------------------------------------------------------------- section 5
def selector_eval(byq, tag, q2d, meta):
    def side(q):
        return "train" if int(hashlib.md5(q2d[q].encode()).hexdigest(), 16) % 2 == 0 \
            else "test"
    acc_mt, acc_m = defaultdict(lambda: defaultdict(list)), defaultdict(
        lambda: defaultdict(list))
    for q, fm in byq.items():
        if side(q) == "train":
            for v, e in fm.items():
                acc_mt[tag[q]][v].append(e)
                acc_m[tag[q][0]][v].append(e)
    best_mt = {k: max(d, key=lambda v: sum(d[v]) / len(d[v])) for k, d in acc_mt.items()}
    best_m = {k: max(d, key=lambda v: sum(d[v]) / len(d[v])) for k, d in acc_m.items()}

    def simple(q):
        return "table_image" if tag[q][0] == "tabular" else "text_only"

    def hand(q):  # v2_analysis._selector_eval rule
        md, task = tag[q]
        if md == "tabular":
            avg = "average" in re.sub(r"'[^']*'", "", meta[q]["question"]).lower()
            if task == "comparison" or (task == "aggregation" and avg):
                return "bar_chart"
            return "table_image"
        return "text_only"
    vals, diffs = defaultdict(list), defaultdict(list)
    for q, fm in sorted(byq.items()):
        if side(q) != "test":
            continue
        mean = sum(fm.values()) / len(fm)
        mt = best_mt.get(tag[q]) or best_m.get(tag[q][0])
        s = {"selector": fm.get(mt, mean), "fixed_mod": fm.get(best_m.get(tag[q][0]), mean),
             "simple_rule": fm.get(simple(q), mean), "hand_rule": fm.get(hand(q), mean),
             "random": mean, "oracle": float(max(fm.values()) >= 1)}
        for k, v in s.items():
            vals[k].append(v)
        for k in ("fixed_mod", "simple_rule", "hand_rule", "random"):
            diffs[f"selector_minus_{k}"].append((q, s["selector"] - s[k]))
    out = {"n_test": len(vals["random"]),
           "n_obj_test": len({q2d[q] for q, _ in diffs["selector_minus_random"]}),
           **{k: rc.mean100(v) for k, v in vals.items()}, "fixed_formats_train": best_m,
           "selector_table_train": {f"{a}|{t}": v for (a, t), v in sorted(best_mt.items())}}
    for k, it in diffs.items():
        r = v2a.contrast(it, q2d)
        out[k] = {"est": r["est"], "lo": r["obj_ci"]["lo"], "hi": r["obj_ci"]["hi"],
                  "n": r["n_units"], "n_obj": r["n_obj"]}
    return out


def s5(EM1, EM2, meta, q2d):
    res = {"v2": {}, "v1_check": {}}
    tag = {q: (b["modality"], b["task"]) for q, b in meta.items()}
    for m in SEL_MODELS:
        byq = {q: {f: v for f, v in fm.items() if f in FORMATS[meta[q]["modality"]]}
               for q, fm in EM2[m].items() if q in meta}
        byq = {q: fm for q, fm in byq.items() if fm}
        res["v2"][m] = selector_eval(byq, tag, q2d, meta)
        res["v2"][m]["n_rows"] = sum(len(v) for v in byq.values())
    for m in rc.MODELS7:
        r = selector_eval(EM1[m], tag, q2d, meta)
        res["v1_check"][m] = {"selector": r["selector"], "fixed_mod": r["fixed_mod"],
                              "paper_selector": PAPER["selector_v1"][m],
                              "reproduces_paper": round(r["selector"], 1)
                              == PAPER["selector_v1"][m]}
    return res


# ---------------------------------------------------------------- main
def main():
    meta = rc.load_bench()
    q2d = {q: b["data_id"] for q, b in meta.items()}
    nc = rc.near_constant()
    cat1 = rc.v1_cats(meta)
    cat2s = rc.v2_cats(meta, strict=True)
    EM1 = load_v1_all()
    EM2 = load_v2_all()
    cqa, cq2d = load_chartqa()
    res = {"definition": __doc__}
    print("[1] selection-corrected gaps")
    res["s1_selection"] = s1(EM1, EM2, meta, q2d, nc, cat2s, cqa, cq2d)
    print("[2] gap decomposition")
    res["s2_decomposition"] = s2(EM1, meta, q2d, cat1)
    print("[3] template exclusion")
    res["s3_template_exclusion"] = s3(EM1, EM2, meta, q2d, nc, cat2s)
    print("[4] graph sensitivity")
    res["s4_graph_sensitivity"] = s4(EM2, meta, q2d, nc, cat2s)
    print("[5] selector on v2")
    res["s5_selector_v2"] = s5(EM1, EM2, meta, q2d)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"wrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
