#!/usr/bin/env python3
"""ACL 2027 mock review, round 3: TTA paired effects, sample, generation cap, ties, test
bookkeeping, latency. CPU only; no inference. Reads existing results, writes only
scripts/analysis/review_acl_r3_tta.json.

    python scripts/analysis/review_acl_r3_tta.py

Conventions: canonical_em (scoring.py), cluster = data_id, percentile 95% CIs from the
object-cluster bootstrap (B=5000, random.Random(0)), two-sided object sign-flip p (B=10000)
via v2_analysis.contrast.

TTA rows. results/review/tta_qwen.jsonl was at one point re-merged from three shards only (695
rows; 2026-11-08 00:48, results/review/logs/merge_tta.log "inputs=3"), after
tta_analysis_results.json / tab_tta.tex had been written from the 750-row merge. The 750 rows are
therefore rebuilt here from tta_qwen_shard0..3.jsonl; the merged file, whatever it holds, must
agree row for row, and every stored per-format accuracy of tta_analysis_results.json is asserted
to reproduce.

1. Paired effects on the TTA sample (same 50 questions per modality): raw text (true data as
   text, results/v2/rawtext_qwen.jsonl) minus the best image format (best = highest image
   accuracy on the sample; for graphs also the best drawing, i.e. excluding the listings), and
   own transcript minus image, per format and pooled per modality, with sign-flip p.
2. Sample: unique data objects per modality; the selection rule is re-run
   (build_review_manifests.build_tta) and checked against tta_qids.txt.
3. Generation cap: fidelity, TTA accuracy and image accuracy per format on transcripts that did
   NOT hit the 2,560-token cap; Spearman of fidelity vs image accuracy across the five tabular
   formats (and all 15 cells) on that subset; questions whose answer is recomputable.
4. Ties in argmax / argmin keys, recomputed from benchmark/realworld_test.jsonl with the
   generator's own definitions (src/generation/*_generator.py); a key is tied when another
   candidate attains the same extreme value (floats: math.isclose, rel 1e-9, abs 1e-12).
   v1 (seven models, canonical_rescore.canon_v1 scoring and gap definition) and v2 (three Qwen
   models, v2_multimodel.analyse with canonical_em on the raw response) are rescored accepting
   any tied candidate under canonical_em.
5. Which test produces the p-values of h1_bound.py, and the paper sentences that rely on it.
6. Latency / token fields stored in the Qwen2.5-VL-7B v2, TTA and raw-text result files.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict

import networkx as nx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import canonical_rescore as cr  # noqa: E402
import review_common as rc  # noqa: E402
import scoring as sc  # noqa: E402
import tta_analysis as ta  # noqa: E402
import v2_analysis as v2a  # noqa: E402
import v2_multimodel as mm  # noqa: E402

REV = os.path.join(ROOT, "results", "review")
OUT = os.path.join(HERE, "review_acl_r3_tta.json")
SHARDS = [os.path.join(REV, f"tta_qwen_shard{i}.jsonl") for i in range(4)]
GRAPH_DRAWINGS = ["node_link", "circular_layout", "adjacency_matrix"]
QWEN_V2 = ["Qwen2.5-VL-7B", "Qwen2.5-VL-32B", "Qwen3-VL-8B"]


def ci(items, q2d, perm=True):
    r = v2a.contrast(items, q2d, perm=perm)
    o = {"est": r["est"], "lo": r["obj_ci"]["lo"], "hi": r["obj_ci"]["hi"], "n": r["n_units"],
         "n_obj": r["n_obj"]}
    if perm:
        o["p"] = r["obj_perm_p"]
    return o


def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def summ(vals):
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return {"n": 0}
    return {"mean": statistics.mean(vals), "median": statistics.median(vals), "n": len(vals)}


# ---------------------------------------------------------------- loading
def load_tta(meta):
    rows = {}
    for p in SHARDS:
        for r in rc.read_jsonl(p):
            k = (r["question_id"], r["viz_type"])
            assert k not in rows, k
            rows[k] = r
    merged = {(r["question_id"], r["viz_type"]): r
              for r in rc.read_jsonl(os.path.join(REV, "tta_qwen.jsonl"))}
    assert all(rows[k] == r for k, r in merged.items())
    man = {(r["question_id"], r["viz_type"]) for r in rc.read_jsonl(
        os.path.join(REV, "tta_manifest.jsonl"))}
    assert set(rows) == man and len(rows) == 750
    return rows, {"rows_from_shards": len(rows), "rows_in_merged_file": len(merged),
                  "merged_rows_identical_to_shards": True}


def selection_check(meta, qids):
    """Re-run build_review_manifests.build_tta's sampling rule."""
    pool_q = [q.strip() for q in open(os.path.join(ROOT, "benchmark", "render_v2",
                                                   "fully_answerable_qids.txt")) if q.strip()]
    by = defaultdict(list)
    for q in pool_q:
        by[meta[q]["modality"]].append(q)
    chosen, pools = [], {}
    for md in ("tabular", "timeseries", "graph"):
        pool = sorted(by[md])
        pools[md] = len(pool)
        chosen += random.Random(0).sample(pool, min(50, len(pool)))
    return sorted(chosen) == sorted(qids), pools


# ---------------------------------------------------------------- 4. ties
def _close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def _argset(cands, best):
    """cands: [(answer, value)]; answers whose value ties the extreme ``best``."""
    return sorted({a for a, v in cands if _close(v, best)})


def tie_candidates(b):
    """(template, candidate answers attaining the extreme) or None when not an argmax key."""
    md, task, q, d = b["modality"], b["task"], b["question"], b["data"]
    quoted = re.findall(r"'([^']*)'", q)
    if md == "tabular" and task == "ranking":
        lab, val = quoted[0], quoted[1]
        cands = [(str(r[lab]), fnum(r[val])) for r in d]
        cands = [c for c in cands if c[1] is not None]
        lowest = "lowest" in q
        best = (min if lowest else max)(v for _, v in cands)
        return f"tabular|ranking_{'lowest' if lowest else 'highest'}", _argset(cands, best)
    if md == "tabular" and task == "outlier_detection":
        lab, val = quoted[0], quoted[1]
        vals = [(str(r[lab]), fnum(r[val])) for r in d]
        vals = [c for c in vals if c[1] is not None]
        mu = float(np.mean([v for _, v in vals]))
        dev = [(a, abs(v - mu)) for a, v in vals]
        return "tabular|outlier_detection", _argset(dev, max(x for _, x in dev))
    if md == "tabular" and task == "comparison" and q.startswith("Which column has the higher"):
        ma = float(np.mean([fnum(r[quoted[0]]) for r in d if fnum(r[quoted[0]]) is not None]))
        mb = float(np.mean([fnum(r[quoted[1]]) for r in d if fnum(r[quoted[1]]) is not None]))
        cands = [(quoted[0], ma), (quoted[1], mb)]
        return "tabular|comparison_higher_average", _argset(cands, max(ma, mb))
    if md == "graph" and task in ("degree_query", "centrality"):
        if task == "degree_query" and "highest degree" not in q:
            return None
        g = nx.Graph()
        g.add_nodes_from(int(n["id"]) for n in d["nodes"])
        g.add_edges_from((int(e["source"]), int(e["target"])) for e in d["edges"])
        if task == "degree_query":
            sc_ = {n: float(g.degree(n)) for n in g.nodes}
            tpl = "graph|degree_query_highest_degree"
        else:
            sc_ = nx.betweenness_centrality(g, normalized=True)
            tpl = "graph|centrality_highest_betweenness"
        cands = [(str(n), v) for n, v in sc_.items()]
        return tpl, _argset(cands, max(sc_.values()))
    if md == "timeseries" and task == "peak_identification":
        vals = [float(v) for v in d]
        mx = "maximum" in q
        best = max(vals) if mx else min(vals)
        return (f"timeseries|peak_identification_{'max' if mx else 'min'}",
                sorted({str(i) for i, v in enumerate(vals) if v == best}, key=int))
    if md == "timeseries" and task == "anomaly_detection":
        vals = [float(v) for v in d]
        mu = statistics.mean(vals)
        dev = [(str(i), abs(v - mu)) for i, v in enumerate(vals)]
        return "timeseries|anomaly_detection", sorted(_argset(dev, max(x for _, x in dev)),
                                                      key=int)
    if md == "timeseries" and task in ("volatility", "change_volatility",
                                       "half_change_intensity"):
        s = np.asarray([float(v) for v in d])
        h = len(s) // 2
        if task == "volatility":
            a, c = float(np.std(s[:h])), float(np.std(s[h:]))
        else:
            a = float(np.mean(np.abs(np.diff(s[:h])))) if h > 1 else 0.0
            c = float(np.mean(np.abs(np.diff(s[h:])))) if len(s) - h > 1 else 0.0
        cands = [("first half", a), ("second half", c)]
        out = _argset(cands, max(a, c))
        if task == "volatility" and abs(a - c) < 1e-9:
            out = sorted(set(out) | {"equal"})
        return f"timeseries|{task}", out
    return None


def build_ties(meta):
    per_tpl = defaultdict(lambda: {"n": 0, "key_in_argmax_set": 0, "tied": 0,
                                   "tied_size": Counter(), "examples": []})
    alt = {}
    for q in sorted(meta):
        b = meta[q]
        r = tie_candidates(b)
        if r is None:
            continue
        tpl, cands = r
        s = per_tpl[tpl]
        s["n"] += 1
        key_ok = any(sc.fold(c) == sc.fold(b["answer"]) or (
            sc.is_numeric_key(c) and sc.is_numeric_key(b["answer"])
            and float(c) == float(b["answer"])) for c in cands)
        s["key_in_argmax_set"] += key_ok
        if len(cands) > 1 and key_ok:
            s["tied"] += 1
            s["tied_size"][len(cands)] += 1
            others = [c for c in cands if sc.fold(c) != sc.fold(b["answer"])]
            alt[q] = others
            if len(s["examples"]) < 3:
                s["examples"].append({"question_id": q, "question": b["question"],
                                      "key": b["answer"], "tied_candidates": cands})
    for s in per_tpl.values():
        s["tied_size"] = {str(k): v for k, v in sorted(s["tied_size"].items())}
    return dict(sorted(per_tpl.items())), alt


def with_ties(base, alt):
    """Wrap a row scorer: also accept any tied candidate key under canonical_em."""
    def f(r, raw=None):
        s = base(r)
        q = r["question_id"]
        if s >= 1 or q not in alt:
            return s
        for a in alt[q]:
            if sc.canonical_em(r.get("prediction", ""), raw(r) if raw else None, a,
                               r.get("task")) >= 1:
                return 1.0
        return s
    return f


def v1_raw(r):
    raw = r.get("prediction_raw")
    return None if raw is None or str(raw) == str(r.get("prediction")) else raw


def v2_raw(r):
    raw = r.get("raw_response")
    return None if raw in (None, "None", "") else raw


def v1_rescore(meta, q2d, alt):
    tie_v1 = with_ties(cr.canon_v1, alt)
    out = {}
    for lab, scorer in (("canonical", cr.canon_v1), ("tie_aware", lambda r: tie_v1(r, v1_raw))):
        res = {"models": {}}
        for m, fn in rc.MODELS7.items():
            EM, _ = cr.em_tables(os.path.join("results", fn), scorer)
            rec = {"gaps": {}}
            vals = [EM[q][f] for q in meta for f in rc.FORMATS[meta[q]["modality"]]
                    if f in EM.get(q, {})]
            rec["overall_acc"] = rc.mean100(vals)
            rec["n_pairs"] = len(vals)
            for md in rc.MODS:
                qs = [q for q in meta if meta[q]["modality"] == md]
                fm = rc.FORMATS[md]
                pf = {f: rc.mean100(EM[q][f] for q in qs) for f in fm}
                b, w = max(pf, key=pf.get), min(pf, key=pf.get)
                g = ci([(q, EM[q][b] - EM[q][w]) for q in qs], q2d, perm=False)
                g.update(best=b, worst=w)
                rec["gaps"][md] = g
            res["models"][m] = rec
        allg = [(m, md, r["gaps"][md]["est"]) for m, r in res["models"].items() for md in rc.MODS]
        res["max_gap_all7"] = list(max(allg, key=lambda t: t[2]))
        core = [t for t in allg if t[0] in rc.CORE4]
        res["max_gap_core4"] = list(max(core, key=lambda t: t[2]))
        out[lab] = res
    return out


def v2_rescore(meta, q2d, alt):
    cat2s = rc.v2_cats(meta, strict=True)
    nc = rc.near_constant()
    pr = v2a.load_prior()
    prior_const = {md: pr[md]["cv_mean"] for md in rc.MODS}
    prior_q = rc.cv_prior_per_question(meta)
    tie_v2 = with_ties(sc.row_score, alt)
    out = {}
    specs = [s for s in mm.MODELS if s["name"] in QWEN_V2]
    for lab, scorer in (("canonical", sc.row_score), ("tie_aware", lambda r: tie_v2(r, v2_raw))):
        mm.SCORER = scorer
        res = {}
        for spec in specs:
            o = mm.analyse(spec, meta, q2d, cat2s, nc, prior_const, prior_q)
            o.pop("_present", None)
            em, _ = mm.load_em(spec["v2"], False)
            vals = [em[q][f] for q in em for f in rc.FORMATS[meta[q]["modality"]] if f in em[q]]
            g = o["gaps"]
            res[spec["name"]] = {
                "overall_acc": rc.mean100(vals), "n_pairs": len(vals),
                "tabular_residual_gap": g["tabular"]["all_answerable_exclNC"],
                "graph_residual_gap": g["graph"]["all_answerable_exclNC"],
                "series_nonlossy3_gap": g["timeseries"]["nonlossy3_all_exclNC"]}
        out[lab] = res
    mm.SCORER = None
    return out


# ---------------------------------------------------------------- 5. h1 bookkeeping
TEX_PATTERNS = [r"sign-flip", r"Holm \$p", r"at or below the question-only prior",
                r"significantly above", r"above the question-only prior"]


def tex_hits():
    d = os.path.join(ROOT, "acl2027")
    out = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".tex"):
            continue
        for i, ln in enumerate(open(os.path.join(d, fn), encoding="utf-8"), 1):
            for p in TEX_PATTERNS:
                if re.search(p, ln):
                    out.append({"file": f"acl2027/{fn}", "line": i, "pattern": p,
                                "text": ln.strip()})
                    break
    return out


def h1_section():
    h = json.load(open(os.path.join(HERE, "h1_bound_results.json")))
    above = [{k: c[k] for k in ("suite", "model", "modality", "n", "diff", "lo", "hi",
                                "p_one_sided", "p_holm")}
             for c in h["cells"] if c.get("n") and c["p_one_sided"] < 0.05]
    return {
        "test_in_h1_bound_py": (
            "h1_bound.boot: object-cluster bootstrap of mean(EM - CV prior) over absent (A) "
            "pairs, B=10000, random.Random(0); 95% CI = percentile; p = ONE-SIDED bootstrap "
            "p = (#{d* <= 0} + 1) / (B + 1); Holm step-down over the 31 tested cells. "
            "No sign-flip permutation test is run in h1_bound.py."),
        "n_cells_tested": h["n_cells_tested"], "n_above_raw_p05": h["n_above_raw"],
        "n_above_holm_p05": h["n_above_holm"], "cells_above_raw": above,
        "tex_hits": tex_hits()}


# ---------------------------------------------------------------- 6. latency
def lat_block(rows, fields):
    out = {}
    for f in fields:
        vals = [fnum(r.get(f)) for r in rows]
        out[f] = summ(vals)
    return out


def latency_section(meta, tta):
    plain, _ = v2a.dedupe(rc.read_jsonl(os.path.join(ROOT, "results", "v2", "v2_qwen.jsonl")))
    fields_v2 = sorted({k for r in plain.values() for k in r})
    v2_fmt = {}
    for md, fmts in ta.FMTS.items():
        for f in fmts:
            rows = [r for (q, ff), r in plain.items() if ff == f and meta[q]["modality"] == md]
            v2_fmt[f"{md}|{f}"] = lat_block(rows, ["latency_s", "n_visual_tokens"])
    tta_fmt, direct_same = {}, {}
    for md, fmts in ta.FMTS.items():
        for f in fmts:
            ks = [k for k in sorted(tta) if k[1] == f and meta[k[0]]["modality"] == md]
            tta_fmt[f"{md}|{f}"] = lat_block([tta[k] for k in ks], [
                "latency_s", "latency_stage1_s", "latency_stage2_s", "transcript_n_tokens"])
            direct_same[f"{md}|{f}"] = lat_block([plain[k] for k in ks], ["latency_s"])
    raw = [r for r in rc.read_jsonl(os.path.join(ROOT, "results", "v2", "rawtext_qwen.jsonl"))
           if str(r.get("prediction")) != "[ERROR]"]
    raw_md = {md: lat_block([r for r in raw if r["modality"] == md],
                            ["latency_s", "n_prompt_tokens"]) for md in rc.MODS}
    tta_all = lat_block(list(tta.values()), ["latency_s", "latency_stage1_s",
                                             "latency_stage2_s", "transcript_n_tokens"])
    direct_all = lat_block([plain[k] for k in tta], ["latency_s"])
    tok_fields = [k for k in fields_v2 if "token" in k]
    return {
        "fields_v2_qwen": fields_v2,
        "generated_tokens_logged": {
            "v2_direct_answer": False, "tta_stage1_transcript": True,
            "tta_stage2_answer": False, "rawtext": False,
            "note": "v2 rows store n_visual_tokens (input image tokens) only; " + str(tok_fields)},
        "v2_qwen7b_direct_by_format": v2_fmt,
        "tta_by_format": tta_fmt, "direct_on_tta_rows_by_format": direct_same,
        "tta_all": tta_all, "direct_on_tta_rows_all": direct_all,
        "rawtext_by_modality": raw_md,
        "caveat": ("wall-clock seconds per query on shared GPUs; the v2 and TTA runs were on "
                   "different dates and GPUs (TTA shards on GPUs 0-3, some shared), so "
                   "latencies compare formats within a run, not runs.")}


# ---------------------------------------------------------------- main
def main():
    meta = rc.load_bench()
    q2d = {q: b["data_id"] for q, b in meta.items()}
    tta, load_info = load_tta(meta)
    qids = [q.strip() for q in open(os.path.join(REV, "tta_qids.txt")) if q.strip()]
    plain, _ = v2a.dedupe(rc.read_jsonl(os.path.join(ROOT, "results", "v2", "v2_qwen.jsonl")))
    raw = {r["question_id"]: r for r in rc.read_jsonl(os.path.join(ROOT, "results", "v2",
                                                                    "rawtext_qwen.jsonl"))
           if str(r.get("prediction")) != "[ERROR]"}
    keys = sorted(tta)
    A = {k: sc.row_score(plain[k]) for k in keys}
    B = {k: sc.row_score(tta[k]) for k in keys}
    C = {k: sc.row_score(raw[k[0]]) for k in keys}
    stored = json.load(open(os.path.join(HERE, "tta_analysis_results.json")))
    for md, fmts in ta.FMTS.items():
        for f in fmts + ["ALL"]:
            ks = [k for k in keys if meta[k[0]]["modality"] == md and (f == "ALL" or k[1] == f)]
            s = stored["accuracy"][f"{md}|{f}"]
            assert abs(rc.mean100(A[k] for k in ks) - s["image"]) < 1e-9
            assert abs(rc.mean100(B[k] for k in ks) - s["tta"]) < 1e-9
            assert abs(rc.mean100(C[k] for k in ks) - s["rawtext"]) < 1e-9
    load_info["reproduces_tta_analysis_results_accuracy"] = True
    res = {"definition": __doc__, "tta_rows": load_info}

    # ---- 1. paired effects
    pe = {}
    for md, fmts in ta.FMTS.items():
        qs = [q for q in qids if meta[q]["modality"] == md]
        img = {f: rc.mean100(A[(q, f)] for q in qs) for f in fmts}
        best = max(fmts, key=lambda f: (img[f], -fmts.index(f)))
        rec = {"image_acc_by_format": img, "rawtext_acc": rc.mean100(C[(q, fmts[0])] for q in qs),
               "best_image_format": best,
               "raw_minus_best_image": ci([(q, C[(q, best)] - A[(q, best)]) for q in qs], q2d)}
        if md == "graph":
            bd = max(GRAPH_DRAWINGS, key=lambda f: (img[f], -fmts.index(f)))
            rec["best_drawing"] = bd
            rec["raw_minus_best_drawing"] = ci([(q, C[(q, bd)] - A[(q, bd)]) for q in qs], q2d)
            m4 = fmts[:4]
            rec["raw_minus_mean_main4_image"] = ci(
                [(q, C[(q, m4[0])] - sum(A[(q, f)] for f in m4) / 4) for q in qs], q2d)
        rec["tta_minus_image_pooled"] = ci([(k[0], B[k] - A[k]) for k in keys
                                            if meta[k[0]]["modality"] == md], q2d)
        rec["tta_minus_image_by_format"] = {f: ci([(q, B[(q, f)] - A[(q, f)]) for q in qs], q2d)
                                            for f in fmts}
        rec["raw_minus_tta_pooled"] = ci([(k[0], C[k] - B[k]) for k in keys
                                          if meta[k[0]]["modality"] == md], q2d)
        pe[md] = rec
    res["paired_effects"] = pe

    # ---- 2. sample
    same, pools = selection_check(meta, qids)
    res["sample"] = {
        "n_questions": len(qids),
        "unique_objects": {md: len({q2d[q] for q in qids if meta[q]["modality"] == md})
                           for md in rc.MODS},
        "questions_per_object_max": {md: max(Counter(q2d[q] for q in qids
                                                     if meta[q]["modality"] == md).values())
                                     for md in rc.MODS},
        "pool_size_fully_answerable": pools, "selection_reproduced": same,
        "rule": ("scripts/build_review_manifests.py build_tta: from the questions answerable in "
                 "every expressive format (benchmark/render_v2/fully_answerable_qids.txt), 50 "
                 "per modality drawn uniformly without replacement with "
                 "random.Random(0).sample(sorted(pool), 50); not stratified by template or "
                 "object."),
        "tasks": dict(sorted(Counter(f"{meta[q]['modality']}|{meta[q]['task']}"
                                     for q in qids).items()))}

    # ---- 3. generation cap
    fid, trunc = {}, {}
    for k in keys:
        b, r = meta[k[0]], tta[k]
        t = r.get("transcript") or ""
        fn = {"tabular": ta.table_fidelity, "timeseries": ta.series_fidelity,
              "graph": ta.graph_fidelity}[b["modality"]]
        fid[k] = fn(t, b["data"])[0][ta.FID_KEY[b["modality"]]]
        trunc[k] = r.get("transcript_truncated") in (True, "True")
    cap = {"n_truncated": sum(trunc.values()), "by_format": {}}
    cells_full, cells_nt = [], []
    for md, fmts in ta.FMTS.items():
        for f in fmts:
            ks = [k for k in keys if k[1] == f and meta[k[0]]["modality"] == md]
            nt = [k for k in ks if not trunc[k]]
            e = {"n": len(ks), "n_not_truncated": len(nt),
                 "truncated_pct": 100 * (len(ks) - len(nt)) / len(ks),
                 "fidelity_all": statistics.mean(fid[k] for k in ks),
                 "fidelity_not_truncated": statistics.mean(fid[k] for k in nt) if nt else None,
                 "fidelity_truncated": statistics.mean(fid[k] for k in ks if trunc[k])
                 if len(nt) < len(ks) else None,
                 "tta_acc_all": rc.mean100(B[k] for k in ks),
                 "tta_acc_not_truncated": rc.mean100(B[k] for k in nt) if nt else None,
                 "image_acc_all": rc.mean100(A[k] for k in ks),
                 "image_acc_not_truncated_rows": rc.mean100(A[k] for k in nt) if nt else None}
            cap["by_format"][f"{md}|{f}"] = e
            cells_full.append((md, f, e["fidelity_all"], e["image_acc_all"]))
            if nt:
                cells_nt.append((md, f, e["fidelity_not_truncated"], e["image_acc_all"],
                                 e["image_acc_not_truncated_rows"]))
    sp = {}
    tab_nt = [c for c in cells_nt if c[0] == "tabular"]
    for lab, cs in (("tabular5", tab_nt), ("all_cells", cells_nt)):
        rho, p = rc.spearman([c[2] for c in cs], [c[3] for c in cs])
        rho2, p2 = rc.spearman([c[2] for c in cs], [c[4] for c in cs])
        sp[lab] = {"n_cells": len(cs),
                   "fidelity_nt_vs_image_full": {"rho": rho, "p": p},
                   "fidelity_nt_vs_image_on_nt_rows": {"rho": rho2, "p": p2},
                   "order_fidelity_nt": [c[1] for c in sorted(cs, key=lambda c: -c[2])],
                   "order_image_full": [c[1] for c in sorted(cs, key=lambda c: -c[3])],
                   "order_image_nt_rows": [c[1] for c in sorted(cs, key=lambda c: -c[4])]}
    tf = [c for c in cells_full if c[0] == "tabular"]
    sp["tabular5_full_sample_check"] = dict(zip(("rho", "p"), rc.spearman(
        [c[2] for c in tf], [c[3] for c in tf])))
    cap["spearman"] = sp
    # recomputable answers
    ok_q = [q for q in qids if ta.implied(meta[q]) != "n/a"
            and ta.answer_ok(ta.implied(meta[q]), meta[q]["answer"])]
    defined = [q for q in qids if ta.implied(meta[q]) != "n/a"]
    comp = Counter()
    nt_comp = Counter()
    src_all, src_nt = defaultdict(Counter), defaultdict(Counter)
    for q in ok_q:
        b = meta[q]
        fmts = ta.FMTS[b["modality"]]
        n_c = n_nt = 0
        for f in fmts:
            k = (q, f)
            imp = ta.implied(b, tta[k].get("transcript") or "")
            computable = imp is not None and imp != []
            n_c += computable
            n_nt += computable and not trunc[k]
            t_ok = ta.answer_ok(imp, b["answer"])
            if B[k] < 1:
                lab = "transcript_right" if t_ok else "transcript_wrong"
                src_all[b["modality"]][lab] += 1
                if not trunc[k]:
                    src_nt[b["modality"]][lab] += 1
        comp["any_format"] += n_c > 0
        comp["all_formats"] += n_c == len(fmts)
        nt_comp["any_format"] += n_nt > 0
        comp[f"n_formats_{n_c}"] += 1

    def share(c):
        n = c["transcript_right"] + c["transcript_wrong"]
        return {"wrong_answers": n, "transcript_right": c["transcript_right"],
                "share_transcript_right": 100 * c["transcript_right"] / n if n else None}
    cap["recomputable"] = {
        "questions_template_defined": len(defined),
        "questions_recomputation_matches_key_on_true_data": len(ok_q),
        "of_those_with_a_parseable_transcript_implied_answer": dict(sorted(comp.items())),
        "of_those_with_a_parseable_non_truncated_transcript": dict(nt_comp),
        "note": ("the paper's 92 = questions whose template answer can be recomputed from the "
                 "TRUE data object and reproduces the key (tta_analysis error_source "
                 "n_questions_used); whether a given transcript yields an answer is counted "
                 "separately here")}
    cap["error_source_wrong_tta_answers"] = {
        md: {"all": share(src_all[md]), "not_truncated": share(src_nt[md])}
        for md in ("tabular", "graph")}
    res["generation_cap"] = cap

    # ---- 4. ties
    tpl, alt = build_ties(meta)
    tied_q = sorted(alt)
    tsum = {"templates": tpl, "tied_questions_total": len(tied_q),
            "tied_questions_by_modality": dict(Counter(meta[q]["modality"] for q in tied_q)),
            "tied_question_ids": tied_q,
            "tied_in_tta_sample": sorted(set(tied_q) & set(qids)),
            "outlier_ties_on_two_row_tables": sum(
                meta[q]["task"] == "outlier_detection" and len(meta[q]["data"]) == 2
                for q in tied_q)}
    V1 = v1_rescore(meta, q2d, alt)
    ref = json.load(open(os.path.join(HERE, "canonical_results.json")))
    assert abs(V1["canonical"]["max_gap_core4"][2]
               - ref["v1"]["canonical"]["max_gap_core4"]["est"]) < 1e-9
    v1d = {}
    for m in rc.MODELS7:
        a, b = V1["canonical"]["models"][m], V1["tie_aware"]["models"][m]
        v1d[m] = {"overall_acc": a["overall_acc"], "overall_acc_tie_aware": b["overall_acc"],
                  "delta_pp": b["overall_acc"] - a["overall_acc"], "n_pairs": a["n_pairs"],
                  "gaps": {md: {"canonical": a["gaps"][md]["est"],
                                "tie_aware": b["gaps"][md]["est"],
                                "tie_aware_ci": [b["gaps"][md]["lo"], b["gaps"][md]["hi"]],
                                "best_worst_canonical": [a["gaps"][md]["best"],
                                                         a["gaps"][md]["worst"]],
                                "best_worst_tie_aware": [b["gaps"][md]["best"],
                                                         b["gaps"][md]["worst"]]}
                           for md in rc.MODS}}
    tsum["v1"] = {"models": v1d,
                  "max_gap_all7": {"canonical": V1["canonical"]["max_gap_all7"],
                                   "tie_aware": V1["tie_aware"]["max_gap_all7"]},
                  "max_gap_core4": {"canonical": V1["canonical"]["max_gap_core4"],
                                    "tie_aware": V1["tie_aware"]["max_gap_core4"]}}
    V2 = v2_rescore(meta, q2d, alt)
    v2d = {}
    for m in QWEN_V2:
        a, b = V2["canonical"][m], V2["tie_aware"][m]
        assert abs(a["tabular_residual_gap"]["est"] - ref["v2"]["canonical"][m]["gaps"][
            "tabular"]["all_answerable_exclNC"]["est"]) < 1e-9
        v2d[m] = {"overall_acc": a["overall_acc"], "overall_acc_tie_aware": b["overall_acc"],
                  "delta_pp": b["overall_acc"] - a["overall_acc"], "n_pairs": a["n_pairs"]}
        for g in ("tabular_residual_gap", "graph_residual_gap", "series_nonlossy3_gap"):
            v2d[m][g] = {"canonical": a[g]["est"], "tie_aware": b[g]["est"],
                         "tie_aware_ci": [b[g]["lo"], b[g]["hi"]], "tie_aware_p": b[g]["p"],
                         "n_q": b[g]["n_q"], "best_worst": [b[g]["best"], b[g]["worst"]]}
    tsum["v2"] = v2d
    res["ties"] = tsum

    # ---- 5 / 6
    res["h1_test_bookkeeping"] = h1_section()
    res["latency"] = latency_section(meta, tta)

    rc.dump(OUT, res)
    report(res)


def report(res):
    print(f"TTA rows: {res['tta_rows']}")
    for md, r in res["paired_effects"].items():
        x = r["raw_minus_best_image"]
        print(f"[1] {md:10s} raw {r['rawtext_acc']:.1f} - best image ({r['best_image_format']}) "
              f"{x['est']:+.1f} [{x['lo']:+.1f},{x['hi']:+.1f}] p={x['p']:.4f} n={x['n']} "
              f"obj={x['n_obj']}")
        if "raw_minus_best_drawing" in r:
            y = r["raw_minus_best_drawing"]
            print(f"    best drawing {r['best_drawing']}: {y['est']:+.1f} [{y['lo']:+.1f},"
                  f"{y['hi']:+.1f}] p={y['p']:.4f}")
        z = r["tta_minus_image_pooled"]
        print(f"    tta-image pooled {z['est']:+.1f} [{z['lo']:+.1f},{z['hi']:+.1f}] "
              f"p={z['p']:.4f} n={z['n']}")
    s = res["sample"]
    print(f"[2] objects {s['unique_objects']} pools {s['pool_size_fully_answerable']} "
          f"reproduced {s['selection_reproduced']}")
    c = res["generation_cap"]
    print(f"[3] truncated {c['n_truncated']}/750")
    for k, e in c["by_format"].items():
        print(f"    {k:28s} n_nt={e['n_not_truncated']:2d} fid {e['fidelity_all']:5.1f}->"
              f"{e['fidelity_not_truncated'] if e['fidelity_not_truncated'] is None else round(e['fidelity_not_truncated'], 1)}"
              f" tta {e['tta_acc_all']:5.1f}->{e['tta_acc_not_truncated']}")
    print(f"    spearman {json.dumps(c['spearman'], default=float)[:400]}")
    print(f"    recomputable {c['recomputable']}")
    print(f"    error source {c['error_source_wrong_tta_answers']}")
    t = res["ties"]
    print(f"[4] tied questions {t['tied_questions_total']} {t['tied_questions_by_modality']}")
    for k, v in t["templates"].items():
        print(f"    {k:40s} n={v['n']:4d} key_ok={v['key_in_argmax_set']:4d} tied={v['tied']}")
    for m, v in t["v1"]["models"].items():
        print(f"    v1 {m:18s} acc {v['overall_acc']:.3f} -> {v['overall_acc_tie_aware']:.3f} "
              f"({v['delta_pp']:+.3f}) gaps " + " ".join(
                  f"{md} {g['canonical']:.2f}->{g['tie_aware']:.2f}" for md, g in v["gaps"].items()))
    print(f"    v1 max gap {t['v1']['max_gap_all7']}")
    for m, v in t["v2"].items():
        print(f"    v2 {m:16s} acc {v['overall_acc']:.3f} -> {v['overall_acc_tie_aware']:.3f} "
              f"tab {v['tabular_residual_gap']['canonical']:.2f}->"
              f"{v['tabular_residual_gap']['tie_aware']:.2f} graph "
              f"{v['graph_residual_gap']['canonical']:.2f}->"
              f"{v['graph_residual_gap']['tie_aware']:.2f}")
    h = res["h1_test_bookkeeping"]
    print(f"[5] {h['test_in_h1_bound_py']}")
    for x in h["tex_hits"]:
        print(f"    {x['file']}:{x['line']} {x['text'][:110]}")
    la = res["latency"]
    print(f"[6] tta all {la['tta_all']} direct {la['direct_on_tta_rows_all']}")
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
