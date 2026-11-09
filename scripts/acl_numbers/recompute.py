#!/usr/bin/env python3
"""Recompute every data-derived number of the paper into one dictionary.

    python scripts/acl_numbers/recompute.py        (writes scripts/acl_numbers/recomputed.json)

Primary metric: canonical_em (scripts/analysis/scoring.py). Sources:
  * scripts/analysis/canonical_results.json  (v1 per-format EM, gaps, Holm, flips, absent vs prior,
    answerable table, selector, raw-text, no-image, assist, scale, CoT, v2 multi-model canonical)
  * canonical REGENERATIONS of the generated appendix tables: error_taxonomy, template_difficulty,
    cot_audit, param_sensitivity and appendix_tables are re-run with review_common.em_val (resp.
    appendix_tables.load_preds) replaced by canonical_em and their outputs redirected to
    scripts/acl_numbers/regen/ (the paper's table fragments can be diffed against these
    files); tab_qual is regenerated with canonical bolding.
  * tta_analysis_results.json, chartqa_results.json, chartqa_lora_paired.json,
    chartqa_gap_noscatter.json, scripts/mitigation/rescore_lora_results.json (LoRA tolerance
    scorer), answerability_results.json, v2_results_strict.json (audit shares), the human
    rating sheets, and direct computations from benchmark/ and results/ (counts, controls,
    ensemble, prompt-style table, F1 table, InternVL 2x2, split-half check, ...).
Deterministic: every resampling uses random.Random(0).
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
AN = os.path.join(ROOT, "scripts", "analysis")
REGEN = os.path.join(HERE, "regen")
sys.path.insert(0, AN)
sys.path.insert(0, ROOT)

import review_common as rc  # noqa: E402
import scoring as sc  # noqa: E402
import v2_analysis as v2a  # noqa: E402

J = lambda p: json.load(open(p))  # noqa: E731


def canon_row(r):
    raw = r.get("raw_response")
    if raw in (None, "None", ""):
        raw = r.get("prediction_raw")
        if raw is not None and str(raw) == str(r.get("prediction")):
            raw = None
    return sc.canonical_em(r.get("prediction", ""), raw, r["answer"], r.get("task"),
                           float(r.get("exact_match", 0) or 0))


def ci_of(items, q2d, perm=False):
    r = v2a.contrast(items, q2d, perm=perm)
    o = {"est": r["est"], "lo": r["obj_ci"]["lo"], "hi": r["obj_ci"]["hi"], "n": r["n_units"]}
    if perm:
        o["p"] = r["obj_perm_p"]
    return o


# ---------------------------------------------------------------- canonical regenerations
def regen_scripts():
    os.makedirs(REGEN, exist_ok=True)
    shutil.copy(os.path.join(AN, "prior_and_subsets_results.json"), REGEN)
    old = (rc.em_val, rc.HERE, rc.TABLES)
    rc.em_val, rc.HERE, rc.TABLES = canon_row, REGEN, REGEN
    out = {}
    try:
        import cot_audit
        import error_taxonomy
        import param_sensitivity
        import template_difficulty
        for mod, js in ((error_taxonomy, "error_taxonomy_results.json"),
                        (template_difficulty, "template_difficulty_results.json"),
                        (cot_audit, "cot_audit_results.json"),
                        (param_sensitivity, "param_sensitivity_results.json")):
            with contextlib.redirect_stdout(io.StringIO()):
                mod.main()
            out[mod.__name__] = J(os.path.join(REGEN, js))
    finally:
        rc.em_val, rc.HERE, rc.TABLES = old
    # appendix_tables: canonical load_preds, outputs into REGEN
    import appendix_tables as at

    def load_preds(fn, field):
        scores, n = {}, 0
        for ln in open(os.path.join(at.RES, fn)):
            if ln.strip():
                r = json.loads(ln)
                n += 1
                scores[(r["question_id"], r["viz_type"])] = canon_row(r)
        return scores, n, 0
    at.load_preds, at.TAB_DIR, at.OUT_JSON = load_preds, REGEN, os.path.join(REGEN,
                                                                              "appendix_tables.json")
    argv = sys.argv
    sys.argv = ["appendix_tables.py"]
    with contextlib.redirect_stdout(io.StringIO()):
        at.main()
    sys.argv = argv
    out["appendix_tables"] = J(os.path.join(REGEN, "appendix_tables.json"))
    regen_qual()
    return out


def regen_qual():
    import qualitative_examples as qe
    meta = rc.load_bench()
    pred = {}
    for name, fn in qe.MODELS:
        for r in rc.read_jsonl(os.path.join(rc.RES, fn)):
            if r["question_id"] in qe.ITEMS:
                pred[(name, r["question_id"], r["viz_type"])] = (r["prediction"], canon_row(r))
    v2 = {}
    for r in rc.read_jsonl(os.path.join(rc.RES, "v2", "v2_qwen.jsonl")):
        if r["question_id"] in qe.ITEMS:
            v2[(r["question_id"], r["viz_type"])] = (r["prediction"], canon_row(r))
    out = []
    for qid in qe.ITEMS:
        b = meta[qid]
        fm = qe.FMT[b["modality"]]
        out.append("\\multicolumn{%d}{@{}p{0.97\\textwidth}@{}}{\\emph{%s} --- %s \\quad key: "
                   "\\texttt{%s}} \\\\" % (len(fm) + 1, qe.esc(qid.split("::")[0]),
                                           qe.esc(b["question"]), qe.esc(b["answer"])))
        out.append("model & " + " & ".join("\\texttt{%s}" % qe.esc(f) for f in fm)
                   + " \\\\ \\midrule")
        for name, _ in qe.MODELS:
            cells = []
            for f in fm:
                p, ok = pred.get((name, qid, f), ("--", 0))
                cells.append(("\\textbf{%s}" if ok == 1 else "%s") % qe.esc(p[:14]))
            out.append(qe.esc(name) + " & " + " & ".join(cells) + " \\\\")
        cells = []
        for f in fm:
            p, ok = v2.get((qid, f), ("--", 0))
            cells.append(("\\textbf{%s}" if ok == 1 else "%s") % qe.esc(p[:14]))
        out.append("Qwen2.5-VL-7B, corrected suite & " + " & ".join(cells) + " \\\\ \\midrule")
    tex = ["% per-format predictions for three example items; bold = exact match with the key",
           "\\begin{tabular}{l" + "c" * 5 + "}", "\\toprule"] + out[:-1] + [
        out[-1].replace(" \\midrule", ""), "\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(REGEN, "tab_qual.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")


# ---------------------------------------------------------------- direct computations
def bench_facts(meta):
    q2d = {q: b["data_id"] for q, b in meta.items()}
    by_obj = Counter(q2d.values())
    cnt = defaultdict(Counter)
    for b in meta.values():
        cnt[(b["modality"], b["task"])][rc.norm(b["answer"])] += 1
    nc = {}
    for k, c in cnt.items():
        a, n = c.most_common(1)[0]
        if n / sum(c.values()) >= 0.9:
            nc[f"{k[0]}|{k[1]}"] = {"n": sum(c.values()), "share": 100 * n / sum(c.values()),
                                    "answer": a}
    src = Counter((b["modality"], "scitab" if b["source"] == "scitabalign" else
                   "synthetic" if b["source"] == "synthetic" else b["source"])
                  for b in meta.values())
    tab_rows = {q: len(b["data"]) for q, b in meta.items() if b["modality"] == "tabular"}
    ans_text = Counter(rc.norm(b["answer"]) for b in meta.values())
    graph_nodes = {q: len(b["data"]["nodes"]) for q, b in meta.items() if b["modality"] == "graph"}
    dq = [q for q, b in meta.items() if b["task"] == "degree_query"]
    out = {"n_q": len(meta), "n_tab": sum(b["modality"] == "tabular" for b in meta.values()),
           "n_ts": sum(b["modality"] == "timeseries" for b in meta.values()),
           "n_graph": sum(b["modality"] == "graph" for b in meta.values()),
           "n_obj": len(by_obj), "q_per_obj_mean": statistics.mean(by_obj.values()),
           "q_per_obj_max": max(by_obj.values()), "n_templates": len(cnt),
           "near_constant": nc, "n_near_constant_cells": len(nc),
           "n_near_constant_q": sum(v["n"] for v in nc.values()),
           "pattern_label_lookup_n": sum(cnt[("timeseries", "pattern_label_lookup")].values()),
           "sources": {f"{a}|{b}": v for (a, b), v in src.items()},
           "ett_label_nx_degree": sum(1 for b in meta.values() if b["source"] == "ett"
                                      and str(b["data_id"]).startswith("nx_")),
           "ett_label_nx_degree_objects": len({b["data_id"] for b in meta.values()
                                               if b["source"] == "ett"
                                               and str(b["data_id"]).startswith("nx_")}),
           "ett_label_true_ett": sum(1 for b in meta.values() if b["source"] == "ett"
                                     and not str(b["data_id"]).startswith("nx_")),
           "scitab_rows_le12": sum(1 for q, n in tab_rows.items()
                                   if meta[q]["source"] == "scitabalign" and n <= 12),
           "tab_rows_le12": sum(1 for n in tab_rows.values() if n <= 12),
           "tab_rows_gt12": sum(1 for n in tab_rows.values() if n > 12),
           "tab_rows_gt20": sum(1 for n in tab_rows.values() if n > 20),
           "value_extraction_n": sum(b["task"] == "value_extraction" and b["modality"] == "tabular"
                                     for b in meta.values()),
           "value_extraction_max_row": max(int(re.search(r"row (\d+)", b["question"]).group(1))
                                           for b in meta.values()
                                           if b["task"] == "value_extraction"),
           "answer_count_east_grid": ans_text["east_grid"],
           "answer_count_first_half": ans_text["first half"],
           "answer_count_second_half": ans_text["second half"],
           "degree_query_n": len(dq),
           "degree_query_ge30": sum(graph_nodes[q] >= 30 for q in dq),
           "ts_len_le43": sum(1 for b in meta.values() if b["modality"] == "timeseries"
                              and len(b["data"]) <= 43),
           "ts_len_gt43": sum(1 for b in meta.values() if b["modality"] == "timeseries"
                              and len(b["data"]) > 43),
           "graph_edges_gt43": sum(1 for b in meta.values() if b["modality"] == "graph"
                                   and len(b["data"].get("edges", b["data"].get("links", []))) > 43),
           "graph_nodes_ge30": sum(1 for n in graph_nodes.values() if n >= 30),
           "correlation_n": sum(b["task"] == "correlation" for b in meta.values()),
           "tab_counterfactual_n": sum(cnt[("tabular", "counterfactual")].values())}
    # tabular counterfactual wording: "If all values in 'A' were multiplied by k, would 'A'
    # remain above 'B' on average?" -- key = ordering unchanged; literal = A > B before and after
    pres_fail = literal_diff = 0
    for b in meta.values():
        if b["modality"] != "tabular" or b["task"] != "counterfactual":
            continue
        lit = literal_cf_key(b)
        if lit is None:
            continue
        pres_fail += not lit[1]
        literal_diff += lit[0] != rc.norm(b["answer"])
    out["tab_cf_presupposition_fails"] = pres_fail
    out["tab_cf_literal_differs"] = literal_diff
    return out


def literal_cf_key(b):
    cols = re.findall(r"'([^']*)'", b["question"])
    k = re.search(r"multiplied by ([\d.]+)", b["question"])
    if len(cols) < 3 or not k:
        return None
    a, c, k = cols[0], cols[2], float(k.group(1))
    va = [r[a] for r in b["data"] if isinstance(r.get(a), (int, float))]
    vc = [r[c] for r in b["data"] if isinstance(r.get(c), (int, float))]
    if not va or not vc:
        return None
    ma, mc = statistics.mean(va), statistics.mean(vc)
    return ("yes" if (ma > mc and k * ma > mc) else "no"), ma > mc


def cf_rekey(meta):
    """Max change of a core-model tabular best-worst gap when the tabular counterfactual
    questions are re-keyed literally."""
    out = {}
    lit = {q: literal_cf_key(b) for q, b in meta.items()
           if b["modality"] == "tabular" and b["task"] == "counterfactual"}
    for m in rc.CORE4:
        rows = rc.load_v1(m)
        gaps = {}
        for lab in ("canonical", "literal"):
            pf = defaultdict(list)
            for (q, f), r in rows.items():
                if meta[q]["modality"] != "tabular":
                    continue
                if lab == "literal" and lit.get(q):
                    r = dict(r, answer=lit[q][0], exact_match=float(
                        sc.exact_match(str(r.get("prediction", "")), lit[q][0])))
                pf[f].append(canon_row(r))
            pf = {f: 100 * sum(v) / len(v) for f, v in pf.items()}
            gaps[lab] = max(pf.values()) - min(pf.values())
        out[m] = gaps
    out["max_abs_change"] = max(abs(v["literal"] - v["canonical"]) for v in out.values()
                                if isinstance(v, dict))
    return out


def example_facts(meta):
    t = meta["tabular_001326::difficulty=2-hop"]
    s = meta["timeseries_000027::difficulty=1-hop"]
    g = meta["graph_000076::difficulty=1-hop"]
    v2 = {(r["question_id"], r["viz_type"]): r for r in rc.read_jsonl(
        os.path.join(rc.RES, "v2", "v2_qwen.jsonl"))}
    teaser = defaultdict(dict)
    for m, fn in rc.MODELS7.items():
        for (q, f), r in rc.load_v1(m).items():
            if q == "tabular_001326::difficulty=2-hop":
                teaser[f][m] = (str(r["prediction"]), canon_row(r))
    vals = [float(v) for v in s["data"]]
    return {"teaser_key": t["answer"],
            "teaser_table_rows": len(t["data"]), "teaser_table_cols": len(t["data"][0]),
            "teaser_correct": {f: sum(ok for _, ok in d.values()) for f, d in teaser.items()},
            "teaser_bar_7577": sum(p == "75.77" for p, _ in teaser["bar_chart"].values()),
            "teaser_v2_bar_pred": v2[("tabular_001326::difficulty=2-hop", "bar_chart")]["prediction"],
            "ts_example_len": len(vals), "ts_example_argmin": vals.index(min(vals)),
            "graph_example_nodes": len(g["data"]["nodes"]),
            "graph_example_v2_text_pred": v2[("graph_000076::difficulty=1-hop", "text_only")][
                "prediction"]}


def human(meta):
    hp, hp2 = os.path.join(ROOT, "human_eval_package"), os.path.join(ROOT, "human_eval_package_v2")
    r = [list(csv.DictReader(open(os.path.join(hp, f"ratings_annotator{i}.csv"), encoding="utf-8")))
         for i in (1, 2)]
    cats = ["Correct", "Ambiguous", "Incorrect"]
    cnt = [{k: sum(x["rating"] == k for x in rows) for k in cats} for rows in r]
    a = {x["item_id"]: x for x in r[0]}
    b = {x["item_id"]: x for x in r[1]}
    ids = sorted(set(a) & set(b))
    n = len(ids)

    def kappa(fa, fb, sub=None):
        S = sub or ids
        m = len(S)
        po = sum(fa(i) == fb(i) for i in S) / m
        vals = {fa(i) for i in S} | {fb(i) for i in S}
        pe = sum((sum(fa(i) == v for i in S) / m) * (sum(fb(i) == v for i in S) / m) for v in vals)
        return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")
    po3, k3 = kappa(lambda i: a[i]["rating"], lambda i: b[i]["rating"])
    po2, k2 = kappa(lambda i: a[i]["rating"] == "Correct", lambda i: b[i]["rating"] == "Correct")
    kmod = {md: kappa(lambda i: a[i]["rating"], lambda i: b[i]["rating"],
                      [i for i in ids if a[i]["modality"] == md])[1]
            for md in ("tabular", "timeseries", "graph")}
    rng = random.Random(0)
    ks = []
    for _ in range(2000):
        S = [ids[int(rng.random() * n)] for _ in range(n)]
        ks.append(kappa(lambda i: a[i]["rating"], lambda i: b[i]["rating"], S)[1])
    klo, khi = v2a.pct([k for k in ks if k == k])
    disagree = [i for i in ids if a[i]["rating"] != b[i]["rating"]]
    sys.path.insert(0, hp)
    from human_eval_aggregate import answer_match
    ans = [x for x in r[1] if x["your_answer"].strip()]
    ok = [x for x in ans if answer_match(x["your_answer"], x["ground_truth"])]
    mod_counts = Counter(x["modality"] for x in r[0])
    fmts = {(x["modality"], x["viz_type"]) for x in r[0]}
    # audit classes of the sampled pairs
    cat = rc.v1_cats(meta)
    items = {json.loads(ln)["item_id"]: json.loads(ln) for ln in open(
        os.path.join(hp, "items.jsonl"), encoding="utf-8") if ln.strip()}
    cls = {i: cat[items[i]["question_id"]][items[i]["viz_type"]] for i in ids}
    absent = [i for i in ids if cls.get(i) == "A"]
    vd = [i for i in ids if cls.get(i) in ("V", "D")]
    out = {"rater_counts": cnt, "n_items": n, "mod_counts": dict(mod_counts), "n_formats": len(fmts),
           "agree3": 100 * po3, "kappa3": k3, "kappa3_ci": [klo, khi], "kappa_by_mod": kmod,
           "agree2": 100 * po2, "kappa2": k2, "n_disagree": len(disagree),
           "disagree_with_ambiguous": sum("Ambiguous" in (a[i]["rating"], b[i]["rating"])
                                          for i in disagree),
           "both_correct": sum(a[i]["rating"] == b[i]["rating"] == "Correct" for i in ids),
           "both_incorrect": sum(a[i]["rating"] == b[i]["rating"] == "Incorrect" for i in ids),
           "r2_answered": len(ans), "r2_match": len(ok),
           "r2_match_ambiguous": sum(x["rating"] == "Ambiguous" for x in ok),
           "r2_ambiguous": cnt[1]["Ambiguous"],
           "audit_absent_n": len(absent),
           "audit_absent_both_incorrect": sum(a[i]["rating"] == b[i]["rating"] == "Incorrect"
                                              for i in absent),
           "audit_absent_any_incorrect": sum("Incorrect" in (a[i]["rating"], b[i]["rating"])
                                             for i in absent),
           "audit_vd_n": len(vd),
           "audit_vd_both_correct": sum(a[i]["rating"] == b[i]["rating"] == "Correct" for i in vd),
           "audit_vd_both_incorrect": sum(a[i]["rating"] == b[i]["rating"] == "Incorrect"
                                          for i in vd)}
    v2s = [{x["item_id"]: x["rating"] for x in csv.DictReader(open(
        os.path.join(hp2, f"ratings_annotator{i}.csv"), encoding="utf-8"))} for i in (1, 2)]
    it2 = {json.loads(ln)["item_id"]: json.loads(ln) for ln in open(
        os.path.join(hp2, "items.jsonl"), encoding="utf-8") if ln.strip()}
    mapped = [i for i, it in it2.items() if it.get("v1_item_id")]
    c2 = Counter(v2s[0].values())
    out.update({"v2_items": len(it2), "v2_mapped": len(mapped),
                "v2_new_ts": sum(1 for i, it in it2.items() if not it.get("v1_item_id")),
                "v2_sheets_identical": all(v2s[0][i] == v2s[1][i] for i in it2),
                "v2_counts": dict(c2),
                "v2_correct_by_mod": {md: sum(v2s[0][i] == "Correct" for i in it2
                                              if it2[i]["modality"] == md)
                                      for md in ("tabular", "graph", "timeseries")},
                "v2_n_by_mod": dict(Counter(it2[i]["modality"] for i in it2)),
                "v2_mapped_before": [sum((a, b)[k][it2[i]["v1_item_id"]]["rating"] == "Correct"
                                         for i in mapped) for k in (0, 1)],
                "v2_mapped_after": [sum(v2s[k][i] == "Correct" for i in mapped) for k in (0, 1)]})
    return out


def ensemble(meta, q2d):
    """cluster_cis.part_c with canonical scores."""
    def norm(p):
        p = str(p).strip().lower()
        m = re.search(r"-?\d+(?:\.\d+)?", p.replace(",", ""))
        return m.group() if m else p
    out = {}
    for m in rc.MODELS7:
        rows = rc.load_v1(m)
        groups = defaultdict(list)
        pf = defaultdict(lambda: defaultdict(list))
        for (q, f), r in rows.items():
            e = canon_row(r)
            groups[q].append((f, e, norm(r.get("prediction", ""))))
            pf[meta[q]["modality"]][f].append(e)
        fixed = {md: max(d, key=lambda f: sum(d[f]) / len(d[f])) for md, d in pf.items()}
        rnd = ens = orc = fix = 0.0
        d_fix = []
        for q, g in groups.items():
            ems = [e for _, e, _ in g]
            mean = sum(ems) / len(ems)
            top = Counter(p for _, _, p in g).most_common(1)[0][0]
            e = 1.0 if max((em for _, em, p in g if p == top), default=0.0) >= 0.5 else 0.0
            fm = {v: em for v, em, _ in g}
            fx = fm.get(fixed[meta[q]["modality"]], mean)
            rnd += mean
            ens += e
            orc += float(any(x >= 0.5 for x in ems))
            fix += fx
            d_fix.append((q, e - fx))
        n = len(groups)
        out[m] = {"random": 100 * rnd / n, "fixed": 100 * fix / n, "ensemble": 100 * ens / n,
                  "oracle": 100 * orc / n, "delta": ci_of(d_fix, q2d)["est"]}
    return out


def prompt_table(meta):
    out = {}
    for m, key in rc.PROMPT_KEYS.items():
        out[m] = {}
        for s in rc.STYLES:
            rows = rc.read_jsonl(os.path.join(rc.RES, "ablation", f"prompt_{key}_{s}.jsonl"))
            em = [canon_row(r) for r in rows]
            pf = defaultdict(list)
            for r, e in zip(rows, em):
                if r["modality"] == "tabular":
                    pf[r["viz_type"]].append(e)
            pf = {f: 100 * sum(v) / len(v) for f, v in pf.items()}
            out[m][s] = {"em": 100 * sum(em) / len(em), "tab_gap": max(pf.values()) - min(pf.values()),
                         "n": len(rows)}
    return out


def f1_table():
    out = {}
    for m in rc.CORE4:
        pf = defaultdict(list)
        for (q, f), r in rc.load_v1(m).items():
            pf[f"{r['modality']}|{f}"].append(float(r.get("f1", 0) or 0))
        out[m] = {k: 100 * sum(v) / len(v) for k, v in pf.items()}
    return out


def ts_control(meta):
    v1 = rc.load_v1("Qwen2.5-VL-7B")
    v2, _ = v2a.dedupe(rc.read_jsonl(os.path.join(rc.RES, "v2", "v2_qwen.jsonl")))
    out = {}
    for f in ("line_plot", "heatmap", "gaf", "recurrence_plot"):
        ks = [(q, f) for q in meta if meta[q]["modality"] == "timeseries"]
        e1 = [canon_row(v1[k]) for k in ks]
        e2 = [canon_row(v2[k]) for k in ks]
        out[f] = {"n": len(ks), "em_agree": 100 * sum(a == b for a, b in zip(e1, e2)) / len(ks),
                  "pred_agree": 100 * sum(rc.norm(v1[k]["prediction"]) == rc.norm(v2[k]["prediction"])
                                          for k in ks) / len(ks),
                  "change": 100 * (sum(e2) - sum(e1)) / len(ks)}
    return out


def run_costs(meta):
    v2rows = rc.read_jsonl(os.path.join(rc.RES, "v2", "v2_qwen.jsonl"))
    lat = [float(r["latency_s"]) for r in v2rows if r.get("latency_s") not in (None, "")]
    toks = [int(r["n_visual_tokens"]) for r in v2rows if r.get("n_visual_tokens")]
    man = rc.read_jsonl(os.path.join(ROOT, "benchmark", "render_v2", "manifest.jsonl"))
    mpx = defaultdict(list)
    for r in man:
        mpx[r["modality"]].append(int(r["width"]) * int(r["height"]) / 1e6)
    raw = rc.read_jsonl(os.path.join(rc.RES, "v2", "rawtext_qwen.jsonl"))
    rtok = defaultdict(list)
    for r in raw:
        if r.get("n_prompt_tokens"):
            rtok[r["modality"]].append(int(r["n_prompt_tokens"]))
    return {"v2_queries": len({(r["question_id"], r["viz_type"]) for r in v2rows}),
            "v2_gpu_hours": sum(lat) / 3600, "v2_mean_latency": statistics.mean(lat),
            "v2_tokens_min": min(toks), "v2_tokens_max": max(toks),
            "v1_mpx": 1024 * 768 / 1e6,
            "v2_mpx_by_mod": {k: statistics.mean(v) for k, v in mpx.items()},
            "noimage_n": len(rc.read_jsonl(os.path.join(rc.RES, "v2", "noimage_qwen.jsonl"))),
            "rawtext_n": len({r["question_id"] for r in raw}),
            "assist_n": len(rc.read_jsonl(os.path.join(rc.RES, "v2", "assist_qwen.jsonl"))),
            "rawtext_tokens": {k: statistics.mean(v) for k, v in rtok.items()}}


def internvl_v2(meta, q2d, C):
    """The 'second open model' paragraph: InternVL2.5-8B, one tile, canonical."""
    v1 = rc.load_v1("InternVL2.5-8B")
    v2, _ = v2a.dedupe(rc.read_jsonl(os.path.join(rc.RES, "v2", "v2_internvl_subset.jsonl")))
    qs = sorted({q for q, f in v2})
    tab = [q for q in qs if meta[q]["modality"] == "tabular"]
    ts = [q for q in qs if meta[q]["modality"] == "timeseries"]
    out = {"n_q": len(qs), "n_pairs": len([k for k in v2 if k[1] in rc.FORMATS[meta[k[0]]["modality"]]]),
           "n_tab": len(tab)}
    for f in ("table_image", "text_only", "heatmap", "bar_chart"):
        out[f"tab_{f}_v1"] = rc.mean100(canon_row(v1[(q, f)]) for q in tab)
        out[f"tab_{f}_v2"] = rc.mean100(canon_row(v2[(q, f)]) for q in tab)
    agree = sum(all(rc.norm(v1[(q, f)]["prediction"]) == rc.norm(v2[(q, f)]["prediction"])
                    for f in ("line_plot", "heatmap", "gaf", "recurrence_plot")) for q in ts)
    out["ts_questions"] = len(ts)
    out["ts_agree_questions"] = agree
    V = C["v2"]["canonical"]["InternVL2.5-8B (1 tile)"]
    out["graph_text_recovery"] = V["recovery"]["graph|text_only"]
    out["degree"] = V["degree_effect"]["degree_query"]
    out["tab_gap"] = V["gaps"]["tabular"]["all_answerable_exclNC"]
    out["graph_gap"] = V["gaps"]["graph"]["all_answerable_exclNC"]
    out["graph_beating_prior"] = sum(v["exceeds_same_q"] for v in V["formats_vs_prior"]["graph"]
                                     .values())
    rows = rc.read_jsonl(os.path.join(rc.RES, "v2", "assist_internvl.jsonl"))
    EMa = defaultdict(dict)
    for r in rows:
        if r["viz_type"] in v2a.ASSIST:
            EMa[r["question_id"]][r["viz_type"]] = canon_row(r)
    with contextlib.redirect_stdout(io.StringIO()):
        s = v2a.s6_assist(EMa, meta, q2d)
    agg = s["aggregation_mean"]
    out["assist_agg_n"] = agg["n"]
    out["assist_agg_means"] = agg["means"]
    out["assist_agg_interaction"] = {
        k: agg["interaction(table_means_eff - bar_means_eff)"][k] for k in ("est",)}
    it = agg["interaction(table_means_eff - bar_means_eff)"]
    out["assist_agg_interaction"] = {"est": it["est"], "lo": it["obj_ci"]["lo"],
                                     "hi": it["obj_ci"]["hi"]}
    out["assist_comparison_means"] = s["comparison"]["means"]
    out["v1_error_rows"] = sum(1 for r in v1.values() if str(r.get("prediction")) == "[ERROR]")
    out["visual_tokens"] = sorted({int(r["n_visual_tokens"]) for r in v2.values()
                                   if r.get("n_visual_tokens")})
    out["max_tab_rows"] = max(len(meta[q]["data"]) for q in tab)
    return out


def split_half(meta):
    B = 10000
    out = []
    for m in rc.CORE4:
        rows = rc.load_v1(m)
        q = defaultdict(dict)
        for (qid, f), r in rows.items():
            q[qid][f] = canon_row(r)
        for md in rc.MODS:
            ids = sorted(x for x in q if meta[x]["modality"] == md)
            fm = rc.FORMATS[md]
            half = {h: [i for i in ids if int(hashlib.md5(i.encode()).hexdigest(), 16) % 2 == h]
                    for h in (0, 1)}
            for sel, test in ((0, 1), (1, 0)):
                means = {f: sum(q[i][f] for i in half[sel]) / len(half[sel]) for f in fm}
                fs = sorted(means)
                b = max(fs, key=lambda f: means[f])
                w = min(fs, key=lambda f: means[f])
                d = [q[i][b] - q[i][w] for i in half[test]]
                obs = abs(sum(d))
                nz = [x for x in d if x != 0]
                rng = random.Random(0)
                hits = sum(1 for _ in range(B)
                           if abs(2 * rng.getrandbits(len(nz)).bit_count() - len(nz)) >= obs - 1e-9)
                p = (hits + 1) / (B + 1)
                out.append({"model": m, "modality": md, "sel": sel, "gap": 100 * sum(d) / len(d),
                            "p": p, "sig_bonf12_05": p * 12 < 0.05})
    return {"rows": out, "n_sig": sum(r["sig_bonf12_05"] for r in out), "n": len(out)}


def correlation_scatter(meta):
    out = {}
    for m in rc.MODELS7:
        rows = rc.load_v1(m)
        ks = [(q, "scatter_plot") for q in meta if meta[q]["task"] == "correlation"
              and meta[q]["modality"] == "tabular"]
        out[m] = {"strict": rc.mean100(float(rows[k].get("exact_match", 0) or 0) for k in ks),
                  "canonical": rc.mean100(canon_row(rows[k]) for k in ks)}
    return out


def regen_extra():
    """Fragments written by canonical_extras / tta_analysis / canonical_rescore, regenerated from
    their JSON into REGEN."""
    import canonical_extras as ce
    import canonical_rescore as cr
    import tta_analysis as ta
    C = J(os.path.join(AN, "canonical_results.json"))
    old = ce.TAB
    ce.TAB = REGEN
    try:
        ce.tex_perviz(C)
        ce.tex_selector(C)
    finally:
        ce.TAB = old
    cr.write_diff_table(C, os.path.join(REGEN, "tab_strict_vs_canonical.tex"))
    old_root = ta.ROOT
    os.makedirs(os.path.join(REGEN, "paper_tables"), exist_ok=True)
    ta.ROOT = REGEN
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ta.write_tex(J(os.path.join(AN, "tta_analysis_results.json")))
    finally:
        ta.ROOT = old_root
    shutil.move(os.path.join(REGEN, "paper_tables", "tab_tta.tex"),
                os.path.join(REGEN, "tab_tta.tex"))


def inline_tables(R, C):
    """Canonical replacements for the three tables typed inline in appendix_acl.tex
    (tab:prompt, tab:answerable, tab:ensemble), same layout, into REGEN."""
    pr = R["prompt"]
    labs = (("GPT-4o", "GPT-4o"), ("Gemini Flash", "Gemini-2.0-Flash"),
            ("Qwen2.5-VL-7B", "Qwen2.5-VL-7B"), ("Claude Sonnet", "Claude Sonnet 4"))
    L = [r"\begin{table*}[h]\centering\small",
         r"\caption{Prompt-style ablation on a stratified \num{500}-question subsample",
         r"(\num{%s} rendered instances per cell). Each entry is overall exact match (\%%) with the"
         % f"{pr['GPT-4o']['cot']['n']:,}".replace(",", "{,}"),
         r"tabular best--worst format gap (pp) in parentheses; \textbf{bold} marks each model's best style.",
         r"Prompt style matters, but the format gap persists in all \num{16} cells, and \textsc{CoT} never",
         r"beats the best non-\textsc{CoT} prompt for any model; Table~\ref{tab:cotaudit} separates the part",
         r"of the \textsc{CoT} deficit due to answer extraction.}",
         r"\label{tab:prompt}", r"\begin{tabular}{lcccc}", r"\toprule",
         r"Model & minimal & concise & detailed & \textsc{CoT} \\", r"\midrule"]
    for lab, m in labs:
        best = max(round(pr[m][s_]["em"], 1) for s_ in rc.STYLES)
        cells = []
        for s_ in rc.STYLES:
            v = f"{pr[m][s_]['em']:.1f}"
            v = rf"\textbf{{{v}}}" if round(pr[m][s_]["em"], 1) == best else v
            cells.append(f"{v}~({pr[m][s_]['tab_gap']:.1f})")
        L.append(f"{lab:14s}& " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    with open(os.path.join(REGEN, "tab_prompt_canonical.tex"), "w") as fh:
        fh.write("% Canonical exact match. Generated by scripts/acl_numbers/recompute.py.\n"
                 + "\n".join(L) + "\n")
    ans = C["answerable"]["canonical"]
    order = {"tabular": ["bar_chart", "heatmap", "scatter_plot", "table_image", "text_only"],
             "timeseries": ["gaf", "heatmap", "line_plot", "recurrence_plot", "text_only"],
             "graph": ["adjacency_matrix", "circular_layout", "node_link", "text_only"]}
    g = ans["GPT-4o"]
    a = R["answerability"]
    L = [r"\begin{table*}[h]\centering\small",
         r"\caption{Answerability audit: exact match (\%) on (question, format) pairs whose answer is",
         r"\emph{present} in the image (visible or derivable) / \emph{absent} by design. Pair counts are the",
         r"same for every model; e.g.\ tabular \texttt{bar\_chart} present $n{=}%d$, absent $n{=}%s$;"
         % (g["tabular|bar_chart"]["VD"]["n"],
            f"{g['tabular|bar_chart']['A']['n']:,}".replace(",", "{,}")),
         r"\texttt{scatter\_plot} present $n{=}%d$; graph \texttt{text\_only} present $n{=}%d$. ``--'' marks"
         % (g["tabular|scatter_plot"]["VD"]["n"], g["graph|text_only"]["VD"]["n"]),
         r"an empty cell. Constant-answer pairs (\num{%d}\%% of series, \num{%d}\%% of graph pairs) are"
         % (round(a["timeseries"]["C"]), round(a["graph"]["C"])),
         r"excluded. Full counts per task and format are in the released \texttt{answerability\_matrix.md}.}",
         r"\label{tab:answerable}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lccccc}",
         r"\toprule"]
    heads = {"tabular": r"\multicolumn{6}{l}{\emph{Tabular}: bar\_chart / heatmap / scatter\_plot / table\_image / text\_only} \\",
             "timeseries": r"\multicolumn{6}{l}{\emph{Time series}: gaf / heatmap / line\_plot / recurrence\_plot / text\_only} \\",
             "graph": r"\multicolumn{6}{l}{\emph{Graph}: adjacency\_matrix / circular\_layout / node\_link / text\_only} \\"}
    for i, (md, fm) in enumerate(order.items()):
        if i:
            L.append(r"\midrule")
        L.append(heads[md])
        for lab, m in labs:
            cells = []
            for f in fm:
                vd, ab = ans[m][f"{md}|{f}"]["VD"], ans[m][f"{md}|{f}"]["A"]
                left = f"{vd['em']:.0f}" if vd["n"] else "--"
                right = f"{ab['em']:.0f}" if ab["n"] else "--"
                cells.append(left + r"\,/\," + right)
            tail = " & \\\\" if md == "graph" else " \\\\"
            L.append(f"{lab:15s}& " + " & ".join(cells) + tail)
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    with open(os.path.join(REGEN, "tab_answerable_canonical.tex"), "w") as fh:
        fh.write("% Canonical exact match. Generated by scripts/acl_numbers/recompute.py.\n"
                 + "\n".join(L) + "\n")
    en = R["ensemble"]
    L = [r"\begin{table*}[h]\centering\small",
         r"\caption{Majority-vote format ensemble vs.\ two single-format references, exact match (\%) over",
         r"all base questions with a per-question denominator: ``random'' is an arbitrary format,",
         r"``fixed (mod)'' the best single format per modality (chosen in-sample), ``oracle''",
         r"correct-if-any-format-correct. The vote beats random but not the fixed format.}",
         r"\label{tab:ensemble}", r"\begin{tabular}{lccccc}", r"\toprule",
         r"Model & random & fixed (mod) & ensemble & $\Delta$ vs.\ fixed & oracle \\", r"\midrule"]
    for lab, m in (("GPT-4o", "GPT-4o"), ("Gemini Flash", "Gemini-2.0-Flash"),
                   ("Gemini-2.5", "Gemini-2.5-Flash"), ("Qwen2.5-VL-7B", "Qwen2.5-VL-7B"),
                   ("Claude Sonnet", "Claude Sonnet 4"), ("Qwen2.5-VL-32B", "Qwen2.5-VL-32B"),
                   ("InternVL2.5-8B", "InternVL2.5-8B")):
        v = en[m]
        L.append(f"{lab:14s}& {v['random']:.1f} & {v['fixed']:.1f} & {v['ensemble']:.1f} & "
                 f"${v['delta']:.1f}$ & {v['oracle']:.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    with open(os.path.join(REGEN, "tab_ensemble_canonical.tex"), "w") as fh:
        fh.write("% Canonical exact match. Generated by scripts/acl_numbers/recompute.py.\n"
                 + "\n".join(L) + "\n")


def more_facts(meta):
    depth = defaultdict(set)
    for b in meta.values():
        depth[(b["modality"], b["task"])].add(b["difficulty"])
    man = rc.read_jsonl(os.path.join(ROOT, "benchmark", "render_v2", "manifest.jsonl"))
    mpx = defaultdict(list)
    for r in man:
        mpx[r["viz_type"]].append(int(r["width"]) * int(r["height"]) / 1e6)
    syn_rows = [len(b["data"]) for b in meta.values() if b["modality"] == "tabular"
                and b["source"] == "synthetic"]
    nx_nodes = [len(b["data"]["nodes"]) for b in meta.values() if b["modality"] == "graph"
                and b["source"] != "synthetic"]
    all_nodes = [len(b["data"]["nodes"]) for b in meta.values() if b["modality"] == "graph"]
    lora = J(os.path.join(ROOT, "scripts", "mitigation", "rescore_lora_results.json"))
    tr = os.path.join(ROOT, "scripts", "mitigation", "pairs_all_train.jsonl")
    n_train = sum(1 for ln in open(tr) if ln.strip())
    try:
        out = subprocess.run([sys.executable, "-m", "pytest",
                              "--collect-only", "-q", "tests/test_scoring.py"], cwd=ROOT,
                             capture_output=True, text=True, timeout=120).stdout
        n_tests = int(re.search(r"(\d+) tests? collected", out).group(1))
    except Exception:  # noqa: BLE001
        n_tests = None
    return {"n_templates_single_depth": sum(len(v) == 1 for v in depth.values()),
            "v2_mpx_by_format": {k: statistics.mean(v) for k, v in mpx.items()},
            "syn_table_rows": [min(syn_rows), max(syn_rows)],
            "networkx_nodes": [min(nx_nodes), max(nx_nodes)],
            "graph_nodes": [min(all_nodes), max(all_nodes)],
            "lora_train_pairs": n_train, "lora_meta": lora["meta"],
            "lora_n_data_id": lora["n_data_id"], "lora_credit_audit": lora.get("credit_audit"),
            "n_scoring_tests": n_tests}


def f1_fragment(R):
    """tables/tab_f1.tex: the per-format token-F1 table of sec_appendix.tex (app:f1), same
    layout, every model (Claude included) from its *_extracted.jsonl file (rc.MODELS7)."""
    f1 = R["f1"]
    labs = (("GPT-4o", "GPT-4o"), ("Gemini Flash", "Gemini-2.0-Flash"),
            ("Qwen2.5-VL-7B", "Qwen2.5-VL-7B"), ("Claude Sonnet", "Claude Sonnet 4"))
    blocks = (("tabular", "Tabular", ["bar_chart", "heatmap", "scatter_plot", "table_image",
                                      "text_only"]),
              ("timeseries", "Time series", ["gaf", "heatmap", "line_plot", "recurrence_plot",
                                             "text_only"]),
              ("graph", "Graph", ["adjacency_matrix", "circular_layout", "node_link",
                                  "text_only"]))
    L = [r"\begin{table*}[htbp]\centering\small",
         r"\caption{F1 (\%) by format. Tabular / time-series columns share the same 5-format "
         r"layout.}",
         r"\begin{tabular}{lccccc}", r"\toprule"]
    for i, (md, name, fm) in enumerate(blocks):
        if i:
            L.append(r"\midrule")
        L.append(r"\multicolumn{6}{l}{\emph{%s}: %s} \\" % (
            name, " / ".join(f.replace("_", r"\_") for f in fm)))
        for lab, m in labs:
            v = [f"{f1[m][f'{md}|{f}']:.1f}" for f in fm]
            if md == "graph":
                L.append(f"{lab:<13s} & " + r"\multicolumn{5}{l}{" + r" \quad ".join(v) + r"} \\")
            else:
                L.append(f"{lab:<13s} & " + " & ".join(v) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    with open(os.path.join(REGEN, "tab_f1.tex"), "w") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    meta = rc.load_bench()
    q2d = {q: b["data_id"] for q, b in meta.items()}
    C = J(os.path.join(AN, "canonical_results.json"))
    R = {"note": __doc__}
    R["bench"] = bench_facts(meta)
    R["examples"] = example_facts(meta)
    R["regen"] = regen_scripts()
    regen_extra()
    R["more"] = more_facts(meta)
    R["human"] = human(meta)
    R["ensemble"] = ensemble(meta, q2d)
    R["prompt"] = prompt_table(meta)
    R["f1"] = f1_table()
    R["ts_control"] = ts_control(meta)
    R["costs"] = run_costs(meta)
    R["internvl_v2"] = internvl_v2(meta, q2d, C)
    R["split_half"] = split_half(meta)
    R["corr_scatter"] = correlation_scatter(meta)
    R["cf_rekey"] = cf_rekey(meta)
    R["answerability"] = J(os.path.join(AN, "answerability_results.json"))["category_fractions"]
    R["v2_categories"] = J(os.path.join(AN, "v2_results_strict.json"))["1_categories"]
    inline_tables(R, C)
    f1_fragment(R)
    with open(os.path.join(HERE, "recomputed.json"), "w") as fh:
        json.dump(R, fh, indent=1, default=float)
    print(f"wrote {os.path.relpath(os.path.join(HERE, 'recomputed.json'), ROOT)}; regenerated "
          f"tables in {os.path.relpath(REGEN, ROOT)}/")


if __name__ == "__main__":
    main()
