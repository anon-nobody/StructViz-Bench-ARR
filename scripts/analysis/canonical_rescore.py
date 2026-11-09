#!/usr/bin/env python3
"""Recompute the paper's numbers under canonical_em (scripts/analysis/scoring.py), side by side
with strict exact match.

Stdlib + the benchmark's own metrics module; no inference.

    python scripts/analysis/canonical_rescore.py

Sections
  0. Folding safety: distinct non-numeric values of one object that fold to the same string.
  1. parse_answer digit-stripping bug. Files with a raw model response (v2 local runs, CoT
     runs) are audited directly (rule (e)). No released v1 file stores the raw response: in
     results/full_*_extracted.jsonl ``prediction_raw`` is the parsed prediction before answer
     extraction (identical to ``prediction`` for GPT-4o / Gemini / Qwen; Claude keeps 237
     verbose raw strings); full_gemini25 / full_qwen32b / full_internvl have no raw field. For
     v1 we therefore count the bug SIGNATURE (prediction is a digit run d, key non-numeric, d a
     strict substring of a key token, e.g. pred "10" for key "10B"), give its EM effect as an
     upper bound, and estimate its precision on the v2 local runs, where raw responses exist
     (share of signature rows that rule (e) confirms).
  2. v1, seven models: per-format EM, within-modality best-worst gaps (best/worst chosen on the
     sample, object-cluster bootstrap CI, object sign-flip p; Holm over the 12 core contrasts,
     cluster_cis.py conventions), flip rates (answerability-agnostic, appendix_tables.py
     definition), EM on answerable (V/D) / absent (A) / constant (C) pairs with the CV prior on
     the same pairs, per-format EM on answerable pairs, best format vs CV prior, and the tabular
     bar_chart value_extraction cell ("1.7% for all four core models").
  3. v2 multi-model: v2_multimodel.analyse under each score (SCORER hook).
  4. CoT: v1 prompt ablation (no raw: rules (a)-(d)), the partial v1 re-run (raw = text after
     the answer marker), and the v2 CoT sample (cot_audit / cot_v2_audit definitions). For CoT
     rows the raw text given to rule (e) is ``cot_final_answer_text`` (the answer line), not the
     whole reasoning, so a key mentioned mid-reasoning cannot be matched.
All statistics: random.Random(0), B_boot=5000, B_perm=10000, cluster = data_id.

Writes scripts/analysis/canonical_results.json and paper_tables/tab_strict_vs_canonical.tex.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import review_common as rc  # noqa: E402
import scoring as sc  # noqa: E402
import v2_analysis as v2a  # noqa: E402
import v2_multimodel as mm  # noqa: E402

MODS = rc.MODS
FORMATS = rc.FORMATS
SCORES = ("strict", "canonical")


def stored(r):
    return float(r.get("exact_match", 0) or 0)


def canon_noraw(r):
    return sc.canonical_em(r.get("prediction", ""), None, r["answer"], r.get("task"),
                           stored=stored(r))


def canon_raw(r):
    return sc.row_score(r, "raw_response")


def canon_v1(r):
    """v1 rows: raw only for the Claude rows whose prediction_raw is a verbose response."""
    raw = r.get("prediction_raw")
    if raw is not None and str(raw) == str(r.get("prediction")):
        raw = None
    return sc.canonical_em(r.get("prediction", ""), raw, r["answer"], r.get("task"),
                           stored=stored(r))


def signature(pred, key):
    d = str(pred).strip()
    if not re.fullmatch(r"\d+", d) or sc.is_numeric_key(key):
        return False
    return any(d in t and t != d for t in re.findall(r"[A-Za-z0-9]+", str(key)))


# ---------------------------------------------------------------- statistics
def ci(items, q2d, perm=False):
    if not items:
        return None
    r = v2a.contrast(items, q2d, perm=perm)
    o = {"est": r["est"], "lo": r["obj_ci"]["lo"], "hi": r["obj_ci"]["hi"], "n": r["n_units"],
         "n_obj": r["n_obj"]}
    if perm:
        o["p"] = r["obj_perm_p"]
    return o


def holm(rows):
    ps = sorted(range(len(rows)), key=lambda i: rows[i]["p"])
    m, running = len(rows), 0.0
    for rank, i in enumerate(ps):
        running = max(running, min(1.0, (m - rank) * rows[i]["p"]))
        rows[i]["p_holm"] = running
    return rows


def em_tables(path, scorer):
    rows, _ = v2a.dedupe(v2a.read_jsonl(os.path.join(ROOT, path)))
    raw_rows = v2a.read_jsonl(os.path.join(ROOT, path))
    for r in raw_rows:  # v1 files: keep error-only keys as EM 0 (released scoring)
        k = (r["question_id"], r["viz_type"])
        if k not in rows:
            rows[k] = r
    EM = defaultdict(dict)
    for (q, f), r in rows.items():
        EM[q][f] = scorer(r)
    return EM, rows


# ---------------------------------------------------------------- section 1
def bug_section(meta):
    out = {"v2_raw_audit": {}, "v1_signature": {}, "v1_files_without_raw": [],
           "notes": "v1 prediction_raw is the pre-extraction parsed prediction, not the model "
                    "response; only 237 Claude rows hold verbose raw text."}
    prec_num = prec_den = 0
    for spec in mm.MODELS:
        rows, _ = v2a.dedupe(v2a.read_jsonl(os.path.join(ROOT, spec["v2"])))
        by, n_sig, n_conf = Counter(), 0, 0
        for (q, f), r in rows.items():
            if f not in FORMATS[meta[q]["modality"]]:
                continue
            raw = r.get("raw_response")
            raw = None if raw in (None, "None", "") else raw
            s0 = stored(r)
            fired = s0 < 1 and sc._raw_fallback(str(r.get("prediction", "")), raw,
                                                str(r["answer"]))
            if fired:
                by[f"{meta[q]['modality']}|{meta[q]['task']}"] += 1
            if s0 < 1 and signature(r.get("prediction", ""), r["answer"]):
                n_sig += 1
                n_conf += fired
        out["v2_raw_audit"][spec["name"]] = {"rows_fixed_by_raw": sum(by.values()),
                                             "by_modality_task": dict(by),
                                             "signature_rows": n_sig,
                                             "signature_confirmed_by_raw": n_conf}
        if spec["name"].startswith("Qwen"):
            prec_num += n_conf
            prec_den += n_sig
    precision = prec_num / prec_den if prec_den else float("nan")
    out["signature_precision_qwen_v2"] = {"confirmed": prec_num, "signature": prec_den,
                                          "precision": precision}
    for m, fn in rc.MODELS7.items():
        rows = rc.load_v1(m)
        r0 = next(iter(rows.values()))
        has_raw = "prediction_raw" in r0 and any(
            str(r.get("prediction_raw")) != str(r.get("prediction")) for r in rows.values())
        if not has_raw:
            out["v1_files_without_raw"].append(fn)
        by, n_q = Counter(), Counter()
        for (q, f), r in rows.items():
            n_q[meta[q]["modality"]] += 1
            if stored(r) < 1 and signature(r.get("prediction", ""), r["answer"]):
                by[(meta[q]["modality"], meta[q]["task"], f)] += 1
        agg = defaultdict(Counter)
        for (md, t, f), v in by.items():
            agg[md][t] += v
        out["v1_signature"][m] = {
            "file": fn, "raw_available": "partial (237 verbose rows)" if has_raw else "no",
            "rows": sum(by.values()),
            "by_modality_task": {md: dict(c) for md, c in agg.items()},
            "upper_bound_em_change_pp_by_modality": {md: 100 * sum(c.values()) / n_q[md]
                                                     for md, c in agg.items()},
            "expected_em_change_pp_by_modality": {md: 100 * precision * sum(c.values()) / n_q[md]
                                                  for md, c in agg.items()},
            "by_format": dict(Counter(f"{md}|{f}" for (md, t, f), v in by.items()
                                      for _ in range(v)))}
    return out


# ---------------------------------------------------------------- section 2
def v1_section(meta, q2d, cat1, prior_q, prior_const):
    out = {}
    for score in SCORES:
        res = {"models": {}}
        gaps12 = []
        for m, fn in rc.MODELS7.items():
            scorer = stored if score == "strict" else canon_v1
            EM, _ = em_tables(os.path.join("results", fn), scorer)
            rec = {"per_format": {}, "gaps": {}, "flip": {}, "by_class": {}, "vd_per_format": {},
                   "best_vs_prior": {}}
            allflip = []
            for md in MODS:
                qs = [q for q in meta if meta[q]["modality"] == md]
                fm = FORMATS[md]
                pf = {f: rc.mean100(EM[q][f] for q in qs) for f in fm}
                rec["per_format"][md] = pf
                b, w = max(pf, key=pf.get), min(pf, key=pf.get)
                g = ci([(q, EM[q][b] - EM[q][w]) for q in qs], q2d, perm=m in rc.CORE4)
                g.update(best=b, worst=w)
                rec["gaps"][md] = g
                if m in rc.CORE4:
                    gaps12.append({"model": m, "modality": md, **g})
                fl = [float(len({EM[q][f] for f in fm}) > 1) for q in qs]
                allflip += fl
                rec["flip"][md] = rc.mean100(fl)
                cls = {}
                for c in ("VD", "A", "C"):
                    items = [(q, f) for q in qs for f in fm if cat1[q][f] in c]
                    if not items:
                        continue
                    d = ci([(q, EM[q][f] - prior_q[q]) for q, f in items], q2d)
                    cls[c] = {"em": rc.mean100(EM[q][f] for q, f in items),
                              "prior": rc.mean100(prior_q[q] for q, f in items),
                              "em_minus_prior": d, "n": len(items)}
                rec["by_class"][md] = cls
                rec["vd_per_format"][md] = {
                    f: {"em": rc.mean100(EM[q][f] for q in qs if cat1[q][f] in "VD"),
                        "n": sum(cat1[q][f] in "VD" for q in qs)} for f in fm}
                bb = ci([(q, EM[q][b]) for q in qs], q2d)
                rec["best_vs_prior"][md] = {"best": b, "em": bb["est"],
                                            "diff": bb["est"] - prior_const[md],
                                            "lo": bb["lo"] - prior_const[md],
                                            "hi": bb["hi"] - prior_const[md]}
            rec["flip"]["all"] = rc.mean100(allflip)
            vq = [q for q in meta if meta[q]["task"] == "value_extraction"
                  and meta[q]["modality"] == "tabular"]
            rec["bar_value_extraction"] = rc.mean100(EM[q]["bar_chart"] for q in vq)
            res["models"][m] = rec
        holm(gaps12)
        res["holm12"] = gaps12
        res["n_holm_sig_05"] = sum(g["p_holm"] < 0.05 for g in gaps12)
        res["n_holm_sig_01"] = sum(g["p_holm"] < 0.01 for g in gaps12)
        res["max_gap_core4"] = max(gaps12, key=lambda g: g["est"])
        allg = [(m, md, r["gaps"][md]["est"]) for m, r in res["models"].items() for md in MODS]
        res["max_gap_all7"] = max(allg, key=lambda t: t[2])
        out[score] = res
    return out


# ---------------------------------------------------------------- section 4
def cot_v1_ablation(meta, q2d, cs):
    out = {}
    for m, key in rc.PROMPT_KEYS.items():
        data = {}
        for s in rc.STYLES:
            p = os.path.join(rc.RES, "ablation", f"prompt_{key}_{s}.jsonl")
            data[s] = {(r["question_id"], r["viz_type"]): r for r in rc.read_jsonl(p)}
        keys = sorted(data["minimal"])

        def rt(r):
            b = meta[r["question_id"]]
            return rc.right_type(r.get("prediction", ""), b["answer"],
                                 cs[(b["modality"], b["task"])])
        rec = {}
        for score in SCORES:
            f = stored if score == "strict" else canon_noraw
            em = {s: {k: f(data[s][k]) for k in keys} for s in rc.STYLES}
            style_em = {s: rc.mean100(em[s].values()) for s in rc.STYLES}
            best = max(("minimal", "concise", "detailed"), key=style_em.get)
            both = [k for k in keys if rt(data["cot"][k]) and rt(data[best][k])]
            rec[score] = {"style_em": style_em, "best": best,
                          "d_all": ci([(k[0], em["cot"][k] - em[best][k]) for k in keys], q2d),
                          "d_both_rt": ci([(k[0], em["cot"][k] - em[best][k]) for k in both],
                                          q2d)}
        out[m] = rec
    return out


def _share_ci(rows, q2d, B=5000, seed=0):
    """Ratio of sums sum(num)/sum(den) with an object-cluster percentile bootstrap.

    rows: (question_id, num, den). Returns est and 95% CI (cluster = data_id)."""
    import random
    by = defaultdict(lambda: [0.0, 0.0])
    for q, a, b in rows:
        by[q2d[q]][0] += a
        by[q2d[q]][1] += b
    objs = sorted(by)
    tot = [sum(by[o][0] for o in objs), sum(by[o][1] for o in objs)]
    rng = random.Random(seed)
    bs = []
    for _ in range(B):
        a = b = 0.0
        for _ in objs:
            o = objs[rng.randrange(len(objs))]
            a += by[o][0]
            b += by[o][1]
        if b:
            bs.append(a / b)
    bs.sort()
    return {"est": 100 * tot[0] / tot[1] if tot[1] else None,
            "lo": 100 * bs[int(0.025 * len(bs))], "hi": 100 * bs[int(0.975 * len(bs)) - 1],
            "n_obj": len(objs)}


def cot_v1_rerun(meta, q2d):
    """Qwen2.5-VL-7B v1 CoT re-run with raw responses (results/review/cot_v1_qwen.jsonl).

    Conditions on the same (question, format) rows:
      new        re-run, max_new_tokens=1024, final-answer parser (cot_final_answer_text)
      legacy     re-run, 1024 tokens, the original extractor (src/models/response_parser)
      trunc256   re-run, raw response truncated to 256 tokens, original extractor: replica of
                 the paper's ablation pipeline (local_models.py max_tokens=256 default)
      original   the paper's CoT cell (results/ablation/prompt_qwen_cot.jsonl)
      concise    Qwen's best non-CoT style (prompt_qwen_concise.jsonl)
    Pipeline artefact = new - trunc256 = (legacy - trunc256: token budget)
                                       + (new - legacy: extractor).
    Share of the deficit explained = (new - trunc256) / (concise - trunc256), with the re-run's
    raw responses as ground truth; also against the paper's own cell (concise - original).
    """
    path = os.path.join(rc.ROOT, "results", "review", "cot_v1_qwen.jsonl")
    if not os.path.exists(path):
        return {"n": 0}
    new = {(r["question_id"], r["viz_type"]): r for r in rc.read_jsonl(path)}

    def keyed(name):
        return {(r["question_id"], r["viz_type"]): r for r in rc.read_jsonl(
            os.path.join(rc.RES, "ablation", name))}
    conc, orig = keyed("prompt_qwen_concise.jsonl"), keyed("prompt_qwen_cot.jsonl")
    ks = sorted(k for k in new if k in conc and k in orig)
    n_man = sum(1 for _ in open(os.path.join(rc.ROOT, "results", "review",
                                             "cot_v1_manifest.jsonl")))
    out = {"n": len(ks), "n_manifest": n_man, "partial": len(ks) < n_man,
           "n_questions": len({k[0] for k in ks}),
           "n_gen_tokens_gt256": sum(int(new[k]["n_gen_tokens"] or 0) > 256 for k in ks),
           "hit_max_new_tokens": sum(new[k]["hit_max_new_tokens"] in (True, "True")
                                     for k in ks),
           "marker_found": sum(new[k]["cot_marker_found"] in (True, "True") for k in ks),
           "trunc256_pred_equals_original": rc.mean100(
               rc.norm(new[k]["prediction_legacy_trunc256"]) == rc.norm(orig[k]["prediction"])
               for k in ks),
           "legacy_pred_equals_original": rc.mean100(
               rc.norm(new[k]["prediction_legacy"]) == rc.norm(orig[k]["prediction"])
               for k in ks)}
    groups = {"all": ks}
    for k in ks:
        b = meta[k[0]]
        groups.setdefault(f"depth|{b['difficulty']}", []).append(k)
        groups.setdefault(f"modality|{b['modality']}", []).append(k)
    for score in SCORES:
        def fnew(r):
            if score == "strict":
                return stored(r)
            raw = r.get("cot_final_answer_text")
            return sc.canonical_em(r["prediction"], None if raw in (None, "None") else raw,
                                   r["answer"], r["task"], stored=stored(r))

        def fold(field, emf):
            def f(r):
                s0 = float(r[emf])
                return s0 if score == "strict" else sc.canonical_em(
                    r[field], None, r["answer"], r["task"], stored=s0)
            return f
        fleg = fold("prediction_legacy", "exact_match_legacy")
        ftr = fold("prediction_legacy_trunc256", "exact_match_legacy_trunc256")
        fc = stored if score == "strict" else canon_noraw
        em = {"new": {k: fnew(new[k]) for k in ks}, "legacy": {k: fleg(new[k]) for k in ks},
              "trunc256": {k: ftr(new[k]) for k in ks}, "original": {k: fc(orig[k]) for k in ks},
              "concise": {k: fc(conc[k]) for k in ks}}
        rec = {}
        for g, gk in groups.items():
            def d(a, b):
                return ci([(k[0], em[a][k] - em[b][k]) for k in gk], q2d)
            r = {"n": len(gk), "n_q": len({k[0] for k in gk})}
            for c in em:
                r[f"em_{c}"] = rc.mean100(em[c][k] for k in gk)
            for a, b in (("new", "concise"), ("new", "legacy"), ("legacy", "trunc256"),
                         ("new", "trunc256"), ("trunc256", "concise"), ("original", "concise"),
                         ("new", "original"), ("trunc256", "original")):
                r[f"{a}_minus_{b}"] = d(a, b)
            r["share_explained_replica"] = _share_ci(
                [(k[0], em["new"][k] - em["trunc256"][k], em["concise"][k] - em["trunc256"][k])
                 for k in gk], q2d)
            r["share_explained_paper_cell"] = _share_ci(
                [(k[0], em["new"][k] - em["original"][k], em["concise"][k] - em["original"][k])
                 for k in gk], q2d)
            rec[g] = r
        # back-compatible top-level fields
        a = rec["all"]
        rec.update({"em_new": a["em_new"], "em_legacy": a["em_legacy"],
                    "em_concise": a["em_concise"], "new_minus_concise": a["new_minus_concise"],
                    "new_minus_legacy": a["new_minus_legacy"]})
        out[score] = rec
    return out


def cot_v2(meta, q2d, cs, nc):
    R = os.path.join(rc.ROOT, "results", "review")
    cot = {(r["question_id"], r["viz_type"]): r for r in rc.read_jsonl(
        os.path.join(R, "cot_v2_qwen.jsonl"))}
    sysd = {(r["question_id"], r["viz_type"]): r for r in rc.read_jsonl(
        os.path.join(R, "cot_v2_qwen_sysprompt_only.jsonl"))}
    plain, _ = v2a.dedupe(v2a.read_jsonl(os.path.join(rc.ROOT, "results", "v2", "v2_qwen.jsonl")))
    main = [k for k in sorted(cot) if k[1] in FORMATS[meta[k[0]]["modality"]]]

    def final_raw(r):
        raw = r.get("cot_final_answer_text")
        return None if raw in (None, "None", "") else raw

    out = {"n": len(main), "n_q": len({k[0] for k in main})}
    for score in SCORES:
        if score == "strict":
            f = {"plain": lambda k: stored(plain[k]), "cot_new": lambda k: stored(cot[k]),
                 "cot_legacy": lambda k: float(cot[k]["exact_match_legacy"]),
                 "sys_only": lambda k: stored(sysd[k])}
        else:
            f = {"plain": lambda k: canon_raw(plain[k]),
                 "cot_new": lambda k: sc.canonical_em(cot[k]["prediction"], final_raw(cot[k]),
                                                      cot[k]["answer"], None, stored(cot[k])),
                 "cot_legacy": lambda k: sc.canonical_em(
                     cot[k]["prediction_legacy"], final_raw(cot[k]), cot[k]["answer"], None,
                     float(cot[k]["exact_match_legacy"])),
                 "sys_only": lambda k: canon_raw(sysd[k])}
        rec = {}
        for g in rc.DEPTHS + MODS + ["overall"]:
            ks = [k for k in main if g == "overall" or meta[k[0]]["difficulty"] == g
                  or meta[k[0]]["modality"] == g]
            if not ks:
                continue
            rec[g] = {c: rc.mean100(fn(k) for k in ks) for c, fn in f.items()}
            rec[g]["n"] = len(ks)
            for c in ("cot_new", "cot_legacy", "sys_only"):
                rec[g][f"{c}_minus_plain"] = ci([(k[0], f[c](k) - f["plain"](k)) for k in ks],
                                                q2d, perm=c == "cot_new")
        ksx = [k for k in main if (meta[k[0]]["modality"], meta[k[0]]["task"]) not in nc]
        rec["overall_exclNC"] = {"cot_new_minus_plain": ci(
            [(k[0], f["cot_new"](k) - f["plain"](k)) for k in ksx], q2d), "n": len(ksx)}
        out[score] = rec
    return out


# ---------------------------------------------------------------- main
def fmt_ci(x, sign=True):
    if not x:
        return "--"
    s = "{:+.1f}" if sign else "{:.1f}"
    return f"{s.format(x['est'])} [{s.format(x['lo'])},{s.format(x['hi'])}]"


def write_diff_table(res, path):
    """paper_tables/tab_strict_vs_canonical.tex from a canonical_results.json dictionary."""
    V1, V2, C1, C2 = res["v1"], res["v2"], res["cot_v1_ablation"], res["cot_v2"]
    # ---------------------------------------------------------------- LaTeX diff table
    rows = []
    S, Cn = V1["strict"], V1["canonical"]
    rows.append(["v1", "Max best-worst gap, core 4 (pp)",
                 f"{S['max_gap_core4']['est']:.1f} ({S['max_gap_core4']['model']}, "
                 f"{S['max_gap_core4']['modality']})",
                 f"{Cn['max_gap_core4']['est']:.1f} ({Cn['max_gap_core4']['model']}, "
                 f"{Cn['max_gap_core4']['modality']})"])
    rows.append(["v1", "Holm-significant core contrasts (p<.05 / .01)",
                 f"{S['n_holm_sig_05']}/12 / {S['n_holm_sig_01']}/12",
                 f"{Cn['n_holm_sig_05']}/12 / {Cn['n_holm_sig_01']}/12"])
    fa = sorted(S["models"][m]["flip"]["all"] for m in rc.MODELS7)
    fb = sorted(Cn["models"][m]["flip"]["all"] for m in rc.MODELS7)
    rows.append(["v1", "Flip rate, seven models (min-max, %)", f"{fa[0]:.1f}-{fa[-1]:.1f}",
                 f"{fb[0]:.1f}-{fb[-1]:.1f}"])
    for m in rc.CORE4:
        for md in MODS:
            ga, gb = S["models"][m]["gaps"][md], Cn["models"][m]["gaps"][md]
            rows.append(["v1", f"Gap {m} {md}", f"{ga['est']:.1f} ({ga['best'][:5]}/{ga['worst'][:5]})",
                         f"{gb['est']:.1f} ({gb['best'][:5]}/{gb['worst'][:5]})"])
    for md in MODS:
        da = [S["models"][m]["by_class"][md]["A"]["em_minus_prior"]["est"] for m in rc.MODELS7]
        db = [Cn["models"][m]["by_class"][md]["A"]["em_minus_prior"]["est"] for m in rc.MODELS7]
        rows.append(["v1", f"Absent pairs EM minus prior, {md} (range, pp)",
                     f"{min(da):+.1f} to {max(da):+.1f}", f"{min(db):+.1f} to {max(db):+.1f}"])
    rows.append(["v1", "Bar chart, value extraction (core 4 mean, %)",
                 f"{sum(S['models'][m]['bar_value_extraction'] for m in rc.CORE4) / 4:.1f}",
                 f"{sum(Cn['models'][m]['bar_value_extraction'] for m in rc.CORE4) / 4:.1f}"])
    for spec in mm.MODELS:
        m = spec["name"]
        for md, lab in (("tabular", "tab."), ("graph", "graph")):
            ga = V2["strict"][m]["gaps"][md]["all_answerable_exclNC"]
            gb = V2["canonical"][m]["gaps"][md]["all_answerable_exclNC"]
            rows.append(["v2", f"Residual {lab} gap, {m}" + (" (subset)" if spec["subset"] else ""),
                         f"{ga['est']:.1f} [{ga['lo']:.1f},{ga['hi']:.1f}]",
                         f"{gb['est']:.1f} [{gb['lo']:.1f},{gb['hi']:.1f}]"])
        if not spec["subset"]:
            ga = V2["strict"][m]["gaps"]["timeseries"]["nonlossy3_all_exclNC"]
            gb = V2["canonical"][m]["gaps"]["timeseries"]["nonlossy3_all_exclNC"]
            rows.append(["v2", f"Series 3-format gap, {m}", f"{ga['est']:.1f}", f"{gb['est']:.1f}"])
    for m, r in C1.items():
        rows.append(["CoT v1", f"CoT minus best plain, {m}", fmt_ci(r["strict"]["d_all"]),
                     fmt_ci(r["canonical"]["d_all"])])
    rows.append(["CoT v2", "CoT minus plain, Qwen-7B, all", fmt_ci(C2["strict"]["overall"][
        "cot_new_minus_plain"]), fmt_ci(C2["canonical"]["overall"]["cot_new_minus_plain"])])
    rc.write_tabular(
        path,
        "Headline numbers under strict exact match vs canonical_em (scripts/analysis/scoring.py:\n"
        "existing normalisation; index-prefix strip, polarity/trend lexicon, case/underscore/edge-\n"
        "punctuation folding for non-numeric keys; raw-response fallback for the parser's digit\n"
        "stripping where a raw response exists). Numeric keys are scored identically. Gaps in pp\n"
        "(best/worst format); v2 residual gaps on questions answerable in every format outside\n"
        "near-constant templates, object-cluster 95% CI.",
        "l l l l", ["Suite", "Quantity", "Strict", "Canonical"], rows)


def main():
    meta = rc.load_bench()
    q2d = {q: b["data_id"] for q, b in meta.items()}
    cs = rc.cell_stats(meta)
    nc = rc.near_constant()
    cat1 = rc.v1_cats(meta)
    cat2s = rc.v2_cats(meta, strict=True)
    prior_q = rc.cv_prior_per_question(meta)
    pr = v2a.load_prior()
    prior_const = {md: pr[md]["cv_mean"] for md in MODS}
    res = {"definition": __doc__, "scorer": sc.__doc__}

    # 0. folding safety
    coll = []
    seen_obj = set()
    for b in meta.values():
        if b["data_id"] in seen_obj or b["modality"] != "tabular":
            continue
        seen_obj.add(b["data_id"])
        vals = {v for row in b["data"] for v in row.values()} | {c for row in b["data"]
                                                                  for c in row}
        coll += [(b["data_id"], a, c) for a, c in sc.object_fold_collisions(sorted(map(str,
                                                                                        vals)))]
    res["fold_collisions"] = coll
    # exposure: canonical 0->1 flips whose prediction is exactly another value of the object
    coll_vals = {(d, x.lower()) for d, a, c in coll for x in (a, c)}
    expo = Counter()
    for m, fn in rc.MODELS7.items():
        for (q, f), r in rc.load_v1(m).items():
            b = meta[q]
            if stored(r) < 1 and canon_v1(r) >= 1 and (
                    b["data_id"], str(r["prediction"]).strip().lower()) in coll_vals \
                    and str(r["prediction"]).strip().lower() != str(b["answer"]).strip().lower():
                expo[m] += 1
    res["fold_collision_exposure_v1"] = dict(expo)
    print(f"    v1 canonical flips whose prediction is a colliding value of the object: "
          f"{dict(expo)}")
    print(f"[0] folding collisions inside tabular objects (distinct values, same fold): "
          f"{len(coll)} {coll[:6]}")

    # 1. bug
    B = bug_section(meta)
    res["parse_bug"] = B
    print("\n[1] parse_answer digit-stripping bug")
    for m, v in B["v2_raw_audit"].items():
        print(f"  v2 {m:26s} fixed by raw {v['rows_fixed_by_raw']:4d}  signature rows "
              f"{v['signature_rows']:4d} (confirmed {v['signature_confirmed_by_raw']}) "
              f"{dict(sorted(v['by_modality_task'].items(), key=lambda kv: -kv[1])[:4])}")
    p = B["signature_precision_qwen_v2"]
    print(f"  signature precision on Qwen v2 runs: {p['confirmed']}/{p['signature']} = "
          f"{p['precision']:.2f}")
    print(f"  v1 files without any raw response: {B['v1_files_without_raw']}")
    for m, v in B["v1_signature"].items():
        print(f"  v1 {m:18s} raw={v['raw_available']:26s} signature rows {v['rows']:4d}; "
              f"upper-bound EM change pp {{{', '.join(f'{k}: {x:.2f}' for k, x in v['upper_bound_em_change_pp_by_modality'].items())}}}; "
              f"tasks {v['by_modality_task']}")

    # 2. v1
    V1 = v1_section(meta, q2d, cat1, prior_q, prior_const)
    res["v1"] = V1
    print("\n[2] v1 seven models, strict -> canonical")
    for m in rc.MODELS7:
        a, b = V1["strict"]["models"][m], V1["canonical"]["models"][m]
        for md in MODS:
            print(f"  {m:17s} {md:10s} " + " ".join(
                f"{f[:5]} {a['per_format'][md][f]:.1f}->{b['per_format'][md][f]:.1f}"
                for f in FORMATS[md])
                + f" | gap {a['gaps'][md]['est']:.1f}->{b['gaps'][md]['est']:.1f} "
                f"({b['gaps'][md]['best'][:5]}/{b['gaps'][md]['worst'][:5]}) | flip "
                f"{a['flip'][md]:.1f}->{b['flip'][md]:.1f}")
        print(f"  {m:17s} flip all {a['flip']['all']:.1f}->{b['flip']['all']:.1f}; bar "
              f"value_extraction {a['bar_value_extraction']:.1f}->{b['bar_value_extraction']:.1f}")
        for md in MODS:
            ca, cb = a["by_class"][md], b["by_class"][md]
            print(f"      {md:10s} " + "  ".join(
                f"{c}: EM {ca[c]['em']:.1f}->{cb[c]['em']:.1f} prior {cb[c]['prior']:.1f} "
                f"EM-prior {fmt_ci(ca[c]['em_minus_prior'])}->{fmt_ci(cb[c]['em_minus_prior'])}"
                for c in ca))
    for score in SCORES:
        v = V1[score]
        print(f"  [{score}] Holm-significant 12 core contrasts: {v['n_holm_sig_05']}/12 at .05, "
              f"{v['n_holm_sig_01']}/12 at .01; max gap core4 {v['max_gap_core4']['est']:.1f} "
              f"({v['max_gap_core4']['model']} {v['max_gap_core4']['modality']}); all7 "
              f"{v['max_gap_all7']}")
        for g in v["holm12"]:
            print(f"      {g['model']:17s} {g['modality']:10s} {g['best']:>15s}/{g['worst']:<15s}"
                  f" {fmt_ci(g, False)} p {g['p']:.4f} holm {g['p_holm']:.4f}")
    # answerable-pairs best tabular format
    for score in SCORES:
        bests = {m: max(V1[score]["models"][m]["vd_per_format"]["tabular"].items(),
                        key=lambda kv: kv[1]["em"]) for m in rc.MODELS7}
        print(f"  [{score}] best tabular format on answerable pairs: "
              + "; ".join(f"{m} {f} {v['em']:.1f} (n={v['n']})" for m, (f, v) in bests.items()))
        res.setdefault("vd_best_tabular", {})[score] = {m: [f, v] for m, (f, v) in bests.items()}
        print(f"  [{score}] answerable-pair EM, tabular bar_chart / scatter_plot / table_image: "
              + "; ".join(f"{m} " + "/".join(
                  f"{V1[score]['models'][m]['vd_per_format']['tabular'][f]['em']:.1f}"
                  for f in ("bar_chart", "scatter_plot", "table_image")) for m in rc.MODELS7)
              + " (n = " + "/".join(str(V1[score]['models']['GPT-4o']['vd_per_format']['tabular'][f]['n'])
                                    for f in ("bar_chart", "scatter_plot", "table_image")) + ")")

    # 3. v2
    print("\n[3] v2 multi-model, strict -> canonical")
    V2 = {}
    for score in SCORES:
        mm.SCORER = None if score == "strict" else canon_raw
        V2[score] = {}
        for spec in mm.MODELS:
            o = mm.analyse(spec, meta, q2d, cat2s, nc, prior_const, prior_q)
            o.pop("_present", None)
            V2[score][spec["name"]] = o
    mm.SCORER = None
    res["v2"] = V2
    for spec in mm.MODELS:
        m = spec["name"]
        a, b = V2["strict"][m], V2["canonical"][m]

        def gl(o, md, k="all_answerable_exclNC"):
            g = o["gaps"][md][k]
            return (f"{g['est']:.1f} [{g['lo']:.1f},{g['hi']:.1f}] p={g['p']:.3f} "
                    f"{g['best'][:5]}/{g['worst'][:5]}") if g.get("n_q") else "--"
        print(f"  {m:26s} tab {gl(a, 'tabular')} -> {gl(b, 'tabular')}")
        print(f"  {'':26s} graph {gl(a, 'graph')} -> {gl(b, 'graph')}")
        print(f"  {'':26s} series {gl(a, 'timeseries', 'nonlossy3_all_exclNC')} -> "
              f"{gl(b, 'timeseries', 'nonlossy3_all_exclNC')}")
        k = "diff_same_q" if spec["subset"] else "diff"
        print(f"  {'':26s} best-prior " + "  ".join(
            f"{md} {a['best_vs_prior'][md][k]:+.1f}->{b['best_vs_prior'][md][k]:+.1f}"
            for md in MODS) + " | beating prior " + " ".join(
            f"{md} {sum(v['exceeds_same_q' if spec['subset'] else 'exceeds'] for v in a['formats_vs_prior'][md].values())}->"
            f"{sum(v['exceeds_same_q' if spec['subset'] else 'exceeds'] for v in b['formats_vs_prior'][md].values())}"
            for md in MODS))
        da, db = a["degree_effect"], b["degree_effect"]
        if da["degree_query"].get("n"):
            print(f"  {'':26s} degree {da['degree_query']['est']:+.1f}/{da['other_tasks']['est']:+.1f}"
                  f" -> {db['degree_query']['est']:+.1f}/{db['other_tasks']['est']:+.1f}")
        if "recovery" in a:
            print(f"  {'':26s} recovery " + " ".join(
                f"{kk.split('|')[1][:5]}({kk[0]}) {a['recovery'][kk]['d']:+.1f}->{b['recovery'][kk]['d']:+.1f}"
                for kk in a["recovery"]))

    # 4. CoT
    C1 = cot_v1_ablation(meta, q2d, cs)
    C1r = cot_v1_rerun(meta, q2d)
    C2 = cot_v2(meta, q2d, cs, nc)
    res["cot_v1_ablation"], res["cot_v1_rerun"], res["cot_v2"] = C1, C1r, C2
    print("\n[4] CoT")
    for m, r in C1.items():
        a, b = r["strict"], r["canonical"]
        print(f"  v1 ablation {m:17s} styles " + " ".join(
            f"{s} {a['style_em'][s]:.1f}->{b['style_em'][s]:.1f}" for s in rc.STYLES)
            + f" | CoT-best({a['best']}->{b['best']}) {fmt_ci(a['d_all'])} -> {fmt_ci(b['d_all'])}"
            f" | right-type {fmt_ci(a['d_both_rt'])} -> {fmt_ci(b['d_both_rt'])}")
    if C1r.get("n"):
        a, b = C1r["strict"], C1r["canonical"]
        print(f"  v1 re-run ({C1r['n']}/{C1r['n_manifest']} rows): new {a['em_new']:.1f}->"
              f"{b['em_new']:.1f}, legacy {a['em_legacy']:.1f}->{b['em_legacy']:.1f}, concise "
              f"{a['em_concise']:.1f}->{b['em_concise']:.1f}; new-concise "
              f"{fmt_ci(a['new_minus_concise'])} -> {fmt_ci(b['new_minus_concise'])}")
    for g in rc.DEPTHS + MODS + ["overall"]:
        a, b = C2["strict"].get(g), C2["canonical"].get(g)
        if not a:
            continue
        print(f"  v2 CoT {g:14s} n={a['n']:4d} plain {a['plain']:.1f}->{b['plain']:.1f} cot "
              f"{a['cot_new']:.1f}->{b['cot_new']:.1f} | cot-plain {fmt_ci(a['cot_new_minus_plain'])}"
              f" -> {fmt_ci(b['cot_new_minus_plain'])} | legacy-plain "
              f"{fmt_ci(a['cot_legacy_minus_plain'])} -> {fmt_ci(b['cot_legacy_minus_plain'])}")

    import canonical_extras  # selector, raw-text, no-image, assist, answerable, scale, errors
    canonical_extras.add_extras(res)
    with open(os.path.join(HERE, "canonical_results.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)

    write_diff_table(res, os.path.join(ROOT, "paper_tables", "tab_strict_vs_canonical.tex"))
    print("\nwrote scripts/analysis/canonical_results.json, paper_tables/tab_strict_vs_canonical.tex")


if __name__ == "__main__":
    main()
