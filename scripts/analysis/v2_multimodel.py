#!/usr/bin/env python3
"""Expressive (v2) suite, multi-model: the paper's Qwen2.5-VL-7B v2 statistics for any
prediction file.

Stdlib only, no inference.

    python scripts/analysis/v2_multimodel.py

Definitions are the ones v2_analysis.py uses for the paper (run with --strict-answerable
--exclude-near-constant), computed with v2_analysis's own functions (classify_v2 +
strictify_v2, dedupe, gap_block, contrast; B_boot=5000, B_perm=10000, random.Random(0),
object = data_id):
  per-format EM     per modality, on every analysed question with that format.
  tab / graph gap   s3b "v2_all_answerable_exclNC": questions answerable (V/D, strict
                    categories) in every main format, near-constant cells excluded; best and
                    worst format chosen on that sample; object-cluster CI and sign-flip p.
  series gap        s3b "v2_nonlossy3_all_exclNC": line_plot / heatmap / text_only on all
                    non-near-constant time-series questions (what the paper's 21.4 is); the
                    answerable-only variant is also reported.
  best vs prior     s5b: EM of the modality's best format (highest EM on the analysed
                    questions) minus the cross-validated question-only prior (prior_cv mean,
                    prior_and_subsets_results.json, a constant); CI = EM object CI shifted.
  degree effect     s7: text_only_deg minus text_only, paired, on degree_query vs other graph
                    tasks.
  recovery          s2: paired v2 - v1 per format on the analysed questions, where a v1 file
                    exists (v1 error-only keys kept as EM 0, as in the released scoring).
Subset models (InternVL family, 1,096-question strict subset, manifest_subset32b) are
analysed on the questions present in their file only; for them the prior is additionally
given as the per-question CV prior averaged over the subset questions (review_common), since
the full-benchmark constant describes a different question mix.

Writes scripts/analysis/v2_multimodel_results.json and paper_tables/tab_v2_multimodel.tex.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import review_common as rc  # noqa: E402
import v2_analysis as v2a  # noqa: E402

MODELS = [
    {"name": "Qwen2.5-VL-7B", "v2": "results/v2/v2_qwen.jsonl",
     "v1": "results/full_qwen_extracted.jsonl", "subset": False},
    {"name": "Qwen2.5-VL-32B", "v2": "results/review/v2_qwen32b.jsonl",
     "v1": "results/full_qwen32b.jsonl", "subset": False},
    {"name": "Qwen3-VL-8B", "v2": "results/review/v2_qwen3vl8b.jsonl", "v1": None,
     "subset": False},
    {"name": "InternVL2-4B", "v2": "results/review/v2sub_internvl2_4b.jsonl", "v1": None,
     "subset": True},
    {"name": "InternVL2-8B", "v2": "results/review/v2sub_internvl2_8b.jsonl", "v1": None,
     "subset": True},
    {"name": "InternVL2.5-8B (6 tiles)", "v2": "results/review/v2sub_internvl25_8b_tiles6.jsonl",
     "v1": "results/full_internvl.jsonl", "subset": True},
    {"name": "InternVL2.5-8B (1 tile)", "v2": "results/v2/v2_internvl_subset.jsonl",
     "v1": "results/full_internvl.jsonl", "subset": True},
]
# Values of an earlier draft BEFORE the 2026-11-08 audit fix of the v2 scatter rules
# (v2_analysis.py docstring): the tabular gap is now computed on a larger all-answerable set, so
# only the graph / series / prior entries are expected to reproduce.
PAPER_7B = {"tab_gap": (11.3, 2.7, 19.9, 186, 0.02), "graph_gap": (4.9, 1.7, 8.1, 528),
            "series": 21.4, "graph_best_prior": -1.9}
MODS = v2a.MODS
FORMATS = v2a.FORMATS


# Optional row scorer (row -> 0/1). None = the stored `exact_match` field (paper scoring);
# v2_multimodel_normalized.py sets it to the index-prefix-normalised score.
SCORER = None


def load_em(path, keep_error_as_zero):
    rows = v2a.read_jsonl(os.path.join(ROOT, path))
    good, st = v2a.dedupe(rows)
    if keep_error_as_zero and st["keys_error_only"]:
        for r in rows:
            k = (r["question_id"], r["viz_type"])
            if k not in good:
                good[k] = r
    em, _ = v2a.em_table(good)
    if SCORER is not None:
        for (q, f), r in good.items():
            em[q][f] = SCORER(r)
    return em, st


def r1(x):
    return None if x is None or x != x else round(x, 2)


def gap_summary(g):
    if not g.get("n_q"):
        return {"n_q": 0}
    return {"est": g["est"], "lo": g["obj_ci"]["lo"], "hi": g["obj_ci"]["hi"],
            "p": g.get("obj_perm_p"), "n_q": g["n_q"], "n_obj": g["n_obj"], "best": g["best"],
            "worst": g["worst"], "per_format": g["per_format"]}


def analyse(spec, meta, q2d, cat2s, nc, prior_const, prior_q):
    EM2, st2 = load_em(spec["v2"], False)
    qs_all = {md: [q for q, r in meta.items() if r["modality"] == md] for md in MODS}
    present = {q for q in EM2 if any(f in EM2[q] for f in FORMATS[meta[q]["modality"]])}
    qs = {md: [q for q in qs_all[md] if q in present] for md in MODS}
    out = {"file": spec["v2"], "subset": spec["subset"], "dedupe": st2, "_present": present,
           "n_questions": {md: len(qs[md]) for md in MODS},
           "n_rows_main": sum(1 for q in EM2 for f in EM2[q]
                              if f in FORMATS[meta[q]["modality"]])}
    # per-format EM
    out["per_format"] = {}
    for md in MODS:
        for f in FORMATS[md]:
            qq = [q for q in qs[md] if f in EM2[q]]
            out["per_format"][f"{md}|{f}"] = {"em": v2a.mean100([EM2[q][f] for q in qq]),
                                              "n": len(qq)}
    # gaps (s3b definitions)
    gaps = {}
    for md in MODS:
        fm = FORMATS[md]
        allx = [q for q in qs[md] if (md, meta[q]["task"]) not in nc]
        fax = [q for q in allx if all(cat2s[q][f] in "VD" for f in fm)]
        gaps[md] = {"all_answerable_exclNC": gap_summary(v2a.gap_block(EM2, fax, fm, q2d)),
                    "n_q_exclNC": len(allx)}
        if md == "timeseries":
            nl_fa = [q for q in allx if all(cat2s[q][f] in "VD" for f in v2a.TS_NONLOSSY)]
            gaps[md]["nonlossy3_all_exclNC"] = gap_summary(
                v2a.gap_block(EM2, allx, v2a.TS_NONLOSSY, q2d))
            gaps[md]["nonlossy3_answerable_exclNC"] = gap_summary(
                v2a.gap_block(EM2, nl_fa, v2a.TS_NONLOSSY, q2d))
    out["gaps"] = gaps
    # best format vs prior (s5b)
    out["best_vs_prior"] = {}
    for md in MODS:
        pf = {f: out["per_format"][f"{md}|{f}"]["em"] for f in FORMATS[md]
              if out["per_format"][f"{md}|{f}"]["n"]}
        b = max(pf, key=pf.get)
        qq = [q for q in qs[md] if b in EM2[q]]
        r = v2a.contrast([(q, EM2[q][b]) for q in qq], q2d)
        pv = prior_const[md]
        rec = {"best": b, "em": r["est"], "prior_cv_const": pv, "diff": r["est"] - pv,
               "lo": r["obj_ci"]["lo"] - pv, "hi": r["obj_ci"]["hi"] - pv, "n_q": r["n_units"],
               "n_obj": r["n_obj"]}
        # per-question CV prior on the same questions (paired; matters for subsets)
        rq = v2a.contrast([(q, EM2[q][b] - prior_q[q]) for q in qq], q2d)
        rec.update(prior_cv_same_q=v2a.mean100([prior_q[q] for q in qq]),
                   diff_same_q=rq["est"], lo_same_q=rq["obj_ci"]["lo"],
                   hi_same_q=rq["obj_ci"]["hi"])
        out["best_vs_prior"][md] = rec
    # every format vs the CV prior constant (s5b "exceeds": EM object-CI lower bound > prior)
    out["formats_vs_prior"] = {}
    for md in MODS:
        rows = {}
        for f in FORMATS[md]:
            qq = [q for q in qs[md] if f in EM2[q]]
            if not qq:
                continue
            r = v2a.contrast([(q, EM2[q][f]) for q in qq], q2d)
            rq = v2a.contrast([(q, EM2[q][f] - prior_q[q]) for q in qq], q2d)
            rows[f] = {"diff": r["est"] - prior_const[md],
                       "lo": r["obj_ci"]["lo"] - prior_const[md],
                       "hi": r["obj_ci"]["hi"] - prior_const[md],
                       "exceeds": r["obj_ci"]["lo"] > prior_const[md],
                       "diff_same_q": rq["est"], "exceeds_same_q": rq["obj_ci"]["lo"] > 0}
        out["formats_vs_prior"][md] = rows
    # degree effect (s7)
    deg = {}
    gq = [q for q in qs["graph"] if v2a.DEG_FMT in EM2.get(q, {}) and "text_only" in EM2[q]]
    for lab, sel in (("degree_query", lambda q: meta[q]["task"] == "degree_query"),
                     ("other_tasks", lambda q: meta[q]["task"] != "degree_query")):
        qq = [q for q in gq if sel(q)]
        if not qq:
            deg[lab] = {"n": 0}
            continue
        r = v2a.contrast([(q, EM2[q][v2a.DEG_FMT] - EM2[q]["text_only"]) for q in qq], q2d,
                         perm=True)
        deg[lab] = {"est": r["est"], "lo": r["obj_ci"]["lo"], "hi": r["obj_ci"]["hi"],
                    "p": r["obj_perm_p"], "n": r["n_units"], "n_obj": r["n_obj"],
                    "em_text_only": v2a.mean100([EM2[q]["text_only"] for q in qq]),
                    "em_text_only_deg": v2a.mean100([EM2[q][v2a.DEG_FMT] for q in qq])}
    out["degree_effect"] = deg
    # recovery v1 -> v2 (s2)
    if spec["v1"]:
        EM1, st1 = load_em(spec["v1"], True)
        out["v1_dedupe"] = st1
        rec = {}
        for md in MODS:
            for f in FORMATS[md]:
                both = [q for q in qs[md] if f in EM2[q] and f in EM1.get(q, {})]
                r = v2a.contrast([(q, EM2[q][f] - EM1[q][f]) for q in both], q2d)
                rec[f"{md}|{f}"] = {"v1": v2a.mean100([EM1[q][f] for q in both]),
                                    "v2": v2a.mean100([EM2[q][f] for q in both]),
                                    "d": r["est"], "lo": r["obj_ci"]["lo"],
                                    "hi": r["obj_ci"]["hi"], "n": r["n_units"]}
        out["recovery"] = rec
    return out


def fmt_gap(g, with_n=True):
    if not g.get("n_q"):
        return "--"
    s = f"{g['est']:.1f} [{g['lo']:.1f},{g['hi']:.1f}]"
    return s + (f" ({g['n_q']})" if with_n else "")


def main():
    meta = rc.load_bench()
    q2d = {q: b["data_id"] for q, b in meta.items()}
    cat2s = rc.v2_cats(meta, strict=True)
    nc = rc.near_constant()
    pr = v2a.load_prior()
    prior_const = {md: pr[md]["cv_mean"] for md in MODS}
    prior_q = rc.cv_prior_per_question(meta)
    man_sub = {r["question_id"] for r in v2a.read_jsonl(
        os.path.join(ROOT, "benchmark", "render_v2", "manifest_subset32b.jsonl"))}
    res = {"definitions": __doc__, "prior_cv_const": prior_const, "models": {},
           "subset_manifest_questions": len(man_sub)}
    for spec in MODELS:
        print(f"\n=== {spec['name']}  ({spec['v2']})")
        o = analyse(spec, meta, q2d, cat2s, nc, prior_const, prior_q)
        present = o.pop("_present")
        if spec["subset"]:
            o["subset_matches_manifest"] = present == man_sub
            print(f"  subset questions {len(present)}; identical to manifest_subset32b "
                  f"({len(man_sub)}): {present == man_sub}")
        res["models"][spec["name"]] = o
        print(f"  dedupe {o['dedupe']}; questions {o['n_questions']} "
              f"(total {sum(o['n_questions'].values())}); main-format rows {o['n_rows_main']}")
        for md in MODS:
            print(f"  {md:10s} EM " + "  ".join(
                f"{f} {v2a.f1(o['per_format'][f'{md}|{f}']['em'])} (n={o['per_format'][f'{md}|{f}']['n']})"
                for f in FORMATS[md]))
        for md in ("tabular", "graph"):
            g = o["gaps"][md]["all_answerable_exclNC"]
            if g.get("n_q"):
                print(f"  {md} residual gap {g['est']:.1f} [{g['lo']:.1f},{g['hi']:.1f}] "
                      f"p={g['p']:.4f} n={g['n_q']} objs={g['n_obj']} best={g['best']} "
                      f"worst={g['worst']}")
            else:
                print(f"  {md} residual gap: no question answerable in every format")
        ts = o["gaps"]["timeseries"]
        for k in ("nonlossy3_all_exclNC", "nonlossy3_answerable_exclNC"):
            g = ts[k]
            if g.get("n_q"):
                print(f"  series {k}: {g['est']:.1f} [{g['lo']:.1f},{g['hi']:.1f}] "
                      f"p={g['p']:.4f} n={g['n_q']} best={g['best']} worst={g['worst']}")
        for md in MODS:
            b = o["best_vs_prior"][md]
            print(f"  best-vs-prior {md:10s} {b['best']:16s} EM {b['em']:.1f} - prior "
                  f"{b['prior_cv_const']:.1f} = {b['diff']:+.1f} [{b['lo']:+.1f},{b['hi']:+.1f}] "
                  f"n={b['n_q']} | same-q CV prior {b['prior_cv_same_q']:.1f}: "
                  f"{b['diff_same_q']:+.1f} [{b['lo_same_q']:+.1f},{b['hi_same_q']:+.1f}]")
        for md in MODS:
            fv = o["formats_vs_prior"][md]
            k = "exceeds_same_q" if spec["subset"] else "exceeds"
            print(f"  formats beating the prior ({'same-q' if spec['subset'] else 'const'}) "
                  f"{md:10s}: {[f for f, v in fv.items() if v[k]]} of {len(fv)}")
        for lab, d in o["degree_effect"].items():
            if d.get("n"):
                print(f"  degree effect {lab:12s} text_only {d['em_text_only']:.1f} -> "
                      f"deg {d['em_text_only_deg']:.1f}: {d['est']:+.1f} "
                      f"[{d['lo']:+.1f},{d['hi']:+.1f}] p={d['p']:.4f} n={d['n']}")
            else:
                print(f"  degree effect {lab}: no text_only_deg rows")
        if "recovery" in o:
            print("  recovery v2-v1: " + "  ".join(
                f"{k.split('|')[1][:9]}({k[0]}) {v['d']:+.1f} [{v['lo']:+.1f},{v['hi']:+.1f}] n={v['n']}"
                for k, v in o["recovery"].items()))

    # ---- reproduction check against the paper
    o = res["models"]["Qwen2.5-VL-7B"]
    tg = o["gaps"]["tabular"]["all_answerable_exclNC"]
    gg = o["gaps"]["graph"]["all_answerable_exclNC"]
    sg = o["gaps"]["timeseries"]["nonlossy3_all_exclNC"]
    gb = o["best_vs_prior"]["graph"]
    got = {"tab_gap": (round(tg["est"], 1), round(tg["lo"], 1), round(tg["hi"], 1), tg["n_q"],
                       round(tg["p"], 2)),
           "graph_gap": (round(gg["est"], 1), round(gg["lo"], 1), round(gg["hi"], 1),
                         gg["n_q"]),
           "series": round(sg["est"], 1), "graph_best_prior": round(gb["diff"], 1)}
    ok = all(tuple(got[k]) == tuple(PAPER_7B[k]) if isinstance(got[k], tuple)
             else got[k] == PAPER_7B[k] for k in PAPER_7B)
    res["reproduction_7b"] = {"paper": PAPER_7B, "got": got, "match": ok}
    ok_rest = all(got[k] == PAPER_7B[k] for k in ("graph_gap", "series", "graph_best_prior"))
    print(f"\n[7B reproduction vs pre-fix draft] paper {PAPER_7B}\n                  got   {got}  -> "
          f"{'MATCH' if ok else 'tabular gap changed by the audit fix (expected); graph/series/prior ' + ('MATCH' if ok_rest else 'MISMATCH')}")

    with open(os.path.join(HERE, "v2_multimodel_results.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)

    # ---- LaTeX
    hdr = ["Model", "Tab.\\ gap [CI] ($n$)", "Tab.\\ best/worst", "Graph gap [CI] ($n$)",
           "Series 3-fmt [CI]", "Best$-$prior tab.", "Best$-$prior ser.",
           "Best$-$prior graph", "Deg.\\ effect deg./other"]
    rows = []
    abbr = {"table_image": "tbl", "text_only": "txt", "heatmap": "heat", "bar_chart": "bar",
            "scatter_plot": "scat", "node_link": "node", "circular_layout": "circ",
            "adjacency_matrix": "adj", "line_plot": "line"}
    for spec in MODELS:
        o = res["models"][spec["name"]]
        tg = o["gaps"]["tabular"]["all_answerable_exclNC"]
        gg = o["gaps"]["graph"]["all_answerable_exclNC"]
        sg = o["gaps"]["timeseries"]["nonlossy3_all_exclNC"]
        bp = o["best_vs_prior"]
        key = "diff_same_q" if spec["subset"] else "diff"
        lo, hi = ("lo_same_q", "hi_same_q") if spec["subset"] else ("lo", "hi")

        def bvp(md):
            b = bp[md]
            return f"{b[key]:+.1f} [{b[lo]:+.1f},{b[hi]:+.1f}]"
        d = o["degree_effect"]
        de = ("--" if not d["degree_query"].get("n") else
              f"{d['degree_query']['est']:+.1f} / {d['other_tasks']['est']:+.1f}")
        bw = (f"{abbr.get(tg['best'], tg['best'])}/{abbr.get(tg['worst'], tg['worst'])}"
              if tg.get("n_q") else "--")
        rows.append([rc.Raw(rc.tex_esc(spec["name"]) + ("$^\\dagger$" if spec["subset"]
                                                         else "")),
                     fmt_gap(tg), bw, fmt_gap(gg), fmt_gap(sg, with_n=False), bvp("tabular"),
                     bvp("timeseries"), bvp("graph"), de])
    path = os.path.join(ROOT, "paper_tables", "tab_v2_multimodel.tex")
    rc.write_tabular(
        path,
        "Expressive (v2) suite per model. Gap = best minus worst format on questions answerable\n"
        "(strict) in every format outside near-constant templates (object-cluster 95% CI; n =\n"
        "questions); best/worst for tables (tbl, txt, heat, bar, scat). Series = best minus worst\n"
        "of line_plot / heatmap / text_only on all non-near-constant series questions. Best-prior\n"
        "= best format's EM minus the question-only CV prior (pp). Deg. effect = text_only_deg\n"
        "minus text_only on degree_query / other graph tasks. dagger = 1,096-question strict\n"
        "subset; for these, gaps are on the subset and the prior is the per-question CV prior\n"
        "averaged over the same questions; their series cells are blank because every subset\n"
        "series question lies in a near-constant template (seasonality / change point).",
        "l l l l l l l l l", hdr, rows)
    print("\nwrote scripts/analysis/v2_multimodel_results.json, "
          "paper_tables/tab_v2_multimodel.tex")


if __name__ == "__main__":
    main()
