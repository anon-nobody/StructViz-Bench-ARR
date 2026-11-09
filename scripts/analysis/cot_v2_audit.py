#!/usr/bin/env python3
"""Chain-of-thought on the expressive (v2) answerable subset, and the partial v1 CoT re-run.

Stdlib only, no inference.

    python scripts/analysis/cot_v2_audit.py

Part A (v2). Questions: results/review/cot_v2_qids.txt, 80 per depth category (fewer where the
pool is smaller) sampled with random.Random(0) from the v2 fully-answerable subset
(benchmark/render_v2/fully_answerable_qids.txt), all formats of each (graph includes the
diagnostic text_only_deg; main analyses use the main formats only). Conditions, all
Qwen2.5-VL-7B on identical v2 images:
  cot_new      results/review/cot_v2_qwen.jsonl, CoT system prompt + CoT user suffix ("Think
               step by step, then end with a line `Final answer: <answer>`"); `exact_match`
               scores `prediction` = text after the last 'Final answer:' marker
               (scripts/eval_local_suite.py cot_extract).
  cot_legacy   same responses, `exact_match_legacy` = parse_answer(raw) as the original
               pipeline stored CoT answers.
  sys_only     results/review/cot_v2_qwen_sysprompt_only.jsonl: CoT instruction in the system
               prompt only, no user suffix (run_review_queue.sh cotv2_sys); new parser.
  plain        the same (question, format) rows of results/v2/v2_qwen.jsonl (default prompt).
EM by depth and modality; paired object-cluster bootstrap 95% CIs and sign-flip p for each
condition minus plain (cluster = data_id; v2_analysis.contrast); wrong-type share
(review_common.right_type, same rule as cot_audit.py); the CoT - plain difference on pairs
where both predictions are of the right type; and all of it with near-constant cells
excluded.

Part B (v1, PARTIAL while the job runs). results/review/cot_v1_qwen.jsonl re-runs the
original ablation CoT prompt on v1 images and stores the raw response, so the original
extraction can be audited: `exact_match` (text after the last 'Answer:'/'Final answer:'),
`exact_match_legacy` (original extractor), vs. results/ablation/prompt_qwen_cot.jsonl (the
paper's CoT cell) and prompt_qwen_concise.jsonl (Qwen's best non-CoT style) on the same rows.

Writes scripts/analysis/cot_v2_audit_results.json and paper_tables/tab_cot_v2.tex.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import review_common as rc  # noqa: E402
import v2_analysis as v2a  # noqa: E402

R = os.path.join(rc.ROOT, "results", "review")
GROUPS = rc.DEPTHS + rc.MODS + ["overall"]
ABBR = {"1-hop": "1-hop", "2-hop": "2-hop", "3-hop": "3-hop", "counterfactual": "CF",
        "tabular": "tabular", "timeseries": "series", "graph": "graph", "overall": "All"}
CONDS = ["plain", "cot_new", "cot_legacy", "sys_only"]


def keyed(path):
    out = {}
    for r in rc.read_jsonl(path):
        k = (r["question_id"], r["viz_type"])
        assert k not in out, (path, k)
        out[k] = r
    return out


def ci(items, q2d, perm=True):
    if not items:
        return None
    r = v2a.contrast(items, q2d, perm=perm)
    return {"est": r["est"], "lo": r["obj_ci"]["lo"], "hi": r["obj_ci"]["hi"],
            "p": r.get("obj_perm_p"), "n": r["n_units"], "n_obj": r["n_obj"]}


def s(x):
    return "--" if x is None else f"{x['est']:+.1f} [{x['lo']:+.1f},{x['hi']:+.1f}]"


def part_a(meta, q2d, cs, nc):
    cot = keyed(os.path.join(R, "cot_v2_qwen.jsonl"))
    sysd = keyed(os.path.join(R, "cot_v2_qwen_sysprompt_only.jsonl"))
    plain_all, _ = rc.load_v2_qwen(meta)
    deg = {k: r for k, r in v2a.dedupe(rc.read_jsonl(os.path.join(
        rc.ROOT, "results", "v2", "v2_qwen.jsonl")))[0].items() if k[1] == v2a.DEG_FMT}
    qids = [q.strip() for q in open(os.path.join(R, "cot_v2_qids.txt")) if q.strip()]
    assert set(cot) == set(sysd)
    keys_all = sorted(cot)
    main = [k for k in keys_all if k[1] in rc.FORMATS[meta[k[0]]["modality"]]]
    assert all(k in plain_all for k in main)
    plain = {k: plain_all[k] for k in main}
    plain.update({k: deg[k] for k in keys_all if k[1] == v2a.DEG_FMT})

    def em(c, k):
        if c == "plain":
            return rc.em_val(plain[k])
        if c == "cot_new":
            return rc.em_val(cot[k])
        if c == "cot_legacy":
            return float(cot[k]["exact_match_legacy"])
        return rc.em_val(sysd[k])

    def pred(c, k):
        return {"plain": plain, "cot_new": cot, "sys_only": sysd}.get(c, cot)[k][
            "prediction_legacy" if c == "cot_legacy" else "prediction"]

    def rtype(c, k):
        b = meta[k[0]]
        return rc.right_type(pred(c, k), b["answer"], cs[(b["modality"], b["task"])])

    def grp(k, g):
        b = meta[k[0]]
        return g == "overall" or b["difficulty"] == g or b["modality"] == g

    info = {"n_questions": len(qids), "n_rows": len(keys_all), "n_main_rows": len(main),
            "questions_by_depth": dict(Counter(meta[q]["difficulty"] for q in qids)),
            "questions_by_modality": dict(Counter(meta[q]["modality"] for q in qids)),
            "questions_in_near_constant": sum((meta[q]["modality"], meta[q]["task"]) in nc
                                              for q in qids),
            "marker_found_cot": sum(r["cot_marker_found"] in (True, "True")
                                    for r in cot.values()),
            "marker_found_sys": sum(r["cot_marker_found"] in (True, "True")
                                    for r in sysd.values()),
            "hit_max_tokens_cot": sum(r["hit_max_new_tokens"] in (True, "True")
                                      for r in cot.values()),
            "ts_tasks": dict(Counter(meta[q]["task"] for q in qids
                                     if meta[q]["modality"] == "timeseries"))}
    out = {"info": info, "samples": {}}
    for samp, keep in (("main", lambda k: True),
                       ("main_exclNC", lambda k: (meta[k[0]]["modality"], meta[k[0]]["task"])
                        not in nc)):
        ks0 = [k for k in main if keep(k)]
        res = {}
        for g in GROUPS:
            ks = [k for k in ks0 if grp(k, g)]
            if not ks:
                continue
            rec = {"n": len(ks), "n_q": len({k[0] for k in ks})}
            for c in CONDS:
                rec[c] = {"em": rc.mean100(em(c, k) for k in ks),
                          "wrong_type": 100 - rc.mean100(rtype(c, k) for k in ks)}
            for c in ("cot_new", "cot_legacy", "sys_only"):
                rec[f"{c}_minus_plain"] = ci([(k[0], em(c, k) - em("plain", k)) for k in ks],
                                             q2d)
            both = [k for k in ks if rtype("cot_new", k) and rtype("plain", k)]
            rec["cot_new_minus_plain_both_right_type"] = ci(
                [(k[0], em("cot_new", k) - em("plain", k)) for k in both], q2d)
            res[g] = rec
        out["samples"][samp] = res
    # diagnostic: text_only_deg rows
    dk = [k for k in keys_all if k[1] == v2a.DEG_FMT]
    out["text_only_deg"] = {"n": len(dk), **{c: rc.mean100(em(c, k) for k in dk)
                                             for c in CONDS},
                            "cot_new_minus_plain": ci([(k[0], em("cot_new", k)
                                                        - em("plain", k)) for k in dk], q2d)}
    # by format (main sample)
    out["by_format"] = {}
    for md in rc.MODS:
        for f in rc.FORMATS[md]:
            ks = [k for k in main if k[1] == f and meta[k[0]]["modality"] == md]
            if ks:
                out["by_format"][f"{md}|{f}"] = {
                    "n": len(ks), **{c: rc.mean100(em(c, k) for k in ks) for c in CONDS},
                    "cot_new_minus_plain": ci([(k[0], em("cot_new", k) - em("plain", k))
                                               for k in ks], q2d, perm=False)}
    # legacy vs new parser disagreement
    dis = [k for k in main if em("cot_new", k) != em("cot_legacy", k)]
    out["parser_disagreement"] = {
        "n": len(dis), "new_right_legacy_wrong": sum(em("cot_new", k) > em("cot_legacy", k)
                                                     for k in dis),
        "examples": [(meta[k[0]]["task"], k[1], cot[k]["prediction"],
                      cot[k]["prediction_legacy"], meta[k[0]]["answer"]) for k in dis[:8]]}
    return out


def part_b(meta, q2d, cs):
    path = os.path.join(R, "cot_v1_qwen.jsonl")
    rows = [r for r in rc.read_jsonl(path)] if os.path.exists(path) else []
    if not rows:
        return {"n": 0}
    new = {}
    for r in rows:
        new[(r["question_id"], r["viz_type"])] = r
    old = keyed(os.path.join(rc.RES, "ablation", "prompt_qwen_cot.jsonl"))
    conc = keyed(os.path.join(rc.RES, "ablation", "prompt_qwen_concise.jsonl"))
    ks = sorted(k for k in new if k in old)
    manifest_n = sum(1 for _ in open(os.path.join(R, "cot_v1_manifest.jsonl")))

    def rt(p, k):
        b = meta[k[0]]
        return rc.right_type(p, b["answer"], cs[(b["modality"], b["task"])])

    out = {"n": len(ks), "n_manifest": manifest_n, "partial": len(ks) < manifest_n,
           "n_questions": len({k[0] for k in ks}),
           "modalities": dict(Counter(meta[k[0]]["modality"] for k in ks)),
           "depths": dict(Counter(meta[k[0]]["difficulty"] for k in ks))}
    cond = {
        "rerun_new_parser": lambda k: rc.em_val(new[k]),
        "rerun_legacy": lambda k: float(new[k]["exact_match_legacy"]),
        "original_ablation_cot": lambda k: rc.em_val(old[k]),
        "original_concise": lambda k: rc.em_val(conc[k]),
    }
    preds = {
        "rerun_new_parser": lambda k: new[k]["prediction"],
        "rerun_legacy": lambda k: new[k]["prediction_legacy"],
        "original_ablation_cot": lambda k: old[k]["prediction"],
        "original_concise": lambda k: conc[k]["prediction"],
    }
    for c in cond:
        out[c] = {"em": rc.mean100(cond[c](k) for k in ks),
                  "wrong_type": 100 - rc.mean100(rt(preds[c](k), k) for k in ks)}
    for a, b in (("rerun_new_parser", "rerun_legacy"), ("rerun_new_parser", "original_concise"),
                 ("rerun_legacy", "original_ablation_cot"),
                 ("original_ablation_cot", "original_concise")):
        out[f"{a}_minus_{b}"] = ci([(k[0], cond[a](k) - cond[b](k)) for k in ks], q2d)
    out["legacy_pred_equals_original_pred"] = rc.mean100(
        rc.norm(new[k]["prediction_legacy"]) == rc.norm(old[k]["prediction"]) for k in ks)
    out["marker_found"] = sum(new[k]["cot_marker_found"] in (True, "True") for k in ks)
    return out


def main():
    meta = rc.load_bench()
    q2d = {q: b["data_id"] for q, b in meta.items()}
    cs = rc.cell_stats(meta)
    nc = rc.near_constant()
    A = part_a(meta, q2d, cs, nc)
    B = part_b(meta, q2d, cs)
    i = A["info"]
    print(f"[A] CoT on the v2 fully-answerable sample: {i['n_questions']} questions, "
          f"{i['n_main_rows']} main-format rows (+{A['text_only_deg']['n']} text_only_deg)")
    print(f"    by depth {i['questions_by_depth']}; by modality {i['questions_by_modality']}; "
          f"in near-constant cells {i['questions_in_near_constant']}; series tasks {i['ts_tasks']}")
    print(f"    'Final answer:' marker found: CoT {i['marker_found_cot']}/{i['n_rows']}, "
          f"sys-only {i['marker_found_sys']}/{i['n_rows']}; CoT hit max tokens "
          f"{i['hit_max_tokens_cot']}")
    for samp, res in A["samples"].items():
        print(f"\n  [{samp}]  EM (wrong-type %) | CoT-new - plain [CI] p | legacy - plain | "
              f"sys - plain | CoT-new - plain, both right type")
        for g, r in res.items():
            print(f"  {ABBR[g]:8s} n={r['n']:4d} q={r['n_q']:3d} " + " ".join(
                f"{c} {r[c]['em']:.1f}({r[c]['wrong_type']:.1f})" for c in CONDS)
                + f" | {s(r['cot_new_minus_plain'])} p={r['cot_new_minus_plain']['p']:.4f}"
                f" | {s(r['cot_legacy_minus_plain'])} | {s(r['sys_only_minus_plain'])}"
                f" | {s(r['cot_new_minus_plain_both_right_type'])} "
                f"(n={r['cot_new_minus_plain_both_right_type']['n']})")
    print("\n  by format (main): " + "  ".join(
        f"{k.split('|')[1]}({k[0]}) n={v['n']} plain {v['plain']:.1f} cot {v['cot_new']:.1f} "
        f"{s(v['cot_new_minus_plain'])}" for k, v in A["by_format"].items()))
    t = A["text_only_deg"]
    print(f"  text_only_deg n={t['n']}: plain {t['plain']:.1f} cot {t['cot_new']:.1f} "
          f"legacy {t['cot_legacy']:.1f} sys {t['sys_only']:.1f}; cot-plain {s(t['cot_new_minus_plain'])}")
    pdz = A["parser_disagreement"]
    print(f"  parser disagreement rows {pdz['n']} (new right / legacy wrong: "
          f"{pdz['new_right_legacy_wrong']}); e.g. {pdz['examples'][:4]}")

    print(f"\n[B] v1 CoT re-run (PARTIAL: {B.get('n')} of {B.get('n_manifest')} rows; "
          f"{B.get('n_questions')} questions; {B.get('modalities')}; {B.get('depths')})")
    if B.get("n"):
        for c in ("rerun_new_parser", "rerun_legacy", "original_ablation_cot",
                  "original_concise"):
            print(f"  {c:24s} EM {B[c]['em']:.1f}  wrong-type {B[c]['wrong_type']:.1f}%")
        for k, v in B.items():
            if k.endswith(("_parser_minus_rerun_legacy", "_minus_original_concise",
                           "_minus_original_ablation_cot")):
                print(f"  {k:52s} {s(v)} p={v['p']:.4f} n={v['n']}")
        print(f"  legacy re-run prediction == original ablation prediction: "
              f"{B['legacy_pred_equals_original_pred']:.1f}% of rows; marker found "
              f"{B['marker_found']}/{B['n']}")
    rc.dump(os.path.join(HERE, "cot_v2_audit_results.json"), {"v2": A, "v1_partial": B})

    # ---- LaTeX
    hdr = ["Sample", "Group", "$n$", "Plain", "CoT", "CoT$_{legacy}$", "Sys-only",
           "WT$_{plain}$", "WT$_{CoT}$", "WT$_{legacy}$", "CoT$-$plain [CI]",
           "Legacy$-$plain [CI]", "CoT$-$plain, right type [CI]"]
    rows, mids = [], []
    for samp, lab in (("main", "All"), ("main_exclNC", rc.Raw("$-$NC"))):
        if rows:
            mids.append(len(rows))
        for g in ["overall"] + rc.DEPTHS + rc.MODS:
            r = A["samples"][samp].get(g)
            if not r:
                continue
            rows.append([lab, ABBR[g], str(r["n"]), f"{r['plain']['em']:.1f}",
                         f"{r['cot_new']['em']:.1f}", f"{r['cot_legacy']['em']:.1f}",
                         f"{r['sys_only']['em']:.1f}", f"{r['plain']['wrong_type']:.1f}",
                         f"{r['cot_new']['wrong_type']:.1f}",
                         f"{r['cot_legacy']['wrong_type']:.1f}", s(r["cot_new_minus_plain"]),
                         s(r["cot_legacy_minus_plain"]),
                         s(r["cot_new_minus_plain_both_right_type"])
                         + f" ({r['cot_new_minus_plain_both_right_type']['n']})"])
    rc.write_tabular(
        os.path.join(rc.ROOT, "paper_tables", "tab_cot_v2.tex"),
        "Chain of thought when the information is visible: Qwen2.5-VL-7B on a stratified sample\n"
        f"of {i['n_questions']} questions answerable in every expressive (v2) format\n"
        f"({i['n_main_rows']} (question, format) pairs, main formats). Plain = default prompt;\n"
        "CoT = step-by-step prompt scored on the text after 'Final answer:'; CoT_legacy = same\n"
        "responses through the original extractor; Sys-only = CoT instruction in the system\n"
        "prompt only. EM (%); WT = wrong answer type (%). Differences: paired, object-cluster\n"
        "95% CI; right type = pairs where both CoT and plain answers have the key's type.\n"
        "-NC = near-constant templates excluded.",
        "l l r r r r r r r r l l l", hdr, rows, midrules=mids)
    print("\nwrote scripts/analysis/cot_v2_audit_results.json, paper_tables/tab_cot_v2.tex")


if __name__ == "__main__":
    main()
