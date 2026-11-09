#!/usr/bin/env python3
"""Chain-of-thought audit of the prompt-style ablation: how much of CoT's deficit is
answer-type-mismatched extraction?

Stdlib only, no inference.

    python scripts/analysis/cot_audit.py

Inputs: results/ablation/prompt_{gpt4o,gemini,qwen,claude}_{minimal,concise,detailed,cot}
.jsonl (500-question stratified subsample, 2,413 (question, format) rows per cell). The
`prediction` field is already the extracted answer; raw responses were not stored, so an
extraction failure can only be inferred from the extracted string. Scoring: `exact_match`.

Expected type comes from the key (review_common.right_type): yes/no, number, or label; a
prediction has the right type iff it is in the template's answer vocabulary, or it is not a
refusal / [EMPTY] / [REFUSED] / unparsed free text and its type equals the key's type (for
closed-vocabulary label templates, <= 10 distinct non-numeric answers, it must also be in the
vocabulary; any substantive answer is the right type when the key is change_point's "none").

For every model x style x depth category (1-hop / 2-hop / 3-hop / counterfactual / overall):
EM, wrong-type share, EM on right-type predictions (with n). The best non-CoT style is the
one with the highest overall EM (as in the paper's Table tab:prompt). CoT deficit vs that
style, all paired on identical (question, format) pairs:
  d_all      EM(cot) - EM(best) over all pairs
  d_cot_rt   same, restricted to pairs where the CoT prediction has the right type
  d_both_rt  same, restricted to pairs where both predictions have the right type
  d_impute   wrong-type CoT pairs are assigned the best style's outcome (extraction failure
             treated as "would have scored like the best prompt"); = sum over CoT-right-type
             pairs of (cot - best) / N_all
  attributable = 1 - d_impute / d_all  (share of the deficit explained by wrong-type CoT
             predictions under that imputation)
Object-cluster bootstrap 95% CIs (cluster = data_id) on d_all and d_both_rt via
cluster_cis.summarize, plus sign-flip p for d_both_rt. Answerable-only (V/D) variants of
d_all and d_both_rt use the v1 answerability classes.

Writes scripts/analysis/cot_audit_results.json and paper_tables/tab_cot_audit.tex.
"""
from __future__ import annotations

import os
from collections import Counter

import review_common as rc

GROUPS = rc.DEPTHS + ["overall"]
ABBR = {"1-hop": "1-hop", "2-hop": "2-hop", "3-hop": "3-hop", "counterfactual": "CF",
        "overall": "All"}


def load_style(key, style):
    rows = rc.read_jsonl(os.path.join(rc.RES, "ablation", f"prompt_{key}_{style}.jsonl"))
    out = {}
    for r in rows:
        k = (r["question_id"], r["viz_type"])
        assert k not in out
        out[k] = r
    return out


def main():
    meta = rc.load_bench()
    q2d = {q: b["data_id"] for q, b in meta.items()}
    cs = rc.cell_stats(meta)
    cat1 = rc.v1_cats(meta)
    res = {"per_style": {}, "deficit": {}, "best_style": {}, "wrong_type_breakdown": {}}

    def rt(r):
        b = meta[r["question_id"]]
        return rc.right_type(r.get("prediction", ""), b["answer"],
                             cs[(b["modality"], b["task"])])

    def grp(k, g):
        return g == "overall" or meta[k[0]]["difficulty"] == g

    print("[per model x style x depth]  EM | wrong-type % | EM on right-type (n)")
    for m, key in rc.PROMPT_KEYS.items():
        data = {s: load_style(key, s) for s in rc.STYLES}
        keys = set(data["minimal"])
        assert all(set(d) == keys for d in data.values()), m
        RT = {s: {k: rt(r) for k, r in data[s].items()} for s in rc.STYLES}
        EMs = {s: {k: rc.em_val(r) for k, r in data[s].items()} for s in rc.STYLES}
        res["per_style"][m] = {}
        for s in rc.STYLES:
            res["per_style"][m][s] = {}
            for g in GROUPS:
                ks = [k for k in keys if grp(k, g)]
                rtk = [k for k in ks if RT[s][k]]
                res["per_style"][m][s][g] = {
                    "n": len(ks), "em": rc.mean100(EMs[s][k] for k in ks),
                    "wrong_type": 100 - rc.mean100(RT[s][k] for k in ks),
                    "em_right_type": rc.mean100(EMs[s][k] for k in rtk), "n_right_type": len(rtk),
                    "n_questions": len({k[0] for k in ks})}
            pf = {}
            for k in keys:
                pf.setdefault(f"{meta[k[0]]['modality']}|{k[1]}", []).append(k)
            res["per_style"][m][s]["per_format"] = {
                fk: {"n": len(ks), "em": rc.mean100(EMs[s][k] for k in ks),
                     "wrong_type": 100 - rc.mean100(RT[s][k] for k in ks),
                     "em_right_type": rc.mean100(EMs[s][k] for k in ks if RT[s][k])}
                for fk, ks in sorted(pf.items())}
            # wrong-type breakdown (overall)
            bd = Counter()
            for k, r in data[s].items():
                if RT[s][k]:
                    continue
                p = r.get("prediction", "")
                b = meta[k[0]]
                voc = cs[(b["modality"], b["task"])]["vocab"]
                bd["refusal/empty" if rc.is_refusal(p, voc) else
                   "free_text" if rc.is_free_text(p) else
                   "out_of_vocab_label" if rc.atype(p) == rc.atype(b["answer"]) else
                   f"{rc.atype(p)}_for_{rc.atype(b['answer'])}"] += 1
            res["wrong_type_breakdown"][f"{m}|{s}"] = dict(bd)
            v = res["per_style"][m][s]
            print(f"  {m:18s} {s:8s} " + "  ".join(
                f"{ABBR[g]} {rc.f1(v[g]['em'])}|{rc.f1(v[g]['wrong_type'])}|"
                f"{rc.f1(v[g]['em_right_type'])}({v[g]['n_right_type']})" for g in GROUPS))
            print(f"  {'':18s} {'':8s} wrong-type breakdown: {dict(bd)}")

        best = max(("minimal", "concise", "detailed"),
                   key=lambda s: res["per_style"][m][s]["overall"]["em"])
        res["best_style"][m] = best
        res["deficit"][m] = {}
        for g in GROUPS:
            ks = sorted(k for k in keys if grp(k, g))
            d = [(k, EMs["cot"][k] - EMs[best][k]) for k in ks]
            d_cot_rt = [(k, v) for k, v in d if RT["cot"][k]]
            d_both = [(k, v) for k, v in d if RT["cot"][k] and RT[best][k]]
            d_vd = [(k, v) for k, v in d if cat1[k[0]][k[1]] in "VD"]
            d_vd_both = [(k, v) for k, v in d_both if cat1[k[0]][k[1]] in "VD"]
            n = len(d)
            d_all = rc.mean100(v for _, v in d)
            d_imp = 100 * sum(v for _, v in d_cot_rt) / n if n else float("nan")
            ci_all = rc.cluster_ci([(k[0], v) for k, v in d], q2d)
            ci_both = rc.cluster_ci([(k[0], v) for k, v in d_both], q2d, perm=True)
            ci_vd = rc.cluster_ci([(k[0], v) for k, v in d_vd], q2d)
            ci_vdb = rc.cluster_ci([(k[0], v) for k, v in d_vd_both], q2d)
            res["deficit"][m][g] = {
                "best": best, "n": n, "em_best": rc.mean100(EMs[best][k] for k in ks),
                "em_cot": rc.mean100(EMs["cot"][k] for k in ks),
                "wt_best": 100 - rc.mean100(RT[best][k] for k in ks),
                "wt_cot": 100 - rc.mean100(RT["cot"][k] for k in ks),
                "d_all": d_all, "d_all_ci": [ci_all["obj_ci"]["lo"], ci_all["obj_ci"]["hi"]],
                "d_cot_rt": rc.mean100(v for _, v in d_cot_rt), "n_cot_rt": len(d_cot_rt),
                "d_both_rt": ci_both["est"] if ci_both else float("nan"),
                "d_both_rt_ci": ([ci_both["obj_ci"]["lo"], ci_both["obj_ci"]["hi"]]
                                 if ci_both else None),
                "d_both_rt_p": ci_both["obj_perm_p"] if ci_both else None,
                "n_both_rt": len(d_both),
                "d_impute": d_imp,
                "attributable_pct": (100 * (1 - d_imp / d_all) if d_all and d_all == d_all
                                     else float("nan")),
                "d_vd": ci_vd["est"] if ci_vd else float("nan"), "n_vd": len(d_vd),
                "d_vd_ci": [ci_vd["obj_ci"]["lo"], ci_vd["obj_ci"]["hi"]] if ci_vd else None,
                "d_vd_both_rt": ci_vdb["est"] if ci_vdb else float("nan"),
                "n_vd_both_rt": len(d_vd_both),
                "d_vd_both_rt_ci": ([ci_vdb["obj_ci"]["lo"], ci_vdb["obj_ci"]["hi"]]
                                    if ci_vdb else None)}

    print("\n[CoT minus best non-CoT style, paired]")
    print(f"  {'model':18s} {'best':8s} {'grp':5s} {'n':>5s} {'EMbest':>6s} {'EMcot':>6s} "
          f"{'WTb':>5s} {'WTc':>5s} {'d_all':>6s} {'CI':>15s} {'d_both':>6s} {'CI':>15s} "
          f"{'n_both':>6s} {'d_imp':>6s} {'attr%':>6s} {'d_vd':>6s} {'d_vd_both':>9s}")
    for m in rc.PROMPT_KEYS:
        for g in GROUPS:
            v = res["deficit"][m][g]
            cib = v["d_both_rt_ci"] or [float("nan")] * 2
            print(f"  {m:18s} {v['best']:8s} {ABBR[g]:5s} {v['n']:5d} {rc.f1(v['em_best']):>6s} "
                  f"{rc.f1(v['em_cot']):>6s} {rc.f1(v['wt_best']):>5s} {rc.f1(v['wt_cot']):>5s} "
                  f"{rc.sgn(v['d_all']):>6s} [{rc.sgn(v['d_all_ci'][0])},{rc.sgn(v['d_all_ci'][1])}]"
                  f" {rc.sgn(v['d_both_rt']):>6s} [{rc.sgn(cib[0])},{rc.sgn(cib[1])}] "
                  f"{v['n_both_rt']:6d} {rc.sgn(v['d_impute']):>6s} {rc.f1(v['attributable_pct']):>6s}"
                  f" {rc.sgn(v['d_vd']):>6s} {rc.sgn(v['d_vd_both_rt']):>9s}")

    print("\n[tabular per format, best style vs CoT: EM | wrong-type % | EM right-type]")
    for m in rc.PROMPT_KEYS:
        b = res["best_style"][m]
        for f in rc.FORMATS["tabular"]:
            x = res["per_style"][m][b]["per_format"][f"tabular|{f}"]
            y = res["per_style"][m]["cot"]["per_format"][f"tabular|{f}"]
            print(f"  {m:18s} {f:13s} {b:8s} {rc.f1(x['em'])}|{rc.f1(x['wrong_type'])}|"
                  f"{rc.f1(x['em_right_type'])}   cot {rc.f1(y['em'])}|{rc.f1(y['wrong_type'])}|"
                  f"{rc.f1(y['em_right_type'])}  (n={x['n']})")

    ov = {m: res["deficit"][m]["overall"] for m in rc.PROMPT_KEYS}
    rng_all = (min(-v["d_all"] for v in ov.values()), max(-v["d_all"] for v in ov.values()))
    rng_both = (min(-v["d_both_rt"] for v in ov.values()),
                max(-v["d_both_rt"] for v in ov.values()))
    any_pos = {m: v["d_both_rt"] > 0 for m, v in ov.items()}
    ci_excl = {m: v["d_both_rt_ci"][1] < 0 for m, v in ov.items()}
    res["verdict"] = {"deficit_range_all_pp": rng_all, "deficit_range_both_right_type_pp":
                      rng_both, "cot_better_on_both_right_type": any_pos,
                      "both_right_type_ci_excludes_zero_below": ci_excl,
                      "mean_d_all": sum(v["d_all"] for v in ov.values()) / 4,
                      "mean_d_both_rt": sum(v["d_both_rt"] for v in ov.values()) / 4,
                      "mean_attributable_pct": sum(v["attributable_pct"] for v in ov.values()) / 4}
    print("\n[verdict] paper: CoT 'never helps (4.4-14.6pp below each model's best non-CoT)'")
    print(f"  all pairs: CoT below best by {rng_all[0]:.1f}-{rng_all[1]:.1f}pp "
          f"(mean {-res['verdict']['mean_d_all']:.1f})")
    print(f"  both-right-type pairs: below by {rng_both[0]:.1f}-{rng_both[1]:.1f}pp "
          f"(mean {-res['verdict']['mean_d_both_rt']:.1f}); CoT better for: "
          f"{[m for m, b in any_pos.items() if b] or 'none'}; CI below zero: {ci_excl}")
    print(f"  mean share of the deficit attributable to wrong-type CoT predictions: "
          f"{res['verdict']['mean_attributable_pct']:.1f}%")

    rc.dump(os.path.join(rc.HERE, "cot_audit_results.json"), res)

    # ---------------------------------------------------------------- LaTeX
    hdr = ["Model", "Depth", "Best", "$n$", "EM$_{best}$", "EM$_{CoT}$", "WT$_{best}$",
           "WT$_{CoT}$", "$\\Delta_{all}$ [CI]", "$\\Delta_{rt}$ [CI]", "$n_{rt}$",
           "Attr.\\,\\%"]
    rows, mids = [], []
    for m in rc.PROMPT_KEYS:
        if rows:
            mids.append(len(rows))
        for g in ["overall"] + rc.DEPTHS:
            v = res["deficit"][m][g]
            cib = v["d_both_rt_ci"] or [float("nan")] * 2
            rows.append([m, ABBR[g], v["best"], str(v["n"]), rc.f1(v["em_best"]),
                         rc.f1(v["em_cot"]), rc.f1(v["wt_best"]), rc.f1(v["wt_cot"]),
                         f"{rc.sgn(v['d_all'])} [{rc.sgn(v['d_all_ci'][0])}, "
                         f"{rc.sgn(v['d_all_ci'][1])}]",
                         f"{rc.sgn(v['d_both_rt'])} [{rc.sgn(cib[0])}, {rc.sgn(cib[1])}]",
                         str(v["n_both_rt"]), rc.f1(v["attributable_pct"])])
    rc.write_tabular(
        os.path.join(rc.TABLES, "tab_cot_audit.tex"),
        "Chain-of-thought audit on the prompt ablation (500-question subsample, 2,413 pairs per\n"
        "model x style). Best = best non-CoT style by overall EM. WT = share (%) of extracted\n"
        "predictions whose answer type does not match the key (yes/no, number, label; refusals,\n"
        "[EMPTY] and unparsed free text count as wrong type). Delta_all = EM(CoT) - EM(best) on\n"
        "all pairs; Delta_rt = the same on pairs where both predictions have the right type;\n"
        "object-cluster 95% CIs. Attr. = share of Delta_all removed when wrong-type CoT\n"
        "predictions are assigned the best style's outcome. Depth: 1/2/3-hop, CF.",
        "l l l r r r r r r r r r", hdr, rows, midrules=mids)
    print("\nwrote scripts/analysis/cot_audit_results.json, paper_tables/tab_cot_audit.tex")


if __name__ == "__main__":
    main()
