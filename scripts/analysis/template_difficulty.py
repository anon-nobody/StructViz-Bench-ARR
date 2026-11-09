#!/usr/bin/env python3
"""Template-level difficulty: answer distribution, question-only prior, accuracy above prior,
and accuracy by reasoning-depth category under answerability / near-constant controls.

Stdlib only, no inference.

    python scripts/analysis/template_difficulty.py

Template = (modality, task). Per template: n questions, number of distinct answers, answer
entropy (bits, lenient answer form), majority-answer share, question-only cross-validated
prior accuracy (prior_and_subsets.py 1(b): 2-fold split over data_id, majority answer per
template from the training fold, 20 seeds; per question = mean hit over seeds), mean EM over
the seven v1 models x all formats on answerable (V/D) pairs, the prior averaged over the
same pairs (pair-weighted, so EM and prior describe identical units), and EM minus prior.

(a) Spearman rho across templates (two-sided permutation p, B=10,000, random.Random(0))
    between answer entropy / majority share and (i) EM on answerable pairs, (ii) EM minus
    prior; templates without answerable pairs are dropped; reported with and without the
    ten near-constant cells.
(b) EM by depth category (1-hop / 2-hop / 3-hop / counterfactual) per model (seven v1
    models + Qwen v2 on the v2 suite with v2 answerability) on: all pairs; all pairs minus
    the near-constant cells; answerable pairs; answerable minus near-constant; strict-
    answerable (v2_analysis.strictify_v2) minus near-constant. Each with the CV prior on the
    same pairs. Pooled 7-model EM minus prior per depth carries an object-cluster bootstrap
    95% CI (cluster_cis.summarize, cluster = data_id). Depth is assigned per question, but is
    a function of the task for 41 of 45 templates; within-template depth contrasts for the
    four mixed templates are printed.

Writes scripts/analysis/template_difficulty_results.json,
paper_tables/tab_template_stats.tex, paper_tables/tab_depth_accuracy.tex.
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict

import review_common as rc

SUBSETS = ["all", "all_noNC", "vd", "vd_noNC", "strict_noNC"]
SUB_LABEL = {"all": "All pairs", "all_noNC": "All pairs, near-constant cells excluded",
             "vd": "Answerable (V/D) pairs", "vd_noNC": "Answerable, near-constant excluded",
             "strict_noNC": "Strict-answerable, near-constant excluded"}


def main():
    meta = rc.load_bench()
    q2d = {q: b["data_id"] for q, b in meta.items()}
    cs = rc.cell_stats(meta)
    nc = rc.near_constant()
    prior = rc.cv_prior_per_question(meta)
    cat1, cat1s = rc.v1_cats(meta), rc.v1_cats(meta, strict=True)
    cat2, cat2s = rc.v2_cats(meta), rc.v2_cats(meta, strict=True)
    EM = {}
    for m in rc.MODELS7:
        EM[m] = {k: rc.em_val(r) for k, r in rc.load_v1(m).items()}
    v2rows, _ = rc.load_v2_qwen(meta)
    EM[rc.QWEN_V2] = {k: rc.em_val(r) for k, r in v2rows.items()}
    models = list(EM)
    v1models = list(rc.MODELS7)

    def in_subset(sub, q, f, m):
        b = meta[q]
        cell = (b["modality"], b["task"])
        v2 = m == rc.QWEN_V2
        if sub.endswith("noNC") and cell in nc:
            return False
        if sub.startswith("vd"):
            return (cat2 if v2 else cat1)[q][f] in "VD"
        if sub.startswith("strict"):
            return (cat2s if v2 else cat1s)[q][f] in "VD"
        return True

    # ---------------------------------------------------------------- per template
    tmpl = defaultdict(list)
    for q, b in meta.items():
        tmpl[(b["modality"], b["task"])].append(q)
    trows = []
    for key in sorted(tmpl, key=lambda k: (rc.MODS.index(k[0]), k[1])):
        qs = tmpl[key]
        c = cs[key]
        depths = Counter(meta[q]["difficulty"] for q in qs)
        em_vd, pr_vd, em_all = [], [], []
        for m in v1models:
            for q in qs:
                for f in rc.FORMATS[key[0]]:
                    e = EM[m][(q, f)]
                    em_all.append(e)
                    if cat1[q][f] in "VD":
                        em_vd.append(e)
                        pr_vd.append(prior[q])
        row = {"modality": key[0], "task": key[1], "n": len(qs), "distinct": c["distinct"],
               "entropy": c["entropy"], "maj_share": 100 * c["maj_share"],
               "majority": c["majority"], "depths": dict(depths),
               "cv_prior": rc.mean100(prior[q] for q in qs),
               "em_all": rc.mean100(em_all), "n_pairs_vd": len(em_vd) // len(v1models),
               "em_vd": rc.mean100(em_vd), "prior_vd": rc.mean100(pr_vd),
               "near_constant": key in nc}
        row["em_minus_prior_vd"] = (row["em_vd"] - row["prior_vd"]) if em_vd else float("nan")
        row["em_minus_prior_all"] = row["em_all"] - row["cv_prior"]
        trows.append(row)

    print("[per template]  (EM over 7 v1 models x formats; VD = answerable pairs)")
    print(f"  {'modality':10s} {'task':27s} {'n':>4s} {'dist':>4s} {'H':>5s} {'maj%':>5s} "
          f"{'prior':>5s} {'EMall':>5s} {'nVD':>4s} {'EMvd':>5s} {'prVD':>5s} {'EM-pr':>6s} NC")
    for r in trows:
        print(f"  {r['modality']:10s} {r['task']:27s} {r['n']:4d} {r['distinct']:4d} "
              f"{r['entropy']:5.2f} {r['maj_share']:5.1f} {rc.f1(r['cv_prior']):>5s} "
              f"{rc.f1(r['em_all']):>5s} {r['n_pairs_vd']:4d} {rc.f1(r['em_vd']):>5s} "
              f"{rc.f1(r['prior_vd']):>5s} {rc.sgn(r['em_minus_prior_vd']):>6s} "
              f"{'NC' if r['near_constant'] else ''}")

    # ---------------------------------------------------------------- (a) Spearman
    corr = {}
    print("\n(a) Spearman across templates")
    for lab, keep in (("all_templates", lambda r: True),
                      ("noNC", lambda r: not r["near_constant"])):
        rs = [r for r in trows if keep(r) and r["em_vd"] == r["em_vd"]]
        for xk in ("entropy", "maj_share"):
            for yk in ("em_vd", "em_minus_prior_vd", "prior_vd"):
                rho, p = rc.spearman([r[xk] for r in rs], [r[yk] for r in rs])
                corr[f"{lab}|{xk}|{yk}"] = {"rho": rho, "p": p, "n_templates": len(rs)}
                print(f"  {lab:14s} {xk:9s} vs {yk:18s} rho={rho:+.3f} p={p:.4f} "
                      f"n={len(rs)}")
        rs_all = [r for r in trows if keep(r)]
        rho, p = rc.spearman([r["entropy"] for r in rs_all], [r["em_minus_prior_all"]
                                                              for r in rs_all])
        corr[f"{lab}|entropy|em_minus_prior_all_pairs"] = {"rho": rho, "p": p,
                                                           "n_templates": len(rs_all)}
        print(f"  {lab:14s} entropy   vs EM-prior(all pairs)  rho={rho:+.3f} p={p:.4f} "
              f"n={len(rs_all)}")

    # ---------------------------------------------------------------- (b) depth
    depth = {}
    print("\n(b) EM by depth (EM / prior on same pairs / n pairs)")
    for sub in SUBSETS:
        depth[sub] = {}
        print(f"  [{SUB_LABEL[sub]}]")
        pooled = {d: [] for d in rc.DEPTHS}
        pooled_pr = {d: [] for d in rc.DEPTHS}
        for m in models:
            depth[sub][m] = {}
            acc = {d: [] for d in rc.DEPTHS}
            pr = {d: [] for d in rc.DEPTHS}
            for (q, f), e in EM[m].items():
                if not in_subset(sub, q, f, m):
                    continue
                d = meta[q]["difficulty"]
                acc[d].append(e)
                pr[d].append(prior[q])
                if m in v1models:
                    pooled[d].append((q, e - prior[q]))
                    pooled_pr[d].append(e)
            for d in rc.DEPTHS:
                depth[sub][m][d] = {"em": rc.mean100(acc[d]), "prior": rc.mean100(pr[d]),
                                    "n": len(acc[d])}
                depth[sub][m][d]["em_minus_prior"] = (depth[sub][m][d]["em"]
                                                      - depth[sub][m][d]["prior"])
            print(f"    {m:22s} " + "  ".join(
                f"{d[:5]} {rc.f1(depth[sub][m][d]['em']):>5s}/{rc.f1(depth[sub][m][d]['prior']):>5s}"
                f" ({depth[sub][m][d]['n']})" for d in rc.DEPTHS))
        depth[sub]["pooled7"] = {}
        for d in rc.DEPTHS:
            ci = rc.cluster_ci(pooled[d], q2d)
            depth[sub]["pooled7"][d] = {
                "em": rc.mean100(pooled_pr[d]), "n_pairs": len(pooled[d]),
                "em_minus_prior": ci["est"] if ci else float("nan"),
                "ci_lo": ci["obj_ci"]["lo"] if ci else None,
                "ci_hi": ci["obj_ci"]["hi"] if ci else None,
                "n_obj": ci["n_obj"] if ci else 0,
                "n_templates": len({meta[q]["task"] + meta[q]["modality"]
                                    for q, _ in pooled[d]})}
        print("    pooled7 EM-prior [obj CI]: " + "  ".join(
            f"{d[:5]} {rc.sgn(depth[sub]['pooled7'][d]['em_minus_prior'])} "
            f"[{rc.sgn(depth[sub]['pooled7'][d]['ci_lo'])},"
            f"{rc.sgn(depth[sub]['pooled7'][d]['ci_hi'])}] "
            f"(tmpl {depth[sub]['pooled7'][d]['n_templates']})" for d in rc.DEPTHS))
        # monotonic order check per model
        mono = {}
        for m in models:
            ems = [depth[sub][m][d]["em"] for d in rc.DEPTHS[:3]]
            mono[m] = ems[0] > ems[1] > ems[2]
        depth[sub]["monotone_1>2>3_hop"] = mono
        print(f"    models with EM 1-hop > 2-hop > 3-hop: "
              f"{sum(mono.values())}/{len(mono)}")

    # within-template depth contrasts
    mixed = {}
    print("\n[within-template depth contrasts, 7 v1 models pooled, answerable pairs]")
    for key, qs in tmpl.items():
        ds = Counter(meta[q]["difficulty"] for q in qs)
        if len(ds) < 2:
            continue
        mixed[f"{key[0]}|{key[1]}"] = {}
        for d in sorted(ds):
            vals = [EM[m][(q, f)] for m in v1models for q in qs
                    if meta[q]["difficulty"] == d for f in rc.FORMATS[key[0]]
                    if cat1[q][f] in "VD"]
            allv = [EM[m][(q, f)] for m in v1models for q in qs
                    if meta[q]["difficulty"] == d for f in rc.FORMATS[key[0]]]
            mixed[f"{key[0]}|{key[1]}"][d] = {"n_q": ds[d], "em_vd": rc.mean100(vals),
                                              "em_all": rc.mean100(allv),
                                              "prior": rc.mean100(prior[q] for q in qs
                                                                  if meta[q]["difficulty"] == d)}
        print(f"  {key[0]:10s} {key[1]:18s} " + "  ".join(
            f"{d}: n={v['n_q']} EMvd={rc.f1(v['em_vd'])} EMall={rc.f1(v['em_all'])} "
            f"prior={rc.f1(v['prior'])}" for d, v in mixed[f"{key[0]}|{key[1]}"].items()))

    rc.dump(os.path.join(rc.HERE, "template_difficulty_results.json"),
            {"templates": trows, "spearman": corr, "depth": depth,
             "within_template_depth": mixed})

    # ---------------------------------------------------------------- LaTeX
    hdr = ["Modality", "Task", "Depth", "$n$", "Ans.", "$H$", "Maj.", "Prior", "EM",
           "$n^{V/D}$", "EM$^{V/D}$", "Prior$^{V/D}$", "$\\Delta^{V/D}$"]
    rows, mids = [], []
    prev = None
    abbrev = {"1-hop": "1", "2-hop": "2", "3-hop": "3", "counterfactual": "CF"}
    for r in trows:
        if prev and r["modality"] != prev:
            mids.append(len(rows))
        prev = r["modality"]
        dp = "/".join(abbrev[d] for d in rc.DEPTHS if d in r["depths"])
        rows.append([r["modality"], rc.Raw(rc.tex_esc(r["task"])
                                           + ("$^{\\dagger}$" if r["near_constant"] else "")),
                     dp, str(r["n"]), str(r["distinct"]), f"{r['entropy']:.2f}",
                     rc.f1(r["maj_share"]), rc.f1(r["cv_prior"]), rc.f1(r["em_all"]),
                     str(r["n_pairs_vd"]), rc.f1(r["em_vd"]), rc.f1(r["prior_vd"]),
                     rc.sgn(r["em_minus_prior_vd"])])
    path = os.path.join(rc.TABLES, "tab_template_stats.tex")
    rc.write_tabular(
        path,
        "Per-template statistics (template = modality x task). Ans. = distinct answers;\n"
        "H = answer entropy (bits); Maj. = majority-answer share (%); Prior = question-only\n"
        "cross-validated prior accuracy (%); EM = mean exact match over the seven v1 models x\n"
        "all formats; V/D columns restrict to answerable (visible/derivable) pairs, n^{V/D} =\n"
        "answerable (question, format) pairs per model; Delta = EM minus prior on those pairs\n"
        "(pp). dagger = near-constant cell (one answer >= 90%). Depth: 1/2/3-hop, CF =\n"
        "counterfactual.",
        "l l l r r r r r r r r r r", hdr, rows, midrules=mids)

    hdr2 = ["Subset", "Model"] + [f"EM {abbrev[d]}" for d in rc.DEPTHS] + \
        [f"$\\Delta$ {abbrev[d]}" for d in rc.DEPTHS]
    rows2, mids2 = [], []
    for sub in ("all", "all_noNC", "vd", "vd_noNC"):
        if rows2:
            mids2.append(len(rows2))
        lab = rc.Raw({"all": "All", "all_noNC": "All $-$NC", "vd": "V/D",
                      "vd_noNC": "V/D $-$NC"}[sub])
        for m in models:
            v = depth[sub][m]
            rows2.append([lab, m] + [rc.f1(v[d]["em"]) for d in rc.DEPTHS]
                         + [rc.sgn(v[d]["em_minus_prior"]) for d in rc.DEPTHS])
        p7 = depth[sub]["pooled7"]
        rows2.append([lab, "7-model pooled"] + [rc.f1(p7[d]["em"]) for d in rc.DEPTHS]
                     + [f"{rc.sgn(p7[d]['em_minus_prior'])} [{rc.sgn(p7[d]['ci_lo'])}, "
                        f"{rc.sgn(p7[d]['ci_hi'])}]" for d in rc.DEPTHS])
    path2 = os.path.join(rc.TABLES, "tab_depth_accuracy.tex")
    rc.write_tabular(
        path2,
        "Exact match (%) by reasoning-depth category (1/2/3-hop, CF = counterfactual) and EM\n"
        "minus the question-only CV prior on the same pairs (Delta, pp). All = every pair;\n"
        "-NC = ten near-constant cells (556 questions) excluded; V/D = answerable pairs only\n"
        "(v2 answerability for the v2 row). Pooled row: seven v1 models, object-cluster 95% CI\n"
        "on Delta. Depth is a function of the task for 41 of 45 templates.",
        "l l r r r r r r r r", hdr2, rows2, midrules=mids2)
    print("\nwrote scripts/analysis/template_difficulty_results.json, "
          "paper_tables/tab_template_stats.tex, paper_tables/tab_depth_accuracy.tex")


if __name__ == "__main__":
    main()
