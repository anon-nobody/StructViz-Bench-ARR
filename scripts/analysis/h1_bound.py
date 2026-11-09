#!/usr/bin/env python3
"""H1, the information upper bound: on absent (A) pairs no model should beat the question-only
prior. Every cell where accuracy is significantly ABOVE the prior is an audit error or a leak.

Stdlib only; no inference.

    python scripts/analysis/h1_bound.py

Pairs: v1 = the seven default-suite models (results/full_*.jsonl) with the v1 answerability
classes (scripts/analysis/answerability.py, default rules: the paper's "absent" pairs);
v2 = the seven expressive-suite entries of v2_multimodel.MODELS with the strict v2 classes
(v2_analysis.classify_v2 + strictify_v2), subset models on their 1,096 subset questions.
Score: canonical_em (scoring.py). Prior: the question-only cross-validated prior per question
(review_common.cv_prior_per_question: template majority answer learned on half of the objects,
20 seeds), so EM and prior are compared on identical (question, format) pairs.

Per model x modality x suite: d = mean(EM - prior) over A pairs; object-cluster bootstrap
(cluster = data_id, B=10000, random.Random(0)) two-sided 95% CI and one-sided p =
(#{d* <= 0} + 1) / (B + 1); "significantly above" = one-sided p < 0.05, also reported after
Holm correction over all cells. For every cell above the prior (raw p < 0.05): EM - prior by
(task, format) on its A pairs and up to five correct examples.

Writes scripts/analysis/h1_bound_results.json and paper_tables/tab_h1_bound.tex.
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import review_common as rc  # noqa: E402
import scoring as sc  # noqa: E402
import v2_analysis as v2a  # noqa: E402
import v2_multimodel as mm  # noqa: E402

B = 10000


def boot(items, q2d):
    """items: [(qid, value)] -> (est, lo95, hi95, one-sided p for est <= 0)."""
    cl = v2a.clusters_from(items, q2d)
    est = v2a.point(cl)
    rng = random.Random(0)
    m = len(cl)
    stats = []
    for _ in range(B):
        s = n = 0.0
        for _ in range(m):
            a, b = cl[int(rng.random() * m)]
            s += a
            n += b
        stats.append(100 * s / n)
    lo, hi = v2a.pct(stats)
    p = (sum(1 for x in stats if x <= 0) + 1) / (B + 1)
    return est, lo, hi, p


def v1_score(r):
    raw = r.get("prediction_raw")
    if raw is not None and str(raw) == str(r.get("prediction")):
        raw = None
    return sc.canonical_em(r.get("prediction", ""), raw, r["answer"], r.get("task"),
                           float(r.get("exact_match", 0) or 0))


def main():
    meta = rc.load_bench()
    q2d = {q: b["data_id"] for q, b in meta.items()}
    prior = rc.cv_prior_per_question(meta)
    cat1 = rc.v1_cats(meta)
    cat2 = rc.v2_cats(meta, strict=True)
    sources = []
    for m in rc.MODELS7:
        rows = rc.load_v1(m)
        sources.append(("v1", m, {k: (r, v1_score(r)) for k, r in rows.items()}, cat1))
    for spec in mm.MODELS:
        rows, _ = v2a.dedupe(rc.read_jsonl(os.path.join(ROOT, spec["v2"])))
        rows = {k: r for k, r in rows.items() if k[1] in rc.FORMATS[meta[k[0]]["modality"]]}
        sources.append(("v2", spec["name"], {k: (r, sc.row_score(r)) for k, r in rows.items()},
                        cat2))
    cells = []
    for suite, m, sc_rows, cats in sources:
        for md in rc.MODS:
            items = [(q, f) for (q, f) in sc_rows if meta[q]["modality"] == md
                     and cats[q][f] == "A"]
            if not items:
                cells.append({"suite": suite, "model": m, "modality": md, "n": 0})
                continue
            vals = [(q, sc_rows[(q, f)][1] - prior[q]) for q, f in items]
            est, lo, hi, p = boot(vals, q2d)
            cell = {"suite": suite, "model": m, "modality": md, "n": len(items),
                    "n_obj": len({q2d[q] for q, _ in items}),
                    "em": rc.mean100(sc_rows[k][1] for k in items),
                    "prior": rc.mean100(prior[q] for q, _ in items),
                    "diff": est, "lo": lo, "hi": hi, "p_one_sided": p}
            if p < 0.05:
                by = defaultdict(list)
                for q, f in items:
                    by[(meta[q]["task"], f)].append((q, sc_rows[(q, f)][1] - prior[q],
                                                     sc_rows[(q, f)][1]))
                cell["by_task_format"] = sorted(
                    [{"task": t, "format": f, "n": len(v),
                      "em": rc.mean100(x[2] for x in v),
                      "em_minus_prior": rc.mean100(x[1] for x in v)}
                     for (t, f), v in by.items()], key=lambda d: -d["em_minus_prior"] * d["n"])
                ex = []
                for d in cell["by_task_format"][:3]:
                    for q, f in items:
                        if meta[q]["task"] == d["task"] and f == d["format"] \
                                and sc_rows[(q, f)][1] >= 1 and len(ex) < 6:
                            r = sc_rows[(q, f)][0]
                            ex.append({"question_id": q, "format": f,
                                       "question": meta[q]["question"],
                                       "key": meta[q]["answer"],
                                       "prediction": r.get("prediction"),
                                       "prior_hit": prior[q]})
                cell["examples"] = ex
            cells.append(cell)
    tested = [c for c in cells if c["n"]]
    ps = sorted(range(len(tested)), key=lambda i: tested[i]["p_one_sided"])
    running = 0.0
    for rank, i in enumerate(ps):
        running = max(running, min(1.0, (len(tested) - rank) * tested[i]["p_one_sided"]))
        tested[i]["p_holm"] = running
    above = [c for c in tested if c["p_one_sided"] < 0.05]
    res = {"definition": __doc__, "cells": cells, "n_cells_tested": len(tested),
           "n_above_raw": len(above), "n_above_holm": sum(c["p_holm"] < 0.05 for c in tested)}
    with open(os.path.join(HERE, "h1_bound_results.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)

    print(f"H1: {len(tested)} model x modality x suite cells with absent pairs; above the prior "
          f"(one-sided p<.05): {len(above)} raw, {res['n_above_holm']} after Holm")
    for c in cells:
        if not c["n"]:
            print(f"  {c['suite']} {c['model']:26s} {c['modality']:10s} no absent pairs")
            continue
        flag = "ABOVE" if c["p_one_sided"] < 0.05 else ""
        print(f"  {c['suite']} {c['model']:26s} {c['modality']:10s} n={c['n']:5d} EM {c['em']:5.1f}"
              f" prior {c['prior']:5.1f} diff {c['diff']:+5.1f} [{c['lo']:+.1f},{c['hi']:+.1f}]"
              f" p1={c['p_one_sided']:.4f} holm={c['p_holm']:.4f} {flag}")
    for c in above:
        print(f"\n  ABOVE: {c['suite']} {c['model']} {c['modality']}: top (task, format) cells")
        for d in c["by_task_format"][:6]:
            print(f"     {d['task']:28s} {d['format']:16s} n={d['n']:4d} EM {d['em']:5.1f} "
                  f"EM-prior {d['em_minus_prior']:+6.1f}")
        for e in c["examples"][:4]:
            print(f"     e.g. {e['format']}: {e['question'][:70]!r} key={e['key']!r} "
                  f"pred={e['prediction']!r} prior_hit={e['prior_hit']:.2f}")

    rows = []
    mids = []
    prev = None
    for c in cells:
        if prev and c["suite"] != prev:
            mids.append(len(rows))
        prev = c["suite"]
        if not c["n"]:
            rows.append([c["suite"], c["model"], c["modality"], "0", "--", "--", "--", "--"])
            continue
        rows.append([c["suite"], c["model"], c["modality"], str(c["n"]), f"{c['em']:.1f}",
                     f"{c['prior']:.1f}", f"{c['diff']:+.1f} [{c['lo']:+.1f},{c['hi']:+.1f}]",
                     f"{c['p_one_sided']:.3f}" + ("$^{*}$" if c["p_one_sided"] < 0.05 else "")])
    rows = [[rc.Raw(x) if "$" in x else x for x in r] for r in rows]
    rc.write_tabular(
        os.path.join(ROOT, "paper_tables", "tab_h1_bound.tex"),
        "Information upper bound: canonical EM on absent (A) pairs vs the question-only CV prior\n"
        "on the same pairs. v1 = default suite, v1 audit classes; v2 = expressive suite, strict\n"
        "classes (InternVL models on the 1,096-question subset). Diff = EM - prior (pp) with\n"
        "object-cluster 95% CI; p = one-sided bootstrap p for diff <= 0; * = above the prior at\n"
        "p < 0.05 (a leak or an audit error).",
        "l l l r r r l l", ["Suite", "Model", "Modality", "$n$", "EM", "Prior", "Diff [CI]", "$p$"],
        rows, midrules=mids)
    print("\nwrote scripts/analysis/h1_bound_results.json, paper_tables/tab_h1_bound.tex")


if __name__ == "__main__":
    main()
