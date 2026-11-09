#!/usr/bin/env python3
"""H7, signal detection on yes/no templates: do the lossy time-series transforms (GAF,
recurrence plot) carry ANY information, or do models only answer with a bias?

Stdlib only; no inference.

    python scripts/analysis/h7_sdt.py

For every yes/no template (modality, task with key in {yes, no}) and every format, per model:
hit rate H = P(answer "yes" | key yes), false-alarm rate F = P(answer "yes" | key no), with a
log-linear correction ((k + 0.5) / (n + 1)); d' = z(H) - z(F); criterion c = -(z(H) + z(F)) / 2
(c < 0 = "yes" bias). Any answer other than "yes" (after canonical folding) counts as not-yes;
the share of non-yes/no answers is reported. 95% CIs: bootstrap over objects (data_id), B=2000,
random.Random(0), recomputing H, F, d', c in each resample. d' CI including 0 = no detectable
information; d' > 0 with |c| >= 0.5 = information plus strong bias; "degenerate" = the model
says "yes" on < 5% or > 95% of the items (d' is then driven by the correction, not by
information); "below chance" = d' CI entirely below 0.

Eligible templates: both classes >= 10 questions (tabular comparison [yes/no subset],
tabular counterfactual, time-series mean_half_comparison, graph counterfactual). Skipped
(listed in the output): templates where one class has < 10 items. Because only
mean_half_comparison is eligible on GAF / recurrence, the time-series yes/no templates are
also POOLED (all time-series yes/no templates, any class size) as a secondary analysis; pooled
d' mixes templates with different base rates and is labelled as such.
Models: the seven v1 models (default suite) and Qwen2.5-VL-7B / 32B / Qwen3-VL-8B on the
expressive suite (GAF and recurrence images are identical in both suites).

Writes scripts/analysis/h7_sdt_results.json and paper_tables/tab_h7_sdt.tex.
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import review_common as rc  # noqa: E402
import scoring as sc  # noqa: E402
import v2_analysis as v2a  # noqa: E402
import v2_multimodel as mm  # noqa: E402

B = 2000
MIN_CLASS = 10
ND = statistics.NormalDist()


def z(p):
    return ND.inv_cdf(p)


def sdt(items):
    """items: [(did, key_yes: bool, said_yes: bool)] -> dict."""
    ny = sum(1 for _, k, _ in items if k)
    nn = len(items) - ny
    h = sum(1 for _, k, s in items if k and s)
    f = sum(1 for _, k, s in items if not k and s)
    H = (h + 0.5) / (ny + 1)
    F = (f + 0.5) / (nn + 1)
    return {"n_yes": ny, "n_no": nn, "H": 100 * h / ny if ny else float("nan"),
            "F": 100 * f / nn if nn else float("nan"), "dprime": z(H) - z(F),
            "c": -(z(H) + z(F)) / 2}


def boot(items):
    by = defaultdict(list)
    for it in items:
        by[it[0]].append(it)
    objs = list(by)
    rng = random.Random(0)
    ds, cs = [], []
    for _ in range(B):
        res = []
        for _ in range(len(objs)):
            res.extend(by[objs[int(rng.random() * len(objs))]])
        if sum(1 for _, k, _ in res if k) == 0 or sum(1 for _, k, _ in res if not k) == 0:
            continue
        s = sdt(res)
        ds.append(s["dprime"])
        cs.append(s["c"])
    return v2a.pct(ds), v2a.pct(cs)


def said(r):
    p = sc.fold(sc.strip_index(r.get("prediction", "")))
    return "yes" if p in ("yes", "true") else "no" if p in ("no", "false") else "other"


def main():
    meta = rc.load_bench()
    yn = defaultdict(Counter)
    for q, b in meta.items():
        a = rc.norm(b["answer"])
        if a in ("yes", "no"):
            yn[(b["modality"], b["task"])][a] += 1
    eligible = {k for k, v in yn.items() if v["yes"] >= MIN_CLASS and v["no"] >= MIN_CLASS}
    skipped = {f"{k[0]}|{k[1]}": dict(v) for k, v in yn.items() if k not in eligible}
    ts_pool = {k for k in yn if k[0] == "timeseries"}

    sources = [("v1", m, rc.load_v1(m)) for m in rc.MODELS7]
    for spec in mm.MODELS[:3]:
        rows, _ = v2a.dedupe(rc.read_jsonl(os.path.join(ROOT, spec["v2"])))
        sources.append(("v2", spec["name"], rows))

    out = []
    for suite, m, rows in sources:
        groups = defaultdict(list)
        other = Counter()
        for (q, f), r in rows.items():
            b = meta[q]
            key = rc.norm(b["answer"])
            if key not in ("yes", "no"):
                continue
            cell = (b["modality"], b["task"])
            s = said(r)
            it = (b["data_id"], key == "yes", s == "yes")
            if cell in eligible:
                groups[(f"{cell[0]}|{cell[1]}", f)].append(it)
                other[(f"{cell[0]}|{cell[1]}", f)] += s == "other"
            if cell in ts_pool:
                groups[("timeseries|ALL_YESNO_POOLED", f)].append(it)
                other[("timeseries|ALL_YESNO_POOLED", f)] += s == "other"
        for (tpl, f), items in sorted(groups.items()):
            if f not in rc.FORMATS[tpl.split("|")[0]]:
                continue
            s = sdt(items)
            (dlo, dhi), (clo, chi) = boot(items)
            s.update(suite=suite, model=m, template=tpl, format=f, d_lo=dlo, d_hi=dhi,
                     c_lo=clo, c_hi=chi, n=len(items), n_obj=len({i[0] for i in items}),
                     other_pct=100 * other[(tpl, f)] / len(items),
                     yes_share=100 * sum(1 for i in items if i[2]) / len(items),
                     verdict=("degenerate (one answer on >95% of items)"
                              if not 5 <= 100 * sum(1 for i in items if i[2]) / len(items) <= 95
                              else "no information (d' CI includes 0)" if dlo <= 0 <= dhi else
                              "information, strong bias" if dlo > 0 and abs(s["c"]) >= 0.5
                              else "information" if dlo > 0 else "below chance"))
            out.append(s)

    res = {"definition": __doc__, "eligible_templates": sorted(f"{a}|{b}" for a, b in eligible),
           "skipped_templates": skipped, "rows": out}
    with open(os.path.join(HERE, "h7_sdt_results.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)

    print(f"eligible (both classes >= {MIN_CLASS}): {res['eligible_templates']}")
    print(f"skipped: {skipped}")
    for tpl in sorted({r["template"] for r in out}):
        print(f"\n[{tpl}]")
        for r in out:
            if r["template"] != tpl:
                continue
            print(f"  {r['suite']} {r['model']:17s} {r['format']:16s} n={r['n']:3d} "
                  f"(yes {r['n_yes']}/no {r['n_no']}) H {r['H']:5.1f} F {r['F']:5.1f} d' "
                  f"{r['dprime']:+.2f} [{r['d_lo']:+.2f},{r['d_hi']:+.2f}] c {r['c']:+.2f} "
                  f"[{r['c_lo']:+.2f},{r['c_hi']:+.2f}] other {r['other_pct']:.0f}% -> {r['verdict']}")
    summ = defaultdict(Counter)
    for r in out:
        summ[(r["template"], r["format"])][r["verdict"].split(" (")[0]] += 1
    print("\n[summary: verdict counts per template x format over models]")
    for k, v in sorted(summ.items()):
        print(f"  {k[0]:36s} {k[1]:16s} {dict(v)}")
    res["summary"] = {f"{k[0]}|{k[1]}": dict(v) for k, v in summ.items()}
    with open(os.path.join(HERE, "h7_sdt_results.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)

    rows_t, mids = [], []
    prev = None
    keep_tpl = ["timeseries|mean_half_comparison", "timeseries|ALL_YESNO_POOLED"]
    for r in out:
        if r["template"] not in keep_tpl or r["suite"] != "v1":
            continue
        if prev and r["template"] != prev:
            mids.append(len(rows_t))
        prev = r["template"]
        rows_t.append([r["template"].split("|")[1], r["model"], r["format"], str(r["n"]),
                       f"{r['H']:.0f}", f"{r['F']:.0f}",
                       f"{r['dprime']:+.2f} [{r['d_lo']:+.2f},{r['d_hi']:+.2f}]",
                       f"{r['c']:+.2f}", f"{r['other_pct']:.0f}"])
    rc.write_tabular(
        os.path.join(ROOT, "paper_tables", "tab_h7_sdt.tex"),
        "Signal detection on time-series yes/no questions, default-suite models. H / F = share\n"
        "of 'yes' answers when the key is yes / no (%); d' = z(H) - z(F) (log-linear correction)\n"
        "with object-bootstrap 95% CI; c = criterion (negative = 'yes' bias); Other = answers\n"
        "that are neither yes nor no (%). mean_half_comparison is the only time-series yes/no\n"
        "template with >= 10 items per class; POOLED combines all time-series yes/no templates\n"
        "(different base rates). Full results incl. tabular / graph templates and v2 models in\n"
        "scripts/analysis/h7_sdt_results.json.",
        "l l l r r r l r r",
        ["Template", "Model", "Format", "$n$", "H", "F", "$d'$ [CI]", "$c$", "Other"],
        rows_t, midrules=mids)
    print("\nwrote scripts/analysis/h7_sdt_results.json, paper_tables/tab_h7_sdt.tex")


if __name__ == "__main__":
    main()
