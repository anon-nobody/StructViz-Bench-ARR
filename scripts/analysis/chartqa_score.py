#!/usr/bin/env python3
"""Score ChartQA external-validity predictions (real charts, human questions, format control).

Reads every prediction file results/review/chartqa/pred_*.jsonl (eval_local_suite.py output
``pred_<model>_chartqa.jsonl`` and eval_lora_chartqa.py output ``pred_lora_<setting>_<name>.jsonl``)
and the item file results/review/chartqa/chartqa_items.jsonl (chart = data_id = cluster).

Metrics, recomputed from the stored ``prediction`` string for every file alike:
  * relaxed accuracy (ChartQA; Masry et al., 2022, as implemented in Pix2Struct/DePlot): if both
    strings parse as floats (a trailing % divides by 100) and gold != 0, correct iff
    |pred-gold|/|gold| <= 5%; else case-insensitive exact string match (both stripped, one
    trailing period removed);
  * our exact match: src.evaluation.metrics.compute_metrics(...).exact (benchmark EM).
Per format: accuracy with object(chart)-cluster bootstrap 95% CI. Best-worst gap over the five
re-renderings (best/worst chosen on the full sample, held fixed in resamples) with cluster
bootstrap CI and cluster sign-flip permutation p; flip rate (share of questions whose
correctness is not identical across the five re-renderings; also across all six);
original_chart minus best re-rendering (CI, p). Helpers reused from cluster_cis.py
(B=5000 bootstraps, B=10000 permutations, seed 0). Analyses use complete questions only
(every format present and non-error); a second block excludes questions whose wording refers
to colours/positions of the original chart (metadata.visual_reference).

    .venv/bin/python scripts/analysis/chartqa_score.py
Writes scripts/analysis/chartqa_results.json and paper_tables/tab_chartqa.tex.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from cluster_cis import cluster_boot, cluster_perm, clusters_from, point  # noqa: E402

RERENDER = ["bar_chart", "heatmap", "table_image", "scatter_plot", "text_only"]
ORIGINAL = "original_chart"
VIZ_TEX = {"bar_chart": "Bar", "heatmap": "Heat", "table_image": "Table",
           "scatter_plot": "Scatter", "text_only": "Text", ORIGINAL: "Orig."}


def _to_float(text: str) -> float | None:
    try:
        if text.endswith("%"):
            return float(text.rstrip("%")) / 100.0
        return float(text)
    except ValueError:
        return None


def _clean(s: str) -> str:
    s = str(s).strip()
    return s[:-1].strip() if s.endswith(".") else s


def relaxed_correct(prediction: str, target: str, max_rel: float = 0.05) -> bool:
    """ChartQA relaxed accuracy (official 5% relative tolerance, else exact string)."""
    p, t = _clean(prediction), _clean(target)
    pf, tf = _to_float(p), _to_float(t)
    if pf is not None and tf:
        return abs(pf - tf) / abs(tf) <= max_rel
    return p.lower() == t.lower()


def em_correct(prediction: str, target: str) -> bool:
    from src.evaluation.metrics import compute_metrics
    return bool(compute_metrics(prediction=str(prediction), answer=str(target)).exact)


def read_jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def model_name(path: Path) -> str:
    m = re.match(r"pred_(.+?)(_chartqa)?\.jsonl$", path.name)
    return m.group(1) if m else path.stem


def summ(cl: list, perm: bool = False) -> dict:
    out = {"est": point(cl), "n_obj": len(cl), "n_q": sum(n for _, n in cl),
           "ci": cluster_boot(cl)}
    if perm:
        out["perm_p"], out["perm_hits"] = cluster_perm(cl)
    return out


def analyse(scores: dict, q2d: dict, qids: list[str], fmts_present: list[str]) -> dict:
    """scores[qid][viz] -> 0/1 for one metric; qids = complete questions."""
    res: dict = {"n_questions": len(qids), "n_charts": len({q2d[q] for q in qids})}
    per = {}
    for v in fmts_present:
        per[v] = summ(clusters_from(((q, scores[q][v]) for q in qids), q2d))
    res["per_format"] = per
    rr = [v for v in RERENDER if v in fmts_present]
    if len(rr) >= 2:
        best = max(rr, key=lambda v: per[v]["est"])
        worst = min(rr, key=lambda v: per[v]["est"])
        res["best_rerender"], res["worst_rerender"] = best, worst
        res["gap_best_worst"] = summ(clusters_from(
            ((q, scores[q][best] - scores[q][worst]) for q in qids), q2d), perm=True)
        res["flip_rate_rerender"] = summ(clusters_from(
            ((q, int(len({scores[q][v] for v in rr}) > 1)) for q in qids), q2d))
        if ORIGINAL in fmts_present:
            allf = rr + [ORIGINAL]
            res["flip_rate_all6"] = summ(clusters_from(
                ((q, int(len({scores[q][v] for v in allf}) > 1)) for q in qids), q2d))
            res["original_minus_best_rerender"] = summ(clusters_from(
                ((q, scores[q][ORIGINAL] - scores[q][best]) for q in qids), q2d), perm=True)
            res["original_minus_worst_rerender"] = summ(clusters_from(
                ((q, scores[q][ORIGINAL] - scores[q][worst]) for q in qids), q2d), perm=True)
    return res


def score_file(path: Path, items: dict, q2d: dict) -> dict:
    rows = read_jsonl(path)
    by_key = {}
    n_err = 0
    for r in rows:
        if str(r.get("prediction")) == "[ERROR]" or r.get("error"):
            n_err += 1
            continue
        if r.get("question_id") in items:
            by_key[(r["question_id"], r["viz_type"])] = r  # last row wins
    fmts = sorted({v for _, v in by_key}, key=lambda v: (RERENDER + [ORIGINAL]).index(v)
                  if v in RERENDER + [ORIGINAL] else 99)
    qids_all = sorted({q for q, _ in by_key})
    complete = [q for q in qids_all if all((q, v) in by_key for v in fmts)]
    out = {"file": os.path.relpath(path, ROOT), "n_rows": len(rows), "n_error_rows": n_err,
           "formats": fmts, "n_questions_seen": len(qids_all), "n_complete": len(complete)}
    metrics = {}
    for mname, fn in [("relaxed", relaxed_correct), ("em", em_correct)]:
        sc = defaultdict(dict)
        for (q, v), r in by_key.items():
            sc[q][v] = int(fn(str(r.get("prediction", "")), str(items[q]["answer"])))
        no_vis = [q for q in complete if not items[q]["metadata"].get("visual_reference")]
        metrics[mname] = {"all": analyse(sc, q2d, complete, fmts),
                          "no_visual_reference": analyse(sc, q2d, no_vis, fmts)}
        # per chart type, per-format point accuracies only
        bytype = defaultdict(list)
        for q in complete:
            bytype[items[q]["metadata"]["chart_type"]].append(q)
        metrics[mname]["per_chart_type"] = {
            t: {"n": len(qs), **{v: 100 * sum(sc[q][v] for q in qs) / len(qs) for v in fmts}}
            for t, qs in sorted(bytype.items())}
    out["metrics"] = metrics
    return out


def ci(c: dict) -> str:
    return f"[{c['lo']:.1f}, {c['hi']:.1f}]"


def fmt_p(s: dict) -> str:
    p = s["perm_p"]
    return r"$<$1e-4" if s["perm_hits"] == 0 else f"{p:.3f}" if p >= 1e-3 else f"{p:.1e}"


def write_tex(results: dict, path: Path, counts: dict) -> None:
    lines = [
        "% Auto-generated by scripts/analysis/chartqa_score.py -- do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{External validity on ChartQA \citep{masry2022chartqa} (test, human-written "
        rf"questions; {counts['n_q']} questions on {counts['n_c']} real charts whose released table "
        r"parses cleanly). Each table is re-rendered in our five formats and compared with the "
        r"original chart. Relaxed accuracy (\%, 5\% numeric tolerance). Gap: best$-$worst "
        r"re-rendering, 95\% chart-cluster bootstrap CI and sign-flip permutation $p$. "
        r"Flip: questions whose correctness changes across the five re-renderings. "
        r"Orig.$-$best: original chart minus the best re-rendering.}",
        r"\label{tab:chartqa}",
        r"\begin{tabular}{l" + "r" * 6 + "rlrrr" + "}",
        r"\toprule",
        "Model & " + " & ".join(VIZ_TEX[v] for v in RERENDER + [ORIGINAL])
        + r" & Gap & 95\% CI & $p$ & Flip & Orig.$-$best \\",
        r"\midrule",
    ]
    for name, res in results.items():
        a = res["metrics"]["relaxed"]["all"]
        pf = a["per_format"]
        cells = [f"{pf[v]['est']:.1f}" if v in pf else "--" for v in RERENDER + [ORIGINAL]]
        if "gap_best_worst" in a:
            g = a["gap_best_worst"]
            cells += [f"{g['est']:.1f}", ci(g["ci"]), fmt_p(g),
                      f"{a['flip_rate_rerender']['est']:.1f}"]
        else:
            cells += ["--"] * 4
        ob = a.get("original_minus_best_rerender")
        cells.append(f"{ob['est']:+.1f}" if ob else "--")
        lines.append(name.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "results/review/chartqa")
    ap.add_argument("--items", type=Path, default=ROOT / "results/review/chartqa/chartqa_items.jsonl")
    ap.add_argument("--out-json", type=Path, default=HERE / "chartqa_results.json")
    ap.add_argument("--out-tex", type=Path, default=ROOT / "paper_tables/tab_chartqa.tex")
    args = ap.parse_args()
    os.chdir(ROOT)
    items = {r["question_id"]: r for r in read_jsonl(args.items)}
    q2d = {q: r["data_id"] for q, r in items.items()}
    results = {}
    for path in sorted(args.pred_dir.glob("pred_*.jsonl")):
        results[model_name(path)] = score_file(path, items, q2d)
    build = json.loads((ROOT / "results/review/chartqa/build_counts.json").read_text())
    payload = {"n_items": len(items), "n_charts": len(set(q2d.values())),
               "build_counts": build, "models": results}
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out_json} ({len(results)} prediction files)")
    for name, res in results.items():
        a = res["metrics"]["relaxed"]["all"]
        pf = "  ".join(f"{v}={s['est']:.1f}" for v, s in a["per_format"].items())
        extra = ""
        if "gap_best_worst" in a:
            g = a["gap_best_worst"]
            extra = (f" | gap={g['est']:.1f} {ci(g['ci'])} p={g['perm_p']:.4f} "
                     f"flip={a['flip_rate_rerender']['est']:.1f}")
        if "original_minus_best_rerender" in a:
            o = a["original_minus_best_rerender"]
            extra += f" | orig-best={o['est']:+.1f} {ci(o['ci'])} p={o['perm_p']:.4f}"
        print(f"[{name}] n={a['n_questions']} relaxed: {pf}{extra}")
    if results:
        n = max(r["metrics"]["relaxed"]["all"]["n_questions"] for r in results.values())
        c = max(r["metrics"]["relaxed"]["all"]["n_charts"] for r in results.values())
        write_tex(results, args.out_tex, {"n_q": n, "n_c": c})
        print(f"wrote {args.out_tex}")
    else:
        print("no prediction files yet; table not written")


if __name__ == "__main__":
    main()
