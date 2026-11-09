#!/usr/bin/env python3
"""Appendix result tables of the paper, recomputed from the released predictions.

Stdlib only, no inference, deterministic (seed 0). Reads

    benchmark/realworld_test.jsonl                 base items (modality, task, data_id, source)
    results/full_<model>[_extracted].jsonl         one row per (question_id, viz_type)

and writes LaTeX fragments into paper_tables/ plus scripts/analysis/appendix_tables.json
holding every number that appears in the fragments.

    python scripts/analysis/appendix_tables.py [--score-field exact_match|exact|auto]

Default is exact_match, the field every other paper script scores with (verify_paper_numbers.py,
cluster_cis.py); in full_claude_extracted.jsonl 31 rows have exact=1 but exact_match=0.

Conventions follow scripts/verify_paper_numbers.py and scripts/analysis/cluster_cis.py:
exact match in percent, best/worst formats chosen once on the full sample, object-cluster
(data_id) percentile bootstrap with B=5000, object sign-flip permutation with B=10000 and
p=(hits+1)/(B+1), fresh random.Random(0) for every resampling call. The resampling functions
are imported from cluster_cis.py so the estimators are literally the same code.

Score field: rows are deduplicated by (question_id, viz_type), last row wins; the score is
`exact` when the row has it, else `exact_match`. Pass --score-field exact_match to reproduce
the paper's main-text convention exactly (the two fields differ on 31 Claude rows).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "results")
BENCH = os.path.join(ROOT, "benchmark", "realworld_test.jsonl")
TAB_DIR = os.path.join(ROOT, "paper_tables")
OUT_JSON = os.path.join(HERE, "appendix_tables.json")

sys.path.insert(0, HERE)
from cluster_cis import B_BOOT, B_PERM, SEED, cluster_boot, cluster_perm, clusters_from, point  # noqa: E402

# (display label, file, json key). The first four are the paper's core models.
MODELS = [
    ("GPT-4o", "full_gpt4o_extracted.jsonl", "gpt4o"),
    ("Gemini-2.0-Flash", "full_gemini_extracted.jsonl", "gemini20"),
    ("Qwen2.5-VL-7B", "full_qwen_extracted.jsonl", "qwen7b"),
    ("Claude Sonnet 4", "full_claude_extracted.jsonl", "claude"),
    ("Gemini-2.5-Flash", "full_gemini25.jsonl", "gemini25"),
    ("Qwen2.5-VL-32B", "full_qwen32b.jsonl", "qwen32b"),
    ("InternVL2.5-8B", "full_internvl.jsonl", "internvl"),
]
CORE = MODELS[:4]
MODS = ["tabular", "timeseries", "graph"]
FORMATS = {
    "tabular": ["table_image", "text_only", "heatmap", "bar_chart", "scatter_plot"],
    "timeseries": ["line_plot", "heatmap", "gaf", "recurrence_plot", "text_only"],
    "graph": ["node_link", "circular_layout", "adjacency_matrix", "text_only"],
}
SOURCES = ["synthetic", "realworld"]
REAL_SOURCES = {"scitabalign", "ett", "networkx_realworld"}
Q_TRUNC = 90


# ---------------------------------------------------------------- loading
def load_bench():
    meta = {}
    for ln in open(BENCH):
        if ln.strip():
            r = json.loads(ln)
            meta[r["question_id"]] = {
                "modality": r["modality"], "task": r["task"], "difficulty": r["difficulty"],
                "data_id": r.get("data_id", r["question_id"]), "source": r["source"],
                "question": r["question"], "answer": str(r["answer"]),
            }
    return meta


def load_preds(fn, field):
    """{(qid, viz): score}; dedupe by key, last row wins. Returns (scores, n_raw, n_field)."""
    p = os.path.join(RES, fn)
    scores, n_raw, n_field = {}, 0, 0
    for ln in open(p):
        if not ln.strip():
            continue
        r = json.loads(ln)
        n_raw += 1
        if field == "auto":
            if "exact" in r:
                n_field += 1
                s = r["exact"]
            else:
                s = r.get("exact_match", 0)
        else:
            s = r.get(field, 0)
        scores[(r["question_id"], r["viz_type"])] = float(s or 0)
    return scores, n_raw, n_field


def by_question(scores, meta, modality):
    g = defaultdict(dict)
    for (q, v), s in scores.items():
        if meta[q]["modality"] == modality:
            g[q][v] = s
    return g


def per_format(scores, meta, modality, pred=lambda q: True):
    acc = defaultdict(list)
    for (q, v), s in scores.items():
        if meta[q]["modality"] == modality and pred(q):
            acc[v].append(s)
    em = {k: 100 * sum(v) / len(v) for k, v in acc.items() if v}
    return em, {k: len(v) for k, v in acc.items()}


def best_worst(pf):
    fmts = sorted(pf)                         # deterministic tie-break: alphabetical
    b = max(fmts, key=lambda f: pf[f])
    w = min(fmts, key=lambda f: pf[f])
    return b, w


# ---------------------------------------------------------------- latex helpers
def esc(s):
    s = str(s)
    s = s.replace("\\", r"\textbackslash{}")
    for ch in ["&", "%", "_", "#", "$", "{", "}"]:
        s = s.replace(ch, "\\" + ch)
    s = s.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
    return s


def esc_br(s):
    """esc() plus a break opportunity after every underscore, for p{} columns."""
    return esc(s).replace(r"\_", r"\_\allowbreak{}")


def f1(x):
    return f"{x:.1f}"


def fp(p, hits):
    return r"$<10^{-4}$" if hits == 0 else f"{p:.4f}"


def caption_comment(text):
    return "% " + text.replace("\n", "\n% ") + "\n"


def write(path, body):
    with open(path, "w") as f:
        f.write(body)
    return os.path.relpath(path, ROOT)


# ---------------------------------------------------------------- tables
SPLIT_ROWS = 60   # stacked task tables taller than this also get a two-part split


def table_task_format(modality, models, meta, tasks, part=""):
    fmts = FORMATS[modality]
    names = ", ".join(lab for lab, _, _ in models)
    lines = [caption_comment(
        f"Exact match (%) per task and rendering format, {modality} modality"
        + (f" (part {part}: {names})" if part else ", four core models") + ". "
        f"Rows are tasks (n = base questions per task; every question is rendered in all "
        f"{len(fmts)} formats), the best format in each row is bold. "
        "Generated by scripts/analysis/appendix_tables.py; do not edit by hand."),
        r"\begingroup\scriptsize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l r " + "c" * len(fmts) + "}", r"\toprule",
        "Task & $n$ & " + " & ".join(esc(f) for f in fmts) + r" \\"]
    data = {}
    for mi, (label, key, scores) in enumerate(models):
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{" + str(2 + len(fmts)) + r"}{l}{\textit{" + esc(label) + r"}} \\")
        data[key] = {}
        for task, n in tasks:
            pf, cnt = per_format(scores, meta, modality, lambda q, t=task: meta[q]["task"] == t)
            vals = [pf.get(f, float("nan")) for f in fmts]
            best = max(v for v in vals if v == v)
            tie_all = len({round(v, 6) for v in vals}) == 1      # no bold when every format ties
            cells = [(r"\textbf{" + f1(v) + "}") if v == best and not tie_all else f1(v)
                     for v in vals]
            lines.append(esc(task) + f" & {n} & " + " & ".join(cells) + r" \\")
            data[key][task] = {"n": n, "em": {f: pf.get(f) for f in fmts},
                               "n_rows": {f: cnt.get(f) for f in fmts},
                               "best": [f for f, v in zip(fmts, vals) if v == best]}
    lines += [r"\bottomrule", r"\end{tabular}", r"\endgroup"]
    return "\n".join(lines) + "\n", data


def gap_rows(models, meta):
    rows = []
    for label, key, scores in models:
        for md in MODS:
            pf, cnt = per_format(scores, meta, md)
            b, w = best_worst(pf)
            bq = by_question(scores, meta, md)
            # file order, not sorted: cluster_cis.py indexes clusters in first-appearance
            # order, and the bootstrap draws depend on that order.
            items = [(q, s[b] - s[w]) for q, s in bq.items() if b in s and w in s]
            q2d = {q: meta[q]["data_id"] for q, _ in items}
            cl = clusters_from(items, q2d)
            ci = cluster_boot(cl)
            p, hits = cluster_perm(cl)
            rows.append({"model": label, "key": key, "modality": md, "best": b, "worst": w,
                         "best_em": pf[b], "worst_em": pf[w], "gap": point(cl),
                         "n_q": len(items), "n_obj": len(cl),
                         "ci_lo": ci["lo"], "ci_hi": ci["hi"], "perm_p": p, "perm_hits": hits})
    return rows


def table_gap_cis(rows):
    lines = [caption_comment(
        "Best-minus-worst format gap per model and modality (exact match, pp), all seven models. "
        "Best/worst formats are chosen once on the full sample. 95% CI: percentile bootstrap over "
        f"source objects (data_id; B={B_BOOT}, seed {SEED}). p: two-sided sign-flip permutation "
        f"over objects (B={B_PERM}, p=(hits+1)/(B+1)). n = questions / objects. "
        "Generated by scripts/analysis/appendix_tables.py; do not edit by hand."),
        r"\begingroup\scriptsize\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l l l r r r c r r}", r"\toprule",
        r"Model & Best & Worst & Best EM & Worst EM & Gap & 95\% CI & $n_q$ / $n_{obj}$ & $p$ \\"]
    for md in MODS:
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{9}{l}{\textit{" + esc(md) + r"}} \\")
        for r in [x for x in rows if x["modality"] == md]:
            lines.append(" & ".join([
                esc(r["model"]), esc(r["best"]), esc(r["worst"]),
                f1(r["best_em"]), f1(r["worst_em"]), f1(r["gap"]),
                f"[{f1(r['ci_lo'])}, {f1(r['ci_hi'])}]", f"{r['n_q']} / {r['n_obj']}",
                fp(r["perm_p"], r["perm_hits"])]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\endgroup"]
    return "\n".join(lines) + "\n"


def flip_rows(models, meta):
    rows = []
    for label, key, scores in models:
        for md in MODS + ["all"]:
            per_q = []
            for m in (MODS if md == "all" else [md]):
                for q, s in sorted(by_question(scores, meta, m).items()):
                    per_q.append(list(s.values()))
            n = len(per_q)
            n_rows = sum(len(v) for v in per_q)
            rows.append({
                "model": label, "key": key, "modality": md, "n_q": n, "n_rows": n_rows,
                "mean_em": 100 * sum(sum(v) for v in per_q) / n_rows,
                "random_em": 100 * sum(sum(v) / len(v) for v in per_q) / n,
                "oracle_em": 100 * sum(max(v) for v in per_q) / n,
                "all_correct": 100 * sum(min(v) for v in per_q) / n,
                "flip": 100 * sum(1 for v in per_q if len(set(v)) > 1) / n,
            })
    return rows


def table_flip(rows):
    lines = [caption_comment(
        "Per-question consistency across formats, all seven models. n = base questions; "
        "Mean EM = exact match over all (question, format) rows; Random = expected EM when one "
        "format is drawn uniformly per question; Oracle = correct in at least one format; "
        "All-correct = correct in every format; Flip = share of questions whose correctness "
        "differs across formats (= 100 - All-correct - all-wrong). Mean EM and Random coincide "
        "because every question is rendered in every format of its modality. "
        "Generated by scripts/analysis/appendix_tables.py; do not edit by hand."),
        r"\begingroup\small",
        r"\begin{tabular}{l l r r r r r r}", r"\toprule",
        r"Model & Modality & $n$ & Mean EM & Random & Oracle & All-correct & Flip \\", r"\midrule"]
    last = None
    for r in rows:
        if last is not None and r["model"] != last:
            lines.append(r"\midrule")
        last = r["model"]
        lines.append(" & ".join([
            esc(r["model"]), esc(r["modality"]), str(r["n_q"]), f1(r["mean_em"]),
            f1(r["random_em"]), f1(r["oracle_em"]), f1(r["all_correct"]), f1(r["flip"])]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\endgroup"]
    return "\n".join(lines) + "\n"


def source_rows(models, meta):
    rows = []
    for label, key, scores in models:
        for md in MODS:
            for src in SOURCES:
                if src == "synthetic":
                    pred = lambda q: meta[q]["source"] == "synthetic"  # noqa: E731
                else:
                    pred = lambda q: meta[q]["source"] in REAL_SOURCES  # noqa: E731
                pf, cnt = per_format(scores, meta, md, pred)
                if not pf:
                    continue
                b, w = best_worst(pf)
                n_q = len({q for (q, v) in scores if meta[q]["modality"] == md and pred(q)})
                names = sorted({meta[q]["source"] for (q, v) in scores
                                if meta[q]["modality"] == md and pred(q)})
                rows.append({"model": label, "key": key, "modality": md, "source": src,
                             "source_names": names, "n_q": n_q, "best": b, "worst": w,
                             "best_em": pf[b], "worst_em": pf[w], "gap": pf[b] - pf[w],
                             "em": {f: pf.get(f) for f in FORMATS[md]}})
    return rows


def table_source(rows):
    lines = [caption_comment(
        "Best-minus-worst format gap (pp) on the synthetic and real-world source subsets, four "
        "core models. Best/worst formats are re-chosen inside each subset. Real-world sources: "
        "SciTabAlign (tabular), ETT (840) plus 30 NetworkX degree-sequence series labelled ett "
        "(time series), NetworkX real-world graphs (graph). n = base "
        "questions in the subset. Generated by scripts/analysis/appendix_tables.py; do not edit by hand."),
        r"\begingroup\scriptsize\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l l l r l l r r r}", r"\toprule",
        r"Model & Modality & Source & $n$ & Best & Worst & Best EM & Worst EM & Gap \\", r"\midrule"]
    last = None
    for r in rows:
        if last is not None and r["model"] != last:
            lines.append(r"\midrule")
        last = r["model"]
        lines.append(" & ".join([
            esc(r["model"]), esc(r["modality"]), esc("real-world" if r["source"] == "realworld" else r["source"]),
            str(r["n_q"]), esc(r["best"]), esc(r["worst"]), f1(r["best_em"]), f1(r["worst_em"]),
            f1(r["gap"])]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\endgroup"]
    return "\n".join(lines) + "\n"


def example_rows(meta, tasks_by_mod):
    rows = []
    for md in MODS:
        for task, n in tasks_by_mod[md]:
            qid = min(q for q, m in meta.items() if m["modality"] == md and m["task"] == task)
            m = meta[qid]
            q = m["question"]
            trunc = len(q) > Q_TRUNC
            rows.append({"modality": md, "task": task, "n": n, "question_id": qid,
                         "difficulty": m["difficulty"],
                         "question": q[:Q_TRUNC] + ("..." if trunc else ""),
                         "question_truncated": trunc, "answer": m["answer"]})
    return rows


def table_examples(rows):
    lines = [caption_comment(
        f"One example per (modality, task): the lexicographically first question_id of the task. "
        f"Questions longer than {Q_TRUNC} characters are truncated. n = base questions in the task. "
        "Generated by scripts/analysis/appendix_tables.py; do not edit by hand."),
        r"\begingroup\scriptsize\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{p{0.5in} p{1.0in} p{0.7in} p{2.0in} p{0.6in} r}", r"\toprule",
        r"Modality & Task & Difficulty & Example question & Answer & $n$ \\", r"\midrule"]
    last = None
    for r in rows:
        if last is not None and r["modality"] != last:
            lines.append(r"\midrule")
        last = r["modality"]
        q = esc(r["question"][:-3]) + r"\ldots" if r["question_truncated"] else esc(r["question"])
        lines.append(" & ".join([esc(r["modality"]), esc_br(r["task"]), esc(r["difficulty"]), q,
                                 esc_br(r["answer"]), str(r["n"])]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\endgroup"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-field", default="exact_match", choices=["auto", "exact", "exact_match"],
                    help="auto = `exact` if the row has it, else `exact_match` (default)")
    args = ap.parse_args()
    os.makedirs(TAB_DIR, exist_ok=True)

    meta = load_bench()
    models, loading = [], {}
    for label, fn, key in MODELS:
        scores, n_raw, n_field = load_preds(fn, args.score_field)
        models.append((label, key, scores))
        loading[key] = {"file": fn, "rows_raw": n_raw, "rows_dedup": len(scores),
                        "rows_with_exact_field": n_field}
    core = models[:4]
    tasks_by_mod = {}
    for md in MODS:
        c = Counter(m["task"] for m in meta.values() if m["modality"] == md)
        tasks_by_mod[md] = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))

    written, out = [], {"config": {
        "score_field": args.score_field, "seed": SEED, "B_boot": B_BOOT, "B_perm": B_PERM,
        "n_questions": len(meta), "n_objects": len({m["data_id"] for m in meta.values()}),
        "formats": FORMATS, "models": [(lab, k) for lab, _, k in MODELS], "loading": loading}}

    out["task_format"] = {}
    for md in MODS:
        body, data = table_task_format(md, core, meta, tasks_by_mod[md])
        written.append(write(os.path.join(TAB_DIR, f"tab_task_format_{md}.tex"), body))
        out["task_format"][md] = {"tasks": tasks_by_mod[md], "models": data, "split": False}
        if len(tasks_by_mod[md]) * len(core) > SPLIT_ROWS:
            # too tall for one page when stacked: also emit two two-model parts
            out["task_format"][md]["split"] = True
            for i, half in enumerate((core[:2], core[2:]), 1):
                b, _ = table_task_format(md, half, meta, tasks_by_mod[md], part=str(i))
                written.append(write(os.path.join(TAB_DIR, f"tab_task_format_{md}_part{i}.tex"), b))

    g = gap_rows(models, meta)
    written.append(write(os.path.join(TAB_DIR, "tab_gap_cis.tex"), table_gap_cis(g)))
    out["gap_cis"] = g

    fl = flip_rows(models, meta)
    written.append(write(os.path.join(TAB_DIR, "tab_flip_rates.tex"), table_flip(fl)))
    out["flip_rates"] = fl

    sr = source_rows(core, meta)
    written.append(write(os.path.join(TAB_DIR, "tab_source_split.tex"), table_source(sr)))
    out["source_split"] = sr

    ex = example_rows(meta, tasks_by_mod)
    written.append(write(os.path.join(TAB_DIR, "tab_task_examples.tex"), table_examples(ex)))
    out["task_examples"] = ex

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1)
    written.append(os.path.relpath(OUT_JSON, ROOT))

    # ---- summary
    print(f"appendix_tables.py  score_field={args.score_field}  seed={SEED}  "
          f"B_boot={B_BOOT}  B_perm={B_PERM}")
    print("files written:")
    for w in written:
        print("  " + w)
    print("\nbest-worst gaps (pp) with object-cluster 95% CI and sign-flip p:")
    for r in g:
        print(f"  {r['model']:17s} {r['modality']:10s} {r['best']:>16s} - {r['worst']:<16s} "
              f"{f1(r['best_em']):>5s} - {f1(r['worst_em']):>5s} = {f1(r['gap']):>5s}  "
              f"[{f1(r['ci_lo'])}, {f1(r['ci_hi'])}]  p={fp(r['perm_p'], r['perm_hits']).replace('$', '')}"
              f"  n={r['n_q']}/{r['n_obj']}")
    print(f"\nlargest within-modality gap: {max(r['gap'] for r in g):.1f} pp")
    print("\nflip rate (%) per model, all modalities pooled | tabular / timeseries / graph:")
    for label, key, _ in models:
        d = {r["modality"]: r for r in fl if r["key"] == key}
        print(f"  {label:17s} {f1(d['all']['flip']):>5s} | "
              + " / ".join(f1(d[m]["flip"]) for m in MODS)
              + f"   (oracle {f1(d['all']['oracle_em'])}, all-correct {f1(d['all']['all_correct'])},"
              f" mean {f1(d['all']['mean_em'])})")
    cf = [r["flip"] for r in fl if r["modality"] == "all" and r["key"] in {k for _, _, k in CORE}]
    af = [r["flip"] for r in fl if r["modality"] == "all"]
    print(f"  four core models: {min(cf):.1f}-{max(cf):.1f}%; all seven: {min(af):.1f}-{max(af):.1f}%")
    print("\nsource split (four core models), gap synthetic vs real-world:")
    for label, key, _ in core:
        parts = []
        for md in MODS:
            d = {r["source"]: r for r in sr if r["key"] == key and r["modality"] == md}
            parts.append(f"{md} {f1(d['synthetic']['gap'])} (n={d['synthetic']['n_q']}) vs "
                         f"{f1(d['realworld']['gap'])} (n={d['realworld']['n_q']})")
        print(f"  {label:17s} " + "; ".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
