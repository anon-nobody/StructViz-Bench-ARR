#!/usr/bin/env python3
"""Question-only prior baseline, near-constant-task removal, truncation contrast,
categorical scoring sensitivity, and per-object question counts.

Stdlib only; mirrors helpers in scripts/verify_paper_numbers.py. No inference.

    python scripts/analysis/prior_and_subsets.py

Writes scripts/analysis/prior_and_subsets_results.json. Seed 0 throughout.
Permutation test: paired sign-flip on per-question (best - worst) EM differences,
two-sided on |mean|, B=10,000, p = (hits+1)/(B+1). Because EM is 0/1, each paired
difference is in {-1,0,+1}; a sign-flipped sum over the k non-zero differences is
drawn as 2*popcount(getrandbits(k)) - k, which has exactly the same distribution
as flipping each sign with probability 1/2 (general path used if any |d| not in {0,1}).
"""
from __future__ import annotations

import json
import os
import random
import re
import statistics
from collections import Counter, defaultdict

ROOT = os.environ.get(
    "STRUCTVIZ_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
RES = os.path.join(ROOT, "results")
BENCH = os.path.join(ROOT, "benchmark", "realworld_test.jsonl")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prior_and_subsets_results.json")

CORE = {
    "GPT-4o": "full_gpt4o_extracted.jsonl",
    "Gemini Flash": "full_gemini_extracted.jsonl",
    "Qwen2.5-VL-7B": "full_qwen_extracted.jsonl",
    "Claude Sonnet": "full_claude_extracted.jsonl",
}
MODS = ["tabular", "timeseries", "graph"]
FMTS = {
    "tabular": ["bar_chart", "heatmap", "table_image", "scatter_plot", "text_only"],
    "timeseries": ["line_plot", "gaf", "recurrence_plot", "heatmap", "text_only"],
    "graph": ["node_link", "adjacency_matrix", "circular_layout", "text_only"],
}
SEED = 0
B_PERM = 10000
N_CV_SEEDS = 20
CONST_THRESH = 0.90
ROW_CUT = 12


# ---------------------------------------------------------------- helpers (mirrored)
def load(fn):
    p = os.path.join(RES, fn)
    return [json.loads(l) for l in open(p) if l.strip()]


def per_format(recs, modality, score=None):
    score = score or (lambda r: float(r.get("exact_match", 0)))
    acc = defaultdict(list)
    for r in recs:
        if r.get("modality") == modality:
            acc[r["viz_type"]].append(score(r))
    return {k: 100 * sum(v) / len(v) for k, v in acc.items() if v}


def by_question(recs, modality=None, score=None):
    score = score or (lambda r: float(r.get("exact_match", 0)))
    g = defaultdict(dict)
    for r in recs:
        if modality is None or r.get("modality") == modality:
            g[r["question_id"]][r["viz_type"]] = score(r)
    return g


def perm_p(paired, rng):
    """Paired sign-flip permutation, two-sided on |mean|, (hits+1)/(B+1)."""
    n = len(paired)
    d = [x - y for x, y in paired]
    obs = sum(d) / n
    thr = abs(obs) * n - 1e-9
    hits = 0
    if all(v in (0.0, 1.0, -1.0) for v in d):
        k = sum(1 for v in d if v != 0)
        for _ in range(B_PERM):
            s = 2 * bin(rng.getrandbits(k)).count("1") - k if k else 0
            if abs(s) >= thr:
                hits += 1
    else:
        for _ in range(B_PERM):
            tot = sum(v if rng.random() < 0.5 else -v for v in d)
            if abs(tot) >= thr:
                hits += 1
    return (hits + 1) / (B_PERM + 1)


def gap_cell(recs, modality, rng, score=None):
    pf = per_format(recs, modality, score)
    b, w = max(pf, key=pf.get), min(pf, key=pf.get)
    q = by_question(recs, modality, score)
    paired = [(s[b], s[w]) for s in q.values() if b in s and w in s]
    return {
        "per_format": pf, "best": b, "worst": w, "gap": pf[b] - pf[w],
        "p": perm_p(paired, rng), "n_questions": len(paired),
    }


def flip_rate(recs):
    multi = [s for s in by_question(recs).values() if len(s) >= 2]
    return 100 * sum(1 for s in multi if len(set(s.values())) > 1) / len(multi), len(multi)


def paired_diff(recs, modality, a, b):
    q = by_question(recs, modality)
    pr = [(v[a], v[b]) for v in q.values() if a in v and b in v]
    return 100 * sum(x - y for x, y in pr) / len(pr), len(pr)


def f1(x):
    return "nan" if x != x else f"{x:.1f}"


def fp(p):
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


def section(t):
    print(f"\n{t}\n" + "-" * 100)


def norm(a):
    return str(a).strip().lower()


# ---------------------------------------------------------------- main
def main():
    rng = random.Random(SEED)
    bench = [json.loads(l) for l in open(BENCH) if l.strip()]
    meta = {b["question_id"]: b for b in bench}
    core = {k: load(v) for k, v in CORE.items()}
    out = {"seed": SEED, "B_perm": B_PERM}

    # ============================================================ 1. prior baseline
    section("1(a). Question-only prior, in-sample majority answer per (modality, task)")
    cnt = defaultdict(Counter)
    for b in bench:
        cnt[(b["modality"], b["task"])][norm(b["answer"])] += 1

    def majority(counter):
        return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]

    maj = {k: majority(v)[0] for k, v in cnt.items()}
    ins = {}
    for md in MODS + ["overall"]:
        qs = [b for b in bench if md == "overall" or b["modality"] == md]
        hit = sum(norm(b["answer"]) == maj[(b["modality"], b["task"])] for b in qs)
        ins[md] = {"acc": 100 * hit / len(qs), "n": len(qs)}
        print(f"  {md:12s} n={len(qs):5d}  in-sample prior acc = {f1(ins[md]['acc'])}%")
    out["prior_in_sample"] = ins

    section(f"1(b). Cross-validated prior: 2-fold split over objects (data_id), {N_CV_SEEDS} seeds")
    objs = sorted({b["data_id"] for b in bench})
    obj_mods = defaultdict(set)
    for b in bench:
        obj_mods[b["data_id"]].add(b["modality"])
    n_multi = sum(1 for v in obj_mods.values() if len(v) > 1)
    print(f"  objects={len(objs)}; data_ids appearing in >1 modality: {n_multi}")
    cv_runs = {md: [] for md in MODS + ["overall"]}
    unseen_runs = []
    for s in range(N_CV_SEEDS):
        r2 = random.Random(s)
        sh = objs[:]
        r2.shuffle(sh)
        half = len(sh) // 2
        folds = [set(sh[:half]), set(sh[half:])]
        hits = Counter()
        tot = Counter()
        unseen = 0
        for i in (0, 1):
            tr = [b for b in bench if b["data_id"] in folds[i]]
            te = [b for b in bench if b["data_id"] in folds[1 - i]]
            c = defaultdict(Counter)
            for b in tr:
                c[(b["modality"], b["task"])][norm(b["answer"])] += 1
            m = {k: majority(v)[0] for k, v in c.items()}
            for b in te:
                k = (b["modality"], b["task"])
                pred = m.get(k)
                if pred is None:
                    unseen += 1
                h = int(pred is not None and norm(b["answer"]) == pred)
                for md in (b["modality"], "overall"):
                    hits[md] += h
                    tot[md] += 1
        for md in MODS + ["overall"]:
            cv_runs[md].append(100 * hits[md] / tot[md])
        unseen_runs.append(unseen)
    cv = {}
    for md in MODS + ["overall"]:
        v = cv_runs[md]
        cv[md] = {"mean": statistics.mean(v), "sd": statistics.stdev(v),
                  "min": min(v), "max": max(v)}
        print(f"  {md:12s} CV prior acc mean={f1(cv[md]['mean'])}%  sd={f1(cv[md]['sd'])}"
              f"  range=[{f1(cv[md]['min'])}, {f1(cv[md]['max'])}]")
    print(f"  test questions whose (modality,task) is absent from the training fold "
          f"(scored wrong): mean {statistics.mean(unseen_runs):.1f} per seed (of {len(bench)})")
    out["prior_cv"] = cv
    out["prior_cv_unseen_task_questions_mean"] = statistics.mean(unseen_runs)

    section(f"1(c). (modality, task) whose most frequent answer covers >= {int(CONST_THRESH*100)}%")
    const = []
    for k in sorted(cnt):
        a, c = majority(cnt[k])
        n = sum(cnt[k].values())
        if c / n >= CONST_THRESH:
            const.append({"modality": k[0], "task": k[1], "answer": a, "count": c, "n": n,
                          "share": 100 * c / n})
            print(f"  {k[0]:10s} {k[1]:28s} answer={a!r:8s} {c:4d}/{n:<4d} ({f1(100*c/n)}%)")
    const_keys = {(c["modality"], c["task"]) for c in const}
    n_const = sum(c["n"] for c in const)
    by_md = Counter()
    for c in const:
        by_md[c["modality"]] += c["n"]
    print(f"  total questions in these tasks: {n_const} of {len(bench)}  "
          + "  ".join(f"{m}={by_md[m]}" for m in MODS))
    out["near_constant_tasks"] = const
    out["near_constant_total_questions"] = n_const
    out["near_constant_questions_by_modality"] = dict(by_md)

    section("1(d). Best single-format EM vs cross-validated prior")
    print(f"  {'model':15s} {'modality':11s} {'best fmt':17s} {'best EM':>8s} {'CV prior':>9s} {'diff':>7s}")
    d1 = {}
    for name, recs in core.items():
        d1[name] = {}
        for md in MODS:
            pf = per_format(recs, md)
            b = max(pf, key=pf.get)
            d1[name][md] = {"best_format": b, "best_em": pf[b], "cv_prior": cv[md]["mean"],
                            "diff": pf[b] - cv[md]["mean"]}
            print(f"  {name:15s} {md:11s} {b:17s} {f1(pf[b]):>8s} {f1(cv[md]['mean']):>9s}"
                  f" {f1(pf[b]-cv[md]['mean']):>7s}")
    out["best_format_vs_prior"] = d1

    # ============================================================ 2. near-constant removed
    section("2. Best-worst gap: full set vs near-constant tasks removed (paired sign-flip, B=10,000)")
    keep = lambda r: (r["modality"], r["task"]) not in const_keys  # noqa: E731
    res2 = {}
    print(f"  {'model':15s} {'modality':11s} | {'n':>5s} {'best':15s} {'worst':15s} {'gap':>5s} {'p':>7s}"
          f" | {'n':>5s} {'best':15s} {'worst':15s} {'gap':>5s} {'p':>7s}")
    for name, recs in core.items():
        res2[name] = {}
        sub = [r for r in recs if keep(r)]
        for md in MODS:
            a = gap_cell(recs, md, rng)
            b = gap_cell(sub, md, rng)
            res2[name][md] = {"full": a, "filtered": b}
            print(f"  {name:15s} {md:11s} | {a['n_questions']:5d} {a['best']:15s} {a['worst']:15s}"
                  f" {f1(a['gap']):>5s} {fp(a['p']):>7s} | {b['n_questions']:5d} {b['best']:15s}"
                  f" {b['worst']:15s} {f1(b['gap']):>5s} {fp(b['p']):>7s}")
    print("\n  per-format EM (full -> filtered)")
    for name in core:
        for md in MODS:
            a, b = res2[name][md]["full"]["per_format"], res2[name][md]["filtered"]["per_format"]
            print(f"  {name:15s} {md:11s} " + "  ".join(
                f"{f}={f1(a[f])}->{f1(b[f])}" for f in FMTS[md]))
    print("\n  per-question flip rate (full -> filtered)")
    fl = {}
    for name, recs in core.items():
        fa, na = flip_rate(recs)
        fb, nb = flip_rate([r for r in recs if keep(r)])
        fl[name] = {"full": fa, "n_full": na, "filtered": fb, "n_filtered": nb}
        print(f"  {name:15s} {f1(fa)}% (n={na})  ->  {f1(fb)}% (n={nb})")
    out["gaps_full_vs_filtered"] = res2
    out["flip_rate_full_vs_filtered"] = fl

    # ============================================================ 3. truncation contrast
    section(f"3. Tabular truncation contrast: SciTabAlign rows<={ROW_CUT} vs rows>{ROW_CUT}; synthetic")
    rows = {qid: len(b["data"]) for qid, b in meta.items() if b["modality"] == "tabular"}
    parts = {
        f"scitab_rows<={ROW_CUT}": lambda q: meta[q]["source"] == "scitabalign" and rows[q] <= ROW_CUT,
        f"scitab_rows>{ROW_CUT}": lambda q: meta[q]["source"] == "scitabalign" and rows[q] > ROW_CUT,
        "synthetic": lambda q: meta[q]["source"] == "synthetic",
    }
    tab_q = [q for q in rows]
    syn_rows = [rows[q] for q in tab_q if meta[q]["source"] == "synthetic"]
    print(f"  synthetic tabular: n={len(syn_rows)}, rows>{ROW_CUT}: {sum(r > ROW_CUT for r in syn_rows)},"
          f" min rows={min(syn_rows)}")
    res3 = {"parts": {}}
    print("\n  task mix (% of questions)")
    all_tasks = sorted({meta[q]["task"] for q in tab_q})
    mix = {}
    for pn, pf_ in parts.items():
        qs = [q for q in tab_q if pf_(q)]
        c = Counter(meta[q]["task"] for q in qs)
        mix[pn] = {"n": len(qs), "pct": {t: 100 * c[t] / len(qs) for t in all_tasks}}
    print(f"  {'task':18s} " + " ".join(f"{pn:>18s}" for pn in parts))
    print(f"  {'n':18s} " + " ".join(f"{mix[pn]['n']:>18d}" for pn in parts))
    for t in all_tasks:
        print(f"  {t:18s} " + " ".join(f"{f1(mix[pn]['pct'][t]):>18s}" for pn in parts))
    res3["task_mix"] = mix
    for pn, pf_ in parts.items():
        print(f"\n  [{pn}]")
        print(f"  {'model':15s} {'n':>5s} " + " ".join(f"{f:>12s}" for f in FMTS["tabular"])
              + f" {'best-worst':>10s} {'p':>7s} {'tbl-txt':>8s}")
        res3["parts"][pn] = {}
        for name, recs in core.items():
            sub = [r for r in recs if r["modality"] == "tabular" and pf_(r["question_id"])]
            g = gap_cell(sub, "tabular", rng)
            dtt, _ = paired_diff(sub, "tabular", "table_image", "text_only")
            g["table_image_minus_text_only"] = dtt
            res3["parts"][pn][name] = g
            print(f"  {name:15s} {g['n_questions']:5d} " + " ".join(
                f"{f1(g['per_format'][f]):>12s}" for f in FMTS["tabular"])
                + f" {f1(g['gap']):>10s} {fp(g['p']):>7s} {f1(dtt):>8s}"
                + f"   ({g['best']} vs {g['worst']})")
    out["truncation_contrast"] = res3

    # ============================================================ 4. lenient categorical
    section("4. Categorical-answer scoring sensitivity (tabular)")
    LEX = {"positively": "positive", "negatively": "negative", "increasing": "increasing",
           "increases": "increasing", "rising": "increasing", "decreasing": "decreasing",
           "decreases": "decreasing", "falling": "decreasing"}

    def lenient(s):
        s = str(s).strip().lower()
        toks = s.split()
        t = toks[0] if toks else ""
        t = re.sub(r"[\.\,\;\:\!\?\)\]\"']+$", "", t)
        t = re.sub(r"^[\(\[\"']+", "", t)
        return LEX.get(t, t)

    def is_cat(qid):
        b = meta[qid]
        if b["modality"] != "tabular":
            return False
        if b["task"] in ("correlation", "trend_analysis"):
            return True
        if b["task"] in ("comparison", "counterfactual"):
            return norm(b["answer"]) in ("yes", "no")
        return False

    cat_q = [q for q in meta if is_cat(q)]
    cc = Counter(meta[q]["task"] for q in cat_q)
    print("  categorical questions: " + ", ".join(f"{t}={cc[t]}" for t in sorted(cc))
          + f"  total={len(cat_q)}")

    def score_len(r):
        if is_cat(r["question_id"]):
            return float(lenient(r.get("prediction", "")) == lenient(r.get("answer", "")))
        return float(r.get("exact_match", 0))

    res4 = {"categorical_counts": dict(cc)}
    print(f"\n  per-format EM on categorical subset, strict -> lenient")
    for name, recs in core.items():
        sub = [r for r in recs if r["modality"] == "tabular" and is_cat(r["question_id"])]
        a = per_format(sub, "tabular")
        b = per_format(sub, "tabular", score_len)
        per_task = {}
        for t in sorted(cc):
            st = [r for r in sub if r["task"] == t]
            per_task[t] = {"strict": per_format(st, "tabular"),
                           "lenient": per_format(st, "tabular", score_len)}
        res4.setdefault("per_model", {})[name] = {"cat_strict": a, "cat_lenient": b,
                                                  "per_task": per_task}
        print(f"  {name:15s} " + "  ".join(f"{f}={f1(a[f])}->{f1(b[f])}" for f in FMTS["tabular"]))
        for t in sorted(cc):
            pa, pb = per_task[t]["strict"], per_task[t]["lenient"]
            print(f"    {t:15s} " + "  ".join(f"{f}={f1(pa[f])}->{f1(pb[f])}"
                                              for f in FMTS["tabular"]))
    print("\n  full tabular gap, strict vs lenient-on-categorical")
    print(f"  {'model':15s} | {'best':13s} {'worst':13s} {'gap':>5s} {'p':>7s} | {'best':13s}"
          f" {'worst':13s} {'gap':>5s} {'p':>7s} | changed?")
    for name, recs in core.items():
        a = gap_cell(recs, "tabular", rng)
        b = gap_cell(recs, "tabular", rng, score_len)
        ch = (a["best"] != b["best"]) or (a["worst"] != b["worst"])
        res4["per_model"][name]["tabular_strict"] = a
        res4["per_model"][name]["tabular_lenient"] = b
        res4["per_model"][name]["best_or_worst_changed"] = ch
        print(f"  {name:15s} | {a['best']:13s} {a['worst']:13s} {f1(a['gap']):>5s} {fp(a['p']):>7s} |"
              f" {b['best']:13s} {b['worst']:13s} {f1(b['gap']):>5s} {fp(b['p']):>7s} |"
              f" best/worst {'CHANGED' if ch else 'same'}, gap {f1(b['gap']-a['gap'])}")
        print(f"  {'':15s}   lenient per-format: " + "  ".join(
            f"{f}={f1(b['per_format'][f])}" for f in FMTS["tabular"]))
    out["categorical_sensitivity"] = res4

    # ============================================================ 5. per-object counts
    section("5. Questions per object (data_id)")
    res5 = {}
    for md in ["overall"] + MODS:
        c = Counter(b["data_id"] for b in bench if md == "overall" or b["modality"] == md)
        v = list(c.values())
        res5[md] = {"objects": len(v), "questions": sum(v), "mean": statistics.mean(v),
                    "median": statistics.median(v), "min": min(v), "max": max(v)}
        print(f"  {md:11s} objects={len(v):4d} questions={sum(v):5d} mean={f1(statistics.mean(v))}"
              f" median={f1(float(statistics.median(v)))} min={min(v)} max={max(v)}")
    out["per_object_counts"] = res5

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
