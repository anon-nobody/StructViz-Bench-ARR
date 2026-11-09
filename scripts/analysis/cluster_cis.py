#!/usr/bin/env python3
"""Object-cluster confidence intervals and baselines for the paper's headline contrasts.

Stdlib only, no inference. Every question in benchmark/realworld_test.jsonl carries a
`data_id` naming its source object (299 objects); questions from one object share data and
are not independent, so each contrast below is recomputed with the object as the resampling
(bootstrap) and exchangeability (sign-flip permutation) unit.

    python scripts/analysis/cluster_cis.py

Conventions: all statistics are ratio estimators sum(d)/sum(n) over clusters, so the point
estimate equals the plain question-level mean. Best/worst formats are chosen once on the
full sample (as the paper does) and held fixed inside every resample. Percentile 95% CIs,
B=5000 bootstraps, B=10000 two-sided sign-flip permutations, p=(hits+1)/(B+1).
Each resampling call uses a fresh random.Random(0).

Writes scripts/analysis/cluster_cis_results.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cluster_cis_results.json")

SEED = 0
B_BOOT = 5000
B_PERM = 10000

CORE = {
    "GPT-4o": "full_gpt4o_extracted.jsonl",
    "Gemini Flash": "full_gemini_extracted.jsonl",
    "Qwen2.5-VL-7B": "full_qwen_extracted.jsonl",
    "Claude Sonnet": "full_claude_extracted.jsonl",
}
EXTRA = {
    "Qwen2.5-VL-32B": "full_qwen32b.jsonl",
    "InternVL2.5-8B": "full_internvl.jsonl",
    "Gemini-2.5": "full_gemini25.jsonl",
}
PROMPT_KEYS = {"GPT-4o": "gpt4o", "Gemini Flash": "gemini", "Qwen2.5-VL-7B": "qwen",
               "Claude Sonnet": "claude"}
STYLES = ["minimal", "concise", "detailed", "cot"]
MODS = ["tabular", "timeseries", "graph"]


# ---------------------------------------------------------------- loading helpers
def load(path):
    if not os.path.exists(path):
        return None
    return [json.loads(ln) for ln in open(path) if ln.strip()]


def load_qid2did():
    q2d = {}
    for ln in open(os.path.join(ROOT, "benchmark", "realworld_test.jsonl")):
        r = json.loads(ln)
        q2d[r["question_id"]] = r["data_id"]
    return q2d


def per_format(recs, modality=None):
    acc = defaultdict(list)
    for r in recs:
        if modality is None or r.get("modality") == modality:
            acc[r["viz_type"]].append(float(r.get("exact_match", 0)))
    return {k: 100 * sum(v) / len(v) for k, v in acc.items() if v}


def by_question(recs, modality=None):
    g = defaultdict(dict)
    for r in recs:
        if modality is None or r.get("modality") == modality:
            g[r["question_id"]][r["viz_type"]] = float(r.get("exact_match", 0))
    return g


def best_worst(recs, modality):
    p = per_format(recs, modality)
    return max(p, key=p.get), min(p, key=p.get), p


# ---------------------------------------------------------------- resampling core
def clusters_from(items, q2d):
    """items: iterable of (question_id, value). Returns list of (sum, n) per data_id."""
    agg = defaultdict(lambda: [0.0, 0])
    for q, v in items:
        a = agg[q2d[q]]
        a[0] += v
        a[1] += 1
    return [tuple(v) for v in agg.values()]


def point(cl):
    return 100 * sum(s for s, _ in cl) / sum(n for _, n in cl)


def pct(xs, lo=2.5, hi=97.5):
    xs = sorted(xs)
    k = len(xs) - 1

    def q(p):
        f = p / 100 * k
        i = int(f)
        j = min(i + 1, k)
        return xs[i] + (xs[j] - xs[i]) * (f - i)

    return q(lo), q(hi)


def cluster_boot(cl, B=B_BOOT):
    rng = random.Random(SEED)
    m = len(cl)
    stats = []
    for _ in range(B):
        s = n = 0.0
        for _ in range(m):
            a, b = cl[int(rng.random() * m)]
            s += a
            n += b
        stats.append(100 * s / n)
    lo, hi = pct(stats)
    return {"lo": lo, "hi": hi, "hw": (hi - lo) / 2}


def question_boot(vals, B=B_BOOT):
    rng = random.Random(SEED)
    m = len(vals)
    stats = []
    for _ in range(B):
        stats.append(100 * sum(vals[int(rng.random() * m)] for _ in range(m)) / m)
    lo, hi = pct(stats)
    return {"lo": lo, "hi": hi, "hw": (hi - lo) / 2}


def cluster_perm(cl, B=B_PERM):
    """Two-sided sign-flip test: flip all paired differences of one object together."""
    rng = random.Random(SEED)
    sums = [s for s, _ in cl]
    N = sum(n for _, n in cl)
    obs = abs(sum(sums) / N)
    hits = 0
    for _ in range(B):
        t = 0.0
        for s in sums:
            t += s if rng.random() < 0.5 else -s
        if abs(t / N) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (B + 1), hits


def summarize(cl, perm=False):
    out = {"est": point(cl), "n_obj": len(cl), "n_units": sum(n for _, n in cl)}
    out["obj_ci"] = cluster_boot(cl)
    if perm:
        out["obj_perm_p"], out["obj_perm_hits"] = cluster_perm(cl)
    return out


# ---------------------------------------------------------------- formatting
def pp(x):
    return f"{x:+.1f}" if x < 0 or x > 0 else f"{x:.1f}"


def f1(x):
    return f"{x:.1f}"


def fp(p, hits=None):
    if hits == 0:
        return "<1e-4"
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


def ci_str(c):
    return f"[{c['lo']:+.1f}, {c['hi']:+.1f}]"


def section(t):
    print(f"\n{t}\n" + "-" * 100)


def side(did):
    return "train" if int(hashlib.md5(did.encode()).hexdigest(), 16) % 2 == 0 else "test"


# ---------------------------------------------------------------- analyses
def part_a(core, q2d):
    section("A. Best-worst gap per (model, modality): question- vs object-cluster bootstrap, "
            "object sign-flip p")
    print(f"{'model':15s} {'mod':10s} {'best':>13s} {'worst':>15s} {'gap':>6s} "
          f"{'q_hw':>6s} {'obj_hw':>7s} {'obj CI':>15s} {'n_q':>5s} {'n_obj':>5s} {'p_obj':>7s}")
    rows = []
    for name, recs in core.items():
        for md in MODS:
            b, w, _ = best_worst(recs, md)
            bq = by_question(recs, md)
            items = [(q, s[b] - s[w]) for q, s in bq.items() if b in s and w in s]
            vals = [v for _, v in items]
            cl = clusters_from(items, q2d)
            r = summarize(cl, perm=True)
            r.update(model=name, modality=md, best=b, worst=w, q_ci=question_boot(vals))
            rows.append(r)
            print(f"{name:15s} {md:10s} {b:>13s} {w:>15s} {f1(r['est']):>6s} "
                  f"{f1(r['q_ci']['hw']):>6s} {f1(r['obj_ci']['hw']):>7s} "
                  f"{ci_str(r['obj_ci']):>15s} {r['n_units']:5d} {r['n_obj']:5d} "
                  f"{fp(r['obj_perm_p'], r['obj_perm_hits']):>7s}")
    hws = [r["obj_ci"]["hw"] for r in rows]
    qhws = [r["q_ci"]["hw"] for r in rows]
    print(f"\nobject-level 95% CI half-width range: {f1(min(hws))}-{f1(max(hws))} pp "
          f"(question-level: {f1(min(qhws))}-{f1(max(qhws))} pp)")
    return {"rows": rows, "obj_hw_min": min(hws), "obj_hw_max": max(hws),
            "q_hw_min": min(qhws), "q_hw_max": max(qhws)}


def selector_items(recs, q2d):
    byq = defaultdict(dict)
    tag = {}
    for r in recs:
        q = r["question_id"]
        byq[q][r["viz_type"]] = float(r.get("exact_match", 0))
        tag[q] = (r["modality"], r.get("task", "?"))

    def sp(q):
        return side(q2d.get(q, q))

    acc_mt = defaultdict(lambda: defaultdict(list))
    acc_m = defaultdict(lambda: defaultdict(list))
    for q, fm in byq.items():
        if sp(q) != "train":
            continue
        for v, e in fm.items():
            acc_mt[tag[q]][v].append(e)
            acc_m[tag[q][0]][v].append(e)
    best_mt = {k: max(d, key=lambda v: sum(d[v]) / len(d[v])) for k, d in acc_mt.items()}
    best_m = {k: max(d, key=lambda v: sum(d[v]) / len(d[v])) for k, d in acc_m.items()}
    d_fix, d_rand, sel_v, fix_v, rnd_v = [], [], [], [], []
    n_text = n = 0
    for q, fm in byq.items():
        if sp(q) != "test":
            continue
        n += 1
        mean = sum(fm.values()) / len(fm)
        mt = best_mt.get(tag[q]) or best_m.get(tag[q][0])
        m = best_m.get(tag[q][0])
        s_mt = fm.get(mt, mean)
        s_m = fm.get(m, mean)
        n_text += mt == "text_only"
        d_fix.append((q, s_mt - s_m))
        d_rand.append((q, s_mt - mean))
        sel_v.append(s_mt)
        fix_v.append(s_m)
        rnd_v.append(mean)
    return {"n": n, "d_fix": d_fix, "d_rand": d_rand, "sel": 100 * sum(sel_v) / n,
            "fixed": 100 * sum(fix_v) / n, "random": 100 * sum(rnd_v) / n,
            "text_only_frac": 100 * n_text / n, "best_m": best_m}


def part_b(models, q2d):
    section("B. Format selector (object-grouped md5(data_id) split), held-out questions")
    print(f"{'model':15s} {'n_test':>6s} {'n_obj':>5s} {'random':>7s} {'fixed':>6s} "
          f"{'select':>7s} {'sel-fix':>8s} {'obj CI':>15s} {'sel-rand':>9s} {'obj CI':>15s} "
          f"{'%text':>6s}")
    out = {}
    for name, recs in models.items():
        s = selector_items(recs, q2d)
        cf = summarize(clusters_from(s["d_fix"], q2d))
        cr = summarize(clusters_from(s["d_rand"], q2d))
        out[name] = {"n_test": s["n"], "n_obj": cf["n_obj"], "random": s["random"],
                     "fixed_mod": s["fixed"], "selector": s["sel"],
                     "fixed_formats_train": s["best_m"],
                     "delta_vs_fixed": cf["est"], "delta_vs_fixed_ci": cf["obj_ci"],
                     "delta_vs_random": cr["est"], "delta_vs_random_ci": cr["obj_ci"],
                     "text_only_frac": s["text_only_frac"]}
        print(f"{name:15s} {s['n']:6d} {cf['n_obj']:5d} {f1(s['random']):>7s} "
              f"{f1(s['fixed']):>6s} {f1(s['sel']):>7s} {pp(cf['est']):>8s} "
              f"{ci_str(cf['obj_ci']):>15s} {pp(cr['est']):>9s} {ci_str(cr['obj_ci']):>15s} "
              f"{f1(s['text_only_frac']):>6s}")
    return out


def norm(p):
    p = str(p).strip().lower()
    m = re.search(r"-?\d+(?:\.\d+)?", p.replace(",", ""))
    return m.group() if m else p


def part_c(models, q2d):
    section("C. Format ensemble (majority vote as tta_format_ensemble.py), all questions; "
            "fixed(mod) chosen in-sample")
    print(f"{'model':15s} {'n_q':>5s} {'random':>7s} {'fixed':>6s} {'ensem':>6s} "
          f"{'oracle':>7s} {'ens-fix':>8s} {'obj CI':>15s} {'ens-rand':>9s} {'obj CI':>15s}")
    out = {}
    for name, recs in models.items():
        groups = defaultdict(list)
        for r in recs:
            groups[r["question_id"]].append(
                (r["viz_type"], float(r.get("exact_match", 0) or 0), norm(r.get("prediction", ""))))
        fixed = {md: best_worst(recs, md)[0] for md in MODS}
        mod = {r["question_id"]: r["modality"] for r in recs}
        rnd = ens = orc = fix = 0.0
        d_fix, d_rand = [], []
        for q, g in groups.items():
            ems = [e for _, e, _ in g]
            mean = sum(ems) / len(ems)
            top = Counter(p for _, _, p in g).most_common(1)[0][0]
            e = 1.0 if max((em for _, em, p in g if p == top), default=0.0) >= 0.5 else 0.0
            o = 1.0 if any(x >= 0.5 for x in ems) else 0.0
            fm = {v: em for v, em, _ in g}
            fx = fm.get(fixed[mod[q]], mean)
            rnd += mean
            ens += e
            orc += o
            fix += fx
            d_fix.append((q, e - fx))
            d_rand.append((q, e - mean))
        n = len(groups)
        cf = summarize(clusters_from(d_fix, q2d))
        cr = summarize(clusters_from(d_rand, q2d))
        out[name] = {"n_q": n, "n_obj": cf["n_obj"], "random": 100 * rnd / n,
                     "fixed_mod": 100 * fix / n, "fixed_formats_full": fixed,
                     "ensemble": 100 * ens / n, "oracle": 100 * orc / n,
                     "ens_minus_fixed": cf["est"], "ens_minus_fixed_ci": cf["obj_ci"],
                     "ens_minus_random": cr["est"], "ens_minus_random_ci": cr["obj_ci"]}
        o = out[name]
        print(f"{name:15s} {n:5d} {f1(o['random']):>7s} {f1(o['fixed_mod']):>6s} "
              f"{f1(o['ensemble']):>6s} {f1(o['oracle']):>7s} {pp(cf['est']):>8s} "
              f"{ci_str(cf['obj_ci']):>15s} {pp(cr['est']):>9s} {ci_str(cr['obj_ci']):>15s}")
    return out


def part_d(q7, q32, q2d):
    section("D. Scale: Qwen2.5-VL-32B minus 7B")
    k7 = {(r["question_id"], r["viz_type"]): float(r.get("exact_match", 0)) for r in q7}
    k32 = {(r["question_id"], r["viz_type"]): float(r.get("exact_match", 0)) for r in q32}
    common = sorted(set(k7) & set(k32))
    ov = summarize(clusters_from([(q, k32[(q, v)] - k7[(q, v)]) for q, v in common], q2d),
                   perm=True)
    ov.update(em_7b=100 * sum(k7[k] for k in common) / len(common),
              em_32b=100 * sum(k32[k] for k in common) / len(common))
    print(f"overall EM: 7B {f1(ov['em_7b'])}  32B {f1(ov['em_32b'])}  diff {pp(ov['est'])}  "
          f"obj CI {ci_str(ov['obj_ci'])}  p_obj {fp(ov['obj_perm_p'], ov['obj_perm_hits'])}  "
          f"(n rows {ov['n_units']}, n_obj {ov['n_obj']})")
    b7, w7, _ = best_worst(q7, "tabular")
    b32, w32, _ = best_worst(q32, "tabular")
    g7, g32 = by_question(q7, "tabular"), by_question(q32, "tabular")
    qs = [q for q in g7 if q in g32 and b7 in g7[q] and w7 in g7[q]
          and b32 in g32[q] and w32 in g32[q]]
    items = [(q, (g32[q][b32] - g32[q][w32]) - (g7[q][b7] - g7[q][w7])) for q in qs]
    tg = summarize(clusters_from(items, q2d), perm=True)
    tg.update(gap_7b=100 * sum(g7[q][b7] - g7[q][w7] for q in qs) / len(qs),
              gap_32b=100 * sum(g32[q][b32] - g32[q][w32] for q in qs) / len(qs),
              fmt_7b=[b7, w7], fmt_32b=[b32, w32])
    print(f"tabular gap: 7B {f1(tg['gap_7b'])} ({b7}-{w7})  32B {f1(tg['gap_32b'])} "
          f"({b32}-{w32})  change {pp(tg['est'])}  obj CI {ci_str(tg['obj_ci'])}  "
          f"p_obj {fp(tg['obj_perm_p'], tg['obj_perm_hits'])}  "
          f"(n_q {tg['n_units']}, n_obj {tg['n_obj']})")
    return {"overall": ov, "tabular_gap_change": tg}


def part_e(q2d):
    section("E. Prompt style (n=500 subsample): CoT minus best non-CoT, object-cluster CI; "
            "tabular best-worst gap per style")
    print(f"{'model':15s} " + " ".join(f"{s:>8s}" for s in STYLES)
          + f" {'bestNon':>9s} {'CoT-best':>9s} {'obj CI':>15s} {'n_obj':>5s} | tab gap "
          + " ".join(f"{s:>8s}" for s in STYLES))
    out = {}
    for name, key in PROMPT_KEYS.items():
        data = {s: load(os.path.join(RES, "ablation", f"prompt_{key}_{s}.jsonl")) for s in STYLES}
        if any(v is None for v in data.values()):
            print(f"{name:15s} missing prompt files")
            continue
        kd = {s: {(r["question_id"], r["viz_type"]): float(r.get("exact_match", 0))
                  for r in recs} for s, recs in data.items()}
        common = sorted(set.intersection(*(set(v) for v in kd.values())))
        ems = {s: 100 * sum(kd[s][k] for k in common) / len(common) for s in STYLES}
        best = max((s for s in STYLES if s != "cot"), key=ems.get)
        r = summarize(clusters_from([(q, kd["cot"][(q, v)] - kd[best][(q, v)])
                                     for q, v in common], q2d))
        gaps = {}
        for s in STYLES:
            _, _, p = best_worst(data[s], "tabular")
            gaps[s] = max(p.values()) - min(p.values())
        out[name] = {"em": ems, "best_non_cot": best, "cot_minus_best": r["est"],
                     "ci": r["obj_ci"], "n_obj": r["n_obj"], "n_rows": r["n_units"],
                     "n_q": len({q for q, _ in common}), "tabular_gap": gaps}
        print(f"{name:15s} " + " ".join(f"{f1(ems[s]):>8s}" for s in STYLES)
              + f" {best:>9s} {pp(r['est']):>9s} {ci_str(r['obj_ci']):>15s} {r['n_obj']:5d} | "
              + "        " + " ".join(f"{f1(gaps[s]):>8s}" for s in STYLES))
    return out


def part_f(a_rows):
    section("F. Holm correction over the 12 main best-worst contrasts (object-level sign-flip p)")
    ps = sorted(((r["obj_perm_p"], r["model"], r["modality"]) for r in a_rows))
    m = len(ps)
    adj, running = [], 0.0
    for i, (p, _, _) in enumerate(ps):
        running = max(running, min(1.0, (m - i) * p))
        adj.append(running)
    res = []
    for (p, mo, md), a in zip(ps, adj):
        res.append({"model": mo, "modality": md, "p_raw": p, "p_holm": a})
        print(f"{mo:15s} {md:10s} p_raw {fp(p):>7s}  p_holm {fp(a) if a >= 1e-4 else '<1e-4':>7s}")
    n05 = sum(a < 0.05 for a in adj)
    n01 = sum(a < 0.01 for a in adj)
    print(f"\nHolm-significant: {n05}/{m} at 0.05, {n01}/{m} at 0.01 "
          f"(permutation p floor = 1/{B_PERM + 1})")
    return {"rows": res, "n_sig_05": n05, "n_sig_01": n01, "m": m}


def main():
    q2d = load_qid2did()
    objs = set(q2d.values())
    by_mod = defaultdict(set)
    for ln in open(os.path.join(ROOT, "benchmark", "realworld_test.jsonl")):
        r = json.loads(ln)
        by_mod[r["modality"]].add(r["data_id"])
    print("StructViz-Bench: object-cluster CIs (seed 0, B_boot=5000, B_perm=10000)")
    print(f"questions {len(q2d)}  objects {len(objs)}  "
          + "  ".join(f"{m} {len(by_mod[m])}" for m in MODS)
          + f"  | split train {sum(side(d) == 'train' for d in objs)} "
            f"test {sum(side(d) == 'test' for d in objs)}")
    core = {k: load(os.path.join(RES, v)) for k, v in CORE.items()}
    six = dict(core)
    for k, v in EXTRA.items():
        recs = load(os.path.join(RES, v))
        if recs is not None:
            six[k] = recs
    results = {"config": {"seed": SEED, "B_boot": B_BOOT, "B_perm": B_PERM,
                          "n_questions": len(q2d), "n_objects": len(objs),
                          "objects_by_modality": {m: len(by_mod[m]) for m in MODS},
                          "models_bc": list(six)}}
    results["A_gaps"] = part_a(core, q2d)
    results["B_selector"] = part_b(six, q2d)
    results["C_ensemble"] = part_c(six, q2d)
    results["D_scale"] = part_d(six["Qwen2.5-VL-7B"], six["Qwen2.5-VL-32B"], q2d)
    results["E_prompt"] = part_e(q2d)
    results["F_holm"] = part_f(results["A_gaps"]["rows"])
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
