#!/usr/bin/env python3
"""Further strict-vs-canonical recomputations, added to scripts/analysis/canonical_results.json.

    python scripts/analysis/canonical_extras.py      (after canonical_rescore.py; also called
                                                     at the end of canonical_rescore.py)

Adds, each as {"strict": ..., "canonical": ...}:
  selector        format selector on held-out objects (md5(data_id) parity split; best format
                  per (modality, task) learned on train, falling back to per modality; a
                  question lacking the chosen format scores its mean over formats) for the seven
                  v1 models: random, fixed per modality, hand rule, selector, oracle (correct in
                  any format), selector - fixed and selector - random with object-cluster CIs,
                  rule - fixed. Uses v2_analysis._selector_eval, the code behind the paper's
                  hand-rule numbers (identical split and selector to cluster_cis.part_b).
  rawtext         Qwen2.5-VL-7B given the raw data as text (results/v2/rawtext_qwen.jsonl), EM per
                  modality (raw_response available, so rule (e) applies).
  noimage         question-only run (results/v2/noimage_qwen.jsonl), EM per modality.
  assist          the 2x2 {full table, row bars} x {means absent, printed} on the mean-comparison
                  questions (v2_analysis.s6_assist).
  answerable      Table answerable: EM on present (V/D, default audit classes) / absent pairs per
                  format for the four core models.
  perviz          Table perviz: per-format EM for the four core models (from section v1).
  scale           Qwen2.5-VL 32B minus 7B: overall EM and tabular-gap change (cluster_cis.part_d).
  errors          error-taxonomy shares cited in the text, with the incorrect set defined by each
                  score: share of all errors on absent pairs per model; for the four core models
                  pooled on answerable pairs, far-off share for graph drawings, near-miss share
                  for the series listing, wrong-label share for GAF / recurrence plots.
Writes paper_tables/tab_perviz_canonical.tex and paper_tables/tab_selector_canonical.tex.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import error_taxonomy as et  # noqa: E402
import review_common as rc  # noqa: E402
import scoring as sc  # noqa: E402
import v2_analysis as v2a  # noqa: E402

SCORES = ("strict", "canonical")
OUT_JSON = os.path.join(HERE, "canonical_results.json")
TAB = os.path.join(ROOT, "paper_tables")
SHORT = {"GPT-4o": "GPT-4o", "Gemini-2.0-Flash": "Gemini-2.0", "Qwen2.5-VL-7B": "Qwen2.5-VL-7B",
         "Claude Sonnet 4": "Claude Son.~4", "Gemini-2.5-Flash": "Gemini-2.5",
         "Qwen2.5-VL-32B": "Qwen2.5-VL-32B", "InternVL2.5-8B": "InternVL2.5-8B"}


def stored(r):
    return float(r.get("exact_match", 0) or 0)


def v1_score(r, score):
    if score == "strict":
        return stored(r)
    raw = r.get("prediction_raw")
    if raw is not None and str(raw) == str(r.get("prediction")):
        raw = None
    return sc.canonical_em(r.get("prediction", ""), raw, r["answer"], r.get("task"), stored(r))


def raw_score(r, score):
    return stored(r) if score == "strict" else sc.row_score(r, "raw_response")


def c(r):
    o = r["obj_ci"]
    return {"est": r["est"], "lo": o["lo"], "hi": o["hi"], "n": r["n_units"],
            "n_obj": r["n_obj"]}


def selector_direct(meta, q2d):
    """Selector numbers, computed directly (same logic as cluster_cis.selector_items) so that
    selector - fixed and selector - random carry their own object-cluster CIs."""
    import hashlib

    out = {}
    for score in SCORES:
        out[score] = {}
        for m in rc.MODELS7:
            byq, tag = defaultdict(dict), {}
            for r in rc.load_v1(m).values():
                q = r["question_id"]
                byq[q][r["viz_type"]] = v1_score(r, score)
                tag[q] = (r["modality"], r["task"])

            def side(q):
                return "train" if int(hashlib.md5(q2d[q].encode()).hexdigest(), 16) % 2 == 0 \
                    else "test"
            acc_mt, acc_m = defaultdict(lambda: defaultdict(list)), defaultdict(
                lambda: defaultdict(list))
            for q, fm in byq.items():
                if side(q) == "train":
                    for v, e in fm.items():
                        acc_mt[tag[q]][v].append(e)
                        acc_m[tag[q][0]][v].append(e)
            best_mt = {k: max(d, key=lambda v: sum(d[v]) / len(d[v])) for k, d in acc_mt.items()}
            best_m = {k: max(d, key=lambda v: sum(d[v]) / len(d[v])) for k, d in acc_m.items()}
            sel, fix, rnd, orc, d_fix, d_rand = [], [], [], [], [], []
            for q, fm in byq.items():
                if side(q) != "test":
                    continue
                mean = sum(fm.values()) / len(fm)
                mt = best_mt.get(tag[q]) or best_m.get(tag[q][0])
                s_mt, s_m = fm.get(mt, mean), fm.get(best_m.get(tag[q][0]), mean)
                sel.append(s_mt)
                fix.append(s_m)
                rnd.append(mean)
                orc.append(float(max(fm.values()) >= 1))
                d_fix.append((q, s_mt - s_m))
                d_rand.append((q, s_mt - mean))
            rule = v2a._selector_eval(
                [dict(r, exact_match=v1_score(r, score)) for r in rc.load_v1(m).values()],
                q2d, meta, None)
            out[score][m] = {
                "n_test": len(sel), "n_obj": len({q2d[q] for q, _ in d_fix}),
                "random": rc.mean100(rnd), "fixed": rc.mean100(fix), "selector": rc.mean100(sel),
                "oracle": rc.mean100(orc), "rule": rule["rule"],
                "sel_minus_fixed": c(v2a.contrast(d_fix, q2d)),
                "sel_minus_random": c(v2a.contrast(d_rand, q2d)),
                "rule_minus_fixed": c(rule["rule_minus_fixed"]),
                "selector_minus_rule": c(rule["selector_minus_rule"])}
    return out


def by_modality_em(path, score, key_by_q=True):
    rows = rc.read_jsonl(os.path.join(ROOT, path))
    d = {}
    for r in rows:
        if str(r.get("prediction")) == "[ERROR]":
            continue
        d[r["question_id"] if key_by_q else (r["question_id"], r["viz_type"])] = r
    out = {}
    for md in rc.MODS:
        xs = [raw_score(r, score) for r in d.values() if r["modality"] == md]
        out[md] = {"em": rc.mean100(xs), "n": len(xs)}
    return out


def assist(meta, q2d):
    good, _ = v2a.dedupe(rc.read_jsonl(os.path.join(ROOT, "results", "v2", "assist_qwen.jsonl")))
    out = {}
    for score in SCORES:
        EMa = defaultdict(dict)
        for (q, f), r in good.items():
            if f in v2a.ASSIST:
                EMa[q][f] = raw_score(r, score)
        with contextlib.redirect_stdout(io.StringIO()):
            s = v2a.s6_assist(EMa, meta, q2d)
        out[score] = {k: {"n": v["n"], "means": v["means"],
                          **{e: c(x) | {"p": x.get("obj_perm_p")} for e, x in v.items()
                             if isinstance(x, dict) and "obj_ci" in x}}
                      for k, v in s.items() if isinstance(v, dict) and "means" in v}
    return out


def answerable(meta):
    cat = rc.v1_cats(meta)
    out = {}
    for score in SCORES:
        out[score] = {}
        for m in rc.CORE4:
            rows = rc.load_v1(m)
            rec = {}
            for md in rc.MODS:
                for f in rc.FORMATS[md]:
                    g = {"VD": [], "A": []}
                    for q in meta:
                        if meta[q]["modality"] != md:
                            continue
                        k = cat[q][f]
                        if k in "VD":
                            g["VD"].append(v1_score(rows[(q, f)], score))
                        elif k == "A":
                            g["A"].append(v1_score(rows[(q, f)], score))
                    rec[f"{md}|{f}"] = {k: {"em": rc.mean100(v), "n": len(v)} for k, v in g.items()}
            out[score][m] = rec
    return out


def scale(q2d):
    out = {}
    for score in SCORES:
        k7 = {k: v1_score(r, score) for k, r in rc.load_v1("Qwen2.5-VL-7B").items()}
        k32 = {k: v1_score(r, score) for k, r in rc.load_v1("Qwen2.5-VL-32B").items()}
        common = sorted(set(k7) & set(k32))
        ov = v2a.contrast([(q, k32[(q, f)] - k7[(q, f)]) for q, f in common], q2d, perm=True)
        mod = {q: (q.split("_")[0]) for q, _ in common}

        def bw(k):
            pf = defaultdict(list)
            for (q, f), v in k.items():
                if mod.get(q) == "tabular":
                    pf[f].append(v)
            pf = {f: sum(v) / len(v) for f, v in pf.items()}
            return max(pf, key=pf.get), min(pf, key=pf.get)
        b7, w7 = bw(k7)
        b32, w32 = bw(k32)
        qs = sorted({q for q, _ in common if mod[q] == "tabular"})
        items = [(q, (k32[(q, b32)] - k32[(q, w32)]) - (k7[(q, b7)] - k7[(q, w7)])) for q in qs]
        tg = v2a.contrast(items, q2d, perm=True)
        out[score] = {"overall": c(ov) | {"p": ov["obj_perm_p"]},
                      "gap_7b": rc.mean100(k7[(q, b7)] - k7[(q, w7)] for q in qs),
                      "gap_32b": rc.mean100(k32[(q, b32)] - k32[(q, w32)] for q in qs),
                      "fmt_7b": [b7, w7], "fmt_32b": [b32, w32],
                      "gap_change": c(tg) | {"p": tg["obj_perm_p"]}}
    return out


def errors(meta):
    cs = rc.cell_stats(meta)
    ov = {q: et.object_values(b) for q, b in meta.items()}
    cat = rc.v1_cats(meta)
    out = {}
    for score in SCORES:
        absent_share, pooled = {}, defaultdict(Counter)
        for m in rc.MODELS7:
            n_err = n_abs = 0
            for (q, f), r in rc.load_v1(m).items():
                if v1_score(r, score) >= 1:
                    continue
                cls = cat[q][f]
                n_err += 1
                n_abs += cls == "A"
                if m in rc.CORE4 and cls in "VD":
                    b = meta[q]
                    k = et.classify(r, b, cls, cs[(b["modality"], b["task"])], ov[q], m)
                    pooled[f"{b['modality']}|{f}"][k] += 1
            absent_share[m] = 100 * n_abs / n_err
        shares = {}
        for key, cnt in pooled.items():
            n = sum(cnt.values())
            shares[key] = {"n_err": n, **{k: 100 * v / n for k, v in cnt.items()}}
        out[score] = {"absent_share_of_errors": absent_share, "core4_vd_by_format": shares}
    return out


def tex_perviz(res):
    order = {"tabular": ["bar_chart", "heatmap", "scatter_plot", "table_image", "text_only"],
             "timeseries": ["gaf", "heatmap", "line_plot", "recurrence_plot", "text_only"],
             "graph": ["adjacency_matrix", "circular_layout", "node_link", "text_only"]}
    head = {"tabular": r"\emph{Tabular}: bar / heatmap / scatter / table image / text",
            "timeseries": r"\emph{Time series}: GAF / heatmap / line / recurrence / text",
            "graph": r"\emph{Graph}: adjacency / circular / node-link / text"}
    L = ["% Canonical exact match (scripts/analysis/scoring.py), same layout as Table perviz in",
         "% main.tex; strict values in parentheses are NOT shown (see tab_strict_vs_canonical).",
         "% Generated by scripts/analysis/canonical_extras.py; do not edit by hand.",
         r"\begin{table}[t]\centering\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
         r"\caption{Default suite: canonical exact match (\%) by format for the four core models. "
         r"The spread within a row is the format gap reported in prior work; "
         r"Section~\ref{sec:results} shows most of it is information availability.}",
         r"\label{tab:perviz}", r"\begin{tabular}{lccccc}", r"\toprule"]
    pv = res["v1"]["canonical"]["models"]
    for i, md in enumerate(rc.MODS):
        if i:
            L.append(r"\midrule")
        L.append(r"\multicolumn{6}{l}{" + head[md] + r"} \\")
        for m in rc.CORE4:
            vals = [f"{pv[m]['per_format'][md][f]:.1f}" for f in order[md]]
            if md == "graph":
                vals.append("")
            L.append(f"{SHORT[m]:14s}& " + " & ".join(vals) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(TAB, "tab_perviz_canonical.tex"), "w") as fh:
        fh.write("\n".join(L) + "\n")


def tex_selector(res):
    S = res["selector"]["canonical"]
    names = {"GPT-4o": "GPT-4o", "Gemini-2.0-Flash": "Gemini Flash",
             "Gemini-2.5-Flash": "Gemini-2.5", "Qwen2.5-VL-7B": "Qwen2.5-VL-7B",
             "Claude Sonnet 4": "Claude Sonnet", "Qwen2.5-VL-32B": "Qwen2.5-VL-32B",
             "InternVL2.5-8B": "InternVL2.5-8B"}
    any_m = next(iter(S.values()))
    L = ["% Canonical exact match; same layout as Table selector in tab_selector_lora.tex plus a",
         "% hand-rule column. Generated by scripts/analysis/canonical_extras.py.",
         r"\begin{table*}[t]\centering\small",
         r"\caption{Learned format selector, split by \emph{source object} (148 train / 151 held-out "
         f"objects; {any_m['n_test']:,}".replace(",", "{,}") + r" held-out questions). Canonical "
         r"exact match (\%) on held-out questions. ``random'' is the expected accuracy of an "
         r"arbitrary format; ``fixed (mod)'' is the best single format per modality, learned on "
         r"train; ``rule'' is the hand rule (tabular comparison and mean aggregation $\to$ bar "
         r"chart, other tabular $\to$ table image, series and graphs $\to$ text listing); "
         r"``selector'' conditions on modality and task. $\Delta$ is the selector's gain over "
         r"the fixed-per-modality baseline with its 95\% object-cluster bootstrap CI (151 held-out "
         r"objects, $B{=}5000$).}",
         r"\label{tab:selector}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lccccclc}",
         r"\toprule",
         r"Model & random & fixed (mod) & rule & \textbf{selector (mod,task)} & $\Delta$ & 95\% CI & oracle \\",
         r"\midrule"]
    for m in ["GPT-4o", "Gemini-2.0-Flash", "Gemini-2.5-Flash", "Qwen2.5-VL-7B",
              "Claude Sonnet 4", "Qwen2.5-VL-32B", "InternVL2.5-8B"]:
        v = S[m]
        d = v["sel_minus_fixed"]
        L.append(f"{names[m]:14s}& {v['random']:.1f} & {v['fixed']:.1f} & {v['rule']:.1f} & "
                 rf"\textbf{{{v['selector']:.1f}}} & ${d['est']:+.1f}$ & "
                 rf"$[{d['lo']:+.1f},{d['hi']:+.1f}]$ & {v['oracle']:.1f} \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    with open(os.path.join(TAB, "tab_selector_canonical.tex"), "w") as fh:
        fh.write("\n".join(L) + "\n")


def add_extras(res):
    meta = rc.load_bench()
    q2d = {q: b["data_id"] for q, b in meta.items()}
    res["selector"] = selector_direct(meta, q2d)
    res["rawtext"] = {s: by_modality_em("results/v2/rawtext_qwen.jsonl", s) for s in SCORES}
    res["noimage"] = {s: by_modality_em("results/v2/noimage_qwen.jsonl", s) for s in SCORES}
    res["assist"] = assist(meta, q2d)
    res["answerable"] = answerable(meta)
    res["perviz"] = {s: {m: res["v1"][s]["models"][m]["per_format"] for m in rc.CORE4}
                     for s in SCORES}
    res["scale"] = scale(q2d)
    res["errors"] = errors(meta)
    tex_perviz(res)
    tex_selector(res)
    return res


def report(res):
    for s in SCORES:
        print(f"\n[{s}] selector (held-out)")
        for m, v in res["selector"][s].items():
            d, r = v["sel_minus_fixed"], v["sel_minus_random"]
            print(f"  {m:17s} n={v['n_test']} random {v['random']:.1f} fixed {v['fixed']:.1f} rule "
                  f"{v['rule']:.1f} sel {v['selector']:.1f} oracle {v['oracle']:.1f} | sel-fix "
                  f"{d['est']:+.1f} [{d['lo']:+.1f},{d['hi']:+.1f}] sel-rand {r['est']:+.1f} | "
                  f"rule-fix {v['rule_minus_fixed']['est']:+.1f}")
        print(f"  rawtext {{{', '.join(f'{k}: {v['em']:.1f} (n={v['n']})' for k, v in res['rawtext'][s].items())}}}")
        print(f"  noimage {{{', '.join(f'{k}: {v['em']:.1f} (n={v['n']})' for k, v in res['noimage'][s].items())}}}")
        a = res["assist"][s]["all"]
        print(f"  assist n={a['n']} means {{{', '.join(f'{k}: {v:.1f}' for k, v in a['means'].items())}}} "
              + " ".join(f"{k} {v['est']:+.1f} [{v['lo']:+.1f},{v['hi']:+.1f}]"
                         for k, v in a.items() if isinstance(v, dict) and "est" in v))
        sc_ = res["scale"][s]
        print(f"  scale overall {sc_['overall']['est']:+.2f} [{sc_['overall']['lo']:+.2f},"
              f"{sc_['overall']['hi']:+.2f}] p={sc_['overall']['p']:.4f}; tabular gap "
              f"{sc_['gap_7b']:.1f}->{sc_['gap_32b']:.1f} change {sc_['gap_change']['est']:+.1f} "
              f"[{sc_['gap_change']['lo']:+.1f},{sc_['gap_change']['hi']:+.1f}] "
              f"p={sc_['gap_change']['p']:.4f}")
        e = res["errors"][s]
        print(f"  absent share of errors: {{{', '.join(f'{k}: {v:.1f}' for k, v in e['absent_share_of_errors'].items())}}}")
        g = e["core4_vd_by_format"]
        print("  core4 V/D errors: " + "; ".join(
            f"{k} n={g[k]['n_err']} far {g[k].get('far_off', 0):.1f} near "
            f"{g[k].get('near_miss', 0):.1f} wrong {g[k].get('wrong_label', 0):.1f}"
            for k in ("graph|node_link", "graph|circular_layout", "graph|adjacency_matrix",
                      "timeseries|text_only", "timeseries|gaf", "timeseries|recurrence_plot")))
        print("  answerable (VD/A) tabular bar/scatter/table: " + "; ".join(
            f"{m} " + "/".join(f"{res['answerable'][s][m][f'tabular|{f}']['VD']['em']:.0f}/"
                               f"{res['answerable'][s][m][f'tabular|{f}']['A']['em']:.0f}"
                               for f in ("bar_chart", "scatter_plot", "table_image"))
            for m in rc.CORE4))


def main():
    res = json.load(open(OUT_JSON))
    add_extras(res)
    with open(OUT_JSON, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    report(res)
    print("\nupdated scripts/analysis/canonical_results.json; wrote "
          "paper_tables/tab_perviz_canonical.tex, paper_tables/tab_selector_canonical.tex")


if __name__ == "__main__":
    main()
