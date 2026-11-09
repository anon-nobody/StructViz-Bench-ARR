#!/usr/bin/env python3
"""Expressive (v2) multi-model analysis re-scored with index-prefix label normalisation,
side by side with the paper's strict exact match; plus v1 7B/32B and a survey of other
surface forms that exact match punishes.

Stdlib only (plus src.evaluation.metrics.exact_match, the benchmark's own scorer); no
inference.

    python scripts/analysis/v2_multimodel_normalized.py

Normalised score (identical for every model, format and task):
    em_norm = 1 if stored exact_match == 1
            else exact_match(strip(prediction), key)   when the key is non-numeric
            else 0                                      (numeric keys never change)
    strip(p) = re.sub(r"^\\s*\\d+\\s*[:.]\\s*", "", p)
The expressive bar chart / scatter plot / heatmap tick labels read "index: label"
(v2_renderers, row_labels), and some models copy the whole tick ("11: cloud" for key
"cloud"; "19.10B" for key "10B"). The score is monotone (it can only turn 0 into 1), so no
correct row is lost. "Numeric key" = float(key) parses, as in metrics._normalize_numeric_string.

All v2 statistics are recomputed with v2_multimodel.analyse (same definitions as the paper:
strict answerability, near-constant cells excluded, gaps on questions answerable in every
format, series over the three non-lossy formats, CV prior, degree effect, recovery vs v1),
once with the stored exact_match and once with em_norm (applied to v1 too for recovery).
v1 7B / 32B: rows changed per format and per-format EM / best-worst gap under both scores.

Survey (v2 main formats, rows still wrong after em_norm): how many more rows each further
surface normalisation would turn correct, per model -- quotes/backticks, %/$/thousands
separators/units after a number, case + punctuation (SQuAD normalisation; non-numeric key and
prediction only, so "-0.07" vs "0.07" is not matched), underscore/hyphen
vs space, polarity lexicon (positively -> positive ...), other index forms ("row 11: x",
"(11) x", "11 - x"), entity prefixes ("node 5", "column x"). An "extended" score applying all
of them is also run through the gap analysis as a robustness check (reported, not proposed).

Writes scripts/analysis/v2_multimodel_normalized_results.json and
paper_tables/tab_v2_multimodel_norm.tex.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import review_common as rc  # noqa: E402
import v2_analysis as v2a  # noqa: E402
import v2_multimodel as mm  # noqa: E402
from src.evaluation.metrics import _normalize_text, exact_match  # noqa: E402

IDX_RE = re.compile(r"^\s*\d+\s*[:.]\s*")
V1_FILES = {"Qwen2.5-VL-7B": "results/full_qwen_extracted.jsonl",
            "Qwen2.5-VL-32B": "results/full_qwen32b.jsonl"}


def key_numeric(a):
    try:
        float(str(a).strip())
        return True
    except ValueError:
        return False


def stored(r):
    return float(r.get("exact_match", 0) or 0)


def em_norm(r):
    if stored(r) >= 1:
        return 1.0
    a = str(r["answer"])
    if key_numeric(a):
        return 0.0
    p = IDX_RE.sub("", str(r.get("prediction", "")))
    return exact_match(p, a)


# ---------------------------------------------------------------- further surface forms
LEX = {"positively": "positive", "negatively": "negative", "increases": "increasing",
       "decreases": "decreasing", "rising": "increasing", "falling": "decreasing"}
NUM_HEAD = re.compile(r"^\s*[\$€£]?\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*(%|[a-zA-Z/°]+)?\s*$")


def _num_eq(p, a):
    m = NUM_HEAD.match(p)
    if not m:
        return False
    try:
        return float(m.group(1).replace(",", "")) == float(a)
    except ValueError:
        return False


PATTERNS = {
    "quotes": lambda p, a: exact_match(p.strip().strip("'\"`"), a),
    "num_units_pct_sep": lambda p, a: key_numeric(a) and _num_eq(p, a),
    "case_punct_space": lambda p, a: (not key_numeric(a)) and (not key_numeric(p))
    and _normalize_text(p) == _normalize_text(a) != "",
    "underscore_hyphen": lambda p, a: re.sub(r"[_\-\s]+", " ", p.strip().lower())
    == re.sub(r"[_\-\s]+", " ", a.strip().lower()),
    "polarity_lexicon": lambda p, a: LEX.get(p.strip().lower().rstrip("."), None)
    == a.strip().lower(),
    "other_index_forms": lambda p, a: (not key_numeric(a)) and exact_match(
        re.sub(r"^\s*(row\s*\d+\s*[:.\-]|\(\d+\)|\d+\s*[-–)]|#\d+\s*[:.\-]?)\s*", "", p,
               flags=re.I), a),
    "entity_prefix": lambda p, a: exact_match(
        re.sub(r"^\s*(node|row|column|col|store|the)\s+", "", p, flags=re.I), a),
}


def em_ext(r):
    if em_norm(r) >= 1:
        return 1.0
    p, a = str(r.get("prediction", "")), str(r["answer"])
    p2 = IDX_RE.sub("", p) if not key_numeric(a) else p
    return float(any(f(p, a) or f(p2, a) for f in PATTERNS.values()))


# ---------------------------------------------------------------- helpers
def change_counts(path, meta, main_only=True):
    rows, _ = v2a.dedupe(v2a.read_jsonl(os.path.join(ROOT, path)))
    by_fmt, by_task, ex = Counter(), Counter(), []
    numeric_changed = 0
    n = 0
    for (q, f), r in rows.items():
        md = meta[q]["modality"]
        if main_only and f not in v2a.FORMATS[md] and f != v2a.DEG_FMT:
            continue
        n += 1
        a, b = stored(r), em_norm(r)
        if a != b:
            assert b > a
            if key_numeric(r["answer"]):
                numeric_changed += 1
            by_fmt[f"{md}|{f}"] += 1
            by_task[f"{md}|{meta[q]['task']}"] += 1
            if len(ex) < 6:
                ex.append((meta[q]["task"], f, r["prediction"], r["answer"]))
    return {"n_rows": n, "changed": sum(by_fmt.values()), "by_format": dict(by_fmt),
            "by_task": dict(by_task), "numeric_key_rows_changed": numeric_changed,
            "examples": ex}


def survey(path, meta):
    rows, _ = v2a.dedupe(v2a.read_jsonl(os.path.join(ROOT, path)))
    c, cf, ex = Counter(), defaultdict(Counter), defaultdict(list)
    n_wrong = 0
    for (q, f), r in rows.items():
        md = meta[q]["modality"]
        if f not in v2a.FORMATS[md]:
            continue
        if em_norm(r) >= 1:
            continue
        n_wrong += 1
        p, a = str(r.get("prediction", "")), str(r["answer"])
        p2 = IDX_RE.sub("", p) if not key_numeric(a) else p
        for name, fn in PATTERNS.items():
            if fn(p, a) or fn(p2, a):
                c[name] += 1
                cf[name][f"{md}|{f}"] += 1
                if len(ex[name]) < 4:
                    ex[name].append((meta[q]["task"], f, p, a))
    return {"n_wrong_after_norm": n_wrong, "counts": dict(c),
            "top_formats": {k: dict(v.most_common(4)) for k, v in cf.items()},
            "examples": dict(ex)}


def parser_survey(path, meta, fa_tab):
    """Rows wrong after em_norm whose stored raw_response (before parse_answer) matches the
    key (after the same index strip): answers destroyed by the response parser."""
    rows, _ = v2a.dedupe(v2a.read_jsonl(os.path.join(ROOT, path)))
    by, ex, n, n_fa = Counter(), [], 0, 0
    for (q, f), r in rows.items():
        md = meta[q]["modality"]
        if f not in v2a.FORMATS[md]:
            continue
        raw, a = str(r.get("raw_response", "")), str(r["answer"])
        if raw in ("", "None") or em_norm(r) >= 1:
            continue
        raw2 = raw if key_numeric(a) else IDX_RE.sub("", raw)
        if exact_match(raw, a) or exact_match(raw2, a):
            n += 1
            n_fa += q in fa_tab
            by[f"{md}|{f}|{meta[q]['task']}"] += 1
            if len(ex) < 5:
                ex.append((meta[q]["task"], f, raw[:40], r["prediction"], a))
    return {"n": n, "n_in_tabular_gap_sample": n_fa, "by": dict(by.most_common(8)),
            "examples": ex}


def run_all(scorer, meta, q2d, cat2s, nc, prior_const, prior_q):
    mm.SCORER = scorer
    out = {}
    for spec in mm.MODELS:
        o = mm.analyse(spec, meta, q2d, cat2s, nc, prior_const, prior_q)
        o.pop("_present", None)
        out[spec["name"]] = o
    mm.SCORER = None
    return out


def v1_block(path, meta, q2d, scorer):
    rows, _ = v2a.dedupe(v2a.read_jsonl(os.path.join(ROOT, path)))
    EM = defaultdict(dict)
    for (q, f), r in rows.items():
        EM[q][f] = scorer(r)
    out = {}
    for md in v2a.MODS:
        qs = [q for q in meta if meta[q]["modality"] == md]
        g = v2a.gap_block(EM, qs, v2a.FORMATS[md], q2d)
        out[md] = {"per_format": g["per_format"], "best": g["best"], "worst": g["worst"],
                   "gap": g["est"], "lo": g["obj_ci"]["lo"], "hi": g["obj_ci"]["hi"],
                   "n_q": g["n_q"]}
    return out


def gap_cell(o, md, key="all_answerable_exclNC"):
    return o["gaps"][md][key]


def fmt(g):
    if not g.get("n_q"):
        return "--"
    return f"{g['est']:.1f} [{g['lo']:.1f},{g['hi']:.1f}]"


ABBR = {"table_image": "tbl", "text_only": "txt", "heatmap": "heat", "bar_chart": "bar",
        "scatter_plot": "scat", "node_link": "node", "circular_layout": "circ",
        "adjacency_matrix": "adj", "line_plot": "line", "gaf": "gaf",
        "recurrence_plot": "rp"}


def main():
    meta = rc.load_bench()
    q2d = {q: b["data_id"] for q, b in meta.items()}
    cat2s = rc.v2_cats(meta, strict=True)
    nc = rc.near_constant()
    pr = v2a.load_prior()
    prior_const = {md: pr[md]["cv_mean"] for md in v2a.MODS}
    prior_q = rc.cv_prior_per_question(meta)

    res = {"definition": __doc__, "changes": {}, "survey": {}, "v1": {}}
    print("[1] rows changed by index-prefix normalisation (0 -> 1), main formats + text_only_deg")
    for spec in mm.MODELS:
        c = change_counts(spec["v2"], meta)
        res["changes"][spec["name"]] = c
        print(f"  {spec['name']:26s} {c['changed']:5d}/{c['n_rows']} rows; numeric-key rows "
              f"changed {c['numeric_key_rows_changed']}")
        print(f"      by format {dict(sorted(c['by_format'].items(), key=lambda kv: -kv[1]))}")
        print(f"      by task   {dict(sorted(c['by_task'].items(), key=lambda kv: -kv[1]))}")
        print(f"      e.g. {c['examples'][:4]}")
    for name, path in V1_FILES.items():
        c = change_counts(path, meta)
        res["changes"][f"{name} v1"] = c
        print(f"  {name + ' v1':26s} {c['changed']:5d}/{c['n_rows']} rows; numeric-key rows "
              f"changed {c['numeric_key_rows_changed']}; by format {c['by_format']}")
    assert all(c["numeric_key_rows_changed"] == 0 for c in res["changes"].values())

    print("\n[2] v2 analyses: strict vs normalised (vs extended surface normalisation)")
    S = run_all(None, meta, q2d, cat2s, nc, prior_const, prior_q)
    N = run_all(em_norm, meta, q2d, cat2s, nc, prior_const, prior_q)
    E = run_all(em_ext, meta, q2d, cat2s, nc, prior_const, prior_q)
    res["strict"], res["normalized"], res["extended"] = S, N, E
    for spec in mm.MODELS:
        m = spec["name"]
        print(f"\n  === {m}")
        for md in v2a.MODS:
            print(f"    EM {md:10s} " + "  ".join(
                f"{f} {S[m]['per_format'][f'{md}|{f}']['em']:.1f}->"
                f"{N[m]['per_format'][f'{md}|{f}']['em']:.1f}" for f in v2a.FORMATS[md]))
        for md in ("tabular", "graph"):
            for lab, X in (("strict", S), ("norm", N), ("ext", E)):
                g = gap_cell(X[m], md)
                if g.get("n_q"):
                    print(f"    {md:8s} gap {lab:6s} {fmt(g)} p={g['p']:.4f} n={g['n_q']} "
                          f"best={g['best']} worst={g['worst']}")
        for lab, X in (("strict", S), ("norm", N)):
            g = gap_cell(X[m], "timeseries", "nonlossy3_all_exclNC")
            if g.get("n_q"):
                print(f"    series 3-fmt {lab:6s} {fmt(g)} n={g['n_q']} best={g['best']} "
                      f"worst={g['worst']}")
        for md in v2a.MODS:
            k = "diff_same_q" if spec["subset"] else "diff"
            a, b = S[m]["best_vs_prior"][md], N[m]["best_vs_prior"][md]
            fa = [f for f, v in S[m]["formats_vs_prior"][md].items()
                  if v["exceeds_same_q" if spec["subset"] else "exceeds"]]
            fb = [f for f, v in N[m]["formats_vs_prior"][md].items()
                  if v["exceeds_same_q" if spec["subset"] else "exceeds"]]
            print(f"    best-prior {md:10s} strict {a['best']} {a[k]:+.1f}  norm {b['best']} "
                  f"{b[k]:+.1f}; beating prior strict {len(fa)} -> norm {len(fb)}")
        for lab in ("degree_query", "other_tasks"):
            a, b = S[m]["degree_effect"][lab], N[m]["degree_effect"][lab]
            if a.get("n"):
                print(f"    degree {lab:12s} strict {a['est']:+.1f} [{a['lo']:+.1f},{a['hi']:+.1f}]"
                      f"  norm {b['est']:+.1f} [{b['lo']:+.1f},{b['hi']:+.1f}] n={a['n']}")
        if "recovery" in S[m]:
            ch = {k: (S[m]["recovery"][k]["d"], N[m]["recovery"][k]["d"])
                  for k in S[m]["recovery"]}
            print("    recovery strict->norm: " + "  ".join(
                f"{k.split('|')[1]}({k[0]}) {a:+.1f}->{b:+.1f}" for k, (a, b) in ch.items()))

    print("\n[2b] v1 (7B, 32B): per-modality best-worst gap on all questions, strict -> norm")
    for name, path in V1_FILES.items():
        a, b = v1_block(path, meta, q2d, stored), v1_block(path, meta, q2d, em_norm)
        res["v1"][name] = {"strict": a, "normalized": b}
        for md in v2a.MODS:
            print(f"  {name:15s} {md:10s} gap {a[md]['gap']:.1f} ({a[md]['best']}/{a[md]['worst']})"
                  f" -> {b[md]['gap']:.1f} ({b[md]['best']}/{b[md]['worst']}); max per-format "
                  f"change {max(b[md]['per_format'][f] - a[md]['per_format'][f] for f in a[md]['per_format']):+.2f}")

    print("\n[3] further surface forms (v2 main formats, rows still wrong after normalisation)")
    for spec in mm.MODELS:
        sv = survey(spec["v2"], meta)
        res["survey"][spec["name"]] = sv
        print(f"  {spec['name']:26s} wrong {sv['n_wrong_after_norm']}: {sv['counts']}")
        for k, v in sv["examples"].items():
            print(f"      {k:18s} {sv['top_formats'][k]}  e.g. {v[:2]}")

    fa_tab = set(q for q in meta if meta[q]["modality"] == "tabular"
                 and ("tabular", meta[q]["task"]) not in nc
                 and all(cat2s[q][f] in "VD" for f in v2a.FORMATS["tabular"]))
    print("\n[3b] raw response correct but destroyed by parse_answer (after normalisation)")
    res["parser_survey"] = {}
    for spec in mm.MODELS:
        ps = parser_survey(spec["v2"], meta, fa_tab)
        res["parser_survey"][spec["name"]] = ps
        print(f"  {spec['name']:26s} {ps['n']:4d} rows (in the n=186 tabular gap sample: "
              f"{ps['n_in_tabular_gap_sample']}); {ps['by']}; e.g. {ps['examples'][:3]}")

    # ---- headline verdict
    def g(X, m, md):
        return gap_cell(X[m], md)
    vt = {m: {lab: g(X, m, "tabular")["est"] for lab, X in (("strict", S), ("norm", N),
                                                             ("ext", E))}
          for m in ("Qwen2.5-VL-7B", "Qwen2.5-VL-32B", "Qwen3-VL-8B")}
    res["verdict_tab_gap"] = vt
    print("\n[verdict] tabular residual gap (n=186) strict / norm / extended:")
    for m, v in vt.items():
        print(f"  {m:16s} {v['strict']:.1f} / {v['norm']:.1f} / {v['ext']:.1f}")

    with open(os.path.join(HERE, "v2_multimodel_normalized_results.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)

    # ---- LaTeX: one row per model, strict | normalised
    hdr = ["Model", "Changed", "Tab.\\ gap strict", "Tab.\\ gap norm.", "b/w norm.",
           "Graph gap strict", "Graph gap norm.", "Series norm.", "Best$-$prior graph s/n",
           "Deg.\\ effect norm."]
    rows = []
    for spec in mm.MODELS:
        m = spec["name"]
        k = "diff_same_q" if spec["subset"] else "diff"
        tn = gap_cell(N[m], "tabular")
        d = N[m]["degree_effect"]
        rows.append([
            rc.Raw(rc.tex_esc(m) + ("$^\\dagger$" if spec["subset"] else "")),
            str(res["changes"][m]["changed"]),
            fmt(gap_cell(S[m], "tabular")), fmt(tn),
            f"{ABBR.get(tn['best'], tn['best'])}/{ABBR.get(tn['worst'], tn['worst'])}",
            fmt(gap_cell(S[m], "graph")), fmt(gap_cell(N[m], "graph")),
            "--" if spec["subset"] else fmt(gap_cell(N[m], "timeseries",
                                                     "nonlossy3_all_exclNC")),
            f"{S[m]['best_vs_prior']['graph'][k]:+.1f} / {N[m]['best_vs_prior']['graph'][k]:+.1f}",
            "--" if not d["degree_query"].get("n") else
            f"{d['degree_query']['est']:+.1f} / {d['other_tasks']['est']:+.1f}"])
    rc.write_tabular(
        os.path.join(ROOT, "paper_tables", "tab_v2_multimodel_norm.tex"),
        "Expressive suite under strict exact match vs index-prefix normalisation (a leading\n"
        "'<index>:' or '<index>.' copied from the 'index: label' tick labels is stripped when the\n"
        "key is non-numeric; applied identically to every model and format). Changed = (question,\n"
        "format) rows turned correct. Gaps: best minus worst format on questions answerable in\n"
        "every format outside near-constant templates (n=186 tabular, 528 graph; object-cluster\n"
        "95% CI); b/w = best/worst format after normalisation; Series = three non-lossy formats.\n"
        "Best-prior graph = best graph format minus the CV prior, strict / normalised. Deg. effect\n"
        "= text_only_deg minus text_only on degree_query / other tasks. dagger = 1,096-question\n"
        "strict subset.",
        "l r l l l l l l l l", hdr, rows)
    print("\nwrote scripts/analysis/v2_multimodel_normalized_results.json, "
          "paper_tables/tab_v2_multimodel_norm.tex")


if __name__ == "__main__":
    main()
