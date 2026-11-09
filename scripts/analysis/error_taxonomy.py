#!/usr/bin/env python3
"""Error taxonomy of every incorrect (question, format) prediction.

Stdlib only, no inference.

    python scripts/analysis/error_taxonomy.py

Scope: all seven v1 models (results/full_*.jsonl) and Qwen2.5-VL-7B on the corrected v2
suite (results/v2/v2_qwen.jsonl, main formats, last non-error row per key). A pair is
incorrect iff `exact_match` == 0 (the `exact` field is ignored). Answerability classes are
scripts/analysis/answerability.py for v1 and v2_analysis.classify_v2 (manifest renderer
constants) for v2.

Rules (applied in this order; the first that fires decides, so categories are mutually
exclusive). "cell" = (modality, task); "vocab" = the set of answers that cell's questions
have (lenient form); lenient = lowercase, '_' -> ' ', edge punctuation stripped, a small
lexicon (positively -> positive, ...), numbers in canonical %g form.
  absent       answerability class A for this (question, format): the key is not
               recoverable from the rendering.
  by_design    class C (constant / unanswerable by design: pattern_label_lookup, the
               always-"yes" connectivity and add-edge counterfactual, ...).
  refusal      prediction is an abstention token (none, n/a, unknown, cannot determine,
               [REFUSED], [EMPTY], [ERROR], "-", empty, ...) or matches a refusal regex
               ("cannot", "unable", "not visible", "does not provide", ...), unless the
               prediction is in the cell vocab (change_point's legitimate answer "none").
  free_text    unparsed output: starts with a reasoning preamble ("Looking at", "I need",
               "Let me", "In", "Based on", "The image", ...) or has >= 5 words, and is not a
               cell value of the source object.
  surface      lenient(prediction) == lenient(key), or the stripped strings are identical
               but exact_match is 0 (31 Claude v1 rows, 29 of them on V/D pairs; reported
               separately as scorer_inconsistency inside this category).
  prior_like   lenient(prediction) == the cell's (in-sample) majority answer, and that
               majority answer is non-numeric (yes/no or label).
  type_mismatch  review_common.right_type is False: the prediction's type (yes/no, number,
               label) differs from the key's, or the cell is closed-vocabulary (<= 10
               distinct non-numeric answers) and the label is outside the vocab.
  near_miss    key and prediction numeric and |p - a| <= 5% of |a|, or one equals the other
               rounded to its own number of decimals (reading-precision error).
  far_off      key and prediction numeric otherwise (off-by-one integer counted inside).
  wrong_label  key and prediction both categorical of the right type (yes/no polarity or a
               label/entity), but wrong.
  other        anything left (expected to be empty; printed).
Shares are % of incorrect pairs. Restricted variant: only pairs with class V or D.

Writes scripts/analysis/error_taxonomy_results.json, paper_tables/tab_error_taxonomy.tex,
paper_tables/tab_error_by_format.tex.
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict

import review_common as rc

CATS = ["absent", "by_design", "refusal", "free_text", "surface", "prior_like",
        "type_mismatch", "near_miss", "far_off", "wrong_label", "other"]
SHORT = {"absent": "Absent", "by_design": "By-design", "refusal": "Refusal",
         "free_text": "Free text", "surface": "Surface", "prior_like": "Prior-like",
         "type_mismatch": "Type", "near_miss": "Near", "far_off": "Far",
         "wrong_label": "Wrong lbl", "other": "Other"}
SUB: Counter = Counter()  # keyed (model, subcategory)


def object_values(b):
    """Lenient set of every cell value / node id of the source object (entity check)."""
    d = b["data"]
    vals = set()
    if isinstance(d, list):
        for row in d:
            if isinstance(row, dict):
                vals.update(rc.lenient(v) for v in row.values())
            else:
                vals.add(rc.lenient(row))
    elif isinstance(d, dict):
        for n in d.get("nodes", []):
            vals.add(rc.lenient(n.get("id")))
    return vals


def classify(r, b, cls, cs, ovals, m=""):
    pred, key = str(r.get("prediction", "")), str(b["answer"])
    if cls == "A":
        return "absent"
    if cls == "C":
        return "by_design"
    lp, lk = rc.lenient(pred), rc.lenient(key)
    if rc.is_refusal(pred, cs["vocab"]):
        return "refusal"
    if rc.is_free_text(pred) and lp not in ovals and lp not in cs["vocab"]:
        return "free_text"
    if rc.norm(pred) == rc.norm(key):
        SUB[(m, "scorer_inconsistency")] += 1
        return "surface"
    if lp == lk:
        return "surface"
    if lp == cs["majority"] and rc.atype(cs["majority"]) != "number":
        return "prior_like"
    if not rc.right_type(pred, key, cs) and not (rc.atype(key) == "label" and lp in ovals):
        return "type_mismatch"
    if rc.atype(key) == "number" and rc.atype(pred) == "number":
        if rc.numeric_near(pred, key):
            return "near_miss"
        pv, av = rc.to_float(pred), rc.to_float(key)
        if pv == int(pv) and av == int(av) and abs(pv - av) == 1:
            SUB[(m, "far_off_by_one")] += 1
        return "far_off"
    if rc.atype(key) in ("yesno", "label"):
        return "wrong_label"
    SUB[(m, "other_examples")] += 1
    return "other"


def shares(c):
    n = sum(c.values())
    return {"n_err": n, **{k: (100 * c[k] / n if n else float("nan")) for k in CATS}}


def main():
    meta = rc.load_bench()
    cs_all = rc.cell_stats(meta)
    ov = {q: object_values(b) for q, b in meta.items()}
    cat1, cat2 = rc.v1_cats(meta), rc.v2_cats(meta)
    preds = {m: rc.load_v1(m) for m in rc.MODELS7}
    v2rows, st = rc.load_v2_qwen(meta)
    preds[rc.QWEN_V2] = v2rows
    print(f"v2 Qwen dedupe: {st}; kept main-format keys {len(v2rows)}")
    models = list(preds)

    # counters
    pooled = {m: {"all": Counter(), "vd": Counter()} for m in models}
    by_mod = {m: {md: {"all": Counter(), "vd": Counter()} for md in rc.MODS} for m in models}
    by_fmt = {m: defaultdict(lambda: {"all": Counter(), "vd": Counter()}) for m in models}
    n_pairs = Counter()
    n_vd = Counter()
    examples = defaultdict(list)
    for m in models:
        cats = cat2 if m == rc.QWEN_V2 else cat1
        for (q, f), r in preds[m].items():
            b = meta[q]
            cls = cats[q][f]
            n_pairs[m] += 1
            n_vd[m] += cls in "VD"
            if rc.em_val(r) >= 1:
                continue
            k = classify(r, b, cls, cs_all[(b["modality"], b["task"])], ov[q], m)
            if len(examples[k]) < 6 and m == "Qwen2.5-VL-7B":
                examples[k].append((b["task"], f, r.get("prediction"), b["answer"]))
            md = b["modality"]
            for scope in ("all", "vd") if cls in "VD" else ("all",):
                pooled[m][scope][k] += 1
                by_mod[m][md][scope][k] += 1
                by_fmt[m][f"{md}|{f}"][scope][k] += 1

    out = {"rules_order": CATS, "subcounts": {f"{a}|{b}": v for (a, b), v in SUB.items()},
           "n_pairs": dict(n_pairs),
           "n_answerable_pairs": dict(n_vd), "examples_qwen_v1": examples,
           "pooled": {}, "by_modality": {}, "by_format": {}, "core4_pooled_by_format": {}}
    for m in models:
        out["pooled"][m] = {s: shares(pooled[m][s]) for s in ("all", "vd")}
        out["by_modality"][m] = {md: {s: shares(by_mod[m][md][s]) for s in ("all", "vd")}
                                 for md in rc.MODS}
        out["by_format"][m] = {k: {s: shares(v[s]) for s in ("all", "vd")}
                               for k, v in by_fmt[m].items()}
    for md in rc.MODS:
        for f in rc.FORMATS[md]:
            agg = {"all": Counter(), "vd": Counter()}
            for m in rc.CORE4:
                for s in agg:
                    agg[s].update(by_fmt[m][f"{md}|{f}"][s])
            out["core4_pooled_by_format"][f"{md}|{f}"] = {s: shares(agg[s]) for s in agg}

    # ---- print
    print("\nrule subcounts:", {f"{a}|{b}": v for (a, b), v in sorted(SUB.items())})
    for scope, lab in (("all", "ALL incorrect pairs"), ("vd", "incorrect ANSWERABLE (V/D) pairs")):
        print(f"\n[{lab}] share of incorrect pairs (%)")
        print(f"  {'model':22s} {'n_err':>6s} " + " ".join(f"{SHORT[c][:9]:>9s}" for c in CATS))
        for m in models:
            s = out["pooled"][m][scope]
            print(f"  {m:22s} {s['n_err']:6d} " + " ".join(f"{rc.f1(s[c]):>9s}" for c in CATS))
    print("\n[per model x modality, all incorrect pairs]")
    for m in models:
        for md in rc.MODS:
            s = out["by_modality"][m][md]["all"]
            print(f"  {m:22s} {md:10s} {s['n_err']:6d} "
                  + " ".join(f"{rc.f1(s[c]):>9s}" for c in CATS))
    print("\n[core-4 pooled per format, all | answerable]")
    for k, v in out["core4_pooled_by_format"].items():
        a, d = v["all"], v["vd"]
        print(f"  {k:28s} n={a['n_err']:5d} " + " ".join(f"{rc.f1(a[c]):>5s}" for c in CATS)
              + f" | vd n={d['n_err']:5d} " + " ".join(f"{rc.f1(d[c]):>5s}" for c in CATS[2:]))
    print("\n[Qwen v1 examples per category]")
    for k, ex in examples.items():
        print(f"  {k}: {ex[:4]}")

    rc.dump(os.path.join(rc.HERE, "error_taxonomy_results.json"), out)

    # ---- LaTeX: compact pooled table, two panels
    hdr = ["Model", "$n_{err}$"] + [SHORT[c] for c in CATS]
    rows, mids = [], []
    for scope in ("all", "vd"):
        if scope == "vd":
            mids.append(len(rows))
        for m in models:
            s = out["pooled"][m][scope]
            cells = [m + (" [V/D]" if scope == "vd" else ""), str(s["n_err"])]
            for c in CATS:
                cells.append("--" if scope == "vd" and c in ("absent", "by_design")
                             else rc.f1(s[c]))
            rows.append(cells)
    rc.write_tabular(
        os.path.join(rc.TABLES, "tab_error_taxonomy.tex"),
        "Error taxonomy: share (%) of incorrect (question, format) predictions per category,\n"
        "pooled over modalities and formats. Top block: all incorrect pairs; bottom block [V/D]:\n"
        "incorrect pairs whose answer is visible or derivable in that rendering. Categories are\n"
        "mutually exclusive and assigned in column order (rules in error_taxonomy.py).\n"
        "Qwen2.5-VL-7B (v2) = corrected rendering suite. Exact match scoring.",
        "l r " + "r" * len(CATS), hdr, rows, midrules=mids)

    hdr2 = ["Modality", "Format", "$n_{err}$"] + [SHORT[c] for c in CATS] + ["$n_{err}^{V/D}$",
                                                                              "Type$^{V/D}$",
                                                                              "Near$^{V/D}$",
                                                                              "Far$^{V/D}$",
                                                                              "Wrong$^{V/D}$"]
    rows2, mids2 = [], []
    for md in rc.MODS:
        if rows2:
            mids2.append(len(rows2))
        for f in rc.FORMATS[md]:
            v = out["core4_pooled_by_format"][f"{md}|{f}"]
            a, d = v["all"], v["vd"]
            rows2.append([md, f, str(a["n_err"])] + [rc.f1(a[c]) for c in CATS]
                         + [str(d["n_err"]), rc.f1(d["type_mismatch"]), rc.f1(d["near_miss"]),
                            rc.f1(d["far_off"]), rc.f1(d["wrong_label"])])
    rc.write_tabular(
        os.path.join(rc.TABLES, "tab_error_by_format.tex"),
        "Error taxonomy by rendering format, pooled over the four core models (v1). Share (%)\n"
        "of incorrect pairs per category; the last five columns repeat selected categories on\n"
        "the answerable (V/D) incorrect pairs only. Per-model breakdowns in\n"
        "scripts/analysis/error_taxonomy_results.json.",
        "l l r " + "r" * len(CATS) + " r r r r r", hdr2, rows2, midrules=mids2)
    print("\nwrote scripts/analysis/error_taxonomy_results.json, "
          "paper_tables/tab_error_taxonomy.tex, paper_tables/tab_error_by_format.tex")


if __name__ == "__main__":
    main()
