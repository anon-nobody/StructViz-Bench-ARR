#!/usr/bin/env python3
"""Re-score the released LoRA before/after records with the paper's common scorer.

Stdlib only. Reads (never modifies):
  scripts/mitigation/base_records_full.jsonl      base Qwen2.5-VL-7B
  scripts/mitigation/after_records_lambda0.jsonl  lambda=0 multi-format SFT
  scripts/mitigation/after_records_full.jsonl     lambda=1 consistency-regularised
  benchmark/realworld_test.jsonl                  gold answers + data_id (source object)

Scorers:
  stored  : the `correct` field written by eval_lora.py (numeric tolerance + substring)
  strict  : src/evaluation/metrics.exact_match (copied verbatim below; the paper's main metric)
  numtol  : strict EM, else float(pred)/float(gold) within max(0.05, 0.01*|gold|); no substring

Consistency Rate (scorer-independent): mean over questions of the fraction of format pairs whose
predictions agree after str().strip().lower() -- the definition used by verify_paper_numbers.py
(Section 6 / Table - LoRA). stats_lora.py compares raw strings; that variant is reported too.

CAVEAT: eval_lora.py stores `preds` AFTER its own norm() (first number in the output if any,
else the lowercased output), while `correct` was computed on the RAW output. Raw outputs are not
in the records, so strict/numtol here score norm(raw). For gold answers that contain digits but
are not numbers (e.g. 'zone_3', 'D4', '2024-01-07 00:00:00'), norm(raw) can never equal the gold
under strict EM; these questions are counted in the "gold-type audit" section.

Writes scripts/mitigation/rescore_lora_results.json.
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SYSTEMS = [
    ("base", "base_records_full.jsonl"),
    ("lambda0", "after_records_lambda0.jsonl"),
    ("lambda1", "after_records_full.jsonl"),
]
MODS = ["tabular", "timeseries", "graph"]
SCORERS = ["stored", "strict", "numtol"]
PAIRS = [("base", "lambda0"), ("lambda0", "lambda1"), ("base", "lambda1")]
B = 5000
SEED = 0
ABS_TOL, REL_TOL = 0.05, 0.01


# ---- verbatim copy of src/evaluation/metrics.py -------------------------------------------
def _normalize_numeric_string(s: str) -> str:
    """Normalize numeric strings to remove trailing zeros (55.130 -> 55.13)."""
    try:
        val = float(s)
        if val == int(val) and "." not in s:
            return str(int(val))
        # Use repr-like formatting that strips trailing zeros.
        normalized = f"{val:g}"
        return normalized
    except ValueError:
        return s.strip().lower()


def exact_match(prediction: str, answer: str) -> float:
    """Case-insensitive exact match with numeric normalization."""
    pred = _normalize_numeric_string(prediction.strip().lower())
    ans = _normalize_numeric_string(answer.strip().lower())
    return float(pred == ans)
# -------------------------------------------------------------------------------------------


def _float(s: str) -> float | None:
    try:
        v = float(s.strip())
    except ValueError:
        return None
    return v if v == v and abs(v) != float("inf") else None


def numtol(pred: str, gold: str) -> int:
    if exact_match(pred, gold):
        return 1
    pf, gf = _float(pred), _float(gold)
    if pf is None or gf is None:
        return 0
    return int(abs(pf - gf) <= max(ABS_TOL, REL_TOL * abs(gf)))


# ---- eval_lora.py scorer (to explain why a stored-correct cell is credited) ---------------
import re  # noqa: E402


def _lora_norm(s):
    s = str(s).strip().lower()
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return m.group() if m else s


def lora_reason(pred: str, gold: str) -> str:
    """Which branch of eval_lora.match credits (pred, gold): 'norm_eq', 'numeric_tol',
    'substring', or 'none'."""
    p, g = _lora_norm(pred), _lora_norm(gold)
    if p == g:
        return "norm_eq"
    try:
        pf, gf = float(p), float(g)
        return "numeric_tol" if abs(pf - gf) <= max(ABS_TOL, REL_TOL * abs(gf)) else "none"
    except ValueError:
        return "substring" if g in str(pred).strip().lower() else "none"


def canon(s) -> str:
    return str(s).strip().lower()


def load_first(path: str) -> tuple[dict, int, int]:
    d, nrows = {}, 0
    for line in open(path):
        if line.strip():
            nrows += 1
            r = json.loads(line)
            d.setdefault(r["qid"], r)  # keep FIRST occurrence
    return d, nrows, nrows - len(d)


def cr_pairs(preds: list, norm: bool) -> float:
    vals = [canon(p) for p in preds] if norm else list(preds)
    k = len(vals)
    if k < 2:
        return 1.0
    agree = sum(1 for i in range(k) for j in range(i + 1, k) if vals[i] == vals[j])
    return agree / (k * (k - 1) / 2)


def ci(vals: list[float]) -> tuple[float, float]:
    s = sorted(vals)  # identical indexing to stats_lora.py
    return s[int(0.025 * len(s))], s[int(0.975 * len(s)) - 1]


def main() -> None:
    recs, meta = {}, {}
    for name, fn in SYSTEMS:
        d, nrows, ndup = load_first(os.path.join(HERE, fn))
        recs[name] = d
        meta[name] = {"file": fn, "rows": nrows, "duplicate_rows_dropped": ndup, "unique_qids": len(d)}
    common = set.intersection(*[set(d) for d in recs.values()])

    gold, data_id = {}, {}
    for line in open(os.path.join(ROOT, "benchmark", "realworld_test.jsonl")):
        if line.strip():
            r = json.loads(line)
            if r["question_id"] in common:
                gold[r["question_id"]] = str(r["answer"])
                data_id[r["question_id"]] = r["data_id"]
    missing_gold = sorted(common - set(gold))
    qids = sorted(q for q in common if q in gold)

    # modality consistency check across files
    mod_of = {}
    for q in qids:
        ms = {recs[s][q]["modality"] for s, _ in SYSTEMS}
        assert len(ms) == 1, (q, ms)
        mod_of[q] = ms.pop()

    out: dict = {"meta": meta, "n_common_qids": len(common), "missing_gold": missing_gold,
                 "n_data_id": len(set(data_id.values())), "B": B, "seed": SEED,
                 "by_modality": {}, "credit_audit": {}}

    # ---- per-question stat rows ----------------------------------------------------------
    print(f"rows / dup-dropped / unique: " + ", ".join(
        f"{s}={m['rows']}/{m['duplicate_rows_dropped']}/{m['unique_qids']}" for s, m in meta.items()))
    print(f"common qids: {len(common)}; with gold: {len(qids)}; missing gold: {len(missing_gold)}; "
          f"distinct data_id: {out['n_data_id']}")

    stored_mismatch = defaultdict(int)
    for mod in MODS:
        qs = [q for q in qids if mod_of[q] == mod]
        fmt_sets = {tuple(sorted(recs[s][q]["correct"])) for q in qs for s, _ in SYSTEMS}
        fmt_sets |= {tuple(sorted(recs[s][q]["preds"])) for q in qs for s, _ in SYSTEMS}
        assert len(fmt_sets) == 1, (mod, fmt_sets)
        fmts = list(fmt_sets.pop())
        F = len(fmts)
        clusters = defaultdict(list)
        for i, q in enumerate(qs):
            clusters[data_id[q]].append(i)
        cl_keys = sorted(clusters)

        # layout per row: for sys, for scorer: F corrects + allc + awsame + awany ; then CR, CRraw
        layout = {}
        rows = []
        for q in qs:
            row = []
            for s, _ in SYSTEMS:
                r = recs[s][q]
                preds = [str(r["preds"][f]) for f in fmts]
                g = gold[q]
                for sc in SCORERS:
                    if sc == "stored":
                        c = [int(r["correct"][f]) for f in fmts]
                    elif sc == "strict":
                        c = [int(exact_match(p, g)) for p in preds]
                    else:
                        c = [numtol(p, g) for p in preds]
                    same = len({canon(p) for p in preds}) == 1
                    layout.setdefault((s, sc), len(row))
                    row += c + [int(all(c)), int(not any(c) and same), int(not any(c))]
                layout.setdefault((s, "CR"), len(row))
                row += [cr_pairs(preds, True), cr_pairs(preds, False)]
                for f in fmts:
                    if int(r["correct"][f]) != int(lora_reason(r["preds"][f], g) != "none"):
                        stored_mismatch[(s, mod)] += 1
            rows.append(row)

        def metrics(sums: list[float], n: int) -> dict:
            res = {}
            for s, _ in SYSTEMS:
                o = layout[(s, "CR")]
                cr, crraw = 100 * sums[o] / n, 100 * sums[o + 1] / n
                for sc in SCORERS:
                    o = layout[(s, sc)]
                    acc = [100 * sums[o + k] / n for k in range(F)]
                    res[(s, sc)] = {
                        "mean_acc": sum(acc) / F, "best": max(acc), "worst": min(acc),
                        "gap": max(acc) - min(acc),
                        "per_format": dict(zip(fmts, acc)),
                        "all_correct": 100 * sums[o + F] / n,
                        "all_wrong_same": 100 * sums[o + F + 1] / n,
                        "all_wrong_any": 100 * sums[o + F + 2] / n,
                        "CR": cr, "CR_raw": crraw,
                    }
            return res

        n = len(qs)
        point = metrics([sum(col) for col in zip(*rows)], n)
        keys = ["gap", "CR", "mean_acc", "all_correct", "all_wrong_same"]

        boot = {}
        for btype in ["question", "cluster"]:
            rng = random.Random(SEED)
            draws = defaultdict(list)  # (pair, sc, key) -> deltas
            for _ in range(B):
                if btype == "question":
                    samp = [rng.randrange(n) for _ in range(n)]
                else:
                    samp = [i for _k in range(len(cl_keys))
                            for i in clusters[cl_keys[rng.randrange(len(cl_keys))]]]
                m = metrics([sum(col) for col in zip(*(rows[i] for i in samp))], len(samp))
                for a, b in PAIRS:
                    for sc in SCORERS:
                        for k in keys:
                            draws[((a, b), sc, k)].append(m[(b, sc)][k] - m[(a, sc)][k])
            for kk, v in draws.items():
                lo, hi = ci(v)
                boot[(btype,) + kk] = {"lo": lo, "hi": hi,
                                       "p_delta_ge0": sum(1 for x in v if x >= 0) / B,
                                       "p_delta_le0": sum(1 for x in v if x <= 0) / B}

        # ---- JSON ---------------------------------------------------------------------------
        mo = {"n_questions": n, "n_data_id": len(cl_keys), "formats": fmts, "point": {}, "deltas": {}}
        for (s, sc), v in point.items():
            mo["point"].setdefault(s, {})[sc] = v
        for a, b in PAIRS:
            for sc in SCORERS:
                for k in keys:
                    d = point[(b, sc)][k] - point[(a, sc)][k]
                    mo["deltas"].setdefault(f"{a}->{b}", {}).setdefault(sc, {})[k] = {
                        "delta": d,
                        "question_boot": boot[("question", (a, b), sc, k)],
                        "cluster_boot": boot[("cluster", (a, b), sc, k)],
                    }
        out["by_modality"][mod] = mo

        # ---- print --------------------------------------------------------------------------
        print(f"\n=== {mod}: n={n} questions, {len(cl_keys)} data_id clusters, {F} formats ===")
        print(f"CR (strip+lower, verify_paper_numbers def): " + "  ".join(
            f"{s}={point[(s, 'stored')]['CR']:.1f}" for s, _ in SYSTEMS)
            + "   | raw-string (stats_lora def): " + "  ".join(
            f"{s}={point[(s, 'stored')]['CR_raw']:.1f}" for s, _ in SYSTEMS))
        print(f"{'scorer':7s} {'system':8s} {'mean':>6s} {'best':>6s} {'worst':>6s} {'gap':>6s} "
              f"{'allC':>6s} {'awSame':>6s} {'awAny':>6s}   per-format")
        for sc in SCORERS:
            for s, _ in SYSTEMS:
                v = point[(s, sc)]
                pf = " ".join(f"{f}={x:.1f}" for f, x in v["per_format"].items())
                print(f"{sc:7s} {s:8s} {v['mean_acc']:6.1f} {v['best']:6.1f} {v['worst']:6.1f} "
                      f"{v['gap']:6.1f} {v['all_correct']:6.1f} {v['all_wrong_same']:6.1f} "
                      f"{v['all_wrong_any']:6.1f}   {pf}")
        print("paired deltas: delta [question-boot 95%CI] p(>=0)/p(<=0) | [cluster-boot 95%CI] p(>=0)/p(<=0)")
        for a, b in PAIRS:
            for sc in SCORERS:
                for k in keys:
                    d = mo["deltas"][f"{a}->{b}"][sc][k]
                    qb, cb = d["question_boot"], d["cluster_boot"]
                    if sc != "stored" and k == "CR":
                        continue  # CR is scorer-independent; printed once under 'stored'
                    print(f"  {a:>7s}->{b:<7s} {sc:6s} {k:14s} {d['delta']:+6.1f} "
                          f"[{qb['lo']:+6.1f},{qb['hi']:+6.1f}] {qb['p_delta_ge0']:.3f}/{qb['p_delta_le0']:.3f} | "
                          f"[{cb['lo']:+6.1f},{cb['hi']:+6.1f}] {cb['p_delta_ge0']:.3f}/{cb['p_delta_le0']:.3f}")

    # ---- credit audit: stored-correct cells that strict EM rejects -------------------------
    print("\n=== stored-correct cells rejected by strict EM (credited only by eval_lora.match) ===")
    print(f"{'system':8s} {'modality':10s} {'cells':>6s} {'stored=1':>8s} {'strict=1':>8s} "
          f"{'st1&EM0':>7s} {'norm_eq':>7s} {'num_tol':>7s} {'substr':>7s} {'st0&EM1':>7s} {'stored!=recomputed':>18s}")
    examples = []
    for s, _ in SYSTEMS:
        for mod in MODS:
            qs = [q for q in qids if mod_of[q] == mod]
            cnt = defaultdict(int)
            for q in qs:
                r, g = recs[s][q], gold[q]
                for f, p in r["preds"].items():
                    p = str(p)
                    st, em = int(r["correct"][f]), int(exact_match(p, g))
                    cnt["cells"] += 1
                    cnt["stored1"] += st
                    cnt["strict1"] += em
                    if st and not em:
                        cnt["st1em0"] += 1
                        why = lora_reason(p, g)
                        cnt[why] += 1
                        if why == "substring" and len(examples) < 10 and (g, p) not in [(e["gold"], e["pred"]) for e in examples]:
                            examples.append({"system": s, "qid": q, "format": f, "gold": g, "pred": p})
                    if em and not st:
                        cnt["st0em1"] += 1
            out["credit_audit"].setdefault(s, {})[mod] = dict(cnt, stored_vs_recomputed_mismatch=stored_mismatch[(s, mod)])
            print(f"{s:8s} {mod:10s} {cnt['cells']:6d} {cnt['stored1']:8d} {cnt['strict1']:8d} "
                  f"{cnt['st1em0']:7d} {cnt['norm_eq']:7d} {cnt['numeric_tol']:7d} {cnt['substring']:7d} "
                  f"{cnt['st0em1']:7d} {stored_mismatch[(s, mod)]:18d}")
    out["substring_examples"] = examples

    # ---- gold-type audit (see CAVEAT in module docstring) -----------------------------------
    print("\n=== gold-type audit: stored preds are eval_lora.norm(raw); digit-bearing non-numeric "
          "golds can never be strict-EM correct ===")
    print(f"{'modality':10s} {'n_q':>4s} {'numeric':>7s} {'text_no_digit':>13s} {'text_with_digit':>15s}"
          f"   stored=1 cells on text_with_digit golds (base/lambda0/lambda1)")
    out["gold_type_audit"] = {}
    for mod in MODS:
        qs = [q for q in qids if mod_of[q] == mod]
        kind = {}
        for q in qs:
            g = gold[q]
            if _float(g) is not None:
                kind[q] = "numeric"
            elif re.search(r"\d", g):
                kind[q] = "text_with_digit"
            else:
                kind[q] = "text_no_digit"
        c = {k: sum(1 for v in kind.values() if v == k) for k in ["numeric", "text_no_digit", "text_with_digit"]}
        st = {s: sum(int(v) for q in qs if kind[q] == "text_with_digit" for v in recs[s][q]["correct"].values())
              for s, _ in SYSTEMS}
        out["gold_type_audit"][mod] = {"n_questions": len(qs), **c, "stored_correct_cells_text_with_digit": st}
        print(f"{mod:10s} {len(qs):4d} {c['numeric']:7d} {c['text_no_digit']:13d} {c['text_with_digit']:15d}   "
              + "/".join(str(st[s]) for s, _ in SYSTEMS))
    print("\nsubstring-credited examples (gold in pred, pred != gold, not numeric-equal):")
    for e in examples:
        print(f"  [{e['system']}] {e['qid']} {e['format']}: gold={e['gold']!r} pred={e['pred']!r}")

    path = os.path.join(HERE, "rescore_lora_results.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
