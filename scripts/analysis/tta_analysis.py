#!/usr/bin/env python3
"""Transcribe-then-answer (TTA) analysis, Qwen2.5-VL-7B on the expressive (v2) suite.

Stdlib only (plus the benchmark's own metrics via scoring.py); no inference.

    python scripts/analysis/tta_analysis.py

Input: results/review/tta_qwen.jsonl, 150 questions (results/review/tta_qids.txt; 50 per
modality) x 5 formats = 750 rows (graph includes the diagnostic text_only_deg listing). Stage 1
transcribes the image to plain text (``transcript``; ``transcript_truncated`` = hit the
2,560-token cap); stage 2 answers from the transcript alone (``prediction`` / ``raw_response``).
All accuracies use canonical_em (scripts/analysis/scoring.py) with the raw-response fallback.

1. Transcription fidelity against the true data object (benchmark/realworld_test.jsonl):
   tables   transcript cleaned (code fences, whole-line quotes), split on the dominant delimiter
            (comma via csv, tab, pipe, else 2+ spaces), read column-wise (header) or transposed
            (a line starting with a column name), whichever recovers more true columns; a row
            label column missing from the header is dropped; rows matched by order.
            cell_acc_all = share of ALL true cells reproduced (numeric: |p - a| <= 1% of |a|, exact when a = 0; label: folded string equality,
            "<idx>: " prefix ignored); cell_acc_shown = the same over the columns the transcript
            contains; numeric_cell_acc over numeric true cells; rows_ok / cols_ok = transcript
            has exactly the true number of data rows / recovers every true column;
            cell_recall_layout_free = share of true cells matched by a distinct transcript token
            regardless of position.
   series   points = lines that are exactly "<int t><sep><number>"; point_acc = share of true
            points t whose transcribed value is within 1% of the true value (point_acc_range:
            within 1% of the series range); length_ok = number of points equals the length. GAF /
            recurrence transcripts list matrix entries, not series values, so they score ~0.
   graphs   "nodes: ..." line and edge lines "<u> <sep> <v>" (sep ",", "--", "-", "|" suffix
            ignored), undirected; edge precision / recall / F1 against the true edge set;
            nodes_ok = node count equals the true count.
   Mean and median per format; share of transcripts truncated at the token cap.
2. Accuracy per format on the same 750 (question, format) rows: (a) image = results/v2/
   v2_qwen.jsonl (plain prompt), (b) TTA = answer from own transcript, (c) raw text =
   results/v2/rawtext_qwen.jsonl (true data serialised as text; one answer per question, so the
   same for every format). Paired object-cluster bootstrap CIs (cluster = data_id; B=5000;
   v2_analysis.contrast) for b-a, c-a, c-b per format and per modality.
3. Decomposition: best-worst format gap per modality under (a) and (b) (best/worst chosen per
   condition), with object CIs and sign-flip p, and the paired change of the gap (b) - (a).
   Spearman across the 15 modality x format cells (rc.spearman, permutation p, B=10000) of the
   fidelity measure (tables: cell_acc_all; series: point_acc; graphs: edge F1) vs (i) image
   accuracy and (ii) TTA - image; also within modality.
4. Error source for wrong TTA answers, for templates where the answer can be recomputed from a
   data object: tabular value_extraction, ranking, outlier_detection, correlation; graph
   degree_query (all five phrasings), edge_count, shortest_path, diameter, cycle_detection,
   bipartite_check, community (largest-component size / fraction), clustering (average
   clustering). Each template's recomputation is validated on the TRUE object first and used only
   where it reproduces the key (agreement reported). The same function applied to the parsed
   transcript gives the transcript-implied answer: wrong TTA answer + implied == key ->
   "transcript right, answer wrong"; implied != key or not computable -> "transcript wrong /
   incomplete on the queried information". Not defined (excluded): time-series change_point and
   seasonality_detection (keys come from generator metadata, not computable from values), graph
   centrality, bridge and counterfactual edge-removal questions.

Writes scripts/analysis/tta_analysis_results.json and paper_tables/tab_tta.tex.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict, deque

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import review_common as rc  # noqa: E402
import scoring as sc  # noqa: E402
import v2_analysis as v2a  # noqa: E402

TTA = os.path.join(ROOT, "results", "review", "tta_qwen.jsonl")
FMTS = {"tabular": ["table_image", "text_only", "heatmap", "bar_chart", "scatter_plot"],
        "timeseries": ["line_plot", "heatmap", "gaf", "recurrence_plot", "text_only"],
        "graph": ["node_link", "circular_layout", "adjacency_matrix", "text_only",
                  "text_only_deg"]}
FID_KEY = {"tabular": "cell_acc_all", "timeseries": "point_acc", "graph": "edge_f1"}


# ---------------------------------------------------------------- parsing helpers
def is_num(x):
    try:
        float(str(x).strip())
        return True
    except ValueError:
        return False


def close(p, a, rel=0.01):
    if a == 0:
        return abs(p) < 1e-9
    return abs(p - a) <= rel * abs(a)


def cell_ok(t, a):
    if t is None:
        return False
    t = str(t).strip()
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        try:
            return close(float(t.replace("%", "")), float(a))
        except ValueError:
            return False
    return sc.fold(sc.strip_index(t)) == sc.fold(a) or sc.fold(t) == sc.fold(a)


def _clean_lines(text):
    out = []
    for ln in text.splitlines():
        t = ln.strip()
        if not t or t.startswith("```"):
            continue
        if len(t) >= 2 and t[0] == t[-1] == '"':
            t = t[1:-1]
        out.append(t)
    return out


def _split_lines(lines):
    counts = {d: sum(ln.count(d) for ln in lines) for d in (",", "\t", "|")}
    d = max(counts, key=counts.get)
    if counts[d] == 0:
        return [re.split(r"\s{2,}", ln) for ln in lines]
    if d == ",":
        return [[c.strip() for c in r] for r in csv.reader(io.StringIO("\n".join(lines)))]
    return [[c.strip() for c in ln.strip("|").split(d)] for ln in lines]


def parse_table(text):
    """(header, rows) of a CSV/TSV/pipe/whitespace transcript (code fences and line quotes
    removed)."""
    rows = _split_lines(_clean_lines(text))
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return [], []
    hdr, body = rows[0], rows[1:]
    body = [r[1:] if len(r) == len(hdr) + 1 else r for r in body]  # unlabelled row-label column
    return hdr, body


def table_columns(text, true_cols):
    """Column values recovered from a transcript, column-wise or transposed (a line whose first
    field is a true column name lists that column's values); returns the orientation that
    recovers more true column names. Values keep the transcript's row order."""
    hdr, body = parse_table(text)
    colwise = {}
    for c in true_cols:
        for j, h in enumerate(hdr):
            if sc.fold(sc.strip_index(h)) == sc.fold(c):
                colwise[c] = [r[j] if j < len(r) else None for r in body]
                break
    trans = {}
    for r in _split_lines(_clean_lines(text)):
        if not r:
            continue
        for c in true_cols:
            if c not in trans and sc.fold(r[0]) == sc.fold(c):
                trans[c] = r[1:]
    return (trans, "transposed") if len(trans) > len(colwise) else (colwise, "columns")


def bag_recall(text, data):
    """Layout-free: share of true cells matched by a distinct transcript token."""
    toks = [t for r in _split_lines(_clean_lines(text)) for t in r if t != ""]
    used = [False] * len(toks)
    hit = tot = 0
    for row in data:
        for a in row.values():
            tot += 1
            for i, t in enumerate(toks):
                if not used[i] and cell_ok(t, a):
                    used[i] = True
                    hit += 1
                    break
    return 100 * hit / tot if tot else 0.0


def table_fidelity(transcript, data):
    cols = list(data[0].keys())
    got, orient = table_columns(transcript, cols)
    hdr, rows = parse_table(transcript)
    n_rows_t = max((len(v) for v in got.values()), default=len(rows))
    tot = ok = tot_shown = ok_shown = tot_num = ok_num = 0
    for i, true_row in enumerate(data):
        for c in cols:
            a = true_row[c]
            numeric = isinstance(a, (int, float)) and not isinstance(a, bool)
            vals = got.get(c)
            t = vals[i] if vals is not None and i < len(vals) else None
            good = cell_ok(t, a)
            tot += 1
            ok += good
            if c in got:
                tot_shown += 1
                ok_shown += good
            if numeric:
                tot_num += 1
                ok_num += good
    return {"cell_acc_all": 100 * ok / tot, "cell_acc_shown": 100 * ok_shown / tot_shown
            if tot_shown else 0.0, "numeric_cell_acc": 100 * ok_num / tot_num if tot_num else
            float("nan"), "cell_recall_layout_free": bag_recall(transcript, data),
            "rows_ok": float(n_rows_t == len(data)),
            "cols_ok": float(len(got) == len(cols)), "cols_matched": len(got),
            "cols_true": len(cols), "rows_transcribed": n_rows_t,
            "transposed": float(orient == "transposed")}, got


PT = re.compile(r"^\s*(\d+)\s*[,:\t ]\s*(-?\d+(?:\.\d+)?(?:[eE]-?\d+)?)\s*$")


def parse_series(text):
    pts = {}
    for ln in text.splitlines():
        m = PT.match(ln)
        if m:
            t = int(m.group(1))
            if t not in pts:
                pts[t] = float(m.group(2))
    return pts


def series_fidelity(transcript, data):
    vals = [float(v) for v in data]
    pts = parse_series(transcript)
    rngv = (max(vals) - min(vals)) or 1.0
    ok = sum(1 for t, a in enumerate(vals) if t in pts and close(pts[t], a))
    ok_r = sum(1 for t, a in enumerate(vals) if t in pts and abs(pts[t] - a) <= 0.01 * rngv)
    return {"point_acc": 100 * ok / len(vals), "point_acc_range": 100 * ok_r / len(vals),
            "length_ok": float(len(pts) == len(vals)), "points_transcribed": len(pts)}, pts


ED = re.compile(r"^\s*(\d+)\s*(?:,|--|-|\s)\s*(\d+)\b")


def parse_graph(text):
    nodes, edges = None, set()
    for ln in text.splitlines():
        s = ln.strip()
        if s.lower().startswith("nodes"):
            nodes = {int(x) for x in re.findall(r"\d+", s.split(":", 1)[-1])}
            continue
        m = ED.match(s)
        if m:
            u, v = int(m.group(1)), int(m.group(2))
            edges.add((min(u, v), max(u, v)))
    if nodes is None:
        nodes = {x for e in edges for x in e}
    return nodes, edges


def true_graph(data):
    key = "edges" if "edges" in data else "links"
    nodes = {int(n["id"]) for n in data["nodes"]}
    edges = {(min(int(e["source"]), int(e["target"])), max(int(e["source"]), int(e["target"])))
             for e in data[key]}
    return nodes, edges


def graph_fidelity(transcript, data):
    tn, te = true_graph(data)
    pn, pe = parse_graph(transcript)
    tp = len(pe & te)
    prec = 100 * tp / len(pe) if pe else 0.0
    rec = 100 * tp / len(te) if te else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"edge_precision": prec, "edge_recall": rec, "edge_f1": f1,
            "nodes_ok": float(len(pn) == len(tn)), "edges_transcribed": len(pe)}, (pn, pe)


# ---------------------------------------------------------------- template recomputation
def adj_of(nodes, edges):
    adj = {n: set() for n in nodes}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


def components(adj):
    seen, comps = set(), []
    for s in adj:
        if s in seen:
            continue
        comp, dq = [], deque([s])
        seen.add(s)
        while dq:
            x = dq.popleft()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    dq.append(y)
        comps.append(comp)
    return comps


def bfs_dist(adj, s):
    d, dq = {s: 0}, deque([s])
    while dq:
        x = dq.popleft()
        for y in adj[x]:
            if y not in d:
                d[y] = d[x] + 1
                dq.append(y)
    return d


def graph_answer(q, nodes, edges):
    adj = adj_of(nodes, edges)
    deg = {n: len(adj[n]) for n in adj}
    ql = q.lower()
    ns = [int(x) for x in re.findall(r"node (\d+)", q)]
    if "average degree among" in ql:
        return sum(deg.get(n, 0) for n in ns) / len(ns)
    if "average degree of the graph" in ql:
        return 2 * len(edges) / len(nodes) if nodes else None
    if "degree of node" in ql or "immediate neighbors" in ql:
        return deg.get(ns[0], 0)
    if "highest degree" in ql:
        if not deg:
            return None
        mx = max(deg.values())
        return [n for n in adj if deg[n] == mx]
    if "how many edges" in ql:
        return len(edges)
    if "shortest path length" in ql:
        if ns[0] not in adj:
            return None
        return bfs_dist(adj, ns[0]).get(ns[1])
    comps = components(adj) if adj else []
    if "largest connected component" in ql and "how many" in ql:
        return max(map(len, comps)) if comps else None
    if "fraction of nodes" in ql:
        return max(map(len, comps)) / len(adj) if comps else None
    if "diameter" in ql:
        big = max(comps, key=len) if comps else []
        return max((max(bfs_dist(adj, s).values()) for s in big), default=None)
    if "contain a cycle" in ql:
        return "yes" if len(edges) > sum(len(c) - 1 for c in comps) else "no"
    if "bipartite" in ql:
        color = {}
        for s in adj:
            if s in color:
                continue
            color[s] = 0
            dq = deque([s])
            while dq:
                x = dq.popleft()
                for y in adj[x]:
                    if y not in color:
                        color[y] = 1 - color[x]
                        dq.append(y)
                    elif color[y] == color[x]:
                        return "no"
        return "yes"
    if "clustering coefficient" in ql:
        cs = []
        for n in adj:
            k = len(adj[n])
            if k < 2:
                cs.append(0.0)
                continue
            nb = list(adj[n])
            links = sum(1 for i in range(k) for j in range(i + 1, k) if nb[j] in adj[nb[i]])
            cs.append(2 * links / (k * (k - 1)))
        return sum(cs) / len(cs) if cs else None
    return None


def table_answer(task, q, cols_rows):
    """cols_rows: dict column -> list of raw cell values (strings or numbers), aligned rows."""
    quoted = re.findall(r"'([^']*)'", q)

    def num(v):
        try:
            return float(str(v).strip())
        except (ValueError, TypeError):
            return None
    if task == "value_extraction":
        col, r = quoted[0], int(re.search(r"row (\d+)", q).group(1))
        vals = cols_rows.get(col)
        return None if vals is None or r >= len(vals) else vals[r]
    if task in ("ranking", "outlier_detection"):
        lab, val = quoted[0], quoted[1]
        L, V = cols_rows.get(lab), cols_rows.get(val)
        if L is None or V is None:
            return None
        pairs = [(lab_v, num(v)) for lab_v, v in zip(L, V) if num(v) is not None]
        if not pairs:
            return None
        if task == "ranking":
            pick = (min if "lowest" in q else max)(pairs, key=lambda p: p[1])
            return pick[0]
        mu = sum(p[1] for p in pairs) / len(pairs)
        return max(pairs, key=lambda p: abs(p[1] - mu))[0]
    if task == "correlation":
        A, B = cols_rows.get(quoted[0]), cols_rows.get(quoted[1])
        if A is None or B is None:
            return None
        xy = [(num(a), num(b)) for a, b in zip(A, B) if num(a) is not None and num(b) is not None]
        if len(xy) < 3:
            return None
        mx = sum(x for x, _ in xy) / len(xy)
        my = sum(y for _, y in xy) / len(xy)
        cov = sum((x - mx) * (y - my) for x, y in xy)
        return "positive" if cov > 0 else "negative"
    return None


def answer_ok(implied, key):
    if implied is None:
        return False
    if isinstance(implied, list):
        return any(answer_ok(x, key) for x in implied) and len(implied) == 1
    if isinstance(implied, float) and is_num(key):
        k = str(key).strip()
        d = len(k.split(".")[1]) if "." in k else 0
        return abs(implied - float(k)) <= 0.5 * 10 ** (-d) + 1e-9 or close(implied, float(k))
    if is_num(implied) and is_num(key):
        return close(float(implied), float(key))
    return sc.fold(sc.strip_index(str(implied))) == sc.fold(key)


GRAPH_TPL = {"degree_query", "edge_count", "shortest_path", "diameter", "cycle_detection",
             "bipartite_check", "community", "clustering"}
TAB_TPL = {"value_extraction", "ranking", "outlier_detection", "correlation"}


def implied(b, transcript=None):
    """Answer recomputed from the true object (transcript None) or from a parsed transcript."""
    md, task, q = b["modality"], b["task"], b["question"]
    if md == "graph" and task in GRAPH_TPL:
        if transcript is None:
            n, e = true_graph(b["data"])
        else:
            n, e = parse_graph(transcript)
        if task == "community" and "bridge" in q.lower():
            return "n/a"
        return graph_answer(q, n, e)
    if md == "tabular" and task in TAB_TPL:
        if transcript is None:
            cols = {c: [row[c] for row in b["data"]] for c in b["data"][0]}
        else:
            cols, _ = table_columns(transcript, list(b["data"][0]))
        return table_answer(task, q, cols)
    return "n/a"


# ---------------------------------------------------------------- main
def summ(vals):
    vals = [v for v in vals if v == v]
    return {"mean": statistics.mean(vals) if vals else float("nan"),
            "median": statistics.median(vals) if vals else float("nan"), "n": len(vals)}


def ci(items, q2d, perm=False):
    r = v2a.contrast(items, q2d, perm=perm)
    o = {"est": r["est"], "lo": r["obj_ci"]["lo"], "hi": r["obj_ci"]["hi"], "n": r["n_units"],
         "n_obj": r["n_obj"]}
    if perm:
        o["p"] = r["obj_perm_p"]
    return o


def main():
    meta = rc.load_bench()
    q2d = {q: b["data_id"] for q, b in meta.items()}
    tta = {(r["question_id"], r["viz_type"]): r for r in rc.read_jsonl(TTA)}
    qids = [q.strip() for q in open(os.path.join(ROOT, "results", "review", "tta_qids.txt"))
            if q.strip()]
    plain, _ = v2a.dedupe(rc.read_jsonl(os.path.join(ROOT, "results", "v2", "v2_qwen.jsonl")))
    raw = {r["question_id"]: r for r in rc.read_jsonl(os.path.join(ROOT, "results", "v2",
                                                                    "rawtext_qwen.jsonl"))
           if str(r.get("prediction")) != "[ERROR]"}
    keys = sorted(tta)
    assert len(keys) == 750 and all(k in plain for k in keys) and all(q in raw for q in qids)
    cat = rc.v2_cats(meta, strict=True)
    nc = rc.near_constant()
    res = {"definition": __doc__, "n_rows": len(keys), "n_questions": len(qids),
           "tasks": dict(Counter(f"{meta[q]['modality']}|{meta[q]['task']}" for q in qids)),
           "questions_in_near_constant_cells": sum((meta[q]["modality"], meta[q]["task"]) in nc
                                                   for q in qids),
           "questions_answerable_all_main_formats_strict": sum(
               all(cat[q][f] in "VD" for f in rc.FORMATS[meta[q]["modality"]]) for q in qids)}

    # ---- 1. fidelity
    fid = {}
    per_row_fid = {}
    for (q, f) in keys:
        b, r = meta[q], tta[(q, f)]
        t = r.get("transcript") or ""
        if b["modality"] == "tabular":
            m, _ = table_fidelity(t, b["data"])
        elif b["modality"] == "timeseries":
            m, _ = series_fidelity(t, b["data"])
        else:
            m, _ = graph_fidelity(t, b["data"])
        m["truncated"] = 100.0 * (r.get("transcript_truncated") in (True, "True"))
        per_row_fid[(q, f)] = m
    for md, fmts in FMTS.items():
        for f in fmts:
            rows = [per_row_fid[k] for k in keys if k[1] == f and meta[k[0]]["modality"] == md]
            fid[f"{md}|{f}"] = {k: summ([x[k] for x in rows]) for k in rows[0]}
    res["fidelity"] = fid

    # ---- 2. accuracy under three conditions
    A = {k: sc.row_score(plain[k]) for k in keys}
    B = {k: sc.row_score(tta[k]) for k in keys}
    Cr = {k: sc.row_score(raw[k[0]]) for k in keys}
    acc = {}
    for md, fmts in FMTS.items():
        for f in fmts + ["ALL"]:
            ks = [k for k in keys if meta[k[0]]["modality"] == md and (f == "ALL" or k[1] == f)]
            acc[f"{md}|{f}"] = {
                "n": len(ks), "image": rc.mean100(A[k] for k in ks),
                "tta": rc.mean100(B[k] for k in ks), "rawtext": rc.mean100(Cr[k] for k in ks),
                "tta_minus_image": ci([(k[0], B[k] - A[k]) for k in ks], q2d),
                "raw_minus_image": ci([(k[0], Cr[k] - A[k]) for k in ks], q2d),
                "raw_minus_tta": ci([(k[0], Cr[k] - B[k]) for k in ks], q2d)}
    res["accuracy"] = acc

    # ---- 3. decomposition
    gaps = {}
    for md, fmts in FMTS.items():
        qs = [q for q in qids if meta[q]["modality"] == md]
        rec = {}
        for lab, E in (("image", A), ("tta", B)):
            EM = {q: {f: E[(q, f)] for f in fmts} for q in qs}
            g = v2a.gap_block(EM, qs, fmts, q2d)
            rec[lab] = {"est": g["est"], "lo": g["obj_ci"]["lo"], "hi": g["obj_ci"]["hi"],
                        "p": g["obj_perm_p"], "best": g["best"], "worst": g["worst"],
                        "per_format": g["per_format"], "n_q": g["n_q"]}
        ba, wa, bb, wb = rec["image"]["best"], rec["image"]["worst"], rec["tta"]["best"], rec[
            "tta"]["worst"]
        rec["gap_change_tta_minus_image"] = ci(
            [(q, (B[(q, bb)] - B[(q, wb)]) - (A[(q, ba)] - A[(q, wa)])) for q in qs], q2d,
            perm=True)
        # same best/worst pair as the image condition
        rec["tta_gap_on_image_pair"] = ci([(q, B[(q, ba)] - B[(q, wa)]) for q in qs], q2d)
        if md == "graph":  # main four formats only (text_only_deg is a diagnostic)
            m4 = fmts[:4]
            for lab, E in (("image_main4", A), ("tta_main4", B)):
                EM = {q: {f: E[(q, f)] for f in m4} for q in qs}
                g = v2a.gap_block(EM, qs, m4, q2d)
                rec[lab] = {"est": g["est"], "lo": g["obj_ci"]["lo"], "hi": g["obj_ci"]["hi"],
                            "best": g["best"], "worst": g["worst"]}
        gaps[md] = rec
    res["gaps"] = gaps

    cells = []
    for md, fmts in FMTS.items():
        for f in fmts:
            cells.append({"cell": f"{md}|{f}", "modality": md,
                          "fidelity": fid[f"{md}|{f}"][FID_KEY[md]]["mean"],
                          "image": acc[f"{md}|{f}"]["image"], "tta": acc[f"{md}|{f}"]["tta"],
                          "tta_minus_image": acc[f"{md}|{f}"]["tta_minus_image"]["est"]})
    corr = {}
    for lab, sel in (("all15", lambda c: True),) + tuple(
            (md, (lambda c, md=md: c["modality"] == md)) for md in FMTS):
        cs = [c for c in cells if sel(c)]
        for y in ("image", "tta", "tta_minus_image"):
            rho, p = rc.spearman([c["fidelity"] for c in cs], [c[y] for c in cs])
            corr[f"{lab}|fidelity_vs_{y}"] = {"rho": rho, "p": p, "n": len(cs)}
    res["cells"] = cells
    res["spearman"] = corr

    # ---- 4. error source
    valid = defaultdict(lambda: [0, 0])
    for q in qids:
        b = meta[q]
        imp = implied(b)
        if imp == "n/a":
            continue
        valid[f"{b['modality']}|{b['task']}"][0] += 1
        valid[f"{b['modality']}|{b['task']}"][1] += answer_ok(imp, b["answer"])
    ok_q = {q for q in qids if implied(meta[q]) != "n/a"
            and answer_ok(implied(meta[q]), meta[q]["answer"])}
    src = defaultdict(Counter)
    tbl = defaultdict(Counter)
    for k in keys:
        q, f = k
        if q not in ok_q:
            continue
        b = meta[q]
        imp_t = implied(b, tta[k].get("transcript") or "")
        t_ok = answer_ok(imp_t, b["answer"])
        a_ok = B[k] >= 1
        lab = ("transcript_right" if t_ok else "transcript_wrong") + "|" + (
            "answer_right" if a_ok else "answer_wrong")
        src[f"{b['modality']}|{f}"][lab] += 1
        src[f"{b['modality']}|ALL"][lab] += 1
        tbl[f"{b['modality']}|{b['task']}"][lab] += 1
    res["error_source"] = {
        "template_validation": {k: {"n": v[0], "recomputation_matches_key": v[1]}
                                for k, v in valid.items()},
        "n_questions_used": len(ok_q),
        "by_format": {k: dict(v) for k, v in src.items()},
        "by_task": {k: dict(v) for k, v in tbl.items()}}

    with open(os.path.join(HERE, "tta_analysis_results.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    report(res)
    write_tex(res)


def report(res):
    print(f"TTA: {res['n_rows']} rows, {res['n_questions']} questions; tasks {res['tasks']}")
    print(f"  questions in near-constant cells: {res['questions_in_near_constant_cells']}; "
          f"answerable in every main format (strict v2): "
          f"{res['questions_answerable_all_main_formats_strict']}")
    print("\n[1] fidelity (mean / median) and truncation share")
    for k, v in res["fidelity"].items():
        md = k.split("|")[0]
        keys = {"tabular": ["cell_acc_all", "cell_acc_shown", "numeric_cell_acc", "rows_ok",
                            "cols_ok"],
                "timeseries": ["point_acc", "point_acc_range", "length_ok"],
                "graph": ["edge_precision", "edge_recall", "edge_f1", "nodes_ok"]}[md]
        print(f"  {k:28s} " + "  ".join(f"{x} {v[x]['mean']:.1f}/{v[x]['median']:.1f}"
                                         for x in keys)
              + f"  truncated {v['truncated']['mean']:.0f}%")
    print("\n[2] accuracy (canonical EM) image / TTA / raw text, paired object CIs")
    for k, v in res["accuracy"].items():
        print(f"  {k:28s} n={v['n']:3d} image {v['image']:5.1f} tta {v['tta']:5.1f} raw "
              f"{v['rawtext']:5.1f} | tta-img {fmt(v['tta_minus_image'])} | raw-img "
              f"{fmt(v['raw_minus_image'])} | raw-tta {fmt(v['raw_minus_tta'])}")
    print("\n[3] best-worst gap per modality")
    for md, g in res["gaps"].items():
        a, b = g["image"], g["tta"]
        print(f"  {md:10s} image {a['est']:.1f} [{a['lo']:.1f},{a['hi']:.1f}] p={a['p']:.4f} "
              f"({a['best']}/{a['worst']}) | tta {b['est']:.1f} [{b['lo']:.1f},{b['hi']:.1f}] "
              f"p={b['p']:.4f} ({b['best']}/{b['worst']}) | change "
              f"{fmt(g['gap_change_tta_minus_image'])} p={g['gap_change_tta_minus_image']['p']:.4f}"
              f" | tta on image pair {fmt(g['tta_gap_on_image_pair'])}")
        if "image_main4" in g:
            print(f"  {'':10s} main 4 formats: image {g['image_main4']['est']:.1f} "
                  f"({g['image_main4']['best']}/{g['image_main4']['worst']}) tta "
                  f"{g['tta_main4']['est']:.1f} ({g['tta_main4']['best']}/{g['tta_main4']['worst']})")
    print("  cells (fidelity, image, tta, tta-image):")
    for c in res["cells"]:
        print(f"    {c['cell']:28s} {c['fidelity']:6.1f} {c['image']:6.1f} {c['tta']:6.1f} "
              f"{c['tta_minus_image']:+6.1f}")
    for k, v in res["spearman"].items():
        print(f"  spearman {k:36s} rho={v['rho']:+.3f} p={v['p']:.4f} n={v['n']}")
    print("\n[4] error source (templates recomputable from data)")
    print(f"  template validation (recomputation from TRUE data matches key): "
          f"{res['error_source']['template_validation']}")
    print(f"  questions used: {res['error_source']['n_questions_used']}")
    for k, v in res["error_source"]["by_format"].items():
        print(f"  {k:28s} {v}")
    for k, v in res["error_source"]["by_task"].items():
        print(f"  {k:28s} {v}")


def fmt(d):
    return f"{d['est']:+.1f} [{d['lo']:+.1f},{d['hi']:+.1f}]"


def write_tex(res):
    rows, mids = [], []
    for i, (md, fmts) in enumerate(FMTS.items()):
        if i:
            mids.append(len(rows))
        for f in fmts:
            fd = res["fidelity"][f"{md}|{f}"]
            a = res["accuracy"][f"{md}|{f}"]
            rows.append([md, f, f"{fd[FID_KEY[md]]['mean']:.1f}",
                         f"{fd['truncated']['mean']:.0f}", str(a["n"]), f"{a['image']:.1f}",
                         f"{a['tta']:.1f}", f"{a['rawtext']:.1f}", fmt(a["tta_minus_image"])])
        g = res["gaps"][md]
        rows.append([md, "gap (best-worst)", "", "", "",
                     f"{g['image']['est']:.1f}", f"{g['tta']['est']:.1f}", "0.0",
                     fmt(g["gap_change_tta_minus_image"])])
    rc.write_tabular(
        os.path.join(ROOT, "paper_tables", "tab_tta.tex"),
        "Transcribe-then-answer, Qwen2.5-VL-7B, 150 questions (50 per modality) x 5 expressive\n"
        "formats. Fidelity: tables = share of true cells reproduced (numeric within 1%); series =\n"
        "share of points within 1%; graphs = edge F1 (all %). Trunc. = transcripts that hit the\n"
        "2,560-token cap (%). Canonical EM (%) answering from the image, from the model's own\n"
        "transcript (TTA), and from the true data as text. Last column: TTA minus image, paired\n"
        "object-cluster 95% CI; on gap rows, the change of the best-worst gap. The 50 series\n"
        "questions are change-point and seasonality questions (near-constant templates).",
        "l l r r r r r r l",
        ["Modality", "Format", "Fidelity", "Trunc.", "$n$", "Image", "TTA", "Raw text",
         "TTA$-$image [CI]"], rows, midrules=mids)
    print("\nwrote scripts/analysis/tta_analysis_results.json, paper_tables/tab_tta.tex")


if __name__ == "__main__":
    main()
