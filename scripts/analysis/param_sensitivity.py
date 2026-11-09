#!/usr/bin/env python3
"""Renderer-limit sensitivity: which questions each v1 legibility limit affects, and how
accuracy differs on affected vs unaffected questions.

Stdlib only, no inference.

    python scripts/analysis/param_sensitivity.py

The v1 renderer limits are canvas-legibility defaults of the 1024x768 renderers
(src/rendering/*_renderers.py), not parameters tuned against question requirements:
  tabular  table_image    first 12 rows        TABLE_IMAGE_MAX_ROWS = 12
  tabular  text_only      first 14 rows        TEXT_VIEW_MAX_ROWS = 14
  tabular  heatmap        first 20 rows        HEATMAP_MAX_ROWS = 20 (numeric columns only)
  timeseries text_only    ~43 visible lines    TEXT_VIEW_MAX_POINTS = 180 listed, canvas clips
  graph    text_only      ~43 visible lines    TEXT_VIEW_MAX_EDGES = 180 listed, canvas clips
  graph    node_link / circular_layout  node labels only when nodes < 30
  graph    adjacency_matrix  every tick labelled only when nodes <= 24 (else every k-th)
For each limit:
  exceeds   questions whose object exceeds the limit (rows / points / edges / nodes);
  flips     questions whose answerability class for that format is A under the limit but V/D
            when the limit is removed (answerability.py re-run with the constant set to
            infinity), i.e. the limit removes information the question needs;
  EM        v1 exact match on the limited format for the four core models, exceeds vs within
            (and flips vs exceeds-but-still-answerable), with an object-cluster bootstrap 95%
            CI on the difference (objects resampled, B=5000, random.Random(0));
  control   the same exceeds-vs-within difference on a format that draws the whole object
            (tabular bar_chart, timeseries line_plot, graph node_link for text_only); for the
            graph label/tick limits, global (non-node-referencing) tasks in the same format
            serve as control. DiD = limited-format difference minus control difference;
  v2        Qwen2.5-VL-7B v2 (answer-complete renderer, no row/line truncation) minus v1 on
            the same format, exceeds vs within.

Writes scripts/analysis/param_sensitivity_results.json and
paper_tables/tab_param_limits.tex.
"""
from __future__ import annotations

import os
import random
from collections import Counter, defaultdict

import review_common as rc

ans = rc.ans
INF = 10 ** 9
B_BOOT = 5000


def n_edges(d):
    return len(d["edges"] if "edges" in d else d["links"])


LIMITS = [
    {"id": "tab_table_rows12", "mod": "tabular", "fmt": "table_image",
     "size": lambda b: len(b["data"]), "exceeds": lambda s: s > 12, "const": "TAB_TABLE_ROWS",
     "control": "bar_chart", "label": "table\\_image: first 12 rows"},
    {"id": "tab_text_rows14", "mod": "tabular", "fmt": "text_only",
     "size": lambda b: len(b["data"]), "exceeds": lambda s: s > 14, "const": "TAB_TEXT_ROWS",
     "control": "bar_chart", "label": "text\\_only (tab.): first 14 rows"},
    {"id": "tab_heat_rows20", "mod": "tabular", "fmt": "heatmap",
     "size": lambda b: len(b["data"]), "exceeds": lambda s: s > 20, "const": "TAB_HEAT_ROWS",
     "control": "bar_chart", "label": "heatmap (tab.): first 20 rows"},
    {"id": "ts_text_lines43", "mod": "timeseries", "fmt": "text_only",
     "size": lambda b: len(b["data"]), "exceeds": lambda s: s > 43, "const": "TS_TEXT_VISIBLE",
     "control": "line_plot", "label": "text\\_only (t.s.): ~43 lines"},
    {"id": "g_text_lines43", "mod": "graph", "fmt": "text_only",
     "size": lambda b: n_edges(b["data"]), "exceeds": lambda s: s > 43,
     "const": "G_TEXT_VISIBLE_EDGES", "control": "node_link",
     "label": "text\\_only (graph): ~43 edges"},
    {"id": "g_nodelabels30_nl", "mod": "graph", "fmt": "node_link",
     "size": lambda b: len(b["data"]["nodes"]), "exceeds": lambda s: s >= 30,
     "const": "G_LABEL_NODES_LT", "control": "global_tasks",
     "label": "node\\_link: labels iff nodes < 30"},
    {"id": "g_nodelabels30_circ", "mod": "graph", "fmt": "circular_layout",
     "size": lambda b: len(b["data"]["nodes"]), "exceeds": lambda s: s >= 30,
     "const": "G_LABEL_NODES_LT", "control": "global_tasks",
     "label": "circular: labels iff nodes < 30"},
    {"id": "g_adjticks24", "mod": "graph", "fmt": "adjacency_matrix",
     "size": lambda b: len(b["data"]["nodes"]), "exceeds": lambda s: s > 24,
     "const": "G_ADJ_TICKS_LE", "control": "global_tasks",
     "label": "adjacency: all ticks iff nodes <= 24"},
]


def two_group_ci(groups, q2d, B=B_BOOT):
    """groups: {"a": [(q, v)], "b": [(q, v)]} -> (est a-b, lo, hi); objects resampled."""
    per_obj = defaultdict(lambda: [0.0, 0, 0.0, 0])
    for g, items in groups.items():
        off = 0 if g == "a" else 2
        for q, v in items:
            o = per_obj[q2d[q]]
            o[off] += v
            o[off + 1] += 1
    objs = list(per_obj.values())
    na = sum(o[1] for o in objs)
    nb = sum(o[3] for o in objs)
    if not na or not nb:
        return float("nan"), None, None
    est = 100 * (sum(o[0] for o in objs) / na - sum(o[2] for o in objs) / nb)
    rng = random.Random(0)
    m = len(objs)
    stats = []
    for _ in range(B):
        sa = ca = sb = cb = 0.0
        for _ in range(m):
            o = objs[int(rng.random() * m)]
            sa += o[0]
            ca += o[1]
            sb += o[2]
            cb += o[3]
        if ca and cb:
            stats.append(100 * (sa / ca - sb / cb))
    lo, hi = rc.cc.pct(stats)
    return est, lo, hi


def main():
    meta = rc.load_bench()
    q2d = {q: b["data_id"] for q, b in meta.items()}
    cat = rc.v1_cats(meta)
    EM = {m: {k: rc.em_val(r) for k, r in rc.load_v1(m).items()} for m in rc.CORE4}
    v2rows, _ = rc.load_v2_qwen(meta)
    EM2 = {k: rc.em_val(r) for k, r in v2rows.items()}

    # descriptive size distributions
    sizes = {
        "tabular_rows": Counter(len(b["data"]) for b in meta.values()
                                if b["modality"] == "tabular"),
        "timeseries_len": Counter(len(b["data"]) for b in meta.values()
                                  if b["modality"] == "timeseries"),
        "graph_nodes": Counter(len(b["data"]["nodes"]) for b in meta.values()
                               if b["modality"] == "graph"),
        "graph_edges": Counter(n_edges(b["data"]) for b in meta.values()
                               if b["modality"] == "graph"),
    }
    import re
    ve_rows = [int(re.search(r"in row (\d+)", b["question"]).group(1)) for b in meta.values()
               if b["modality"] == "tabular" and b["task"] == "value_extraction"]
    objs_by_mod = defaultdict(dict)
    for b in meta.values():
        objs_by_mod[b["modality"]][b["data_id"]] = b
    print("[object sizes]")
    for md, key, fn in (("tabular", "rows", lambda b: len(b["data"])),
                        ("timeseries", "length", lambda b: len(b["data"])),
                        ("graph", "nodes", lambda b: len(b["data"]["nodes"])),
                        ("graph", "edges", lambda b: n_edges(b["data"]))):
        v = sorted(fn(b) for b in objs_by_mod[md].values())
        print(f"  {md:10s} {key:6s} objects={len(v)} min={v[0]} median={v[len(v)//2]} "
              f"max={v[-1]}")
    print(f"  tabular value_extraction row index: max={max(ve_rows)} (n={len(ve_rows)})")

    res = {"sizes": {k: dict(v) for k, v in sizes.items()},
           "value_extraction_max_row_index": max(ve_rows), "limits": {}}
    for L in LIMITS:
        md, f = L["mod"], L["fmt"]
        qs = [q for q, b in meta.items() if b["modality"] == md]
        # answerability with the limit removed
        old = getattr(ans, L["const"])
        setattr(ans, L["const"], INF)
        cat_nolim = {q: ans.CLASSIFY[md](meta[q]) for q in qs}
        setattr(ans, L["const"], old)
        exc = [q for q in qs if L["exceeds"](L["size"](meta[q]))]
        wit = [q for q in qs if not L["exceeds"](L["size"](meta[q]))]
        flips = [q for q in qs if cat[q][f] == "A" and cat_nolim[q][f] in "VD"]
        exc_ok = [q for q in exc if cat[q][f] in "VD"]
        other_changed = [q for q in qs for g in ans.FORMATS[md]
                         if g != f and cat[q][g] != cat_nolim[q][g]]
        tasks_flip = Counter(meta[q]["task"] for q in flips)
        n_obj_exc = len({q2d[q] for q in exc})
        r = {"label": L["label"], "modality": md, "format": f, "n_questions": len(qs),
             "n_exceeds": len(exc), "n_objects_exceeds": n_obj_exc,
             "n_objects": len({q2d[q] for q in qs}), "n_flips": len(flips),
             "flip_tasks": dict(tasks_flip), "n_exceeds_still_answerable": len(exc_ok),
             "n_other_format_class_changes": len(other_changed), "models": {}}
        if L["control"] == "global_tasks":
            ctrl_fmt, ctrl_q = f, lambda q: meta[q]["task"] in ans.G_GLOBAL_TASKS
            prim_q = lambda q: meta[q]["task"] in ans.G_NODE_TASKS  # noqa: E731
        else:
            ctrl_fmt, ctrl_q = L["control"], (lambda q: True)
            prim_q = lambda q: True  # noqa: E731
        r["control"] = (f"{f}, global tasks" if L["control"] == "global_tasks"
                        else L["control"])
        for m in rc.CORE4 + ["Qwen v2-v1"]:
            if m == "Qwen v2-v1":
                val = lambda q, g: EM2[(q, g)] - EM["Qwen2.5-VL-7B"][(q, g)]  # noqa: E731
            else:
                val = lambda q, g, m=m: EM[m][(q, g)]  # noqa: E731
            pe = [(q, val(q, f)) for q in exc if prim_q(q)]
            pw = [(q, val(q, f)) for q in wit if prim_q(q)]
            d, lo, hi = two_group_ci({"a": pe, "b": pw}, q2d)
            ce = [(q, val(q, ctrl_fmt)) for q in exc if ctrl_q(q)]
            cw = [(q, val(q, ctrl_fmt)) for q in wit if ctrl_q(q)]
            dc, clo, chi = two_group_ci({"a": ce, "b": cw}, q2d)
            fl = [(q, val(q, f)) for q in flips]
            ok = [(q, val(q, f)) for q in exc_ok]
            dfl, flo, fhi = two_group_ci({"a": fl, "b": ok}, q2d)
            r["models"][m] = {
                "em_exceeds": rc.mean100(v for _, v in pe), "n_exceeds": len(pe),
                "em_within": rc.mean100(v for _, v in pw), "n_within": len(pw),
                "diff": d, "diff_ci": [lo, hi],
                "ctrl_exceeds": rc.mean100(v for _, v in ce),
                "ctrl_within": rc.mean100(v for _, v in cw),
                "ctrl_diff": dc, "ctrl_diff_ci": [clo, chi], "did": d - dc,
                "em_flips": rc.mean100(v for _, v in fl), "n_flips": len(fl),
                "em_exceeds_answerable": rc.mean100(v for _, v in ok), "n_exceeds_answerable":
                len(ok), "diff_flips_vs_answerable": dfl, "diff_flips_ci": [flo, fhi]}
        res["limits"][L["id"]] = r
        print(f"\n[{L['id']}] {md}/{f}: exceeds {len(exc)}/{len(qs)} questions "
              f"({n_obj_exc} objects); flips V/D->A {len(flips)} {dict(tasks_flip)}; "
              f"exceeds-but-answerable {len(exc_ok)}; control={r['control']}")
        for m, v in r["models"].items():
            print(f"  {m:18s} EM exc {rc.f1(v['em_exceeds']):>5s} (n={v['n_exceeds']}) "
                  f"within {rc.f1(v['em_within']):>5s} (n={v['n_within']}) "
                  f"diff {rc.sgn(v['diff'])} [{rc.sgn(v['diff_ci'][0])},{rc.sgn(v['diff_ci'][1])}]"
                  f" | ctrl diff {rc.sgn(v['ctrl_diff'])} DiD {rc.sgn(v['did'])} | flips "
                  f"{rc.f1(v['em_flips'])} (n={v['n_flips']}) vs exc-answerable "
                  f"{rc.f1(v['em_exceeds_answerable'])} (n={v['n_exceeds_answerable']})")

    rc.dump(os.path.join(rc.HERE, "param_sensitivity_results.json"), res)

    hdr = ["Limit", "$n_q$", "Exceeds", "Flips", "Model", "EM$_{exc}$", "EM$_{within}$",
           "$\\Delta$ [CI]", "$\\Delta_{ctrl}$", "DiD", "EM$_{flip}$"]
    rows, mids = [], []
    for L in LIMITS:
        r = res["limits"][L["id"]]
        if rows:
            mids.append(len(rows))
        for i, m in enumerate(rc.CORE4 + ["Qwen v2-v1"]):
            v = r["models"][m]
            rows.append([rc.Raw(r["label"]) if i == 0 else "",
                         str(r["n_questions"]) if i == 0 else "",
                         str(r["n_exceeds"]) if i == 0 else "",
                         str(r["n_flips"]) if i == 0 else "", m,
                         rc.f1(v["em_exceeds"]) if m != "Qwen v2-v1" else rc.sgn(v["em_exceeds"]),
                         rc.f1(v["em_within"]) if m != "Qwen v2-v1" else rc.sgn(v["em_within"]),
                         f"{rc.sgn(v['diff'])} [{rc.sgn(v['diff_ci'][0])}, "
                         f"{rc.sgn(v['diff_ci'][1])}]",
                         rc.sgn(v["ctrl_diff"]), rc.sgn(v["did"]),
                         rc.f1(v["em_flips"]) if m != "Qwen v2-v1" else rc.sgn(v["em_flips"])])
    rc.write_tabular(
        os.path.join(rc.TABLES, "tab_param_limits.tex"),
        "v1 renderer legibility limits. n_q = questions of the modality; Exceeds = questions\n"
        "whose object exceeds the limit; Flips = questions whose answer becomes absent in that\n"
        "format because of the limit (answerability A with the limit, V/D without it). EM on the\n"
        "limited format for exceeding vs within-limit questions, Delta = difference (pp,\n"
        "object-cluster 95% CI); Delta_ctrl = the same split on a format drawing the whole\n"
        "object (tabular bar_chart, t.s. line_plot, graph node_link) or, for graph label/tick\n"
        "limits, on global tasks of the same format (Delta then uses node-referencing tasks);\n"
        "DiD = Delta - Delta_ctrl; EM_flip = EM on flipped questions. Qwen v2-v1 rows give the\n"
        "change from the answer-complete v2 renderer (pp).",
        "l r r r l r r r r r r", hdr, rows, midrules=mids)
    print("\nwrote scripts/analysis/param_sensitivity_results.json, "
          "paper_tables/tab_param_limits.tex")


if __name__ == "__main__":
    main()
