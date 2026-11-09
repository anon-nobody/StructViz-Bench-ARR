#!/usr/bin/env python3
"""Shared helpers for the follow-up analyses.

Used by error_taxonomy.py, template_difficulty.py, cot_audit.py and param_sensitivity.py.
Stdlib only, no inference. Scoring always uses the `exact_match` field (never `exact`).
Clustered statistics reuse scripts/analysis/cluster_cis.py (cluster = data_id).
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "results")
BENCH = os.path.join(ROOT, "benchmark", "realworld_test.jsonl")
TABLES = os.path.join(ROOT, "paper_tables")
sys.path.insert(0, HERE)

import answerability as ans  # noqa: E402
import cluster_cis as cc  # noqa: E402
import v2_analysis as v2a  # noqa: E402

MODS = ["tabular", "timeseries", "graph"]
FORMATS = ans.FORMATS
MODELS7 = {
    "GPT-4o": "full_gpt4o_extracted.jsonl",
    "Gemini-2.0-Flash": "full_gemini_extracted.jsonl",
    "Qwen2.5-VL-7B": "full_qwen_extracted.jsonl",
    "Claude Sonnet 4": "full_claude_extracted.jsonl",
    "Gemini-2.5-Flash": "full_gemini25.jsonl",
    "Qwen2.5-VL-32B": "full_qwen32b.jsonl",
    "InternVL2.5-8B": "full_internvl.jsonl",
}
CORE4 = ["GPT-4o", "Gemini-2.0-Flash", "Qwen2.5-VL-7B", "Claude Sonnet 4"]
QWEN_V2 = "Qwen2.5-VL-7B (v2)"
PROMPT_KEYS = {"GPT-4o": "gpt4o", "Gemini-2.0-Flash": "gemini", "Qwen2.5-VL-7B": "qwen",
               "Claude Sonnet 4": "claude"}
STYLES = ["minimal", "concise", "detailed", "cot"]
DEPTHS = ["1-hop", "2-hop", "3-hop", "counterfactual"]
N_CV_SEEDS = 20


# ---------------------------------------------------------------- loading
def read_jsonl(path):
    return [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]


def load_bench():
    bench = read_jsonl(BENCH)
    return {b["question_id"]: b for b in bench}


def em_val(r):
    return float(r.get("exact_match", 0) or 0)


def load_v1(name):
    """{(qid, viz_type): row}; asserts one row per key (true for all seven released files)."""
    rows = read_jsonl(os.path.join(RES, MODELS7[name]))
    out = {}
    for r in rows:
        k = (r["question_id"], r["viz_type"])
        assert k not in out, (name, k)
        out[k] = r
    return out


def load_v2_qwen(meta):
    """Qwen v2 main-format predictions; last non-error row per key (v2_analysis.dedupe)."""
    rows = read_jsonl(os.path.join(RES, "v2", "v2_qwen.jsonl"))
    good, st = v2a.dedupe(rows)
    out = {k: r for k, r in good.items()
           if k[0] in meta and k[1] in FORMATS[meta[k[0]]["modality"]]}
    return out, st


# ---------------------------------------------------------------- answerability
def v1_cats(meta, strict=False):
    cat = {q: ans.CLASSIFY[r["modality"]](r) for q, r in meta.items()}
    if strict:
        cat = {q: v2a.strictify_v2(meta[q], c) for q, c in cat.items()}
    return cat


def v2_cats(meta, strict=False):
    """v2 answerability with the renderer constants recorded in the v2 manifest."""
    man = os.path.join(ROOT, "benchmark", "render_v2", "manifest.jsonl")
    cman = {}
    for r in read_jsonl(man):
        cman[(r["question_id"], r["viz_type"])] = r
    out = {}
    for q, r in meta.items():
        c = dict(v2a.V2_DEFAULTS)
        if r["modality"] == "tabular":
            rc = {}
            for f in ("bar_chart", "scatter_plot"):
                rc.update(cman.get((q, f), {}).get("renderer_constants") or {})
            for k in c:
                if k in rc:
                    c[k] = rc[k]
        cat = v2a.classify_v2(r, c)
        cat = {f: cat[f] for f in FORMATS[r["modality"]]}
        if strict:
            cat = v2a.strictify_v2(r, cat)
        out[q] = cat
    return out


def near_constant():
    d = json.load(open(os.path.join(HERE, "prior_and_subsets_results.json")))
    cells = {(c["modality"], c["task"]) for c in d["near_constant_tasks"]}
    assert len(cells) == 10 and d["near_constant_total_questions"] == 556
    return cells


# ---------------------------------------------------------------- answer typing
NUM_RE = re.compile(r"^[-+]?\$?\d[\d,]*(\.\d+)?%?$|^[-+]?\.\d+%?$")
LEX = {"positively": "positive", "negatively": "negative", "increases": "increasing",
       "rising": "increasing", "decreases": "decreasing", "falling": "decreasing",
       "increase": "increasing", "decrease": "decreasing", "upward": "increasing",
       "downward": "decreasing", "higher": "higher", "lower": "lower"}


def norm(s):
    return str(s).strip().lower()


def to_float(s):
    t = norm(s).replace(",", "").replace("$", "").rstrip("%")
    try:
        v = float(t)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def atype(s):
    """'yesno' | 'number' | 'label' for a key or an extracted prediction."""
    t = norm(s)
    if t in ("yes", "no"):
        return "yesno"
    if NUM_RE.match(t) and to_float(t) is not None:
        return "number"
    return "label"


def lenient(s):
    """Surface normalisation: lowercase, '_'/'-' -> space, strip edge punctuation, LEX map,
    numeric canonical form."""
    t = norm(s)
    v = to_float(t) if NUM_RE.match(t) else None
    if v is not None:
        return f"{v:g}"
    t = re.sub(r"[_]+", " ", t)
    t = re.sub(r"^[\s\(\[\"'`]+|[\s\.\,\;\:\!\?\)\]\"'`]+$", "", t)
    t = re.sub(r"\s+", " ", t)
    return LEX.get(t, t)


REFUSAL_TOKENS = {"none", "n/a", "na", "unknown", "[refused]", "[empty]", "[error]", "",
                  "cannot determine", "can't determine", "not determinable", "undetermined",
                  "i don't know", "not available", "not visible", "not shown", "not provided",
                  "unanswerable", "insufficient data", "insufficient information", "-",
                  "null", "nan", "?", "not applicable", "not possible", "no data"}
REFUSAL_RE = re.compile(r"\b(cannot|can't|can not|unable|not (possible|enough|visible|"
                        r"available|provided|shown|determinable|clear)|insufficient|"
                        r"does not (provide|show|contain)|doesn't (provide|show|contain)|"
                        r"no (information|data)|not specified|don't know)\b")
PREAMBLE_RE = re.compile(r"^(looking at|i can see|i need|i'll|i will|let me|let's|to (find|"
                         r"answer|determine|solve|identify|calculate)|based on|in this|in the|"
                         r"in|the image|the chart|the graph|this (image|chart|graph|is)|"
                         r"first,|step \d|from the|analy[sz]ing)\b")


def is_refusal(p, vocab):
    t = norm(p)
    if lenient(t) in vocab:
        return False
    return t in REFUSAL_TOKENS or bool(REFUSAL_RE.search(t))


def is_free_text(p):
    """Unparsed free text: a reasoning preamble fragment or >= 5 words."""
    t = norm(p)
    return bool(PREAMBLE_RE.match(t)) or len(t.split()) >= 5


def cell_stats(meta):
    """Per (modality, task): answer Counter (lenient), vocab set, majority answer, closed flag."""
    cnt = defaultdict(Counter)
    for b in meta.values():
        cnt[(b["modality"], b["task"])][lenient(b["answer"])] += 1
    out = {}
    for k, c in cnt.items():
        maj, mc = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        n = sum(c.values())
        nonnum = all(atype(a) != "number" for a in c)
        out[k] = {"counter": c, "vocab": set(c), "majority": maj, "maj_share": mc / n, "n": n,
                  "closed": nonnum and len(c) <= 10, "distinct": len(c),
                  "entropy": abs(-sum(v / n * math.log2(v / n) for v in c.values()))}
    return out


def numeric_near(p, a, rel=0.05):
    pv, av = to_float(p), to_float(a)
    if pv is None or av is None:
        return False
    if abs(pv - av) <= rel * max(abs(av), 1e-12):
        return True
    # rounding-equivalent: either value equals the other rounded to its own decimals
    def dec(s):
        t = norm(s).replace(",", "").rstrip("%")
        return len(t.split(".")[1]) if "." in t else 0
    return round(av, dec(p)) == pv or round(pv, dec(a)) == av


def right_type(pred, key, cs):
    """Does the extracted prediction have the type the key requires?

    In-vocabulary predictions (lenient match to any answer of the cell) are always the right
    type; refusals and free text never are; any substantive prediction is the right type
    when the key is the absence answer "none" (change_point); in closed-vocabulary label cells (<=10 distinct
    non-numeric answers) a label outside the vocabulary is the wrong type; otherwise
    atype(pred) must equal atype(key)."""
    lp = lenient(pred)
    if lp in cs["vocab"]:
        return True
    if is_refusal(pred, cs["vocab"]) or is_free_text(pred):
        return False
    if lenient(key) == "none":  # change_point "no change": any substantive answer is typed
        return True
    tk, tp = atype(key), atype(pred)
    if tk != tp:
        return False
    if tk == "label" and cs["closed"]:
        return False
    return True


# ---------------------------------------------------------------- question-only CV prior
def cv_prior_per_question(meta, seeds=N_CV_SEEDS):
    """prior_and_subsets.py 1(b): 2-fold split over data_id, majority answer per
    (modality, task) learned on the training fold; returns qid -> mean hit over seeds
    (each question is in exactly one test fold per seed). Unseen cells score 0."""
    bench = list(meta.values())
    objs = sorted({b["data_id"] for b in bench})
    hits = Counter()

    def majority(counter):
        return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    for s in range(seeds):
        r2 = random.Random(s)
        sh = objs[:]
        r2.shuffle(sh)
        half = len(sh) // 2
        folds = [set(sh[:half]), set(sh[half:])]
        for i in (0, 1):
            c = defaultdict(Counter)
            for b in bench:
                if b["data_id"] in folds[i]:
                    c[(b["modality"], b["task"])][norm(b["answer"])] += 1
            m = {k: majority(v) for k, v in c.items()}
            for b in bench:
                if b["data_id"] in folds[1 - i]:
                    pred = m.get((b["modality"], b["task"]))
                    hits[b["question_id"]] += int(pred is not None and norm(b["answer"]) == pred)
    return {q: hits[q] / seeds for q in meta}


# ---------------------------------------------------------------- statistics
def mean100(v):
    v = list(v)
    return 100 * sum(v) / len(v) if v else float("nan")


def ranks(x):
    idx = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[idx[j + 1]] == x[idx[i]]:
            j += 1
        for k in range(i, j + 1):
            r[idx[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else float("nan")


def spearman(a, b, B=10000):
    """Spearman rho with two-sided permutation p ((hits+1)/(B+1), random.Random(0))."""
    ra, rb = ranks(a), ranks(b)
    rho = pearson(ra, rb)
    rng = random.Random(0)
    rb2 = rb[:]
    hits = 0
    for _ in range(B):
        rng.shuffle(rb2)
        if abs(pearson(ra, rb2)) >= abs(rho) - 1e-12:
            hits += 1
    return rho, (hits + 1) / (B + 1)


def cluster_ci(items, q2d, perm=False):
    """items: [(qid, value)] -> cluster_cis.summarize over data_id (values in 0/1 or diffs)."""
    if not items:
        return None
    cl = cc.clusters_from(items, q2d)
    return cc.summarize(cl, perm=perm)


# ---------------------------------------------------------------- LaTeX
class Raw(str):
    """A table cell that write_tabular must not escape (already valid LaTeX)."""


def tex_esc(s):
    if isinstance(s, Raw):
        return str(s)
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"),
                 ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def write_tabular(path, caption, colspec, header, rows, midrules=()):
    """booktabs tabular in \\scriptsize; caption as a % comment; rows are lists of cells
    (already formatted; strings are escaped here). midrules: row indices preceded by \\midrule."""
    lines = []
    for ln in caption.splitlines():
        lines.append("% " + ln)
    lines.append("% Generated by scripts/analysis/" + os.path.basename(sys.argv[0])
                 + "; do not edit by hand.")
    lines.append(r"\begingroup\scriptsize")
    lines.append(r"\begin{tabular}{" + colspec + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    for i, r in enumerate(rows):
        if i in midrules:
            lines.append(r"\midrule")
        lines.append(" & ".join(tex_esc(c) for c in r) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\endgroup")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def f1(x):
    return "--" if x is None or x != x else f"{x:.1f}"


def sgn(x):
    return "--" if x is None or x != x else f"{x:+.1f}"


def dump(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else
                  float(o))
