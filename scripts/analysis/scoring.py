#!/usr/bin/env python3
"""Canonical exact-match scorer for StructViz-Bench (revision primary metric).

``canonical_em(pred, raw, key, task, stored=None)`` returns 1.0 or 0.0. It is MONOTONE with
respect to the stored strict score: if ``stored`` (the released ``exact_match`` field) is 1 it
returns 1, and otherwise it can only add matches, never remove them. Rules, applied in order;
the first that matches returns 1:

(a) Existing normalisation: ``src.evaluation.metrics.exact_match(pred, key)`` (strip, lowercase,
    numeric canonical form: "55.130" == "55.13", "3.0" == "3").
        exact_match("Increasing", "increasing") -> 1
    NUMERIC KEYS (``float(key)`` parses) stop here: they keep the existing rule unchanged, so
    "48.36" vs "48.17", "10%" vs "10", or "-0.07" vs "0.07" stay wrong.

Non-numeric keys only:

(b) Index-prefix strip. The expressive bar chart / scatter plot / heatmap label rows as
    "<index>: <label>"; a leading ``^\\s*\\d+\\s*[:.]\\s*`` is removed from the prediction.
        pred "11: cloud", key "cloud" -> 1;  pred "19.10B", key "10B" -> 1
        pred "11: cloud", key "rain" -> 0
(c) Polarity / trend lexicon, keyed by the answer:
        key positive: {"positive", "positively", "positive correlation",
                       "positively correlated"};  key negative: the same with "negative"
        key increasing: {"increasing", "rising"};  key decreasing: {"decreasing", "falling"}
    Only applies when the key is one of those four words, so "rising" never matches a
    "higher"/"first half" key, and "increasing" is never accepted for a correlation key.
        pred "positively", key "positive" -> 1;  pred "falling", key "decreasing" -> 1
        pred "increasing", key "positive" -> 0
(d) Folding: lowercase, "_" -> " ", whitespace collapsed, and quotes / trailing sentence
    punctuation (. , ; : ! ? " ' `) stripped at the ends. Interior punctuation is kept, so
    distinct table entities do not merge.
        pred "second_half", key "second half" -> 1;  pred "w. & h", key "W. & H." -> 1
        pred "svm-w/o neg", key "SVM-w/o neg." -> 1;  pred "HDSA", key "HDSA (+)" -> 0
    (b) and (c) are evaluated on the folded, index-stripped prediction as well.
(e) Raw-response fallback for the parse_answer digit-stripping bug (``_extract_number`` keeps
    the last digit run for "numeric" tasks such as value_extraction): if ``raw`` is given, the
    parsed prediction is a pure digit run, and that run is a STRICT substring of an
    alphanumeric token of ``raw`` that equals the key after folding, it is counted correct.
        pred "10", raw "10B", key "10B" -> 1;  pred "08", raw "CoNLL08", key "CoNLL08" -> 1
        pred "10", raw "10", key "10B" -> 0 (no longer token);  pred "10", raw None -> 0
    Without a raw response (all released v1 files) this rule cannot fire.
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.evaluation.metrics import exact_match  # noqa: E402

INDEX_PREFIX = re.compile(r"^\s*\d+\s*[:.]\s*")
_EDGE = " \t\"'`.,;:!?"
LEXICON = {
    "positive": {"positive", "positively", "positive correlation", "positively correlated"},
    "negative": {"negative", "negatively", "negative correlation", "negatively correlated"},
    "increasing": {"increasing", "rising"},
    "decreasing": {"decreasing", "falling"},
}
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-/+]*")


def is_numeric_key(key: object) -> bool:
    """True when the key parses as a float (the existing numeric branch of exact_match)."""
    try:
        float(str(key).strip())
    except ValueError:
        return False
    return True


def fold(s: object) -> str:
    """Rule (d): lowercase, '_' -> ' ', collapse whitespace, strip edge quotes/punctuation."""
    t = str(s).strip().lower().replace("_", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip(_EDGE)


def strip_index(s: object) -> str:
    """Rule (b): drop a leading '<digits>:' or '<digits>.' tick-label prefix."""
    return INDEX_PREFIX.sub("", str(s))


def _surface_match(p: str, key: str) -> bool:
    fk = fold(key)
    if not fk:
        return False
    for cand in (p, strip_index(p)):
        fp = fold(cand)
        if fp == fk:                                     # (b) + (d)
            return True
        if fk in LEXICON and fp in LEXICON[fk]:          # (c)
            return True
    return False


def _raw_fallback(pred: str, raw: object, key: str) -> bool:
    """Rule (e)."""
    if raw is None:
        return False
    d = str(pred).strip()
    if not re.fullmatch(r"\d+", d):
        return False
    fk = fold(key)
    for tok in _TOKEN.findall(str(raw)):
        tok = tok.rstrip("._-/+")
        if d in tok and tok != d and fold(tok) == fk:
            return True
    return False


def canonical_em(pred: object, raw: object, key: object, task: str | None = None,
                 stored: float | None = None) -> float:
    """Canonical exact match (see module docstring). ``task`` is accepted for interface
    stability; no rule depends on it (rule (c) is keyed by the answer)."""
    del task
    if stored is not None and float(stored) >= 1:
        return 1.0
    p, k = "" if pred is None else str(pred), str(key)
    if exact_match(p, k) >= 1:                           # (a)
        return 1.0
    if is_numeric_key(k):
        return 0.0
    if _surface_match(p, k):
        return 1.0
    return 1.0 if _raw_fallback(p, raw, k) else 0.0


def row_score(r: dict, raw_field: str | None = "raw_response") -> float:
    """canonical_em for a result row; raw from ``raw_field`` when present ('None' -> None)."""
    raw = r.get(raw_field) if raw_field else None
    if raw in (None, "None", ""):
        raw = None
    return canonical_em(r.get("prediction", ""), raw, r["answer"], r.get("task"),
                        stored=float(r.get("exact_match", 0) or 0))


def object_fold_collisions(values) -> list[tuple[str, str]]:
    """Distinct non-numeric values of one object that fold to the same string (safety check:
    folding must not merge two different entities)."""
    seen: dict[str, str] = {}
    out = []
    for v in values:
        sv = str(v)
        if is_numeric_key(sv):
            continue
        f = fold(sv)
        if f in seen and seen[f] != sv:
            out.append((seen[f], sv))
        seen.setdefault(f, sv)
    return out
