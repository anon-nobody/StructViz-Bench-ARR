from __future__ import annotations

import re
import string
from dataclasses import dataclass


# Pre-built translation table for stripping ASCII punctuation.
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(s: str) -> str:
    """SQuAD-style text normalization: lowercase, strip punctuation, collapse spaces."""
    s = s.lower().translate(_PUNCT_TABLE)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


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


def token_f1(prediction: str, answer: str) -> float:
    """SQuAD-style token-level F1.

    Applies SQuAD text normalization (lowercase + strip ASCII punctuation +
    collapse whitespace) before tokenizing on whitespace.  Duplicate tokens
    are counted via multiset intersection.
    """
    from collections import Counter

    pred_tokens = _normalize_text(prediction).split()
    ans_tokens = _normalize_text(answer).split()
    if not pred_tokens and not ans_tokens:
        return 1.0
    if not pred_tokens or not ans_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    ans_counter = Counter(ans_tokens)
    common_count = sum((pred_counter & ans_counter).values())
    if common_count == 0:
        return 0.0
    precision = common_count / len(pred_tokens)
    recall = common_count / len(ans_tokens)
    return 2 * precision * recall / (precision + recall)


def numerical_accuracy(
    prediction: str,
    answer: str,
    abs_tolerance: float = 0.05,
    rel_tolerance: float = 0.01,
) -> float:
    """Check numeric closeness with combined absolute + relative tolerance.

    Returns 1.0 if the prediction is within *either* the absolute tolerance
    or the symmetric relative tolerance of the answer.  The relative tolerance
    uses ``max(|answer|, |prediction|)`` as the denominator, ensuring that the
    same percent-error magnitude yields the same verdict regardless of which
    side is larger.

    Args:
        prediction: Model's predicted string.
        answer: Ground-truth answer string.
        abs_tolerance: Maximum absolute difference allowed (default 0.05).
        rel_tolerance: Maximum symmetric relative difference allowed
            (default 1%, i.e. 0.01).

    Returns:
        1.0 if within tolerance, 0.0 otherwise.
    """
    try:
        pred_val = float(prediction)
        ans_val = float(answer)
    except ValueError:
        return 0.0

    abs_diff = abs(pred_val - ans_val)

    # Absolute tolerance check.
    if abs_diff <= abs_tolerance:
        return 1.0

    # Symmetric relative tolerance: max(|ans|, |pred|) as denominator.
    denom = max(abs(ans_val), abs(pred_val))
    if denom > 0.0 and abs_diff / denom <= rel_tolerance:
        return 1.0

    return 0.0


@dataclass(slots=True)
class MetricBundle:
    """Container for all evaluation metrics."""

    exact: float
    f1: float
    numeric: float


def compute_metrics(
    prediction: str,
    answer: str,
    abs_tolerance: float = 0.05,
    rel_tolerance: float = 0.01,
    extract: bool = True,
) -> MetricBundle:
    """Compute all metrics for a single prediction-answer pair.

    Args:
        prediction: Model prediction string.
        answer: Ground-truth answer string.
        abs_tolerance: Absolute numeric tolerance.
        rel_tolerance: Relative numeric tolerance.
        extract: If True, apply answer extraction to verbose predictions
            before computing metrics. This improves EM for models that
            produce explanatory text around their final answer.

    Returns:
        MetricBundle with exact, f1, and numeric scores.
    """
    if extract:
        from src.evaluation.answer_extractor import extract_answer

        prediction = extract_answer(prediction)

    return MetricBundle(
        exact=exact_match(prediction, answer),
        f1=token_f1(prediction, answer),
        numeric=numerical_accuracy(
            prediction,
            answer,
            abs_tolerance=abs_tolerance,
            rel_tolerance=rel_tolerance,
        ),
    )
