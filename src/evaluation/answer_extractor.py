"""Extract concise final answers from verbose model predictions.

Many MLLMs (especially Claude) produce multi-paragraph explanations instead
of short answers.  This module applies a cascade of heuristics to pull out
the final answer token(s), dramatically improving EM for verbose responders.
"""

from __future__ import annotations

import re


# ── Ordered extraction patterns (first match wins) ──────────────────────────

_ANSWER_PATTERNS: list[re.Pattern[str]] = [
    # Explicit "the answer is X" / "Answer: X" — capture up to 2 short tokens
    # (~91% of benchmark answers are 1 token, ~7% are 2 tokens like dates or
    # short labels). Stopword filter is applied post-hoc to reject sentence
    # continuations like "42 tokens long" or "yes and the model".
    re.compile(
        r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)\s*['\"]?"
        r"([A-Za-z0-9][A-Za-z0-9._\-]*(?:\s+[A-Za-z0-9][A-Za-z0-9._\-:]*)?)",
        re.IGNORECASE,
    ),
    # "Therefore, X" / "Thus, X" / "So, X" / "Hence, X"
    re.compile(
        r"(?:therefore|thus|hence|so)\s*[,:]\s*['\"]?"
        r"([A-Za-z0-9][A-Za-z0-9._\-]*(?:\s+[A-Za-z0-9][A-Za-z0-9._\-:]*)?)",
        re.IGNORECASE,
    ),
    # "Result: X" / "Output: X"
    re.compile(
        r"(?:result|output)\s*[:=]\s*['\"]?"
        r"([A-Za-z0-9][A-Za-z0-9._\-]*(?:\s+[A-Za-z0-9][A-Za-z0-9._\-:]*)?)",
        re.IGNORECASE,
    ),
    # Boxed answer: \boxed{X}
    re.compile(r"\\boxed\{([^{}]+?)\}", re.IGNORECASE),
    # **bold answer** at the very end (constrain to short content)
    re.compile(r"\*\*([A-Za-z0-9][A-Za-z0-9._\-\s]{0,30}?)\*\*\s*\.?\s*$", re.MULTILINE),
]

# Stopwords that should never appear as the 2nd token of an extracted answer;
# their presence indicates a sentence continuation rather than a multi-word
# answer (e.g., "yes and the model", "42 tokens long", "negative trend in").
_STOPWORD_2ND = frozenset({
    "and", "or", "but", "with", "in", "on", "at", "by", "for", "of", "to",
    "from", "the", "a", "an", "is", "was", "were", "are", "be", "been",
    "tokens", "trend", "result", "value", "model", "based", "showing",
    "indicates", "indicating", "significantly", "approximately", "above",
    "below", "shown", "data", "table", "chart", "across", "throughout",
})


def _trim_answer_candidate(candidate: str) -> str:
    """Apply stopword-based truncation to a multi-token answer candidate.

    If the second token is a sentence-continuation stopword, return only the
    first token. Otherwise return the full candidate.
    """
    parts = candidate.strip().rstrip(".,!?;:").split()
    if len(parts) <= 1:
        return parts[0] if parts else ""
    if parts[1].lower() in _STOPWORD_2ND:
        return parts[0]
    return " ".join(parts[:2])

# ── Terminal answer patterns (single word/number after final sentence) ───────
# Matches "... some sentence. yes" or "... some sentence. 42.5" at end of text.
# Uses a strict single-token pattern (no embedded spaces) to avoid over-capture
# like "Therefore, yes and the model agrees" → "yes and the model agrees".
_TERMINAL_ANSWER = re.compile(
    r"[.!?]\s+([A-Za-z0-9][A-Za-z0-9._\-]*)\s*\.?\s*$",
)

# Common short answers that models append at the end of verbose text
_COMMON_ANSWERS = frozenset({
    "yes", "no", "none", "true", "false",
    "increasing", "decreasing", "stable", "positive", "negative",
})

# Numeric-only line (standalone number, possibly with sign/decimal)
_NUMERIC_LINE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$")

# Short final line (≤60 chars, no sentence-like structure)
_SHORT_FINAL = re.compile(r"^[^.!?]{1,60}$")


def extract_answer(prediction: str) -> str:
    """Extract the concise final answer from a (possibly verbose) prediction.

    The function applies a cascade of heuristics in priority order:
    1. Explicit answer markers ("the answer is …", "Answer: …")
    2. Conclusion markers ("Therefore, …", "Thus, …")
    3. Result markers ("Result: …", "Output: …")
    4. LaTeX boxed answers
    5. Bold markers at end
    6. Last line if it is purely numeric
    7. Last non-empty line if short (≤60 chars)
    8. Fallback: return stripped original

    Args:
        prediction: Raw model output string.

    Returns:
        Extracted answer string, stripped of surrounding whitespace.
    """
    text = prediction.strip()
    if not text:
        return text

    # Very short predictions are likely already concise answers.
    if len(text) <= 40:
        return text

    # Try each regex pattern in priority order.
    for pattern in _ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            # Apply stopword-aware trimming: if the captured candidate has a
            # 2nd token that is a sentence-continuation stopword (e.g., "and",
            # "tokens", "trend"), keep only the first token. Then strip
            # trailing sentence punctuation while preserving internal decimals.
            candidate = _trim_answer_candidate(match.group(1))
            if candidate:
                return candidate

    # Terminal answer: "... some explanation. no" or "... analysis. 42.5"
    # This catches Claude's pattern of appending a bare answer after a sentence.
    term_match = _TERMINAL_ANSWER.search(text)
    if term_match:
        candidate = term_match.group(1).strip().rstrip(".")
        # Accept if it's a known common answer or a number.
        if candidate.lower() in _COMMON_ANSWERS or _NUMERIC_LINE.match(candidate):
            return candidate

    # Try last non-empty line heuristics.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        last = lines[-1]

        # Pure numeric last line.
        num_match = _NUMERIC_LINE.match(last)
        if num_match:
            return num_match.group(1)

        # Short last line (likely the final answer).
        if _SHORT_FINAL.match(last):
            # Strip trailing period for EM matching.
            return last.rstrip(".")

    # Last resort: extract the final token after the last period in single-line text.
    if len(lines) == 1 and ". " in text:
        after_last_period = text.rsplit(". ", 1)[-1].strip().rstrip(".")
        if len(after_last_period) <= 40:
            return after_last_period

    # Fallback: return original stripped text.
    return text


def recompute_with_extraction(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Re-extract answers from existing result rows and recompute metrics.

    Mutates each row in-place by adding ``prediction_extracted`` and updating
    metric fields (``exact_match``, ``f1``, ``numeric_accuracy``).

    Args:
        rows: List of result dictionaries with ``prediction`` and ``answer`` keys.

    Returns:
        The same list, mutated in-place.
    """
    from src.evaluation.metrics import compute_metrics

    for row in rows:
        raw = str(row.get("prediction", ""))
        extracted = extract_answer(raw)
        row["prediction_extracted"] = extracted
        bundle = compute_metrics(prediction=extracted, answer=str(row.get("answer", "")))
        row["exact_match"] = bundle.exact
        row["f1"] = bundle.f1
        row["numeric_accuracy"] = bundle.numeric
    return rows
