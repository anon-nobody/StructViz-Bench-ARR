from __future__ import annotations

import re


_MARKDOWN_PATTERN = re.compile(r"`{1,3}([^`]*)`{1,3}")

# Task names that produce yes/no answers — expanded to cover all StructViz tasks.
_YES_NO_TASKS = frozenset(
    {
        "connectivity",
        "counterfactual",
        "counterfactual_half_shift",
        "counterfactual_sign_flip",
        "split_trend_consistency",
        "seasonality_detection",
    }
)

# Task names that produce numeric answers.
_NUMERIC_TASKS = frozenset(
    {
        "aggregation",
        "degree_query",
        "peak_identification",
        "range_query",
        "shortest_path",
        "threshold_count",
        "value_extraction",
        "value_lookup",
        "mean_shift_magnitude",
    }
)

# Task names that produce directional word answers.
_DIRECTION_TASKS = frozenset(
    {
        "trend_analysis",
        "endpoint_comparison",
        "volatility",
    }
)

# Task names that produce label/name answers.
_LABEL_TASKS = frozenset(
    {
        "ranking",
        "comparison",
        "outlier_detection",
        "pattern_label_lookup",
        "change_point",
    }
)

# Fallback keyword hints (for tasks not explicitly listed above).
_NUMERIC_TASK_HINTS = (
    "number",
    "count",
    "numeric",
    "how many",
    "value",
    "total",
)
_YES_NO_TASK_HINTS = (
    "yes/no",
    "true/false",
    "boolean",
)

# Preamble patterns to strip from verbose model responses.
_PREAMBLE_PATTERNS = [
    r"(?i)^based on (the )?(image|chart|graph|table|data|visualization|plot)[,\s.]*",
    r"(?i)^(from|looking at|according to) (the )?(image|chart|graph|table|data|visualization|plot)[,\s.]*",
    r"(?i)^the answer is[:\s]+",
    r"(?i)^answer[:\s]+",
    r"(?i)^final answer[:\s]+",
    r"(?i)^the (value|result|degree|number|count|shortest path( length)?|range|average|mean) (is|of|=)[:\s]*",
    r"(?i)^(it is|this is|that is|they are)[:\s]+",
    r"(?i)^(yes|no)[,.\s]+(the |it |they |this |based |because |since |as )",
    r"(?i)^the column ['\"]?",
]

# Patterns that indicate inability to answer.
_REFUSAL_PATTERNS = [
    r"(?i)^i('m| am) (unable|not able|cannot)",
    r"(?i)^i can('t|not) (determine|tell|read|see|identify)",
    r"(?i)^(sorry|unfortunately)",
    r"(?i)^it('s| is) (not possible|impossible|unclear|difficult) to (determine|tell|read)",
]


def _strip_response_text(raw_response: str) -> str:
    """Remove markdown formatting and normalize whitespace."""
    text = _MARKDOWN_PATTERN.sub(r"\1", raw_response)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" \t\"'`.,;:!?")
    return text


def _extract_answer_clause(cleaned: str) -> str:
    """Strip common preamble phrases to isolate the answer."""
    reduced = cleaned
    for pattern in _PREAMBLE_PATTERNS:
        reduced = re.sub(pattern, "", reduced).strip()
    return reduced.strip(" \t\"'`.,;:!?")


def _is_refusal(text: str) -> bool:
    """Check if model response is a refusal to answer."""
    return any(re.search(p, text) for p in _REFUSAL_PATTERNS)


def _task_likely_numeric(task: str) -> bool:
    if task.lower() in _NUMERIC_TASKS:
        return True
    lowered = task.lower()
    return any(hint in lowered for hint in _NUMERIC_TASK_HINTS)


def _task_likely_yes_no(task: str) -> bool:
    if task.lower() in _YES_NO_TASKS:
        return True
    lowered = task.lower()
    return any(hint in lowered for hint in _YES_NO_TASK_HINTS)


def _extract_yes_no(text: str) -> str | None:
    """Try to extract yes/no from text, checking start-of-text first."""
    lowered = text.lower().strip()
    # Direct match at start (handles "yes", "Yes, because...", etc.)
    if re.match(r"^yes\b", lowered):
        return "yes"
    if re.match(r"^no\b", lowered):
        return "no"
    # Word boundary search anywhere in text
    if re.search(r"\b(yes|true)\b", lowered):
        return "yes"
    if re.search(r"\b(no|false|not connected|not directly|cannot|wouldn't|would not)\b", lowered):
        return "no"
    # Numeric 0/1 for yes/no tasks (0=no, 1=yes is a common model pattern)
    stripped = lowered.strip()
    if stripped == "0":
        return "no"
    if stripped == "1":
        return "yes"
    return None


def _extract_number(text: str) -> str | None:
    """Extract a numeric value, preferring the last number in a sentence like 'degree of node 0 is 1'."""
    # If the text IS just a number, return it directly.
    stripped = text.strip()
    if re.fullmatch(r"[-+]?\d*\.?\d+", stripped):
        return stripped

    # Look for "is X" or "= X" patterns at the end.
    is_match = re.search(r"(?:is|=|:)\s*([-+]?\d*\.?\d+)\s*[.!]?\s*$", text)
    if is_match:
        return is_match.group(1)

    # Fall back to last number in the text.
    all_nums = re.findall(r"[-+]?\d*\.?\d+", text)
    if all_nums:
        return all_nums[-1]

    return None


def _extract_direction(text: str) -> str | None:
    """Extract directional/categorical answer from verbose text."""
    lowered = text.lower().strip()
    # Direct single-word/phrase match.
    direction_map = {
        "increasing": "increasing",
        "decreasing": "decreasing",
        "higher": "higher",
        "lower": "lower",
        "first half": "first half",
        "second half": "second half",
    }
    # Check if text IS the direction word.
    if lowered in direction_map:
        return direction_map[lowered]
    # Search for direction words in verbose text.
    for keyword, normalized in direction_map.items():
        if keyword in lowered:
            return normalized
    return None


def _extract_label(text: str) -> str:
    """Extract a label/name answer, stripping verbose wrapping."""
    # If wrapped in quotes, extract the quoted part.
    quoted = re.search(r"['\"]([^'\"]+)['\"]", text)
    if quoted:
        return quoted.group(1).strip()

    # Take the first meaningful token(s) — stop at common sentence connectors.
    reduced = re.split(
        r"\b(has|have|is|are|was|were|the|with|which|that)\b", text, maxsplit=1
    )
    if reduced:
        candidate = reduced[0].strip(" \t\"'`.,;:!?")
        if candidate and len(candidate) < 80:
            return candidate

    return text.strip(" \t\"'`.,;:!?")


def parse_answer(raw_response: str, task: str) -> str:
    """Parse model output into a normalized benchmark answer.

    Args:
        raw_response: Raw text generated by a model.
        task: Task name or task hint used for extraction strategy.

    Returns:
        Cleaned and normalized answer string.
    """
    cleaned = _strip_response_text(raw_response)
    reduced = _extract_answer_clause(cleaned)

    # Handle refusals — model can't answer.
    if _is_refusal(reduced):
        # For numeric tasks, try extracting a number anyway.
        num = _extract_number(reduced)
        if num is not None:
            return num
        return "[REFUSED]"

    # Yes/No tasks — check on cleaned text BEFORE preamble stripping (so "Yes, the..." is caught).
    if _task_likely_yes_no(task):
        yn = _extract_yes_no(cleaned)
        if yn is not None:
            return yn
        # Fallback: also try on preamble-stripped text.
        yn = _extract_yes_no(reduced)
        if yn is not None:
            return yn

    # Numeric tasks.
    if _task_likely_numeric(task):
        num = _extract_number(reduced)
        if num is not None:
            return num

    # Direction tasks.
    task_lower = task.lower()
    if task_lower in _DIRECTION_TASKS:
        direction = _extract_direction(reduced)
        if direction is not None:
            return direction

    # Label tasks — extract the core label.
    if task_lower in _LABEL_TASKS:
        # For "change_point" task, check for "none" type answers first.
        if task_lower == "change_point":
            if re.search(
                r"\b(none|no significant|no clear|no change)\b", reduced.lower()
            ):
                return "none"
            num = _extract_number(reduced)
            if num is not None:
                return num
        return _extract_label(reduced)

    # Generic fallback: if response is very short (<=3 words), use as-is.
    if len(reduced.split()) <= 3:
        return reduced.lower()

    # Try yes/no extraction even for unknown tasks (response-based detection).
    yn = _extract_yes_no(reduced)
    if yn is not None and len(reduced.split()) <= 6:
        return yn

    # Try numeric extraction for verbose responses.
    num = _extract_number(reduced)
    if num is not None and len(reduced.split()) > 5:
        return num

    # Try direction extraction.
    direction = _extract_direction(reduced)
    if direction is not None:
        return direction

    return reduced.strip(" \t\"'`.,;:!?").lower()
