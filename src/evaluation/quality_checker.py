from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import re
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from typing import Any


def _to_row(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if is_dataclass(item) and not isinstance(item, type):
        return asdict(item)
    raise TypeError(f"Unsupported benchmark item type: {type(item)!r}")


def _normalize_question(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text.strip().lower())
    return re.sub(r"[^a-z0-9 ]", "", collapsed)


def validate_answers(items: list[Any]) -> dict[str, Any]:
    """Check answer fields for emptiness and basic formatting quality."""
    invalid: list[dict[str, Any]] = []
    for item in items:
        row = _to_row(item)
        answer = str(row.get("answer", "")).strip()
        if not answer:
            invalid.append(
                {
                    "question_id": row.get("question_id", "unknown"),
                    "reason": "empty_answer",
                }
            )
            continue
        if len(answer) > 512:
            invalid.append(
                {
                    "question_id": row.get("question_id", "unknown"),
                    "reason": "answer_too_long",
                }
            )
    return {
        "total_items": len(items),
        "invalid_count": len(invalid),
        "invalid_items": invalid,
    }


def deduplicate(items: list[Any]) -> list[dict[str, Any]]:
    """Remove semantically duplicate questions via normalized text signatures."""
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        row = _to_row(item)
        signature = (
            str(row.get("data_id", "")),
            str(row.get("task", "")),
            _normalize_question(str(row.get("question", ""))),
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(row)
    return deduped


def check_consistency(items: list[Any]) -> dict[str, Any]:
    """Verify same data_id groups keep consistent metadata fields."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        row = _to_row(item)
        grouped[str(row.get("data_id", "unknown"))].append(row)

    inconsistencies: list[dict[str, Any]] = []
    for data_id, group in grouped.items():
        modalities = {str(row.get("modality", "")) for row in group}
        if len(modalities) > 1:
            inconsistencies.append(
                {
                    "data_id": data_id,
                    "reason": "modality_mismatch",
                    "values": sorted(modalities),
                }
            )
        tasks = {str(row.get("task", "")) for row in group}
        if not tasks:
            inconsistencies.append(
                {
                    "data_id": data_id,
                    "reason": "missing_task",
                }
            )
    return {
        "group_count": len(grouped),
        "inconsistency_count": len(inconsistencies),
        "inconsistencies": inconsistencies,
    }


def check_difficulty_distribution(
    items: list[Any],
    target_distribution: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compare actual difficulty percentages against configured targets."""
    target = target_distribution or {
        "1-hop": 0.30,
        "2-hop": 0.30,
        "3-hop": 0.25,
        "counterfactual": 0.15,
    }
    counts = Counter(str(_to_row(item).get("difficulty", "unknown")) for item in items)
    total = max(len(items), 1)
    rows: dict[str, dict[str, float]] = {}
    for difficulty, expected in target.items():
        actual = counts.get(difficulty, 0) / total
        rows[difficulty] = {
            "actual": actual,
            "target": expected,
            "delta": actual - expected,
        }
    return {
        "total_items": len(items),
        "distribution": rows,
    }


def generate_quality_report(items: list[Any]) -> dict[str, Any]:
    """Run complete quality checks and return structured report."""
    deduped = deduplicate(items)
    return {
        "original_count": len(items),
        "deduplicated_count": len(deduped),
        "duplicates_removed": len(items) - len(deduped),
        "answer_validation": validate_answers(deduped),
        "consistency": check_consistency(deduped),
        "difficulty_distribution": check_difficulty_distribution(deduped),
    }
