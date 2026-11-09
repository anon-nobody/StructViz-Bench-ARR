"""Diversify template-based questions using rule-based paraphrasing.

Generates multiple surface-form variants of each benchmark question to
reduce sensitivity to exact wording.  All transformations are purely
rule-based (synonym dictionaries + regex patterns) -- no LLM API calls.
"""

from __future__ import annotations

import random
import re
from typing import Any


# ── Synonym / phrase substitution tables ────────────────────────────────────

_QUESTION_STARTERS: list[list[str]] = [
    ["What is", "Determine", "Find", "Calculate", "Identify"],
    ["What are", "List", "Enumerate", "Find all", "Identify all"],
    ["How many", "Count the number of", "What is the count of"],
    ["Which", "What", "Identify which"],
]

_PHRASE_SYNONYMS: dict[str, list[str]] = {
    "maximum": ["highest", "largest", "greatest", "peak"],
    "minimum": ["lowest", "smallest", "least"],
    "average": ["mean", "arithmetic mean"],
    "total": ["sum", "aggregate", "combined"],
    "value": ["amount", "quantity", "figure"],
    "between": ["from", "connecting"],
    "adjacent": ["neighboring", "directly connected"],
    "connected to": ["linked to", "joined to"],
    "path": ["route", "trail"],
    "shortest path": ["minimum-length path", "shortest route"],
    "node": ["vertex"],
    "edge": ["link", "connection"],
    "degree": ["number of connections"],
    "weight": ["cost", "distance"],
}

_FORMAT_TRANSFORMS: list[tuple[re.Pattern[str], str]] = [
    # "What is X?" -> "Report X."
    (re.compile(r"^What is (.+)\?$", re.IGNORECASE), r"Report \1."),
    # "What is X?" -> "State X."
    (re.compile(r"^What is (.+)\?$", re.IGNORECASE), r"State \1."),
    # "What is X?" -> "X is?"
    (re.compile(r"^What is (.+)\?$", re.IGNORECASE), r"\1 is?"),
    # "How many X?" -> "Count X."
    (re.compile(r"^How many (.+)\?$", re.IGNORECASE), r"Count \1."),
    # "Which X?" -> "Identify X."
    (re.compile(r"^Which (.+)\?$", re.IGNORECASE), r"Identify \1."),
]


class QuestionParaphraser:
    """Diversify template-based questions using rule-based paraphrasing.

    Applies systematic transformations:
        1. Synonym substitution ("What is" -> "Determine", "Find",
           "Calculate").
        2. Voice transformation ("What is the value of X?" ->
           "The value of X is?").
        3. Specificity variation ("What is the maximum?" ->
           "What is the highest value?").
        4. Question format variation ("What is X?" -> "Identify X." ->
           "Report X.").
    """

    def __init__(self, seed: int = 42) -> None:
        """Initialize with a random seed for reproducible paraphrasing.

        Args:
            seed: Random seed for variant selection.
        """
        self._rng = random.Random(seed)

    def paraphrase(self, question: str, n_variants: int = 3) -> list[str]:
        """Generate *n* rule-based paraphrases of a question.

        Args:
            question: Original question string.
            n_variants: Number of distinct paraphrases to produce.

        Returns:
            List of paraphrased question strings.  Duplicates and the
            original are filtered out, so the result length may be
            smaller than *n_variants*.
        """
        candidates: list[str] = []

        # Strategy 1: question-starter synonym substitution
        candidates.extend(self._apply_starter_synonyms(question))

        # Strategy 2: phrase-level synonym substitution
        candidates.extend(self._apply_phrase_synonyms(question))

        # Strategy 3: format transforms (question -> imperative, etc.)
        candidates.extend(self._apply_format_transforms(question))

        # Deduplicate and remove original
        seen: set[str] = {question.strip()}
        unique: list[str] = []
        for c in candidates:
            c_stripped = c.strip()
            if c_stripped and c_stripped not in seen:
                seen.add(c_stripped)
                unique.append(c_stripped)

        # Sample if we have more than requested
        if len(unique) > n_variants:
            unique = self._rng.sample(unique, n_variants)

        return unique

    def paraphrase_benchmark(
        self, items: list[dict[str, Any]], n_variants: int = 2
    ) -> list[dict[str, Any]]:
        """Add paraphrased variants to benchmark items.

        For each item, generates paraphrased versions and adds them as
        new items with the same metadata but a different ``question``
        field and a ``"paraphrase_of"`` back-reference.

        Args:
            items: List of benchmark item dicts, each containing at least
                a ``"question"`` field.
            n_variants: Number of paraphrase variants per item.

        Returns:
            Extended list including the originals plus paraphrased copies.
        """
        augmented: list[dict[str, Any]] = []
        for item in items:
            augmented.append(item)
            question = str(item.get("question", ""))
            if not question:
                continue
            variants = self.paraphrase(question, n_variants=n_variants)
            for idx, variant in enumerate(variants):
                new_item = dict(item)
                new_item["question"] = variant
                q_id = item.get("question_id", item.get("id", ""))
                new_item["question_id"] = f"{q_id}_para{idx}"
                new_item["paraphrase_of"] = q_id
                augmented.append(new_item)
        return augmented

    # ── private helpers ─────────────────────────────────────────────────

    def _apply_starter_synonyms(self, question: str) -> list[str]:
        """Replace question-starting phrases with synonyms."""
        results: list[str] = []
        q_lower = question.lstrip()
        for group in _QUESTION_STARTERS:
            for starter in group:
                if q_lower.lower().startswith(starter.lower()):
                    rest = q_lower[len(starter):]
                    for alt in group:
                        if alt.lower() != starter.lower():
                            results.append(alt + rest)
                    return results
        return results

    def _apply_phrase_synonyms(self, question: str) -> list[str]:
        """Replace domain-specific phrases with synonyms."""
        results: list[str] = []
        for phrase, synonyms in _PHRASE_SYNONYMS.items():
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            if pattern.search(question):
                chosen = self._rng.choice(synonyms)
                # Preserve the case of the first character
                match = pattern.search(question)
                if match:
                    original = match.group()
                    replacement = chosen
                    if original[0].isupper():
                        replacement = chosen[0].upper() + chosen[1:]
                    results.append(pattern.sub(replacement, question, count=1))
        return results

    @staticmethod
    def _apply_format_transforms(question: str) -> list[str]:
        """Apply regex-based question format transformations."""
        results: list[str] = []
        for pattern, replacement in _FORMAT_TRANSFORMS:
            match = pattern.match(question.strip())
            if match:
                transformed = pattern.sub(replacement, question.strip())
                results.append(transformed)
        return results
