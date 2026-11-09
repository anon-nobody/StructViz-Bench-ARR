"""Quantitative failure mode classifier for StructViz-Bench evaluation.

Categorizes prediction errors into the three failure modes identified in
the paper, enabling fine-grained analysis of *why* models fail rather than
just *how often* they fail.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

# Visualization types that involve lossy transforms where exact numeric
# values are discarded during encoding.
_LOSSY_VIZ_TYPES: set[str] = {"gaf", "recurrence_plot"}

# Graph visualization types where the same underlying data can be rendered
# in different spatial layouts.
_GRAPH_VIZ_TYPES: set[str] = {"adjacency_matrix", "node_link", "circular_layout"}

# Multi-hop difficulty levels that require chained reasoning.
_MULTIHOP_DIFFICULTIES: set[str] = {"2-hop", "3-hop"}


class FailureClassifier:
    """Classify prediction errors into three failure modes.

    Failure modes:
        1. representation_projection_loss -- Model fails on lossy transforms
           (GAF, recurrence_plot) where the correct answer requires exact
           values that the transform discards.
        2. layout_misalignment -- Same graph/data yields different answers
           under different layouts (adjacency_matrix vs node_link).
        3. surface_overreliance -- Model extracts surface statistics (max/min)
           instead of performing multi-hop reasoning (2-hop or 3-hop questions).
        4. correct -- Prediction matches the ground-truth answer.
        5. other -- Errors that do not fit any of the above categories.
    """

    # ── public API ──────────────────────────────────────────────────────

    def classify_single(self, record: dict[str, Any]) -> str:
        """Classify one evaluation record into a failure mode.

        Args:
            record: Dictionary containing at least ``exact`` (or
                ``exact_match``), ``viz_type``, ``modality``, ``difficulty``,
                ``prediction``, and ``answer`` fields.

        Returns:
            One of ``"correct"``, ``"representation_projection_loss"``,
            ``"layout_misalignment"``, ``"surface_overreliance"``, or
            ``"other"``.
        """
        if self._is_correct(record):
            return "correct"

        viz_type = str(record.get("viz_type", "")).lower()
        modality = str(record.get("modality", "")).lower()
        difficulty = str(record.get("difficulty", "")).lower()

        # Mode 1 -- lossy visual encoding
        if viz_type in _LOSSY_VIZ_TYPES:
            return "representation_projection_loss"

        # Mode 2 -- graph layout sensitivity
        if modality == "graph" and viz_type in _GRAPH_VIZ_TYPES:
            return "layout_misalignment"

        # Mode 3 -- surface statistic shortcut on multi-hop questions
        if difficulty in _MULTIHOP_DIFFICULTIES and self._looks_like_surface_stat(record):
            return "surface_overreliance"

        return "other"

    def classify_batch(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Classify all records and add a ``failure_mode`` field.

        Args:
            records: List of evaluation record dicts.

        Returns:
            The same list with an added ``"failure_mode"`` key in each dict.
        """
        for record in records:
            record["failure_mode"] = self.classify_single(record)
        return records

    def compute_failure_distribution(
        self, records: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Compute the fraction of each failure mode across all errors.

        Args:
            records: List of evaluation record dicts (will be classified
                in-place if not already).

        Returns:
            Mapping from failure-mode name to its fraction among all
            *incorrect* predictions.  The ``"correct"`` key is excluded.
        """
        classified = self._ensure_classified(records)
        error_records = [r for r in classified if r["failure_mode"] != "correct"]
        if not error_records:
            return {}
        counts: dict[str, int] = defaultdict(int)
        for record in error_records:
            counts[record["failure_mode"]] += 1
        total = len(error_records)
        return {mode: count / total for mode, count in sorted(counts.items())}

    def compute_failure_by_model(
        self, records: list[dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
        """Per-model failure mode distribution.

        Args:
            records: List of evaluation record dicts.

        Returns:
            Nested mapping ``{model_name: {failure_mode: fraction}}``.
        """
        classified = self._ensure_classified(records)
        by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in classified:
            model = str(record.get("model", "unknown"))
            by_model[model].append(record)
        return {
            model: self.compute_failure_distribution(recs)
            for model, recs in sorted(by_model.items())
        }

    def compute_failure_by_modality(
        self, records: list[dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
        """Per-modality failure mode distribution.

        Args:
            records: List of evaluation record dicts.

        Returns:
            Nested mapping ``{modality: {failure_mode: fraction}}``.
        """
        classified = self._ensure_classified(records)
        by_modality: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in classified:
            modality = str(record.get("modality", "unknown"))
            by_modality[modality].append(record)
        return {
            mod: self.compute_failure_distribution(recs)
            for mod, recs in sorted(by_modality.items())
        }

    # ── private helpers ─────────────────────────────────────────────────

    @staticmethod
    def _is_correct(record: dict[str, Any]) -> bool:
        """Return True if the prediction is marked as correct."""
        # Support both naming conventions found in the codebase.
        exact = record.get("exact", record.get("exact_match"))
        if exact is not None:
            return float(exact) == 1.0
        return False

    @staticmethod
    def _looks_like_surface_stat(record: dict[str, Any]) -> bool:
        """Heuristic: check if the prediction looks like a surface statistic.

        Returns True when the prediction is a bare number that could plausibly
        be a max/min value read directly off the visualization, rather than a
        value obtained through multi-hop reasoning.
        """
        prediction = str(record.get("prediction", "")).strip()
        answer = str(record.get("answer", "")).strip()

        # If the prediction is not numeric, it is unlikely to be a simple
        # surface statistic.
        try:
            float(prediction)
        except ValueError:
            return False

        # If prediction equals answer it would be correct (handled earlier).
        # Here we only flag cases where the prediction is numeric but wrong,
        # which suggests the model grabbed an easy-to-read value instead of
        # reasoning through the chain.
        try:
            pred_val = float(prediction)
            ans_val = float(answer)
            # If the values are very different, the model likely grabbed a
            # surface-level number rather than performing multi-hop reasoning.
            if ans_val != 0.0 and abs(pred_val - ans_val) / abs(ans_val) > 0.05:
                return True
        except ValueError:
            pass

        return False

    def _ensure_classified(
        self, records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Classify records in-place if they lack a ``failure_mode`` field."""
        for record in records:
            if "failure_mode" not in record:
                record["failure_mode"] = self.classify_single(record)
        return records
