"""Human evaluation analysis for StructViz-Bench.

Computes human accuracy, visualization sensitivity, human-model performance
gaps, inter-annotator agreement, and generates comparison reports.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.evaluation.metrics import exact_match


@dataclass(slots=True)
class HumanAccuracyResult:
    """Human exact-match accuracy breakdown.

    Attributes:
        overall: Overall human EM accuracy.
        by_modality: EM accuracy per modality.
        by_viz_type: EM accuracy per visualization type.
        by_modality_viz: EM accuracy per (modality, viz_type) pair.
        total_annotations: Total number of annotations evaluated.
    """

    overall: float
    by_modality: dict[str, float]
    by_viz_type: dict[str, float]
    by_modality_viz: dict[tuple[str, str], float]
    total_annotations: int


@dataclass(slots=True)
class HumanVizSensitivityResult:
    """Human visualization sensitivity analysis.

    Attributes:
        overall_sensitivity: Mean std-dev of human EM across viz types per item.
        per_item: Mapping item_id -> {viz_type: accuracy}.
        flip_rate: Fraction of viz pairs where human correctness changes.
        by_modality: Mean sensitivity per modality.
    """

    overall_sensitivity: float
    per_item: dict[str, dict[str, float]]
    flip_rate: float
    by_modality: dict[str, float]


@dataclass(slots=True)
class HumanModelGapResult:
    """Gap between human and model performance.

    Attributes:
        overall_gap: Overall EM gap (human - model).
        human_overall: Human overall EM.
        model_overall: Model overall EM.
        by_modality: Per-modality gap dict with human, model, gap keys.
        by_viz_type: Per-viz_type gap dict with human, model, gap keys.
    """

    overall_gap: float
    human_overall: float
    model_overall: float
    by_modality: dict[str, dict[str, float]]
    by_viz_type: dict[str, dict[str, float]]


@dataclass(slots=True)
class AgreementResult:
    """Inter-annotator agreement statistics.

    Attributes:
        metric_name: Name of the agreement metric used (cohen_kappa or fleiss_kappa).
        overall_kappa: Overall agreement score.
        by_modality: Agreement per modality.
        num_annotators: Number of annotators detected.
        num_items: Number of unique (item_id, viz_type) pairs evaluated.
    """

    metric_name: str
    overall_kappa: float
    by_modality: dict[str, float]
    num_annotators: int
    num_items: int


@dataclass(slots=True)
class HumanEvalReport:
    """Full human evaluation comparison report.

    Attributes:
        accuracy: Human accuracy breakdown.
        sensitivity: Human visualization sensitivity.
        gap: Human-model performance gap.
        agreement: Inter-annotator agreement.
        summary: Prose summary of key findings.
    """

    accuracy: HumanAccuracyResult
    sensitivity: HumanVizSensitivityResult
    gap: HumanModelGapResult | None
    agreement: AgreementResult
    summary: str


class HumanEvalAnalyzer:
    """Analyze human evaluation annotations for StructViz-Bench.

    Expects annotations as a list of dicts with keys:
        item_id, question, answer (ground truth), human_answer, viz_type,
        modality, annotator_id, confidence, time_seconds.
    """

    @staticmethod
    def _em_score(human_answer: str, ground_truth: str) -> float:
        """Compute exact-match score between human answer and ground truth.

        Args:
            human_answer: The annotator's answer string.
            ground_truth: The ground-truth answer string.

        Returns:
            1.0 if exact match, 0.0 otherwise.
        """
        return exact_match(human_answer, ground_truth)

    @staticmethod
    def _population_std(values: list[float]) -> float:
        """Compute population standard deviation.

        Args:
            values: List of numeric values.

        Returns:
            Population standard deviation. Returns 0.0 for empty input.
        """
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance**0.5

    @staticmethod
    def _mean(values: list[float]) -> float:
        """Compute arithmetic mean, returning 0.0 for empty input.

        Args:
            values: List of numeric values.

        Returns:
            Arithmetic mean.
        """
        if not values:
            return 0.0
        return sum(values) / len(values)

    def compute_human_accuracy(
        self,
        annotations: list[dict[str, Any]],
    ) -> HumanAccuracyResult:
        """Compute human exact-match accuracy per modality and viz_type.

        Args:
            annotations: List of annotation dicts. Each must contain keys
                ``answer`` (ground truth), ``human_answer``, ``modality``,
                and ``viz_type``.

        Returns:
            HumanAccuracyResult with overall, per-modality, per-viz, and
            per-(modality, viz_type) breakdowns.
        """
        if not annotations:
            return HumanAccuracyResult(
                overall=0.0,
                by_modality={},
                by_viz_type={},
                by_modality_viz={},
                total_annotations=0,
            )

        scores: list[float] = []
        by_modality: dict[str, list[float]] = defaultdict(list)
        by_viz: dict[str, list[float]] = defaultdict(list)
        by_mod_viz: dict[tuple[str, str], list[float]] = defaultdict(list)

        for ann in annotations:
            em = self._em_score(str(ann["human_answer"]), str(ann["answer"]))
            scores.append(em)
            modality = str(ann["modality"])
            viz_type = str(ann["viz_type"])
            by_modality[modality].append(em)
            by_viz[viz_type].append(em)
            by_mod_viz[(modality, viz_type)].append(em)

        return HumanAccuracyResult(
            overall=self._mean(scores),
            by_modality={k: self._mean(v) for k, v in sorted(by_modality.items())},
            by_viz_type={k: self._mean(v) for k, v in sorted(by_viz.items())},
            by_modality_viz={k: self._mean(v) for k, v in sorted(by_mod_viz.items())},
            total_annotations=len(annotations),
        )

    def compute_human_viz_sensitivity(
        self,
        annotations: list[dict[str, Any]],
    ) -> HumanVizSensitivityResult:
        """Analyze whether humans show visualization sensitivity.

        Groups annotations by item_id, computes mean human EM per viz_type
        for each item, then measures the standard deviation across viz types.

        Args:
            annotations: List of annotation dicts.

        Returns:
            HumanVizSensitivityResult with per-item accuracy maps,
            overall sensitivity, flip rate, and per-modality sensitivity.
        """
        # Group by (item_id, viz_type) -> list of EM scores.
        item_viz_scores: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        item_modality: dict[str, str] = {}

        for ann in annotations:
            item_id = str(ann["item_id"])
            viz_type = str(ann["viz_type"])
            em = self._em_score(str(ann["human_answer"]), str(ann["answer"]))
            item_viz_scores[item_id][viz_type].append(em)
            item_modality[item_id] = str(ann["modality"])

        # Compute mean EM per (item, viz_type).
        per_item: dict[str, dict[str, float]] = {}
        for item_id, viz_dict in item_viz_scores.items():
            per_item[item_id] = {
                viz: self._mean(scores) for viz, scores in viz_dict.items()
            }

        # Sensitivity = std-dev of per-viz accuracies for each item.
        sensitivities: list[float] = []
        for item_id, viz_accs in per_item.items():
            values = list(viz_accs.values())
            if len(values) >= 2:
                sensitivities.append(self._population_std(values))

        # Flip rate: fraction of viz pairs where correctness changes per item.
        total_pairs = 0
        total_flips = 0
        for item_id, viz_accs in per_item.items():
            values = list(viz_accs.values())
            for i, j in combinations(range(len(values)), 2):
                total_pairs += 1
                if values[i] != values[j]:
                    total_flips += 1

        flip_rate = total_flips / total_pairs if total_pairs > 0 else 0.0

        # Per-modality sensitivity.
        modality_sensitivities: dict[str, list[float]] = defaultdict(list)
        for item_id, viz_accs in per_item.items():
            values = list(viz_accs.values())
            if len(values) >= 2:
                modality = item_modality.get(item_id, "unknown")
                modality_sensitivities[modality].append(self._population_std(values))

        return HumanVizSensitivityResult(
            overall_sensitivity=self._mean(sensitivities),
            per_item=per_item,
            flip_rate=flip_rate,
            by_modality={
                k: self._mean(v) for k, v in sorted(modality_sensitivities.items())
            },
        )

    def compute_human_model_gap(
        self,
        human_records: list[dict[str, Any]],
        model_records: list[dict[str, Any]],
    ) -> HumanModelGapResult:
        """Compute performance gap between human annotators and an MLLM.

        Both inputs should cover the same set of (item_id, viz_type) pairs.
        Human records use ``human_answer``; model records use ``prediction``
        and ``exact`` (or recomputed EM).

        Args:
            human_records: Human annotation dicts with keys ``item_id``,
                ``viz_type``, ``modality``, ``answer``, ``human_answer``.
            model_records: Model evaluation dicts with keys ``question_id``
                (or ``item_id``), ``viz_type``, ``modality``, ``answer``,
                ``prediction``, and optionally ``exact``.

        Returns:
            HumanModelGapResult with overall and per-slice gaps.
        """
        # Build human EM by (item_id, viz_type).
        human_em: dict[tuple[str, str], list[float]] = defaultdict(list)
        human_modality: dict[tuple[str, str], str] = {}
        for rec in human_records:
            key = (str(rec["item_id"]), str(rec["viz_type"]))
            em = self._em_score(str(rec["human_answer"]), str(rec["answer"]))
            human_em[key].append(em)
            human_modality[key] = str(rec["modality"])

        # Build model EM by (item_id, viz_type).
        model_em: dict[tuple[str, str], float] = {}
        model_modality: dict[tuple[str, str], str] = {}
        for rec in model_records:
            qid = str(rec.get("question_id", rec.get("item_id", "")))
            key = (qid, str(rec["viz_type"]))
            if "exact" in rec:
                model_em[key] = float(rec["exact"])
            else:
                model_em[key] = self._em_score(
                    str(rec.get("prediction", "")), str(rec["answer"])
                )
            model_modality[key] = str(rec["modality"])

        # Find common keys.
        common_keys = set(human_em.keys()) & set(model_em.keys())

        human_scores: list[float] = []
        model_scores: list[float] = []
        by_modality_human: dict[str, list[float]] = defaultdict(list)
        by_modality_model: dict[str, list[float]] = defaultdict(list)
        by_viz_human: dict[str, list[float]] = defaultdict(list)
        by_viz_model: dict[str, list[float]] = defaultdict(list)

        for key in common_keys:
            h_em = self._mean(human_em[key])
            m_em = model_em[key]
            human_scores.append(h_em)
            model_scores.append(m_em)

            modality = human_modality[key]
            viz_type = key[1]
            by_modality_human[modality].append(h_em)
            by_modality_model[modality].append(m_em)
            by_viz_human[viz_type].append(h_em)
            by_viz_model[viz_type].append(m_em)

        human_overall = self._mean(human_scores)
        model_overall = self._mean(model_scores)

        by_modality: dict[str, dict[str, float]] = {}
        for mod in sorted(set(by_modality_human.keys()) | set(by_modality_model.keys())):
            h = self._mean(by_modality_human.get(mod, []))
            m = self._mean(by_modality_model.get(mod, []))
            by_modality[mod] = {"human": h, "model": m, "gap": h - m}

        by_viz_type: dict[str, dict[str, float]] = {}
        for vt in sorted(set(by_viz_human.keys()) | set(by_viz_model.keys())):
            h = self._mean(by_viz_human.get(vt, []))
            m = self._mean(by_viz_model.get(vt, []))
            by_viz_type[vt] = {"human": h, "model": m, "gap": h - m}

        return HumanModelGapResult(
            overall_gap=human_overall - model_overall,
            human_overall=human_overall,
            model_overall=model_overall,
            by_modality=by_modality,
            by_viz_type=by_viz_type,
        )

    def compute_inter_annotator_agreement(
        self,
        annotations: list[dict[str, Any]],
    ) -> AgreementResult:
        """Compute inter-annotator agreement using Cohen's or Fleiss' kappa.

        For exactly 2 annotators, uses Cohen's kappa. For 3+ annotators,
        uses Fleiss' kappa. Agreement is measured on binary correctness
        (exact match = 1 or 0).

        Args:
            annotations: List of annotation dicts. Must include ``item_id``,
                ``viz_type``, ``annotator_id``, ``answer``, ``human_answer``.

        Returns:
            AgreementResult with kappa scores overall and per modality.
        """
        # Group annotations by (item_id, viz_type) -> list of (annotator_id, em).
        grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
        item_modality: dict[tuple[str, str], str] = {}

        for ann in annotations:
            key = (str(ann["item_id"]), str(ann["viz_type"]))
            annotator = str(ann["annotator_id"])
            em = self._em_score(str(ann["human_answer"]), str(ann["answer"]))
            grouped[key][annotator] = em
            item_modality[key] = str(ann["modality"])

        all_annotators = set()
        for annotator_map in grouped.values():
            all_annotators.update(annotator_map.keys())
        num_annotators = len(all_annotators)

        if num_annotators < 2:
            return AgreementResult(
                metric_name="n/a",
                overall_kappa=1.0,
                by_modality={},
                num_annotators=num_annotators,
                num_items=len(grouped),
            )

        if num_annotators == 2:
            kappa_overall = self._cohens_kappa(grouped)
            metric_name = "cohen_kappa"
        else:
            kappa_overall = self._fleiss_kappa(grouped, num_annotators)
            metric_name = "fleiss_kappa"

        # Per-modality agreement.
        modality_groups: dict[str, dict[tuple[str, str], dict[str, float]]] = defaultdict(
            dict
        )
        for key, annotator_map in grouped.items():
            mod = item_modality[key]
            modality_groups[mod][key] = annotator_map

        by_modality: dict[str, float] = {}
        for mod, mod_grouped in sorted(modality_groups.items()):
            if num_annotators == 2:
                by_modality[mod] = self._cohens_kappa(mod_grouped)
            else:
                by_modality[mod] = self._fleiss_kappa(mod_grouped, num_annotators)

        return AgreementResult(
            metric_name=metric_name,
            overall_kappa=kappa_overall,
            by_modality=by_modality,
            num_annotators=num_annotators,
            num_items=len(grouped),
        )

    @staticmethod
    def _cohens_kappa(
        grouped: dict[tuple[str, str], dict[str, float]],
    ) -> float:
        """Compute Cohen's kappa for two annotators on binary labels.

        Args:
            grouped: Mapping (item_id, viz_type) -> {annotator_id: em_score}.

        Returns:
            Cohen's kappa value.
        """
        annotators = set()
        for am in grouped.values():
            annotators.update(am.keys())
        annotator_list = sorted(annotators)
        if len(annotator_list) < 2:
            return 1.0

        a1, a2 = annotator_list[0], annotator_list[1]

        # Build contingency counts: (a1_label, a2_label).
        counts: dict[tuple[int, int], int] = defaultdict(int)
        n = 0
        for key, am in grouped.items():
            if a1 in am and a2 in am:
                label1 = int(am[a1] >= 0.5)
                label2 = int(am[a2] >= 0.5)
                counts[(label1, label2)] += 1
                n += 1

        if n == 0:
            return 0.0

        # Observed agreement.
        p_o = sum(counts[(c, c)] for c in [0, 1]) / n

        # Expected agreement.
        p_a1 = {c: sum(counts[(c, j)] for j in [0, 1]) / n for c in [0, 1]}
        p_a2 = {c: sum(counts[(i, c)] for i in [0, 1]) / n for c in [0, 1]}
        p_e = sum(p_a1[c] * p_a2[c] for c in [0, 1])

        if p_e >= 1.0:
            return 1.0
        return (p_o - p_e) / (1.0 - p_e)

    @staticmethod
    def _fleiss_kappa(
        grouped: dict[tuple[str, str], dict[str, float]],
        num_annotators: int,
    ) -> float:
        """Compute Fleiss' kappa for multiple annotators on binary labels.

        Args:
            grouped: Mapping (item_id, viz_type) -> {annotator_id: em_score}.
            num_annotators: Total number of annotators.

        Returns:
            Fleiss' kappa value.
        """
        # Categories: 0 (incorrect) and 1 (correct).
        n_items = 0
        # For each item, count how many annotators assigned each category.
        item_counts: list[tuple[int, int]] = []  # (n_incorrect, n_correct)

        for key, am in grouped.items():
            n_correct = sum(1 for v in am.values() if v >= 0.5)
            n_total = len(am)
            if n_total < 2:
                continue
            item_counts.append((n_total - n_correct, n_correct))
            n_items += 1

        if n_items == 0:
            return 0.0

        # Compute P_bar (mean per-item agreement).
        p_items: list[float] = []
        for n_inc, n_cor in item_counts:
            n = n_inc + n_cor
            if n <= 1:
                continue
            p_i = (n_inc * (n_inc - 1) + n_cor * (n_cor - 1)) / (n * (n - 1))
            p_items.append(p_i)

        if not p_items:
            return 0.0

        p_bar = sum(p_items) / len(p_items)

        # Compute P_e (expected agreement by chance).
        total_ratings = sum(n_inc + n_cor for n_inc, n_cor in item_counts)
        if total_ratings == 0:
            return 0.0
        total_correct = sum(n_cor for _, n_cor in item_counts)
        total_incorrect = sum(n_inc for n_inc, _ in item_counts)
        p_correct = total_correct / total_ratings
        p_incorrect = total_incorrect / total_ratings
        p_e = p_correct**2 + p_incorrect**2

        if p_e >= 1.0:
            return 1.0
        return (p_bar - p_e) / (1.0 - p_e)

    def generate_human_eval_report(
        self,
        annotations: list[dict[str, Any]],
        model_records: list[dict[str, Any]] | None = None,
    ) -> HumanEvalReport:
        """Generate a full human evaluation comparison report.

        Computes accuracy, visualization sensitivity, inter-annotator
        agreement, and optionally a human-model gap analysis.

        Args:
            annotations: Human annotation dicts with keys ``item_id``,
                ``question``, ``answer``, ``human_answer``, ``viz_type``,
                ``modality``, ``annotator_id``, ``confidence``, ``time_seconds``.
            model_records: Optional model evaluation dicts for gap analysis.
                Expected keys: ``question_id`` (or ``item_id``), ``viz_type``,
                ``modality``, ``answer``, ``prediction``, optionally ``exact``.

        Returns:
            HumanEvalReport with all analysis results and a text summary.
        """
        accuracy = self.compute_human_accuracy(annotations)
        sensitivity = self.compute_human_viz_sensitivity(annotations)
        agreement = self.compute_inter_annotator_agreement(annotations)

        gap: HumanModelGapResult | None = None
        if model_records:
            gap = self.compute_human_model_gap(annotations, model_records)

        # Build summary text.
        lines: list[str] = [
            "=== StructViz-Bench Human Evaluation Report ===",
            "",
            f"Total annotations: {accuracy.total_annotations}",
            f"Overall human EM: {accuracy.overall:.3f}",
            "",
            "Human EM by modality:",
        ]
        for mod, em in accuracy.by_modality.items():
            lines.append(f"  {mod}: {em:.3f}")

        lines.append("")
        lines.append("Human EM by visualization type:")
        for vt, em in accuracy.by_viz_type.items():
            lines.append(f"  {vt}: {em:.3f}")

        lines.append("")
        lines.append(
            f"Human viz sensitivity (mean std-dev): {sensitivity.overall_sensitivity:.4f}"
        )
        lines.append(f"Human viz flip rate: {sensitivity.flip_rate:.4f}")

        lines.append("")
        lines.append(
            f"Inter-annotator agreement ({agreement.metric_name}): "
            f"{agreement.overall_kappa:.3f}"
        )
        lines.append(f"  Annotators: {agreement.num_annotators}")
        lines.append(f"  Items evaluated: {agreement.num_items}")

        if gap is not None:
            lines.append("")
            lines.append(f"Human-model gap (overall): {gap.overall_gap:+.3f}")
            lines.append(f"  Human EM: {gap.human_overall:.3f}")
            lines.append(f"  Model EM: {gap.model_overall:.3f}")
            lines.append("")
            lines.append("Gap by modality:")
            for mod, vals in gap.by_modality.items():
                lines.append(
                    f"  {mod}: human={vals['human']:.3f}  "
                    f"model={vals['model']:.3f}  gap={vals['gap']:+.3f}"
                )

        summary = "\n".join(lines)

        return HumanEvalReport(
            accuracy=accuracy,
            sensitivity=sensitivity,
            gap=gap,
            agreement=agreement,
            summary=summary,
        )
