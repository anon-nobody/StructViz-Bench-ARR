from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from src.evaluation.metrics import MetricBundle, compute_metrics
from src.evaluation.viz_sensitivity import VizSensitivityAnalyzer
from src.utils.io_utils import read_jsonl, write_jsonl


@dataclass(slots=True)
class EvaluationRecord:
    """One evaluated prediction with metrics for a rendered benchmark item.

    Attributes:
        model: Model name.
        question_id: Unique question identifier.
        question: Question text.
        answer: Ground-truth answer.
        prediction: Model prediction.
        viz_type: Visualization type used for rendering.
        modality: Data modality.
        metrics: Computed metric bundle.
    """

    model: str
    question_id: str
    question: str
    answer: str
    prediction: str
    viz_type: str
    modality: str
    metrics: MetricBundle


class Evaluator:
    """Run model evaluation loops and aggregate benchmark metrics."""

    def _resolve_image(self, item: dict[str, Any]) -> Image.Image:
        if "image" in item and isinstance(item["image"], Image.Image):
            return item["image"]
        image_path_raw = item.get("image_path")
        if not image_path_raw:
            raise ValueError("Rendered item must include 'image' or 'image_path'.")
        image_path = Path(str(image_path_raw))
        return Image.open(image_path).convert("RGB")

    @staticmethod
    def _record_to_row(record: EvaluationRecord) -> dict[str, Any]:
        return {
            "model": record.model,
            "question_id": record.question_id,
            "question": record.question,
            "answer": record.answer,
            "prediction": record.prediction,
            "viz_type": record.viz_type,
            "modality": record.modality,
            "exact": record.metrics.exact,
            "f1": record.metrics.f1,
            "numeric": record.metrics.numeric,
        }

    @staticmethod
    def _record_from_row(row: dict[str, Any]) -> EvaluationRecord:
        return EvaluationRecord(
            model=str(row.get("model", "")),
            question_id=str(row.get("question_id", "")),
            question=str(row.get("question", "")),
            answer=str(row.get("answer", "")),
            prediction=str(row.get("prediction", "")),
            viz_type=str(row.get("viz_type", "")),
            modality=str(row.get("modality", "")),
            metrics=MetricBundle(
                exact=float(row.get("exact", 0.0)),
                f1=float(row.get("f1", 0.0)),
                numeric=float(row.get("numeric", 0.0)),
            ),
        )

    def evaluate_model(
        self,
        model: Any,
        rendered_items: list[dict[str, Any]],
        show_progress: bool = True,
    ) -> list[EvaluationRecord]:
        """Evaluate one model on rendered items.

        Args:
            model: Model wrapper implementing answer().
            rendered_items: Rendered benchmark rows.
            show_progress: Whether to show tqdm progress bar.

        Returns:
            Per-item evaluation records.
        """
        return self.evaluate_batch(
            model=model,
            items=rendered_items,
            batch_size=1,
            show_progress=show_progress,
        )

    def evaluate_model_batch(
        self,
        model: Any,
        rendered_items: list[dict[str, Any]],
        batch_size: int = 1,
        show_progress: bool = True,
    ) -> list[EvaluationRecord]:
        """Evaluate one model using batched inference where supported.

        Args:
            model: Model wrapper implementing answer()/answer_batch().
            rendered_items: Rendered benchmark rows.
            batch_size: Inference batch size.
            show_progress: Whether to show tqdm progress bar.

        Returns:
            Per-item evaluation records.
        """
        return self.evaluate_batch(
            model=model,
            items=rendered_items,
            batch_size=batch_size,
            show_progress=show_progress,
        )

    def evaluate_batch(
        self,
        model: Any,
        items: list[dict[str, Any]],
        batch_size: int = 1,
        show_progress: bool = True,
    ) -> list[EvaluationRecord]:
        """Evaluate rendered items with progress tracking and optional batching.

        Args:
            model: Model wrapper implementing answer()/answer_batch().
            items: Rendered benchmark rows.
            batch_size: Number of items per batch.
            show_progress: Whether to display tqdm progress.

        Returns:
            Per-item evaluation records.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be >= 1")

        records: list[EvaluationRecord] = []
        starts = list(range(0, len(items), batch_size))
        iterator: Any = starts
        if show_progress:
            iterator = tqdm(starts, desc=f"Evaluating {model.name}", leave=False)

        for start in iterator:
            chunk = items[start : start + batch_size]
            images = [self._resolve_image(item) for item in chunk]

            if len(chunk) == 1:
                prediction_list = [
                    model.answer(
                        question=str(chunk[0]["question"]),
                        image=images[0],
                        metadata=chunk[0],
                    )
                ]
            else:
                prediction_list = model.answer_batch(
                    questions=[str(item["question"]) for item in chunk],
                    images=images,
                    metadata_list=chunk,
                )

            for item, prediction in zip(chunk, prediction_list):
                records.append(
                    EvaluationRecord(
                        model=str(model.name),
                        question_id=str(item["question_id"]),
                        question=str(item["question"]),
                        answer=str(item["answer"]),
                        prediction=str(prediction),
                        viz_type=str(item.get("viz_type", "unknown")),
                        modality=str(item.get("modality", "unknown")),
                        metrics=compute_metrics(
                            prediction=str(prediction),
                            answer=str(item["answer"]),
                        ),
                    )
                )
        return records

    def summarize(self, records: list[EvaluationRecord]) -> dict[str, float]:
        """Compute overall benchmark metrics.

        Args:
            records: Evaluation records.

        Returns:
            Overall exact/f1/numeric means.
        """
        if not records:
            return {"overall_exact": 0.0, "overall_f1": 0.0, "overall_numeric": 0.0}
        total = len(records)
        return {
            "overall_exact": sum(record.metrics.exact for record in records) / total,
            "overall_f1": sum(record.metrics.f1 for record in records) / total,
            "overall_numeric": sum(record.metrics.numeric for record in records)
            / total,
        }

    def summarize_by_modality(
        self, records: list[EvaluationRecord]
    ) -> dict[str, dict[str, float]]:
        """Compute per-modality metric breakdown.

        Args:
            records: Evaluation records.

        Returns:
            Mapping modality -> metric means.
        """
        grouped: dict[str, list[EvaluationRecord]] = defaultdict(list)
        for record in records:
            grouped[record.modality].append(record)
        return {
            modality: {
                "exact": sum(item.metrics.exact for item in group) / len(group),
                "f1": sum(item.metrics.f1 for item in group) / len(group),
                "numeric": sum(item.metrics.numeric for item in group) / len(group),
                "count": float(len(group)),
            }
            for modality, group in grouped.items()
            if group
        }

    def summarize_by_difficulty(
        self,
        records: list[EvaluationRecord],
    ) -> dict[str, dict[str, float]]:
        """Compute per-difficulty metric breakdown.

        Args:
            records: Evaluation records with difficulty inside question text payload.

        Returns:
            Mapping difficulty -> metric means.
        """
        grouped: dict[str, list[EvaluationRecord]] = defaultdict(list)
        for record in records:
            difficulty = "unknown"
            if "::difficulty=" in record.question_id:
                difficulty = record.question_id.split("::difficulty=")[-1]
            grouped[difficulty].append(record)

        return {
            difficulty: {
                "exact": sum(item.metrics.exact for item in group) / len(group),
                "f1": sum(item.metrics.f1 for item in group) / len(group),
                "numeric": sum(item.metrics.numeric for item in group) / len(group),
                "count": float(len(group)),
            }
            for difficulty, group in grouped.items()
            if group
        }

    def summarize_viz_sensitivity(
        self, records: list[EvaluationRecord]
    ) -> dict[str, float]:
        """Aggregate visualization sensitivity across grouped question variants.

        Args:
            records: Evaluation records.

        Returns:
            Dictionary with average sensitivity.
        """
        grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
        for record in records:
            grouped[(record.question_id, record.model)][record.viz_type] = (
                record.metrics.exact
            )
        analyzed = VizSensitivityAnalyzer().analyze_group(grouped)
        if not analyzed:
            return {"avg_sensitivity": 0.0}
        return {
            "avg_sensitivity": sum(row.sensitivity_score for row in analyzed.values())
            / len(analyzed)
        }

    def export_records(self, records: list[EvaluationRecord], path: Path) -> None:
        """Write evaluation records to JSONL.

        Args:
            records: Evaluation records.
            path: Output JSONL path.
        """
        write_jsonl(path, [self._record_to_row(record) for record in records])

    @staticmethod
    def load_records(path: Path, as_rows: bool = False) -> list[Any]:
        """Load evaluation records from JSONL.

        Args:
            path: Input JSONL path.
            as_rows: Whether to return raw dict rows.

        Returns:
            Loaded EvaluationRecord list or raw rows.
        """
        rows = read_jsonl(path)
        if as_rows:
            return rows
        return [Evaluator._record_from_row(row) for row in rows]

    def serialize_records(
        self, records: list[EvaluationRecord]
    ) -> list[dict[str, Any]]:
        """Serialize records into JSONL-safe dictionaries.

        Args:
            records: Evaluation records.

        Returns:
            JSON-serializable row list.
        """
        return [self._record_to_row(record) for record in records]

    def save_records(self, path: Path, records: list[EvaluationRecord]) -> None:
        """Persist evaluation records as JSONL.

        Args:
            path: Output JSONL path.
            records: Evaluation records.
        """
        self.export_records(records, path)


def export_records(records: list[EvaluationRecord], path: Path) -> None:
    """Write evaluation records to JSONL.

    Args:
        records: Evaluation records.
        path: Output JSONL path.
    """
    Evaluator().export_records(records, path)


def load_records(path: Path) -> list[EvaluationRecord]:
    """Load evaluation records from JSONL.

    Args:
        path: Input JSONL path.

    Returns:
        Loaded evaluation records.
    """
    return Evaluator.load_records(path)


def build_rendered_item(
    question_id: str,
    question: str,
    answer: str,
    viz_type: str,
    modality: str,
    image: Image.Image,
) -> dict[str, Any]:
    """Build one in-memory rendered item dictionary.

    Args:
        question_id: Question ID.
        question: Question text.
        answer: Ground-truth answer.
        viz_type: Visualization type.
        modality: Data modality.
        image: Rendered image.

    Returns:
        Rendered-item dict compatible with evaluator.
    """
    return {
        "question_id": question_id,
        "question": question,
        "answer": answer,
        "viz_type": viz_type,
        "modality": modality,
        "image": image,
    }
