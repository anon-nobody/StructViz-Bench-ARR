from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

from dataclasses import dataclass
from statistics import mean

import numpy as np

from src.generation import QAPair
from src.generation.data_sources.timeseries_sources import TimeSeriesDataset
from src.generation.difficulty_classifier import Difficulty, DifficultyClassifier


@dataclass(slots=True)
class TimeSeriesBenchmarkGenerator:
    """Generate QA items for time-series reasoning tasks."""

    classifier: DifficultyClassifier

    def __init__(self) -> None:
        """Initialize time-series generator with default classifier."""
        self.classifier = DifficultyClassifier()

    def generate_forecasting(self, ts: list[float]) -> list[QAPair]:
        """Generate forecasting questions for next-step estimation."""
        if len(ts) < 4:
            return []
        delta = ts[-1] - ts[-2]
        predicted = round(ts[-1] + delta, 3)
        difficulty = self.classifier.classify(
            reasoning_steps=2,
            requires_arithmetic=True,
            has_counterfactual=False,
        )
        return [
            QAPair(
                question="Based on the most recent change, what is the next expected value?",
                answer=str(predicted),
                difficulty=difficulty.value,
                data_id="timeseries-forecast",
                task="forecasting",
            )
        ]

    def generate_anomaly_detection(self, ts: list[float]) -> list[QAPair]:
        """Generate anomaly detection questions over deviations."""
        if len(ts) < 5:
            return []
        avg = mean(ts)
        deviations = [abs(v - avg) for v in ts]
        anomaly_idx = max(range(len(ts)), key=lambda i: deviations[i])
        difficulty = self.classifier.classify(
            reasoning_steps=2,
            requires_arithmetic=True,
            has_counterfactual=False,
        )
        return [
            QAPair(
                question="Which timestep appears most anomalous relative to the sequence average?",
                answer=str(anomaly_idx),
                difficulty=difficulty.value,
                data_id="timeseries-anomaly",
                task="anomaly_detection",
            )
        ]

    def generate_pattern_classification(self, ts: list[float]) -> list[QAPair]:
        """Generate trend/pattern classification questions."""
        if len(ts) < 4:
            return []
        slope = ts[-1] - ts[0]
        pattern = "upward" if slope > 0 else "downward"
        return [
            QAPair(
                question="Is the sequence trend upward or downward overall?",
                answer=pattern,
                difficulty=Difficulty.ONE_HOP.value,
                data_id="timeseries-pattern",
                task="pattern_classification",
            ),
            QAPair(
                question="If each point increased by 5, would the trend direction change?",
                answer="no",
                difficulty=Difficulty.COUNTERFACTUAL.value,
                data_id="timeseries-pattern",
                task="pattern_classification",
            ),
        ]

    def generate_seasonality_detection(
        self, dataset: TimeSeriesDataset
    ) -> list[QAPair]:
        """Generate seasonality detection questions for a dataset."""
        periodic_patterns = {"seasonal", "trend_seasonal", "multimodal"}
        has_seasonality = dataset.metadata.pattern_type in periodic_patterns
        return [
            QAPair(
                question="Does this series show periodic/seasonal behavior?",
                answer="yes" if has_seasonality else "no",
                difficulty=Difficulty.ONE_HOP.value,
                data_id=dataset.data_id,
                task="seasonality_detection",
            )
        ]

    def generate_peak_identification(self, dataset: TimeSeriesDataset) -> list[QAPair]:
        """Generate peak and trough identification questions for a dataset."""
        values = dataset.values
        max_index = int(np.argmax(values))
        min_index = int(np.argmin(values))
        return [
            QAPair(
                question="At which timestep does the maximum value occur?",
                answer=str(max_index),
                difficulty=Difficulty.ONE_HOP.value,
                data_id=dataset.data_id,
                task="peak_identification",
            ),
            QAPair(
                question="At which timestep does the minimum value occur?",
                answer=str(min_index),
                difficulty=Difficulty.ONE_HOP.value,
                data_id=dataset.data_id,
                task="peak_identification",
            ),
        ]

    def generate_range_query(self, dataset: TimeSeriesDataset) -> list[QAPair]:
        """Generate value range questions for a dataset."""
        values = dataset.values
        value_range = max(values) - min(values)
        return [
            QAPair(
                question="What is the range (max - min) of values?",
                answer=f"{value_range:.3f}",
                difficulty=Difficulty.TWO_HOP.value,
                data_id=dataset.data_id,
                task="range_query",
            )
        ]

    def generate_change_point(self, dataset: TimeSeriesDataset) -> list[QAPair]:
        """Generate change-point questions for a dataset."""
        known_points = dataset.metadata.known_anomaly_indices
        if known_points:
            answer = str(known_points[0])
        else:
            answer = "none"
        return [
            QAPair(
                question="At approximately which timestep does a significant change occur?",
                answer=answer,
                difficulty=Difficulty.ONE_HOP.value,
                data_id=dataset.data_id,
                task="change_point",
            )
        ]

    def generate_volatility(self, dataset: TimeSeriesDataset) -> list[QAPair]:
        """Generate volatility comparison questions for a dataset."""
        values = dataset.values
        split = len(values) // 2
        first_std = float(np.std(values[:split]))
        second_std = float(np.std(values[split:]))
        if abs(first_std - second_std) < 1e-9:
            answer = "equal"
        else:
            answer = "first half" if first_std > second_std else "second half"
        return [
            QAPair(
                question="Is this series more volatile in the first or second half?",
                answer=answer,
                difficulty=Difficulty.TWO_HOP.value,
                data_id=dataset.data_id,
                task="volatility",
            )
        ]

    def generate_dataset_questions(
        self,
        dataset: TimeSeriesDataset,
        dataset_index: int,
        target_items: int = 25,
    ) -> list[QAPair]:
        """Generate a fixed number of QA items for one synthetic TS dataset.

        Args:
            dataset: Time-series dataset with metadata.
            dataset_index: Position of the dataset in generation order.
            target_items: Number of QA items to emit.

        Returns:
            Generated QA pairs with controlled difficulty mix.
        """

        if target_items <= 0:
            return []

        one_hop_pool = self._build_one_hop_pool(dataset)
        two_hop_pool = self._build_two_hop_pool(dataset)
        three_hop_pool = self._build_three_hop_pool(dataset)
        counterfactual_pool = self._build_counterfactual_pool(dataset)
        one_count, two_count, three_count, counterfactual_count = self._difficulty_plan(
            dataset_index
        )

        selected: list[QAPair] = []
        selected.extend(one_hop_pool[:one_count])
        selected.extend(two_hop_pool[:two_count])
        selected.extend(three_hop_pool[:three_count])
        selected.extend(counterfactual_pool[:counterfactual_count])

        if len(selected) < target_items:
            fallback = (
                one_hop_pool + two_hop_pool + three_hop_pool + counterfactual_pool
            )
            needed = target_items - len(selected)
            selected.extend(fallback[:needed])

        return selected[:target_items]

    def generate_benchmark_items(
        self,
        datasets: list[TimeSeriesDataset],
        items_per_dataset: int = 25,
    ) -> list[QAPair]:
        """Generate QA items across multiple synthetic TS datasets.

        Args:
            datasets: Synthetic datasets from the TS data factory.
            items_per_dataset: Number of QA pairs per dataset.

        Returns:
            Full list of generated QA items.
        """

        pairs: list[QAPair] = []
        for index, dataset in enumerate(datasets):
            pairs.extend(
                self.generate_dataset_questions(
                    dataset=dataset,
                    dataset_index=index,
                    target_items=items_per_dataset,
                )
            )
        return pairs

    def _build_one_hop_pool(self, dataset: TimeSeriesDataset) -> list[QAPair]:
        values = dataset.values
        anomaly_count = len(dataset.metadata.known_anomaly_indices)
        trend = "upward" if values[-1] > values[0] else "downward"
        base: list[QAPair] = []
        base.extend(self.generate_seasonality_detection(dataset))
        base.extend(self.generate_peak_identification(dataset))
        base.extend(self.generate_change_point(dataset))
        base.extend(
            [
                QAPair(
                    question="What is the first value in the series?",
                    answer=f"{values[0]:.3f}",
                    difficulty=Difficulty.ONE_HOP.value,
                    data_id=dataset.data_id,
                    task="value_lookup",
                ),
                QAPair(
                    question="What is the last value in the series?",
                    answer=f"{values[-1]:.3f}",
                    difficulty=Difficulty.ONE_HOP.value,
                    data_id=dataset.data_id,
                    task="value_lookup",
                ),
                QAPair(
                    question="What is the pattern type label for this dataset?",
                    answer=dataset.metadata.pattern_type,
                    difficulty=Difficulty.ONE_HOP.value,
                    data_id=dataset.data_id,
                    task="pattern_label_lookup",
                ),
                QAPair(
                    question="Is the final value higher or lower than the first value?",
                    answer="higher" if values[-1] > values[0] else "lower",
                    difficulty=Difficulty.ONE_HOP.value,
                    data_id=dataset.data_id,
                    task="endpoint_comparison",
                ),
                QAPair(
                    question="How many known anomaly indices are annotated in metadata?",
                    answer=str(anomaly_count),
                    difficulty=Difficulty.ONE_HOP.value,
                    data_id=dataset.data_id,
                    task="metadata_count",
                ),
                QAPair(
                    question="Is the overall trend upward or downward?",
                    answer=trend,
                    difficulty=Difficulty.ONE_HOP.value,
                    data_id=dataset.data_id,
                    task="trend_direction",
                ),
            ]
        )
        return base

    def _build_two_hop_pool(self, dataset: TimeSeriesDataset) -> list[QAPair]:
        values = dataset.values
        series = np.asarray(values)
        split = len(values) // 2
        first_half = series[:split]
        second_half = series[split:]
        mean_delta = float(np.mean(second_half) - np.mean(first_half))
        median = float(np.median(series))
        threshold = float(np.mean(series) + np.std(series))
        threshold_count = int(np.sum(series > threshold))

        base: list[QAPair] = []
        base.extend(self.generate_range_query(dataset))
        base.extend(self.generate_volatility(dataset))
        base.extend(
            [
                QAPair(
                    question="Is the mean of the second half higher than the first half?",
                    answer="yes" if mean_delta > 0 else "no",
                    difficulty=Difficulty.TWO_HOP.value,
                    data_id=dataset.data_id,
                    task="mean_half_comparison",
                ),
                QAPair(
                    question="By how much does second-half mean exceed first-half mean?",
                    answer=f"{mean_delta:.3f}",
                    difficulty=Difficulty.TWO_HOP.value,
                    data_id=dataset.data_id,
                    task="mean_shift_magnitude",
                ),
                QAPair(
                    question="How many timesteps are above mean plus one standard deviation?",
                    answer=str(threshold_count),
                    difficulty=Difficulty.TWO_HOP.value,
                    data_id=dataset.data_id,
                    task="threshold_count",
                ),
                QAPair(
                    question="Is the median value above or below the overall mean?",
                    answer="above" if median > float(np.mean(series)) else "below",
                    difficulty=Difficulty.TWO_HOP.value,
                    data_id=dataset.data_id,
                    task="median_mean_relation",
                ),
                QAPair(
                    question="Is the average absolute change larger in first or second half?",
                    answer=self._volatility_by_abs_change(series),
                    difficulty=Difficulty.TWO_HOP.value,
                    data_id=dataset.data_id,
                    task="change_volatility",
                ),
                QAPair(
                    question="How many sign changes are in first-order differences?",
                    answer=str(self._count_diff_sign_changes(series)),
                    difficulty=Difficulty.TWO_HOP.value,
                    data_id=dataset.data_id,
                    task="oscillation_count",
                ),
            ]
        )
        return base

    def _build_three_hop_pool(self, dataset: TimeSeriesDataset) -> list[QAPair]:
        values = dataset.values
        split = len(values) // 2
        first_half = values[:split]
        second_half = values[split:]
        slope_first = first_half[-1] - first_half[0]
        slope_second = second_half[-1] - second_half[0]
        max_idx = int(np.argmax(values))
        min_idx = int(np.argmin(values))
        peak_after_trough = "yes" if max_idx > min_idx else "no"
        ratio = (max(values) - min(values)) / max(abs(np.mean(values)), 1e-8)

        base: list[QAPair] = []
        base.extend(
            self._retag_pairs(
                self.generate_forecasting(values),
                dataset.data_id,
                difficulty_override=Difficulty.THREE_HOP.value,
            )
        )
        base.extend(
            self._retag_pairs(
                self.generate_anomaly_detection(values),
                dataset.data_id,
                difficulty_override=Difficulty.THREE_HOP.value,
            )
        )
        base.extend(
            [
                QAPair(
                    question="Do both halves move in the same trend direction?",
                    answer="yes" if slope_first * slope_second >= 0 else "no",
                    difficulty=Difficulty.THREE_HOP.value,
                    data_id=dataset.data_id,
                    task="split_trend_consistency",
                ),
                QAPair(
                    question="Does the peak occur after the trough?",
                    answer=peak_after_trough,
                    difficulty=Difficulty.THREE_HOP.value,
                    data_id=dataset.data_id,
                    task="peak_trough_order",
                ),
                QAPair(
                    question="Is normalized amplitude (range/|mean|) greater than 1.0?",
                    answer="yes" if ratio > 1.0 else "no",
                    difficulty=Difficulty.THREE_HOP.value,
                    data_id=dataset.data_id,
                    task="normalized_amplitude",
                ),
                QAPair(
                    question="Which half has larger mean absolute first difference?",
                    answer=self._half_with_larger_abs_diff(np.asarray(values)),
                    difficulty=Difficulty.THREE_HOP.value,
                    data_id=dataset.data_id,
                    task="half_change_intensity",
                ),
                QAPair(
                    question="Is the median of local maxima greater than overall median?",
                    answer=self._local_maxima_median_relation(np.asarray(values)),
                    difficulty=Difficulty.THREE_HOP.value,
                    data_id=dataset.data_id,
                    task="local_peak_median_relation",
                ),
            ]
        )
        return base

    def _build_counterfactual_pool(self, dataset: TimeSeriesDataset) -> list[QAPair]:
        values = np.asarray(dataset.values)
        scaled = values * 1.1
        scaled_trend = "upward" if scaled[-1] > scaled[0] else "downward"
        base: list[QAPair] = []
        pattern_pairs = self.generate_pattern_classification(dataset.values)
        base.extend(
            self._retag_pairs(
                [
                    pair
                    for pair in pattern_pairs
                    if pair.difficulty == Difficulty.COUNTERFACTUAL.value
                ],
                dataset.data_id,
            )
        )
        base.extend(
            [
                QAPair(
                    question="If every value increased by 10%, would max index change?",
                    answer="no",
                    difficulty=Difficulty.COUNTERFACTUAL.value,
                    data_id=dataset.data_id,
                    task="counterfactual_scale_peak",
                ),
                QAPair(
                    question="If we add 5 only to second half, would second-half mean exceed first-half mean?",
                    answer=self._counterfactual_half_mean_answer(values),
                    difficulty=Difficulty.COUNTERFACTUAL.value,
                    data_id=dataset.data_id,
                    task="counterfactual_half_shift",
                ),
                QAPair(
                    question="If the series were multiplied by -1, would max and min indices swap?",
                    answer="yes",
                    difficulty=Difficulty.COUNTERFACTUAL.value,
                    data_id=dataset.data_id,
                    task="counterfactual_sign_flip",
                ),
                QAPair(
                    question="If each point had +2 added, would trend direction change?",
                    answer="no",
                    difficulty=Difficulty.COUNTERFACTUAL.value,
                    data_id=dataset.data_id,
                    task="counterfactual_translation",
                ),
                QAPair(
                    question="After scaling by 10%, would the trend be upward or downward?",
                    answer=scaled_trend,
                    difficulty=Difficulty.COUNTERFACTUAL.value,
                    data_id=dataset.data_id,
                    task="counterfactual_scaled_trend",
                ),
            ]
        )
        return base

    def _difficulty_plan(self, dataset_index: int) -> tuple[int, int, int, int]:
        mod_index = dataset_index % 20
        if mod_index < 5:
            return (8, 8, 6, 3)
        if mod_index < 10:
            return (7, 7, 7, 4)
        if mod_index < 15:
            return (7, 8, 6, 4)
        return (8, 7, 6, 4)

    def _retag_pairs(
        self,
        pairs: list[QAPair],
        data_id: str,
        difficulty_override: str | None = None,
    ) -> list[QAPair]:
        return [
            QAPair(
                question=pair.question,
                answer=pair.answer,
                difficulty=difficulty_override or pair.difficulty,
                data_id=data_id,
                task=pair.task,
            )
            for pair in pairs
        ]

    def _volatility_by_abs_change(self, series: np.ndarray) -> str:
        split = len(series) // 2
        first_changes = np.abs(np.diff(series[:split]))
        second_changes = np.abs(np.diff(series[split:]))
        first_score = float(np.mean(first_changes)) if len(first_changes) else 0.0
        second_score = float(np.mean(second_changes)) if len(second_changes) else 0.0
        return "first half" if first_score >= second_score else "second half"

    def _count_diff_sign_changes(self, series: np.ndarray) -> int:
        diffs = np.diff(series)
        if len(diffs) < 2:
            return 0
        signs = np.sign(diffs)
        return int(np.sum(signs[1:] * signs[:-1] < 0))

    def _half_with_larger_abs_diff(self, series: np.ndarray) -> str:
        split = len(series) // 2
        first = np.abs(np.diff(series[:split]))
        second = np.abs(np.diff(series[split:]))
        first_mean = float(np.mean(first)) if len(first) else 0.0
        second_mean = float(np.mean(second)) if len(second) else 0.0
        return "first half" if first_mean >= second_mean else "second half"

    def _local_maxima_median_relation(self, series: np.ndarray) -> str:
        if len(series) < 3:
            return "no"
        maxima_indices = [
            index
            for index in range(1, len(series) - 1)
            if series[index] > series[index - 1] and series[index] > series[index + 1]
        ]
        if not maxima_indices:
            return "no"
        maxima_median = float(np.median(series[maxima_indices]))
        series_median = float(np.median(series))
        return "yes" if maxima_median > series_median else "no"

    def _counterfactual_half_mean_answer(self, values: np.ndarray) -> str:
        split = len(values) // 2
        first_half = values[:split]
        second_half = values[split:] + 5.0
        return (
            "yes" if float(np.mean(second_half)) > float(np.mean(first_half)) else "no"
        )
