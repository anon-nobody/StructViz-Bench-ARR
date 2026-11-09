from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

from dataclasses import dataclass, replace
from statistics import median

import numpy as np
import pandas as pd

from src.generation import QAPair
from src.generation.data_sources.tabular_sources import (
    TabularDataFactory,
    TabularDataset,
)
from src.generation.difficulty_classifier import Difficulty, DifficultyClassifier


@dataclass(slots=True)
class TabularBenchmarkGenerator:
    """Generate QA items for tabular reasoning tasks."""

    classifier: DifficultyClassifier
    data_factory: TabularDataFactory

    def __init__(self) -> None:
        """Initialize tabular generator with default classifier."""
        self.classifier = DifficultyClassifier()
        self.data_factory = TabularDataFactory()

    def generate_value_extraction(self, df: pd.DataFrame) -> list[QAPair]:
        """Generate direct value lookup questions from a dataframe."""
        if df.empty:
            return []
        pairs: list[QAPair] = []
        first_row = df.iloc[0]
        data_id = str(first_row.get("data_id", "tabular-item-0"))
        for column in df.columns[:3]:
            answer = str(first_row[column])
            difficulty = self.classifier.classify(
                reasoning_steps=1,
                requires_arithmetic=False,
                has_counterfactual=False,
            )
            pairs.append(
                QAPair(
                    question=f"What is the value of '{column}' in row 0?",
                    answer=answer,
                    difficulty=difficulty.value,
                    data_id=data_id,
                    task="value_extraction",
                )
            )
        return pairs

    def generate_trend_analysis(self, df: pd.DataFrame) -> list[QAPair]:
        """Generate trend questions over numeric columns."""
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.empty or len(numeric_df) < 3:
            return []
        pairs: list[QAPair] = []
        for column in numeric_df.columns[:2]:
            start_value = float(numeric_df[column].iloc[0])
            end_value = float(numeric_df[column].iloc[-1])
            trend = "increasing" if end_value > start_value else "decreasing"
            difficulty = self.classifier.classify(
                reasoning_steps=2,
                requires_arithmetic=True,
                has_counterfactual=False,
            )
            pairs.append(
                QAPair(
                    question=f"Is '{column}' generally increasing or decreasing across rows?",
                    answer=trend,
                    difficulty=difficulty.value,
                    data_id=str(df.index.name or "tabular-trend"),
                    task="trend_analysis",
                )
            )
        return pairs

    def generate_comparison(self, df: pd.DataFrame) -> list[QAPair]:
        """Generate comparative questions between values or aggregates."""
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] < 2:
            return []
        pairs: list[QAPair] = []
        col_a, col_b = numeric_df.columns[:2]
        mean_a = float(numeric_df[col_a].mean())
        mean_b = float(numeric_df[col_b].mean())
        answer = col_a if mean_a > mean_b else col_b
        difficulty: Difficulty = self.classifier.classify(
            reasoning_steps=3,
            requires_arithmetic=True,
            has_counterfactual=False,
        )
        pairs.append(
            QAPair(
                question=f"Which column has the higher average value: '{col_a}' or '{col_b}'?",
                answer=answer,
                difficulty=difficulty.value,
                data_id="tabular-compare",
                task="comparison",
            )
        )
        pairs.append(
            QAPair(
                question=f"If all values in '{col_a}' were increased by 10%, would it still exceed '{col_b}' on average?",
                answer="yes" if mean_a * 1.1 > mean_b else "no",
                difficulty=Difficulty.COUNTERFACTUAL.value,
                data_id="tabular-compare",
                task="counterfactual",
            )
        )
        return pairs

    def generate_aggregation(
        self, df: pd.DataFrame, variant_idx: int = 0
    ) -> list[QAPair]:
        """Generate aggregation questions on deterministic numeric columns."""
        numeric_columns = self._numeric_columns(df)
        if not numeric_columns:
            return []
        column = numeric_columns[variant_idx % len(numeric_columns)]
        operation = ["sum", "average", "median"][variant_idx % 3]
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            return []
        answer_map = {
            "sum": round(float(series.sum()), 2),
            "average": round(float(series.mean()), 2),
            "median": round(float(median(series.tolist())), 2),
        }
        difficulty = self.classifier.classify(
            reasoning_steps=2,
            requires_arithmetic=True,
            has_counterfactual=False,
        )
        return [
            QAPair(
                question=(
                    f"What is the {operation} of '{column}' across all rows? "
                    "Round to 2 decimals."
                ),
                answer=str(answer_map[operation]),
                difficulty=difficulty.value,
                data_id=self._resolve_data_id(df),
                task="aggregation",
            )
        ]

    def generate_filtering(
        self, df: pd.DataFrame, variant_idx: int = 0
    ) -> list[QAPair]:
        """Generate filtering/count questions with numeric thresholds."""
        numeric_columns = self._numeric_columns(df)
        if not numeric_columns:
            return []
        column = numeric_columns[variant_idx % len(numeric_columns)]
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            return []
        quantile = 0.45 if variant_idx % 2 == 0 else 0.6
        threshold = round(float(series.quantile(quantile)), 2)
        count = int((series > threshold).sum())
        difficulty = self.classifier.classify(
            reasoning_steps=2,
            requires_arithmetic=True,
            has_counterfactual=False,
        )
        return [
            QAPair(
                question=f"How many rows have '{column}' strictly greater than {threshold}?",
                answer=str(count),
                difficulty=difficulty.value,
                data_id=self._resolve_data_id(df),
                task="filtering",
            )
        ]

    def generate_ranking(self, df: pd.DataFrame, variant_idx: int = 0) -> list[QAPair]:
        """Generate ranking questions for highest/lowest values."""
        numeric_columns = self._numeric_columns(df)
        if not numeric_columns:
            return []
        value_col = numeric_columns[variant_idx % len(numeric_columns)]
        label_col = self._label_column(df, exclude={value_col})
        if label_col is None:
            return []
        ascending = variant_idx % 2 == 1
        sorted_df = df.sort_values(by=value_col, ascending=ascending, kind="mergesort")
        target_row = sorted_df.iloc[0]
        direction = "lowest" if ascending else "highest"
        difficulty = self.classifier.classify(
            reasoning_steps=1,
            requires_arithmetic=False,
            has_counterfactual=False,
        )
        return [
            QAPair(
                question=f"Which '{label_col}' has the {direction} '{value_col}' value?",
                answer=str(target_row[label_col]),
                difficulty=difficulty.value,
                data_id=self._resolve_data_id(df),
                task="ranking",
            )
        ]

    def generate_outlier_detection(
        self, df: pd.DataFrame, variant_idx: int = 0
    ) -> list[QAPair]:
        """Generate outlier detection questions based on mean deviation."""
        numeric_columns = self._numeric_columns(df)
        if not numeric_columns:
            return []
        value_col = numeric_columns[variant_idx % len(numeric_columns)]
        label_col = self._label_column(df, exclude={value_col})
        if label_col is None:
            return []
        series = pd.to_numeric(df[value_col], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            return []
        mean_value = float(valid.mean())
        deviations = (valid - mean_value).abs()
        outlier_idx = deviations.idxmax()
        outlier_label = str(df.loc[outlier_idx, label_col])
        difficulty = self.classifier.classify(
            reasoning_steps=3,
            requires_arithmetic=True,
            has_counterfactual=False,
        )
        return [
            QAPair(
                question=(
                    f"Which '{label_col}' entry deviates most from the mean of '{value_col}'?"
                ),
                answer=outlier_label,
                difficulty=difficulty.value,
                data_id=self._resolve_data_id(df),
                task="outlier_detection",
            )
        ]

    def generate_counterfactual(
        self, df: pd.DataFrame, variant_idx: int = 0
    ) -> list[QAPair]:
        """Generate counterfactual ranking-change questions."""
        numeric_columns = self._numeric_columns(df)
        if len(numeric_columns) < 2:
            return []
        base_col = numeric_columns[variant_idx % len(numeric_columns)]
        compare_col = numeric_columns[(variant_idx + 1) % len(numeric_columns)]
        base_mean = float(pd.to_numeric(df[base_col], errors="coerce").dropna().mean())
        compare_mean = float(
            pd.to_numeric(df[compare_col], errors="coerce").dropna().mean()
        )
        multiplier = 1.2 if variant_idx % 2 == 0 else 0.85
        relation_before = base_mean > compare_mean
        relation_after = base_mean * multiplier > compare_mean
        answer = "yes" if relation_before == relation_after else "no"
        return [
            QAPair(
                question=(
                    f"If all values in '{base_col}' were multiplied by {multiplier}, "
                    f"would '{base_col}' remain above '{compare_col}' on average?"
                ),
                answer=answer,
                difficulty=Difficulty.COUNTERFACTUAL.value,
                data_id=self._resolve_data_id(df),
                task="counterfactual",
            )
        ]

    def generate_correlation(
        self, df: pd.DataFrame, variant_idx: int = 0
    ) -> list[QAPair]:
        """Generate correlation-sign questions for two numeric columns."""
        numeric_columns = self._numeric_columns(df)
        if len(numeric_columns) < 2:
            return []
        col_x = numeric_columns[variant_idx % len(numeric_columns)]
        col_y = numeric_columns[(variant_idx + 1) % len(numeric_columns)]
        x = pd.to_numeric(df[col_x], errors="coerce")
        y = pd.to_numeric(df[col_y], errors="coerce")
        joint = pd.concat([x, y], axis=1).dropna()
        if len(joint) < 3:
            return []
        corr = float(joint.iloc[:, 0].corr(joint.iloc[:, 1]))
        answer = "positive" if corr >= 0 else "negative"
        difficulty = self.classifier.classify(
            reasoning_steps=3,
            requires_arithmetic=True,
            has_counterfactual=False,
        )
        return [
            QAPair(
                question=(
                    f"Do '{col_x}' and '{col_y}' appear positively or negatively correlated?"
                ),
                answer=answer,
                difficulty=difficulty.value,
                data_id=self._resolve_data_id(df),
                task="correlation",
            )
        ]

    def generate_dataset_items(
        self,
        dataset: TabularDataset,
        items_per_dataset: int = 25,
        difficulty_targets: dict[Difficulty, int] | None = None,
    ) -> list[QAPair]:
        """Generate fixed-distribution tabular QA items for one dataset.

        Args:
            dataset: Generated tabular dataset container.
            items_per_dataset: Number of QA items to produce.
            difficulty_targets: Optional explicit per-difficulty item counts.

        Returns:
            Deterministic QA items with a balanced reasoning-depth distribution.
        """
        df = dataset.dataframe.copy()
        df["data_id"] = dataset.meta.name
        target_counts = difficulty_targets or self._difficulty_targets(
            items_per_dataset
        )

        items: list[QAPair] = []
        one_hop: list[list[QAPair]] = [
            self._normalize_pairs(
                self.generate_value_extraction(df), dataset.meta.name
            ),
        ]
        for idx in range(8):
            one_hop.append(
                self._normalize_pairs(self.generate_ranking(df, idx), dataset.meta.name)
            )
        items.extend(self._take_flat(one_hop, target_counts[Difficulty.ONE_HOP]))

        two_hop: list[list[QAPair]] = [
            self._normalize_pairs(self.generate_trend_analysis(df), dataset.meta.name),
        ]
        for idx in range(5):
            two_hop.append(
                self._normalize_pairs(
                    self.generate_aggregation(df, idx), dataset.meta.name
                )
            )
            two_hop.append(
                self._normalize_pairs(
                    self.generate_filtering(df, idx), dataset.meta.name
                )
            )
        items.extend(self._take_flat(two_hop, target_counts[Difficulty.TWO_HOP]))

        comparison_pairs = self._normalize_pairs(
            self.generate_comparison(df), dataset.meta.name
        )
        three_hop: list[list[QAPair]] = [
            [
                pair
                for pair in comparison_pairs
                if pair.difficulty == Difficulty.THREE_HOP.value
            ],
        ]
        for idx in range(6):
            three_hop.append(
                self._normalize_pairs(
                    self.generate_outlier_detection(df, idx), dataset.meta.name
                )
            )
            three_hop.append(
                self._normalize_pairs(
                    self.generate_correlation(df, idx), dataset.meta.name
                )
            )
        items.extend(self._take_flat(three_hop, target_counts[Difficulty.THREE_HOP]))

        counterfactual: list[list[QAPair]] = [
            [
                pair
                for pair in comparison_pairs
                if pair.difficulty == Difficulty.COUNTERFACTUAL.value
            ],
        ]
        for idx in range(8):
            counterfactual.append(
                self._normalize_pairs(
                    self.generate_counterfactual(df, idx), dataset.meta.name
                )
            )
        items.extend(
            self._take_flat(counterfactual, target_counts[Difficulty.COUNTERFACTUAL])
        )
        return items[:items_per_dataset]

    def generate_tabular_benchmark(
        self,
        num_datasets: int = 20,
        items_per_dataset: int = 25,
    ) -> list[QAPair]:
        """Generate full tabular benchmark QA items from synthetic sources."""
        datasets = self.data_factory.generate_all()[:num_datasets]
        schedule = self._benchmark_target_schedule(num_datasets, items_per_dataset)
        pairs: list[QAPair] = []
        for idx, dataset in enumerate(datasets):
            pairs.extend(
                self.generate_dataset_items(
                    dataset,
                    items_per_dataset=items_per_dataset,
                    difficulty_targets=schedule[idx],
                )
            )
        return pairs

    def _take_flat(self, nested_pairs: list[list[QAPair]], count: int) -> list[QAPair]:
        flat: list[QAPair] = []
        for group in nested_pairs:
            flat.extend(group)
            if len(flat) >= count:
                break
        return flat[:count]

    def _difficulty_targets(self, item_count: int) -> dict[Difficulty, int]:
        cf_count = int(round(item_count * 0.15))
        one_count = int(round(item_count * 0.30))
        two_count = int(round(item_count * 0.30))
        three_count = item_count - one_count - two_count - cf_count
        return {
            Difficulty.ONE_HOP: one_count,
            Difficulty.TWO_HOP: two_count,
            Difficulty.THREE_HOP: three_count,
            Difficulty.COUNTERFACTUAL: cf_count,
        }

    def _benchmark_target_schedule(
        self,
        num_datasets: int,
        items_per_dataset: int,
    ) -> list[dict[Difficulty, int]]:
        total_items = num_datasets * items_per_dataset
        remaining = {
            Difficulty.ONE_HOP: int(round(total_items * 0.30)),
            Difficulty.TWO_HOP: int(round(total_items * 0.30)),
            Difficulty.THREE_HOP: int(round(total_items * 0.25)),
        }
        remaining[Difficulty.COUNTERFACTUAL] = (
            total_items
            - remaining[Difficulty.ONE_HOP]
            - remaining[Difficulty.TWO_HOP]
            - remaining[Difficulty.THREE_HOP]
        )

        schedule: list[dict[Difficulty, int]] = []
        order = [
            Difficulty.ONE_HOP,
            Difficulty.TWO_HOP,
            Difficulty.THREE_HOP,
            Difficulty.COUNTERFACTUAL,
        ]
        for dataset_idx in range(num_datasets):
            left = num_datasets - dataset_idx
            allocation = {
                difficulty: remaining[difficulty] // left for difficulty in order
            }
            allocated = sum(allocation.values())
            extras = items_per_dataset - allocated
            if extras > 0:
                ranked = sorted(
                    order,
                    key=lambda difficulty: (
                        (remaining[difficulty] / left) - allocation[difficulty],
                        -order.index(difficulty),
                    ),
                    reverse=True,
                )
                for difficulty in ranked:
                    if extras == 0:
                        break
                    if remaining[difficulty] > allocation[difficulty]:
                        allocation[difficulty] += 1
                        extras -= 1
            for difficulty in order:
                remaining[difficulty] -= allocation[difficulty]
            schedule.append(allocation)
        return schedule

    def _normalize_pairs(self, pairs: list[QAPair], data_id: str) -> list[QAPair]:
        return [replace(pair, data_id=data_id) for pair in pairs]

    def _numeric_columns(self, df: pd.DataFrame) -> list[str]:
        candidates = df.select_dtypes(include="number").columns.tolist()
        filtered = [column for column in candidates if "id" not in column.lower()]
        return filtered if filtered else candidates

    def _label_column(self, df: pd.DataFrame, exclude: set[str]) -> str | None:
        for column in df.columns:
            if column in exclude or column == "data_id":
                continue
            if pd.api.types.is_string_dtype(
                df[column]
            ) or pd.api.types.is_categorical_dtype(df[column]):
                return str(column)
        for column in df.columns:
            if column in exclude or column == "data_id":
                continue
            if np.issubdtype(df[column].dtype, np.datetime64):
                return str(column)
        return None

    def _resolve_data_id(self, df: pd.DataFrame) -> str:
        if "data_id" in df.columns and not df.empty:
            return str(df["data_id"].iloc[0])
        return str(df.index.name or "tabular-item")
