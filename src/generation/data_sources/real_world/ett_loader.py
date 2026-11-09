from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.generation.data_sources.timeseries_sources import (
    TimeSeriesDataset,
    TimeSeriesMeta,
)


@dataclass(slots=True)
class ETTLoader:
    data_dir: Path = Path("data/external/ett/repo/ETT-small")
    window_size: int = 200
    windows_per_column: int = 3
    seed: int = 42

    def load_datasets(self) -> list[TimeSeriesDataset]:
        datasets: list[TimeSeriesDataset] = []
        csv_files = sorted(self.data_dir.glob("ETT*.csv"))
        for file_index, csv_path in enumerate(csv_files):
            frame = pd.read_csv(csv_path)
            numeric_columns = [
                column
                for column in frame.columns
                if column != "date" and pd.api.types.is_numeric_dtype(frame[column])
            ]

            for column_index, column in enumerate(numeric_columns):
                values = [float(value) for value in frame[column].tolist()]
                windows = self._select_non_overlapping_windows(
                    values=values,
                    seed_offset=file_index * 100 + column_index,
                )
                for window_index, window in enumerate(windows):
                    data_id = f"ett_{csv_path.stem}_{column}_w{window_index:02d}"
                    datasets.append(
                        TimeSeriesDataset(
                            data_id=data_id,
                            values=window,
                            metadata=TimeSeriesMeta(
                                name=f"{csv_path.stem}:{column}:window_{window_index}",
                                domain="energy",
                                description=f"ETT measurement column {column} from {csv_path.stem}",
                                pattern_type="real_world_measurement",
                                length=len(window),
                                sampling_rate="hourly"
                                if "ETTh" in csv_path.stem
                                else "15min",
                                sampling_info="windowed non-overlapping segment",
                                known_anomaly_indices=[],
                            ),
                        )
                    )

        return datasets

    def _select_non_overlapping_windows(
        self, values: list[float], seed_offset: int
    ) -> list[list[float]]:
        if not values:
            return []
        if len(values) <= self.window_size:
            return [values]

        starts = list(range(0, len(values) - self.window_size + 1, self.window_size))
        if not starts:
            return [values[: self.window_size]]

        count = min(self.windows_per_column, len(starts))
        rng = random.Random(self.seed + seed_offset)
        chosen = sorted(rng.sample(starts, k=count))
        return [values[start : start + self.window_size] for start in chosen]

