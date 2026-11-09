from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.generation.data_sources.tabular_sources import DatasetMeta, TabularDataset

_TAG_PATTERN = re.compile(r"\[(?:BOLD|ITALIC)\]")
_NUMERIC_PATTERN = re.compile(r"^-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?%?$")


@dataclass(slots=True)
class SciTabAlignLoader:
    data_path: Path = Path("data/external/scitabalign/data/scitab_align_plus/data.json")

    def load_datasets(self) -> list[TabularDataset]:
        rows = self._read_rows()
        by_table: dict[str, dict[str, Any]] = {}
        for row in rows:
            table_id = str(row.get("table_id", "")).strip()
            if table_id and table_id not in by_table:
                by_table[table_id] = row

        datasets: list[TabularDataset] = []
        for table_id, row in by_table.items():
            columns = [
                self._clean_text(str(value))
                for value in row.get("table_column_names", [])
            ]
            if not columns:
                continue

            raw_values = row.get("table_content_values", [])
            cleaned_rows: list[list[Any]] = []
            for raw_row in raw_values:
                values = [self._to_scalar(value) for value in raw_row]
                if len(values) < len(columns):
                    values.extend(["" for _ in range(len(columns) - len(values))])
                cleaned_rows.append(values[: len(columns)])

            dataframe = pd.DataFrame(cleaned_rows, columns=columns)
            caption = self._clean_text(str(row.get("table_caption", "")))
            column_descriptions = {
                column: f"Column from SciTabAlign table: {column}" for column in columns
            }
            if dataframe.select_dtypes(include="number").empty:
                dataframe["__row_index__"] = list(range(len(dataframe)))
                column_descriptions["__row_index__"] = "Fallback numeric row index"
            datasets.append(
                TabularDataset(
                    meta=DatasetMeta(
                        name=table_id,
                        domain="scientific",
                        description=caption or f"Table from SciTabAlign ({table_id})",
                        column_descriptions=column_descriptions,
                    ),
                    dataframe=dataframe,
                )
            )
        return datasets

    def load_claims(self) -> list[dict[str, Any]]:
        """Return raw claim data for reference (claim text, label, table_id)."""
        rows = self._read_rows()
        return [
            {
                "id": str(row.get("id", "")),
                "claim": str(row.get("claim", "")),
                "label": str(row.get("label", row.get("human_label", ""))),
                "table_id": str(row.get("table_id", "")),
            }
            for row in rows
        ]

    def _read_rows(self) -> list[dict[str, Any]]:
        if not self.data_path.exists():
            return []
        payload = json.loads(self.data_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return []
        return [dict(row) for row in payload if isinstance(row, dict)]

    def _clean_text(self, value: str) -> str:
        no_tags = _TAG_PATTERN.sub("", value)
        return re.sub(r"\s+", " ", no_tags).strip()

    def _to_scalar(self, value: Any) -> Any:
        if isinstance(value, (int, float)):
            return float(value)
        text = self._clean_text(str(value))
        compact = text.replace(" ", "")
        if _NUMERIC_PATTERN.match(compact):
            normalized = compact.replace(",", "")
            if normalized.endswith("%"):
                normalized = normalized[:-1]
            try:
                return float(normalized)
            except ValueError:
                return text
        return text
