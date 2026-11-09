from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonlines
import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with jsonlines.open(path, mode="r") as reader:
        return [dict(record) for record in reader]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with jsonlines.open(path, mode="w", dumps=_safe_dumps) as writer:
        for row in rows:
            writer.write(row)


def _safe_default(obj: object) -> object:
    """JSON encoder fallback for non-serializable types."""
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return getattr(obj, "item")()
        except (TypeError, ValueError):
            pass
    to_list = getattr(obj, "tolist", None)
    if callable(to_list):
        return to_list()
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    isoformat = getattr(obj, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if isinstance(obj, set):
        return (
            sorted(obj)
            if all(isinstance(x, (str, int, float)) for x in obj)
            else list(obj)
        )
    if isinstance(obj, bool):
        return obj
    if hasattr(obj, "__dict__"):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _safe_dumps(obj: object) -> str:
    return json.dumps(obj, default=_safe_default, ensure_ascii=False)


def save_image(path: Path, image: Image.Image) -> None:
    ensure_dir(path.parent)
    image.save(path)


@dataclass(slots=True)
class BenchmarkItem:
    """Canonical benchmark item with serialization helpers.

    Attributes:
        question_id: Unique item identifier.
        question: Natural-language question.
        answer: Ground-truth answer.
        modality: Data modality for this item.
        data_id: Source dataset identifier.
        task: Task category name.
        difficulty: Difficulty label (1-hop, 2-hop, 3-hop, counterfactual).
        viz_methods: Applicable visualization methods.
        data: Structured source payload (dataframe, list, graph, dict).
        metadata: Additional metadata fields.
        image_path: Optional rendered image path.
        viz_type: Optional visualization type for rendered variants.
    """

    question_id: str
    question: str
    answer: str
    modality: str
    data_id: str
    task: str
    difficulty: str
    viz_methods: list[str]
    data: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    image_path: str | None = None
    viz_type: str | None = None
    source: str = "synthetic"

    def to_dict(self) -> dict[str, Any]:
        """Serialize item into a JSONL-safe dictionary."""
        row: dict[str, Any] = {
            "question_id": self.question_id,
            "question": self.question,
            "answer": self.answer,
            "modality": self.modality,
            "data_id": self.data_id,
            "task": self.task,
            "difficulty": self.difficulty,
            "viz_methods": self.viz_methods,
            "metadata": self.metadata,
        }
        row.update(_serialize_data_payload(self.data))
        if self.image_path is not None:
            row["image_path"] = self.image_path
        if self.viz_type is not None:
            row["viz_type"] = self.viz_type
        row["source"] = self.source
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> BenchmarkItem:
        """Deserialize one benchmark item row from JSONL."""
        data = _deserialize_data_payload(row)
        return cls(
            question_id=str(row["question_id"]),
            question=str(row["question"]),
            answer=str(row["answer"]),
            modality=str(row["modality"]),
            data_id=str(row["data_id"]),
            task=str(row["task"]),
            difficulty=str(row["difficulty"]),
            viz_methods=[str(v) for v in row.get("viz_methods", [])],
            data=data,
            metadata=dict(row.get("metadata", {})),
            image_path=str(row["image_path"]) if row.get("image_path") else None,
            viz_type=str(row["viz_type"]) if row.get("viz_type") else None,
            source=str(row.get("source", "synthetic")),
        )


def _serialize_data_payload(data: Any) -> dict[str, Any]:
    if isinstance(data, pd.DataFrame):
        # Convert to records with native Python types to ensure JSON safety
        records = json.loads(data.to_json(orient="records", date_format="iso"))
        return {
            "data_format": "dataframe_records",
            "data": records,
            "data_columns": list(data.columns),
        }
    if isinstance(data, nx.Graph):
        return {
            "data_format": "networkx_node_link",
            "data": nx.node_link_data(data),
        }
    if isinstance(data, list):
        return {
            "data_format": "timeseries_list",
            "data": data,
        }
    if isinstance(data, dict):
        return {
            "data_format": "json_dict",
            "data": data,
        }
    return {
        "data_format": "raw",
        "data": data,
    }


def _deserialize_data_payload(row: dict[str, Any]) -> Any:
    data_format = str(row.get("data_format", "raw"))
    payload = row.get("data")
    if data_format == "dataframe_records":
        columns = row.get("data_columns")
        if isinstance(columns, list):
            return pd.DataFrame(payload, columns=[str(column) for column in columns])
        return pd.DataFrame(payload)
    if data_format == "networkx_node_link":
        if not isinstance(payload, dict):
            return nx.Graph()
        return nx.node_link_graph(payload)
    if data_format == "timeseries_list":
        if isinstance(payload, list):
            return [float(value) for value in payload]
        return []
    if data_format == "json_dict":
        return dict(payload) if isinstance(payload, dict) else {}
    return payload


def write_benchmark_items(path: Path, items: list[BenchmarkItem]) -> None:
    """Write benchmark items as JSONL."""
    write_jsonl(path, [item.to_dict() for item in items])


def read_benchmark_items(path: Path) -> list[BenchmarkItem]:
    """Read benchmark items from JSONL."""
    rows = read_jsonl(path)
    return [BenchmarkItem.from_dict(row) for row in rows]
