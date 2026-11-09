from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import networkx as nx
import pandas as pd
import pytest

from src.generation.data_sources.real_world import (
    ETTLoader,
    RealWorldGraphLoader,
    SciTabAlignLoader,
)
from src.utils.io_utils import BenchmarkItem

# The raw SciTabAlign and ETT sources are not redistributed with this repository
# (see the dataset card); these tests run only after the sources are placed locally.
_needs_scitab = pytest.mark.skipif(
    not SciTabAlignLoader().load_datasets(), reason="SciTabAlign source data not present",
)
_needs_ett = pytest.mark.skipif(
    not ETTLoader(window_size=200, windows_per_column=3, seed=42).load_datasets(),
    reason="ETT source data not present (see scripts/data/rebuild_ett.py)",
)


@_needs_scitab
def test_scitabalign_loader_loads_datasets() -> None:
    datasets = SciTabAlignLoader().load_datasets()
    assert datasets
    for dataset in datasets:
        assert isinstance(dataset.dataframe, pd.DataFrame)
        numeric_columns = dataset.dataframe.select_dtypes(include="number").columns
        assert len(numeric_columns) > 0


@_needs_scitab
def test_scitabalign_loader_loads_claims() -> None:
    claims = SciTabAlignLoader().load_claims()
    assert len(claims) == 372
    assert all(
        "claim" in claim and "label" in claim and "table_id" in claim
        for claim in claims
    )


@_needs_ett
def test_ett_loader_loads_datasets() -> None:
    datasets = ETTLoader(window_size=200, windows_per_column=3, seed=42).load_datasets()
    assert datasets
    ett_datasets = [
        dataset for dataset in datasets if dataset.data_id.startswith("ett_")
    ]
    assert ett_datasets
    assert all(len(dataset.values) == 200 for dataset in ett_datasets)
    for dataset in ett_datasets[:10]:
        assert isinstance(dataset.values, list)
        assert all(isinstance(value, float) for value in dataset.values)


def test_realworld_graph_loader_loads_datasets() -> None:
    datasets = RealWorldGraphLoader(seed=42).load_datasets()
    assert datasets
    assert 15 <= len(datasets) <= 20
    for dataset in datasets:
        assert isinstance(dataset.graph, nx.Graph)
        assert 15 <= dataset.graph.number_of_nodes() <= 100


def test_benchmark_item_source_field() -> None:
    item = BenchmarkItem(
        question_id="tabular_000001::difficulty=1-hop",
        question="What is value x?",
        answer="1",
        modality="tabular",
        data_id="demo",
        task="value_extraction",
        difficulty="1-hop",
        viz_methods=["table_image"],
        data={"x": 1},
        source="scitabalign",
    )
    row = item.to_dict()
    assert row["source"] == "scitabalign"

    loaded = BenchmarkItem.from_dict(row)
    assert loaded.source == "scitabalign"

    legacy_loaded = BenchmarkItem.from_dict(
        {
            "question_id": "legacy",
            "question": "q",
            "answer": "a",
            "modality": "tabular",
            "data_id": "id",
            "task": "task",
            "difficulty": "1-hop",
            "viz_methods": [],
            "metadata": {},
            "data_format": "json_dict",
            "data": {},
        }
    )
    assert legacy_loaded.source == "synthetic"
