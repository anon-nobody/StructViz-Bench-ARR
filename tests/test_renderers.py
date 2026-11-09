from __future__ import annotations

import networkx as nx
import pandas as pd

from src.rendering.render_pipeline import RenderPipeline


def test_render_pipeline_tabular_outputs_all_methods() -> None:
    pipeline = RenderPipeline()
    rendered = pipeline.render_all(
        {
            "modality": "tabular",
            "data": pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]}),
        }
    )
    assert {
        "bar_chart",
        "heatmap",
        "table_image",
        "scatter_plot",
        "text_only",
    }.issubset(rendered)


def test_render_pipeline_timeseries_outputs_all_methods() -> None:
    pipeline = RenderPipeline()
    rendered = pipeline.render_all(
        {"modality": "timeseries", "data": [1.0, 1.2, 1.1, 1.3]}
    )
    assert {"line_plot", "gaf", "recurrence_plot", "heatmap", "text_only"}.issubset(
        rendered
    )


def test_render_pipeline_graph_outputs_all_methods() -> None:
    pipeline = RenderPipeline()
    rendered = pipeline.render_all({"modality": "graph", "data": nx.cycle_graph(5)})
    assert {"node_link", "adjacency_matrix", "circular_layout", "text_only"}.issubset(
        rendered
    )
