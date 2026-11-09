from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, cast

import networkx as nx
import yaml

from src.generation import QAPair
from src.generation.data_sources.graph_sources import GraphDataFactory, GraphDataset
from src.generation.data_sources.tabular_sources import (
    TabularDataFactory,
    TabularDataset,
)
from src.generation.data_sources.timeseries_sources import (
    TimeSeriesDataFactory,
    TimeSeriesDataset,
)
from src.generation.mixed_generator import MixedBenchmarkGenerator
from src.utils.io_utils import BenchmarkItem, write_benchmark_items

VIZ_METHODS_BY_MODALITY: dict[str, list[str]] = {
    "mixed_tab_ts": [
        "bar_chart+line_plot",
        "heatmap+gaf",
        "scatter_plot+recurrence_plot",
        "table_image+heatmap",
        "text_only+text_only",
    ],
    "mixed_tab_graph": [
        "bar_chart+node_link",
        "heatmap+adjacency_matrix",
        "scatter_plot+circular_layout",
        "table_image+node_link",
        "text_only+text_only",
    ],
    "mixed_ts_graph": [
        "line_plot+node_link",
        "gaf+adjacency_matrix",
        "recurrence_plot+circular_layout",
        "heatmap+node_link",
        "text_only+text_only",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate mixed-type (Level 2) StructViz-Bench benchmark JSONL.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/generation.yaml"),
        help="Generation config YAML path (default: configs/generation.yaml).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)."
    )
    parser.add_argument(
        "--items-per-combo",
        type=int,
        default=None,
        help="Override items per mixed combination; defaults to level_2 config value.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/mixed_items.jsonl"),
        help="Output JSONL path (default: benchmark/mixed_items.jsonl).",
    )
    return parser.parse_args()


def _load_level2_items_per_combo(config_path: Path) -> int:
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = cast(dict[str, Any], yaml.safe_load(handle))
    level_2 = cast(dict[str, dict[str, Any]], cfg["level_2"])
    tab_ts = int(level_2["tab_ts"]["num_items"])
    tab_graph = int(level_2["tab_graph"]["num_items"])
    ts_graph = int(level_2["ts_graph"]["num_items"])
    values = {tab_ts, tab_graph, ts_graph}
    if len(values) != 1:
        raise ValueError("level_2 counts must match for generate_all_mixed().")
    return tab_ts


def _select_source_datasets(
    seed: int,
) -> tuple[TabularDataset, TimeSeriesDataset, GraphDataset]:
    tab_sets = TabularDataFactory(seed=seed).generate_all()
    ts_factory = TimeSeriesDataFactory(seed=seed)
    create_datasets = getattr(ts_factory, "create_datasets", None)
    ts_sets = (
        cast(list[TimeSeriesDataset], create_datasets())
        if callable(create_datasets)
        else ts_factory.generate_datasets(len(tab_sets))
    )
    graph_sets = GraphDataFactory(seed=seed).create_datasets()

    tab = tab_sets[seed % len(tab_sets)]
    ts = ts_sets[(seed + 1) % len(ts_sets)]
    graph = graph_sets[(seed + 2) % len(graph_sets)]
    return tab, ts, graph


def _modality_from_data_id(data_id: str) -> str:
    if data_id.startswith("mixed-tab_ts-"):
        return "mixed_tab_ts"
    if data_id.startswith("mixed-tab_graph-"):
        return "mixed_tab_graph"
    if data_id.startswith("mixed-ts_graph-"):
        return "mixed_ts_graph"
    raise ValueError(f"Unknown mixed data_id prefix: {data_id}")


def _tabular_payload(tab: TabularDataset) -> dict[str, Any]:
    return {
        "records": tab.dataframe.to_dict(orient="records"),
        "columns": [str(column) for column in tab.dataframe.columns],
        "metadata": {
            "domain": tab.meta.domain,
            "description": tab.meta.description,
            "column_descriptions": tab.meta.column_descriptions,
        },
    }


def _timeseries_payload(ts: TimeSeriesDataset) -> dict[str, Any]:
    return {
        "values": list(ts.values),
        "metadata": {
            "domain": ts.metadata.domain,
            "pattern_type": ts.metadata.pattern_type,
            "sampling_rate": ts.metadata.sampling_rate,
            "sampling_info": ts.metadata.sampling_info,
        },
    }


def _graph_payload(graph: GraphDataset) -> dict[str, Any]:
    return {
        "node_link": nx.node_link_data(graph.graph),
        "metadata": {
            "domain": graph.meta.domain,
            "topology_type": graph.meta.topology_type,
            "num_nodes": graph.meta.num_nodes,
            "num_edges": graph.meta.num_edges,
            "is_connected": graph.meta.is_connected,
            "num_components": graph.meta.num_components,
        },
    }


def _build_mixed_payload(
    modality: str,
    tab: TabularDataset,
    ts: TimeSeriesDataset,
    graph: GraphDataset,
) -> dict[str, Any]:
    if modality == "mixed_tab_ts":
        return {
            "left": {"modality": "tabular", "data": _tabular_payload(tab)},
            "right": {"modality": "timeseries", "data": _timeseries_payload(ts)},
        }
    if modality == "mixed_tab_graph":
        return {
            "left": {"modality": "tabular", "data": _tabular_payload(tab)},
            "right": {"modality": "graph", "data": _graph_payload(graph)},
        }
    if modality == "mixed_ts_graph":
        return {
            "left": {"modality": "timeseries", "data": _timeseries_payload(ts)},
            "right": {"modality": "graph", "data": _graph_payload(graph)},
        }
    raise ValueError(f"Unsupported mixed modality: {modality}")


def _to_benchmark_item(
    pair: QAPair,
    idx: int,
    seed: int,
    tab: TabularDataset,
    ts: TimeSeriesDataset,
    graph: GraphDataset,
) -> BenchmarkItem:
    modality = _modality_from_data_id(pair.data_id)
    payload = _build_mixed_payload(modality, tab, ts, graph)
    return BenchmarkItem(
        question_id=f"{modality}_{idx:06d}::difficulty={pair.difficulty}",
        question=pair.question,
        answer=pair.answer,
        modality=modality,
        data_id=pair.data_id,
        task=pair.task,
        difficulty=pair.difficulty,
        viz_methods=list(VIZ_METHODS_BY_MODALITY[modality]),
        data=payload,
        metadata={"seed": seed, "level": "level_2", "combination": modality},
        source="synthetic",
    )


def _print_summary(items: list[BenchmarkItem]) -> None:
    by_combo = Counter(item.modality for item in items)
    by_difficulty = Counter(item.difficulty for item in items)
    print(f"Total items: {len(items)}")
    print("Per-combination counts:")
    for combo in ["mixed_tab_ts", "mixed_tab_graph", "mixed_ts_graph"]:
        print(f"  - {combo}: {by_combo.get(combo, 0)}")
    print("Difficulty distribution:")
    for difficulty, count in sorted(by_difficulty.items()):
        print(f"  - {difficulty}: {count}")


def main() -> None:
    args = parse_args()
    items_per_combo = (
        int(args.items_per_combo)
        if args.items_per_combo is not None
        else _load_level2_items_per_combo(args.config)
    )

    generator = MixedBenchmarkGenerator(seed=args.seed)
    pairs = generator.generate_all_mixed(
        seed=args.seed,
        items_per_combination=items_per_combo,
    )

    tab, ts, graph = _select_source_datasets(seed=args.seed)
    items = [
        _to_benchmark_item(pair, idx, args.seed, tab, ts, graph)
        for idx, pair in enumerate(pairs)
    ]

    write_benchmark_items(args.output, items)
    _print_summary(items)
    print(f"Benchmark JSONL saved: {args.output}")


if __name__ == "__main__":
    main()
