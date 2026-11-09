from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.generation import QAPair
from src.generation.data_sources.graph_sources import GraphDataFactory
from src.generation.data_sources.real_world.ett_loader import ETTLoader
from src.generation.data_sources.real_world.graph_loader import RealWorldGraphLoader
from src.generation.data_sources.real_world.scitabalign_loader import SciTabAlignLoader
from src.generation.data_sources.tabular_sources import TabularDataFactory
from src.generation.data_sources.timeseries_sources import TimeSeriesDataFactory
from src.generation.graph_generator import GraphBenchmarkGenerator
from src.generation.tabular_generator import TabularBenchmarkGenerator
from src.generation.timeseries_generator import TimeSeriesBenchmarkGenerator
from src.utils.io_utils import BenchmarkItem, write_benchmark_items


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for benchmark generation."""
    parser = argparse.ArgumentParser(
        description="Generate StructViz-Bench base benchmark JSONL."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Generation config YAML path."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output benchmark JSONL path."
    )
    parser.add_argument(
        "--include-realworld",
        action="store_true",
        help="Append real-world datasets (SciTabAlign, ETT, NetworkX patterns).",
    )
    return parser.parse_args()


def _load_timeseries_datasets(
    factory: TimeSeriesDataFactory, num_datasets: int
) -> list[Any]:
    """Load time-series datasets using available factory API."""
    return factory.generate_datasets(num_datasets)


def _to_item(
    pair: QAPair,
    idx: int,
    modality: str,
    viz_methods: list[str],
    data: Any,
    metadata: dict[str, Any],
    source: str = "synthetic",
) -> BenchmarkItem:
    """Convert QAPair output into canonical BenchmarkItem."""
    return BenchmarkItem(
        question_id=f"{modality}_{idx:06d}::difficulty={pair.difficulty}",
        question=pair.question,
        answer=pair.answer,
        modality=modality,
        data_id=pair.data_id,
        task=pair.task,
        difficulty=pair.difficulty,
        viz_methods=viz_methods,
        data=data,
        metadata=metadata,
        source=source,
    )


def _build_tabular_items(
    cfg: dict[str, Any],
    seed: int,
    start_idx: int = 0,
) -> list[BenchmarkItem]:
    """Generate tabular BenchmarkItem rows from factory and generator outputs."""
    tab_cfg = cfg["level_1"]["tabular"]
    num_datasets = int(tab_cfg["num_datasets"])
    items_per_dataset = int(tab_cfg["items_per_dataset"])
    viz_methods = [str(v) for v in tab_cfg["viz_methods"]]

    datasets = TabularDataFactory(seed=seed).generate_all()[:num_datasets]
    pairs = TabularBenchmarkGenerator().generate_tabular_benchmark(
        num_datasets=num_datasets,
        items_per_dataset=items_per_dataset,
    )
    dataset_map = {dataset.meta.name: dataset for dataset in datasets}

    items: list[BenchmarkItem] = []
    for idx, pair in enumerate(pairs):
        dataset = dataset_map.get(pair.data_id, datasets[idx % len(datasets)])
        items.append(
            _to_item(
                pair=pair,
                idx=start_idx + idx,
                modality="tabular",
                viz_methods=viz_methods,
                data=dataset.dataframe.copy(),
                metadata={
                    "domain": dataset.meta.domain,
                    "description": dataset.meta.description,
                    "column_descriptions": dataset.meta.column_descriptions,
                },
            )
        )
    return items


def _build_timeseries_items(
    cfg: dict[str, Any],
    seed: int,
    start_idx: int = 0,
) -> list[BenchmarkItem]:
    """Generate time-series BenchmarkItem rows from factory and generator outputs."""
    ts_cfg = cfg["level_1"]["timeseries"]
    num_datasets = int(ts_cfg["num_datasets"])
    items_per_dataset = int(ts_cfg["items_per_dataset"])
    viz_methods = [str(v) for v in ts_cfg["viz_methods"]]

    datasets = _load_timeseries_datasets(TimeSeriesDataFactory(seed=seed), num_datasets)
    pairs = TimeSeriesBenchmarkGenerator().generate_benchmark_items(
        datasets, items_per_dataset
    )
    dataset_map = {dataset.data_id: dataset for dataset in datasets}

    items: list[BenchmarkItem] = []
    for idx, pair in enumerate(pairs):
        dataset = dataset_map[pair.data_id]
        items.append(
            _to_item(
                pair=pair,
                idx=start_idx + idx,
                modality="timeseries",
                viz_methods=viz_methods,
                data=list(dataset.values),
                metadata={
                    "domain": dataset.metadata.domain,
                    "pattern_type": dataset.metadata.pattern_type,
                    "sampling_rate": dataset.metadata.sampling_rate,
                    "sampling_info": dataset.metadata.sampling_info,
                },
            )
        )
    return items


def _build_graph_items(
    cfg: dict[str, Any],
    seed: int,
    start_idx: int = 0,
) -> list[BenchmarkItem]:
    """Generate graph BenchmarkItem rows from factory and generator outputs."""
    graph_cfg = cfg["level_1"]["graph"]
    num_datasets = int(graph_cfg["num_datasets"])
    items_per_dataset = int(graph_cfg["items_per_dataset"])
    viz_methods = [str(v) for v in graph_cfg["viz_methods"]]

    datasets = GraphDataFactory(seed=seed).create_datasets()[:num_datasets]
    generator = GraphBenchmarkGenerator()
    items: list[BenchmarkItem] = []
    idx = start_idx
    for dataset in datasets:
        pairs = generator.generate_dataset_qa(
            graph=dataset.graph,
            data_id=dataset.meta.name,
            items_per_dataset=items_per_dataset,
        )
        for pair in pairs:
            items.append(
                _to_item(
                    pair=pair,
                    idx=idx,
                    modality="graph",
                    viz_methods=viz_methods,
                    data=dataset.graph,
                    metadata={
                        "domain": dataset.meta.domain,
                        "topology_type": dataset.meta.topology_type,
                        "num_nodes": dataset.meta.num_nodes,
                        "num_edges": dataset.meta.num_edges,
                    },
                )
            )
            idx += 1
    return items


def _build_realworld_tabular_items(
    cfg: dict[str, Any],
    start_idx: int = 0,
) -> list[BenchmarkItem]:
    rw_tab_cfg = cfg["real_world"]["tabular"]
    items_per_dataset = int(rw_tab_cfg["items_per_dataset"])
    viz_methods = [str(method) for method in rw_tab_cfg["viz_methods"]]

    loader = SciTabAlignLoader(data_path=Path(str(rw_tab_cfg["data_path"])))
    datasets = loader.load_datasets()
    generator = TabularBenchmarkGenerator()

    items: list[BenchmarkItem] = []
    idx = start_idx
    for dataset in datasets:
        pairs = generator.generate_dataset_items(
            dataset=dataset,
            items_per_dataset=items_per_dataset,
        )
        for pair in pairs:
            items.append(
                _to_item(
                    pair=pair,
                    idx=idx,
                    modality="tabular",
                    viz_methods=viz_methods,
                    data=dataset.dataframe.copy(),
                    metadata={
                        "domain": dataset.meta.domain,
                        "description": dataset.meta.description,
                        "column_descriptions": dataset.meta.column_descriptions,
                    },
                    source="scitabalign",
                )
            )
            idx += 1
    return items


def _build_realworld_timeseries_items(
    cfg: dict[str, Any],
    start_idx: int = 0,
) -> list[BenchmarkItem]:
    rw_ts_cfg = cfg["real_world"]["timeseries"]
    items_per_dataset = int(rw_ts_cfg["items_per_dataset"])
    viz_methods = [str(method) for method in rw_ts_cfg["viz_methods"]]

    datasets = ETTLoader(
        data_dir=Path(str(rw_ts_cfg["data_dir"])),
        window_size=int(rw_ts_cfg["window_size"]),
        windows_per_column=int(rw_ts_cfg["windows_per_column"]),
        seed=int(cfg["benchmark"]["seed"]),
    ).load_datasets()
    pairs = TimeSeriesBenchmarkGenerator().generate_benchmark_items(
        datasets,
        items_per_dataset,
    )
    dataset_map = {dataset.data_id: dataset for dataset in datasets}

    items: list[BenchmarkItem] = []
    for idx, pair in enumerate(pairs):
        dataset = dataset_map[pair.data_id]
        items.append(
            _to_item(
                pair=pair,
                idx=start_idx + idx,
                modality="timeseries",
                viz_methods=viz_methods,
                data=list(dataset.values),
                metadata={
                    "domain": dataset.metadata.domain,
                    "pattern_type": dataset.metadata.pattern_type,
                    "sampling_rate": dataset.metadata.sampling_rate,
                    "sampling_info": dataset.metadata.sampling_info,
                },
                source="ett",
            )
        )
    return items


def _build_realworld_graph_items(
    cfg: dict[str, Any],
    seed: int,
    start_idx: int = 0,
) -> list[BenchmarkItem]:
    rw_graph_cfg = cfg["real_world"]["graph"]
    items_per_dataset = int(rw_graph_cfg["items_per_dataset"])
    viz_methods = [str(method) for method in rw_graph_cfg["viz_methods"]]

    datasets = RealWorldGraphLoader(seed=seed).load_datasets()
    generator = GraphBenchmarkGenerator()

    items: list[BenchmarkItem] = []
    idx = start_idx
    for dataset in datasets:
        pairs = generator.generate_dataset_qa(
            graph=dataset.graph,
            data_id=dataset.meta.name,
            items_per_dataset=items_per_dataset,
        )
        for pair in pairs:
            items.append(
                _to_item(
                    pair=pair,
                    idx=idx,
                    modality="graph",
                    viz_methods=viz_methods,
                    data=dataset.graph,
                    metadata={
                        "domain": dataset.meta.domain,
                        "topology_type": dataset.meta.topology_type,
                        "num_nodes": dataset.meta.num_nodes,
                        "num_edges": dataset.meta.num_edges,
                    },
                    source="networkx_realworld",
                )
            )
            idx += 1
    return items


def _print_summary(items: list[BenchmarkItem]) -> None:
    """Print benchmark generation counts by modality and difficulty."""
    by_modality = Counter(item.modality for item in items)
    by_difficulty = Counter(item.difficulty for item in items)
    print(f"Generated items: {len(items)}")
    print("Modality counts:")
    for modality, count in sorted(by_modality.items()):
        print(f"  - {modality}: {count}")
    print("Difficulty counts:")
    for difficulty, count in sorted(by_difficulty.items()):
        print(f"  - {difficulty}: {count}")


def main() -> None:
    """Run full benchmark item generation for all three modalities."""
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    seed = int(cfg["benchmark"]["seed"])
    tabular_items = _build_tabular_items(cfg, seed)
    timeseries_items = _build_timeseries_items(cfg, seed)
    graph_items = _build_graph_items(cfg, seed)
    all_items = tabular_items + timeseries_items + graph_items

    if args.include_realworld:
        all_items += _build_realworld_tabular_items(cfg, start_idx=len(tabular_items))
        all_items += _build_realworld_timeseries_items(
            cfg,
            start_idx=len(tabular_items) + len(timeseries_items),
        )
        all_items += _build_realworld_graph_items(
            cfg,
            seed,
            start_idx=len(tabular_items) + len(timeseries_items) + len(graph_items),
        )

    write_benchmark_items(args.output, all_items)
    _print_summary(all_items)
    print(f"Benchmark JSONL saved: {args.output}")


if __name__ == "__main__":
    main()
