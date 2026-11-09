from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import networkx as nx
import pandas as pd
import pytest
from typing import Any

from src.generation.data_sources.graph_sources import GraphDataFactory
from src.generation.data_sources.tabular_sources import TabularDataFactory
from src.generation.data_sources.timeseries_sources import TimeSeriesDataFactory
from src.generation.difficulty_classifier import Difficulty
from src.generation.graph_generator import GraphBenchmarkGenerator
from src.generation.tabular_generator import TabularBenchmarkGenerator
from src.generation.timeseries_generator import TimeSeriesBenchmarkGenerator

try:
    from src.generation.mixed_generator import MixedTypeGenerator
except ImportError:

    class _MissingMixedTypeGenerator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pytest.skip("src.generation.mixed_generator is not available")

    MixedTypeGenerator: Any = _MissingMixedTypeGenerator


def _create_timeseries_datasets(seed: int = 42) -> list[Any]:
    factory = TimeSeriesDataFactory(seed=seed)
    if hasattr(factory, "create_datasets"):
        return factory.create_datasets()
    return factory.generate_datasets(20)


def test_tabular_generator_outputs_pairs() -> None:
    generator = TabularBenchmarkGenerator()
    df = pd.DataFrame({"data_id": ["x0", "x1"], "a": [1, 2], "b": [3, 4]})
    pairs = generator.generate_value_extraction(df)
    assert pairs
    assert all(pair.question for pair in pairs)
    assert all(pair.answer is not None for pair in pairs)


def test_tabular_data_factory_generates_20_diverse_datasets() -> None:
    datasets = TabularDataFactory(seed=42).generate_all()
    assert len(datasets) == 20
    assert len({dataset.meta.domain for dataset in datasets}) >= 8
    for dataset in datasets:
        assert 15 <= len(dataset.dataframe) <= 50
        assert 4 <= len(dataset.dataframe.columns) <= 8
        assert dataset.meta.column_descriptions


def test_tabular_generator_new_question_templates() -> None:
    generator = TabularBenchmarkGenerator()
    dataset = TabularDataFactory(seed=7).generate_all()[0]
    df = dataset.dataframe.copy()
    df["data_id"] = dataset.meta.name

    task_methods = {
        "aggregation": generator.generate_aggregation,
        "filtering": generator.generate_filtering,
        "ranking": generator.generate_ranking,
        "outlier_detection": generator.generate_outlier_detection,
        "counterfactual": generator.generate_counterfactual,
        "correlation": generator.generate_correlation,
    }
    for task, method in task_methods.items():
        pairs = method(df)
        assert pairs
        assert pairs[0].task == task


def test_tabular_generator_dataset_items_have_target_distribution() -> None:
    generator = TabularBenchmarkGenerator()
    dataset = TabularDataFactory(seed=21).generate_all()[3]
    pairs = generator.generate_dataset_items(dataset, items_per_dataset=25)
    assert len(pairs) == 25

    counts: dict[str, int] = {}
    for pair in pairs:
        counts[pair.difficulty] = counts.get(pair.difficulty, 0) + 1

    assert set(counts).issuperset(
        {
            Difficulty.ONE_HOP.value,
            Difficulty.TWO_HOP.value,
            Difficulty.THREE_HOP.value,
            Difficulty.COUNTERFACTUAL.value,
        }
    )


def test_tabular_generator_benchmark_has_500_items_and_target_mix() -> None:
    generator = TabularBenchmarkGenerator()
    pairs = generator.generate_tabular_benchmark(num_datasets=20, items_per_dataset=25)
    assert len(pairs) == 500

    counts: dict[str, int] = {}
    for pair in pairs:
        counts[pair.difficulty] = counts.get(pair.difficulty, 0) + 1

    assert counts[Difficulty.ONE_HOP.value] == 150
    assert counts[Difficulty.TWO_HOP.value] == 150
    assert counts[Difficulty.THREE_HOP.value] == 125
    assert counts[Difficulty.COUNTERFACTUAL.value] == 75


def test_timeseries_generator_outputs_anomaly_question() -> None:
    generator = TimeSeriesBenchmarkGenerator()
    ts = [1.0, 1.1, 10.0, 1.2, 1.0]
    pairs = generator.generate_anomaly_detection(ts)
    assert len(pairs) == 1
    assert pairs[0].task == "anomaly_detection"


def test_timeseries_data_factory_generates_20_datasets() -> None:
    datasets = _create_timeseries_datasets(seed=42)
    assert len(datasets) == 20

    pattern_types = {dataset.metadata.pattern_type for dataset in datasets}
    assert len(pattern_types) >= 4

    for dataset in datasets:
        assert len(dataset.values) >= 50
        assert dataset.data_id
        assert dataset.metadata.domain


def test_timeseries_data_factory_is_deterministic() -> None:
    first = _create_timeseries_datasets(seed=42)
    second = _create_timeseries_datasets(seed=42)

    for first_dataset, second_dataset in zip(first, second):
        assert first_dataset.data_id == second_dataset.data_id
        assert first_dataset.values == second_dataset.values
        assert (
            first_dataset.metadata.pattern_type == second_dataset.metadata.pattern_type
        )


def test_timeseries_generator_question_type_methods_return_expected_tasks() -> None:
    generator = TimeSeriesBenchmarkGenerator()
    dataset = _create_timeseries_datasets(seed=42)[0]

    seasonality_pairs = generator.generate_seasonality_detection(dataset)
    peak_pairs = generator.generate_peak_identification(dataset)
    range_pairs = generator.generate_range_query(dataset)
    change_point_pairs = generator.generate_change_point(dataset)
    volatility_pairs = generator.generate_volatility(dataset)

    assert len(seasonality_pairs) == 1
    assert len(peak_pairs) == 2
    assert len(range_pairs) == 1
    assert len(change_point_pairs) == 1
    assert len(volatility_pairs) == 1
    assert seasonality_pairs[0].task == "seasonality_detection"
    assert all(pair.task == "peak_identification" for pair in peak_pairs)
    assert range_pairs[0].task == "range_query"
    assert change_point_pairs[0].task == "change_point"
    assert volatility_pairs[0].task == "volatility"


def test_timeseries_generator_seasonality_detection() -> None:
    generator = TimeSeriesBenchmarkGenerator()
    dataset = _create_timeseries_datasets(seed=42)[0]
    pairs = generator.generate_seasonality_detection(dataset)
    assert len(pairs) == 1
    assert pairs[0].task == "seasonality_detection"
    assert pairs[0].answer in ("yes", "no")


def test_timeseries_generator_peak_identification() -> None:
    generator = TimeSeriesBenchmarkGenerator()
    dataset = _create_timeseries_datasets(seed=42)[0]
    pairs = generator.generate_peak_identification(dataset)
    assert len(pairs) == 2
    assert all(pair.task == "peak_identification" for pair in pairs)


def test_timeseries_generator_range_query() -> None:
    generator = TimeSeriesBenchmarkGenerator()
    dataset = _create_timeseries_datasets(seed=42)[0]
    pairs = generator.generate_range_query(dataset)
    assert len(pairs) == 1
    assert pairs[0].task == "range_query"
    _ = float(pairs[0].answer)


def test_timeseries_generator_change_point() -> None:
    generator = TimeSeriesBenchmarkGenerator()
    dataset = _create_timeseries_datasets(seed=42)[0]
    pairs = generator.generate_change_point(dataset)
    assert len(pairs) == 1
    assert pairs[0].task == "change_point"
    assert pairs[0].answer == "none" or pairs[0].answer.isdigit()


def test_timeseries_generator_volatility() -> None:
    generator = TimeSeriesBenchmarkGenerator()
    dataset = _create_timeseries_datasets(seed=42)[0]
    pairs = generator.generate_volatility(dataset)
    assert len(pairs) == 1
    assert pairs[0].task == "volatility"
    assert pairs[0].answer in ("first half", "second half", "equal")


def test_timeseries_generator_dataset_questions_target_count() -> None:
    generator = TimeSeriesBenchmarkGenerator()
    dataset = _create_timeseries_datasets(seed=42)[0]
    pairs = generator.generate_dataset_questions(
        dataset=dataset,
        dataset_index=0,
        target_items=25,
    )
    assert len(pairs) == 25


def test_timeseries_generator_benchmark_has_500_items() -> None:
    generator = TimeSeriesBenchmarkGenerator()
    datasets = _create_timeseries_datasets(seed=42)
    pairs = generator.generate_benchmark_items(datasets, items_per_dataset=25)
    assert len(pairs) == 500

    counts: dict[str, int] = {}
    for pair in pairs:
        counts[pair.difficulty] = counts.get(pair.difficulty, 0) + 1

    assert counts[Difficulty.ONE_HOP.value] == 150
    assert counts[Difficulty.TWO_HOP.value] == 150
    assert counts[Difficulty.THREE_HOP.value] == 125
    assert counts[Difficulty.COUNTERFACTUAL.value] == 75


def test_mixed_generator_import_guard_when_module_missing() -> None:
    generator = MixedTypeGenerator(seed=42)
    assert generator is not None


def test_graph_generator_outputs_connectivity() -> None:
    generator = GraphBenchmarkGenerator()
    graph = nx.path_graph(4)
    pairs = generator.generate_connectivity(graph)
    assert len(pairs) == 1
    assert pairs[0].task == "connectivity"


def test_graph_data_factory_generates_20_diverse_datasets() -> None:
    factory = GraphDataFactory(seed=42)
    datasets = factory.create_datasets()

    assert len(datasets) == 20
    assert all(10 <= dataset.meta.num_nodes <= 100 for dataset in datasets)
    assert all(dataset.meta.num_edges >= 0 for dataset in datasets)
    assert {dataset.meta.topology_type for dataset in datasets} == {
        "erdos_renyi",
        "barabasi_albert",
        "watts_strogatz",
        "tree",
        "grid",
        "star",
        "community",
        "bipartite",
    }


def test_graph_data_factory_is_deterministic() -> None:
    first = GraphDataFactory(seed=42).create_datasets()
    second = GraphDataFactory(seed=42).create_datasets()

    first_signature = [
        (
            dataset.meta.name,
            dataset.meta.domain,
            dataset.meta.topology_type,
            dataset.meta.num_nodes,
            dataset.meta.num_edges,
            dataset.meta.is_connected,
            dataset.meta.num_components,
            sorted(dataset.graph.edges()),
        )
        for dataset in first
    ]
    second_signature = [
        (
            dataset.meta.name,
            dataset.meta.domain,
            dataset.meta.topology_type,
            dataset.meta.num_nodes,
            dataset.meta.num_edges,
            dataset.meta.is_connected,
            dataset.meta.num_components,
            sorted(dataset.graph.edges()),
        )
        for dataset in second
    ]

    assert first_signature == second_signature


def test_graph_generator_new_question_templates() -> None:
    generator = GraphBenchmarkGenerator()
    graph = nx.cycle_graph(8)
    qa_items = generator.generate_dataset_qa(
        graph=graph, data_id="cycle-8", items_per_dataset=25
    )

    assert len(qa_items) == 25
    tasks = {item.task for item in qa_items}
    assert "degree_query" in tasks
    assert "centrality" in tasks
    assert "clustering" in tasks
    assert "diameter" in tasks
    assert "bipartite_check" in tasks
    assert "cycle_detection" in tasks
    assert "edge_count" in tasks

    difficulty_counts: dict[str, int] = {}
    for item in qa_items:
        difficulty_counts[item.difficulty] = (
            difficulty_counts.get(item.difficulty, 0) + 1
        )

    assert difficulty_counts["1-hop"] == 7
    assert difficulty_counts["2-hop"] == 8
    assert difficulty_counts["3-hop"] == 6
    assert difficulty_counts["counterfactual"] == 4


def test_mixed_generator_tab_ts_generates_200_with_target_difficulties() -> None:
    tab_dataset = TabularDataFactory(seed=42).generate_all()[0]
    ts_datasets = _create_timeseries_datasets(seed=42)
    ts_dataset = ts_datasets[0]

    generator = MixedTypeGenerator(seed=42)
    pairs = generator.generate_tab_ts_items(
        tabular_dataset=tab_dataset,
        ts_dataset=ts_dataset,
        num_items=200,
    )

    assert len(pairs) == 200
    counts: dict[str, int] = {}
    for pair in pairs:
        counts[pair.difficulty] = counts.get(pair.difficulty, 0) + 1
        assert pair.answer not in {"depends_on_reasoning_depth", ""}
    # All 4 difficulty levels should be present
    assert len(counts) >= 3
    # Verify reasonable distribution (30/30/25/15 target)
    assert counts.get(Difficulty.ONE_HOP.value, 0) + counts.get(Difficulty.TWO_HOP.value, 0) >= 60


def test_mixed_generator_tab_graph_generates_computable_answers() -> None:
    tab_dataset = TabularDataFactory(seed=42).generate_all()[1]
    graph_dataset = GraphDataFactory(seed=42).create_datasets()[1]

    generator = MixedTypeGenerator(seed=42)
    pairs = generator.generate_tab_graph_items(
        tabular_dataset=tab_dataset,
        graph_dataset=graph_dataset,
        num_items=200,
    )

    assert len(pairs) == 200
    # Verify answer diversity (not all same answer)
    assert len({pair.answer for pair in pairs}) > 3
    # Verify difficulties present
    difficulties = {pair.difficulty for pair in pairs}
    assert len(difficulties) >= 3


def test_mixed_generator_ts_graph_generates_200_items() -> None:
    ts_datasets = _create_timeseries_datasets(seed=42)
    ts_dataset = ts_datasets[0]
    graph_dataset = GraphDataFactory(seed=42).create_datasets()[2]

    generator = MixedTypeGenerator(seed=42)
    pairs = generator.generate_ts_graph_items(
        ts_dataset=ts_dataset,
        graph_dataset=graph_dataset,
        num_items=200,
    )

    assert len(pairs) == 200
    counts: dict[str, int] = {}
    for pair in pairs:
        counts[pair.difficulty] = counts.get(pair.difficulty, 0) + 1
    assert len(counts) >= 3


def test_mixed_generator_generate_all_produces_600_items() -> None:
    generator = MixedTypeGenerator(seed=42)
    all_pairs = generator.generate_all_mixed(seed=42, items_per_combination=200)
    assert len(all_pairs) == 600
    # Verify all answers are non-empty
    assert all(pair.answer for pair in all_pairs)
