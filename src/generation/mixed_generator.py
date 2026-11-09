from __future__ import annotations
# pyright: reportMissingImports=false, reportMissingModuleSource=false

from collections.abc import Callable
from dataclasses import dataclass, field
import random
from typing import Any, cast

import networkx as nx
import numpy as np
import pandas as pd

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
from src.generation.difficulty_classifier import Difficulty, DifficultyClassifier


@dataclass(slots=True)
class _Template:
    task: str
    reasoning_steps: int
    requires_arithmetic: bool
    counterfactual: bool
    render: Callable[[int], tuple[str, str]]


@dataclass(slots=True)
class MixedBenchmarkGenerator:
    seed: int = 42
    classifier: DifficultyClassifier = field(default_factory=DifficultyClassifier)

    def generate_tab_ts_items(
        self,
        tabular_dataset: TabularDataset,
        ts_dataset: TimeSeriesDataset,
        num_items: int = 200,
    ) -> list[QAPair]:
        """Generate table+time-series cross-modal QA items.

        Args:
            tabular_dataset: Tabular dataset from `TabularDataFactory`.
            ts_dataset: Time-series dataset from `TimeSeriesDataFactory`.
            num_items: Number of QA pairs to generate.

        Returns:
            Deterministic QA pairs with 30/30/25/15 difficulty distribution.
        """

        c = self._tab_ts_context(tabular_dataset, ts_dataset)
        return self._emit_items("tab_ts", self._tab_ts_templates(c), num_items, 101)

    def generate_tab_graph_items(
        self,
        tabular_dataset: TabularDataset,
        graph_dataset: GraphDataset,
        num_items: int = 200,
    ) -> list[QAPair]:
        """Generate table+graph cross-modal QA items.

        Args:
            tabular_dataset: Tabular dataset from `TabularDataFactory`.
            graph_dataset: Graph dataset from `GraphDataFactory`.
            num_items: Number of QA pairs to generate.

        Returns:
            Deterministic QA pairs with 30/30/25/15 difficulty distribution.
        """

        c = self._tab_graph_context(tabular_dataset, graph_dataset)
        return self._emit_items(
            "tab_graph", self._tab_graph_templates(c), num_items, 211
        )

    def generate_ts_graph_items(
        self,
        ts_dataset: TimeSeriesDataset,
        graph_dataset: GraphDataset,
        num_items: int = 200,
    ) -> list[QAPair]:
        """Generate time-series+graph cross-modal QA items.

        Args:
            ts_dataset: Time-series dataset from `TimeSeriesDataFactory`.
            graph_dataset: Graph dataset from `GraphDataFactory`.
            num_items: Number of QA pairs to generate.

        Returns:
            Deterministic QA pairs with 30/30/25/15 difficulty distribution.
        """

        c = self._ts_graph_context(ts_dataset, graph_dataset)
        return self._emit_items("ts_graph", self._ts_graph_templates(c), num_items, 307)

    def generate_all_mixed(
        self, seed: int = 42, items_per_combination: int = 200
    ) -> list[QAPair]:
        """Generate all mixed combinations: tab_ts, tab_graph, ts_graph.

        Args:
            seed: Seed used for all source factories.
            items_per_combination: Number of items per combination.

        Returns:
            Combined list containing exactly `3 * items_per_combination` items.
        """

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
        return (
            self.generate_tab_ts_items(tab, ts, items_per_combination)
            + self.generate_tab_graph_items(tab, graph, items_per_combination)
            + self.generate_ts_graph_items(ts, graph, items_per_combination)
        )

    def _tab_ts_context(
        self, tab_ds: TabularDataset, ts_ds: TimeSeriesDataset
    ) -> dict[str, Any]:
        col = self._first_numeric_column(tab_ds.dataframe)
        table = (
            pd.to_numeric(tab_ds.dataframe[col], errors="coerce")
            .dropna()
            .to_numpy(float)
        )
        ts = np.asarray(ts_ds.values, dtype=float)
        aligned = np.interp(
            np.linspace(0, len(ts) - 1, len(table)), np.arange(len(ts)), ts
        )
        table_mean = float(np.mean(table))
        ts_mean = float(np.mean(ts))
        return {
            "col": col,
            "table": table,
            "ts": ts,
            "table_mean": table_mean,
            "ts_mean": ts_mean,
            "table_max": float(np.max(table)),
            "ts_peak": float(np.max(ts)),
            "table_std": float(np.std(table)),
            "ts_std": float(np.std(ts)),
            "table_range": float(np.ptp(table)),
            "ts_range": float(np.ptp(ts)),
            "table_trend": "increasing" if table[-1] >= table[0] else "decreasing",
            "ts_trend": "increasing" if ts[-1] >= ts[0] else "decreasing",
            "table_above_mean": int(np.sum(table > table_mean)),
            "ts_above_mean": int(np.sum(ts > ts_mean)),
            "corr": self._safe_corr(table, aligned),
            "ts_anoms": int(len(ts_ds.metadata.known_anomaly_indices)),
        }

    def _tab_graph_context(
        self, tab_ds: TabularDataset, graph_ds: GraphDataset
    ) -> dict[str, Any]:
        col = self._first_numeric_column(tab_ds.dataframe)
        table = (
            pd.to_numeric(tab_ds.dataframe[col], errors="coerce")
            .dropna()
            .to_numpy(float)
        )
        graph = graph_ds.graph
        nodes = sorted(int(node) for node in graph.nodes())
        mapped = {node: float(table[i % len(table)]) for i, node in enumerate(nodes)}
        degrees = dict(graph.degree())
        edge_gaps = [abs(mapped[int(u)] - mapped[int(v)]) for u, v in graph.edges()]
        max_deg_node = max(nodes, key=lambda n: (degrees[n], -n))
        max_val_node = max(nodes, key=lambda n: (mapped[n], -n))
        mean_degree = float(
            (2 * graph.number_of_edges()) / max(graph.number_of_nodes(), 1)
        )
        return {
            "col": col,
            "table_mean": float(np.mean(table)),
            "table_median": float(np.median(table)),
            "table_std": float(np.std(table)),
            "max_deg_node": int(max_deg_node),
            "max_val_node": int(max_val_node),
            "max_edge_gap": float(max(edge_gaps) if edge_gaps else 0.0),
            "density": float(nx.density(graph)),
            "num_nodes": int(graph.number_of_nodes()),
            "num_edges": int(graph.number_of_edges()),
            "num_components": int(nx.number_connected_components(graph)),
            "mean_degree": mean_degree,
            "deg_gt_table_median": int(
                np.sum(
                    np.array([degrees[n] for n in nodes], dtype=float)
                    > np.median(table)
                )
            ),
        }

    def _ts_graph_context(
        self, ts_ds: TimeSeriesDataset, graph_ds: GraphDataset
    ) -> dict[str, Any]:
        ts = np.asarray(ts_ds.values, dtype=float)
        graph = graph_ds.graph
        degrees = np.asarray([d for _, d in graph.degree()], dtype=float)
        vol = float(np.std(np.diff(ts))) if len(ts) > 1 else 0.0
        anoms = int(len(ts_ds.metadata.known_anomaly_indices))
        if anoms == 0:
            z = (ts - np.mean(ts)) / max(float(np.std(ts)), 1e-8)
            anoms = int(np.sum(np.abs(z) > 2.0))
        return {
            "ts_mean": float(np.mean(ts)),
            "ts_peak": float(np.max(ts)),
            "ts_median": float(np.median(ts)),
            "ts_std": float(np.std(ts)),
            "ts_vol": vol,
            "anoms": anoms,
            "density": float(nx.density(graph)),
            "num_nodes": int(graph.number_of_nodes()),
            "num_edges": int(graph.number_of_edges()),
            "num_components": int(nx.number_connected_components(graph)),
            "diameter": self._largest_component_diameter(graph),
            "max_degree": int(np.max(degrees)) if len(degrees) else 0,
            "avg_degree": float(np.mean(degrees)) if len(degrees) else 0.0,
        }

    def _tab_ts_templates(self, c: dict[str, Any]) -> list[_Template]:
        t = lambda task, steps, arith, cf, fn: _Template(task, steps, arith, cf, fn)
        return [
            t(
                "cross_modal_comparison",
                1,
                False,
                False,
                lambda _: (
                    f"Is the trend in table column '{c['col']}' consistent with the time-series trend?",
                    "yes" if c["table_trend"] == c["ts_trend"] else "no",
                ),
            ),
            t(
                "cross_modal_threshold",
                1,
                False,
                False,
                lambda _: (
                    f"Does max('{c['col']}') exceed the series peak?",
                    "yes" if c["table_max"] > c["ts_peak"] else "no",
                ),
            ),
            t(
                "cross_modal_comparison",
                1,
                False,
                False,
                lambda _: (
                    "Which has larger range: table column or time series?",
                    "table" if c["table_range"] >= c["ts_range"] else "time_series",
                ),
            ),
            t(
                "cross_modal_threshold",
                2,
                True,
                False,
                lambda _: (
                    f"Are counts above mean larger in '{c['col']}' than in the series?",
                    "yes" if c["table_above_mean"] > c["ts_above_mean"] else "no",
                ),
            ),
            t(
                "cross_modal_alignment",
                2,
                True,
                False,
                lambda _: (
                    "Is aligned table-series correlation non-negative?",
                    "yes" if c["corr"] >= 0 else "no",
                ),
            ),
            t(
                "cross_modal_aggregation",
                2,
                True,
                False,
                lambda _: (
                    "What is |table mean - series mean|?",
                    f"{abs(c['table_mean'] - c['ts_mean']):.3f}",
                ),
            ),
            t(
                "cross_modal_comparison",
                3,
                True,
                False,
                lambda _: (
                    "Is table std greater than series std while table range is also greater?",
                    "yes"
                    if (
                        c["table_std"] > c["ts_std"]
                        and c["table_range"] > c["ts_range"]
                    )
                    else "no",
                ),
            ),
            t(
                "cross_modal_aggregation",
                3,
                True,
                False,
                lambda _: (
                    "Is (table mean / series mean) greater than 1?",
                    "yes" if c["table_mean"] / max(c["ts_mean"], 1e-8) > 1 else "no",
                ),
            ),
            t(
                "cross_modal_counterfactual",
                4,
                True,
                True,
                lambda _: (
                    "If the series had one extra anomaly, would anomalies exceed table values above mean?",
                    "yes" if (c["ts_anoms"] + 1) > c["table_above_mean"] else "no",
                ),
            ),
            t(
                "cross_modal_counterfactual",
                4,
                True,
                True,
                lambda _: (
                    f"If '{c['col']}' increased by 10%, would its mean exceed the current series peak?",
                    "yes" if (c["table_mean"] * 1.1) > c["ts_peak"] else "no",
                ),
            ),
        ]

    def _tab_graph_templates(self, c: dict[str, Any]) -> list[_Template]:
        t = lambda task, steps, arith, cf, fn: _Template(task, steps, arith, cf, fn)
        return [
            t(
                "cross_modal_comparison",
                1,
                False,
                False,
                lambda _: (
                    f"Does the highest-degree node match the max mapped '{c['col']}' node?",
                    "yes" if c["max_deg_node"] == c["max_val_node"] else "no",
                ),
            ),
            t(
                "cross_modal_threshold",
                1,
                False,
                False,
                lambda _: (
                    f"How many nodes have degree greater than median('{c['col']}')?",
                    str(c["deg_gt_table_median"]),
                ),
            ),
            t(
                "cross_modal_comparison",
                1,
                False,
                False,
                lambda _: (
                    "Is graph density > table_std/100?",
                    "yes" if c["density"] > (c["table_std"] / 100.0) else "no",
                ),
            ),
            t(
                "cross_modal_aggregation",
                2,
                True,
                False,
                lambda _: (
                    "What is |mean_degree - table_mean|?",
                    f"{abs(c['mean_degree'] - c['table_mean']):.3f}",
                ),
            ),
            t(
                "cross_modal_threshold",
                2,
                True,
                False,
                lambda _: (
                    "Is number of components greater than table_mean/10?",
                    "yes" if c["num_components"] > (c["table_mean"] / 10.0) else "no",
                ),
            ),
            t(
                "cross_modal_comparison",
                2,
                True,
                False,
                lambda _: (
                    "Is max mapped edge gap greater than table std?",
                    "yes" if c["max_edge_gap"] > c["table_std"] else "no",
                ),
            ),
            t(
                "cross_modal_comparison",
                3,
                True,
                False,
                lambda _: (
                    "Which is larger: graph density*100 or table mean?",
                    "graph_density_scaled"
                    if (c["density"] * 100.0) > c["table_mean"]
                    else "table_mean",
                ),
            ),
            t(
                "cross_modal_aggregation",
                3,
                True,
                False,
                lambda _: (
                    "Is (edges + nodes) greater than rounded table mean?",
                    "yes"
                    if (c["num_edges"] + c["num_nodes"]) > round(c["table_mean"])
                    else "no",
                ),
            ),
            t(
                "cross_modal_counterfactual",
                4,
                True,
                True,
                lambda _: (
                    f"If the graph had one more edge, would density exceed average('{c['col']}')/100?",
                    "yes"
                    if self._density_with_extra_edge(c["num_nodes"], c["num_edges"], 1)
                    > (c["table_mean"] / 100.0)
                    else "no",
                ),
            ),
            t(
                "cross_modal_counterfactual",
                4,
                True,
                True,
                lambda _: (
                    f"If '{c['col']}' dropped by 20%, would max mapped edge gap stay above mean degree?",
                    "yes" if (c["max_edge_gap"] * 0.8) > c["mean_degree"] else "no",
                ),
            ),
        ]

    def _ts_graph_templates(self, c: dict[str, Any]) -> list[_Template]:
        t = lambda task, steps, arith, cf, fn: _Template(task, steps, arith, cf, fn)
        return [
            t(
                "cross_modal_comparison",
                1,
                False,
                False,
                lambda _: (
                    "Is number of graph components greater than number of series anomalies?",
                    "yes" if c["num_components"] > c["anoms"] else "no",
                ),
            ),
            t(
                "cross_modal_threshold",
                1,
                False,
                False,
                lambda _: (
                    "Does graph max degree exceed series standard deviation?",
                    "yes" if c["max_degree"] > c["ts_std"] else "no",
                ),
            ),
            t(
                "cross_modal_comparison",
                1,
                False,
                False,
                lambda _: (
                    "Is graph density > 0.1 and series volatility > 1.0?",
                    "yes" if (c["density"] > 0.1 and c["ts_vol"] > 1.0) else "no",
                ),
            ),
            t(
                "cross_modal_aggregation",
                2,
                True,
                False,
                lambda _: (
                    "What is |avg_degree - series_mean/10|?",
                    f"{abs(c['avg_degree'] - (c['ts_mean'] / 10.0)):.3f}",
                ),
            ),
            t(
                "cross_modal_threshold",
                2,
                True,
                False,
                lambda _: (
                    "Is graph diameter larger than anomaly count?",
                    "yes" if c["diameter"] > c["anoms"] else "no",
                ),
            ),
            t(
                "cross_modal_comparison",
                2,
                True,
                False,
                lambda _: (
                    "Which is larger: series peak or graph edge count?",
                    "series_peak" if c["ts_peak"] > c["num_edges"] else "graph_edges",
                ),
            ),
            t(
                "cross_modal_comparison",
                3,
                True,
                False,
                lambda _: (
                    "Does density correlate with lower volatility under rule density*10 > volatility?",
                    "yes" if (c["density"] * 10.0) > c["ts_vol"] else "no",
                ),
            ),
            t(
                "cross_modal_aggregation",
                3,
                True,
                False,
                lambda _: (
                    "Is (graph nodes + components) greater than rounded series median?",
                    "yes"
                    if (c["num_nodes"] + c["num_components"]) > round(c["ts_median"])
                    else "no",
                ),
            ),
            t(
                "cross_modal_counterfactual",
                4,
                True,
                True,
                lambda _: (
                    "If graph had one more edge, would density exceed series_mean/100?",
                    "yes"
                    if self._density_with_extra_edge(c["num_nodes"], c["num_edges"], 1)
                    > (c["ts_mean"] / 100.0)
                    else "no",
                ),
            ),
            t(
                "cross_modal_counterfactual",
                4,
                True,
                True,
                lambda _: (
                    "If one anomaly were added, would anomaly count exceed graph components?",
                    "yes" if (c["anoms"] + 1) > c["num_components"] else "no",
                ),
            ),
        ]

    def _emit_items(
        self,
        prefix: str,
        templates: list[_Template],
        num_items: int,
        seed_offset: int,
    ) -> list[QAPair]:
        if num_items <= 0:
            return []
        targets = self._difficulty_targets(num_items)
        buckets: dict[Difficulty, list[_Template]] = {
            Difficulty.ONE_HOP: [],
            Difficulty.TWO_HOP: [],
            Difficulty.THREE_HOP: [],
            Difficulty.COUNTERFACTUAL: [],
        }
        for template in templates:
            diff = self.classifier.classify(
                reasoning_steps=template.reasoning_steps,
                requires_arithmetic=template.requires_arithmetic,
                has_counterfactual=template.counterfactual,
            )
            buckets[diff].append(template)

        pairs: list[QAPair] = []
        ordered = [
            Difficulty.ONE_HOP,
            Difficulty.TWO_HOP,
            Difficulty.THREE_HOP,
            Difficulty.COUNTERFACTUAL,
        ]
        for diff in ordered:
            for i in range(targets[diff]):
                template = buckets[diff][i % len(buckets[diff])]
                q, a = template.render(i)
                pairs.append(
                    QAPair(
                        question=q,
                        answer=str(a),
                        difficulty=diff.value,
                        data_id="",
                        task=template.task,
                    )
                )

        rng = random.Random(self.seed + seed_offset + num_items)
        rng.shuffle(pairs)
        return [
            QAPair(
                pair.question,
                pair.answer,
                pair.difficulty,
                f"mixed-{prefix}-{idx}",
                pair.task,
            )
            for idx, pair in enumerate(pairs[:num_items])
        ]

    def _difficulty_targets(self, num_items: int) -> dict[Difficulty, int]:
        one = int(round(num_items * 0.30))
        two = int(round(num_items * 0.30))
        three = int(round(num_items * 0.25))
        return {
            Difficulty.ONE_HOP: one,
            Difficulty.TWO_HOP: two,
            Difficulty.THREE_HOP: three,
            Difficulty.COUNTERFACTUAL: num_items - one - two - three,
        }

    def _first_numeric_column(self, frame: pd.DataFrame) -> str:
        cols = [
            col
            for col in frame.select_dtypes(include="number").columns
            if "id" not in str(col).lower()
        ]
        if not cols:
            raise ValueError("Tabular dataset requires at least one numeric column.")
        return str(cols[0])

    def _safe_corr(self, left: np.ndarray, right: np.ndarray) -> float:
        if len(left) < 2 or len(right) < 2:
            return 0.0
        if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
            return 0.0
        return float(np.corrcoef(left, right)[0, 1])

    def _largest_component_diameter(self, graph: nx.Graph[Any]) -> int:
        if graph.number_of_nodes() <= 1:
            return 0
        component = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(component)
        if subgraph.number_of_nodes() <= 1:
            return 0
        return int(nx.diameter(subgraph))

    def _density_with_extra_edge(
        self, nodes: int, edges: int, extra_edges: int
    ) -> float:
        if nodes <= 1:
            return 0.0
        max_edges = nodes * (nodes - 1) / 2.0
        return min((edges + extra_edges) / max_edges, 1.0)


MixedTypeGenerator = MixedBenchmarkGenerator
