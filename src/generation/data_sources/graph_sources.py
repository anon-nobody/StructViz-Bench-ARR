from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportMissingTypeArgument=false

from dataclasses import dataclass

import networkx as nx
from networkx.generators.trees import random_labeled_tree


@dataclass(slots=True)
class GraphMeta:
    """Metadata summary for one synthetic graph dataset.

    Args:
        name: Unique dataset name.
        domain: Synthetic domain label.
        topology_type: Generator topology family.
        num_nodes: Number of nodes.
        num_edges: Number of edges.
        is_connected: Whether all nodes are connected.
        num_components: Number of connected components.
    """

    name: str
    domain: str
    topology_type: str
    num_nodes: int
    num_edges: int
    is_connected: bool
    num_components: int
    source_category: str = "synthetic"


@dataclass(slots=True)
class GraphSpec:
    """Generation spec for a deterministic synthetic graph.

    Args:
        name: Unique dataset name.
        domain: Synthetic domain label.
        topology_type: Generator topology family.
        num_nodes: Target number of nodes.
        params: Topology-specific numeric parameters.
    """

    name: str
    domain: str
    topology_type: str
    num_nodes: int
    params: dict[str, object]


@dataclass(slots=True)
class GraphDataset:
    """Pair of graph object and metadata.

    Args:
        graph: Generated networkx graph.
        meta: Associated metadata.
    """

    graph: nx.Graph
    meta: GraphMeta


@dataclass(slots=True)
class GraphDataFactory:
    """Factory for deterministic synthetic graph datasets.

    Args:
        seed: Global deterministic seed.
    """

    seed: int = 42

    def create_datasets(self) -> list[GraphDataset]:
        """Generate the canonical 20 graph datasets.

        Returns:
            A list of 20 synthetic graph datasets.
        """

        datasets: list[GraphDataset] = []
        for index, spec in enumerate(self._build_specs()):
            graph_seed = self.seed + index * 997
            graph = self._build_graph(spec=spec, graph_seed=graph_seed)
            meta = self._build_meta(spec=spec, graph=graph)
            datasets.append(GraphDataset(graph=graph, meta=meta))
        return datasets

    def _build_specs(self) -> list[GraphSpec]:
        return [
            GraphSpec(
                "citation_network_small", "scholarly", "erdos_renyi", 24, {"p": 0.12}
            ),
            GraphSpec(
                "trade_network_dense", "economics", "erdos_renyi", 42, {"p": 0.20}
            ),
            GraphSpec(
                "sensor_reliability_sparse", "iot", "erdos_renyi", 68, {"p": 0.06}
            ),
            GraphSpec(
                "social_influencer_core", "social", "barabasi_albert", 38, {"m": 2}
            ),
            GraphSpec(
                "protein_interaction_scale",
                "bioinformatics",
                "barabasi_albert",
                75,
                {"m": 3},
            ),
            GraphSpec(
                "api_dependency_graph", "software", "barabasi_albert", 56, {"m": 4}
            ),
            GraphSpec(
                "regional_transport_small_world",
                "mobility",
                "watts_strogatz",
                60,
                {"k": 4, "p": 0.15},
            ),
            GraphSpec(
                "neighborhood_contacts_small_world",
                "public_health",
                "watts_strogatz",
                84,
                {"k": 6, "p": 0.08},
            ),
            GraphSpec(
                "warehouse_robot_links",
                "robotics",
                "watts_strogatz",
                48,
                {"k": 4, "p": 0.25},
            ),
            GraphSpec("energy_distribution_tree", "utilities", "tree", 45, {}),
            GraphSpec("file_system_hierarchy", "storage", "tree", 96, {}),
            GraphSpec("city_blocks_grid", "urban", "grid", 64, {"rows": 8, "cols": 8}),
            GraphSpec(
                "factory_floor_grid",
                "manufacturing",
                "grid",
                50,
                {"rows": 5, "cols": 10},
            ),
            GraphSpec("broadcast_hub_star", "telecom", "star", 32, {}),
            GraphSpec("airline_hub_star", "aviation", "star", 71, {}),
            GraphSpec(
                "research_groups_community",
                "academia",
                "community",
                72,
                {"sizes": [24, 24, 24], "p_in": 0.30, "p_out": 0.02},
            ),
            GraphSpec(
                "market_segments_community",
                "commerce",
                "community",
                90,
                {"sizes": [30, 30, 30], "p_in": 0.22, "p_out": 0.03},
            ),
            GraphSpec(
                "campus_clubs_community",
                "education",
                "community",
                54,
                {"sizes": [18, 18, 18], "p_in": 0.35, "p_out": 0.05},
            ),
            GraphSpec(
                "authors_papers_bipartite",
                "publishing",
                "bipartite",
                58,
                {"left_size": 22, "right_size": 36, "p": 0.16},
            ),
            GraphSpec(
                "users_items_bipartite",
                "recommender",
                "bipartite",
                80,
                {"left_size": 30, "right_size": 50, "p": 0.10},
            ),
        ]

    def _get_float(self, params: dict[str, object], key: str) -> float:
        value = params.get(key)
        if not isinstance(value, (int, float)):
            raise ValueError(f"Expected numeric parameter '{key}'.")
        return float(value)

    def _get_int(self, params: dict[str, object], key: str) -> int:
        value = params.get(key)
        if not isinstance(value, int):
            raise ValueError(f"Expected integer parameter '{key}'.")
        return value

    def _get_int_list(self, params: dict[str, object], key: str) -> list[int]:
        value = params.get(key)
        if not isinstance(value, list) or not all(
            isinstance(item, int) for item in value
        ):
            raise ValueError(f"Expected list[int] parameter '{key}'.")
        return value

    def _build_graph(self, spec: GraphSpec, graph_seed: int) -> nx.Graph:
        topology = spec.topology_type
        if topology == "erdos_renyi":
            graph = nx.erdos_renyi_graph(
                spec.num_nodes, self._get_float(spec.params, "p"), seed=graph_seed
            )
        elif topology == "barabasi_albert":
            graph = nx.barabasi_albert_graph(
                spec.num_nodes, self._get_int(spec.params, "m"), seed=graph_seed
            )
        elif topology == "watts_strogatz":
            graph = nx.watts_strogatz_graph(
                spec.num_nodes,
                self._get_int(spec.params, "k"),
                self._get_float(spec.params, "p"),
                seed=graph_seed,
            )
        elif topology == "tree":
            graph = random_labeled_tree(spec.num_nodes, seed=graph_seed)
        elif topology == "grid":
            graph = nx.grid_2d_graph(
                self._get_int(spec.params, "rows"),
                self._get_int(spec.params, "cols"),
            )
            graph = nx.convert_node_labels_to_integers(
                graph, first_label=0, ordering="default"
            )
        elif topology == "star":
            graph = nx.star_graph(spec.num_nodes - 1)
        elif topology == "community":
            sizes = self._get_int_list(spec.params, "sizes")
            p_in = self._get_float(spec.params, "p_in")
            p_out = self._get_float(spec.params, "p_out")
            graph = nx.stochastic_block_model(
                sizes=sizes,
                p=[
                    [p_in if i == j else p_out for j in range(len(sizes))]
                    for i in range(len(sizes))
                ],
                seed=graph_seed,
            )
            graph = nx.Graph(graph)
        elif topology == "bipartite":
            graph = nx.bipartite.random_graph(
                self._get_int(spec.params, "left_size"),
                self._get_int(spec.params, "right_size"),
                self._get_float(spec.params, "p"),
                seed=graph_seed,
            )
            graph = nx.convert_node_labels_to_integers(
                graph, first_label=0, ordering="default"
            )
        else:
            raise ValueError(f"Unsupported topology type: {topology}")

        return graph

    def _build_meta(self, spec: GraphSpec, graph: nx.Graph) -> GraphMeta:
        num_components = nx.number_connected_components(graph)
        return GraphMeta(
            name=spec.name,
            domain=spec.domain,
            topology_type=spec.topology_type,
            num_nodes=graph.number_of_nodes(),
            num_edges=graph.number_of_edges(),
            is_connected=nx.is_connected(graph),
            num_components=num_components,
        )
