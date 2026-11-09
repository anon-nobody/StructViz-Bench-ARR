from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportMissingTypeArgument=false

from dataclasses import dataclass

import networkx as nx

from src.generation.data_sources.graph_sources import GraphDataset, GraphMeta


@dataclass(slots=True)
class RealWorldGraphLoader:
    seed: int = 42

    def load_datasets(self) -> list[GraphDataset]:
        """Load real-world and topology-template graphs with clear provenance.

        Returns two categories:
        - **Real-world**: Classic social network datasets from NetworkX
          (Karate Club, Les Miserables, Florentine Families, Davis Southern Women).
        - **Topology-template**: Graphs generated from well-known topology models
          (Watts-Strogatz, Barabasi-Albert, Caveman, Random Geometric) with
          controlled parameters. These are NOT real-world but use standard
          graph-theory models as structured templates.

        Each dataset carries a ``source_category`` field in its meta to
        distinguish provenance.
        """
        datasets: list[GraphDataset] = []

        # ── Real-world graphs (published social network datasets) ────────
        datasets.append(
            self._wrap_graph(
                self._relabel_to_integers(nx.karate_club_graph()),
                "karate_club",
                "social_network",
                "social",
                source_category="real_world",
            )
        )
        datasets.append(
            self._wrap_graph(
                self._relabel_to_integers(nx.les_miserables_graph()),
                "les_miserables",
                "social_network",
                "social",
                source_category="real_world",
            )
        )
        datasets.append(
            self._wrap_graph(
                self._relabel_to_integers(nx.florentine_families_graph()),
                "florentine_families",
                "historical_social_network",
                "social",
                source_category="real_world",
            )
        )
        datasets.append(self._davis_women_projection())

        # ── Topology-template graphs (generated from standard models) ────
        generated_graphs: list[tuple[str, str, str, nx.Graph]] = [
            (
                "social_small_world_24",
                "social_network",
                "small_world",
                nx.watts_strogatz_graph(24, 4, 0.15, seed=self.seed + 1),
            ),
            (
                "social_small_world_40",
                "social_network",
                "small_world",
                nx.watts_strogatz_graph(40, 6, 0.10, seed=self.seed + 2),
            ),
            (
                "social_small_world_56",
                "social_network",
                "small_world",
                nx.watts_strogatz_graph(56, 6, 0.20, seed=self.seed + 3),
            ),
            (
                "citation_scale_free_28",
                "citation_network",
                "scale_free",
                nx.barabasi_albert_graph(28, 2, seed=self.seed + 4),
            ),
            (
                "citation_scale_free_52",
                "citation_network",
                "scale_free",
                nx.barabasi_albert_graph(52, 3, seed=self.seed + 5),
            ),
            (
                "citation_scale_free_76",
                "citation_network",
                "scale_free",
                nx.barabasi_albert_graph(76, 4, seed=self.seed + 6),
            ),
            (
                "community_caveman_30",
                "community_network",
                "community",
                nx.connected_caveman_graph(6, 5),
            ),
            (
                "community_caveman_48",
                "community_network",
                "community",
                nx.connected_caveman_graph(8, 6),
            ),
            (
                "community_caveman_70",
                "community_network",
                "community",
                nx.connected_caveman_graph(10, 7),
            ),
            (
                "proximity_geometric_26",
                "mobility_network",
                "geometric",
                nx.random_geometric_graph(26, 0.35, seed=self.seed + 7),
            ),
            (
                "proximity_geometric_44",
                "mobility_network",
                "geometric",
                nx.random_geometric_graph(44, 0.28, seed=self.seed + 8),
            ),
            (
                "proximity_geometric_68",
                "mobility_network",
                "geometric",
                nx.random_geometric_graph(68, 0.23, seed=self.seed + 9),
            ),
        ]

        for name, domain, topology, graph in generated_graphs:
            datasets.append(
                self._wrap_graph(
                    graph, name, domain, topology, source_category="topology_template"
                )
            )

        filtered = [
            dataset
            for dataset in datasets
            if 15 <= dataset.graph.number_of_nodes() <= 100
        ]
        return filtered

    def _davis_women_projection(self) -> GraphDataset:
        graph = nx.davis_southern_women_graph()
        women_nodes = [
            node
            for node, attrs in graph.nodes(data=True)
            if attrs.get("bipartite") == 0
        ]
        projected = nx.bipartite.weighted_projected_graph(graph, women_nodes)
        return self._wrap_graph(
            self._relabel_to_integers(nx.Graph(projected)),
            "davis_southern_women_projection",
            "social_network",
            "bipartite_projection",
            source_category="real_world",
        )

    def _relabel_to_integers(self, graph: nx.Graph) -> nx.Graph:
        return nx.convert_node_labels_to_integers(
            graph, label_attribute="original_label"
        )

    def _wrap_graph(
        self,
        graph: nx.Graph,
        name: str,
        domain: str,
        topology: str,
        source_category: str = "real_world",
    ) -> GraphDataset:
        return GraphDataset(
            graph=graph,
            meta=GraphMeta(
                name=name,
                domain=domain,
                topology_type=topology,
                num_nodes=graph.number_of_nodes(),
                num_edges=graph.number_of_edges(),
                is_connected=nx.is_connected(graph),
                num_components=nx.number_connected_components(graph),
                source_category=source_category,
            ),
        )
