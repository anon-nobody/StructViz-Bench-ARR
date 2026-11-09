from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportMissingTypeArgument=false

from dataclasses import dataclass
from statistics import mean

import networkx as nx

from src.generation import QAPair
from src.generation.difficulty_classifier import Difficulty, DifficultyClassifier


@dataclass(slots=True)
class GraphBenchmarkGenerator:
    """Generate QA items for graph reasoning tasks."""

    classifier: DifficultyClassifier

    def __init__(self) -> None:
        """Initialize graph generator with default classifier."""
        self.classifier = DifficultyClassifier()

    def _difficulty_targets(self, items_per_dataset: int) -> dict[str, int]:
        fractions = {
            Difficulty.ONE_HOP.value: 0.30,
            Difficulty.TWO_HOP.value: 0.30,
            Difficulty.THREE_HOP.value: 0.25,
            Difficulty.COUNTERFACTUAL.value: 0.15,
        }
        base = {key: int(items_per_dataset * value) for key, value in fractions.items()}
        remainder = items_per_dataset - sum(base.values())
        remainders = sorted(
            (
                (items_per_dataset * fraction) - base[key],
                key,
            )
            for key, fraction in fractions.items()
        )
        if remainder > 0:
            for _, key in reversed(remainders[-remainder:]):
                base[key] += 1
        return base

    def generate_degree_query(self, graph: nx.Graph, data_id: str) -> list[QAPair]:
        nodes = sorted(graph.nodes)
        if not nodes:
            return []

        pivot_nodes = [nodes[0], nodes[len(nodes) // 2], nodes[-1]]
        highest_degree_node = max(
            nodes, key=lambda node: (graph.degree(node), -int(node))
        )

        pairs = [
            QAPair(
                question=f"What is the degree of node {node}?",
                answer=str(int(graph.degree(node))),
                difficulty=Difficulty.ONE_HOP.value,
                data_id=data_id,
                task="degree_query",
            )
            for node in pivot_nodes
        ]
        pairs.append(
            QAPair(
                question="Which node has the highest degree?",
                answer=str(int(highest_degree_node)),
                difficulty=Difficulty.TWO_HOP.value,
                data_id=data_id,
                task="degree_query",
            )
        )
        return pairs

    def generate_centrality(self, graph: nx.Graph, data_id: str) -> list[QAPair]:
        centrality = nx.betweenness_centrality(graph, normalized=True)
        node = max(sorted(graph.nodes), key=lambda n: (centrality[n], -int(n)))
        return [
            QAPair(
                question="Which node has the highest betweenness centrality?",
                answer=str(int(node)),
                difficulty=Difficulty.THREE_HOP.value,
                data_id=data_id,
                task="centrality",
            )
        ]

    def generate_clustering(self, graph: nx.Graph, data_id: str) -> list[QAPair]:
        coefficient = round(float(nx.average_clustering(graph)), 3)
        return [
            QAPair(
                question="What is the approximate clustering coefficient of the graph?",
                answer=f"{coefficient:.3f}",
                difficulty=Difficulty.TWO_HOP.value,
                data_id=data_id,
                task="clustering",
            )
        ]

    def generate_diameter(self, graph: nx.Graph, data_id: str) -> list[QAPair]:
        components = list(nx.connected_components(graph))
        largest_component_nodes = max(components, key=len)
        largest_component = graph.subgraph(largest_component_nodes).copy()
        diameter = (
            nx.diameter(largest_component)
            if largest_component.number_of_nodes() > 1
            else 0
        )
        return [
            QAPair(
                question="What is the diameter of the graph (or largest component)?",
                answer=str(int(diameter)),
                difficulty=Difficulty.THREE_HOP.value,
                data_id=data_id,
                task="diameter",
            )
        ]

    def generate_bipartite_check(self, graph: nx.Graph, data_id: str) -> list[QAPair]:
        return [
            QAPair(
                question="Is this graph bipartite?",
                answer="yes" if nx.is_bipartite(graph) else "no",
                difficulty=Difficulty.TWO_HOP.value,
                data_id=data_id,
                task="bipartite_check",
            )
        ]

    def generate_cycle_detection(self, graph: nx.Graph, data_id: str) -> list[QAPair]:
        has_cycle = any(nx.cycle_basis(graph))
        return [
            QAPair(
                question="Does this graph contain a cycle?",
                answer="yes" if has_cycle else "no",
                difficulty=Difficulty.TWO_HOP.value,
                data_id=data_id,
                task="cycle_detection",
            )
        ]

    def generate_edge_count(self, graph: nx.Graph, data_id: str) -> list[QAPair]:
        return [
            QAPair(
                question="How many edges does this graph have?",
                answer=str(graph.number_of_edges()),
                difficulty=Difficulty.ONE_HOP.value,
                data_id=data_id,
                task="edge_count",
            )
        ]

    def generate_counterfactual(self, graph: nx.Graph, data_id: str) -> list[QAPair]:
        nodes = sorted(graph.nodes)
        if len(nodes) < 2:
            return []

        u = nodes[0]
        v = nodes[-1]
        existing_edge = graph.has_edge(u, v)
        components_now = nx.number_connected_components(graph)
        components_if_removed = components_now
        if graph.has_edge(u, v):
            graph_removed = graph.copy()
            graph_removed.remove_edge(u, v)
            components_if_removed = nx.number_connected_components(graph_removed)

        return [
            QAPair(
                question=f"If an edge were added between node {u} and node {v}, would they be directly connected?",
                answer="yes",
                difficulty=Difficulty.COUNTERFACTUAL.value,
                data_id=data_id,
                task="counterfactual",
            ),
            QAPair(
                question=(
                    f"If the edge between node {u} and node {v} were removed, would the number "
                    "of connected components increase?"
                ),
                answer="yes"
                if existing_edge and components_if_removed > components_now
                else "no",
                difficulty=Difficulty.COUNTERFACTUAL.value,
                data_id=data_id,
                task="counterfactual",
            ),
            QAPair(
                question="If one random edge were removed, would the total edge count decrease by one?",
                answer="yes" if graph.number_of_edges() > 0 else "no",
                difficulty=Difficulty.COUNTERFACTUAL.value,
                data_id=data_id,
                task="counterfactual",
            ),
        ]

    def generate_dataset_qa(
        self,
        graph: nx.Graph,
        data_id: str,
        items_per_dataset: int = 25,
    ) -> list[QAPair]:
        base_connectivity = self.generate_connectivity(graph)
        base_shortest = self.generate_shortest_path(graph)
        base_community = self.generate_community(graph)

        candidate_pairs: list[QAPair] = []
        candidate_pairs.extend(
            [
                QAPair(
                    question=item.question,
                    answer=item.answer,
                    difficulty=item.difficulty,
                    data_id=data_id,
                    task=item.task,
                )
                for item in base_connectivity + base_shortest + base_community
            ]
        )
        candidate_pairs.extend(self.generate_degree_query(graph, data_id=data_id))
        candidate_pairs.extend(self.generate_centrality(graph, data_id=data_id))
        candidate_pairs.extend(self.generate_clustering(graph, data_id=data_id))
        candidate_pairs.extend(self.generate_diameter(graph, data_id=data_id))
        candidate_pairs.extend(self.generate_bipartite_check(graph, data_id=data_id))
        candidate_pairs.extend(self.generate_cycle_detection(graph, data_id=data_id))
        candidate_pairs.extend(self.generate_edge_count(graph, data_id=data_id))
        candidate_pairs.extend(self.generate_counterfactual(graph, data_id=data_id))

        nodes = sorted(graph.nodes)
        top_degrees = sorted(
            ((node, graph.degree(node)) for node in nodes),
            key=lambda item: (-item[1], item[0]),
        )
        for node, degree in top_degrees[: min(6, len(top_degrees))]:
            candidate_pairs.append(
                QAPair(
                    question=f"How many immediate neighbors does node {node} have?",
                    answer=str(int(degree)),
                    difficulty=Difficulty.ONE_HOP.value,
                    data_id=data_id,
                    task="degree_query",
                )
            )

        for node in nodes[1 : min(len(nodes), 6)]:
            candidate_pairs.append(
                QAPair(
                    question=f"Is there a path between node {nodes[0]} and node {node}?",
                    answer="yes" if nx.has_path(graph, nodes[0], node) else "no",
                    difficulty=Difficulty.TWO_HOP.value,
                    data_id=data_id,
                    task="connectivity",
                )
            )

        if len(nodes) >= 3:
            candidate_pairs.append(
                QAPair(
                    question=(
                        f"What is the average degree among node {nodes[0]}, node {nodes[len(nodes) // 2]}, "
                        f"and node {nodes[-1]}?"
                    ),
                    answer=f"{mean([graph.degree(nodes[0]), graph.degree(nodes[len(nodes) // 2]), graph.degree(nodes[-1])]):.3f}",
                    difficulty=Difficulty.THREE_HOP.value,
                    data_id=data_id,
                    task="degree_query",
                )
            )

        largest_component_size = max(
            len(component) for component in nx.connected_components(graph)
        )
        candidate_pairs.append(
            QAPair(
                question="How many nodes are in the largest connected component?",
                answer=str(largest_component_size),
                difficulty=Difficulty.THREE_HOP.value,
                data_id=data_id,
                task="community",
            )
        )
        candidate_pairs.append(
            QAPair(
                question="What is the average degree of the graph?",
                answer=f"{((2 * graph.number_of_edges()) / max(graph.number_of_nodes(), 1)):.3f}",
                difficulty=Difficulty.THREE_HOP.value,
                data_id=data_id,
                task="degree_query",
            )
        )
        candidate_pairs.append(
            QAPair(
                question="How many nodes have degree 1?",
                answer=str(sum(1 for _, degree in graph.degree() if degree == 1)),
                difficulty=Difficulty.THREE_HOP.value,
                data_id=data_id,
                task="degree_query",
            )
        )
        candidate_pairs.append(
            QAPair(
                question="What is the graph density?",
                answer=f"{nx.density(graph):.3f}",
                difficulty=Difficulty.THREE_HOP.value,
                data_id=data_id,
                task="edge_count",
            )
        )

        if len(nodes) >= 2:
            start = nodes[0]
            end = nodes[-1]
            try:
                shortest = nx.shortest_path_length(graph, source=start, target=end)
                candidate_pairs.append(
                    QAPair(
                        question=(
                            f"If one new edge were added between node {start} and node {end}, "
                            "would the shortest path between them become 1?"
                        ),
                        answer="yes",
                        difficulty=Difficulty.COUNTERFACTUAL.value,
                        data_id=data_id,
                        task="shortest_path",
                    )
                )
                candidate_pairs.append(
                    QAPair(
                        question=(
                            f"Is the current shortest path between node {start} and node {end} greater than 1?"
                        ),
                        answer="yes" if shortest > 1 else "no",
                        difficulty=Difficulty.TWO_HOP.value,
                        data_id=data_id,
                        task="shortest_path",
                    )
                )
            except nx.NetworkXNoPath:
                candidate_pairs.append(
                    QAPair(
                        question=(
                            f"If one new edge were added between node {start} and node {end}, "
                            "would they become connected by a path?"
                        ),
                        answer="yes",
                        difficulty=Difficulty.COUNTERFACTUAL.value,
                        data_id=data_id,
                        task="shortest_path",
                    )
                )

        targets = self._difficulty_targets(items_per_dataset)

        selected: list[QAPair] = []
        for difficulty, target_count in targets.items():
            filtered = [
                pair for pair in candidate_pairs if pair.difficulty == difficulty
            ]
            for pair in filtered[:target_count]:
                selected.append(pair)

        return selected[:items_per_dataset]

    def generate_connectivity(self, graph: nx.Graph) -> list[QAPair]:
        """Generate connectivity questions for selected node pairs."""
        nodes = list(graph.nodes)
        if len(nodes) < 2:
            return []
        u, v = nodes[0], nodes[-1]
        connected = nx.has_path(graph, u, v)
        difficulty = self.classifier.classify(
            reasoning_steps=1,
            requires_arithmetic=False,
            has_counterfactual=False,
        )
        return [
            QAPair(
                question=f"Are node {u} and node {v} connected?",
                answer="yes" if connected else "no",
                difficulty=difficulty.value,
                data_id="graph-connectivity",
                task="connectivity",
            )
        ]

    def generate_shortest_path(self, graph: nx.Graph) -> list[QAPair]:
        """Generate shortest-path questions over graph pairs."""
        nodes = list(graph.nodes)
        if len(nodes) < 2:
            return []
        u, v = nodes[0], nodes[-1]
        try:
            length = nx.shortest_path_length(graph, source=u, target=v)
            answer = str(length)
        except nx.NetworkXNoPath:
            answer = "no_path"
        difficulty = self.classifier.classify(
            reasoning_steps=2,
            requires_arithmetic=True,
            has_counterfactual=False,
        )
        return [
            QAPair(
                question=f"What is the shortest path length between node {u} and node {v}?",
                answer=answer,
                difficulty=difficulty.value,
                data_id="graph-shortest-path",
                task="shortest_path",
            )
        ]

    def generate_community(self, graph: nx.Graph) -> list[QAPair]:
        """Generate community-level comparison and counterfactual questions."""
        components = list(nx.connected_components(graph))
        if not components:
            return []
        largest = max(len(component) for component in components)
        total = graph.number_of_nodes()
        ratio = round(largest / total, 3) if total else 0.0
        return [
            QAPair(
                question="What fraction of nodes belong to the largest connected component?",
                answer=str(ratio),
                difficulty=Difficulty.THREE_HOP.value,
                data_id="graph-community",
                task="community",
            ),
            QAPair(
                question="If one bridge edge were removed, would the number of connected components likely increase?",
                answer="yes"
                if graph.number_of_edges() > graph.number_of_nodes() - 1
                else "no",
                difficulty=Difficulty.COUNTERFACTUAL.value,
                data_id="graph-community",
                task="community",
            ),
        ]
