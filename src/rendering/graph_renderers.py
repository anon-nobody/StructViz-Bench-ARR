from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import io
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.rendering.style_config import (
    ANNOTATION_FONT_SIZE,
    DEFAULT_DPI,
    LABEL_FONT_SIZE,
    TITLE_FONT_SIZE,
    apply_global_style,
    apply_theme,
    figure_size,
)
from src.rendering import v2_renderers as v2



# The text view lists at most this many edges; on the default 1024x768 canvas only ~43 lines are
# visible (see the paper's rendering completeness audit). Change to re-render without truncation.
TEXT_VIEW_MAX_EDGES = 180

class GraphRenderers:
    """Render graph data in publication-quality visualization methods."""

    def __init__(self, suite: str = "v1") -> None:
        """Select the rendering suite ("v1" = released benchmark images, "v2" = all labels)."""
        self.suite = suite

    def node_link(
        self,
        graph: nx.Graph[Any],
        width: int = 1024,
        height: int = 768,
        title: str = "",
        data_meta: dict[str, object] | None = None,
        style: str | None = None,
    ) -> Image.Image:
        """Render a degree-aware node-link diagram.

        Args:
            graph: Input graph.
            width: Target image width in pixels.
            height: Target image height in pixels.
            title: Optional title override.
            data_meta: Optional metadata for context.
            style: Optional matplotlib style name or alias.

        Returns:
            Rendered node-link diagram as a PIL image.
        """
        if self.suite == "v2":
            return v2.graph_node_link(graph, title=title, data_meta=data_meta, style=style)
        apply_global_style(style)
        fig, ax = plt.subplots(figsize=figure_size(width, height), dpi=DEFAULT_DPI)
        theme = apply_theme(ax)
        pos = nx.spring_layout(graph, seed=42)
        node_degrees = dict(graph.degree())
        degree_values = np.array(list(node_degrees.values()), dtype=float)
        if degree_values.size == 0:
            degree_values = np.array([1.0])
        size_scale = 220 + 520 * (degree_values / (degree_values.max() + 1e-9))
        node_sizes = [float(size_scale[idx]) for idx in range(len(graph.nodes))]
        edge_widths = [
            1.0 + float((node_degrees[u] + node_degrees[v]) * 0.15)
            for u, v in graph.edges()
        ]
        nx.draw_networkx_nodes(
            graph,
            pos=pos,
            ax=ax,
            node_size=node_sizes,
            node_color=theme.primary_palette[0],
            alpha=0.9,
            edgecolors="white",
            linewidths=0.8,
        )
        nx.draw_networkx_edges(
            graph,
            pos=pos,
            ax=ax,
            width=edge_widths,
            alpha=0.6,
            edge_color=theme.primary_palette[5],
        )
        if graph.number_of_nodes() < 30:
            nx.draw_networkx_labels(
                graph,
                pos=pos,
                ax=ax,
                font_size=ANNOTATION_FONT_SIZE,
                font_color="#0f172a",
            )
        ax.set_title(
            title
            or str(
                data_meta.get("title", "Graph Structure (Node-Link View)")
                if data_meta
                else "Graph Structure (Node-Link View)"
            ),
            fontsize=TITLE_FONT_SIZE,
        )
        ax.axis("off")
        return self._to_image(fig)

    def adjacency_matrix(
        self,
        graph: nx.Graph[Any],
        width: int = 1024,
        height: int = 768,
        title: str = "",
        data_meta: dict[str, object] | None = None,
        style: str | None = None,
    ) -> Image.Image:
        """Render a labeled adjacency matrix heatmap.

        Args:
            graph: Input graph.
            width: Target image width in pixels.
            height: Target image height in pixels.
            title: Optional title override.
            data_meta: Optional metadata for context.
            style: Optional matplotlib style name or alias.

        Returns:
            Rendered adjacency matrix as a PIL image.
        """
        if self.suite == "v2":
            return v2.graph_adjacency_matrix(
                graph, title=title, data_meta=data_meta, style=style
            )
        apply_global_style(style)
        nodes = list(graph.nodes())
        matrix = nx.to_numpy_array(graph, nodelist=nodes)
        fig, ax = plt.subplots(figsize=figure_size(width, height), dpi=DEFAULT_DPI)
        theme = apply_theme(ax)
        im = ax.imshow(matrix, cmap=theme.sequential_cmap, interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, shrink=0.86)
        cbar.ax.set_ylabel("Connectivity", fontsize=LABEL_FONT_SIZE)
        labels = [str(node) for node in nodes]
        max_labels = 24
        if len(labels) > max_labels:
            step = max(1, len(labels) // max_labels)
            ticks = list(range(0, len(labels), step))
            tick_labels = [labels[idx] for idx in ticks]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right")
            ax.set_yticklabels(tick_labels)
        else:
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels)
        if matrix.size and len(nodes) <= 15:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.0f}",
                        ha="center",
                        va="center",
                        fontsize=ANNOTATION_FONT_SIZE,
                        color="#111827",
                    )
        ax.set_xlabel(
            str(data_meta.get("x_label", "Target Node") if data_meta else "Target Node")
        )
        ax.set_ylabel(
            str(data_meta.get("y_label", "Source Node") if data_meta else "Source Node")
        )
        ax.set_title(
            title
            or str(
                data_meta.get("title", "Graph Adjacency Matrix")
                if data_meta
                else "Graph Adjacency Matrix"
            ),
            fontsize=TITLE_FONT_SIZE,
        )
        return self._to_image(fig)

    def circular_layout(
        self,
        graph: nx.Graph[Any],
        width: int = 1024,
        height: int = 768,
        title: str = "",
        data_meta: dict[str, object] | None = None,
        style: str | None = None,
    ) -> Image.Image:
        """Render graph with circular layout and curved edges.

        Args:
            graph: Input graph.
            width: Target image width in pixels.
            height: Target image height in pixels.
            title: Optional title override.
            data_meta: Optional metadata for context.
            style: Optional matplotlib style name or alias.

        Returns:
            Rendered circular graph view as a PIL image.
        """
        if self.suite == "v2":
            return v2.graph_circular_layout(
                graph,
                self._community_map(graph),
                title=title,
                data_meta=data_meta,
                style=style,
            )
        apply_global_style(style)
        fig, ax = plt.subplots(figsize=figure_size(width, height), dpi=DEFAULT_DPI)
        theme = apply_theme(ax)
        pos = nx.circular_layout(graph)
        communities = self._community_map(graph)
        node_colors = [
            theme.primary_palette[communities[node] % len(theme.primary_palette)]
            for node in graph.nodes()
        ]
        node_sizes = [240 + graph.degree(node) * 35 for node in graph.nodes()]
        nx.draw_networkx_nodes(
            graph,
            pos=pos,
            ax=ax,
            node_size=node_sizes,
            node_color=node_colors,
            edgecolors="white",
            linewidths=0.8,
        )
        if graph.number_of_edges() > 0:
            nx.draw_networkx_edges(
                graph,
                pos=pos,
                ax=ax,
                width=1.2,
                edge_color="#64748b",
                alpha=0.65,
                connectionstyle="arc3,rad=0.15",
            )
        if graph.number_of_nodes() < 30:
            nx.draw_networkx_labels(
                graph,
                pos=pos,
                ax=ax,
                font_size=ANNOTATION_FONT_SIZE,
                font_color="#0f172a",
            )
        ax.set_title(
            title
            or str(
                data_meta.get("title", "Graph Circular Layout by Community")
                if data_meta
                else "Graph Circular Layout by Community"
            ),
            fontsize=TITLE_FONT_SIZE,
        )
        ax.axis("off")
        return self._to_image(fig)

    def text_only(
        self,
        graph: nx.Graph[Any],
        width: int = 1024,
        height: int = 768,
        title: str = "",
        data_meta: dict[str, object] | None = None,
        style: str | None = None,
    ) -> Image.Image:
        """Render graph edges and node degree statistics as text.

        Args:
            graph: Input graph.
            width: Target image width in pixels.
            height: Target image height in pixels.
            title: Optional title override.
            data_meta: Optional metadata for context.
            style: Unused style token kept for API consistency.

        Returns:
            Rendered text image as a PIL image.
        """
        if self.suite == "v2":
            return v2.graph_text_only(graph, title=title, data_meta=data_meta)
        image = Image.new("RGB", (width, height), color=(250, 252, 255))
        draw = ImageDraw.Draw(image)
        try:
            title_font = ImageFont.truetype("DejaVuSansMono.ttf", 16)
            body_font = ImageFont.truetype("DejaVuSansMono.ttf", 12)
        except OSError:
            title_font = ImageFont.load_default()
            body_font = ImageFont.load_default()
        title_text = title or str(
            data_meta.get("title", "Graph Text View")
            if data_meta
            else "Graph Text View"
        )
        draw.rectangle((16, 16, width - 16, 54), fill=(220, 230, 242))
        draw.text((24, 24), title_text, fill=(15, 23, 42), font=title_font)

        node_degrees = dict(graph.degree())
        edges = list(graph.edges())
        lines = [
            "edge list (u -- v) | degree(u), degree(v)",
            "----------------------------------------",
        ]
        lines.extend(
            f"{str(u):>8} -- {str(v):<8} | {node_degrees.get(u, 0):>3d},"
            f" {node_degrees.get(v, 0):>3d}"
            for u, v in edges[:TEXT_VIEW_MAX_EDGES]
        )
        if not edges:
            lines.append("(no edges)")
        draw.multiline_text(
            (24, 70), "\n".join(lines), fill=(31, 41, 51), font=body_font, spacing=4
        )
        return image

    def text_only_deg(
        self,
        graph: nx.Graph[Any],
        width: int = 1024,
        height: int = 768,
        title: str = "",
        data_meta: dict[str, object] | None = None,
        style: str | None = None,
    ) -> Image.Image:
        """v2 text view listing every edge together with v1's degree columns (assist condition)."""
        del width, height, style
        return v2.graph_text_only(graph, title=title, data_meta=data_meta, with_degree=True)

    def _to_image(self, fig: plt.Figure) -> Image.Image:
        """Convert a matplotlib figure into a PIL image."""
        buffer = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def _community_map(self, graph: nx.Graph[Any]) -> dict[object, int]:
        """Assign nodes to community identifiers for coloring."""
        if graph.number_of_nodes() == 0:
            return {}
        if graph.number_of_edges() == 0:
            return {node: idx for idx, node in enumerate(graph.nodes())}
        communities = nx.algorithms.community.greedy_modularity_communities(graph)
        mapping: dict[object, int] = {}
        for idx, community in enumerate(communities):
            for node in community:
                mapping[node] = idx
        return mapping
