from __future__ import annotations

# pyright: reportMissingImports=false

from dataclasses import dataclass, field
from typing import Any

Image = object

from src.rendering.graph_renderers import GraphRenderers
from src.rendering.tabular_renderers import TabularRenderers
from src.rendering.timeseries_renderers import TimeSeriesRenderers

# "v1": released benchmark images (byte-stable); "v2": answer-complete renderings;
# "assist": tabular calculation-aid 2x2 (full table / row bars, with / without column means).
SUITES = ("v1", "v2", "assist")


@dataclass(slots=True)
class RenderPipeline:
    """Render one data item across all applicable visualization formats."""

    width: int = 1024
    height: int = 768
    dpi: int = 150
    style: str | None = "default"
    suite: str = "v1"
    tabular: TabularRenderers = field(init=False)
    timeseries: TimeSeriesRenderers = field(init=False)
    graph: GraphRenderers = field(init=False)

    def __post_init__(self) -> None:
        """Initialize renderer families."""
        if self.suite not in SUITES:
            raise ValueError(f"Unknown render suite {self.suite!r}; expected one of {SUITES}")
        renderer_suite = "v1" if self.suite == "v1" else "v2"
        self.tabular = TabularRenderers(suite=renderer_suite)
        self.timeseries = TimeSeriesRenderers(suite=renderer_suite)
        self.graph = GraphRenderers(suite=renderer_suite)

    def render_all(
        self,
        data_item: dict[str, Any],
        style: str | None = None,
    ) -> dict[str, Image]:
        """Return mapping viz_name to rendered image for one item.

        Args:
            data_item: Benchmark sample including modality and payload.
            style: Optional style override for this render call.

        Returns:
            Mapping from visualization method name to image.
        """
        modality = str(data_item["modality"])
        payload = data_item["data"]
        data_meta = data_item.get("data_meta")
        viz_titles = data_item.get("viz_titles", {})
        effective_style = style or self.style
        rendered: dict[str, Image] = {}

        if self.suite == "assist":
            if modality != "tabular":
                raise ValueError("The assist suite is defined for tabular items only")
            return self.tabular.render_assist(payload, style=effective_style)

        if modality == "tabular":
            rendered["bar_chart"] = self.tabular.bar_chart(
                payload,
                self.width,
                self.height,
                title=str(viz_titles.get("bar_chart", "")),
                data_meta=data_meta,
                style=effective_style,
            )
            rendered["heatmap"] = self.tabular.heatmap(
                payload,
                self.width,
                self.height,
                title=str(viz_titles.get("heatmap", "")),
                data_meta=data_meta,
                style=effective_style,
            )
            rendered["table_image"] = self.tabular.table_image(
                payload,
                self.width,
                self.height,
                title=str(viz_titles.get("table_image", "")),
                data_meta=data_meta,
                style=effective_style,
            )
            rendered["scatter_plot"] = self.tabular.scatter_plot(
                payload,
                self.width,
                self.height,
                title=str(viz_titles.get("scatter_plot", "")),
                data_meta=data_meta,
                style=effective_style,
            )
            rendered["text_only"] = self.tabular.text_only(
                payload,
                self.width,
                self.height,
                title=str(viz_titles.get("text_only", "")),
                data_meta=data_meta,
                style=effective_style,
            )
        elif modality == "timeseries":
            rendered["line_plot"] = self.timeseries.line_plot(
                payload,
                self.width,
                self.height,
                title=str(viz_titles.get("line_plot", "")),
                data_meta=data_meta,
                style=effective_style,
            )
            rendered["gaf"] = self.timeseries.gaf(
                payload,
                self.width,
                self.height,
                title=str(viz_titles.get("gaf", "")),
                data_meta=data_meta,
                style=effective_style,
            )
            rendered["recurrence_plot"] = self.timeseries.recurrence_plot(
                payload,
                self.width,
                self.height,
                title=str(viz_titles.get("recurrence_plot", "")),
                data_meta=data_meta,
                style=effective_style,
            )
            rendered["heatmap"] = self.timeseries.heatmap(
                payload,
                self.width,
                self.height,
                title=str(viz_titles.get("heatmap", "")),
                data_meta=data_meta,
                style=effective_style,
            )
            rendered["text_only"] = self.timeseries.text_only(
                payload,
                self.width,
                self.height,
                title=str(viz_titles.get("text_only", "")),
                data_meta=data_meta,
                style=effective_style,
            )
        elif modality == "graph":
            rendered["node_link"] = self.graph.node_link(
                payload,
                self.width,
                self.height,
                title=str(viz_titles.get("node_link", "")),
                data_meta=data_meta,
                style=effective_style,
            )
            rendered["adjacency_matrix"] = self.graph.adjacency_matrix(
                payload,
                self.width,
                self.height,
                title=str(viz_titles.get("adjacency_matrix", "")),
                data_meta=data_meta,
                style=effective_style,
            )
            rendered["circular_layout"] = self.graph.circular_layout(
                payload,
                self.width,
                self.height,
                title=str(viz_titles.get("circular_layout", "")),
                data_meta=data_meta,
                style=effective_style,
            )
            rendered["text_only"] = self.graph.text_only(
                payload,
                self.width,
                self.height,
                title=str(viz_titles.get("text_only", "")),
                data_meta=data_meta,
                style=effective_style,
            )
            if self.suite == "v2":
                rendered["text_only_deg"] = self.graph.text_only_deg(
                    payload,
                    self.width,
                    self.height,
                    title=str(viz_titles.get("text_only_deg", "")),
                    data_meta=data_meta,
                    style=effective_style,
                )
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        return rendered
