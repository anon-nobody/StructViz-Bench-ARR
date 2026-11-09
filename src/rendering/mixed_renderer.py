from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.rendering.render_pipeline import RenderPipeline


@dataclass(slots=True)
class MixedRenderer:
    width: int = 1536
    height: int = 768
    style: str | None = "default"
    pipeline: RenderPipeline = field(init=False)
    _panel_labels: tuple[str, str] = field(
        init=False,
        default=("Panel A: left", "Panel B: right"),
    )

    def __post_init__(self) -> None:
        panel_width = max(256, self.width // 2)
        self.pipeline = RenderPipeline(
            width=panel_width, height=self.height, style=self.style
        )

    def render_composite(
        self,
        left_image: Image.Image,
        right_image: Image.Image,
        width: int,
        height: int,
    ) -> Image.Image:
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()

        panel_width = width // 2
        header_height = 36
        content_height = max(1, height - header_height - 8)

        left_fit = ImageOps.contain(
            left_image.convert("RGB"), (panel_width - 12, content_height)
        )
        right_fit = ImageOps.contain(
            right_image.convert("RGB"), (panel_width - 12, content_height)
        )

        left_x = (panel_width - left_fit.width) // 2
        right_x = panel_width + (panel_width - right_fit.width) // 2
        y = header_height + (content_height - left_fit.height) // 2
        canvas.paste(left_fit, (left_x, y))
        canvas.paste(right_fit, (right_x, y))

        draw.text((8, 10), self._panel_labels[0], fill="black", font=font)
        draw.text((panel_width + 8, 10), self._panel_labels[1], fill="black", font=font)
        draw.line([(panel_width, 0), (panel_width, height)], fill="#444444", width=1)
        return canvas

    def render_mixed_item(
        self, data_item: dict[str, Any], viz_combo: str
    ) -> Image.Image:
        left_viz, right_viz = self._split_viz_combo(viz_combo)
        left_component, right_component = self._extract_components(data_item)

        left_input = self._to_pipeline_input(left_component)
        right_input = self._to_pipeline_input(right_component)

        left_rendered = self.pipeline.render_all(left_input)
        right_rendered = self.pipeline.render_all(right_input)

        if left_viz not in left_rendered:
            raise ValueError(
                f"Unsupported left viz '{left_viz}' for {left_input['modality']}"
            )
        if right_viz not in right_rendered:
            raise ValueError(
                f"Unsupported right viz '{right_viz}' for {right_input['modality']}"
            )

        self._panel_labels = (
            f"Panel A: {left_input['modality']}",
            f"Panel B: {right_input['modality']}",
        )
        return self.render_composite(
            left_rendered[left_viz],
            right_rendered[right_viz],
            self.width,
            self.height,
        )

    def _split_viz_combo(self, viz_combo: str) -> tuple[str, str]:
        parts = [part.strip() for part in viz_combo.split("+")]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Invalid mixed viz combo: {viz_combo}")
        return parts[0], parts[1]

    def _extract_components(
        self,
        data_item: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = data_item.get("data", data_item)
        if not isinstance(payload, dict):
            raise ValueError("Mixed data payload must be a dict.")
        left = payload.get("left")
        right = payload.get("right")
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise ValueError("Mixed data payload must contain left/right components.")
        return left, right

    def _to_pipeline_input(self, component: dict[str, Any]) -> dict[str, Any]:
        modality = str(component.get("modality", ""))
        raw_data = component.get("data", {})
        if not isinstance(raw_data, dict):
            raise ValueError("Mixed component data must be a dict.")

        converted_data: Any
        data_meta = raw_data.get("metadata")
        if modality == "tabular":
            records = raw_data.get("records", [])
            columns = raw_data.get("columns")
            if isinstance(columns, list):
                converted_data = pd.DataFrame(
                    records, columns=[str(c) for c in columns]
                )
            else:
                converted_data = pd.DataFrame(records)
        elif modality == "timeseries":
            values = raw_data.get("values", [])
            converted_data = (
                [float(value) for value in values] if isinstance(values, list) else []
            )
        elif modality == "graph":
            node_link = raw_data.get("node_link", {})
            converted_data = (
                nx.node_link_graph(node_link)
                if isinstance(node_link, dict)
                else nx.Graph()
            )
        else:
            raise ValueError(f"Unsupported mixed component modality: {modality}")

        return {
            "modality": modality,
            "data": converted_data,
            "data_meta": data_meta if isinstance(data_meta, dict) else None,
            "viz_titles": {},
        }
