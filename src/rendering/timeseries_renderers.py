from __future__ import annotations

# pyright: reportMissingImports=false

import io

import matplotlib.pyplot as plt
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



# The text view lists at most this many points; on the default 1024x768 canvas only ~43 lines are
# visible (see the paper's rendering completeness audit). Change to re-render without truncation.
TEXT_VIEW_MAX_POINTS = 180

class TimeSeriesRenderers:
    """Render time-series samples in publication-quality visualization styles."""

    def __init__(self, suite: str = "v1") -> None:
        """Select the rendering suite; v2 only changes text_only (all points, multi-column)."""
        self.suite = suite

    def line_plot(
        self,
        ts: list[float],
        width: int = 1024,
        height: int = 768,
        title: str = "",
        data_meta: dict[str, object] | None = None,
        style: str | None = None,
    ) -> Image.Image:
        """Render a sequence as a styled line chart.

        Args:
            ts: Input time-series values.
            width: Target image width in pixels.
            height: Target image height in pixels.
            title: Optional title override.
            data_meta: Optional metadata for axis labels and context.
            style: Optional matplotlib style name or alias.

        Returns:
            Rendered chart as a PIL image.
        """
        apply_global_style(style)
        arr = self._prepare_series(ts)
        steps = np.arange(arr.size)
        fig, ax = plt.subplots(figsize=figure_size(width, height), dpi=DEFAULT_DPI)
        theme = apply_theme(ax)
        ax.plot(
            steps,
            arr,
            color=theme.primary_palette[0],
            linewidth=2.2,
            marker="o",
            markersize=3.5,
            markerfacecolor=theme.primary_palette[1],
            markeredgecolor="white",
            alpha=0.95,
        )
        max_idx = int(np.argmax(arr))
        min_idx = int(np.argmin(arr))
        key_indices = sorted({0, min_idx, max_idx, arr.size - 1})
        ax.scatter(
            key_indices,
            arr[key_indices],
            s=56,
            color=theme.primary_palette[3],
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
            label="Key points",
        )
        ax.legend(loc="upper right", fontsize=ANNOTATION_FONT_SIZE)
        ax.set_xlabel(
            str(data_meta.get("x_label", "Timestep") if data_meta else "Timestep")
        )
        ax.set_ylabel(str(data_meta.get("y_label", "Value") if data_meta else "Value"))
        ax.set_title(
            title
            or str(
                data_meta.get("title", "Time Series Trend by Timestep")
                if data_meta
                else "Time Series Trend by Timestep"
            ),
            fontsize=TITLE_FONT_SIZE,
        )
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.8)
        return self._to_image(fig)

    def gaf(
        self,
        ts: list[float],
        width: int = 1024,
        height: int = 768,
        title: str = "",
        data_meta: dict[str, object] | None = None,
        style: str | None = None,
    ) -> Image.Image:
        """Render a real Gramian Angular Summation Field matrix.

        Args:
            ts: Input time-series values.
            width: Target image width in pixels.
            height: Target image height in pixels.
            title: Optional title override.
            data_meta: Optional metadata for axis labels and context.
            style: Optional matplotlib style name or alias.

        Returns:
            Rendered GAF image as a PIL image.
        """
        apply_global_style(style)
        arr = self._prepare_series(ts)
        matrix = self._compute_gaf(arr)
        fig, ax = plt.subplots(figsize=figure_size(width, height), dpi=DEFAULT_DPI)
        theme = apply_theme(ax)
        im = ax.imshow(
            matrix, cmap=theme.diverging_cmap, origin="lower", interpolation="nearest"
        )
        cbar = fig.colorbar(im, ax=ax, shrink=0.86)
        cbar.ax.set_ylabel("GAF Intensity", fontsize=LABEL_FONT_SIZE)
        ax.set_xlabel(
            str(data_meta.get("x_label", "Timestep i") if data_meta else "Timestep i")
        )
        ax.set_ylabel(
            str(data_meta.get("y_label", "Timestep j") if data_meta else "Timestep j")
        )
        ax.set_title(
            title
            or str(
                data_meta.get("title", "Gramian Angular Summation Field")
                if data_meta
                else "Gramian Angular Summation Field"
            ),
            fontsize=TITLE_FONT_SIZE,
        )
        return self._to_image(fig)

    def recurrence_plot(
        self,
        ts: list[float],
        width: int = 1024,
        height: int = 768,
        title: str = "",
        data_meta: dict[str, object] | None = None,
        style: str | None = None,
    ) -> Image.Image:
        """Render a thresholded binary recurrence plot.

        Args:
            ts: Input time-series values.
            width: Target image width in pixels.
            height: Target image height in pixels.
            title: Optional title override.
            data_meta: Optional metadata for axis labels and context.
            style: Optional matplotlib style name or alias.

        Returns:
            Rendered recurrence plot as a PIL image.
        """
        apply_global_style(style)
        arr = self._prepare_series(ts)
        dist = np.abs(arr[:, None] - arr[None, :])
        threshold = float(np.percentile(dist, 20))
        recurrence = (dist <= threshold).astype(float)
        fig, ax = plt.subplots(figsize=figure_size(width, height), dpi=DEFAULT_DPI)
        theme = apply_theme(ax)
        im = ax.imshow(
            recurrence,
            cmap=theme.sequential_cmap,
            origin="lower",
            interpolation="nearest",
        )
        cbar = fig.colorbar(im, ax=ax, shrink=0.86)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Non-recurrent", "Recurrent"])
        ax.set_xlabel(
            str(data_meta.get("x_label", "Timestep i") if data_meta else "Timestep i")
        )
        ax.set_ylabel(
            str(data_meta.get("y_label", "Timestep j") if data_meta else "Timestep j")
        )
        ax.set_title(
            title
            or str(
                data_meta.get("title", "Binary Recurrence Plot")
                if data_meta
                else "Binary Recurrence Plot"
            ),
            fontsize=TITLE_FONT_SIZE,
        )
        return self._to_image(fig)

    def heatmap(
        self,
        ts: list[float],
        width: int = 1024,
        height: int = 768,
        title: str = "",
        data_meta: dict[str, object] | None = None,
        style: str | None = None,
    ) -> Image.Image:
        """Render a sequence as a single-row heatmap.

        Args:
            ts: Input time-series values.
            width: Target image width in pixels.
            height: Target image height in pixels.
            title: Optional title override.
            data_meta: Optional metadata for axis labels and context.
            style: Optional matplotlib style name or alias.

        Returns:
            Rendered heatmap as a PIL image.
        """
        apply_global_style(style)
        arr = self._prepare_series(ts)
        heat = arr[np.newaxis, :]
        fig, ax = plt.subplots(figsize=figure_size(width, height), dpi=DEFAULT_DPI)
        theme = apply_theme(ax)
        im = ax.imshow(heat, aspect="auto", cmap=theme.sequential_cmap)
        cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.ax.set_ylabel("Value", fontsize=LABEL_FONT_SIZE)
        tick_count = min(10, arr.size)
        tick_positions = np.linspace(0, arr.size - 1, num=tick_count, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(pos) for pos in tick_positions])
        ax.set_yticks([0])
        ax.set_yticklabels(
            [str(data_meta.get("series_label", "Series") if data_meta else "Series")]
        )
        ax.set_xlabel(
            str(data_meta.get("x_label", "Timestep") if data_meta else "Timestep")
        )
        ax.set_ylabel(
            str(data_meta.get("y_label", "Signal") if data_meta else "Signal")
        )
        ax.set_title(
            title
            or str(
                data_meta.get("title", "Time Series Intensity Heatmap")
                if data_meta
                else "Time Series Intensity Heatmap"
            ),
            fontsize=TITLE_FONT_SIZE,
        )
        return self._to_image(fig)

    def text_only(
        self,
        ts: list[float],
        width: int = 1024,
        height: int = 768,
        title: str = "",
        data_meta: dict[str, object] | None = None,
        style: str | None = None,
    ) -> Image.Image:
        """Render time-series values as a structured text table.

        Args:
            ts: Input time-series values.
            width: Target image width in pixels.
            height: Target image height in pixels.
            title: Optional title override.
            data_meta: Optional metadata for context.
            style: Unused style token kept for API consistency.

        Returns:
            Rendered text table as a PIL image.
        """
        if self.suite == "v2":
            return v2.timeseries_text_only(
                self._prepare_series(ts), title=title, data_meta=data_meta
            )
        arr = self._prepare_series(ts)
        image = Image.new("RGB", (width, height), color=(250, 252, 255))
        draw = ImageDraw.Draw(image)
        try:
            title_font = ImageFont.truetype("DejaVuSansMono.ttf", 16)
            body_font = ImageFont.truetype("DejaVuSansMono.ttf", 12)
        except OSError:
            title_font = ImageFont.load_default()
            body_font = ImageFont.load_default()
        title_text = title or str(
            data_meta.get("title", "Time Series Text View")
            if data_meta
            else "Time Series Text View"
        )
        draw.rectangle((16, 16, width - 16, 54), fill=(220, 230, 242))
        draw.text((24, 24), title_text, fill=(15, 23, 42), font=title_font)
        lines = ["timestep | value", "----------------"]
        lines.extend(
            f"{idx:>7d} | {value:>8.4f}" for idx, value in enumerate(arr[:TEXT_VIEW_MAX_POINTS])
        )
        draw.multiline_text(
            (24, 70), "\n".join(lines), fill=(31, 41, 51), font=body_font, spacing=4
        )
        return image

    def _to_image(self, fig: plt.Figure) -> Image.Image:
        """Convert a matplotlib figure into a PIL image."""
        buffer = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def _prepare_series(self, ts: list[float]) -> np.ndarray:
        """Convert input sequence to a safe float array."""
        if not ts:
            return np.array([0.0], dtype=float)
        return np.asarray(ts, dtype=float)

    def _compute_gaf(self, arr: np.ndarray) -> np.ndarray:
        """Compute Gramian Angular Summation Field from a 1D signal."""
        scaled = 2.0 * (arr - arr.min()) / (arr.max() - arr.min() + 1e-9) - 1.0
        scaled = np.clip(scaled, -1.0, 1.0)
        try:
            from pyts.image import GramianAngularField

            transformer = GramianAngularField(method="summation", sample_range=(-1, 1))
            return transformer.fit_transform(scaled.reshape(1, -1))[0]
        except Exception:
            phi = np.arccos(scaled)
            return np.cos(phi[:, None] + phi[None, :])
