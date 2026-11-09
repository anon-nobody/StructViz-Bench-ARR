from __future__ import annotations

# pyright: reportMissingImports=false

import io
from typing import Any

import numpy as np

Image = Any

from src.rendering.style_config import (
    ANNOTATION_FONT_SIZE,
    DEFAULT_DPI,
    HEADER_BG_COLOR,
    LABEL_FONT_SIZE,
    ROW_ALT_BG_COLOR,
    TITLE_FONT_SIZE,
    apply_global_style,
    apply_theme,
    figure_size,
)
from src.rendering import v2_renderers as v2



# Row limits used by the released benchmark images (see the paper's rendering completeness
# audit). table_image and the text view show only the first rows of long tables; bar_chart and
# scatter_plot draw every row. Change these to re-render without truncation.
TABLE_IMAGE_MAX_ROWS = 12
TEXT_VIEW_MAX_ROWS = 14
HEATMAP_MAX_ROWS = 20

class TabularRenderers:
    """Render tabular data with publication-ready visualization methods."""

    def __init__(self, suite: str = "v1") -> None:
        """Select the rendering suite ("v1" = released benchmark images, "v2" = full-data)."""
        self.suite = suite

    def bar_chart(
        self,
        df: Any,
        width: int = 1024,
        height: int = 768,
        title: str = "",
        data_meta: dict[str, Any] | None = None,
        style: str | None = None,
    ) -> Image:
        """Render sorted numeric aggregates as a horizontal bar chart.

        Args:
            df: Input dataframe.
            width: Target image width in pixels.
            height: Target image height in pixels.
            title: Optional title override.
            data_meta: Optional metadata used for axis labels and context.
            style: Optional matplotlib style name or alias.

        Returns:
            Rendered chart as a PIL image.
        """
        if self.suite == "v2":
            return v2.tabular_bar_chart(df, title=title, data_meta=data_meta, style=style)
        plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        pd = __import__("pandas")
        apply_global_style(style)
        numeric = df.select_dtypes(include="number")
        values = (
            numeric.mean() if not numeric.empty else pd.Series([1], index=["empty"])
        )
        values = values.sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=figure_size(width, height), dpi=DEFAULT_DPI)
        theme = apply_theme(ax)
        y_labels = values.index.astype(str)
        colors = [
            theme.primary_palette[i % len(theme.primary_palette)]
            for i in range(len(values))
        ]
        bars = ax.barh(y_labels, values.values, color=colors, alpha=0.95)
        x_label = str(
            data_meta.get("value_label", "Mean Value") if data_meta else "Mean Value"
        )
        y_label = str(
            data_meta.get("category_label", "Columns") if data_meta else "Columns"
        )
        default_title = "Mean Value by Column"
        ax.set_xlabel(x_label, fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel(y_label, fontsize=LABEL_FONT_SIZE)
        ax.set_title(
            title
            or str(
                data_meta.get("title", default_title) if data_meta else default_title
            )
        )
        ax.grid(axis="x", linestyle="--", alpha=0.7)
        ax.grid(axis="y", visible=False)
        value_range = float(np.nanmax(np.abs(values.values))) if len(values) else 1.0
        pad = value_range * 0.08 if value_range > 0 else 0.5
        for bar in bars:
            val = bar.get_width()
            ax.text(
                val + pad,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center",
                fontsize=ANNOTATION_FONT_SIZE,
                color=theme.text_color,
            )
        ax.set_xlim(right=ax.get_xlim()[1] + pad * 2)
        return self._to_image(fig)

    def heatmap(
        self,
        df: Any,
        width: int = 1024,
        height: int = 768,
        title: str = "",
        data_meta: dict[str, Any] | None = None,
        style: str | None = None,
    ) -> Image:
        """Render numeric table values as an annotated heatmap.

        Args:
            df: Input dataframe.
            width: Target image width in pixels.
            height: Target image height in pixels.
            title: Optional title override.
            data_meta: Optional metadata used for axis labels and context.
            style: Optional matplotlib style name or alias.

        Returns:
            Rendered heatmap as a PIL image.
        """
        if self.suite == "v2":
            return v2.tabular_heatmap(df, title=title, data_meta=data_meta, style=style)
        plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        pd = __import__("pandas")
        sns = __import__("seaborn")
        apply_global_style(style)
        numeric = df.select_dtypes(include="number")
        matrix = (
            numeric.head(HEATMAP_MAX_ROWS)
            if not numeric.empty
            else pd.DataFrame([[0.0]], columns=["value"])
        )
        fig, ax = plt.subplots(figsize=figure_size(width, height), dpi=DEFAULT_DPI)
        theme = apply_theme(ax)
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=theme.sequential_cmap,
            annot=True,
            fmt=".2f",
            linewidths=0.4,
            linecolor="white",
            cbar_kws={"shrink": 0.85, "label": "Value"},
            annot_kws={"fontsize": ANNOTATION_FONT_SIZE},
        )
        ax.set_xlabel(
            str(data_meta.get("x_label", "Columns") if data_meta else "Columns")
        )
        ax.set_ylabel(str(data_meta.get("y_label", "Rows") if data_meta else "Rows"))
        ax.set_title(
            title
            or str(
                data_meta.get("title", "Tabular Value Heatmap")
                if data_meta
                else "Tabular Value Heatmap"
            ),
            fontsize=TITLE_FONT_SIZE,
        )
        return self._to_image(fig)

    def table_image(
        self,
        df: Any,
        width: int = 1024,
        height: int = 768,
        title: str = "",
        data_meta: dict[str, Any] | None = None,
        style: str | None = None,
    ) -> Image:
        """Render dataframe values as a styled table image.

        Args:
            df: Input dataframe.
            width: Target image width in pixels.
            height: Target image height in pixels.
            title: Optional title override.
            data_meta: Optional metadata used for title context.
            style: Optional matplotlib style name or alias.

        Returns:
            Rendered table as a PIL image.
        """
        if self.suite == "v2":
            return v2.tabular_table_image(df, title=title, data_meta=data_meta, style=style)
        plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        apply_global_style(style)
        fig, ax = plt.subplots(figsize=figure_size(width, height), dpi=DEFAULT_DPI)
        _ = apply_theme(ax)
        ax.axis("off")
        table_data = df.head(TABLE_IMAGE_MAX_ROWS).copy()
        table_data = table_data.map(
            lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)
        )
        table = ax.table(
            cellText=table_data.values,
            colLabels=[str(column) for column in table_data.columns],
            rowLabels=[str(i) for i in table_data.index],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(ANNOTATION_FONT_SIZE)
        table.scale(1.0, 1.35)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor(HEADER_BG_COLOR)
                cell.set_text_props(weight="bold", color="#0f172a")
            elif row % 2 == 0:
                cell.set_facecolor(ROW_ALT_BG_COLOR)
            else:
                cell.set_facecolor("white")
            cell.set_edgecolor("#d0d7de")
            cell.set_linewidth(0.6)
        ax.set_title(
            title
            or str(
                data_meta.get("title", "Tabular Data Snapshot")
                if data_meta
                else "Tabular Data Snapshot"
            ),
            fontsize=TITLE_FONT_SIZE,
            pad=10,
        )
        return self._to_image(fig)

    def scatter_plot(
        self,
        df: Any,
        width: int = 1024,
        height: int = 768,
        title: str = "",
        data_meta: dict[str, Any] | None = None,
        style: str | None = None,
    ) -> Image:
        """Render first two numeric columns as a scatter plot with regression.

        Args:
            df: Input dataframe.
            width: Target image width in pixels.
            height: Target image height in pixels.
            title: Optional title override.
            data_meta: Optional metadata used for axis labels and context.
            style: Optional matplotlib style name or alias.

        Returns:
            Rendered scatter plot as a PIL image.
        """
        if self.suite == "v2":
            return v2.tabular_scatter_plot(df, title=title, data_meta=data_meta, style=style)
        plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        apply_global_style(style)
        numeric = df.select_dtypes(include="number")
        fig, ax = plt.subplots(figsize=figure_size(width, height), dpi=DEFAULT_DPI)
        theme = apply_theme(ax)
        if numeric.shape[1] >= 2:
            x_values = numeric.iloc[:, 0].to_numpy(dtype=float)
            y_values = numeric.iloc[:, 1].to_numpy(dtype=float)
            ax.scatter(
                x_values,
                y_values,
                s=42,
                alpha=0.72,
                color=theme.primary_palette[0],
                edgecolor="white",
                linewidth=0.4,
            )
            if len(x_values) >= 2 and len(np.unique(x_values)) >= 2:
                slope, intercept = np.polyfit(x_values, y_values, deg=1)
                x_line = np.linspace(float(x_values.min()), float(x_values.max()), 100)
                y_line = slope * x_line + intercept
                ax.plot(
                    x_line,
                    y_line,
                    color=theme.primary_palette[3],
                    linewidth=2.0,
                    label="Trend line",
                )
                ax.legend(loc="best", fontsize=ANNOTATION_FONT_SIZE)
            ax.set_xlabel(
                str(
                    data_meta.get("x_label", numeric.columns[0])
                    if data_meta
                    else numeric.columns[0]
                ),
                fontsize=LABEL_FONT_SIZE,
            )
            ax.set_ylabel(
                str(
                    data_meta.get("y_label", numeric.columns[1])
                    if data_meta
                    else numeric.columns[1]
                ),
                fontsize=LABEL_FONT_SIZE,
            )
        else:
            ax.scatter([0], [0], s=50, color=theme.primary_palette[0])
            ax.set_xlabel("Feature 1", fontsize=LABEL_FONT_SIZE)
            ax.set_ylabel("Feature 2", fontsize=LABEL_FONT_SIZE)
        ax.set_title(
            title
            or str(
                data_meta.get("title", "Relationship Between Numeric Variables")
                if data_meta
                else "Relationship Between Numeric Variables"
            ),
            fontsize=TITLE_FONT_SIZE,
        )
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
        return self._to_image(fig)

    def text_only(
        self,
        df: Any,
        width: int = 1024,
        height: int = 768,
        title: str = "",
        data_meta: dict[str, Any] | None = None,
        style: str | None = None,
    ) -> Image:
        """Render tabular values in styled monospaced text format.

        Args:
            df: Input dataframe.
            width: Target image width in pixels.
            height: Target image height in pixels.
            title: Optional title override.
            data_meta: Optional metadata used for title context.
            style: Unused style token kept for API consistency.

        Returns:
            Rendered text image as a PIL image.
        """
        if self.suite == "v2":
            return v2.tabular_text_only(df, title=title, data_meta=data_meta, style=style)
        pil = __import__("PIL.Image", fromlist=["Image"])
        draw_module = __import__("PIL.ImageDraw", fromlist=["ImageDraw"])
        font_module = __import__("PIL.ImageFont", fromlist=["ImageFont"])
        image = pil.new("RGB", (width, height), color=(250, 252, 255))
        draw = draw_module.Draw(image)
        try:
            title_font = font_module.truetype("DejaVuSansMono.ttf", 16)
            body_font = font_module.truetype("DejaVuSansMono.ttf", 12)
        except OSError:
            title_font = font_module.load_default()
            body_font = font_module.load_default()

        text_df = df.head(TEXT_VIEW_MAX_ROWS).copy()
        text_df = text_df.map(
            lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)
        )
        header = " | ".join(str(col) for col in text_df.columns)
        separator = "-" * min(120, len(header) + 6)
        body_lines = text_df.to_string(index=False).splitlines()
        title_text = title or str(
            data_meta.get("title", "Tabular Text View")
            if data_meta
            else "Tabular Text View"
        )
        draw.rectangle((16, 16, width - 16, 54), fill=(220, 230, 242))
        draw.text((24, 24), title_text, fill=(15, 23, 42), font=title_font)
        content = "\n".join([header, separator, *body_lines])
        draw.multiline_text(
            (24, 70), content, fill=(31, 41, 51), font=body_font, spacing=4
        )
        return image

    def render_assist(self, df: Any, style: str | None = None) -> dict[str, Image]:
        """Return the 2x2 calculation-aid images (full table / row bars, with / without means)."""
        return v2.tabular_assist(df, style=style)

    def _to_image(self, fig: Any) -> Image:
        """Convert a matplotlib figure into a PIL image."""
        plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        pil = __import__("PIL.Image", fromlist=["Image"])
        buffer = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buffer, format="png")
        plt.close(fig)
        _ = buffer.seek(0)
        return pil.open(buffer).convert("RGB")
