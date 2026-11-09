"""Answer-complete ("v2") rendering suite and the tabular calculation-aid ("assist") images.

The v1 renderers reproduce the released benchmark images and are left untouched; each renderer
class dispatches here only when constructed with ``suite="v2"``. v2 images are designed so that
the queried information is visible for (almost) every question: every table row is drawn,
bar charts show row-level values instead of column means, every graph node is labelled, and
text views list every point / edge in multiple columns on a canvas that grows as needed.
Gramian angular fields and recurrence plots stay intentionally lossy (identical to v1).
"""

from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import io
import math
from itertools import combinations
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.rendering.style_config import (
    DEFAULT_DPI,
    HEADER_BG_COLOR,
    PRIMARY_PALETTE,
    ROW_ALT_BG_COLOR,
    TITLE_FONT_SIZE,
    apply_global_style,
    apply_theme,
    figure_size,
)

V2_WIDTH = 1024
V2_MIN_HEIGHT = 768
V2_MAX_CHART_HEIGHT = 1024
V2_MAX_TEXT_HEIGHT = 1024
V2_TABLE_ROW_PX = 24
V2_TABLE_FONT_SIZES = (13, 12, 11, 10, 9)
V2_HEATMAP_PANEL_MAX_ROWS = 60
V2_BAR_ANNOTATE_MAX_ROWS = 35
V2_BAR_PANEL_PX = 300
V2_ROW_LABEL_MAX_CHARS = 12
V2_SCATTER_MAX_PAIR_PANELS = 4
V2_BAR_MAX_PANELS = 6
V2_SCATTER_PAIR_ANNOTATE_MAX_ROWS = 40
V2_TEXT_FONT_SIZE = 12
V2_TEXT_MIN_FONT_SIZE = 8
V2_TS_TEXT_FONT_SIZE = 11
V2_TS_TEXT_MAX_COLUMNS = 4
V2_TS_TEXT_MAX_HEIGHT = 1024
V2_GRAPH_TEXT_MAX_COLUMNS = 8
V2_GRAPH_SQUARE_MIN_NODES = 60
V2_GRAPH_LABEL_MAX_PT = 9
V2_GRAPH_LABEL_MIN_PT = 6
V2_ADJ_LABEL_MIN_PT = 5
V2_ADJ_ALTERNATE_MIN_NODES = 40
V2_LAYOUT_SEED = 42

_TEXT_TOP = 70
_TEXT_BOTTOM = 20
_TEXT_LEFT = 24
_TEXT_BG = (250, 252, 255)
_TEXT_FG = (31, 41, 51)
_TITLE_BG = (220, 230, 242)
_TITLE_FG = (15, 23, 42)


def v2_constants() -> dict[str, Any]:
    """Return the v2 layout constants (recorded in render manifests)."""
    return {
        key: value
        for key, value in globals().items()
        if key.startswith("V2_") and isinstance(value, (int, float, tuple))
    }


# --------------------------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------------------------


def _fig_to_image(fig: Any) -> Image.Image:
    """Convert a matplotlib figure to an RGB PIL image (same procedure as v1)."""
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _mono(size: int, bold: bool = False) -> Any:
    name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _sans(size: int, bold: bool = False) -> Any:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _line_height(size: int) -> int:
    return int(round(size * 1.2)) + 4


def fmt_cell(value: Any) -> str:
    """Cell formatting shared with the v1 table/text views."""
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def fmt_value(value: Any) -> str:
    """Compact exact-ish number formatting for chart annotations."""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return str(val)
    if val.is_integer() and abs(val) < 1e15:
        return str(int(val))
    text = f"{val:.3f}".rstrip("0").rstrip(".")
    return text if text not in ("-0", "") else "0"


def _numeric(df: Any) -> Any:
    return df.select_dtypes(include="number")


def _first_categorical(df: Any) -> Any:
    numeric_cols = set(_numeric(df).columns)
    for column in df.columns:
        if column not in numeric_cols:
            return column
    return None


def row_labels(df: Any, max_chars: int = V2_ROW_LABEL_MAX_CHARS) -> list[str]:
    """Row labels ``"<index>: <first categorical value>"`` (index only if no categorical)."""
    cat = _first_categorical(df)
    labels: list[str] = []
    for pos, idx in enumerate(df.index):
        if cat is None:
            labels.append(str(idx))
            continue
        val = str(df[cat].iloc[pos])
        if len(val) > max_chars:
            val = val[: max_chars - 1] + "~"
        labels.append(f"{idx}: {val}")
    return labels


def _title_bar(draw: Any, width: int, title: str) -> None:
    draw.rectangle((16, 16, width - 16, 54), fill=_TITLE_BG)
    draw.text((24, 24), title, fill=_TITLE_FG, font=_mono(16))


def render_text_columns(
    title: str,
    columns: list[list[str]],
    font_size: int,
    width: int = V2_WIDTH,
    min_height: int = V2_MIN_HEIGHT,
) -> Image.Image:
    """Draw pre-split monospaced text columns side by side under a title bar."""
    lh = _line_height(font_size)
    n_lines = max((len(col) for col in columns), default=0)
    height = max(min_height, _TEXT_TOP + n_lines * lh + _TEXT_BOTTOM)
    image = Image.new("RGB", (width, height), color=_TEXT_BG)
    draw = ImageDraw.Draw(image)
    _title_bar(draw, width, title)
    font = _mono(font_size)
    avail = width - 2 * _TEXT_LEFT
    slot = avail / max(1, len(columns))
    for k, col in enumerate(columns):
        x = _TEXT_LEFT + int(round(k * slot))
        if k > 0:
            draw.line(
                (x - 8, _TEXT_TOP, x - 8, _TEXT_TOP + n_lines * lh),
                fill=(203, 213, 225),
                width=1,
            )
        for i, line in enumerate(col):
            draw.text((x, _TEXT_TOP + i * lh), line, fill=_TEXT_FG, font=font)
    return image


def plan_text_columns(
    header: list[str],
    body: list[str],
    base_font: int,
    max_cols: int,
    max_height: int,
    width: int = V2_WIDTH,
    min_height: int = V2_MIN_HEIGHT,
    min_font: int = V2_TEXT_MIN_FONT_SIZE,
    footer: list[str] | None = None,
) -> tuple[int, list[list[str]]]:
    """Choose font size and column count so every body line is visible.

    Prefers the fewest columns that fit the minimum canvas; otherwise uses the most columns
    allowed and grows the canvas up to ``max_height``; otherwise shrinks the font.
    """
    footer = footer or []
    avail = width - 2 * _TEXT_LEFT
    n = max(1, len(body))
    maxlen = max((len(line) for line in header + body), default=1)
    choice: tuple[int, int] | None = None
    for size in range(base_font, min_font - 1, -1):
        cw = _mono(size).getlength("M")
        lh = _line_height(size)
        gap = 3 * cw
        col_w = maxlen * cw
        possible = max(1, min(max_cols, int((avail + gap) // (col_w + gap))))

        def needed(ncols: int, lh: int = lh) -> int:
            rows = math.ceil(n / ncols)
            return _TEXT_TOP + (len(header) + rows) * lh + _TEXT_BOTTOM + len(footer) * lh

        fit_min = next((c for c in range(1, possible + 1) if needed(c) <= min_height), None)
        if fit_min is not None:
            choice = (size, fit_min)
            break
        if needed(possible) <= max_height:
            choice = (size, possible)
            break
    if choice is None:
        size = min_font
        cw = _mono(size).getlength("M")
        possible = max(1, min(max_cols, int((avail + 3 * cw) // (maxlen * cw + 3 * cw))))
        choice = (size, possible)
    size, ncols = choice
    rows = math.ceil(len(body) / ncols) if body else 0
    columns = [header + body[k * rows : (k + 1) * rows] for k in range(ncols)]
    columns = [col for col in columns if len(col) > len(header)] or [header + body]
    if footer:
        columns[0] = columns[0] + [""] * (max(len(c) for c in columns) - len(columns[0]))
        columns[0] = columns[0] + footer
    return size, columns


# --------------------------------------------------------------------------------------------
# tabular
# --------------------------------------------------------------------------------------------


def tabular_bar_chart(
    df: Any,
    title: str = "",
    data_meta: dict[str, Any] | None = None,
    style: str | None = None,
    means_panel: bool = False,
    width: int = V2_WIDTH,
) -> Image.Image:
    """Row-level bars: one panel per numeric column, one bar per row (value annotated)."""
    apply_global_style(style)
    numeric = _numeric(df)
    all_cols = list(numeric.columns)
    cols = all_cols[:V2_BAR_MAX_PANELS]
    n_rows = len(df)
    labels = row_labels(df)
    n_panels = max(1, len(cols))
    grid_cols = 1 if n_panels <= 3 else 2
    grid_rows = math.ceil(n_panels / grid_cols)
    height = V2_MIN_HEIGHT if n_panels <= 2 else V2_MAX_CHART_HEIGHT
    fig = plt.figure(figsize=figure_size(width, height), dpi=DEFAULT_DPI)
    side_ratio = (0.32 if grid_cols == 1 else 0.42) if means_panel else 0.0
    if means_panel:
        grid = fig.add_gridspec(
            grid_rows, grid_cols + 1, width_ratios=[1.0] * grid_cols + [side_ratio]
        )
    else:
        grid = fig.add_gridspec(grid_rows, grid_cols)
    # Rotated tick/annotation text must fit the horizontal pixels available per bar
    # (1pt = dpi/72 px; rotated text width ~ 1.2 * font size).
    panel_px = 0.92 * width / (grid_cols + side_ratio) - 55
    px_per_bar = panel_px / max(1, n_rows)
    fit_pt = px_per_bar / (1.2 * DEFAULT_DPI / 72)
    tick_fs = float(np.clip(fit_pt, 4.0, 7.0))
    ann_fs = float(np.clip(fit_pt, 4.0, 6.0))
    rotate = n_rows > 8 or grid_cols > 1 or means_panel
    x = np.arange(n_rows)
    for k in range(grid_rows * grid_cols):
        ax = fig.add_subplot(grid[k // grid_cols, k % grid_cols])
        if k >= len(cols):
            ax.axis("off")
            if not cols and k == 0:
                ax.text(0.5, 0.5, "(no numeric columns)", ha="center", va="center")
            continue
        theme = apply_theme(ax)
        col = cols[k]
        vals = numeric[col].to_numpy(dtype=float)
        ax.bar(
            x,
            np.nan_to_num(vals),
            width=0.8,
            color=theme.primary_palette[k % len(theme.primary_palette)],
            alpha=0.9,
        )
        ax.set_title(str(col), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=tick_fs)
        ax.tick_params(axis="y", labelsize=7)
        ax.axhline(0, color="#475569", linewidth=0.6)
        ax.grid(axis="x", visible=False)
        finite = vals[np.isfinite(vals)] if vals.size else np.array([0.0])
        top = max(float(finite.max()) if finite.size else 0.0, 0.0)
        bottom = min(float(finite.min()) if finite.size else 0.0, 0.0)
        span = (top - bottom) or 1.0
        head = 0.55 if rotate else 0.15
        ax.set_ylim(bottom - (head * span if bottom < 0 else 0.0), top + head * span)
        ax.set_xlim(-0.6, n_rows - 0.4)
        if n_rows <= V2_BAR_ANNOTATE_MAX_ROWS:
            for xi, val in zip(x, vals):
                if not np.isfinite(val):
                    continue
                ax.annotate(
                    fmt_value(val),
                    (xi, val),
                    xytext=(0, 2 if val >= 0 else -2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if val >= 0 else "top",
                    rotation=90 if rotate else 0,
                    fontsize=ann_fs,
                    color="#111827",
                )
    if means_panel:
        side = fig.add_subplot(grid[:, grid_cols])
        side.axis("off")
        lines = ["Column means", f"(all {n_rows} rows)", ""]
        for col in all_cols:
            name = str(col)
            if len(name) > 18:
                name = name[:17] + "~"
            lines.append(f"{name}:")
            lines.append(f"  {float(numeric[col].mean()):.2f}")
        side.text(
            0.02,
            0.98,
            "\n".join(lines),
            va="top",
            ha="left",
            family="monospace",
            fontsize=8 if len(all_cols) <= 8 else 6.5,
            bbox={"boxstyle": "round", "facecolor": "#f1f5f9", "edgecolor": "#94a3b8"},
            transform=side.transAxes,
        )
    default_title = f"Row values by numeric column ({len(cols)} columns x {n_rows} rows)"
    if len(all_cols) > len(cols):
        default_title = (
            f"Row values by numeric column (first {len(cols)} of {len(all_cols)} numeric "
            f"columns x {n_rows} rows)"
        )
    fig.suptitle(
        title or str(data_meta.get("title", default_title) if data_meta else default_title),
        fontsize=12,
    )
    return _fig_to_image(fig)


def tabular_scatter_plot(
    df: Any,
    title: str = "",
    data_meta: dict[str, Any] | None = None,
    style: str | None = None,
    width: int = V2_WIDTH,
) -> Image.Image:
    """First-two-column scatter with every point labelled, plus up to a 2x2 grid of the
    remaining column pairs with the strongest absolute Pearson correlation."""
    apply_global_style(style)
    numeric = _numeric(df)
    cols = list(numeric.columns)
    labels = row_labels(df, max_chars=8)
    idx_labels = [str(i) for i in df.index]
    pairs = [p for p in combinations(range(len(cols)), 2) if p != (0, 1)]
    if pairs:
        corr = numeric.astype(float).corr().abs().fillna(0.0).to_numpy()
        pairs.sort(key=lambda p: (-float(corr[p[0], p[1]]), p))
    pairs = pairs[:V2_SCATTER_MAX_PAIR_PANELS]
    height = V2_MAX_CHART_HEIGHT if pairs else V2_MIN_HEIGHT
    fig = plt.figure(figsize=figure_size(width, height), dpi=DEFAULT_DPI)
    if pairs:
        pair_rows = math.ceil(len(pairs) / 2)
        grid = fig.add_gridspec(1 + pair_rows, 2, height_ratios=[2.2] + [1.0] * pair_rows)
        ax = fig.add_subplot(grid[0, :])
    else:
        ax = fig.add_subplot(1, 1, 1)
    theme = apply_theme(ax)
    if len(cols) >= 2:
        xv = numeric.iloc[:, 0].to_numpy(dtype=float)
        yv = numeric.iloc[:, 1].to_numpy(dtype=float)
        x_name, y_name = str(cols[0]), str(cols[1])
    elif len(cols) == 1:
        xv = np.arange(len(df), dtype=float)
        yv = numeric.iloc[:, 0].to_numpy(dtype=float)
        x_name, y_name = "row index", str(cols[0])
    else:
        xv, yv = np.array([0.0]), np.array([0.0])
        x_name, y_name = "Feature 1", "Feature 2"
        labels = ["(no numeric columns)"]
    ax.scatter(xv, yv, s=36, alpha=0.8, color=theme.primary_palette[0], edgecolor="white",
               linewidth=0.4)
    finite = np.isfinite(xv) & np.isfinite(yv)
    if len(cols) >= 2 and finite.sum() >= 2 and len(np.unique(xv[finite])) >= 2:
        slope, intercept = np.polyfit(xv[finite], yv[finite], deg=1)
        x_line = np.linspace(float(xv[finite].min()), float(xv[finite].max()), 100)
        ax.plot(x_line, slope * x_line + intercept, color=theme.primary_palette[3],
                linewidth=1.6, label="Trend line")
        ax.legend(loc="best", fontsize=7)
    for xi, yi, lab in zip(xv, yv, labels):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.annotate(lab, (xi, yi), xytext=(3, 3), textcoords="offset points", fontsize=6,
                        color="#0f172a")
    ax.set_xlabel(x_name, fontsize=10)
    ax.set_ylabel(y_name, fontsize=10)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
    default_title = "Relationship Between Numeric Variables (points labelled by row)"
    ax.set_title(
        title or str(data_meta.get("title", default_title) if data_meta else default_title),
        fontsize=12,
    )
    for k, (i, j) in enumerate(pairs):
        sub = fig.add_subplot(grid[1 + k // 2, k % 2])
        apply_theme(sub)
        xs = numeric.iloc[:, i].to_numpy(dtype=float)
        ys = numeric.iloc[:, j].to_numpy(dtype=float)
        sub.scatter(xs, ys, s=12, alpha=0.8, color=theme.primary_palette[(k + 1) % 6])
        if len(df) <= V2_SCATTER_PAIR_ANNOTATE_MAX_ROWS:
            for xi, yi, lab in zip(xs, ys, idx_labels):
                if np.isfinite(xi) and np.isfinite(yi):
                    sub.annotate(lab, (xi, yi), xytext=(2, 2), textcoords="offset points",
                                 fontsize=5, color="#334155")
        sub.set_xlabel(str(cols[i])[:28], fontsize=7)
        sub.set_ylabel(str(cols[j])[:28], fontsize=7)
        sub.tick_params(labelsize=6)
    return _fig_to_image(fig)


def _table_layout(
    header: list[str], rows: list[list[str]], size: int, pad: int = 6
) -> list[float]:
    """Column widths; header, index column and summary rows are measured in bold."""
    font, bold = _sans(size), _sans(size, bold=True)
    widths = [bold.getlength(h) for h in header]
    for row in rows:
        for c, cell in enumerate(row):
            widths[c] = max(widths[c], (bold if c == 0 else font).getlength(cell))
    return [w + 2 * pad for w in widths]


def _balanced_groups(widths: list[float], avail: float) -> list[list[int]]:
    """Split data columns 1..n into contiguous groups of similar width, each fitting ``avail``
    together with the repeated index column 0."""
    cap = avail - widths[0]
    data = widths[1:]
    k = max(2, math.ceil(sum(data) / cap))
    target = sum(data) / k
    groups: list[list[int]] = []
    current: list[int] = []
    used = 0.0
    for c in range(1, len(widths)):
        w = widths[c]
        if current and (used + w > cap or used + w / 2 > target * (len(groups) + 1) - sum(
            widths[i] for g in groups for i in g
        )):
            groups.append(current)
            current, used = [], 0.0
        current.append(c)
        used += w
    if current:
        groups.append(current)
    return groups


def tabular_table_image(
    df: Any,
    title: str = "",
    data_meta: dict[str, Any] | None = None,
    style: str | None = None,
    extra_rows: list[tuple[str, dict[Any, str]]] | None = None,
    width: int = V2_WIDTH,
) -> Image.Image:
    """Full table (all rows and columns) drawn with PIL on a canvas of at most 1024x1024.

    Tables too wide for the canvas are split into column groups stacked vertically (each
    repeats the row-index column); font and row height shrink until everything fits. ``extra_rows`` appends labelled summary
    rows (e.g. ``("mean", {col: "12.34"})``) with a distinct background.
    """
    del style
    cells = df.map(fmt_cell)
    col_names = [str(c) for c in df.columns]
    body = [[str(idx)] + list(cells.iloc[pos]) for pos, idx in enumerate(df.index)]
    extras = [
        [label] + [values.get(c, "") for c in df.columns] for label, values in (extra_rows or [])
    ]
    header = [""] + col_names
    avail = width - 32
    n_table_rows = 1 + len(body) + len(extras)
    title_h = 60
    group_gap = 14

    def stacked_height(n_groups: int, rh: int) -> int:
        return title_h + n_groups * n_table_rows * rh + (n_groups - 1) * group_gap + 16

    # Largest font (then roomiest row height) whose layout fits the 1024px cap; column groups
    # are stacked vertically when the table is too wide for one group at that font.
    choice = None
    for size in V2_TABLE_FONT_SIZES:
        widths = _table_layout(header, body + extras, size)
        groups = (
            [list(range(1, len(header)))] if sum(widths) <= avail
            else _balanced_groups(widths, avail)
        )
        for row_h in (max(V2_TABLE_ROW_PX, size + 11), size + 9, size + 7):
            if stacked_height(len(groups), row_h) <= V2_MAX_TEXT_HEIGHT:
                choice = (size, widths, groups, row_h)
                break
        if choice:
            break
    halves = 1
    if choice is None:
        size = V2_TABLE_FONT_SIZES[-1]
        widths = _table_layout(header, body + extras, size)
        groups = (
            [list(range(1, len(header)))] if sum(widths) <= avail
            else _balanced_groups(widths, avail)
        )
        if len(groups) == 1 and sum(widths) * 2 + 24 <= avail:
            halves = 2
            row_h = max(size + 7, (V2_MAX_TEXT_HEIGHT - title_h - 16) // (
                math.ceil(len(body) / 2) + len(extras) + 1))
        else:
            row_h = max(12, (V2_MAX_TEXT_HEIGHT - title_h - 16 - (len(groups) - 1) * group_gap)
                        // (len(groups) * n_table_rows))
        choice = (size, widths, groups, row_h)
    font_size, widths, groups, row_h = choice
    font = _sans(font_size)
    bold = _sans(font_size, bold=True)
    if halves == 2:
        half = math.ceil(len(body) / 2)
        blocks = [(groups[0], body[:half], []), (groups[0], body[half:], extras)]
        rows_max = max(len(b[1]) + len(b[2]) for b in blocks) + 1
        needed = title_h + rows_max * row_h + 16
    else:
        blocks = [(g, body, extras) for g in groups]
        needed = stacked_height(len(groups), row_h)
    height = max(V2_MIN_HEIGHT, needed)
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    title_text = title or str(
        data_meta.get("title", "Tabular Data (all rows)") if data_meta else "Tabular Data (all rows)"
    )
    tf = _sans(TITLE_FONT_SIZE + 2, bold=True)
    draw.text(((width - tf.getlength(title_text)) / 2, 18), title_text, fill=_TITLE_FG, font=tf)

    def draw_block(x0: float, y0: float, cols: list[int], rows: list[list[str]],
                   extra: list[list[str]]) -> float:
        idxs = [0] + cols
        y = y0
        all_rows = [header] + rows + extra
        for r, row in enumerate(all_rows):
            if r == 0:
                fill = HEADER_BG_COLOR
            elif r > len(rows):
                fill = "#fde68a"
            elif r % 2 == 0:
                fill = ROW_ALT_BG_COLOR
            else:
                fill = "white"
            x = x0
            for c in idxs:
                w = widths[c]
                draw.rectangle((x, y, x + w, y + row_h), fill=fill, outline="#d0d7de")
                text = row[c]
                f = bold if (r == 0 or c == 0 or r > len(rows)) else font
                tw = f.getlength(text)
                draw.text((x + (w - tw) / 2, y + (row_h - font_size) / 2 - 1), text,
                          fill="#0f172a", font=f)
                x += w
            y += row_h
        return y

    y = title_h
    if halves == 2:
        block_w = sum(widths[c] for c in [0] + groups[0])
        x_start = (width - (2 * block_w + 24)) / 2
        for k, (cols, rows, extra) in enumerate(blocks):
            draw_block(x_start + k * (block_w + 24), y, cols, rows, extra)
    else:
        for cols, rows, extra in blocks:
            block_w = sum(widths[c] for c in [0] + cols)
            y = draw_block((width - block_w) / 2, y, cols, rows, extra) + group_gap
    return image


def tabular_text_only(
    df: Any,
    title: str = "",
    data_meta: dict[str, Any] | None = None,
    style: str | None = None,
    width: int = V2_WIDTH,
) -> Image.Image:
    """Monospaced text view of every row (with row index); 2 columns or taller canvas if needed."""
    del style
    title_text = title or str(
        data_meta.get("title", "Tabular Text View (all rows)")
        if data_meta
        else "Tabular Text View (all rows)"
    )
    text_df = df.map(fmt_cell)
    avail = width - 2 * _TEXT_LEFT
    for size in range(V2_TEXT_FONT_SIZE, V2_TEXT_MIN_FONT_SIZE - 1, -1):
        cw = _mono(size).getlength("M")
        lh = _line_height(size)
        lines = text_df.to_string(index=True).splitlines()
        maxlen = max((len(line) for line in lines), default=1)
        if maxlen * cw <= avail:
            one_col_h = _TEXT_TOP + len(lines) * lh + _TEXT_BOTTOM
            if one_col_h <= V2_MIN_HEIGHT:
                return render_text_columns(title_text, [lines], size, width)
            if len(df) > 1 and 2 * maxlen * cw + 3 * cw <= avail:
                half = math.ceil(len(df) / 2)
                cols = [
                    text_df.iloc[:half].to_string(index=True).splitlines(),
                    text_df.iloc[half:].to_string(index=True).splitlines(),
                ]
                h = _TEXT_TOP + max(len(c) for c in cols) * lh + _TEXT_BOTTOM
                if h <= V2_MAX_TEXT_HEIGHT:
                    return render_text_columns(title_text, cols, size, width)
            if one_col_h <= V2_MAX_TEXT_HEIGHT:
                return render_text_columns(title_text, [lines], size, width)
            continue
        wrapped = text_df.to_string(index=True, line_width=int(avail // cw)).splitlines()
        wrapped = [line.rstrip(" \\") if line.endswith("\\") else line for line in wrapped]
        if max(len(line) for line in wrapped) * cw <= avail + cw and (
            _TEXT_TOP + len(wrapped) * lh + _TEXT_BOTTOM <= V2_MAX_TEXT_HEIGHT
            or size == V2_TEXT_MIN_FONT_SIZE
        ):
            return render_text_columns(title_text, [wrapped], size, width)
    size = V2_TEXT_MIN_FONT_SIZE
    cw = _mono(size).getlength("M")
    wrapped = text_df.to_string(index=True, line_width=int(avail // cw)).splitlines()
    return render_text_columns(title_text, [wrapped], size, width)


def tabular_heatmap(
    df: Any,
    title: str = "",
    data_meta: dict[str, Any] | None = None,
    style: str | None = None,
    width: int = V2_WIDTH,
) -> Image.Image:
    """Annotated heatmap of every row (<=60 per panel), row ticks ``idx: category``.

    Colour is min-max scaled within each column so columns with different units remain
    readable; the exact value is printed in every cell.
    """
    apply_global_style(style)
    sns = __import__("seaborn")
    pd = __import__("pandas")
    numeric = _numeric(df)
    if numeric.empty:
        numeric = pd.DataFrame({"(no numeric columns)": [0.0] * max(1, len(df))},
                               index=df.index if len(df) else [0])
    labels = row_labels(df) if len(df) else ["0"]
    n_rows = len(numeric)
    n_panels = 1 if n_rows <= V2_HEATMAP_PANEL_MAX_ROWS else 2
    per_panel = math.ceil(n_rows / n_panels)
    height = int(min(V2_MAX_CHART_HEIGHT, max(V2_MIN_HEIGHT, 150 + per_panel * 22)))
    fig, axes = plt.subplots(1, n_panels, figsize=figure_size(width, height), dpi=DEFAULT_DPI,
                             squeeze=False)
    values = numeric.to_numpy(dtype=float)
    col_min = np.nanmin(values, axis=0) if values.size else np.zeros(1)
    col_max = np.nanmax(values, axis=0) if values.size else np.ones(1)
    rng = np.where((col_max - col_min) > 0, col_max - col_min, 1.0)
    scaled = (values - col_min) / rng
    annot = np.vectorize(fmt_value)(values) if values.size else values
    n_cols = numeric.shape[1]
    ann_fs = 8 if n_cols <= 6 else (7 if n_cols <= 10 else 6)
    if n_panels == 2:
        ann_fs = max(5, ann_fs - 2)
    for p in range(n_panels):
        ax = axes[0][p]
        apply_theme(ax)
        lo, hi = p * per_panel, min(n_rows, (p + 1) * per_panel)
        sns.heatmap(
            scaled[lo:hi],
            ax=ax,
            cmap="YlGnBu",
            annot=annot[lo:hi],
            fmt="",
            linewidths=0.4,
            linecolor="white",
            cbar=False,
            vmin=0.0,
            vmax=1.0,
            annot_kws={"fontsize": ann_fs},
            xticklabels=[str(c) for c in numeric.columns],
            yticklabels=labels[lo:hi],
        )
        ax.tick_params(axis="y", labelsize=7 if per_panel <= 35 else 6, rotation=0)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        ax.set_xlabel("")
        ax.set_ylabel("")
    default_title = "Tabular Value Heatmap (all rows; colour scaled per column)"
    fig.suptitle(
        title or str(data_meta.get("title", default_title) if data_meta else default_title),
        fontsize=12,
    )
    return _fig_to_image(fig)


def tabular_assist(df: Any, style: str | None = None) -> dict[str, Image.Image]:
    """2x2 calculation-aid images: {table, bar} x {without, with} precomputed column means."""
    numeric = _numeric(df)
    means = {col: f"{float(numeric[col].mean()):.2f}" for col in numeric.columns}
    return {
        "table_full": tabular_table_image(df, style=style),
        "table_full_means": tabular_table_image(df, style=style, extra_rows=[("mean", means)]),
        "bar_rows": tabular_bar_chart(df, style=style),
        "bar_rows_means": tabular_bar_chart(df, style=style, means_panel=True),
    }


# --------------------------------------------------------------------------------------------
# time series
# --------------------------------------------------------------------------------------------


def timeseries_text_only(
    arr: np.ndarray,
    title: str = "",
    data_meta: dict[str, object] | None = None,
    width: int = V2_WIDTH,
) -> Image.Image:
    """Every point as ``timestep | value`` in up to four columns (11pt mono)."""
    title_text = title or str(
        data_meta.get("title", "Time Series Text View (all points)")
        if data_meta
        else "Time Series Text View (all points)"
    )
    header = ["timestep | value", "---------+---------"]
    body = [f"{idx:>8d} | {float(value):>8.4f}" for idx, value in enumerate(arr)]
    size, columns = plan_text_columns(
        header,
        body,
        base_font=V2_TS_TEXT_FONT_SIZE,
        max_cols=V2_TS_TEXT_MAX_COLUMNS,
        max_height=V2_TS_TEXT_MAX_HEIGHT,
        width=width,
    )
    return render_text_columns(title_text, columns, size, width)


# --------------------------------------------------------------------------------------------
# graph
# --------------------------------------------------------------------------------------------


def _graph_canvas_height(n_nodes: int) -> int:
    return V2_WIDTH if n_nodes >= V2_GRAPH_SQUARE_MIN_NODES else V2_MIN_HEIGHT


def _graph_label_pt(n_nodes: int) -> int:
    if n_nodes < 30:
        return V2_GRAPH_LABEL_MAX_PT
    pt = int(round(V2_GRAPH_LABEL_MAX_PT - (n_nodes - 30) / 15))
    return max(V2_GRAPH_LABEL_MIN_PT, min(V2_GRAPH_LABEL_MAX_PT, pt))


def _node_base_size(graph: nx.Graph, label_pt: int) -> float:
    """Marker area (pt^2) whose diameter fits the longest node label."""
    chars = max((len(str(node)) for node in graph.nodes()), default=1)
    diameter = label_pt * 0.62 * chars + label_pt * 0.9
    return float(diameter**2)


def _lighten(hex_color: str, amount: float = 0.55) -> tuple[float, float, float]:
    rgb = np.array([int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)])
    return tuple(rgb + (1 - rgb) * amount)  # type: ignore[return-value]


def graph_node_link(
    graph: nx.Graph,
    title: str = "",
    data_meta: dict[str, object] | None = None,
    style: str | None = None,
    width: int = V2_WIDTH,
) -> Image.Image:
    """Spring layout (seed 42, as v1) with every node labelled."""
    apply_global_style(style)
    n = graph.number_of_nodes()
    fig, ax = plt.subplots(figsize=figure_size(width, _graph_canvas_height(n)), dpi=DEFAULT_DPI)
    apply_theme(ax)
    pos = nx.spring_layout(graph, seed=V2_LAYOUT_SEED)
    label_pt = _graph_label_pt(n)
    base = _node_base_size(graph, label_pt)
    degrees = dict(graph.degree())
    max_deg = max(degrees.values(), default=1) or 1
    sizes = [base * (1.0 + 0.6 * degrees[node] / max_deg) for node in graph.nodes()]
    nx.draw_networkx_edges(graph, pos=pos, ax=ax, width=0.8, alpha=0.45, edge_color="#64748b")
    nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_size=sizes,
                           node_color=[_lighten(PRIMARY_PALETTE[0])], edgecolors="#1f4e79",
                           linewidths=0.6)
    nx.draw_networkx_labels(graph, pos=pos, ax=ax, font_size=label_pt, font_color="#0f172a")
    default_title = "Graph Structure (Node-Link View, all nodes labelled)"
    ax.set_title(
        title or str(data_meta.get("title", default_title) if data_meta else default_title),
        fontsize=TITLE_FONT_SIZE,
    )
    ax.axis("off")
    ax.margins(0.03)
    return _fig_to_image(fig)


def graph_circular_layout(
    graph: nx.Graph,
    communities: dict[object, int],
    title: str = "",
    data_meta: dict[str, object] | None = None,
    style: str | None = None,
    width: int = V2_WIDTH,
) -> Image.Image:
    """Circular layout coloured by community with every node labelled."""
    apply_global_style(style)
    n = graph.number_of_nodes()
    fig, ax = plt.subplots(figsize=figure_size(width, _graph_canvas_height(n)), dpi=DEFAULT_DPI)
    apply_theme(ax)
    pos = nx.circular_layout(graph)
    label_pt = _graph_label_pt(n)
    base = _node_base_size(graph, label_pt)
    degrees = dict(graph.degree())
    max_deg = max(degrees.values(), default=1) or 1
    sizes = [base * (1.0 + 0.4 * degrees[node] / max_deg) for node in graph.nodes()]
    colors = [
        _lighten(PRIMARY_PALETTE[communities.get(node, 0) % len(PRIMARY_PALETTE)], 0.45)
        for node in graph.nodes()
    ]
    if graph.number_of_edges() > 0:
        nx.draw_networkx_edges(graph, pos=pos, ax=ax, width=0.8, edge_color="#64748b",
                               alpha=0.5)
    nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_size=sizes, node_color=colors,
                           edgecolors="#334155", linewidths=0.6)
    nx.draw_networkx_labels(graph, pos=pos, ax=ax, font_size=label_pt, font_color="#0f172a")
    default_title = "Graph Circular Layout by Community (all nodes labelled)"
    ax.set_title(
        title or str(data_meta.get("title", default_title) if data_meta else default_title),
        fontsize=TITLE_FONT_SIZE,
    )
    ax.axis("off")
    ax.set_aspect("equal")
    ax.margins(0.03)
    return _fig_to_image(fig)


def graph_adjacency_matrix(
    graph: nx.Graph,
    title: str = "",
    data_meta: dict[str, object] | None = None,
    style: str | None = None,
    width: int = V2_WIDTH,
) -> Image.Image:
    """Adjacency matrix on a square canvas with a tick label for every node.

    For large graphs even-indexed labels sit on the bottom/left axes and odd-indexed labels on
    the top/right axes, doubling the label spacing so each stays legible.
    """
    apply_global_style(style)
    nodes = list(graph.nodes())
    n = len(nodes)
    matrix = nx.to_numpy_array(graph, nodelist=nodes)
    fig, ax = plt.subplots(figsize=figure_size(width, width), dpi=DEFAULT_DPI)
    apply_theme(ax)
    ax.grid(False)
    binary = bool(matrix.size) and set(np.unique(matrix)).issubset({0.0, 1.0})
    im = ax.imshow(matrix, cmap="Blues" if binary else "YlGnBu", interpolation="nearest",
                   vmin=0.0)
    if not binary:
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.set_ylabel("Edge weight", fontsize=9)
    for k in range(n + 1):
        strong = k % 10 == 0
        ax.axhline(k - 0.5, color="#94a3b8" if strong else "#e2e8f0",
                   linewidth=0.6 if strong else 0.3)
        ax.axvline(k - 0.5, color="#94a3b8" if strong else "#e2e8f0",
                   linewidth=0.6 if strong else 0.3)
    labels = [str(node) for node in nodes]
    fs = V2_GRAPH_LABEL_MAX_PT if n <= 24 else max(V2_ADJ_LABEL_MIN_PT, int(9 * 24 / n))
    if n > V2_ADJ_ALTERNATE_MIN_NODES:
        fs = max(V2_ADJ_LABEL_MIN_PT, min(8, int(2 * 9 * 24 / n)))
        even = list(range(0, n, 2))
        odd = list(range(1, n, 2))
        ax.set_xticks(even)
        ax.set_xticklabels([labels[i] for i in even], rotation=90, fontsize=fs)
        ax.set_yticks(even)
        ax.set_yticklabels([labels[i] for i in even], fontsize=fs)
        ax.set_xticks(odd, minor=True)
        ax.set_xticklabels([labels[i] for i in odd], minor=True, rotation=90, fontsize=fs)
        ax.set_yticks(odd, minor=True)
        ax.set_yticklabels([labels[i] for i in odd], minor=True, fontsize=fs)
        ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False, top=True,
                       labeltop=True, length=2)
        ax.tick_params(axis="y", which="minor", left=False, labelleft=False, right=True,
                       labelright=True, length=2)
        ax.tick_params(axis="both", which="major", length=2)
        subtitle = " (even-position labels bottom/left, odd top/right)"
    else:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90 if n > 24 else 45, fontsize=fs)
        ax.set_yticklabels(labels, fontsize=fs)
        subtitle = ""
    if matrix.size and n <= 15:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if matrix[i, j] > 0.5 * matrix.max() else "#111827")
    ax.set_xlabel(str(data_meta.get("x_label", "Target Node") if data_meta else "Target Node"),
                  fontsize=9)
    ax.set_ylabel(str(data_meta.get("y_label", "Source Node") if data_meta else "Source Node"),
                  fontsize=9)
    default_title = "Graph Adjacency Matrix" + subtitle
    ax.set_title(
        title or str(data_meta.get("title", default_title) if data_meta else default_title),
        fontsize=11,
    )
    return _fig_to_image(fig)


def graph_text_only(
    graph: nx.Graph,
    title: str = "",
    data_meta: dict[str, object] | None = None,
    with_degree: bool = False,
    width: int = V2_WIDTH,
) -> Image.Image:
    """Every edge ``u -- v`` in multiple columns (optionally with v1's degree columns).

    Nodes with no incident edge are listed on a footer line so the node set is complete.
    """
    default = "Graph Text View (all edges" + (
        "; du, dv = degree of u, v)" if with_degree else ")"
    )
    title_text = title or str(data_meta.get("title", default) if data_meta else default)
    edges = list(graph.edges())
    degrees = dict(graph.degree())
    lw = max((len(str(node)) for node in graph.nodes()), default=1)
    if with_degree:
        dw = max((len(str(d)) for d in degrees.values()), default=1)
        header = ["u -- v | du,dv", "--------------"]
        body = [
            f"{str(u):>{lw}} -- {str(v):<{lw}} | {degrees.get(u, 0):>{dw}d},"
            f"{degrees.get(v, 0):>{dw}d}"
            for u, v in edges
        ]
    else:
        header = ["u -- v", "------"]
        body = [f"{str(u):>{lw}} -- {str(v):<{lw}}" for u, v in edges]
    if not edges:
        body = ["(no edges)"]
    isolated = [str(node) for node in graph.nodes() if degrees.get(node, 0) == 0]
    footer = [f"nodes without edges: {', '.join(isolated)}"] if isolated else None
    size, columns = plan_text_columns(
        header,
        body,
        base_font=V2_TEXT_FONT_SIZE,
        max_cols=V2_GRAPH_TEXT_MAX_COLUMNS,
        max_height=V2_MAX_TEXT_HEIGHT,
        width=width,
        footer=footer,
    )
    return render_text_columns(title_text, columns, size, width)
