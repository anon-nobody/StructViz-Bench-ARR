from __future__ import annotations

# pyright: reportMissingImports=false

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


PRIMARY_PALETTE = [
    "#1f4e79",
    "#2a9d8f",
    "#f4a261",
    "#e76f51",
    "#6d597a",
    "#577590",
]
SEQUENTIAL_CMAP = "YlGnBu"
DIVERGING_CMAP = "coolwarm"
TEXT_COLOR = "#1f2933"
GRID_COLOR = "#d9e2ec"
BACKGROUND_COLOR = "#fbfdff"
HEADER_BG_COLOR = "#dce6f2"
ROW_ALT_BG_COLOR = "#f3f7fb"

TITLE_FONT_SIZE = 14
LABEL_FONT_SIZE = 11
TICK_FONT_SIZE = 9
ANNOTATION_FONT_SIZE = 9

DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 768
DEFAULT_DPI = 150

DEFAULT_STYLE = "seaborn-v0_8-whitegrid"
STYLE_LIBRARY = {
    "default": "seaborn-v0_8-whitegrid",
    "whitegrid": "seaborn-v0_8-whitegrid",
    "ticks": "seaborn-v0_8-ticks",
}


@dataclass(slots=True)
class ThemeConfig:
    """Styling values shared by all rendering methods."""

    title_size: int = TITLE_FONT_SIZE
    label_size: int = LABEL_FONT_SIZE
    tick_size: int = TICK_FONT_SIZE
    annotation_size: int = ANNOTATION_FONT_SIZE
    dpi: int = DEFAULT_DPI
    text_color: str = TEXT_COLOR
    grid_color: str = GRID_COLOR
    background_color: str = BACKGROUND_COLOR
    primary_palette: tuple[str, ...] = tuple(PRIMARY_PALETTE)
    sequential_cmap: str = SEQUENTIAL_CMAP
    diverging_cmap: str = DIVERGING_CMAP


def resolve_style(style: str | None) -> str:
    """Map a style alias to a matplotlib style name."""
    if not style:
        return DEFAULT_STYLE
    return STYLE_LIBRARY.get(style, style)


def figure_size(width: int, height: int, dpi: int = DEFAULT_DPI) -> tuple[float, float]:
    """Convert pixel dimensions into matplotlib figure dimensions."""
    return width / dpi, height / dpi


def apply_theme(ax: Axes, theme: ThemeConfig | None = None) -> ThemeConfig:
    """Apply a publication-ready visual theme to an axis.

    Args:
        ax: Target matplotlib axis.
        theme: Optional theme override.

    Returns:
        Effective theme object used by the axis.
    """
    config = theme or ThemeConfig()
    ax.set_facecolor(config.background_color)
    ax.figure.set_facecolor("white")
    ax.tick_params(axis="both", labelsize=config.tick_size, colors=config.text_color)
    for spine in ax.spines.values():
        spine.set_alpha(0.25)
        spine.set_color(config.grid_color)
    ax.grid(True, color=config.grid_color, linewidth=0.7, alpha=0.8)
    return config


def apply_global_style(style: str | None = None) -> None:
    """Apply a matplotlib style preset and shared rcParams."""
    plt.style.use(resolve_style(style))
    plt.rcParams["axes.titlesize"] = TITLE_FONT_SIZE
    plt.rcParams["axes.labelsize"] = LABEL_FONT_SIZE
    plt.rcParams["xtick.labelsize"] = TICK_FONT_SIZE
    plt.rcParams["ytick.labelsize"] = TICK_FONT_SIZE
