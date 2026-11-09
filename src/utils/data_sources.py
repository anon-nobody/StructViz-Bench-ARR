from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DataSourceSpec:
    name: str
    root_url: str
    citation: str


def get_default_sources() -> dict[str, DataSourceSpec]:
    return {
        "uci": DataSourceSpec(
            name="UCI Machine Learning Repository",
            root_url="https://archive.ics.uci.edu/",
            citation="Dua and Graff, UCI ML Repository",
        ),
        "monash": DataSourceSpec(
            name="Monash Time Series Repository",
            root_url="https://forecastingdata.org/",
            citation="Godahewa et al., Monash TS Forecasting Archive",
        ),
        "ogb": DataSourceSpec(
            name="Open Graph Benchmark",
            root_url="https://ogb.stanford.edu/",
            citation="Hu et al., Open Graph Benchmark",
        ),
    }


def list_source_names() -> list[str]:
    return sorted(get_default_sources())
