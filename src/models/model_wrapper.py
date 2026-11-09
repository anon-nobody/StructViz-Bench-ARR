from __future__ import annotations

# pyright: reportMissingImports=false

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Protocol

import yaml


class ModelWrapper(Protocol):
    """Protocol for unified model wrappers."""

    name: str

    def answer(self, question: str, image: object, metadata: Dict[str, Any]) -> str:
        """Answer one benchmark question from image input."""
        ...

    def answer_batch(
        self,
        questions: List[str],
        images: List[object],
        metadata_list: List[Dict[str, Any]],
    ) -> List[str]:
        """Answer a batch of benchmark questions from image input."""
        ...


@dataclass(slots=True)
class BaseModel:
    """Base class for model wrappers."""

    name: str
    max_tokens: int = 256
    temperature: float = 0.0
    system_prompt: str = "Answer concisely. Give only the final answer."

    def answer(self, question: str, image: object, metadata: Dict[str, Any]) -> str:
        """Answer one benchmark question from image input."""
        raise NotImplementedError

    def answer_batch(
        self,
        questions: List[str],
        images: List[object],
        metadata_list: List[Dict[str, Any]],
    ) -> List[str]:
        """Answer a batch by delegating to per-item inference."""
        if not (len(questions) == len(images) == len(metadata_list)):
            raise ValueError(
                "questions, images, and metadata_list must have the same length"
            )
        return [
            self.answer(question=question, image=image, metadata=metadata)
            for question, image, metadata in zip(questions, images, metadata_list)
        ]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> BaseModel:
        """Create an instance from a config mapping."""
        allowed_fields = {field_info.name for field_info in fields(cls)}
        kwargs = {key: value for key, value in config.items() if key in allowed_fields}
        return cls(**kwargs)


def load_model_config(config_path: Path) -> Dict[str, Any]:
    """Load one model configuration YAML file."""
    with config_path.open("r", encoding="utf-8") as file_handle:
        loaded = yaml.safe_load(file_handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return loaded
