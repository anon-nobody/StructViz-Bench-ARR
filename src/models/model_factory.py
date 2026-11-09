from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

from .api_models import ClaudeModel, GPT4oModel, GeminiModel
from .local_models import InternVLModel, LlavaModel, QwenVLModel
from .model_wrapper import BaseModel, load_model_config

_MODEL_TYPE_TO_CLASS: dict[str, type[BaseModel]] = {
    "gpt4o": GPT4oModel,
    "claude": ClaudeModel,
    "gemini": GeminiModel,
    "qwen": QwenVLModel,
    "llava": LlavaModel,
    "internvl": InternVLModel,
}


def _filter_kwargs(
    model_cls: type[BaseModel], config: dict[str, Any]
) -> dict[str, Any]:
    allowed = {field_info.name for field_info in fields(model_cls)}
    return {key: value for key, value in config.items() if key in allowed}


def create_model(config_path: Path) -> BaseModel:
    """Create a configured model instance from a YAML config file.

    Args:
        config_path: Path to one model YAML file.

    Returns:
        Instantiated model wrapper.

    Raises:
        ValueError: If required fields are missing or unsupported.
    """
    config = load_model_config(config_path)
    model_type = str(config.get("model_type", "")).strip().lower()
    if not model_type:
        raise ValueError(f"Missing 'model_type' in config: {config_path}")

    model_cls = _MODEL_TYPE_TO_CLASS.get(model_type)
    if model_cls is None:
        supported = ", ".join(sorted(_MODEL_TYPE_TO_CLASS))
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Supported: {supported}"
        )

    kwargs = _filter_kwargs(model_cls, config)
    if "name" not in kwargs:
        kwargs["name"] = str(config.get("name", model_type))
    return model_cls(**kwargs)
