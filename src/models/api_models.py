from __future__ import annotations

import base64
import io
import importlib
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from .model_wrapper import BaseModel
from .rate_limiter import RateLimiter
from .response_parser import parse_answer


def _encode_image_to_base64_png(image: Any) -> str:
    """Encode a PIL image as a base64 PNG string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _extract_openai_content(content: Any) -> str:
    """Extract text content from OpenAI-style response message content."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif hasattr(block, "text"):
                parts.append(str(getattr(block, "text")))
        return "\n".join(part for part in parts if part)

    return str(content)


def _run_with_retry(fn: Callable[[], str], retries: int = 3) -> str:
    """Execute a callable with exponential-backoff retry."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as error:  # noqa: BLE001
            last_error = error
            if attempt == retries - 1:
                break
            time.sleep(2**attempt)
    raise RuntimeError("Request failed after retries") from last_error


_STRUCTVIZ_SYSTEM_PROMPT = (
    "You are a precise visual data analyst. "
    "Look at the image and answer the question. "
    "Reply with ONLY the final answer value — no explanation, no units, no sentence.\n"
    "\n"
    "Answer format rules:\n"
    "- For yes/no questions: answer exactly 'yes' or 'no'\n"
    "- For numeric questions: answer with just the number (e.g. 48.17)\n"
    "- For name/label questions: answer with just the name (e.g. east_grid)\n"
    "- For direction questions: answer with just the word (e.g. increasing, higher, first half)\n"
    "- If no clear answer exists: answer 'none'\n"
    "- NEVER start with 'The', 'Based on', 'I can', or any other preamble\n"
    "- NEVER include explanation or reasoning\n"
)

@dataclass(slots=True)
class GPT4oModel(BaseModel):
    """Wrapper for GPT-4o style API models via OpenAI API."""

    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 256
    temperature: float = 0.0
    system_prompt: str = _STRUCTVIZ_SYSTEM_PROMPT
    timeout_seconds: float = 30.0
    requests_per_minute: int = 60

    def answer(self, question: str, image: object, metadata: dict[str, Any]) -> str:
        """Answer one question with GPT-4o vision API."""
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(
                f"Missing API key in environment variable: {self.api_key_env}"
            )

        limiter = RateLimiter(requests_per_minute=self.requests_per_minute)
        limiter.wait()

        encoded_image = _encode_image_to_base64_png(image)
        openai_module = importlib.import_module("openai")
        client = openai_module.OpenAI(api_key=api_key)

        def _request() -> str:
            response = client.chat.completions.create(
                model=self.name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout_seconds,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded_image}"
                                },
                            },
                        ],
                    },
                ],
            )
            raw = _extract_openai_content(response.choices[0].message.content)
            task = str(metadata.get("task", "generic"))
            return parse_answer(raw_response=raw, task=task)

        return _run_with_retry(_request)


@dataclass(slots=True)
class ClaudeModel(BaseModel):
    """Wrapper for Claude API models."""

    api_key_env: str = "ANTHROPIC_API_KEY"
    max_tokens: int = 256
    temperature: float = 0.0
    system_prompt: str = _STRUCTVIZ_SYSTEM_PROMPT
    timeout_seconds: float = 30.0
    requests_per_minute: int = 60

    def answer(self, question: str, image: object, metadata: dict[str, Any]) -> str:
        """Answer one question with Anthropic Claude messages API."""
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(
                f"Missing API key in environment variable: {self.api_key_env}"
            )

        limiter = RateLimiter(requests_per_minute=self.requests_per_minute)
        limiter.wait()

        encoded_image = _encode_image_to_base64_png(image)
        anthropic_module = importlib.import_module("anthropic")
        client = anthropic_module.Anthropic(
            api_key=api_key, timeout=self.timeout_seconds
        )

        def _request() -> str:
            response = client.messages.create(
                model=self.name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": encoded_image,
                                },
                            },
                            {"type": "text", "text": question},
                        ],
                    }
                ],
            )
            raw = "\n".join(
                block.text
                for block in response.content
                if hasattr(block, "text") and block.text
            )
            task = str(metadata.get("task", "generic"))
            return parse_answer(raw_response=raw, task=task)

        return _run_with_retry(_request)


@dataclass(slots=True)
class GeminiModel(BaseModel):
    """Wrapper for Gemini models using OpenAI-compatible endpoint."""

    api_key_env: str = "GEMINI_API_KEY"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    max_tokens: int = 256
    temperature: float = 0.0
    system_prompt: str = _STRUCTVIZ_SYSTEM_PROMPT
    timeout_seconds: float = 30.0
    requests_per_minute: int = 60

    def answer(self, question: str, image: object, metadata: dict[str, Any]) -> str:
        """Answer one question with Gemini vision API."""
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(
                f"Missing API key in environment variable: {self.api_key_env}"
            )

        limiter = RateLimiter(requests_per_minute=self.requests_per_minute)
        limiter.wait()

        encoded_image = _encode_image_to_base64_png(image)
        openai_module = importlib.import_module("openai")
        client = openai_module.OpenAI(api_key=api_key, base_url=self.base_url)

        def _request() -> str:
            response = client.chat.completions.create(
                model=self.name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout_seconds,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded_image}"
                                },
                            },
                        ],
                    },
                ],
            )
            raw = _extract_openai_content(response.choices[0].message.content)
            task = str(metadata.get("task", "generic"))
            return parse_answer(raw_response=raw, task=task)

        return _run_with_retry(_request)
