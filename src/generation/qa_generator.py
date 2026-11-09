from __future__ import annotations

# pyright: reportMissingImports=false

from dataclasses import dataclass

from src.generation import QAPair


@dataclass(slots=True)
class QAGenerator:
    """Create QA pairs with a model-assisted generation interface."""

    model_name: str = "gpt-4o"
    temperature: float = 0.7

    def generate_from_context(
        self, context: str, data_id: str, task: str
    ) -> list[QAPair]:
        """Generate deterministic placeholder QA pairs from context text."""
        context_summary = context.strip().split(".")[0][:140] or "structured sample"
        return [
            QAPair(
                question=f"From the given context, what is the key insight for {task}?",
                answer=context_summary,
                difficulty="2-hop",
                data_id=data_id,
                task=task,
            )
        ]
