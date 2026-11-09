from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class QAPair:
    question: str
    answer: str
    difficulty: str
    data_id: str
    task: str
