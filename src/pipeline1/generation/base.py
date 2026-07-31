from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationResult:
    answer: str
    input_tokens: int
    output_tokens: int
    # Optional provider-specific diagnostics (finish_reason, reasoning_tokens, etc.).
    # None means the generator did not supply diagnostics; {} means it did but found nothing notable.
    completion_diagnostics: dict[str, Any] | None = None


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> GenerationResult:
        raise NotImplementedError
