from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricResult:
    metric_name: str
    value: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseSemanticMetric(ABC):
    name: str

    @abstractmethod
    def compute(self, **kwargs: Any) -> MetricResult:
        ...
