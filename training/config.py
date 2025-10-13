"""Lightweight BasicSR configuration wrapper used across utilities."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict

__all__ = ["TrainingConfig"]


@dataclass
class TrainingConfig:
    """Minimal container that stores BasicSR-ready options."""

    basicsr_options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        if not isinstance(data, dict):
            raise TypeError("TrainingConfig expects a mapping loaded from YAML.")
        return cls(basicsr_options=deepcopy(data))

    def to_basicsr_options(self) -> Dict[str, Any]:
        return deepcopy(self.basicsr_options)
