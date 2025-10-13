"""BasicSR training configuration wrapper."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["TrainingConfig"]


def _slugify(text: Optional[Any]) -> str:
    if text is None:
        return "run"
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(text).strip())
    return cleaned or "run"


@dataclass
class TrainingConfig:
    """Wrapper that forwards BasicSR-compatible options to the trainer."""

    basicsr_options: Dict[str, Any] = field(default_factory=dict)
    backend: str = "basicsr"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        if not isinstance(data, dict):
            raise TypeError("TrainingConfig expects a mapping loaded from YAML.")
        return cls(basicsr_options=deepcopy(data))

    @property
    def name(self) -> str:
        return str(self.basicsr_options.get("name") or "run")

    def ensure_experiments_root(self) -> Path:
        paths = self.basicsr_options.setdefault("path", {})
        root_raw = paths.get("experiments_root")
        if root_raw:
            root = Path(str(root_raw)).expanduser()
        else:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
            slug = _slugify(self.basicsr_options.get("name"))
            root = Path("runs") / f"{timestamp}_{slug}"
            paths["experiments_root"] = str(root)
        if not root.is_absolute():
            root = (Path.cwd() / root).resolve()
            paths["experiments_root"] = str(root)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def to_basicsr_options(self) -> Dict[str, Any]:
        return deepcopy(self.basicsr_options)
