from typing import Any, Dict

from .basicsr_adapter import run_with_basicsr
from .config import TrainingConfig

__all__ = ["run"]


def run(config: TrainingConfig | Dict[str, Any]) -> None:
    """Normalise configs and trigger the BasicSR training pipeline."""

    if not isinstance(config, TrainingConfig):
        config = TrainingConfig.from_dict(dict(config))

    run_with_basicsr(config)

