"""Training package exports."""

from .config import TrainingConfig
from .trainer import run

__all__ = ["run", "TrainingConfig"]
