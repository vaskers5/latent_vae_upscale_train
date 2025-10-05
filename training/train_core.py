"""Backward compatibility shim for the refactored training pipeline."""

from __future__ import annotations

from .config import TrainingConfig
from .trainer import VAETrainer, run

__all__ = ["run", "TrainingConfig", "VAETrainer"]
