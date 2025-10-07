"""Wrapper around wandb for safer logging."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

import wandb
from torch import nn

__all__ = ["WandbLogger"]


class WandbLogger:
    """Guarded interface for interacting with Weights & Biases."""

    def __init__(
        self,
        *,
        project: str,
        run_name: str,
        enabled: bool,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.project = project
        self.run_name = run_name
        self._enabled = bool(enabled)
        self._logger = logger
        self._run: Optional[wandb.sdk.wandb_run.Run] = None
        self._active = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def is_active(self) -> bool:
        return self._active and self._run is not None

    def _disable(self, reason: str) -> None:
        if self._logger is not None:
            self._logger.warning("Disabling W&B logging: %s", reason)
        self._enabled = False
        self._active = False
        self._run = None

    def start(
        self,
        *,
        config: Mapping[str, Any],
        model: Optional[nn.Module] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        if not self._enabled:
            return False
        if self._active:
            return True
        try:
            run = wandb.init(project=self.project, name=self.run_name, config=dict(config))
        except Exception as exc:  # pragma: no cover - defensive guard
            self._disable(f"wandb.init failed with error: {exc}")
            return False

        if run is None:
            self._disable("wandb.init returned None")
            return False

        self._run = run
        self._active = True

        if model is not None:
            try:
                wandb.watch(model, log="all", log_freq=100)
            except Exception as exc:  # pragma: no cover - defensive guard
                if self._logger is not None:
                    self._logger.warning("Failed to attach W&B model watch: %s", exc)

        if metadata:
            self.log(metadata, step=0)

        if self._logger is not None:
            self._logger.info("Logging enabled with Weights & Biases | project=%s | run=%s", run.project, run.name)

        return True

    def log(self, payload: Mapping[str, Any], *, step: Optional[int] = None) -> None:
        if not self.is_active:
            return
        if not payload:
            return
        try:
            wandb.log(dict(payload), step=step)
        except Exception as exc:  # pragma: no cover - defensive guard
            self._disable(f"wandb.log failed with error: {exc}")

    def log_summary(self, summary: Mapping[str, Any]) -> None:
        if not self.is_active:
            return
        if not summary:
            return
        try:
            for key, value in summary.items():
                self._run.summary[key] = value
        except Exception as exc:  # pragma: no cover - defensive guard
            if self._logger is not None:
                self._logger.warning("Failed to write W&B summary: %s", exc)

    def finish(self) -> None:
        if self._run is None:
            return
        try:
            wandb.finish()
        except Exception as exc:  # pragma: no cover - defensive guard
            if self._logger is not None:
                self._logger.warning("wandb.finish raised an error: %s", exc)
        finally:
            self._run = None
            self._active = False
