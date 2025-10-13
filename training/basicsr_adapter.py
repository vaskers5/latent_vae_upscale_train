"""Adapter for running BasicSR training with pass-through configs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict

import yaml

from .config import TrainingConfig

LOGGER = logging.getLogger(__name__)


def _write_options_file(options: Dict[str, object], experiments_root: Path) -> Path:
    experiments_root.mkdir(parents=True, exist_ok=True)
    config_path = experiments_root / "basicsr_options.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(options, handle, sort_keys=False)
    return config_path


def run_with_basicsr(config: TrainingConfig) -> None:
    options = config.to_basicsr_options()
    options.setdefault("name", config.name)
    experiments_root = config.ensure_experiments_root()
    path_section = options.setdefault("path", {})
    path_section["experiments_root"] = str(experiments_root)

    config_path = _write_options_file(options, experiments_root)
    LOGGER.info("Launching BasicSR training with config: %s", config_path)

    repo_root = Path(__file__).resolve().parent.parent
    argv_backup = sys.argv[:]
    sys.argv = [argv_backup[0], "-opt", str(config_path)]
    try:
        from basicsr.train import train_pipeline

        train_pipeline(str(repo_root))
    finally:
        sys.argv = argv_backup
