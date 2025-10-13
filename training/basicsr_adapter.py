"""Adapter for running BasicSR training with pass-through configs."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .config import TrainingConfig

LOGGER = logging.getLogger(__name__)


def _write_options_file(options: Dict[str, object], experiments_root: Path) -> Path:
    experiments_root.mkdir(parents=True, exist_ok=True)
    config_path = experiments_root / "basicsr_options.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(options, handle, sort_keys=False)
    return config_path


def _slugify(text: Optional[Any]) -> str:
    if text is None:
        return "run"
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(text).strip())
    return cleaned or "run"


def _resolve_run_name(value: Optional[Any]) -> str:
    if value is None:
        return "run"
    name = str(value).strip()
    return name or "run"


def _ensure_experiments_root(options: Dict[str, Any]) -> Path:
    paths = options.setdefault("path", {})
    root_raw = paths.get("experiments_root")
    if root_raw:
        root = Path(str(root_raw)).expanduser()
    else:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        slug = _slugify(options.get("name"))
        root = Path("runs") / f"{timestamp}_{slug}"
        paths["experiments_root"] = str(root)
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
        paths["experiments_root"] = str(root)
    root.mkdir(parents=True, exist_ok=True)
    paths["experiments_root"] = str(root)
    return root


def run_with_basicsr(config: TrainingConfig) -> None:
    options = config.to_basicsr_options()
    options["name"] = _resolve_run_name(options.get("name"))
    experiments_root = _ensure_experiments_root(options)

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
