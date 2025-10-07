"""YAML configuration loader with deep-merge support."""

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


class ConfigError(RuntimeError):
    """Raised when a configuration file cannot be loaded or is invalid."""


def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config(paths: Iterable[Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:  # pragma: no cover
            raise ConfigError(f"Failed to parse YAML config: {path}\n{exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"Expected a mapping in config file: {path}")
        merged = _deep_merge(merged, data)
    if overrides:
        merged = _deep_merge(merged, overrides)
    return merged
