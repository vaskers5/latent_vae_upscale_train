#!/usr/bin/env python3
"""Distributed training entry point (use with accelerate launch)."""

import argparse
from pathlib import Path
from typing import Any, Dict

from training.config_loader import load_config
from training.train_core import run as run_training

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "configs" / "vae_2x_1024ch" / "train_config.yaml"
DISTRIBUTED_OVERRIDE = BASE_DIR / "configs" / "vae_2x_1024ch" / "6xA100_conf.yaml"


def _parse_value(raw: str) -> Any:
    text = raw.strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _apply_override(target: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    current = target
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed VAE training using YAML configuration files")
    parser.add_argument(
        "-c",
        "--config",
        action="append",
        dest="configs",
        help="Path to an additional YAML config file. Defaults include baseline + distributed override.",
    )
    parser.add_argument(
        "--exp-name",
        dest="exp_name",
        help="Optional experiment name used for directories and logging.",
    )
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        dest="overrides",
        default=[],
        metavar="KEY=VALUE",
        help="Manual override (dotted keys supported).",
    )
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if args.exp_name:
        overrides["exp_name"] = args.exp_name
    for item in args.overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        _apply_override(overrides, key.strip(), _parse_value(value))
    return overrides


def main() -> None:
    args = parse_args()
    default_paths = [str(DEFAULT_CONFIG), str(DISTRIBUTED_OVERRIDE)]
    config_paths = args.configs if args.configs else default_paths
    overrides = build_overrides(args)
    config = load_config(config_paths, overrides=overrides)
    run_training(config)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
