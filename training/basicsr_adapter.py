"""Adapter for running BasicSR training with the project configuration."""

from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import yaml

from .config import TrainingConfig

LOGGER = logging.getLogger(__name__)


def _clone_without_control_tokens(node: Any) -> Any:
    if isinstance(node, dict):
        return {key: _clone_without_control_tokens(value) for key, value in node.items() if key != "__replace__"}
    if isinstance(node, list):
        return [_clone_without_control_tokens(item) for item in node]
    return copy.deepcopy(node)


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict):
            cleaned = _clone_without_control_tokens(value)
            replace = bool(value.get("__replace__"))
            current = base.get(key)
            type_changed = False
            if isinstance(current, dict) and isinstance(cleaned, dict):
                type_changed = "type" in cleaned and "type" in current and cleaned["type"] != current["type"]
            if replace or not isinstance(current, dict) or type_changed:
                base[key] = cleaned
            else:
                base[key] = _merge_dicts(current, cleaned)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _resolve_scale(config: TrainingConfig) -> int:
    low = config.dataset.model_resolution or config.dataset.high_resolution
    high = config.dataset.high_resolution
    if low <= 0 or high <= 0:
        raise ValueError("Dataset resolutions must be positive to compute scale.")
    if high % low != 0:
        raise ValueError(f"High resolution {high} not divisible by low resolution {low}.")
    return high // low


def _sanitize_loss_component(component: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    drop_keys = {"name", "inputs", "pred_key", "target_key"}
    for key, value in component.items():
        if key in drop_keys:
            continue
        cleaned[key] = value
    cleaned.setdefault("loss_weight", 1.0)
    cleaned.setdefault("reduction", "mean")
    space = component.get("space", "latent")
    cleaned["space"] = str(space).strip().lower()
    cleaned["loss_weight"] = float(cleaned.get("loss_weight", 1.0))
    return cleaned


def _build_loss_entries(config: TrainingConfig) -> Dict[str, Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {}
    for component in config.losses.components:
        name = str(component.get("name") or component.get("type") or "loss").strip()
        key = f"{name}_opt"
        if key in entries:
            LOGGER.warning("Duplicate loss key '%s'; overriding previous definition.", key)
        entries[key] = _sanitize_loss_component(component)
    return entries


def _map_optimizer_type(name: str) -> str:
    lowered = name.lower()
    if lowered == "adamw":
        return "AdamW"
    if lowered in {"adam", "adam8bit"}:
        if lowered == "adam8bit":
            LOGGER.warning("BasicSR backend does not support 8-bit Adam; falling back to standard Adam.")
        return "Adam"
    if lowered == "sgd":
        return "SGD"
    LOGGER.warning("Unsupported optimizer '%s'; defaulting to Adam.", name)
    return "Adam"


def _build_scheduler_config(config: TrainingConfig) -> Dict[str, Any]:
    scheduler_type = (config.optimiser.scheduler or "").strip().lower()
    max_iters = int(config.optimiser.max_train_steps)
    if scheduler_type == "cosine":
        return {
            "type": "CosineAnnealingRestartLR",
            "periods": [max_iters],
            "restart_weights": [1],
            "eta_min": float(config.optimiser.min_learning_rate),
        }
    LOGGER.warning("Scheduler '%s' not supported; using constant learning rate.", scheduler_type)
    return {
        "type": "MultiStepRestartLR",
        "milestones": [max_iters],
        "gamma": 1.0,
    }


def _build_network_config(config: TrainingConfig, scale: int) -> Dict[str, Any]:
    extra = dict(config.latent_upscaler.extra)
    window_size = config.latent_upscaler.window or extra.get("window_size") or 8
    mlp_ratio = config.latent_upscaler.mlp_ratio or extra.get("mlp_ratio") or 4.0
    embed_dim = extra.get("embed_dim", 180)
    depths = extra.get("depths") or ([config.latent_upscaler.depth] * 4 if config.latent_upscaler.depth else [6, 6, 6, 6])
    num_heads = extra.get("num_heads") or ([config.latent_upscaler.heads] * len(depths) if config.latent_upscaler.heads else [6] * len(depths))
    drop_path_rate = extra.get("drop_path_rate", 0.0)
    upsampler = extra.get("upsampler", "pixelshuffle")
    resi_connection = extra.get("resi_connection", "1conv")
    patch_size = config.latent_upscaler.patch_size or extra.get("patch_size") or 1
    img_range = extra.get("img_range", 1.0)
    img_size = extra.get("img_size") or (config.dataset.model_resolution // max(scale, 1) if config.dataset.model_resolution else 64)

    network: Dict[str, Any] = {
        "type": "SwinIR",
        "upscale": scale,
        "in_chans": extra.get("in_chans", 16),
        "img_size": img_size,
        "patch_size": patch_size,
        "embed_dim": embed_dim,
        "depths": depths,
        "num_heads": num_heads,
        "window_size": window_size,
        "mlp_ratio": mlp_ratio,
        "drop_path_rate": drop_path_rate,
        "upsampler": upsampler,
        "resi_connection": resi_connection,
        "img_range": img_range,
    }
    for optional_key in ("qkv_bias", "qk_scale", "drop_rate", "attn_drop_rate", "ape", "patch_norm", "use_checkpoint"):
        if optional_key in extra:
            network[optional_key] = extra[optional_key]
    return network


def _build_dataset_config(config: TrainingConfig, scale: int) -> Dict[str, Any]:
    cache_dirs: Iterable[Path] = config.embeddings.vae_cache_dirs or (config.embeddings.cache_dir,)
    cache_paths = [str(path) for path in cache_dirs]
    if not cache_paths:
        raise ValueError("No embedding cache directories provided for BasicSR dataset.")
    low_res = config.dataset.model_resolution or config.dataset.high_resolution
    high_res = config.dataset.high_resolution
    dataset_cfg: Dict[str, Any] = {
        "name": "latent-train",
        "type": "LatentCacheDataset",
        "cache_dirs": cache_paths,
        "vae_names": list(config.embeddings.vae_names) or [Path(path).name for path in cache_paths],
        "low_res": int(low_res),
        "high_res": int(high_res),
        "scale": scale,
        "phase": "train",
        "prefetch_mode": "cpu",
        "num_prefetch_queue": 1,
        "batch_size_per_gpu": int(config.optimiser.batch_size),
        "num_worker_per_gpu": int(config.dataset.num_workers),
        "dataset_enlarge_ratio": 1,
        "use_shuffle": True,
        "pin_memory": True,
        "persistent_workers": False,
    }
    if config.embeddings.csv_path is not None:
        dataset_cfg["csv_path"] = str(config.embeddings.csv_path)
    return dataset_cfg


def _build_logger_config(config: TrainingConfig) -> Dict[str, Any]:
    save_freq = int(config.logging.save_each_n_steps or config.optimiser.max_train_steps)
    if save_freq <= 0:
        save_freq = config.optimiser.max_train_steps
    print_freq = max(1, int(config.logging.global_sample_interval // 10) if config.logging.global_sample_interval else 100)
    use_wandb = bool(config.logging.use_wandb)
    logger_cfg: Dict[str, Any] = {
        "print_freq": print_freq,
        "save_checkpoint_freq": save_freq,
        "use_tb_logger": use_wandb,
        "keep_last_ckpt": 1,
        "wandb": {
            "project": config.paths.project if use_wandb else None,
            "name": config.logging.wandb_run_name if use_wandb else None,
            "resume_id": None,
        },
    }
    return logger_cfg


def _build_train_section(config: TrainingConfig) -> Dict[str, Any]:
    optimizer_type = _map_optimizer_type(config.optimiser.optimizer_type)
    train_cfg: Dict[str, Any] = {
        "ema_decay": 0.0,
        "total_iter": int(config.optimiser.max_train_steps),
        "warmup_iter": -1,
        "val_freq": int(config.logging.global_sample_interval or config.optimiser.max_train_steps),
        "optim_g": {
            "type": optimizer_type,
            "lr": float(config.optimiser.base_learning_rate),
            "betas": (0.9, float(config.optimiser.beta2)),
            "eps": float(config.optimiser.eps),
            "weight_decay": float(config.optimiser.weight_decay if config.optimiser.use_decay else 0.0),
        },
        "scheduler": _build_scheduler_config(config),
    }
    return train_cfg


def build_basicsr_options(config: TrainingConfig) -> Dict[str, Any]:
    scale = _resolve_scale(config)
    dataset_cfg = _build_dataset_config(config, scale)
    train_cfg = _build_train_section(config)
    train_cfg.update(_build_loss_entries(config))
    logger_cfg = _build_logger_config(config)
    network_cfg = _build_network_config(config, scale)
    experiments_root = config.paths.run_dir / "basicsr"
    num_gpu = torch.cuda.device_count()
    if num_gpu <= 0:
        num_gpu = 1
    opt: Dict[str, Any] = {
        "name": config.paths.exp_name or f"latent-upscale-{config.paths.timestamp}",
        "model_type": "SwinIRLatentModel",
        "manual_seed": int(config.seed),
        "is_train": True,
        "scale": scale,
        "num_gpu": num_gpu,
        "dist": False,
        "rank": 0,
        "world_size": 1,
        "datasets": {"train": dataset_cfg},
        "network_g": network_cfg,
        "path": {
            "experiments_root": str(experiments_root),
            "pretrain_network_g": str(config.latent_upscaler.load_from) if config.latent_upscaler.load_from else None,
            "strict_load_g": True,
            "resume_state": None,
        },
        "train": train_cfg,
        "logger": logger_cfg,
        "val": None,
        "dist_params": {
            "backend": "nccl",
            "init_method": None,
            "port": None,
            "rank": 0,
            "world_size": 1,
        },
        "resume_state": None,
        "load_networks_only": False,
        "launcher": "none",
    }
    if config.basicsr:
        opt = _merge_dicts(opt, config.basicsr)
    return opt


def run_with_basicsr(config: TrainingConfig) -> None:
    config.paths.ensure_directories()
    opt = build_basicsr_options(config)
    experiments_root = Path(opt["path"]["experiments_root"])
    experiments_root.parent.mkdir(parents=True, exist_ok=True)
    basicsr_config_path = experiments_root / "basicsr_options.yaml"
    basicsr_config_path.parent.mkdir(parents=True, exist_ok=True)
    with basicsr_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(opt, handle, sort_keys=False)

    repo_root = Path(__file__).resolve().parent.parent
    argv_backup = sys.argv[:]
    sys.argv = [argv_backup[0], "-opt", str(basicsr_config_path)]
    try:
        from basicsr.train import train_pipeline

        train_pipeline(str(repo_root))
    finally:
        sys.argv = argv_backup
