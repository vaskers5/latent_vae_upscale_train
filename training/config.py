"""Configuration schema and helper utilities for VAE training."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

__all__ = [
    "_resolve_bool",
    "_resolve_dtype",
    "PathsConfig",
    "DatasetConfig",
    "OptimizerConfig",
    "EMAConfig",
    "ModelConfig",
    "LossConfig",
    "LoggingConfig",
    "LatentUpscalerConfig",
    "TrainingConfig",
]


def _slugify(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return slug or None


def _resolve_dtype(value: Any) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if isinstance(value, torch.dtype):
        return value
    try:
        return mapping[str(value).lower()]
    except KeyError as exc:  # pragma: no cover - config guard
        raise ValueError(f"Unsupported dtype value: {value!r}") from exc


def _resolve_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return bool(value)


@dataclass
class PathsConfig:
    dataset_root: Path
    project: str
    save_root: Path
    samples_name: str
    exp_name: Optional[str]
    timestamp: str
    run_dir: Path
    samples_dir: Path
    checkpoints_dir: Path
    best_dir: Path
    final_dir: Path

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PathsConfig":
        dataset_root = Path(data.get("ds_path", "./data"))
        project = str(data.get("project", "vae_project"))
        save_root = Path(data.get("save_root_dir", project))
        samples_name = str(data.get("generated_folder_name", "samples"))
        exp_name = data.get("exp_name")
        timestamp = datetime.now().strftime("%Y_%m_%d_%H")
        slug = _slugify(exp_name)
        run_folder = f"{timestamp}_{slug}".strip("_") if slug else timestamp
        run_dir = save_root / run_folder
        return cls(
            dataset_root=dataset_root,
            project=project,
            save_root=save_root,
            samples_name=samples_name,
            exp_name=exp_name,
            timestamp=timestamp,
            run_dir=run_dir,
            samples_dir=run_dir / samples_name,
            checkpoints_dir=run_dir / "checkpoints",
            best_dir=run_dir / "best",
            final_dir=run_dir / "final",
        )

    def ensure_directories(self) -> None:
        for directory in (
            self.save_root,
            self.run_dir,
            self.samples_dir,
            self.checkpoints_dir,
            self.best_dir,
            self.final_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class DatasetConfig:
    high_resolution: int
    model_resolution: int
    resize_long_side: int
    limit: int
    num_workers: int
    horizontal_flip_prob: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetConfig":
        section = data.get("dataset", {})
        return cls(
            high_resolution=int(section.get("high_resolution", data.get("high_resolution", 256))),
            model_resolution=int(section.get("model_resolution", data.get("model_resolution", 256))),
            resize_long_side=int(section.get("resize_long_side", data.get("resize_long_side", 0))),
            limit=int(section.get("limit", data.get("limit", 0))),
            num_workers=int(section.get("num_workers", data.get("num_workers", 4))),
            horizontal_flip_prob=float(section.get("horizontal_flip_prob", data.get("horizontal_flip_prob", 0.0))),
        )


@dataclass
class OptimizerConfig:
    batch_size: int
    base_learning_rate: float
    min_learning_rate: float
    num_epochs: int
    optimizer_type: str
    beta2: float
    eps: float
    clip_grad_norm: float
    use_decay: bool
    weight_decay: float
    warmup_percent: float
    gradient_accumulation_steps: int
    scheduler: str
    cosine_min_ratio: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizerConfig":
        section = data.get("optimizer", {})
        return cls(
            batch_size=int(section.get("batch_size", data.get("batch_size", 8))),
            base_learning_rate=float(section.get("base_learning_rate", data.get("base_learning_rate", 1e-4))),
            min_learning_rate=float(section.get("min_learning_rate", data.get("min_learning_rate", 1e-5))),
            num_epochs=int(section.get("num_epochs", data.get("num_epochs", 10))),
            optimizer_type=str(section.get("optimizer_type", data.get("optimizer_type", "adam8bit"))),
            beta2=float(section.get("beta2", data.get("beta2", 0.99))),
            eps=float(section.get("eps", data.get("eps", 1e-6))),
            clip_grad_norm=float(section.get("clip_grad_norm", data.get("clip_grad_norm", 1.0))),
            use_decay=bool(section.get("use_decay", data.get("use_decay", True))),
            weight_decay=float(section.get("weight_decay", data.get("weight_decay", 0.01))),
            warmup_percent=float(section.get("warmup_percent", data.get("warmup_percent", 0.01))),
            gradient_accumulation_steps=int(section.get("gradient_accumulation_steps", data.get("gradient_accumulation_steps", 1))),
            scheduler=str(section.get("scheduler", data.get("scheduler", "cosine"))).lower(),
            cosine_min_ratio=float(section.get("cosine_min_ratio", data.get("cosine_min_ratio", 0.1))),
        )


@dataclass
class EMAConfig:
    enabled: bool
    decay: float
    update_after_step: int
    update_interval: int
    device: Optional[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EMAConfig":
        section = data.get("ema", {})
        return cls(
            enabled=_resolve_bool(section.get("enabled", data.get("ema_enabled", False))),
            decay=float(section.get("decay", data.get("ema_decay", 0.999))),
            update_after_step=int(section.get("update_after_step", data.get("ema_update_after_step", 0))),
            update_interval=max(1, int(section.get("update_interval", data.get("ema_update_interval", 1)))),
            device=section.get("device") or data.get("ema_device"),
        )


@dataclass
class ModelConfig:
    load_from: Optional[str]
    hf_repo: Optional[str]
    hf_subfolder: Optional[str]
    hf_revision: Optional[str]
    hf_auth_token: Optional[str]
    weights_dtype: torch.dtype
    mixed_precision: str
    train_decoder_only: bool
    full_training: bool
    vae_kind: str
    kl_ratio: float
    use_torch_compile: bool
    torch_compile_backend: Optional[str]
    torch_compile_mode: Optional[str]
    torch_compile_fullgraph: Optional[bool]

    @classmethod
    def from_dict(cls, data: Dict[str, Any], project: str) -> "ModelConfig":
        section = data.get("model", {})
        load_from = (
            section.get("LOAD_FROM")
            or section.get("load_from")
            or data.get("LOAD_FROM")
            or data.get("load_from")
        )
        load_from = str(load_from) if load_from else None
        compile_fullgraph = section.get("torch_compile_fullgraph")
        if compile_fullgraph is None:
            compile_fullgraph = data.get("torch_compile_fullgraph")
        if isinstance(compile_fullgraph, str):
            compile_fullgraph = _resolve_bool(compile_fullgraph, default=False)
        elif compile_fullgraph is not None:
            compile_fullgraph = bool(compile_fullgraph)
        return cls(
            load_from=load_from,
            hf_repo=section.get("hf_repo") or data.get("hf_repo"),
            hf_subfolder=section.get("hf_subfolder") or data.get("hf_subfolder"),
            hf_revision=section.get("hf_revision") or data.get("hf_revision"),
            hf_auth_token=section.get("hf_auth_token") or data.get("hf_auth_token"),
            weights_dtype=_resolve_dtype(section.get("weights_dtype", data.get("weights_dtype", "float32"))),
            mixed_precision=str(section.get("mixed_precision", data.get("mixed_precision", "no"))).strip().lower(),
            train_decoder_only=_resolve_bool(section.get("train_decoder_only", data.get("train_decoder_only", True))),
            full_training=_resolve_bool(section.get("full_training", data.get("full_training", False))),
            vae_kind=str(section.get("vae_kind", data.get("vae_kind", "kl"))),
            kl_ratio=float(section.get("kl_ratio", data.get("kl_ratio", 0.0))),
            use_torch_compile=_resolve_bool(section.get("use_torch_compile", data.get("use_torch_compile", True))),
            torch_compile_backend=section.get("torch_compile_backend") or data.get("torch_compile_backend"),
            torch_compile_mode=section.get("torch_compile_mode") or data.get("torch_compile_mode"),
            torch_compile_fullgraph=compile_fullgraph,
        )


@dataclass
class LossConfig:
    ratios: Dict[str, float]
    enabled: Dict[str, bool]
    active_losses: Tuple[str, ...]
    median_coeff_steps: int
    lpips_backbone: str
    lpips_eval_resolution: int
    focal_frequency_alpha: float
    focal_frequency_patch_factor: int
    focal_frequency_log_weight: bool
    focal_frequency_ave_spectrum: bool
    focal_frequency_normalize: bool
    focal_frequency_eps: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any], default_resolution: int, default_kl: float) -> "LossConfig":
        section = data.get("loss", {})

        ratio_section_raw = section.get("ratios") if isinstance(section.get("ratios"), dict) else {}
        fallback_ratio_raw = data.get("loss_ratios") if isinstance(data.get("loss_ratios"), dict) else {}

        ratio_section = {str(key).lower(): value for key, value in ratio_section_raw.items()}
        fallback_ratio_section = {str(key).lower(): value for key, value in fallback_ratio_raw.items()}

        defaults = {
            "mae": 1.0,
            "mse": 1.0,
            "edge": 0.1,
            "lpips": 0.0,
            "kl": default_kl,
            "ffl": 0.0,
        }

        def resolve_ratio(name: str, default: float) -> float:
            raw = ratio_section.get(name, fallback_ratio_section.get(name, default))
            try:
                return float(raw)
            except (TypeError, ValueError):  # pragma: no cover - config guard
                return default

        ratios_full: Dict[str, float] = {}
        for key, default in defaults.items():
            ratios_full[key] = resolve_ratio(key, default)

        global_enabled = True
        toggles: Dict[str, bool] = {}

        raw_enabled = section.get("enabled")
        if isinstance(raw_enabled, dict):
            for key, value in raw_enabled.items():
                toggles[str(key).lower()] = _resolve_bool(value, default=True)
        elif raw_enabled is not None:
            global_enabled = _resolve_bool(raw_enabled, default=True)

        raw_enable = section.get("enable")
        if isinstance(raw_enable, dict):
            for key, value in raw_enable.items():
                toggles[str(key).lower()] = _resolve_bool(value, default=True)
        elif raw_enable is not None and not isinstance(raw_enable, dict):
            global_enabled = _resolve_bool(raw_enable, default=global_enabled)

        disabled_raw = section.get("disabled")
        if disabled_raw is None:
            disabled_raw = section.get("disable")
        if isinstance(disabled_raw, str):
            disabled = {disabled_raw.lower()}
        elif isinstance(disabled_raw, (list, tuple, set)):
            disabled = {str(item).lower() for item in disabled_raw}
        elif disabled_raw is None:
            disabled = set()
        else:  # pragma: no cover - config guard
            disabled = {str(disabled_raw).lower()}

        enabled_map: Dict[str, bool] = {}
        filtered_ratios: Dict[str, float] = {}
        active_losses: List[str] = []

        for key, value in ratios_full.items():
            enabled = global_enabled and key not in disabled
            if key in toggles:
                enabled = toggles[key]
            enabled_map[key] = bool(enabled)
            if enabled and value != 0.0:
                filtered_ratios[key] = value
                active_losses.append(key)

        return cls(
            ratios=filtered_ratios,
            enabled=enabled_map,
            active_losses=tuple(active_losses),
            median_coeff_steps=int(section.get("median_coeff_steps", data.get("median_coeff_steps", 256))),
            lpips_backbone=str(section.get("lpips_backbone", data.get("lpips_backbone", "vgg"))),
            lpips_eval_resolution=int(section.get("lpips_eval_resolution", data.get("lpips_eval_resolution", default_resolution))),
            focal_frequency_alpha=float(section.get("focal_frequency", {}).get("alpha", data.get("focal_frequency_alpha", 1.0))),
            focal_frequency_patch_factor=int(section.get("focal_frequency", {}).get("patch_factor", data.get("focal_frequency_patch_factor", 1))),
            focal_frequency_log_weight=_resolve_bool(section.get("focal_frequency", {}).get("log_weight", data.get("focal_frequency_log_weight", True))),
            focal_frequency_ave_spectrum=_resolve_bool(section.get("focal_frequency", {}).get("ave_spectrum", data.get("focal_frequency_ave_spectrum", False))),
            focal_frequency_normalize=_resolve_bool(section.get("focal_frequency", {}).get("normalize", data.get("focal_frequency_normalize", True))),
            focal_frequency_eps=float(section.get("focal_frequency", {}).get("eps", data.get("focal_frequency_eps", 1e-8))),
        )


@dataclass
class LoggingConfig:
    use_wandb: bool
    wandb_run_name: Optional[str]
    sample_interval_share: int
    global_sample_interval: int
    global_save_interval: Optional[int]
    save_model: bool
    save_barrier: float
    log_grad_norm: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any], timestamp: str) -> "LoggingConfig":
        section = data.get("logging", {})
        save_interval = int(section.get("global_save_interval", data.get("GLOBAL_SAVE_INTERVAL", 1000)))
        return cls(
            use_wandb=bool(section.get("use_wandb", data.get("use_wandb", False))),
            wandb_run_name=section.get("wandb_run_name") or data.get("wandb_run_name") or timestamp,
            sample_interval_share=int(section.get("sample_interval_share", data.get("sample_interval_share", 1000))),
            global_sample_interval=int(section.get("global_sample_interval", data.get("GLOBAL_SAMPLE_INTERVAL", 500))),
            global_save_interval=save_interval if save_interval > 0 else None,
            save_model=bool(section.get("save_model", data.get("save_model", True))),
            save_barrier=float(section.get("save_barrier", data.get("save_barrier", 1.003))),
            log_grad_norm=bool(section.get("log_grad_norm", data.get("log_grad_norm", True))),
        )


@dataclass
class LatentUpscalerConfig:
    enabled: bool
    width_multiplier: int
    scale_factor: Optional[int]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LatentUpscalerConfig":
        section = data.get("latent_upscaler", {})
        scale = section.get("scale_factor")
        return cls(
            enabled=bool(section.get("enabled", data.get("train_latent_upscaler", False))),
            width_multiplier=int(section.get("width_multiplier", data.get("latent_upscaler_width", 2))),
            scale_factor=int(scale) if scale is not None else None,
        )


@dataclass
class TrainingConfig:
    paths: PathsConfig
    dataset: DatasetConfig
    optimiser: OptimizerConfig
    model: ModelConfig
    losses: LossConfig
    logging: LoggingConfig
    latent_upscaler: LatentUpscalerConfig
    ema: EMAConfig
    seed: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        paths = PathsConfig.from_dict(data)
        dataset = DatasetConfig.from_dict(data)
        optimiser = OptimizerConfig.from_dict(data)
        model = ModelConfig.from_dict(data, project=paths.project)
        losses = LossConfig.from_dict(data, dataset.model_resolution, model.kl_ratio)
        logging = LoggingConfig.from_dict(data, timestamp=paths.timestamp)
        latent_upscaler = LatentUpscalerConfig.from_dict(data)
        ema = EMAConfig.from_dict(data)
        seed = int(data.get("seed", int(datetime.now().strftime("%Y%m%d"))))
        return cls(
            paths=paths,
            dataset=dataset,
            optimiser=optimiser,
            model=model,
            losses=losses,
            logging=logging,
            latent_upscaler=latent_upscaler,
            ema=ema,
            seed=seed,
        )
