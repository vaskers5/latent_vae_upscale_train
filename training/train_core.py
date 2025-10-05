"""Shared training utilities for VAE fine-tuning.

The module exposes a configuration driven training pipeline that powers both
``train_vae.py`` and ``train_vae_distributed.py``.  It keeps the heavy lifting
(e.g. dataset preparation, optimiser setup, latent upscaler handling) in one
place so the CLI entry points remain lightweight and readable.
"""

import gc
import math
import os
import random
import re
import shutil
from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import bitsandbytes as bnb
import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import wandb
from accelerate import Accelerator
from diffusers import (
    AsymmetricAutoencoderKL,
    AutoencoderKL,
    AutoencoderKLQwenImage,
    AutoencoderKLWan,
)
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import spectral_norm as apply_spectral_norm
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration dataclasses


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
    persistent_workers: bool

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
            persistent_workers=_resolve_bool(
                section.get("persistent_workers", data.get("persistent_workers", True))
            ),
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
    dtype: torch.dtype
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
        load_from = section.get("LOAD_FROM") or section.get("load_from") or data.get("LOAD_FROM") or data.get("load_from")
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
            dtype=_resolve_dtype(section.get("dtype", data.get("dtype", "float32"))),
            mixed_precision=str(section.get("mixed_precision", data.get("mixed_precision", "no"))),
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
            "adv": 0.0,
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
class AdversarialConfig:
    enabled: bool
    loss_type: str
    discriminator_channels: int
    discriminator_max_channels: int
    discriminator_layers: int
    discriminator_norm: str
    spectral_norm: bool
    lr: float
    beta1: float
    beta2: float
    weight_decay: float
    clip_grad_norm: float
    update_frequency: int
    warmup_steps: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdversarialConfig":
        section = data.get("adversarial", {})
        return cls(
            enabled=_resolve_bool(section.get("enabled", data.get("adversarial_enabled", False))),
            loss_type=str(section.get("loss_type", data.get("adversarial_loss", "ragan"))).lower(),
            discriminator_channels=int(section.get("discriminator", {}).get("base_channels", 64)),
            discriminator_max_channels=int(section.get("discriminator", {}).get("max_channels", 512)),
            discriminator_layers=int(section.get("discriminator", {}).get("num_layers", 4)),
            discriminator_norm=str(section.get("discriminator", {}).get("norm", "instance")).lower(),
            spectral_norm=_resolve_bool(section.get("discriminator", {}).get("spectral_norm", True)),
            lr=float(section.get("optimizer", {}).get("lr", section.get("lr", 1e-4))),
            beta1=float(section.get("optimizer", {}).get("beta1", 0.9)),
            beta2=float(section.get("optimizer", {}).get("beta2", 0.99)),
            weight_decay=float(section.get("optimizer", {}).get("weight_decay", 0.0)),
            clip_grad_norm=float(section.get("optimizer", {}).get("clip_grad_norm", 1.0)),
            update_frequency=int(section.get("update_frequency", 1)),
            warmup_steps=int(section.get("warmup_steps", 0)),
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
    adversarial: AdversarialConfig
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
        adversarial = AdversarialConfig.from_dict(data)
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
            adversarial=adversarial,
            seed=seed,
        )


# ---------------------------------------------------------------------------
# Dataset utilities


class ImageFolderDataset(Dataset):
    """Recursively reads images and returns random crops normalised to [-1, 1]."""

    def __init__(
        self,
        root: Path,
        high_resolution: int,
        resize_long_side: int = 0,
        limit: int = 0,
        horizontal_flip_prob: float = 0.0,
    ) -> None:
        self.root = root
        self.high_resolution = high_resolution
        self.resize_long_side = resize_long_side
        self.horizontal_flip_prob = horizontal_flip_prob
        self.paths: List[Path] = []

        exts = {".png", ".jpg", ".jpeg", ".webp"}
        for current_root, _dirs, files in os.walk(root):
            for name in files:
                if Path(name).suffix.lower() in exts:
                    self.paths.append(Path(current_root) / name)
        if limit:
            self.paths = self.paths[:limit]

        self.paths = self._filter_valid_images(self.paths)
        if not self.paths:
            raise RuntimeError(f"No valid images found under '{root}'")
        random.shuffle(self.paths)

    @staticmethod
    def _filter_valid_images(paths: Iterable[Path]) -> List[Path]:
        valid: List[Path] = []
        for path in paths:
            try:
                with Image.open(path) as img:
                    img.verify()
                valid.append(path)
            except (OSError, UnidentifiedImageError):
                continue
        return valid

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.paths[index % len(self.paths)]
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = self._resize_if_needed(img)
            img = self._maybe_flip(img)
            img = self._crop_to_resolution(img)
            tensor = TF.to_tensor(img)
            tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            return tensor

    def _resize_if_needed(self, img: Image.Image) -> Image.Image:
        if self.resize_long_side <= 0:
            return img
        width, height = img.size
        longest = max(width, height)
        if longest <= self.resize_long_side:
            return img
        scale = self.resize_long_side / float(longest)
        new_size = (int(round(width * scale)), int(round(height * scale)))
        return img.resize(new_size, Image.LANCZOS)

    def _maybe_flip(self, img: Image.Image) -> Image.Image:
        if self.horizontal_flip_prob > 0 and random.random() < self.horizontal_flip_prob:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def _crop_to_resolution(self, img: Image.Image) -> Image.Image:
        if self.high_resolution <= 0:
            return img
        width, height = img.size
        if width < self.high_resolution or height < self.high_resolution:
            img = img.resize((max(width, self.high_resolution), max(height, self.high_resolution)), Image.LANCZOS)
            width, height = img.size
        if width == self.high_resolution and height == self.high_resolution:
            return img
        x = random.randint(0, width - self.high_resolution)
        y = random.randint(0, height - self.high_resolution)
        return img.crop((x, y, x + self.high_resolution, y + self.high_resolution))


# ---------------------------------------------------------------------------
# Helper modules


class LatentUpscaler(nn.Module):
    def __init__(self, channels: int, scale_factor: int, hidden_multiplier: int) -> None:
        super().__init__()
        if scale_factor < 2:
            raise ValueError("Latent upscaler requires scale_factor >= 2")
        hidden_channels = max(channels * hidden_multiplier, channels)
        self.scale_factor = scale_factor
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.dim() == 5:  # video VAE (B, C, T, H, W)
            b, c, t, h, w = latents.shape
            latents_2d = latents.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            upsampled = self._forward_2d(latents_2d)
            _, c2, h2, w2 = upsampled.shape
            upsampled = upsampled.reshape(b, t, c2, h2, w2).permute(0, 2, 1, 3, 4)
            return upsampled
        if latents.dim() != 4:
            raise ValueError(f"Expected 4D or 5D latents, got shape {latents.shape}")
        return self._forward_2d(latents)

    def _forward_2d(self, latents: torch.Tensor) -> torch.Tensor:
        residual = F.interpolate(latents, scale_factor=self.scale_factor, mode="nearest")
        out = self.net(latents)
        return self.act(out + residual.to(out.dtype))


class MedianLossNormalizer:
    def __init__(self, desired_ratios: Dict[str, float], window_steps: int, device: Optional[torch.device] = None) -> None:
        total = sum(desired_ratios.values()) or 1.0
        self.target = {k: v / total for k, v in desired_ratios.items()}
        self.buffers = {k: deque(maxlen=window_steps) for k in desired_ratios.keys()}
        self.device = device

    def update(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:
        if not losses:
            zero = torch.zeros((), device=self.device) if self.device is not None else torch.tensor(0.0)
            return zero, {}, {}
        for key, value in losses.items():
            if key in self.buffers:
                self.buffers[key].append(float(value.detach().abs().cpu()))
        medians = {k: (np.median(buf) if buf else 1.0) for k, buf in self.buffers.items()}
        coeffs: Dict[str, float] = {}
        for key in losses.keys():
            denom = max(medians.get(key, 1.0), 1e-12)
            coeffs[key] = self.target.get(key, 0.0) / denom
        first_loss = next(iter(losses.values()))
        total_loss = torch.zeros((), device=first_loss.device, dtype=first_loss.dtype)
        for key, value in losses.items():
            total_loss = total_loss + coeffs[key] * value
        return total_loss, coeffs, medians


class FocalFrequencyLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        patch_factor: int = 1,
        ave_spectrum: bool = False,
        log_weight: bool = True,
        normalize: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.patch_factor = max(1, patch_factor)
        self.ave_spectrum = ave_spectrum
        self.log_weight = log_weight
        self.normalize = normalize
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = prediction.to(torch.float32)
        ref = target.to(torch.float32)
        pred_freq = torch.fft.rfft2(pred, dim=(-2, -1), norm="ortho")
        ref_freq = torch.fft.rfft2(ref, dim=(-2, -1), norm="ortho")
        diff = pred_freq - ref_freq
        magnitude = diff.abs()
        if self.patch_factor > 1:
            magnitude = magnitude[..., :: self.patch_factor, :: self.patch_factor]
        if self.ave_spectrum:
            weight = magnitude.mean(dim=(0, 1), keepdim=True)
        else:
            weight = magnitude
        if self.log_weight:
            weight = torch.log1p(weight)
        weight = torch.pow(weight + self.eps, self.alpha)
        loss_matrix = weight * (magnitude ** 2)
        if self.normalize:
            return loss_matrix.mean()
        return loss_matrix.sum() / loss_matrix.numel()


class PatchDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        max_channels: int = 512,
        num_layers: int = 4,
        norm: str = "instance",
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        current_in = in_channels
        current_out = base_channels
        norm = norm.lower()

        for layer_idx in range(max(1, num_layers)):
            stride = 1 if layer_idx == 0 else 2
            conv = nn.Conv2d(current_in, current_out, kernel_size=3, stride=stride, padding=1)
            if use_spectral_norm:
                conv = apply_spectral_norm(conv)
            block: List[nn.Module] = [conv]
            if norm == "instance":
                block.append(nn.InstanceNorm2d(current_out))
            elif norm == "layer":
                block.append(nn.GroupNorm(1, current_out))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Sequential(*block))
            current_in = current_out
            current_out = min(current_out * 2, max_channels)

        final = nn.Conv2d(current_in, 1, kernel_size=3, padding=1)
        if use_spectral_norm:
            final = apply_spectral_norm(final)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class RaGANLoss:
    def __init__(self) -> None:
        self.criterion = nn.BCEWithLogitsLoss()

    def discriminator_loss(self, real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        real_labels = torch.ones_like(real_pred)
        fake_labels = torch.zeros_like(fake_pred)
        loss_real = self.criterion(real_pred - fake_pred.mean(), real_labels)
        loss_fake = self.criterion(fake_pred - real_pred.mean(), fake_labels)
        return 0.5 * (loss_real + loss_fake)

    def generator_loss(self, real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        real_labels = torch.zeros_like(real_pred)
        fake_labels = torch.ones_like(fake_pred)
        loss_fake = self.criterion(fake_pred - real_pred.mean(), fake_labels)
        loss_real = self.criterion(real_pred - fake_pred.mean(), real_labels)
        return 0.5 * (loss_fake + loss_real)


# ---------------------------------------------------------------------------
# Trainer implementation


class VAETrainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.cfg = config
        self.cfg.paths.ensure_directories()

        torch.backends.cudnn.benchmark = True

        self.accelerator = Accelerator(
            mixed_precision=config.model.mixed_precision,
            gradient_accumulation_steps=config.optimiser.gradient_accumulation_steps,
        )
        self.device = self.accelerator.device
        self._seed_everything(config.seed)

        self.dataset = ImageFolderDataset(
            root=self.cfg.paths.dataset_root,
            high_resolution=self.cfg.dataset.high_resolution,
            resize_long_side=self.cfg.dataset.resize_long_side,
            limit=self.cfg.dataset.limit,
            horizontal_flip_prob=self.cfg.dataset.horizontal_flip_prob,
        )
        if len(self.dataset) < self.cfg.optimiser.batch_size:
            raise RuntimeError(
                f"Found only {len(self.dataset)} images but batch_size is {self.cfg.optimiser.batch_size}."
            )

        persistent_workers = self.cfg.dataset.persistent_workers and self.cfg.dataset.num_workers > 0
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.cfg.optimiser.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
        )

        self.vae = self._load_vae()
        self.latent_upscaler: Optional[LatentUpscaler] = self._build_latent_upscaler(self.vae)
        self._freeze_parameters()

        self.discriminator: Optional[PatchDiscriminator] = None
        self.discriminator_optimizer: Optional[torch.optim.Optimizer] = None
        self.adversarial_loss: Optional[RaGANLoss] = None

        adv_ratio = self.cfg.losses.ratios.get("adv", 0.0)
        self.adv_active = self.cfg.adversarial.enabled and adv_ratio > 0.0
        if adv_ratio > 0.0 and not self.cfg.adversarial.enabled:
            self.accelerator.print(
                "[WARN] Adversarial loss ratio set but adversarial training disabled; ignoring adversarial loss"
            )
        if not self.adv_active and "adv" in self.cfg.losses.ratios:
            self.cfg.losses.ratios.pop("adv")
            self.cfg.losses.enabled["adv"] = False
            self.cfg.losses.active_losses = tuple(
                key for key in self.cfg.losses.active_losses if key != "adv"
            )

        if self.adv_active:
            self.discriminator = self._build_discriminator()
            self.adversarial_loss = self._build_adversarial_loss()
            self.discriminator_optimizer = self._build_discriminator_optimizer()

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        prepare_items: List[Any] = [self.dataloader, self.vae]
        if self.latent_upscaler is not None:
            prepare_items.append(self.latent_upscaler)
        if self.discriminator is not None:
            prepare_items.append(self.discriminator)
        prepare_items.extend([self.optimizer, self.scheduler])
        if self.discriminator_optimizer is not None:
            prepare_items.append(self.discriminator_optimizer)
        prepared = self.accelerator.prepare(*prepare_items)
        idx = 0
        self.dataloader = prepared[idx]; idx += 1
        self.vae = prepared[idx]; idx += 1
        if self.latent_upscaler is not None:
            self.latent_upscaler = prepared[idx]; idx += 1
        if self.discriminator is not None:
            self.discriminator = prepared[idx]; idx += 1
        self.optimizer = prepared[idx]; idx += 1
        self.scheduler = prepared[idx]; idx += 1
        if self.discriminator_optimizer is not None:
            self.discriminator_optimizer = prepared[idx]

        self.compile_enabled = False
        if self.cfg.model.use_torch_compile and hasattr(torch, "compile"):
            compile_kwargs: Dict[str, Any] = {}
            if self.cfg.model.torch_compile_backend:
                compile_kwargs["backend"] = self.cfg.model.torch_compile_backend
            if self.cfg.model.torch_compile_mode:
                compile_kwargs["mode"] = self.cfg.model.torch_compile_mode
            if self.cfg.model.torch_compile_fullgraph is not None:
                compile_kwargs["fullgraph"] = self.cfg.model.torch_compile_fullgraph
            try:
                self.vae = torch.compile(self.vae, **compile_kwargs)
                self.compile_enabled = True
            except Exception as exc:  # pragma: no cover - diagnostic only
                self.accelerator.print(f"[WARN] torch.compile failed for VAE: {exc}")
            if self.latent_upscaler is not None:
                # Compiling the upscaler tends to capture CUDA graph state that later
                # replays with stale buffers, so keep it eager for stability.
                self.accelerator.print("[INFO] Skipping torch.compile for latent upscaler (eager mode forced)")

        self.trainable_params = [p for p in self.vae.parameters() if p.requires_grad]
        if self.latent_upscaler is not None:
            self.trainable_params.extend(p for p in self.latent_upscaler.parameters() if p.requires_grad)
        if self.discriminator is not None:
            self.discriminator_optimizer.zero_grad(set_to_none=True)

        self.loss_normalizer = MedianLossNormalizer(
            self.cfg.losses.ratios,
            self.cfg.losses.median_coeff_steps,
            device=self.device,
        )
        self.ffl_module: Optional[FocalFrequencyLoss] = None
        if self.cfg.losses.ratios.get("ffl", 0.0) > 0:
            self.ffl_module = FocalFrequencyLoss(
                alpha=self.cfg.losses.focal_frequency_alpha,
                patch_factor=self.cfg.losses.focal_frequency_patch_factor,
                ave_spectrum=self.cfg.losses.focal_frequency_ave_spectrum,
                log_weight=self.cfg.losses.focal_frequency_log_weight,
                normalize=self.cfg.losses.focal_frequency_normalize,
                eps=self.cfg.losses.focal_frequency_eps,
            )

        self.lpips_module: Optional[lpips.LPIPS] = None
        if self.cfg.losses.ratios.get("lpips", 0.0) > 0:
            self.lpips_module = lpips.LPIPS(net=self.cfg.losses.lpips_backbone, verbose=False)
            self.lpips_module = self.lpips_module.to(self.device).eval()
            for p in self.lpips_module.parameters():
                p.requires_grad_(False)

        self.discriminator_steps = 0

        self.ema_vae: Optional[AveragedModel] = None
        self.ema_latent_upscaler: Optional[AveragedModel] = None
        if self.cfg.ema.enabled:
            if self.cfg.ema.device:
                try:
                    ema_device = torch.device(self.cfg.ema.device)
                except (TypeError, RuntimeError):  # pragma: no cover - config fallback
                    self.accelerator.print(
                        f"[WARN] Invalid EMA device '{self.cfg.ema.device}', falling back to training device"
                    )
                    ema_device = self.device
            else:
                ema_device = self.device
            base_vae = self._unwrap_model(self.vae)
            self.ema_vae = AveragedModel(base_vae, avg_fn=self._ema_avg_fn, device=ema_device)
            if self.latent_upscaler is not None:
                base_up = self._unwrap_model(self.latent_upscaler)
                self.ema_latent_upscaler = AveragedModel(base_up, avg_fn=self._ema_avg_fn, device=ema_device)

        self.fixed_samples = self._sample_fixed_batch()

        if self.cfg.logging.use_wandb and self.accelerator.is_main_process:
            wandb.init(
                project=self.cfg.paths.project,
                name=self.cfg.logging.wandb_run_name,
                config=self._wandb_payload(),
            )
            wandb.config.update({"run_dir": str(self.cfg.paths.run_dir)}, allow_val_change=True)

        if self.cfg.logging.save_model and self.accelerator.is_main_process:
            self.accelerator.print("Generating reference samples before training...")
            self._generate_and_log_samples(step=0)
        self.accelerator.wait_for_everyone()

    # ------------------------------------------------------------------ helpers
    def _seed_everything(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False

    def _load_vae(self) -> nn.Module:
        cfg = self.cfg.model
        load_path = Path(cfg.load_from) if cfg.load_from else None
        path_exists = load_path is not None and load_path.exists()

        hf_source = cfg.hf_repo if cfg.hf_repo else None
        if not path_exists and cfg.load_from and not cfg.hf_repo:
            # if user supplied a path but it does not exist, fall back to treating it as repo id
            hf_source = cfg.load_from

        vae: nn.Module
        kind = (cfg.vae_kind or "").strip().lower()

        if kind == "qwen":
            if path_exists:
                source = str(load_path)
                kwargs: Dict[str, Any] = {}
            else:
                source = hf_source or "Qwen/Qwen-Image"
                kwargs = {}
                if cfg.hf_subfolder or not hf_source:
                    kwargs["subfolder"] = cfg.hf_subfolder or "vae"
                if cfg.hf_revision:
                    kwargs["revision"] = cfg.hf_revision
                if cfg.hf_auth_token:
                    kwargs["use_auth_token"] = cfg.hf_auth_token
            vae = AutoencoderKLQwenImage.from_pretrained(source, **kwargs)
        else:
            if path_exists:
                source = str(load_path)
                kwargs = {}
            else:
                source = hf_source or self.cfg.paths.project
                kwargs = {}
                if cfg.hf_subfolder:
                    kwargs["subfolder"] = cfg.hf_subfolder
                if cfg.hf_revision:
                    kwargs["revision"] = cfg.hf_revision
                if cfg.hf_auth_token:
                    kwargs["use_auth_token"] = cfg.hf_auth_token
            if kind == "wan":
                vae = AutoencoderKLWan.from_pretrained(source, **kwargs)
            else:
                if kind in {"kl", "autoencoderkl", "autoencoder_kl"}:
                    vae = AutoencoderKL.from_pretrained(source, **kwargs)
                elif kind in {"asymmetric_kl", "kl_asymmetric", "kl_asym", "asym_kl"}:
                    vae = AsymmetricAutoencoderKL.from_pretrained(source, **kwargs)
                else:
                    if self.cfg.dataset.model_resolution == self.cfg.dataset.high_resolution:
                        vae = AutoencoderKL.from_pretrained(source, **kwargs)
                    else:
                        vae = AsymmetricAutoencoderKL.from_pretrained(source, **kwargs)
        display_source = source if not path_exists else str(load_path)
        self.accelerator.print(f"[INFO] Loading VAE from: {display_source}")
        return vae.to(cfg.dtype)

    def _determine_latent_channels(self, vae: nn.Module) -> Optional[int]:
        core = self._unwrap_model(vae)
        cfg = getattr(core, "config", None)
        latent_channels = getattr(cfg, "latent_channels", None)
        if latent_channels:
            return latent_channels
        decoder = getattr(core, "decoder", None)
        conv_in = getattr(decoder, "conv_in", None) if decoder is not None else None
        return getattr(conv_in, "in_channels", None)

    def _build_latent_upscaler(self, vae: nn.Module) -> Optional[LatentUpscaler]:
        cfg = self.cfg
        if not cfg.latent_upscaler.enabled:
            return None
        scale_factor = cfg.latent_upscaler.scale_factor
        if scale_factor is None:
            if cfg.dataset.high_resolution <= cfg.dataset.model_resolution:
                raise ValueError("Latent upscaler enabled but high_resolution <= model_resolution")
            if cfg.dataset.high_resolution % cfg.dataset.model_resolution != 0:
                raise ValueError("High resolution must be integer multiple of model resolution")
            scale_factor = cfg.dataset.high_resolution // cfg.dataset.model_resolution
        latent_channels = self._determine_latent_channels(vae)
        if latent_channels is None:
            raise RuntimeError("Unable to determine latent channels for upscaler")
        upscaler = LatentUpscaler(latent_channels, scale_factor=scale_factor, hidden_multiplier=cfg.latent_upscaler.width_multiplier)
        return upscaler.to(self.device, dtype=self.cfg.model.dtype)

    def _freeze_parameters(self) -> None:
        core = self._unwrap_model(self.vae)
        for param in core.parameters():
            param.requires_grad = False

        names: List[str] = []
        if self.latent_upscaler is not None:
            for name, param in self.latent_upscaler.named_parameters():
                param.requires_grad = True
                names.append(f"latent_upscaler.{name}")
        elif self.cfg.model.full_training and not self.cfg.model.train_decoder_only:
            for name, param in core.named_parameters():
                param.requires_grad = True
                names.append(name)
        else:
            decoder = getattr(core, "decoder", None)
            if decoder is not None:
                for name, param in decoder.named_parameters():
                    param.requires_grad = True
                    names.append(f"decoder.{name}")
            post_quant = getattr(core, "post_quant_conv", None)
            if post_quant is not None:
                for name, param in post_quant.named_parameters():
                    param.requires_grad = True
                    names.append(f"post_quant_conv.{name}")
        self.accelerator.print(f"[INFO] Unfrozen {len(names)} parameter tensors")

    def _collect_param_groups(self) -> List[Dict[str, Any]]:
        modules: List[nn.Module] = [self.vae]
        if self.latent_upscaler is not None:
            modules.append(self.latent_upscaler)
        decay, no_decay = [], []
        tokens = ("bias", "norm", "rms", "layernorm")
        for module in modules:
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                (no_decay if any(tok in name.lower() for tok in tokens) else decay).append(param)
        return [
            {"params": decay, "weight_decay": self.cfg.optimiser.weight_decay if self.cfg.optimiser.use_decay else 0.0},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def _ema_avg_fn(self, averaged_param: torch.Tensor, model_param: torch.Tensor, _num_averaged: int) -> torch.Tensor:
        decay = self.cfg.ema.decay
        if averaged_param is None:
            return model_param.detach()
        return averaged_param * decay + model_param.detach() * (1.0 - decay)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        groups = self._collect_param_groups()
        cfg = self.cfg.optimiser
        name = cfg.optimizer_type.lower()
        if name == "adam8bit":
            return bnb.optim.AdamW8bit(groups, lr=cfg.base_learning_rate, betas=(0.9, cfg.beta2), eps=cfg.eps)
        if name == "adamw":
            return torch.optim.AdamW(groups, lr=cfg.base_learning_rate, betas=(0.9, cfg.beta2), eps=cfg.eps)
        if name == "adam":
            return torch.optim.Adam(groups, lr=cfg.base_learning_rate, betas=(0.9, cfg.beta2), eps=cfg.eps)
        if name in {"dadapt", "dadaptadam", "d_adapt_adam", "dadapt_adam"}:
            try:
                from dadaptation import DAdaptAdam
            except ImportError as exc:  # pragma: no cover - optional dependency guard
                raise ImportError(
                    "Optimizer type 'dadapt_adam' requested but the 'dadaptation' package is not installed. "
                    "Install it with 'pip install dadaptation'."
                ) from exc
            return DAdaptAdam(
                groups,
                lr=max(cfg.base_learning_rate, 1e-12),
                betas=(0.9, cfg.beta2),
                eps=cfg.eps,
                weight_decay=cfg.weight_decay if cfg.use_decay else 0.0,
                decouple=cfg.use_decay,
            )
        raise ValueError(f"Unsupported optimizer type: {self.cfg.optimiser.optimizer_type}")

    def _build_discriminator(self) -> PatchDiscriminator:
        cfg = self.cfg.adversarial
        return PatchDiscriminator(
            in_channels=3,
            base_channels=cfg.discriminator_channels,
            max_channels=cfg.discriminator_max_channels,
            num_layers=cfg.discriminator_layers,
            norm=cfg.discriminator_norm,
            use_spectral_norm=cfg.spectral_norm,
        )

    def _build_discriminator_optimizer(self) -> torch.optim.Optimizer:
        if self.discriminator is None:
            raise RuntimeError("Discriminator not initialised")
        cfg = self.cfg.adversarial
        return torch.optim.Adam(
            self.discriminator.parameters(),
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )

    def _build_adversarial_loss(self) -> RaGANLoss:
        if self.cfg.adversarial.loss_type not in {"ragan", "relativistic", "relativistic-average"}:
            raise ValueError(
                f"Unsupported adversarial loss type '{self.cfg.adversarial.loss_type}'. Only 'ragan' is implemented."
            )
        return RaGANLoss()

    def _total_steps(self) -> int:
        batches_per_epoch = len(self.dataloader)
        effective_batches = batches_per_epoch / max(1, self.cfg.optimiser.gradient_accumulation_steps)
        total = int(math.ceil(effective_batches)) * self.cfg.optimiser.num_epochs
        self.accelerator.print(
            f"[INFO] batches/epoch={batches_per_epoch}, optimiser steps/epoch={int(math.ceil(effective_batches))}, total={total}"
        )
        return max(1, total)

    def _build_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        cfg = self.cfg.optimiser
        total_steps = self._total_steps()
        if cfg.scheduler == "cosine":
            eta_min = cfg.base_learning_rate * cfg.cosine_min_ratio
            return CosineAnnealingLR(self.optimizer, total_steps, eta_min=eta_min)

        warmup_steps = max(1, int(total_steps * cfg.warmup_percent))
        min_ratio = cfg.min_learning_rate / max(cfg.base_learning_rate, 1e-12)

        def schedule(step: int) -> float:
            step += 1
            if step <= warmup_steps:
                return min_ratio + (1.0 - min_ratio) * (step / warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(self.optimizer, schedule)

    def _toggle_discriminator_grad(self, enabled: bool) -> None:
        if self.discriminator is None:
            return
        for param in self.discriminator.parameters():
            param.requires_grad_(enabled)

    def _should_train_discriminator(self) -> bool:
        frequency = max(1, self.cfg.adversarial.update_frequency)
        return self.discriminator_steps % frequency == 0

    def _discriminator_step(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> Tuple[Optional[float], Optional[float]]:
        if self.discriminator is None or self.discriminator_optimizer is None or self.adversarial_loss is None:
            return None, None
        self._toggle_discriminator_grad(True)
        disc_loss_value: Optional[float] = None
        grad_norm_value: Optional[float] = None
        with self.accelerator.accumulate(self.discriminator):
            real_pred = self.discriminator(real_images.to(torch.float32))
            fake_pred = self.discriminator(fake_images.to(torch.float32))
            loss = self.adversarial_loss.discriminator_loss(real_pred, fake_pred)
            self.accelerator.backward(loss)
            disc_loss_value = float(loss.detach().item())
            grad_norm = torch.tensor(0.0, device=self.device)
            if self.accelerator.sync_gradients:
                if self.cfg.adversarial.clip_grad_norm > 0:
                    grad_norm = self.accelerator.clip_grad_norm_(
                        self.discriminator.parameters(), self.cfg.adversarial.clip_grad_norm
                    )
                self.discriminator_optimizer.step()
                self.discriminator_optimizer.zero_grad(set_to_none=True)
                grad_norm_value = float(grad_norm.detach().item()) if isinstance(grad_norm, torch.Tensor) else 0.0
        return disc_loss_value, grad_norm_value

    def _generator_adversarial_loss(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> torch.Tensor:
        if self.discriminator is None or self.adversarial_loss is None:
            return torch.zeros((), device=self.device)
        self._toggle_discriminator_grad(False)
        with torch.no_grad():
            real_pred = self.discriminator(real_images.to(torch.float32))
        fake_pred = self.discriminator(fake_images.to(torch.float32))
        loss = self.adversarial_loss.generator_loss(real_pred.detach(), fake_pred)
        return loss

    def _update_ema(self, step: int) -> None:
        if self.ema_vae is None:
            return
        if step < self.cfg.ema.update_after_step:
            return
        if (step - self.cfg.ema.update_after_step) % self.cfg.ema.update_interval != 0:
            return
        base_vae = self._unwrap_model(self.vae)
        self.ema_vae.update_parameters(base_vae)
        if self.ema_latent_upscaler is not None and self.latent_upscaler is not None:
            base_up = self._unwrap_model(self.latent_upscaler)
            self.ema_latent_upscaler.update_parameters(base_up)

    def _sample_fixed_batch(self, count: int = 4) -> torch.Tensor:
        indices = random.sample(range(len(self.dataset)), min(count, len(self.dataset)))
        samples = torch.stack([self.dataset[idx] for idx in indices])
        return samples.to(self.device)

    def _wandb_payload(self) -> Dict[str, Any]:
        cfg = self.cfg
        return {
            "batch_size": cfg.optimiser.batch_size,
            "base_learning_rate": cfg.optimiser.base_learning_rate,
            "num_epochs": cfg.optimiser.num_epochs,
            "optimizer_type": cfg.optimiser.optimizer_type,
            "gradient_accumulation_steps": cfg.optimiser.gradient_accumulation_steps,
            "model_resolution": cfg.dataset.model_resolution,
            "high_resolution": cfg.dataset.high_resolution,
            "train_decoder_only": cfg.model.train_decoder_only,
            "full_training": cfg.model.full_training,
            "vae_kind": cfg.model.vae_kind,
            "latent_upscaler_enabled": cfg.latent_upscaler.enabled,
            "adversarial_enabled": self.adv_active,
            "adversarial_loss_type": cfg.adversarial.loss_type,
        }

    # ---------------------------------------------------------------- training
    def train(self) -> None:
        total_steps = self._total_steps()
        sample_interval = self._resolve_interval(self.cfg.logging.global_sample_interval, total_steps, self.cfg.logging.sample_interval_share)
        save_interval = self.cfg.logging.global_save_interval

        progress = tqdm(total=total_steps, disable=not self.accelerator.is_local_main_process)
        best_loss = float("inf")
        best_step = -1
        global_step = 0

        for epoch in range(self.cfg.optimiser.num_epochs):
            self.vae.train()
            if self.latent_upscaler is not None:
                self.latent_upscaler.train()
            if self.discriminator is not None:
                self.discriminator.train()
            batch_losses: List[float] = []
            batch_grads: List[float] = []
            disc_losses: List[float] = []
            disc_grads: List[float] = []

            for batch in self.dataloader:
                batch = batch.to(self.device)
                low_res = self._downsample_for_model(batch)
                reconstruction, encode_out = self._forward(latents_input=low_res)
                reconstruction = self._match_spatial(reconstruction, batch)

                disc_loss_value = None
                disc_grad_value = None
                if (
                    self.adv_active
                    and self.discriminator is not None
                    and global_step >= self.cfg.adversarial.warmup_steps
                    and self._should_train_discriminator()
                ):
                    disc_loss_value, disc_grad_value = self._discriminator_step(
                        batch, reconstruction.detach()
                    )

                with self.accelerator.accumulate(self.vae):
                    losses = self._compute_losses(reconstruction, batch, encode_out)
                    if self.adv_active and global_step >= self.cfg.adversarial.warmup_steps:
                        losses["adv"] = self._generator_adversarial_loss(batch, reconstruction)
                    total_loss, coeffs, medians = self.loss_normalizer.update(losses)
                    if not torch.isfinite(total_loss):
                        raise RuntimeError("Encountered NaN/Inf loss")

                    self.accelerator.backward(total_loss)

                    grad_norm = torch.tensor(0.0, device=self.device)
                    if self.accelerator.sync_gradients:
                        if self.cfg.optimiser.clip_grad_norm > 0:
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.trainable_params, self.cfg.optimiser.clip_grad_norm
                            )
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        self._update_ema(global_step)
                        progress.update(1)

                    if self.accelerator.is_main_process:
                        lr = self.optimizer.param_groups[0]["lr"]
                        batch_losses.append(float(total_loss.detach().item()))
                        batch_grads.append(float(grad_norm.detach().item()) if isinstance(grad_norm, torch.Tensor) else 0.0)
                        if disc_loss_value is not None:
                            disc_losses.append(disc_loss_value)
                            disc_grads.append(disc_grad_value or 0.0)
                        if self.cfg.logging.use_wandb and self.accelerator.sync_gradients:
                            log = {
                                "total_loss": float(total_loss.detach().item()),
                                "learning_rate": lr,
                                "epoch": epoch,
                            }
                            if self.cfg.logging.log_grad_norm:
                                log["grad_norm"] = batch_grads[-1]
                                if disc_loss_value is not None:
                                    log["grad_norm_discriminator"] = disc_grad_value or 0.0
                            for key, value in losses.items():
                                log[f"loss_{key}"] = float(value.detach().item())
                            for key, value in coeffs.items():
                                log[f"coeff_{key}"] = float(value)
                                log[f"median_{key}"] = float(medians.get(key, 0.0))
                            if disc_loss_value is not None:
                                log["loss_discriminator"] = disc_loss_value
                            wandb.log(log, step=global_step)

                    if (
                        self.cfg.logging.save_model
                        and self.accelerator.is_main_process
                        and save_interval
                        and global_step > 0
                        and global_step % save_interval == 0
                    ):
                        self._save_checkpoint(self.cfg.paths.checkpoints_dir / f"step_{global_step:08d}")

                    if global_step > 0 and global_step % sample_interval == 0:
                        if self.accelerator.is_main_process:
                            self._generate_and_log_samples(global_step)
                        self.accelerator.wait_for_everyone()
                        avg_loss = float(np.mean(batch_losses[-sample_interval:])) if batch_losses else float("nan")
                        avg_grad = float(np.mean(batch_grads[-sample_interval:])) if batch_grads else 0.0
                        if self.accelerator.is_main_process:
                            self.accelerator.print(
                                f"Epoch {epoch} step {global_step}: loss={avg_loss:.6f}, grad_norm={avg_grad:.6f}, lr={lr:.6e}"
                            )
                            if (
                                self.cfg.logging.save_model
                                and math.isfinite(avg_loss)
                                and avg_loss < best_loss * self.cfg.logging.save_barrier
                            ):
                                best_loss = avg_loss
                                best_step = global_step
                                self._save_best_checkpoint(best_loss, best_step)
                            if self.cfg.logging.use_wandb:
                                wandb.log({"interm_loss": avg_loss, "interm_grad": avg_grad}, step=global_step)

                if self.adv_active:
                    self.discriminator_steps += 1

            if self.accelerator.is_main_process:
                epoch_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
                self.accelerator.print(f"Epoch {epoch} finished, avg loss {epoch_loss:.6f}")
                if self.cfg.logging.use_wandb:
                    wandb.log({"epoch_loss": epoch_loss, "epoch": epoch + 1}, step=global_step)

        if self.accelerator.is_main_process:
            self.accelerator.print("Training complete – storing final model")
            if self.cfg.logging.save_model:
                self._save_final_model()
        self.accelerator.free_memory()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        self.accelerator.print("Training run complete")

    # ---------------------------------------------------------------- private
    def _downsample_for_model(self, batch: torch.Tensor) -> torch.Tensor:
        if self.cfg.dataset.high_resolution == self.cfg.dataset.model_resolution:
            return batch
        return F.interpolate(
            batch,
            size=(self.cfg.dataset.model_resolution, self.cfg.dataset.model_resolution),
            mode="bilinear",
            align_corners=False,
        )

    def _forward(self, latents_input: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        encode_input = latents_input.unsqueeze(2) if self._is_video_vae(self.vae) else latents_input
        dtype = next(self.vae.parameters()).dtype
        encode_input = encode_input.to(dtype)
        freeze_encoder = self.cfg.model.train_decoder_only or self.latent_upscaler is not None
        ctx = torch.no_grad() if freeze_encoder else nullcontext()
        with ctx:
            encoding = self.vae.encode(encode_input)
        latents = encoding.latent_dist.mean if freeze_encoder else encoding.latent_dist.sample()
        if freeze_encoder:
            latents = latents.detach()
        if self.latent_upscaler is not None:
            compiler_mod = getattr(torch, "compiler", None)
            mark_step = getattr(compiler_mod, "cudagraph_mark_step_begin", None)
            if callable(mark_step):
                mark_step()
            # Clone avoids the upstream compiled graph reusing the input buffer across iterations.
            latents = self.latent_upscaler(latents.clone())
        latents = latents.to(dtype)
        if self._is_video_vae(self.vae):
            decoded = self.vae.decode(latents).sample.squeeze(2)
        else:
            decoded = self.vae.decode(latents).sample
        return decoded, {"encoding": encoding, "latents": latents}

    def _match_spatial(self, tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if tensor.shape[-2:] == reference.shape[-2:]:
            return tensor
        return F.interpolate(tensor, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def _compute_losses(self, reconstruction: torch.Tensor, target: torch.Tensor, encode_result: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        rec = reconstruction.to(torch.float32)
        tgt = target.to(torch.float32)
        active = set(self.cfg.losses.active_losses)
        losses: Dict[str, torch.Tensor] = {}

        if "mae" in active:
            losses["mae"] = F.l1_loss(rec, tgt)
        if "mse" in active:
            losses["mse"] = F.mse_loss(rec, tgt)
        if "edge" in active:
            losses["edge"] = F.l1_loss(self._sobel(rec), self._sobel(tgt))

        if "ffl" in active:
            if self.ffl_module is None:
                losses["ffl"] = torch.zeros((), device=self.device, dtype=rec.dtype)
            else:
                losses["ffl"] = self.ffl_module(rec, tgt)

        if "lpips" in active:
            losses["lpips"] = self._lpips_loss(rec, tgt)

        if "kl" in active:
            if self.cfg.model.full_training and not self.cfg.model.train_decoder_only:
                dist = encode_result["encoding"].latent_dist
                mean = dist.mean
                logvar = dist.logvar
                losses["kl"] = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
            else:
                losses["kl"] = torch.zeros((), device=self.device, dtype=rec.dtype)
        return losses

    def _lpips_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.lpips_module is None:
            return torch.tensor(0.0, device=self.device)
        size = self.cfg.losses.lpips_eval_resolution
        pred = prediction
        tgt = target
        if size > 0 and prediction.shape[-2:] != (size, size):
            pred = F.interpolate(prediction, size=(size, size), mode="bilinear", align_corners=False)
        if size > 0 and target.shape[-2:] != (size, size):
            tgt = F.interpolate(target, size=(size, size), mode="bilinear", align_corners=False)
        return self.lpips_module(pred, tgt).mean()

    @staticmethod
    def _sobel(tensor: torch.Tensor) -> torch.Tensor:
        channels = tensor.shape[1]
        kernel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], device=tensor.device, dtype=tensor.dtype)
        kernel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], device=tensor.device, dtype=tensor.dtype)
        kernel_x = kernel_x.repeat(channels, 1, 1, 1)
        kernel_y = kernel_y.repeat(channels, 1, 1, 1)
        grad_x = F.conv2d(tensor, kernel_x, padding=1, groups=channels)
        grad_y = F.conv2d(tensor, kernel_y, padding=1, groups=channels)
        return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-12)

    @contextmanager
    def _evaluation_context(self, module: Optional[nn.Module]):
        if module is None:
            yield None
            return
        prev_mode = module.training
        first_param = next((p for p in module.parameters()), None)
        original_device = first_param.device if first_param is not None else self.device
        if original_device != self.device:
            module = module.to(self.device)
        module.eval()
        try:
            yield module
        finally:
            module.train(prev_mode)
            if original_device != self.device:
                module.to(original_device)

    def _generate_and_log_samples(self, step: int) -> None:
        try:
            vae_source = self.ema_vae.module if self.ema_vae is not None else self._unwrap_model(self.vae)
            upscaler_source: Optional[nn.Module]
            if self.ema_latent_upscaler is not None:
                upscaler_source = self.ema_latent_upscaler.module
            else:
                upscaler_source = self._unwrap_model(self.latent_upscaler)
            with self._evaluation_context(vae_source) as vae, self._evaluation_context(upscaler_source) as upscaler:
                if vae is None:
                    return
                with torch.no_grad():
                    high = self.fixed_samples
                    low = self._downsample_for_model(high)
                    first_param = next((p for p in vae.parameters()), None)
                    dtype = first_param.dtype if first_param is not None else torch.float32
                    low = low.to(dtype)
                    if self._is_video_vae(vae):
                        latents = vae.encode(low.unsqueeze(2)).latent_dist.mean
                        if upscaler is not None:
                            latents = upscaler(latents)
                        latents = latents.to(dtype)
                        recon = vae.decode(latents).sample.squeeze(2)
                    else:
                        latents = vae.encode(low).latent_dist.mean
                        if upscaler is not None:
                            latents = upscaler(latents)
                        latents = latents.to(dtype)
                        recon = vae.decode(latents).sample
                    recon = self._match_spatial(recon, high)

            self.cfg.paths.samples_dir.mkdir(parents=True, exist_ok=True)
            grid_path = self._save_sample_grid(high, recon, step)
            if self.cfg.logging.use_wandb and self.accelerator.is_main_process:
                wandb.log({"samples/pairs": wandb.Image(str(grid_path))}, step=step)
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    def _save_sample_grid(self, originals: torch.Tensor, decoded: torch.Tensor, step: int) -> Path:
        def to_pil(tensor: torch.Tensor) -> Image.Image:
            array = ((tensor.clamp(-1, 1) + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
            array = array.permute(1, 2, 0).cpu().numpy()
            return Image.fromarray(array)

        num = decoded.shape[0]
        fig, axes = plt.subplots(2, num, figsize=(num * 2.5, 6))
        if num == 1:
            axes = np.array([[axes[0]], [axes[1]]])
        for idx in range(num):
            axes[0, idx].imshow(to_pil(originals[idx]))
            axes[0, idx].axis("off")
            axes[0, idx].set_title(f"Real {idx}")
            axes[1, idx].imshow(to_pil(decoded[idx]))
            axes[1, idx].axis("off")
            axes[1, idx].set_title(f"Decoded {idx}")
        plt.tight_layout()
        grid_path = self.cfg.paths.samples_dir / f"grid_{step}.png"
        fig.savefig(grid_path, dpi=150)
        plt.close(fig)
        to_pil(originals[0]).save(self.cfg.paths.samples_dir / "sample_real.jpg", quality=95)
        to_pil(decoded[0]).save(self.cfg.paths.samples_dir / "sample_decoded.jpg", quality=95)
        return grid_path

    def _save_checkpoint(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        model = self.accelerator.unwrap_model(self.vae)
        model.save_pretrained(str(directory))
        torch.save(self.optimizer.state_dict(), directory / "optimizer.pt")
        torch.save(self.scheduler.state_dict(), directory / "scheduler.pt")
        if self.latent_upscaler is not None:
            torch.save(
                {
                    "state_dict": self.accelerator.unwrap_model(self.latent_upscaler).state_dict(),
                    "scale_factor": self.cfg.dataset.high_resolution // self.cfg.dataset.model_resolution
                    if self.cfg.dataset.high_resolution > self.cfg.dataset.model_resolution
                    else self.cfg.latent_upscaler.scale_factor,
                    "width_multiplier": self.cfg.latent_upscaler.width_multiplier,
                },
                directory / "latent_upscaler.pt",
            )
        if self.ema_vae is not None:
            torch.save(
                {
                    "state_dict": self.ema_vae.module.state_dict(),
                    "decay": self.cfg.ema.decay,
                    "update_after_step": self.cfg.ema.update_after_step,
                    "update_interval": self.cfg.ema.update_interval,
                },
                directory / "ema_vae.pt",
            )
            if self.ema_latent_upscaler is not None:
                torch.save(
                    {
                        "state_dict": self.ema_latent_upscaler.module.state_dict(),
                        "decay": self.cfg.ema.decay,
                        "update_after_step": self.cfg.ema.update_after_step,
                        "update_interval": self.cfg.ema.update_interval,
                    },
                    directory / "ema_latent_upscaler.pt",
                )

    def _save_best_checkpoint(self, loss: float, step: int) -> None:
        if self.cfg.paths.best_dir.exists():
            shutil.rmtree(self.cfg.paths.best_dir)
        self.cfg.paths.best_dir.mkdir(parents=True, exist_ok=True)
        self._save_checkpoint(self.cfg.paths.best_dir)
        (self.cfg.paths.best_dir / "best_summary.txt").write_text(f"step={step}\nloss={loss}\n", encoding="utf-8")
        self.accelerator.print(f"[CHECKPOINT] Updated best model at step {step} (loss {loss:.6f})")

    def _save_final_model(self) -> None:
        if self.cfg.paths.final_dir.exists():
            shutil.rmtree(self.cfg.paths.final_dir)
        self.cfg.paths.final_dir.mkdir(parents=True, exist_ok=True)
        self._save_checkpoint(self.cfg.paths.final_dir)

    def _unwrap_model(self, module: Optional[nn.Module]) -> Optional[nn.Module]:
        if module is None:
            return None
        if hasattr(self.accelerator, "unwrap_model"):
            try:
                module = self.accelerator.unwrap_model(module)
            except Exception:  # pragma: no cover - defensive
                pass
        if hasattr(module, "_orig_mod"):
            module = module._orig_mod
        return module

    def _is_video_vae(self, module: nn.Module) -> bool:
        encoder = getattr(module, "encoder", None)
        conv_in = getattr(encoder, "conv_in", None) if encoder is not None else None
        weight = getattr(conv_in, "weight", None) if conv_in is not None else None
        return isinstance(weight, torch.nn.Parameter) and weight.ndimension() == 5

    @staticmethod
    def _resolve_interval(configured: int, total_steps: int, share: int) -> int:
        if configured > 0:
            return configured
        return max(1, total_steps // max(1, share))


# ---------------------------------------------------------------------------
# Public entry point


def run(config: Dict[str, Any]) -> None:
    training_config = TrainingConfig.from_dict(config)
    trainer = VAETrainer(training_config)
    trainer.train()


__all__ = ["run", "TrainingConfig", "VAETrainer"]
