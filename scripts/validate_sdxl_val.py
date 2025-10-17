#!/usr/bin/env python3
"""Validation script for latent super-resolution models using SDXL validation images.

The script reads a YAML configuration that specifies:
  * the location of the `sdxl_val` image set (or a parent directory that contains it),
  * model construction parameters,
  * the checkpoint file to evaluate.

It encodes the SDXL images into VAE latents on-the-fly, applies the super-resolution
model, decodes the predictions back to pixel space, and reports aggregate metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from basicsr.models.swinir_latent_model import SwinIRLatentModel

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    MATPLOTLIB_AVAILABLE = False


try:
    from diffusers import (
        AsymmetricAutoencoderKL,
        AutoencoderKL,
        AutoencoderKLQwenImage,
        AutoencoderKLWan,
    )

    DIFFUSERS_AVAILABLE = True
except ImportError:  # pragma: no cover - defensive guard
    DIFFUSERS_AVAILABLE = False


RESAMPLE_MODES = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}

DEFAULT_RESAMPLE = Image.Resampling.BICUBIC

DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

SWINIR_FALLBACK_TARGET = "basicsr.archs.swinir_arch.SwinIR"


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    subsets: Optional[List[str]]
    extensions: Tuple[str, ...]
    scale: int
    limit: Optional[int]
    batch_size: int
    num_workers: int
    resample: Image.Resampling
    high_size: Optional[Tuple[int, int]]
    low_size: Optional[Tuple[int, int]]


@dataclass(frozen=True)
class ModelConfig:
    target: str
    params: Dict[str, object]
    device: torch.device
    dtype: torch.dtype


@dataclass(frozen=True)
class VAEConfig:
    target: str
    params: Dict[str, object]
    dtype: Optional[torch.dtype]


@dataclass(frozen=True)
class WeightsConfig:
    path: Path
    key_preference: Optional[str]
    strict: bool


@dataclass(frozen=True)
class MetricsConfig:
    compute_latent_l1: bool
    compute_latent_mse: bool
    compute_pixel_psnr: bool
    compute_pixel_l1: bool


@dataclass(frozen=True)
class OutputConfig:
    directory: Optional[Path]
    metrics_path: Optional[Path]
    per_image_path: Optional[Path]
    visualization_dir: Optional[Path]
    visualization_limit: int
    visualization_channels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an SR model on the SDXL validation set.")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the YAML configuration file with dataset, model, and weight settings.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise TypeError(f"Configuration root must be a mapping, received {type(data)!r}.")
    return dict(data)


def _parse_resize_size(value: object) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("Resize dimension must be positive.")
        return (value, value)
    if isinstance(value, Sequence) and len(value) == 2:
        first, second = value
        first_int = int(first)
        second_int = int(second)
        if first_int <= 0 or second_int <= 0:
            raise ValueError("Resize dimensions must be positive.")
        return (first_int, second_int)
    raise TypeError(f"Unsupported resize specification: {value!r}")


def _resolve_dataset_root(raw: object) -> Path:
    if not raw:
        raise ValueError("dataset.path is required in the configuration.")
    candidate = Path(str(raw)).expanduser().resolve()
    if candidate.is_dir() and candidate.name.lower() == "sdxl_val":
        return candidate
    nested = candidate / "sdxl_val"
    if nested.is_dir():
        return nested
    # Fallback to widely used naming so the script remains usable if the folder
    # is called `sdxl_validation`.
    if candidate.is_dir() and candidate.name.lower() == "sdxl_validation":
        return candidate
    validation_nested = candidate / "sdxl_validation"
    if validation_nested.is_dir():
        return validation_nested
    raise FileNotFoundError(
        f"Unable to locate 'sdxl_val' images. Checked: {candidate}, {nested}, {validation_nested}"
    )


def _normalise_subsets(value: object) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        return [trimmed] if trimmed else None
    if isinstance(value, Sequence):
        subsets = [str(item).strip() for item in value if str(item).strip()]
        return subsets or None
    raise TypeError("dataset.subset must be a string or sequence of strings.")


def _normalise_extensions(value: object) -> Tuple[str, ...]:
    if value is None:
        return DEFAULT_EXTENSIONS
    if isinstance(value, str):
        parts = [value]
    elif isinstance(value, Sequence):
        parts = value
    else:
        raise TypeError("dataset.extensions must be a string or list of strings.")
    normalized: List[str] = []
    for item in parts:
        ext = str(item).lower().strip()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.append(ext)
    return tuple(normalized) if normalized else DEFAULT_EXTENSIONS


def _parse_resample(value: object) -> Image.Resampling:
    if value is None:
        return DEFAULT_RESAMPLE
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in RESAMPLE_MODES:
            raise KeyError(f"Unsupported resample method '{value}'. Available: {sorted(RESAMPLE_MODES)}")
        return RESAMPLE_MODES[key]
    raise TypeError("dataset.resample must be a string.")


def parse_dataset_config(raw: Mapping[str, object]) -> DatasetConfig:
    root = _resolve_dataset_root(raw.get("path"))
    subsets = _normalise_subsets(raw.get("subset") or raw.get("subsets"))
    extensions = _normalise_extensions(raw.get("extensions"))
    scale = int(raw.get("scale") or 2)
    if scale <= 0:
        raise ValueError("dataset.scale must be positive.")
    limit_value = raw.get("limit")
    limit = int(limit_value) if limit_value is not None else None
    batch_size = int(raw.get("batch_size") or 1)
    num_workers = int(raw.get("num_workers") or 0)
    resample = _parse_resample(raw.get("resample"))
    high_size = _parse_resize_size(raw.get("high_size") or raw.get("high_resolution"))
    low_size = _parse_resize_size(raw.get("low_size") or raw.get("low_resolution"))
    return DatasetConfig(
        root=root,
        subsets=subsets,
        extensions=extensions,
        scale=scale,
        limit=limit,
        batch_size=batch_size,
        num_workers=num_workers,
        resample=resample,
        high_size=high_size,
        low_size=low_size,
    )


def _resolve_target_parts(target: str) -> Tuple[str, str]:
    if "." not in target:
        raise ValueError(f"Target must be a fully qualified name: {target!r}")
    module_name, class_name = target.rsplit(".", 1)
    return module_name, class_name


def _parse_dtype(spec: object, *, default: Optional[torch.dtype] = None) -> torch.dtype:
    if spec is None:
        if default is None:
            return torch.float32
        return default
    if isinstance(spec, torch.dtype):
        return spec
    if isinstance(spec, str):
        text = spec.strip().lower()
        if not text:
            return torch.float32
        if not text.startswith("torch."):
            text = f"torch.{text}"
        attr = text.split(".")[-1]
        if not hasattr(torch, attr):
            raise ValueError(f"Unsupported dtype specifier: {spec}")
        dtype = getattr(torch, attr)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Dtype specifier did not resolve to torch.dtype: {spec}")
        return dtype
    raise TypeError(f"Unsupported dtype specification: {type(spec)!r}")


def _parse_device(value: object) -> torch.device:
    if value is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(value, torch.device):
        return value
    text = str(value).strip().lower()
    if text == "cpu":
        return torch.device("cpu")
    if text.startswith("cuda"):
        return torch.device(text)
    if text in {"auto", "gpu"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text.isdigit():
        return torch.device(f"cuda:{text}")
    raise ValueError(f"Unsupported device specification: {value!r}")


def parse_model_config(raw: Mapping[str, object]) -> Tuple[ModelConfig, VAEConfig]:
    arch_cfg = raw.get("arch")
    if arch_cfg is None:
        raise ValueError("model.arch section is required.")
    if isinstance(arch_cfg, str):
        target = arch_cfg
        params = dict(raw.get("params", {}))
    elif isinstance(arch_cfg, Mapping):
        target = str(
            arch_cfg.get("target")
            or arch_cfg.get("class_path")
            or arch_cfg.get("cls")
            or arch_cfg.get("type")
            or SWINIR_FALLBACK_TARGET
        )
        params = dict(arch_cfg.get("params", {}))
    else:
        raise TypeError("model.arch must be a mapping or dotted class path string.")

    if "." not in target:
        if target.lower() == "swinir":
            target = SWINIR_FALLBACK_TARGET
        else:
            raise ValueError(f"Unknown model type '{target}'. Provide a fully qualified class path.")

    device = _parse_device(raw.get("device"))
    dtype = _parse_dtype(raw.get("dtype"), default=torch.float32)

    vae_cfg = raw.get("vae", {})
    if not isinstance(vae_cfg, Mapping):
        raise TypeError("model.vae must be a mapping with VAE parameters.")
    vae_target = str(
        vae_cfg.get("target")
        or vae_cfg.get("class_path")
        or vae_cfg.get("type")
        or vae_cfg.get("kind")
        or "AutoencoderKL"
    )
    vae_params = dict(vae_cfg.get("params", {}))
    vae_dtype = None
    if "dtype" in vae_cfg:
        vae_dtype = _parse_dtype(vae_cfg.get("dtype"), default=None)

    return (
        ModelConfig(target=target, params=params, device=device, dtype=dtype),
        VAEConfig(target=vae_target, params=vae_params, dtype=vae_dtype),
    )


def parse_weights_config(raw: Mapping[str, object]) -> WeightsConfig:
    weight_path = raw.get("path")
    if not weight_path:
        raise ValueError("weights.path is required.")
    key = raw.get("key") or raw.get("state_key")
    strict = bool(raw.get("strict", False))
    return WeightsConfig(path=Path(str(weight_path)).expanduser().resolve(), key_preference=key, strict=strict)


def parse_metrics_config(raw: Mapping[str, object] | None) -> MetricsConfig:
    if raw is None:
        raw = {}
    return MetricsConfig(
        compute_latent_l1=bool(raw.get("latent_l1", True)),
        compute_latent_mse=bool(raw.get("latent_mse", True)),
        compute_pixel_psnr=bool(raw.get("pixel_psnr", True)),
        compute_pixel_l1=bool(raw.get("pixel_l1", True)),
    )


def parse_output_config(raw: Mapping[str, object] | None) -> OutputConfig:
    if raw is None:
        raw = {}
    base_dir_raw = raw.get("dir") or raw.get("directory")
    base_dir = Path(base_dir_raw).expanduser().resolve() if base_dir_raw else None

    def _resolve(path_value: Optional[object], default_name: str) -> Optional[Path]:
        if path_value:
            return Path(str(path_value)).expanduser().resolve()
        if base_dir is not None:
            return base_dir / default_name
        return None

    metrics_path = _resolve(raw.get("metrics_path"), "metrics.json")
    per_image_path = _resolve(raw.get("per_image_path"), "per_image_metrics.json")
    viz_dir_raw = raw.get("visualizations_dir") or raw.get("visualization_dir")
    if not viz_dir_raw and base_dir is not None:
        viz_dir_raw = base_dir / "visualizations"
    viz_limit_raw = raw.get("visualizations_limit") or raw.get("visualization_limit")
    viz_channels_raw = raw.get("visualizations_channels") or raw.get("visualization_channels")
    viz_limit = int(viz_limit_raw) if viz_limit_raw is not None else 8
    if viz_limit < 0:
        raise ValueError("output.visualizations_limit must be non-negative.")
    viz_channels = int(viz_channels_raw) if viz_channels_raw is not None else 4
    if viz_channels <= 0:
        raise ValueError("output.visualizations_channels must be positive.")
    return OutputConfig(
        directory=base_dir,
        metrics_path=metrics_path,
        per_image_path=per_image_path,
        visualization_dir=Path(viz_dir_raw).expanduser().resolve() if viz_dir_raw else None,
        visualization_limit=viz_limit,
        visualization_channels=viz_channels,
    )


class SdxlValDataset(Dataset):
    """Dataset that loads SDXL validation images and produces paired low/high tensors."""

    def __init__(
        self,
        image_paths: Sequence[Path],
        *,
        scale: int,
        resample: Image.Resampling,
        high_size: Optional[Tuple[int, int]],
        low_size: Optional[Tuple[int, int]],
    ) -> None:
        self.image_paths = list(image_paths)
        self.scale = scale
        self.resample = resample
        self.high_size = high_size
        self.low_size = low_size
        if not self.image_paths:
            raise RuntimeError("No validation images were found under the requested directory.")

    def __len__(self) -> int:
        return len(self.image_paths)

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor * 2.0 - 1.0

    def _resize_if_needed(self, image: Image.Image, size: Optional[Tuple[int, int]]) -> Image.Image:
        if size is None:
            return image
        if image.size == size:
            return image
        return image.resize(size, self.resample)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        path = self.image_paths[idx]
        with Image.open(path) as handle:
            image = handle.convert("RGB")

        high_image = self._resize_if_needed(image, self.high_size)

        if self.low_size is not None:
            low_image = self._resize_if_needed(image, self.low_size)
        else:
            width, height = high_image.size
            low_w = max(1, width // self.scale)
            low_h = max(1, height // self.scale)
            low_image = high_image.resize((low_w, low_h), self.resample)

        gt_tensor = self._to_tensor(high_image)
        lq_tensor = self._to_tensor(low_image)

        return {
            "lq_pixels": lq_tensor,
            "gt_pixels": gt_tensor,
            "path": str(path),
        }


def gather_image_paths(cfg: DatasetConfig) -> List[Path]:
    candidates: List[Path] = []
    if cfg.subsets:
        for subset in cfg.subsets:
            subset_dir = cfg.root / subset
            if not subset_dir.is_dir():
                raise FileNotFoundError(f"Subset '{subset}' does not exist under {cfg.root}")
            candidates.extend(_scan_images(subset_dir, cfg.extensions))
    else:
        candidates.extend(_scan_images(cfg.root, cfg.extensions))
    candidates = sorted(candidates)
    if cfg.limit is not None:
        candidates = candidates[: cfg.limit]
    if not candidates:
        raise RuntimeError(f"No images found in {cfg.root} matching extensions {cfg.extensions}")
    return candidates


def _scan_images(root: Path, extensions: Iterable[str]) -> List[Path]:
    allowed = tuple(ext.lower() for ext in extensions)
    result: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed:
            result.append(path)
    return result


def instantiate_class(target: str, *, params: Mapping[str, object]) -> object:
    module_name, class_name = _resolve_target_parts(target)
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)
    return cls(**params)


def instantiate_model(model_cfg: ModelConfig) -> torch.nn.Module:
    model = instantiate_class(model_cfg.target, params=model_cfg.params)
    model = model.to(device=model_cfg.device, dtype=model_cfg.dtype)
    model.eval()
    return model


def instantiate_vae(vae_cfg: VAEConfig, device: torch.device) -> torch.nn.Module:
    if not DIFFUSERS_AVAILABLE:
        raise RuntimeError("diffusers is required for VAE encoding/decoding but is not installed.")

    target = vae_cfg.target.lower()
    params = dict(vae_cfg.params)
    dtype = vae_cfg.dtype

    def _apply_dtype(module: torch.nn.Module) -> torch.nn.Module:
        if dtype is not None:
            module = module.to(dtype=dtype)
        return module.to(device).eval()

    if target in {"autoencoderkl", "kl", "vae"}:
        return _apply_dtype(AutoencoderKL.from_pretrained(**params))
    if target in {"asym_autoencoderkl", "asymmetric_autoencoderkl", "asym_kl"}:
        return _apply_dtype(AsymmetricAutoencoderKL.from_pretrained(**params))
    if target in {"qwen", "autoencoderklqwenimage"}:
        return _apply_dtype(AutoencoderKLQwenImage.from_pretrained(**params))
    if target in {"wan", "autoencoderklwan"}:
        return _apply_dtype(AutoencoderKLWan.from_pretrained(**params))
    # Assume a HuggingFace repo string was supplied under params["pretrained_model_name_or_path"]
    if "pretrained_model_name_or_path" in params or "model_id" in params:
        return _apply_dtype(AutoencoderKL.from_pretrained(**params))
    # Backwards compatible helpers: allow hf_repo/hf_revision keys.
    repo_id = params.pop("hf_repo", None) or params.pop("repo_id", None)
    if not repo_id:
        repo_id = "stabilityai/sdxl-vae"
    if "subfolder" not in params and "hf_subfolder" in params:
        params["subfolder"] = params.pop("hf_subfolder")
    if "revision" not in params and "hf_revision" in params:
        params["revision"] = params.pop("hf_revision")
    if "use_auth_token" not in params and "hf_auth_token" in params:
        params["use_auth_token"] = params.pop("hf_auth_token")
    return _apply_dtype(AutoencoderKL.from_pretrained(repo_id, **params))


def _module_device(module: torch.nn.Module, fallback: Optional[torch.device] = None) -> torch.device:
    tensor = next(module.parameters(), None)
    if tensor is None:
        tensor = next(module.buffers(), None)
    if tensor is not None:
        return tensor.device
    if fallback is not None:
        return fallback
    return torch.device("cpu")


def load_checkpoint(
    model: torch.nn.Module,
    weights_cfg: WeightsConfig,
    fallback_device: Optional[torch.device] = None,
) -> None:
    device = _module_device(model, fallback=fallback_device)
    checkpoint = torch.load(weights_cfg.path, map_location=device)
    state_dict = None
    candidate_keys = []
    if weights_cfg.key_preference:
        candidate_keys.append(weights_cfg.key_preference)
    candidate_keys.extend(["params_ema", "params", "state_dict", "model"])
    for key in candidate_keys:
        if isinstance(checkpoint, Mapping) and key in checkpoint:
            state_dict = checkpoint[key]
            break
    if state_dict is None:
        if isinstance(checkpoint, Mapping):
            # Potentially nested BasicSR format.
            for value in checkpoint.values():
                if isinstance(value, Mapping) and all(isinstance(k, str) for k in value.keys()):
                    state_dict = value
                    break
        if state_dict is None:
            state_dict = checkpoint

    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Unexpected checkpoint structure in {weights_cfg.path}")

    state_dict = dict(state_dict)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=weights_cfg.strict)
    if missing:
        print(f"[warn] Missing keys when loading checkpoint: {missing}", file=sys.stderr)
    if unexpected:
        print(f"[warn] Unexpected keys when loading checkpoint: {unexpected}", file=sys.stderr)


def encode_latents(vae: torch.nn.Module, pixels: torch.Tensor) -> torch.Tensor:
    if pixels.dim() != 4:
        raise ValueError("Expected pixel tensor with shape [B, C, H, W].")
    with torch.no_grad():
        posterior = vae.encode(pixels).latent_dist
        mean = posterior.mean
        if mean.dim() == 5:  # diffusers sometimes inserts an extra temporal dimension
            mean = mean.squeeze(2)
    return mean


def decode_latents(vae: torch.nn.Module, latents: torch.Tensor) -> torch.Tensor:
    if latents.dim() != 4:
        raise ValueError("Expected latent tensor with shape [B, C, H, W].")
    with torch.no_grad():
        decoded = vae.decode(latents).sample
        decoded = torch.clamp(decoded, -1.0, 1.0)
    return decoded


def _mean_reduce(tensor: torch.Tensor) -> torch.Tensor:
    dims = list(range(1, tensor.ndim))
    return tensor.mean(dim=dims)


def _relative_output_path(image_path: str, dataset_root: Path) -> Path:
    path_obj = Path(image_path)
    try:
        return path_obj.resolve().relative_to(dataset_root.resolve())
    except Exception:
        return Path(path_obj.name)


def _infer_dataset_name(dataset_cfg: DatasetConfig) -> str:
    if dataset_cfg.subsets:
        return dataset_cfg.subsets[0]
    return dataset_cfg.root.name


def build_val_metrics_config(metrics_cfg: MetricsConfig) -> Dict[str, Dict[str, object]]:
    metrics: Dict[str, Dict[str, object]] = {}
    if metrics_cfg.compute_latent_l1:
        metrics["latent_l1"] = {"type": "L1Loss", "space": "latent", "better": "lower"}
    if metrics_cfg.compute_latent_mse:
        metrics["latent_mse"] = {"type": "MSELoss", "space": "latent", "better": "lower"}
    if metrics_cfg.compute_pixel_psnr:
        metrics["pixel_psnr"] = {"type": "calculate_psnr", "space": "pixel", "better": "higher"}
    if metrics_cfg.compute_pixel_l1:
        metrics["pixel_l1"] = {"type": "L1Loss", "space": "pixel", "better": "lower"}
    return metrics


def _vae_source_from_config(dataset_name: str, vae_cfg: VAEConfig) -> Tuple[str, Dict[str, object]]:
    params = dict(vae_cfg.params)
    mapping: Dict[str, object] = {}

    load_from = params.get("load_from")
    if load_from:
        mapping["load_from"] = str(load_from)

    repo = (
        params.get("pretrained_model_name_or_path")
        or params.get("model_id")
        or params.get("hf_repo")
    )
    if repo:
        mapping["hf_repo"] = str(repo)

    if "hf_subfolder" in params:
        mapping["hf_subfolder"] = params["hf_subfolder"]
    if "subfolder" in params:
        mapping["hf_subfolder"] = params["subfolder"]

    if "hf_revision" in params:
        mapping["hf_revision"] = params["hf_revision"]
    if "revision" in params:
        mapping["hf_revision"] = params["revision"]

    if "hf_auth_token" in params:
        mapping["hf_auth_token"] = params["hf_auth_token"]
    if "use_auth_token" in params:
        mapping["hf_auth_token"] = params["use_auth_token"]

    if vae_cfg.dtype is not None:
        mapping["weights_dtype"] = str(vae_cfg.dtype).split(".")[-1]

    target_lower = vae_cfg.target.lower()
    if "qwen" in target_lower:
        mapping["vae_kind"] = "qwen"
    elif "wan" in target_lower:
        mapping["vae_kind"] = "wan"
    elif "asym" in target_lower:
        mapping["vae_kind"] = "kl"
    else:
        mapping["vae_kind"] = "kl"

    vae_name = f"{dataset_name}_vae"
    return vae_name, mapping


def resolve_visualization_root(dataset_cfg: DatasetConfig, output_cfg: OutputConfig) -> Path:
    if output_cfg.visualization_dir is not None:
        return output_cfg.visualization_dir.resolve()
    if output_cfg.directory is not None:
        return (output_cfg.directory / "visualizations").resolve()
    return (dataset_cfg.root / "visualizations").resolve()


def build_model_options(
    *,
    dataset_cfg: DatasetConfig,
    model_cfg: ModelConfig,
    vae_cfg: VAEConfig,
    weights_cfg: WeightsConfig,
    metrics_cfg: MetricsConfig,
    output_cfg: OutputConfig,
    dataset_name: str,
) -> Tuple[Dict[str, object], Path, str]:
    visualization_root = resolve_visualization_root(dataset_cfg, output_cfg)
    visualization_root.mkdir(parents=True, exist_ok=True)

    metrics_opt = build_val_metrics_config(metrics_cfg)

    network_g_opt = dict(model_cfg.params)
    network_g_opt["type"] = model_cfg.target.rsplit(".", 1)[-1]

    vae_name, vae_source = _vae_source_from_config(dataset_name, vae_cfg)

    opt: Dict[str, object] = {
        "name": dataset_name,
        "is_train": False,
        "dist": False,
        "rank": 0,
        "num_gpu": 1 if model_cfg.device.type == "cuda" and torch.cuda.is_available() else 0,
        "scale": dataset_cfg.scale,
        "network_g": network_g_opt,
        "path": {
            "pretrain_network_g": None,
            "param_key_g": weights_cfg.key_preference or "params",
            "strict_load_g": weights_cfg.strict,
            "visualization": str(visualization_root),
        },
        "val": {
            "metrics": metrics_opt,
            "suffix": dataset_name,
            "pbar": True,
        },
        "logger": {
            "use_tb_logger": False,
            "wandb": {},
        },
        "datasets": {
            "val": {
                "name": dataset_name,
                "vae_names": vae_name,
            }
        },
        "vae_sources": {
            vae_name: vae_source,
        },
    }

    return opt, visualization_root, vae_name


def run_validation(
    dataset_cfg: DatasetConfig,
    model_cfg: ModelConfig,
    vae_cfg: VAEConfig,
    weights_cfg: WeightsConfig,
    metrics_cfg: MetricsConfig,
    output_cfg: OutputConfig,
) -> None:
    image_paths = gather_image_paths(dataset_cfg)
    dataset = SdxlValDataset(
        image_paths,
        scale=dataset_cfg.scale,
        resample=dataset_cfg.resample,
        high_size=dataset_cfg.high_size,
        low_size=dataset_cfg.low_size,
    )
    dataloader_kwargs = {}
    if dataset_cfg.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
    dataloader = DataLoader(
        dataset,
        batch_size=dataset_cfg.batch_size,
        shuffle=False,
        num_workers=dataset_cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        **dataloader_kwargs,
    )

    dataset_name = _infer_dataset_name(dataset_cfg)
    opt, visualization_root, vae_name = build_model_options(
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        vae_cfg=vae_cfg,
        weights_cfg=weights_cfg,
        metrics_cfg=metrics_cfg,
        output_cfg=output_cfg,
        dataset_name=dataset_name,
    )

    print(f"[info] Loaded {len(dataset)} validation images from {dataset_cfg.root}")
    print(f"[info] Using device: {model_cfg.device}")
    print(f"[info] Visualization root: {visualization_root}")

    model = SwinIRLatentModel(opt)
    load_checkpoint(model.net_g, weights_cfg, fallback_device=model.device)
    vae = instantiate_vae(vae_cfg, model.device)

    metrics_config: Dict[str, Dict[str, object]] = opt["val"]["metrics"]
    metric_sums: Dict[str, float] = {name: 0.0 for name in metrics_config.keys()}
    per_image_records: List[Dict[str, object]] = []

    visualization_limit = max(0, output_cfg.visualization_limit)
    visualization_enabled = visualization_limit > 0 and MATPLOTLIB_AVAILABLE
    if visualization_limit > 0 and not MATPLOTLIB_AVAILABLE:
        print("[warn] Matplotlib is not available; skipping visualization exports.")
    save_dir = Path(opt["path"]["visualization"]) / dataset_name
    if visualization_enabled:
        save_dir.mkdir(parents=True, exist_ok=True)

    if VAE_AVAILABLE:
        model._get_cache_dir(dataset_name)
    model._val_decode_mem_cache.clear()

    total_images = 0
    visualization_count = 0
    progress = tqdm(dataloader, desc="Validating", unit="image")

    with torch.no_grad():
        for batch in progress:
            lq_pixels = batch["lq_pixels"].to(model_cfg.device, dtype=model_cfg.dtype)
            gt_pixels = batch["gt_pixels"].to(model_cfg.device, dtype=model_cfg.dtype)
            paths = batch["path"]
            batch_size = lq_pixels.shape[0]

            lq_latents = encode_latents(vae, lq_pixels).to(model.device, dtype=model_cfg.dtype)
            gt_latents = encode_latents(vae, gt_pixels).to(model.device, dtype=model_cfg.dtype)

            feed_dict: Dict[str, object] = {
                "lq": lq_latents,
                "gt": gt_latents,
                "lq_path": paths,
                "gt_path": paths,
                "vae_name": [vae_name] * batch_size,
            }
            model.feed_data(feed_dict)
            model.test()

            pred_latent = model.output.detach()
            lq_latent = model.lq.detach()
            gt_latent = model.gt.detach()

            for idx in range(batch_size):
                total_images += 1
                img_path = paths[idx]
                img_name = Path(img_path).stem
                rel_path = _relative_output_path(img_path, dataset_cfg.root)

                lq_single = lq_latent[idx : idx + 1]
                pred_single = pred_latent[idx : idx + 1]
                gt_single = gt_latent[idx : idx + 1]

                decoded_lq = model._decode_with_cache(
                    lq_single, img_name=img_name, role="lq", dataset_name=dataset_name
                )
                decoded_gt = model._decode_with_cache(
                    gt_single, img_name=img_name, role="gt", dataset_name=dataset_name
                )
                decoded_pred = model._decode_with_cache(
                    pred_single, img_name=img_name, role="pred", dataset_name=dataset_name
                )

                sample_metrics: Dict[str, float] = {}
                for metric_name, metric_opt in metrics_config.items():
                    value = model.calculate_metric_in_space(
                        pred_single,
                        gt_single,
                        metric_name,
                        metric_opt,
                        decoded_pred=decoded_pred,
                        decoded_gt=decoded_gt,
                    )
                    metric_sums[metric_name] += value
                    sample_metrics[metric_name] = float(value)

                per_image_records.append(
                    {
                        "path": str(img_path),
                        "relative_path": str(rel_path),
                        "metrics": sample_metrics,
                    }
                )

                if visualization_enabled and visualization_count < visualization_limit:
                    save_path = save_dir / f"{img_name}_{opt['name']}.png"
                    model._save_comparison_plot(
                        save_path,
                        img_name,
                        lq_single.cpu(),
                        pred_single.cpu(),
                        gt_single.cpu(),
                        decoded_lq.float() if decoded_lq is not None else None,
                        decoded_pred.float() if decoded_pred is not None else None,
                        decoded_gt.float() if decoded_gt is not None else None,
                        log_to_wandb=False,
                        current_iter=0,
                    )
                    visualization_count += 1

            if metrics_config:
                running = {
                    name: metric_sums[name] / total_images for name in metrics_config.keys()
                }
                progress.set_postfix({key: f"{value:.4f}" for key, value in running.items()})

    if total_images == 0:
        print("[warn] Validation dataset is empty; no metrics computed.")
        return

    averages = {name: metric_sums[name] / total_images for name in metrics_config.keys()}
    averages_float = {name: float(value) for name, value in averages.items()}

    print("\nValidation complete.")
    print(f"num_samples: {total_images}")
    for name, value in averages_float.items():
        print(f"{name}: {value:.6f}")

    if output_cfg.metrics_path:
        output_cfg.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"num_samples": total_images, "metrics": averages_float}
        with output_cfg.metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"[info] Wrote aggregated metrics to {output_cfg.metrics_path}")

    if output_cfg.per_image_path and per_image_records:
        output_cfg.per_image_path.parent.mkdir(parents=True, exist_ok=True)
        with output_cfg.per_image_path.open("w", encoding="utf-8") as handle:
            json.dump(per_image_records, handle, indent=2)
        print(f"[info] Wrote per-image metrics to {output_cfg.per_image_path}")

    if visualization_enabled:
        print(f"[info] Saved {visualization_count} visualization figure(s) to {save_dir}")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    dataset_cfg = parse_dataset_config(config.get("dataset", {}))
    model_cfg, vae_cfg = parse_model_config(config.get("model", {}))
    weights_cfg = parse_weights_config(config.get("weights", {}))
    metrics_cfg = parse_metrics_config(config.get("metrics"))
    output_cfg = parse_output_config(config.get("output"))

    run_validation(dataset_cfg, model_cfg, vae_cfg, weights_cfg, metrics_cfg, output_cfg)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
