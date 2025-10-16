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
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

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


def compute_metrics(
    *,
    pred_latents: torch.Tensor,
    gt_latents: torch.Tensor,
    pred_pixels: Optional[torch.Tensor],
    gt_pixels: Optional[torch.Tensor],
    metrics_cfg: MetricsConfig,
) -> Dict[str, torch.Tensor]:
    results: Dict[str, torch.Tensor] = {}
    if metrics_cfg.compute_latent_l1:
        results["latent_l1"] = _mean_reduce(torch.abs(pred_latents - gt_latents))
    if metrics_cfg.compute_latent_mse:
        diff = pred_latents - gt_latents
        results["latent_mse"] = _mean_reduce(diff * diff)
    if metrics_cfg.compute_pixel_psnr and pred_pixels is not None and gt_pixels is not None:
        mse = _mean_reduce((pred_pixels - gt_pixels) ** 2)
        eps = torch.finfo(mse.dtype).eps
        psnr = 10 * torch.log10(1.0 / torch.clamp(mse, min=eps))
        results["pixel_psnr"] = psnr
    if metrics_cfg.compute_pixel_l1 and pred_pixels is not None and gt_pixels is not None:
        results["pixel_l1"] = _mean_reduce(torch.abs(pred_pixels - gt_pixels))
    return results


def _tensor_to_image_range(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)


def _relative_output_path(image_path: str, dataset_root: Path) -> Path:
    path_obj = Path(image_path)
    try:
        return path_obj.resolve().relative_to(dataset_root.resolve())
    except Exception:
        return Path(path_obj.name)


def _latent_preview(latent: torch.Tensor, max_channels: int) -> np.ndarray:
    latent_np = latent.detach().cpu().numpy()
    channels, height, width = latent_np.shape
    preview_channels = max(1, min(max_channels, channels))
    cols = 2 if preview_channels > 1 else 1
    rows = int(math.ceil(preview_channels / cols))
    canvas = np.zeros((rows * height, cols * width), dtype=latent_np.dtype)
    for idx in range(preview_channels):
        row, col = divmod(idx, cols)
        canvas[row * height : (row + 1) * height, col * width : (col + 1) * width] = latent_np[idx]
    return canvas


def _tensor_to_numpy_rgb(tensor: torch.Tensor) -> np.ndarray:
    data = tensor.detach().cpu()
    if data.dim() == 4:
        data = data.squeeze(0)
    data = torch.clamp(data, -1.0, 1.0)
    data = (data + 1.0) * 0.5
    data = data.permute(1, 2, 0).contiguous()
    return data.numpy()


def _pixel_psnr(pred: torch.Tensor, target: torch.Tensor) -> Optional[float]:
    pred01 = _tensor_to_image_range(pred)
    target01 = _tensor_to_image_range(target)
    mse = torch.mean((pred01 - target01) ** 2).item()
    if mse <= 0:
        return None
    return 10.0 * math.log10(1.0 / mse)


def _sync_if_cuda(device: torch.device) -> None:
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_memory(device: torch.device) -> None:
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _max_memory_allocated(device: torch.device) -> int:
    if isinstance(device, torch.device) and device.type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    return 0


def _timed_call(device: torch.device, fn, *args, **kwargs):
    _sync_if_cuda(device)
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    _sync_if_cuda(device)
    duration = time.perf_counter() - start
    return result, duration


def _save_visualization_figure(
    *,
    save_path: Path,
    image_name: str,
    lq_latent: torch.Tensor,
    pred_latent: torch.Tensor,
    gt_latent: torch.Tensor,
    decoded_lq: Optional[torch.Tensor],
    decoded_pred: Optional[torch.Tensor],
    decoded_gt: Optional[torch.Tensor],
    channels: int,
) -> None:
    if not MATPLOTLIB_AVAILABLE:
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    # Latent panels
    lq_panel = _latent_preview(lq_latent, channels)
    im0 = axes[0, 0].imshow(lq_panel, cmap="viridis")
    axes[0, 0].set_title(
        f"Input Latents\n{lq_latent.shape[0]} ch @ {lq_latent.shape[1]}x{lq_latent.shape[2]}"
    )
    axes[0, 0].axis("off")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    pred_panel = _latent_preview(pred_latent, channels)
    im1 = axes[0, 1].imshow(pred_panel, cmap="viridis")
    axes[0, 1].set_title(
        f"Predicted Latents\n{pred_latent.shape[0]} ch @ {pred_latent.shape[1]}x{pred_latent.shape[2]}"
    )
    axes[0, 1].axis("off")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    gt_panel = _latent_preview(gt_latent, channels)
    im2 = axes[0, 2].imshow(gt_panel, cmap="viridis")
    axes[0, 2].set_title(
        f"Ground Truth Latents\n{gt_latent.shape[0]} ch @ {gt_latent.shape[1]}x{gt_latent.shape[2]}"
    )
    axes[0, 2].axis("off")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    latent_diff = torch.mean(torch.abs(gt_latent - pred_latent), dim=0).detach().cpu().numpy()
    im3 = axes[0, 3].imshow(latent_diff, cmap="magma")
    axes[0, 3].set_title("Latent Difference |GT - Pred| (mean over channels)")
    axes[0, 3].axis("off")
    fig.colorbar(im3, ax=axes[0, 3], fraction=0.046, pad=0.04)

    pixel_metrics: Dict[str, float] = {}
    decoded_available = decoded_lq is not None and decoded_pred is not None and decoded_gt is not None

    if decoded_available:
        decoded_lq_np = _tensor_to_numpy_rgb(decoded_lq)  # already [-1, 1] -> 0, 1 conversions inside
        decoded_pred_np = _tensor_to_numpy_rgb(decoded_pred)
        decoded_gt_np = _tensor_to_numpy_rgb(decoded_gt)

        axes[1, 0].imshow(decoded_lq_np)
        axes[1, 0].set_title("Decoded Input\n(low-resolution reconstruction)")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(decoded_pred_np)
        axes[1, 1].set_title("Decoded Prediction")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(decoded_gt_np)
        axes[1, 2].set_title("Decoded Ground Truth")
        axes[1, 2].axis("off")

        pred_tensor = decoded_pred.detach()
        gt_tensor = decoded_gt.detach()
        psnr_value = _pixel_psnr(pred_tensor, gt_tensor)
        if psnr_value is not None:
            pixel_metrics["psnr"] = psnr_value
        l1_value = torch.mean(torch.abs(_tensor_to_image_range(pred_tensor) - _tensor_to_image_range(gt_tensor))).item()
        pixel_metrics["pixel_l1"] = l1_value

        diff_map = torch.mean(
            torch.abs(_tensor_to_image_range(gt_tensor) - _tensor_to_image_range(pred_tensor)), dim=0
        ).cpu()
        diff_np = diff_map.numpy()
        if diff_np.max() > 0:
            diff_np = diff_np / diff_np.max()
        im_bottom = axes[1, 3].imshow(diff_np, cmap="magma")
        axes[1, 3].set_title("Pixel Difference |GT - Pred|\n(mean absolute error)")
        axes[1, 3].axis("off")
        fig.colorbar(im_bottom, ax=axes[1, 3], fraction=0.046, pad=0.04)
    else:
        for col in range(4):
            axes[1, col].text(
                0.5,
                0.5,
                "No decoded images\n(VAE unavailable)",
                ha="center",
                va="center",
                fontsize=12,
            )
            axes[1, col].axis("off")

    latent_l1 = torch.mean(torch.abs(pred_latent - gt_latent)).item()
    latent_mse = torch.mean((pred_latent - gt_latent) ** 2).item()
    title_parts = [
        image_name,
        f"Latent L1: {latent_l1:.6f}",
        f"Latent MSE: {latent_mse:.6f}",
    ]
    if "psnr" in pixel_metrics:
        title_parts.append(f"Pixel PSNR: {pixel_metrics['psnr']:.2f} dB")
    if "pixel_l1" in pixel_metrics:
        title_parts.append(f"Pixel L1: {pixel_metrics['pixel_l1']:.6f}")
    fig.suptitle(" | ".join(title_parts), fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.35)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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

    print(f"[info] Loaded {len(dataset)} validation images from {dataset_cfg.root}")
    print(f"[info] Using device: {model_cfg.device}")

    model = instantiate_model(model_cfg)
    load_checkpoint(model, weights_cfg, fallback_device=model_cfg.device)
    vae = instantiate_vae(vae_cfg, model_cfg.device)

    total_samples = 0
    metric_sums: Dict[str, float] = defaultdict(float)
    per_image_records: List[Dict[str, object]] = []
    using_cuda = isinstance(model_cfg.device, torch.device) and model_cfg.device.type == "cuda"
    total_encode_time = 0.0
    total_model_time = 0.0
    total_decode_time = 0.0
    total_batch_time = 0.0
    max_peak_bytes = 0
    memory_per_image_accum = 0.0
    memory_samples = 0
    visualization_enabled = (
        output_cfg.visualization_dir is not None and output_cfg.visualization_limit > 0
    )
    if visualization_enabled and not MATPLOTLIB_AVAILABLE:
        print("[warn] Matplotlib is not available; skipping visualization exports.")
        visualization_enabled = False
    if visualization_enabled:
        output_cfg.visualization_dir.mkdir(parents=True, exist_ok=True)
    visualizations_saved = 0
    pixel_metric_warning_issued = False

    progress = tqdm(dataloader, desc="Validating", unit="batch")
    inference_context = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
    with inference_context():
        for batch in progress:
            if using_cuda:
                _sync_if_cuda(model_cfg.device)
                _reset_peak_memory(model_cfg.device)
            batch_start = time.perf_counter()
            lq_pixels = batch["lq_pixels"].to(
                model_cfg.device, dtype=model_cfg.dtype, non_blocking=using_cuda
            )
            gt_pixels = batch["gt_pixels"].to(
                model_cfg.device, dtype=model_cfg.dtype, non_blocking=using_cuda
            )
            paths = batch["path"]

            lq_latents, encode_time_lq = _timed_call(model_cfg.device, encode_latents, vae, lq_pixels)
            gt_latents, encode_time_gt = _timed_call(model_cfg.device, encode_latents, vae, gt_pixels)
            total_encode_time += encode_time_lq + encode_time_gt

            pred_latents, model_time = _timed_call(model_cfg.device, model, lq_latents)
            total_model_time += model_time

            pred_pixels = None
            gt_pixels_range = None
            need_pixel_metrics = metrics_cfg.compute_pixel_psnr or metrics_cfg.compute_pixel_l1

            decoded_pred_batch: Optional[torch.Tensor] = None
            decoded_gt_batch: Optional[torch.Tensor] = None
            decoded_lq_batch: Optional[torch.Tensor] = None
            decode_time_batch = 0.0

            if need_pixel_metrics or visualization_enabled:
                decoded_pred_batch, decode_time = _timed_call(
                    model_cfg.device, decode_latents, vae, pred_latents
                )
                decode_time_batch += decode_time
            if visualization_enabled:
                decoded_lq_batch, decode_time = _timed_call(
                    model_cfg.device, decode_latents, vae, lq_latents
                )
                decode_time_batch += decode_time
                decoded_gt_batch, decode_time = _timed_call(
                    model_cfg.device, decode_latents, vae, gt_latents
                )
                decode_time_batch += decode_time

            total_decode_time += decode_time_batch

            if need_pixel_metrics and decoded_pred_batch is not None:
                pred_pixels = _tensor_to_image_range(decoded_pred_batch.to(dtype=torch.float32))
                gt_pixels_range = _tensor_to_image_range(gt_pixels.to(dtype=torch.float32))
            elif need_pixel_metrics and not pixel_metric_warning_issued:
                print("[warn] Unable to compute pixel metrics because decoded predictions are unavailable.")
                pixel_metric_warning_issued = True

            batch_metrics = compute_metrics(
                pred_latents=pred_latents.to(dtype=torch.float32),
                gt_latents=gt_latents.to(dtype=torch.float32),
                pred_pixels=pred_pixels,
                gt_pixels=gt_pixels_range,
                metrics_cfg=metrics_cfg,
            )
            batch_metrics_cpu = {name: values.detach().cpu() for name, values in batch_metrics.items()}

            _sync_if_cuda(model_cfg.device)
            batch_time = time.perf_counter() - batch_start
            total_batch_time += batch_time

            batch_size = lq_pixels.shape[0]
            total_samples += batch_size
            batch_peak_bytes = 0
            batch_memory_per_image_mb: Optional[float] = None
            if using_cuda:
                batch_peak_bytes = _max_memory_allocated(model_cfg.device)
                if batch_peak_bytes > max_peak_bytes:
                    max_peak_bytes = batch_peak_bytes
                if batch_size > 0:
                    memory_per_image_accum += batch_peak_bytes / batch_size
                    memory_samples += 1
                    batch_memory_per_image_mb = (batch_peak_bytes / batch_size) / (1024**2)

            for name, values_cpu in batch_metrics_cpu.items():
                metric_sums[name] += values_cpu.sum().item()

            if output_cfg.per_image_path:
                time_per_image = batch_time / batch_size if batch_size > 0 else 0.0
                per_image_values = {name: values_cpu.tolist() for name, values_cpu in batch_metrics_cpu.items()}
                for idx in range(batch_size):
                    record = {"path": paths[idx]}
                    for name, values_list in per_image_values.items():
                        record[name] = float(values_list[idx])
                    record["time_per_image_sec"] = time_per_image
                    if batch_memory_per_image_mb is not None:
                        record["cuda_memory_per_image_mb"] = batch_memory_per_image_mb
                    per_image_records.append(record)

            if visualization_enabled and visualizations_saved < output_cfg.visualization_limit:
                for sample_idx in range(batch_size):
                    if visualizations_saved >= output_cfg.visualization_limit:
                        break
                    rel_path = _relative_output_path(paths[sample_idx], dataset_cfg.root)
                    save_path = output_cfg.visualization_dir / rel_path.with_suffix(".png")
                    decoded_lq_item = (
                        decoded_lq_batch[sample_idx] if decoded_lq_batch is not None else None
                    )
                    decoded_pred_item = (
                        decoded_pred_batch[sample_idx] if decoded_pred_batch is not None else None
                    )
                    decoded_gt_item = (
                        decoded_gt_batch[sample_idx] if decoded_gt_batch is not None else None
                    )
                    _save_visualization_figure(
                        save_path=save_path,
                        image_name=str(rel_path),
                        lq_latent=lq_latents[sample_idx],
                        pred_latent=pred_latents[sample_idx],
                        gt_latent=gt_latents[sample_idx],
                        decoded_lq=decoded_lq_item,
                        decoded_pred=decoded_pred_item,
                        decoded_gt=decoded_gt_item,
                        channels=output_cfg.visualization_channels,
                    )
                    visualizations_saved += 1

            if batch_metrics_cpu:
                running = {name: metric_sums[name] / total_samples for name in batch_metrics_cpu}
                progress.set_postfix({key: f"{value:.4f}" for key, value in running.items()})

    averages = {name: value / total_samples for name, value in metric_sums.items()}
    runtime_summary: Dict[str, float] = {}
    if total_samples > 0:
        runtime_summary = {
            "total_runtime_sec": total_batch_time,
            "avg_time_per_image_sec": total_batch_time / total_samples if total_samples else 0.0,
            "avg_encode_time_per_image_sec": total_encode_time / total_samples if total_samples else 0.0,
            "avg_model_time_per_image_sec": total_model_time / total_samples if total_samples else 0.0,
            "avg_decode_time_per_image_sec": total_decode_time / total_samples if total_samples else 0.0,
        }
        if total_batch_time > 0:
            runtime_summary["images_per_second"] = total_samples / total_batch_time
        if using_cuda and memory_samples > 0:
            avg_memory_per_image_bytes = memory_per_image_accum / memory_samples
            runtime_summary["avg_cuda_memory_per_image_mb"] = avg_memory_per_image_bytes / (1024**2)
            runtime_summary["max_cuda_memory_mb"] = max_peak_bytes / (1024**2)

    print("\nValidation complete.")
    for name, value in averages.items():
        print(f"{name}: {value:.6f}")
    print(f"num_samples: {total_samples}")
    if runtime_summary:
        print("-- Inference metrics --")
        for name, value in runtime_summary.items():
            if "memory" in name:
                print(f"{name}: {value:.2f}")
            else:
                print(f"{name}: {value:.6f}")

    if output_cfg.metrics_path:
        output_cfg.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"num_samples": total_samples, "metrics": averages}
        if runtime_summary:
            payload["runtime"] = runtime_summary
        with output_cfg.metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"[info] Wrote aggregated metrics to {output_cfg.metrics_path}")

    if output_cfg.per_image_path and per_image_records:
        output_cfg.per_image_path.parent.mkdir(parents=True, exist_ok=True)
        with output_cfg.per_image_path.open("w", encoding="utf-8") as handle:
            json.dump(per_image_records, handle, indent=2)
        print(f"[info] Wrote per-image metrics to {output_cfg.per_image_path}")
    if visualization_enabled:
        print(
            f"[info] Saved {visualizations_saved} visualization figure(s) to {output_cfg.visualization_dir}"
        )


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
