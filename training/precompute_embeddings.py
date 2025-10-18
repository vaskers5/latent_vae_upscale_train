"""Distributed latent embedding precomputation using Accelerate."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from accelerate import Accelerator
from diffusers import (
    AsymmetricAutoencoderKL,
    AutoencoderKL,
    AutoencoderKLQwenImage,
    AutoencoderKLWan,
)
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from PIL import Image

from .config_loader import load_config
from .embedding_io import save_record
from .image_folder_dataset import ImageFolderDataset

__all__ = ["main", "parse_args"]


@dataclass(frozen=True)
class ResolutionTask:
    resolution: int
    batch_size: int


class SimpleEmbeddingsDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings_dir: Path, img_dir: Path) -> None:
        self.embeddings_dir = embeddings_dir
        self.images = [os.path.join(img_dir, p) for p in os.listdir(img_dir) if p.split(".")[-1] in {"png", "jpg", "jpeg", "webp"}]
        self.embeddings_paths = [embeddings_dir / (Path(p).name.rsplit(".", 1)[0] + ".pt") for p in self.images]
        if not self.embeddings_paths:
            raise RuntimeError(f"No embedding files found under '{self.embeddings_dir}'")

    def __len__(self) -> int:
        return len(self.embeddings_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        embed_path = self.embeddings_paths[index]
        img_path = self.images[index]
        try:
            torch.load(embed_path)
            return ""
        except Exception:
            try: 
                img = Image.open(img_path)
                img = img.convert("RGB")
                img.close()
                return img_path
            except Exception:
                return ""


@dataclass(frozen=True)
class ModelConfig:
    name: str
    cache_subdir: str
    hf_repo: Optional[str]
    vae_kind: str
    hf_subfolder: Optional[str]
    hf_revision: Optional[str]
    hf_auth_token: Optional[str]
    load_from: Optional[Path]
    weights_dtype: Optional[torch.dtype]
    tasks: List[ResolutionTask]


DeviceSpec = Union[str, int]


def _parse_dtype(value: Any, *, default: Optional[torch.dtype]) -> Optional[torch.dtype]:
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        name = value.strip().lower().replace(" ", "")
        if not name.startswith("torch."):
            name = f"torch.{name}"
        try:
            return getattr(torch, name.split(".")[-1])
        except AttributeError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported dtype specification: {value}") from exc
    raise TypeError(f"Expected dtype string or torch.dtype, received: {value!r}")


def _collate_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    model_inputs = torch.stack([item["model_input"] for item in batch])
    paths = [item["path"] for item in batch]
    return {
        "model_input": model_inputs,
        "paths": paths,
    }


def _prepare_output_path(
    image_path: Path,
    dataset_dir: Path,
    embeddings_resolution_dir: Path,
) -> Path:
    try:
        relative_path = Path(image_path).resolve().relative_to(dataset_dir.resolve())
    except ValueError:
        relative_path = Path(image_path).name
    return (embeddings_resolution_dir / relative_path).with_suffix(".pt")


def _build_dataloader(
    dataset: torch.utils.data.Dataset,
    *,
    accelerator: Accelerator,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    collate_fn: Callable[[Sequence[Dict[str, Any]]], Dict[str, Any]],
) -> DataLoader:
    # Let Accelerate handle sharding. Do NOT attach a DistributedSampler here.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
    )
    return dataloader

def _load_vae(model_cfg: ModelConfig) -> torch.nn.Module:
    source: Optional[str]
    kwargs: Dict[str, Any] = {}
    load_path = model_cfg.load_from
    if load_path is not None and load_path.exists():
        source = str(load_path)
    else:
        source = model_cfg.hf_repo
        if source is None:
            raise RuntimeError(
                f"Model '{model_cfg.name}' must define either 'hf_repo' or 'load_from'."
            )
        if model_cfg.hf_subfolder:
            kwargs["subfolder"] = model_cfg.hf_subfolder
        if model_cfg.hf_revision:
            kwargs["revision"] = model_cfg.hf_revision
        if model_cfg.hf_auth_token:
            kwargs["use_auth_token"] = model_cfg.hf_auth_token
    if model_cfg.weights_dtype is not None:
        kwargs["torch_dtype"] = model_cfg.weights_dtype

    kind = model_cfg.vae_kind.lower()
    if kind == "qwen":
        vae = AutoencoderKLQwenImage.from_pretrained(source, **kwargs)
    elif kind == "wan":
        vae = AutoencoderKLWan.from_pretrained(source, **kwargs)
    elif kind in {"kl", "autoencoderkl", "autoencoder_kl"}:
        vae = AutoencoderKL.from_pretrained(source, **kwargs)
    else:
        vae = AsymmetricAutoencoderKL.from_pretrained(source, **kwargs)
    return vae


def _parse_model_configs(raw: Dict[str, Any]) -> List[ModelConfig]:
    models: List[ModelConfig] = []
    for name, cfg in raw.items():
        load_from = Path(cfg["load_from"]).expanduser() if cfg.get("load_from") else None
        weights_dtype = _parse_dtype(cfg.get("weights_dtype"), default=None)
        tasks = [
            ResolutionTask(resolution=int(pair[0]), batch_size=int(pair[1]))
            for pair in cfg.get("resolutions_with_batchsize", [])
        ]
        if not tasks:
            raise ValueError(f"Model '{name}' must define 'resolutions_with_batchsize'.")
        models.append(
            ModelConfig(
                name=name,
                cache_subdir=str(cfg.get("cache_subdir", name)),
                hf_repo=cfg.get("hf_repo"),
                vae_kind=str(cfg.get("vae_kind", "kl")).lower(),
                hf_subfolder=cfg.get("hf_subfolder"),
                hf_revision=cfg.get("hf_revision"),
                hf_auth_token=cfg.get("hf_auth_token"),
                load_from=load_from,
                weights_dtype=weights_dtype,
                tasks=tasks,
            )
        )
    return models


def _parse_dataset_pairs(config: Dict[str, Any]) -> List[Tuple[Path, Path]]:
    dataset_pairs_cfg = config.get("datasets") or config.get("dataset_pairs")
    dataset_pairs: List[Tuple[Path, Path]] = []
    if dataset_pairs_cfg is not None:
        if not isinstance(dataset_pairs_cfg, Sequence):
            raise TypeError("The 'datasets' configuration must be a sequence of [images, cache] pairs.")
        for idx, pair in enumerate(dataset_pairs_cfg):
            if (
                not isinstance(pair, Sequence)
                or isinstance(pair, (str, bytes))
                or len(pair) != 2
            ):
                raise ValueError(
                    "Each dataset entry must be a two-item sequence: [images_folder, cache_folder]."
                )
            dataset_path, cache_path = pair
            dataset_pairs.append(
                (
                    Path(dataset_path).expanduser().resolve(),
                    Path(cache_path).expanduser().resolve(),
                )
            )
    else:
        if "dataset_root" not in config or "cache_root" not in config:
            raise ValueError(
                "Configuration must define either 'dataset_root' and 'cache_root', or the 'datasets' list."
            )
        dataset_pairs.append(
            (
                Path(config["dataset_root"]).expanduser().resolve(),
                Path(config["cache_root"]).expanduser().resolve(),
            )
        )
    return dataset_pairs


def _gather_models(config: Dict[str, Any]) -> Tuple[List[Tuple[Path, Path]], Dict[str, Any], List[ModelConfig]]:
    dataset_pairs = _parse_dataset_pairs(config)
    defaults = config.get("defaults", {})
    models = _parse_model_configs(config.get("models", {}))
    return dataset_pairs, defaults, models


def _normalize_cuda_device(device: DeviceSpec) -> Optional[str]:
    if device is None:
        return None
    if isinstance(device, int):
        if device < 0:
            raise ValueError(f"Negative CUDA device index: {device}")
        return str(device)
    if isinstance(device, str):
        value = device.strip()
        if not value:
            return None
        lower = value.lower()
        if lower in {"auto", "all", "cpu"}:
            return None
        if "," in value:
            raise ValueError("Comma-separated device specifications must be split beforehand.")
        for prefix in ("cuda", "gpu"):
            if lower.startswith(prefix):
                remainder = value[len(prefix) :]
                if remainder.startswith(":"):
                    remainder = remainder[1:]
                value = remainder
                break
        value = value.strip()
        if not value:
            raise ValueError(f"Empty CUDA device specification: {device!r}")
        if not value.isdigit():
            raise ValueError(f"Unsupported CUDA device specification: {device!r}")
        return value
    raise TypeError(f"Unsupported device specification type: {type(device)!r}")


def _configure_cuda_devices(devices: Optional[Sequence[DeviceSpec]]) -> Optional[List[str]]:
    if not devices:
        return None
    normalized: List[str] = []
    for spec in devices:
        if isinstance(spec, str) and "," in spec:
            parts = [part.strip() for part in spec.split(",") if part.strip()]
        else:
            parts = [spec]
        for part in parts:
            normalized_device = _normalize_cuda_device(part)
            if normalized_device is not None:
                normalized.append(normalized_device)
    if not normalized:
        return None
    unique_devices = list(dict.fromkeys(normalized))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(unique_devices)
    return unique_devices


def _dataset_info(resolution: int, defaults: Dict[str, Any]) -> Dict[str, Any]:
    resize_long_side = int(defaults.get("resize_long_side", 0) or 0)
    return {
        "high_resolution": int(resolution),
        "model_resolution": int(resolution),
        "resize_long_side": resize_long_side,
    }


def _encode_resolution(
    *,
    accelerator: Accelerator,
    vae: torch.nn.Module,
    dataloader: DataLoader,
    dataset_dir: Path,
    embeddings_dir: Path,
    resolution: int,
    total_items: int,
    embeddings_dtype: torch.dtype,
    encode_dtype: torch.dtype,
    store_distribution: bool,
    defaults: Dict[str, Any],
) -> None:
    device = accelerator.device
    vae.eval()
    vae_module = accelerator.unwrap_model(vae)
    progress = None
    if accelerator.is_main_process:
        progress = tqdm(total=total_items, unit="img", desc=f"{resolution}px")
    dataset_meta = _dataset_info(resolution, defaults)

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["model_input"].to(device=device, dtype=encode_dtype)
            encoding = vae_module.encode(inputs)
            
            latent_mean = encoding.latent_dist.mean.detach()
            latent_logvar = (
                encoding.latent_dist.logvar.detach() if store_distribution else None
            )

            if latent_mean.ndim == 5:
                latent_mean = latent_mean.squeeze(2)
                if latent_logvar is not None:
                    latent_logvar = latent_logvar.squeeze(2)
            latents = latent_mean.to(embeddings_dtype)
            for idx, image_path in enumerate(batch["paths"]):
                record_path = _prepare_output_path(image_path, dataset_dir, embeddings_dir)
                record_path.parent.mkdir(parents=True, exist_ok=True)
                mean_tensor = latent_mean[idx] if store_distribution else None
                logvar_tensor = latent_logvar[idx] if latent_logvar is not None else None
                save_record(
                    record_path,
                    latents=latents[idx],
                    dataset_info=dataset_meta,
                    mean=mean_tensor,
                    logvar=logvar_tensor,
                )

            local_count = torch.tensor(len(batch["paths"]), device=device, dtype=torch.long)
            completed = accelerator.reduce(local_count, reduction="sum")
            if progress is not None and accelerator.is_main_process:
                progress.update(int(completed.item()))

    if progress is not None:
        progress.close()
    accelerator.wait_for_everyone()


def run_precompute(args: argparse.Namespace) -> None:
    config = load_config([args.config])
    dataset_pairs, defaults, models = _gather_models(config)
    num_workers = defaults.get("num_workers", 0)
    embeddings_dtype = _parse_dtype(defaults.get("embeddings_dtype"), default=torch.float32)
    store_distribution = bool(defaults.get("store_distribution", False))
    default_devices = defaults.get("devices")
    if isinstance(default_devices, (list, tuple)):
        device_specs = list(default_devices)
    elif default_devices is None:
        device_specs = None
    else:
        device_specs = [default_devices]
    selected_devices = _configure_cuda_devices(device_specs)
    accelerator = Accelerator()

    if accelerator.is_main_process:
        if selected_devices:
            formatted_devices = ", ".join(f"cuda:{dev}" for dev in selected_devices)
            print(
                f"[Embeddings] Using CUDA visible devices: {formatted_devices}"
            )
        for _, cache_root in dataset_pairs:
            cache_root.mkdir(parents=True, exist_ok=True)
        print(
            f"[Embeddings] Starting precompute with {accelerator.num_processes} process(es) on {accelerator.device}."
        )

    for model_cfg in models:
        if accelerator.is_main_process:
            print(f"[Embeddings] Loading VAE '{model_cfg.name}' from {model_cfg.hf_repo or model_cfg.load_from}...")
        vae = _load_vae(model_cfg)
        vae = accelerator.prepare(vae)
        encode_dtype = next(accelerator.unwrap_model(vae).parameters()).dtype

        for dataset_root, cache_root in dataset_pairs:
            if accelerator.is_main_process:
                print(
                    f"[Embeddings] Processing dataset '{dataset_root}' with cache '{cache_root}'."
                )

            for task in model_cfg.tasks:
                resolution_dir = dataset_root / f"{task.resolution}px"
                if not resolution_dir.exists():
                    if accelerator.is_main_process:
                        print(f"[Embeddings] Skipping {task.resolution}px (missing directory: {resolution_dir})")
                    continue
                embeddings_dir = cache_root / model_cfg.cache_subdir / f"{task.resolution}px"
                embeddings_dir.mkdir(parents=True, exist_ok=True)

                images_to_process: List[str] = []
                for img_batch in tqdm(
                    torch.utils.data.DataLoader(
                        SimpleEmbeddingsDataset(embeddings_dir, resolution_dir),
                        batch_size=task.batch_size,
                        num_workers=num_workers,
                    )
                ):
                    for img in img_batch:
                        if img != "":
                            images_to_process.append(img)

                dataset = ImageFolderDataset(
                    paths=images_to_process,
                )
                per_device_batch_size = task.batch_size
                dataloader = _build_dataloader(
                    dataset,
                    accelerator=accelerator,
                    batch_size=per_device_batch_size,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available(),
                    collate_fn=_collate_batch,
                )
                dataloader = accelerator.prepare(dataloader)

                if accelerator.is_main_process:
                    global_bs = per_device_batch_size * accelerator.num_processes
                    print(
                        f"[Embeddings] Encoding {len(dataset)} images at {task.resolution}px with per-device batch size {per_device_batch_size} (global {global_bs})."
                    )

                _encode_resolution(
                    accelerator=accelerator,
                    vae=vae,
                    dataloader=dataloader,
                    dataset_dir=resolution_dir,
                    embeddings_dir=embeddings_dir,
                    resolution=task.resolution,
                    total_items=len(dataset),
                    embeddings_dtype=embeddings_dtype,
                    encode_dtype=encode_dtype,
                    store_distribution=store_distribution,
                    defaults=defaults,
                )

        accelerator.wait_for_everyone()
        del vae
        torch.cuda.empty_cache()

    if accelerator.is_main_process:
        print("[Embeddings] Precomputation complete.")


def parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute latent embeddings with Accelerate")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config file")
    return parser.parse_args(args)


def main(cli_args: Optional[Iterable[str]] = None) -> None:
    args = parse_args(cli_args)
    run_precompute(args)


if __name__ == "__main__":  # pragma: no cover
    main()
