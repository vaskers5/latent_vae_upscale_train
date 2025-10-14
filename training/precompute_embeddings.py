"""Distributed latent embedding precomputation using Accelerate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from accelerate import Accelerator
from diffusers import (
    AsymmetricAutoencoderKL,
    AutoencoderKL,
    AutoencoderKLQwenImage,
    AutoencoderKLWan,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config_loader import load_config
from .embedding_io import TransformParams, save_record
from .image_folder_dataset import ImageFolderDataset

__all__ = ["main", "parse_args"]


@dataclass(frozen=True)
class ResolutionTask:
    resolution: int
    batch_size: int


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
    params = [item["params"] for item in batch]
    paths = [item["path"] for item in batch]
    return {
        "model_input": model_inputs,
        "params": params,
        "paths": paths,
    }


def _prepare_output_path(
    image_path: Path,
    dataset_resolution_dir: Path,
    embeddings_resolution_dir: Path,
) -> Path:
    image_path = image_path.resolve()
    dataset_resolution_dir = dataset_resolution_dir.resolve()
    dataset_str = str(dataset_resolution_dir)
    image_str = str(image_path)
    if image_str.startswith(dataset_str):
        suffix = image_str[len(dataset_str) :].lstrip("/\\")
        relative = Path(suffix) if suffix else Path(image_path.name)
    else:
        relative = Path(image_path.name)
    return (embeddings_resolution_dir / relative).with_suffix(".pt")


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


def _gather_models(config: Dict[str, Any]) -> Tuple[Path, Path, Dict[str, Any], List[ModelConfig]]:
    dataset_root = Path(config["dataset_root"]).expanduser().resolve()
    cache_root = Path(config["cache_root"]).expanduser().resolve()
    defaults = config.get("defaults", {})
    models = _parse_model_configs(config.get("models", {}))
    return dataset_root, cache_root, defaults, models


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
    progress = None
    if accelerator.is_main_process:
        progress = tqdm(total=total_items, unit="img", desc=f"{resolution}px")
    dataset_meta = _dataset_info(resolution, defaults)

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["model_input"].to(device=device, dtype=encode_dtype)
            encoding = vae.encode(inputs)
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
                params: TransformParams = batch["params"][idx]
                record_path = _prepare_output_path(image_path, dataset_dir, embeddings_dir)
                mean_tensor = latent_mean[idx] if store_distribution else None
                logvar_tensor = latent_logvar[idx] if latent_logvar is not None else None
                save_record(
                    record_path,
                    latents=latents[idx],
                    params=params,
                    dataset_info=dataset_meta,
                    mean=mean_tensor,
                    logvar=logvar_tensor,
                )

            if progress is not None:
                local_count = torch.tensor(len(batch["paths"]), device=device, dtype=torch.long)
                completed = accelerator.reduce(local_count, reduction="sum")
                progress.update(int(completed.item()))

    if progress is not None:
        progress.close()
    accelerator.wait_for_everyone()


def run_precompute(args: argparse.Namespace) -> None:
    config = load_config([args.config])
    dataset_root, cache_root, defaults, models = _gather_models(config)

    if args.dataset_root:
        dataset_root = Path(args.dataset_root).expanduser().resolve()
    if args.cache_root:
        cache_root = Path(args.cache_root).expanduser().resolve()

    num_workers = int(args.num_workers or defaults.get("num_workers", 0))
    embeddings_dtype = _parse_dtype(defaults.get("embeddings_dtype"), default=torch.float32)
    store_distribution = bool(defaults.get("store_distribution", False))
    accelerator = Accelerator()

    if accelerator.is_main_process:
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

        for task in model_cfg.tasks:
            resolution_dir = dataset_root / f"{task.resolution}px"
            if not resolution_dir.exists():
                if accelerator.is_main_process:
                    print(f"[Embeddings] Skipping {task.resolution}px (missing directory: {resolution_dir})")
                continue
            embeddings_dir = cache_root / model_cfg.cache_subdir / f"{task.resolution}px"
            embeddings_dir.mkdir(parents=True, exist_ok=True)

            dataset = ImageFolderDataset(
                resolution_dir,
                high_resolution=task.resolution,
                resize_long_side=int(defaults.get("resize_long_side", 0) or 0),
                limit=0,
                embedding_cache=None,
                model_resolution=task.resolution,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=int(args.batch_size or task.batch_size),
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                collate_fn=_collate_batch,
            )
            dataloader = accelerator.prepare(dataloader)

            if accelerator.is_main_process:
                effective_bs = getattr(dataloader, "batch_size", task.batch_size)
                print(
                    f"[Embeddings] Encoding {len(dataset)} images at {task.resolution}px with batch size {effective_bs}."
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
    parser.add_argument("--dataset-root", type=Path, default=None, help="Override dataset root")
    parser.add_argument("--cache-root", type=Path, default=None, help="Override embeddings cache root")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers")
    parser.add_argument("--batch-size", type=int, default=None, help="Override encoding batch size")
    return parser.parse_args(args)


def main(cli_args: Optional[Iterable[str]] = None) -> None:
    args = parse_args(cli_args)
    run_precompute(args)


if __name__ == "__main__":  # pragma: no cover
    main()
