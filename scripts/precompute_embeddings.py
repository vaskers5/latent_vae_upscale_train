"""CLI utility to precompute VAE embeddings for a dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List

import torch
from diffusers import (
    AsymmetricAutoencoderKL,
    AutoencoderKL,
    AutoencoderKLQwenImage,
    AutoencoderKLWan,
)

from training.config import TrainingConfig
from training.config_loader import load_config
from training.dataset import ImageFolderDataset
from training.embeddings import EmbeddingCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute latent embeddings for a dataset")
    parser.add_argument("--config", type=Path, required=True, help="Path to the training YAML config")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override dataset root directory defined in the config",
    )
    parser.add_argument(
        "--cache-subdir",
        type=str,
        default="cache_embeddings",
        help="Subdirectory (relative to the dataset root) where embeddings are stored",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to use (defaults to cuda if available, otherwise cpu)",
    )
    parser.add_argument(
        "--variants-per-sample",
        type=int,
        default=1,
        help="Number of cached variants to generate per image",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing cached embeddings before recomputing",
    )
    return parser.parse_args()


def _load_vae(config: TrainingConfig) -> torch.nn.Module:
    model_cfg = config.model
    load_path = Path(model_cfg.load_from) if model_cfg.load_from else None
    path_exists = load_path is not None and load_path.exists()

    hf_source = model_cfg.hf_repo if model_cfg.hf_repo else None
    if not path_exists and model_cfg.load_from and not model_cfg.hf_repo:
        hf_source = model_cfg.load_from

    kind = (model_cfg.vae_kind or "").strip().lower()

    if kind == "qwen":
        if path_exists:
            source = str(load_path)
            kwargs = {}
        else:
            source = hf_source or "Qwen/Qwen-Image"
            kwargs = {}
            if model_cfg.hf_subfolder or not hf_source:
                kwargs["subfolder"] = model_cfg.hf_subfolder or "vae"
            if model_cfg.hf_revision:
                kwargs["revision"] = model_cfg.hf_revision
            if model_cfg.hf_auth_token:
                kwargs["use_auth_token"] = model_cfg.hf_auth_token
        vae = AutoencoderKLQwenImage.from_pretrained(source, **kwargs)
    else:
        if path_exists:
            source = str(load_path)
            kwargs = {}
        else:
            source = hf_source or config.paths.project
            kwargs = {}
            if model_cfg.hf_subfolder:
                kwargs["subfolder"] = model_cfg.hf_subfolder
            if model_cfg.hf_revision:
                kwargs["revision"] = model_cfg.hf_revision
            if model_cfg.hf_auth_token:
                kwargs["use_auth_token"] = model_cfg.hf_auth_token
        if kind == "wan":
            vae = AutoencoderKLWan.from_pretrained(source, **kwargs)
        else:
            if kind in {"kl", "autoencoderkl", "autoencoder_kl"}:
                vae = AutoencoderKL.from_pretrained(source, **kwargs)
            elif kind in {"asymmetric_kl", "kl_asymmetric", "kl_asym", "asym_kl"}:
                vae = AsymmetricAutoencoderKL.from_pretrained(source, **kwargs)
            else:
                if config.dataset.model_resolution == config.dataset.high_resolution:
                    vae = AutoencoderKL.from_pretrained(source, **kwargs)
                else:
                    vae = AsymmetricAutoencoderKL.from_pretrained(source, **kwargs)

    display_source = source if not path_exists else str(load_path)
    print(f"[Embeddings] Loading VAE from: {display_source}")
    return vae.to(model_cfg.weights_dtype)


def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_expected_counts(cache: EmbeddingCache, image_paths: Iterable[Path], expected_variants: int) -> None:
    paths = list(image_paths)
    missing: List[Path] = []
    for path in paths:
        variants = cache.available_variants(path)
        if len(variants) != expected_variants:
            missing.append(path)

    if missing:
        raise RuntimeError(
            f"Embedding cache is incomplete. Expected {expected_variants} variant(s) per image but "
            f"{len(missing)} file(s) are missing."
        )

    expected_total = len(paths) * expected_variants
    produced = len([p for p in cache.cache_root.rglob("*.pt") if p.is_file()])
    if produced != expected_total:
        raise RuntimeError(
            f"Embedding count mismatch: expected {expected_total} cache files, found {produced}."
        )


def main() -> None:
    args = parse_args()

    raw_config = load_config([args.config])
    dataset_root = args.dataset_root or raw_config.get("ds_path")
    if not dataset_root:
        raise RuntimeError("Dataset root must be provided via --dataset-root or ds_path in the config")
    dataset_root = Path(dataset_root).expanduser().resolve()
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root {dataset_root} does not exist or is not a directory")

    cache_dir = dataset_root / args.cache_subdir
    if args.overwrite and cache_dir.exists():
        print(f"[Embeddings] Removing existing cache directory: {cache_dir}")
        shutil.rmtree(cache_dir)

    raw_config["ds_path"] = str(dataset_root)
    embeddings_section = raw_config.setdefault("embeddings", {})
    embeddings_section["enabled"] = True
    embeddings_section["cache_dir"] = str(cache_dir)
    embeddings_section["variants_per_sample"] = max(1, args.variants_per_sample)
    embeddings_section.setdefault("store_distribution", True)
    if args.overwrite:
        embeddings_section["overwrite"] = True

    config = TrainingConfig.from_dict(raw_config)

    if config.embeddings.variants_per_sample != args.variants_per_sample:
        print(
            f"[Embeddings] Adjusted variants_per_sample from {args.variants_per_sample} to "
            f"{config.embeddings.variants_per_sample} based on configuration",
        )

    cache = EmbeddingCache(config.embeddings, config.dataset, config.paths.dataset_root)

    dataset = ImageFolderDataset(
        root=config.paths.dataset_root,
        high_resolution=config.dataset.high_resolution,
        resize_long_side=config.dataset.resize_long_side,
        limit=config.dataset.limit,
        embedding_cache=None,
        model_resolution=config.dataset.model_resolution,
    )

    vae = _load_vae(config)
    device = _resolve_device(args.device)
    vae = vae.to(device)
    vae.eval()

    cache.ensure_populated(
        dataset,
        vae,
        device=device,
        encode_dtype=next(vae.parameters()).dtype,
        seed=config.seed,
        accelerator=None,
    )

    _ensure_expected_counts(cache, dataset.paths, config.embeddings.variants_per_sample)
    try:
        cache_display = cache.cache_root.relative_to(dataset_root)
    except ValueError:
        cache_display = cache.cache_root
    print(
        f"[Embeddings] Completed precomputation for {len(dataset)} image(s) "
        f"into {cache_display}"
    )


if __name__ == "__main__":
    main()

