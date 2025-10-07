"""CLI utility to precompute VAE embeddings for multiple VAE backends."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable, List

import torch
from diffusers import (
    AsymmetricAutoencoderKL,
    AutoencoderKL,
    AutoencoderKLQwenImage,
    AutoencoderKLWan,
)

from .config import DatasetConfig, EmbeddingsConfig
from .config_loader import load_config
from .dataset import ImageFolderDataset
from .embeddings import EmbeddingCache
from .multi_precompute_config import MultiPrecomputeConfig, MultiPrecomputeTask

__all__ = ["main", "parse_args"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute latent embeddings for one or more VAEs")
    parser.add_argument("--config", type=Path, required=True, help="Path to the multi-VAE YAML config")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override dataset root directory defined in the config",
    )
    parser.add_argument(
        "--cache-subdir",
        type=str,
        default=None,
        help=(
            "Subdirectory or path for cached embeddings."
            " If relative it is resolved inside the dataset root."
        ),
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
        default=None,
        help="Number of cached variants to generate per image",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size to use for VAE encoding (defaults to config value)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker count for embedding precomputation (defaults to config value)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing cached embeddings before recomputing",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tasks whose caches are already complete (validated before encoding)",
    )

    return parser.parse_args()


def _load_vae(task: MultiPrecomputeTask) -> torch.nn.Module:
    load_path = Path(task.load_from).expanduser() if task.load_from else None
    path_exists = load_path is not None and load_path.exists()

    hf_source = task.hf_repo or None
    if not path_exists and task.load_from and not task.hf_repo:
        hf_source = task.load_from

    kind = (task.vae_kind or "").strip().lower()

    if kind == "qwen":
        if path_exists:
            source = str(load_path)
            kwargs = {}
        else:
            source = hf_source or "Qwen/Qwen-Image"
            kwargs = {}
            if task.hf_subfolder or not hf_source:
                kwargs["subfolder"] = task.hf_subfolder or "vae"
            if task.hf_revision:
                kwargs["revision"] = task.hf_revision
            if task.hf_auth_token:
                kwargs["use_auth_token"] = task.hf_auth_token
        vae = AutoencoderKLQwenImage.from_pretrained(source, **kwargs)
    else:
        if path_exists:
            source = str(load_path)
            kwargs = {}
        else:
            source = hf_source
            if not source:
                raise RuntimeError(
                    f"Task '{task.vae_name}' must provide either a local 'load_from' path or an 'hf_repo'."
                )
            kwargs = {}
            if task.hf_subfolder:
                kwargs["subfolder"] = task.hf_subfolder
            if task.hf_revision:
                kwargs["revision"] = task.hf_revision
            if task.hf_auth_token:
                kwargs["use_auth_token"] = task.hf_auth_token
        if kind == "wan":
            vae = AutoencoderKLWan.from_pretrained(source, **kwargs)
        elif kind in {"kl", "autoencoderkl", "autoencoder_kl"}:
            vae = AutoencoderKL.from_pretrained(source, **kwargs)
        elif kind in {"asymmetric_kl", "kl_asymmetric", "kl_asym", "asym_kl"}:
            vae = AsymmetricAutoencoderKL.from_pretrained(source, **kwargs)
        else:
            if task.model_resolution == task.high_resolution:
                vae = AutoencoderKL.from_pretrained(source, **kwargs)
            else:
                vae = AsymmetricAutoencoderKL.from_pretrained(source, **kwargs)

    display_source = str(load_path) if path_exists else source
    print(f"[Embeddings] Loading VAE from: {display_source}")
    return vae


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


def _run_from_config(args: argparse.Namespace) -> None:
    raw_config = load_config([args.config])
    multi_cfg = MultiPrecomputeConfig.from_dict(raw_config)

    dataset_override = args.dataset_root.expanduser() if args.dataset_root else None
    cache_override = Path(args.cache_subdir).expanduser() if args.cache_subdir else None
    variants_override = int(args.variants_per_sample) if args.variants_per_sample is not None else None
    batch_override = int(args.batch_size) if args.batch_size is not None else None
    workers_override = int(args.num_workers) if args.num_workers is not None else None

    tasks = multi_cfg.generate_tasks(
        dataset_root_override=dataset_override,
        cache_override=cache_override,
        variants_override=variants_override,
        batch_override=batch_override,
        workers_override=workers_override,
    )

    if not tasks:
        raise RuntimeError("No VAE tasks were defined in the configuration file")

    device = _resolve_device(args.device or multi_cfg.defaults.device)
    print(f"[Embeddings] Prepared {len(tasks)} task(s) for multi-VAE precomputation")

    overall_start = time.perf_counter()
    for index, task in enumerate(tasks, start=1):
        print(
            f"[Embeddings] Task {index}/{len(tasks)} :: {task.vae_name} at {task.display_resolution}"
        )

        if not task.dataset_path.exists() or not task.dataset_path.is_dir():
            raise FileNotFoundError(
                f"Dataset root {task.dataset_path} for task '{task.vae_name}' does not exist or is not a directory"
            )

        dataset_cfg = DatasetConfig(
            high_resolution=task.high_resolution,
            model_resolution=task.model_resolution,
            resize_long_side=task.resize_long_side,
            limit=task.limit,
            num_workers=task.num_workers,
        )

        embeddings_cfg = EmbeddingsConfig(
            enabled=True,
            cache_dir=task.cache_dir,
            dtype=task.embeddings_dtype,
            variants_per_sample=task.variants_per_sample,
            overwrite=args.overwrite,
            precompute_batch_size=task.batch_size,
            num_workers=task.num_workers,
            store_distribution=task.store_distribution,
            vae_names=(task.vae_name,),
            vae_cache_dirs=(task.cache_dir,),
        )

        cache = EmbeddingCache(embeddings_cfg, dataset_cfg, task.dataset_path)

        dataset = ImageFolderDataset(
            root=task.dataset_path,
            high_resolution=task.high_resolution,
            resize_long_side=task.resize_long_side,
            limit=task.limit,
            embedding_cache=None,
            model_resolution=task.model_resolution,
        )

        if args.skip_existing and not args.overwrite:
            try:
                _ensure_expected_counts(cache, dataset.paths, embeddings_cfg.variants_per_sample)
            except RuntimeError:
                pass
            else:
                print(
                    f"[Embeddings] Cache already complete for {task.vae_name}"
                    f" ({task.display_resolution}); skipping."
                )
                continue

        start = time.perf_counter()
        vae = _load_vae(task)
        to_kwargs = {"device": device}
        if task.weights_dtype is not None:
            to_kwargs["dtype"] = task.weights_dtype
        vae = vae.to(**to_kwargs)
        vae.eval()

        cache.ensure_populated(
            dataset,
            vae,
            device=device,
            encode_dtype=next(vae.parameters()).dtype,
            seed=0,
            accelerator=None,
        )

        _ensure_expected_counts(cache, dataset.paths, embeddings_cfg.variants_per_sample)
        elapsed = time.perf_counter() - start

        print(
            f"[Embeddings] Finished {task.vae_name} ({task.display_resolution}) in {elapsed:.1f}s"
        )

        del vae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_elapsed = time.perf_counter() - overall_start
    print(f"[Embeddings] Multi-VAE precomputation finished in {total_elapsed:.1f}s total")


def main() -> None:
    args = parse_args()
    _run_from_config(args)


if __name__ == "__main__":
    main()
