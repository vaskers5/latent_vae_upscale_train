"""CLI utility to precompute VAE embeddings for multiple VAE backends."""

from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List, Sequence

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
        help="Torch device to use for all tasks (defaults to cuda if available, otherwise cpu)",
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of torch devices used to run tasks in parallel (e.g. cuda:0 cuda:1)",
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

def _parse_device_string(spec: str) -> torch.device:
    spec = spec.strip()
    if not spec:
        raise ValueError("Device specifications must not be empty")
    if spec.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA devices requested but torch.cuda.is_available() is False")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device.index} is out of range for {torch.cuda.device_count()} visible device(s)"
            )
    return device


def _resolve_device_list(
    *,
    single_device: str | None,
    device_list: Sequence[str] | None,
    default_device: str | None,
) -> List[torch.device]:
    if single_device and device_list:
        raise ValueError("Specify either --device or --devices, not both")

    devices: List[torch.device] = []

    if device_list:
        for item in device_list:
            parts = [part for part in item.split(",") if part.strip()]
            if not parts:
                continue
            for part in parts:
                devices.append(_parse_device_string(part))
    elif single_device:
        devices.append(_parse_device_string(single_device))
    elif default_device:
        devices.append(_parse_device_string(default_device))
    else:
        devices.append(_parse_device_string("cuda" if torch.cuda.is_available() else "cpu"))

    # Deduplicate while preserving order to avoid redundant workers.
    seen: set[str] = set()
    unique_devices: List[torch.device] = []
    for device in devices:
        key = str(device)
        if key in seen:
            continue
        seen.add(key)
        unique_devices.append(device)

    return unique_devices


def _ensure_cuda_context(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_device(device)


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


def _execute_task(
    task: MultiPrecomputeTask,
    *,
    device: torch.device,
    overwrite: bool,
    skip_existing: bool,
    index: int,
    total: int,
) -> None:
    print(
        f"[Embeddings] Task {index}/{total} :: {task.vae_name} at {task.display_resolution}"
        f" [device={device}]"
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
        overwrite=overwrite,
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

    if skip_existing and not overwrite:
        try:
            _ensure_expected_counts(cache, dataset.paths, embeddings_cfg.variants_per_sample)
        except RuntimeError:
            pass
        else:
            print(
                (
                    f"[Embeddings] Cache already complete for {task.vae_name}"
                    f" ({task.display_resolution}); skipping."
                )
            )
            return

    start = time.perf_counter()

    _ensure_cuda_context(device)
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
    if device.type == "cuda":
        _ensure_cuda_context(device)
        torch.cuda.empty_cache()


def _run_parallel_tasks(
    tasks: List[MultiPrecomputeTask],
    devices: Sequence[torch.device],
    *,
    overwrite: bool,
    skip_existing: bool,
) -> None:
    lock = threading.Lock()
    enumerator = iter(enumerate(tasks, start=1))
    total = len(tasks)

    def worker(device: torch.device) -> None:
        _ensure_cuda_context(device)
        while True:
            with lock:
                try:
                    index, task = next(enumerator)
                except StopIteration:
                    return
            _execute_task(
                task,
                device=device,
                overwrite=overwrite,
                skip_existing=skip_existing,
                index=index,
                total=total,
            )

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(worker, device) for device in devices]
        for future in futures:
            # Propagate any exception raised inside the worker threads.
            future.result()


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

    devices = _resolve_device_list(
        single_device=args.device,
        device_list=args.devices,
        default_device=multi_cfg.defaults.device,
    )

    print(
        f"[Embeddings] Prepared {len(tasks)} task(s) for multi-VAE precomputation"
        f" using {len(devices)} device(s)"
    )

    overall_start = time.perf_counter()

    if len(devices) == 1:
        device = devices[0]
        for index, task in enumerate(tasks, start=1):
            _execute_task(
                task,
                device=device,
                overwrite=args.overwrite,
                skip_existing=args.skip_existing,
                index=index,
                total=len(tasks),
            )
    else:
        _run_parallel_tasks(
            tasks,
            devices,
            overwrite=args.overwrite,
            skip_existing=args.skip_existing,
        )

    total_elapsed = time.perf_counter() - overall_start
    print(f"[Embeddings] Multi-VAE precomputation finished in {total_elapsed:.1f}s total")


def main() -> None:
    args = parse_args()
    _run_from_config(args)


if __name__ == "__main__":
    main()
