"""Unified embedding cache, configuration, and precompute CLI utilities."""

from __future__ import annotations

import argparse
import hashlib
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from diffusers import (
    AsymmetricAutoencoderKL,
    AutoencoderKL,
    AutoencoderKLQwenImage,
    AutoencoderKLWan,
)
from tqdm.auto import tqdm

from .config import DatasetConfig, EmbeddingsConfig, _resolve_bool, _resolve_dtype, _slugify
from .config_loader import load_config

if TYPE_CHECKING:  # pragma: no cover
    from accelerate import Accelerator
    from .dataset import ImageFolderDataset

__all__ = [
    "EmbeddingCache",
    "MultiPrecomputeConfig",
    "MultiPrecomputeTask",
    "TransformParams",
    "main",
    "parse_args",
]


# --------------------------------------------------------------------------------------
# Embedding cache utilities

_CACHE_VERSION = 1
_LOG_BATCH_FLUSH_INTERVAL = 100
_LOG_BUFFER_MAX_ENTRIES = 10_000


@dataclass(frozen=True)
class TransformParams:
    """Metadata describing how an image tensor was prepared for the model."""

    flip: bool
    crop_x: int
    crop_y: int

    def to_dict(self) -> Dict[str, Any]:
        return {"flip": self.flip, "crop_x": int(self.crop_x), "crop_y": int(self.crop_y)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformParams":
        return cls(
            flip=bool(data.get("flip", False)),
            crop_x=int(data.get("crop_x", 0)),
            crop_y=int(data.get("crop_y", 0)),
        )


@dataclass
class EmbeddingRecord:
    """Data container returned when loading cached embeddings."""

    latents: torch.Tensor
    mean: Optional[torch.Tensor]
    logvar: Optional[torch.Tensor]
    params: TransformParams
    variant_index: int
    config: Dict[str, Any]


class EmbeddingCache:
    """Manages storage and retrieval of precomputed VAE embeddings."""

    def __init__(
        self,
        embeddings_config: EmbeddingsConfig,
        dataset_config: DatasetConfig,
        dataset_root: Path,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.cfg = embeddings_config
        self.dataset_cfg = dataset_config
        self.dataset_root = dataset_root.resolve()
        self.cache_root = self.cfg.cache_dir.resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.log = log or print
        self._log_path = self._resolve_log_path()
        self._cache_index = self._load_cache_log()
        self._pending_log_entries: List[Path] = []
        self._log_cleared_for_overwrite = False
        self.log(
            f"Initialized EmbeddingCache with cache_root: {self.cache_root} "
            f"(log: {self._log_path})"
        )

    # ------------------------------------------------------------------ storage helpers
    def _relative_path(self, image_path: Path) -> Path:
        try:
            return image_path.resolve().relative_to(self.dataset_root)
        except ValueError:
            digest = hashlib.sha256(str(image_path).encode("utf-8")).hexdigest()[:16]
            safe_name = image_path.stem + f"_{digest}"
            return Path("__external__") / safe_name

    def _record_path(self, image_path: Path) -> Path:
        relative = self._relative_path(image_path)
        if relative.suffix:
            record_relative = relative.with_suffix(".pt")
        else:
            record_relative = relative.parent / f"{relative.name}.pt"
        return (self.cache_root / record_relative).resolve()

    @staticmethod
    def _sanitize_fragment(text: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text.strip())
        return cleaned or "cache"

    def _resolve_log_path(self) -> Path:
        parent = self.cache_root.parent if self.cache_root.parent != self.cache_root else self.cache_root
        slug_candidate = next(iter(self.cfg.vae_names), parent.name or self.cache_root.name)
        resolution = self.cache_root.name or "latents"
        filename = (
            f"{self._sanitize_fragment(slug_candidate)}_"
            f"{self._sanitize_fragment(resolution)}_embed_cache_log"
        )
        return parent / filename

    def _load_cache_log(self) -> Set[Path]:
        index: Set[Path] = set()
        if not hasattr(self, "_log_path") or not self._log_path.exists():
            return index
        try:
            with self._log_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    index.add(Path(line))
        except OSError as exc:
            self.log(f"[Embeddings] Warning: failed to read cache log '{self._log_path}': {exc}")
        return index

    def _register_cached_path(self, image_path: Path) -> None:
        relative = self._relative_path(image_path)
        if relative in self._cache_index:
            return
        self._cache_index.add(relative)
        self._pending_log_entries.append(relative)
        if len(self._pending_log_entries) >= _LOG_BUFFER_MAX_ENTRIES:
            self._flush_log_buffer()

    def _flush_log_buffer(self) -> None:
        if not self._pending_log_entries:
            return
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._log_path.open("a", encoding="utf-8") as handle:
                for relative in self._pending_log_entries:
                    handle.write(f"{relative.as_posix()}\n")
        except OSError as exc:
            self.log(f"[Embeddings] Warning: failed to write cache log '{self._log_path}': {exc}")
        finally:
            self._pending_log_entries.clear()

    def _prepare_for_overwrite(self) -> None:
        if not self.cfg.overwrite or self._log_cleared_for_overwrite:
            return
        if self._log_path.exists():
            try:
                self._log_path.unlink()
            except OSError as exc:
                self.log(f"[Embeddings] Warning: unable to remove cache log '{self._log_path}': {exc}")
        self._cache_index = set()
        self._pending_log_entries = []
        self._log_cleared_for_overwrite = True

    def available_variants(self, image_path: Path) -> List[int]:
        relative = self._relative_path(image_path)
        return [0] if relative in self._cache_index else []

    def has_variant(self, image_path: Path, variant_index: int) -> bool:
        if variant_index != 0:
            return False
        relative = self._relative_path(image_path)
        return relative in self._cache_index

    def save_record(
        self,
        image_path: Path,
        latents: torch.Tensor,
        params: TransformParams,
        *,
        mean: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
    ) -> None:
        record_path = self._record_path(image_path)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "version": _CACHE_VERSION,
            "latents": latents.cpu(),
            "params": params.to_dict(),
            "variant": 0,
            "dataset": {
                "high_resolution": self.dataset_cfg.high_resolution,
                "model_resolution": self.dataset_cfg.model_resolution,
                "resize_long_side": self.dataset_cfg.resize_long_side,
            },
            "storage_dtype": str(latents.dtype).replace("torch.", ""),
        }
        if mean is not None:
            payload["mean"] = mean.cpu()
        if logvar is not None:
            payload["logvar"] = logvar.cpu()
        torch.save(payload, record_path)

    def load_record(self, image_path: Path) -> EmbeddingRecord:
        self.log(f"Loading embedding for {image_path}")
        record_path = self._record_path(image_path)
        if not record_path.exists():
            raise FileNotFoundError(f"Missing embedding cache for '{image_path}'")
        payload = torch.load(record_path, map_location="cpu")
        version = int(payload.get("version", -1))
        if version != _CACHE_VERSION:
            raise RuntimeError(
                "Embedding cache version mismatch for '{record_path}'. "
                f"Expected {_CACHE_VERSION}, found {version}."
            )
        params = TransformParams.from_dict(payload.get("params", {}))
        latents = payload["latents"]
        mean = payload.get("mean")
        logvar = payload.get("logvar")
        config = payload.get("dataset", {})
        return EmbeddingRecord(
            latents=latents,
            mean=mean,
            logvar=logvar,
            params=params,
            variant_index=0,
            config=config,
        )

    def choose_record(self, image_path: Path, rng: Optional[random.Random] = None) -> EmbeddingRecord:
        if not self.has_variant(image_path, 0):
            raise FileNotFoundError(f"No cached embeddings available for '{image_path}'")
        del rng  # No variants yet; seed unused but kept for compatibility.
        self.log(f"Chose cached embedding for {image_path}")
        return self.load_record(image_path)

    def validate_dataset(self, dataset: "ImageFolderDataset") -> None:
        if not self.cfg.enabled:
            return
        missing: List[Path] = []
        for path in dataset.paths:
            if not self.has_variant(path, 0):
                missing.append(path)
        if missing:
            self.log(f"Validation failed: {len(missing)} images missing cached embeddings")
            preview = "\n".join(f"- {path}" for path in missing[:10])
            suffix = "" if len(missing) <= 10 else f"\n... and {len(missing) - 10} more"
            raise RuntimeError(
                "Embedding cache incomplete. Missing the following entries:\n"
                f"{preview}{suffix}"
            )
        self.log("Dataset validation passed: all embeddings available")

    # ------------------------------------------------------------------ population helpers
    def ensure_populated(
        self,
        dataset: "ImageFolderDataset",
        vae: nn.Module,
        *,
        device: torch.device,
        encode_dtype: torch.dtype,
        seed: int,
        accelerator: Optional["Accelerator"] = None,
        progress_position: Optional[int] = None,
        progress_desc: Optional[str] = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        if not self.cfg.enabled:
            return

        self._prepare_for_overwrite()
        dataset_lookup: Dict[Path, Path] = {
            self._relative_path(path): path for path in dataset.paths
        }
        dataset_relatives = list(dataset_lookup.keys())

        if self.cfg.overwrite:
            missing: List[Path] = list(dataset_lookup.values())
        else:
            cached_relatives = set(self._cache_index)
            existing_relatives = set(dataset_relatives).intersection(cached_relatives)
            missing_relatives = [relative for relative in dataset_relatives if relative not in existing_relatives]
            missing = [dataset_lookup[relative] for relative in missing_relatives]

        main_process = accelerator is None or accelerator.is_main_process

        if not missing:
            if main_process:
                (log or print)("[Embeddings] Cache already populated")
            return

        if not main_process:
            return

        display_desc = progress_desc or "Precomputing latents"
        logger = log or print
        logger(f"[Embeddings] Populating cache for {len(missing)} images ...")

        vae_prev_mode = vae.training
        vae = vae.to(device)
        vae.eval()

        dataset_loader = _EmbeddingGenerationDataset(dataset=dataset, pending=missing, cache=self, seed=seed)
        loader = DataLoader(
            dataset_loader,
            batch_size=self.cfg.precompute_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=_collate_embedding_batches,
        )

        progress = None
        if main_process:
            progress = tqdm(
                total=len(missing),
                desc=display_desc,
                unit="img",
                position=progress_position,
                leave=progress_position is None,
                dynamic_ncols=True,
            )

        batch_counter = 0
        try:
            with torch.no_grad():
                for batch in loader:
                    batch_counter += 1
                    model_inputs = batch["model_input"].to(device=device, dtype=encode_dtype)
                    if self._is_video_vae(vae):
                        model_inputs = model_inputs.unsqueeze(2)
                    encoding = vae.encode(model_inputs)
                    latent_mean = encoding.latent_dist.mean.detach()
                    latent_logvar = (
                        encoding.latent_dist.logvar.detach() if self.cfg.store_distribution else None
                    )

                    latents = latent_mean
                    if self._is_video_vae(vae):
                        latents = latents.squeeze(2)
                        latent_mean = latent_mean.squeeze(2)
                        if latent_logvar is not None:
                            latent_logvar = latent_logvar.squeeze(2)

                    latents = latents.to(self.cfg.dtype)

                    for idx, image_path in enumerate(batch["paths"]):
                        params = batch["params"][idx]
                        mean_tensor = latent_mean[idx] if self.cfg.store_distribution else None
                        logvar_tensor = latent_logvar[idx] if latent_logvar is not None else None
                        self.save_record(
                            image_path,
                            latents[idx],
                            params,
                            mean=mean_tensor,
                            logvar=logvar_tensor,
                        )
                        self._register_cached_path(image_path)

                    if progress is not None:
                        progress.update(len(batch["paths"]))
                    if batch_counter % _LOG_BATCH_FLUSH_INTERVAL == 0:
                        self._flush_log_buffer()
        finally:
            self._flush_log_buffer()

        if progress is not None:
            progress.close()

        print("[Embeddings] Cache population complete")
        vae.train(vae_prev_mode)

    @staticmethod
    def _is_video_vae(module: nn.Module) -> bool:
        encoder = getattr(module, "encoder", None)
        conv_in = getattr(encoder, "conv_in", None) if encoder is not None else None
        weight = getattr(conv_in, "weight", None) if conv_in is not None else None
        return isinstance(weight, torch.nn.Parameter) and weight.ndimension() == 5

    @staticmethod
    def _build_rng(seed: int, image_path: Path) -> random.Random:
        payload = f"{seed}:{image_path.as_posix()}".encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()[:16]
        seed_value = int(digest, 16) & 0xFFFFFFFF
        return random.Random(seed_value)


class _EmbeddingGenerationDataset(Dataset):
    """Dataset wrapper that yields tensors ready for VAE encoding."""

    def __init__(
        self,
        *,
        dataset: "ImageFolderDataset",
        pending: Sequence[Path],
        cache: EmbeddingCache,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.pending = list(pending)
        self.cache = cache
        self.seed = seed

    def __len__(self) -> int:
        return len(self.pending)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_path = self.pending[index]
        rng = self.cache._build_rng(self.seed, image_path)
        sample_tensor, params = self.dataset.build_tensor_sample(image_path, rng=rng, params=None)
        model_input = self.dataset.prepare_model_input(sample_tensor).contiguous()
        return {
            "path": image_path,
            "model_input": model_input,
            "params": params,
        }


def _collate_embedding_batches(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    model_inputs = torch.stack([item["model_input"] for item in batch], dim=0)
    return {
        "model_input": model_inputs,
        "params": [item["params"] for item in batch],
        "paths": [item["path"] for item in batch],
    }


# --------------------------------------------------------------------------------------
# Multi-precompute configuration helpers


@dataclass
class ResolutionSpec:
    """Container describing a single resolution precompute target."""

    high_resolution: int
    model_resolution: int
    resize_long_side: Optional[int] = None
    batch_size: Optional[int] = None

    @classmethod
    def from_raw(cls, raw: Any) -> "ResolutionSpec":
        if isinstance(raw, int):
            value = int(raw)
            return cls(high_resolution=value, model_resolution=value, resize_long_side=None)
        if isinstance(raw, (list, tuple)):
            if not raw:
                raise ValueError("Resolution list entries must include at least the resolution value")
            high = int(raw[0])
            if len(raw) == 1:
                return cls(high_resolution=high, model_resolution=high)
            if len(raw) == 2:
                return cls(high_resolution=high, model_resolution=high, batch_size=int(raw[1]))
            model = int(raw[1])
            batch = int(raw[2])
            resize = int(raw[3]) if len(raw) > 3 and raw[3] is not None else None
            return cls(
                high_resolution=high,
                model_resolution=model,
                resize_long_side=resize,
                batch_size=batch,
            )
        if isinstance(raw, dict):
            high = (
                raw.get("high_resolution")
                or raw.get("high")
                or raw.get("size")
                or raw.get("resolution")
            )
            if high is None:
                raise ValueError("Resolution entry must define 'high_resolution' or be an integer")
            model = (
                raw.get("model_resolution")
                or raw.get("model")
                or raw.get("downsample")
                or high
            )
            resize = raw.get("resize_long_side") or raw.get("resize")
            batch = raw.get("batch_size") or raw.get("batch")
            return cls(
                high_resolution=int(high),
                model_resolution=int(model),
                resize_long_side=int(resize) if resize is not None else None,
                batch_size=int(batch) if batch is not None else None,
            )
        raise TypeError(f"Unsupported resolution specification: {raw!r}")

    @property
    def folder_name(self) -> str:
        if self.model_resolution == self.high_resolution:
            return f"{self.high_resolution}px"
        return f"{self.high_resolution}to{self.model_resolution}px"

    @property
    def display(self) -> str:
        if self.model_resolution == self.high_resolution:
            return f"{self.high_resolution}px"
        return f"{self.high_resolution}px→{self.model_resolution}px"


@dataclass
class MultiPrecomputeDefaults:
    cache_subdir: str = "cache_embeddings"
    batch_size: int = 16
    num_workers: int = 4
    resize_long_side: Optional[int] = None
    limit: Optional[int] = None
    store_distribution: bool = True
    embeddings_dtype: torch.dtype = torch.float16
    device: Optional[str] = None
    devices: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiPrecomputeDefaults":
        cache_subdir = str(data.get("cache_subdir", "cache_embeddings"))
        batch = int(data.get("batch_size", 16))
        workers = int(data.get("num_workers", 4))
        resize = data.get("resize_long_side")
        limit = data.get("limit")
        store_distribution = data.get("store_distribution")
        dtype_value = data.get("embeddings_dtype")
        device = data.get("device")
        devices_raw = data.get("devices")

        devices: Optional[List[str]]
        if devices_raw is None:
            devices = None
        elif isinstance(devices_raw, str):
            items = [part.strip() for part in devices_raw.split(",")]
            devices = [item for item in items if item]
        elif isinstance(devices_raw, (list, tuple, set)):
            cleaned: List[str] = []
            for entry in devices_raw:
                text = str(entry).strip()
                if text:
                    cleaned.append(text)
            devices = cleaned or None
        else:
            raise TypeError("defaults.devices must be a string, list, or tuple of device names")

        return cls(
            cache_subdir=cache_subdir,
            batch_size=max(1, batch),
            num_workers=max(0, workers),
            resize_long_side=int(resize) if resize is not None else None,
            limit=int(limit) if limit is not None else None,
            store_distribution=_resolve_bool(store_distribution, default=True),
            embeddings_dtype=_resolve_dtype(dtype_value or "float16"),
            device=str(device) if device else None,
            devices=devices,
        )


@dataclass
class MultiVaeConfig:
    name: str
    slug: str
    load_from: Optional[str]
    hf_repo: Optional[str]
    hf_subfolder: Optional[str]
    hf_revision: Optional[str]
    hf_auth_token: Optional[str]
    vae_kind: Optional[str]
    weights_dtype: Optional[torch.dtype]
    embeddings_dtype: Optional[torch.dtype]
    dataset_root: Optional[Path]
    dataset_subdir: Optional[Path]
    cache_subdir: Optional[Path]
    batch_size: Optional[int]
    num_workers: Optional[int]
    resize_long_side: Optional[int]
    limit: Optional[int]
    store_distribution: Optional[bool]
    resolutions: List[ResolutionSpec] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, name: str, data: Dict[str, Any]) -> "MultiVaeConfig":
        resolutions_raw = (
            data.get("resolutions_with_batchsize")
            or data.get("resolutions")
            or data.get("resolutions_list")
        )
        if not resolutions_raw:
            raise ValueError(
                f"VAE entry '{name}' must define a 'resolutions' or 'resolutions_with_batchsize' list"
            )
        if isinstance(resolutions_raw, dict):
            iterable: Iterable[Any] = resolutions_raw.values()
        else:
            iterable = resolutions_raw
        resolutions = [ResolutionSpec.from_raw(item) for item in iterable]

        dataset_root = data.get("dataset_root")
        dataset_subdir = data.get("dataset_subdir")
        cache_subdir = data.get("cache_subdir")
        weights_dtype = data.get("weights_dtype")
        embeddings_dtype = data.get("embeddings_dtype")
        batch = data.get("batch_size")
        workers = data.get("num_workers")
        resize = data.get("resize_long_side")
        limit = data.get("limit")
        store_distribution = data.get("store_distribution")
        slug = _slugify(name) or name.replace("/", "_")

        return cls(
            name=name,
            slug=slug,
            load_from=str(data["load_from"]) if data.get("load_from") else None,
            hf_repo=str(data["hf_repo"]) if data.get("hf_repo") else None,
            hf_subfolder=str(data["hf_subfolder"]) if data.get("hf_subfolder") else None,
            hf_revision=str(data["hf_revision"]) if data.get("hf_revision") else None,
            hf_auth_token=str(data["hf_auth_token"]) if data.get("hf_auth_token") else None,
            vae_kind=str(data["vae_kind"]) if data.get("vae_kind") else None,
            weights_dtype=_resolve_dtype(weights_dtype) if weights_dtype else None,
            embeddings_dtype=_resolve_dtype(embeddings_dtype) if embeddings_dtype else None,
            dataset_root=Path(dataset_root).expanduser() if dataset_root else None,
            dataset_subdir=Path(dataset_subdir).expanduser() if dataset_subdir else None,
            cache_subdir=Path(cache_subdir).expanduser() if cache_subdir else None,
            batch_size=int(batch) if batch is not None else None,
            num_workers=int(workers) if workers is not None else None,
            resize_long_side=int(resize) if resize is not None else None,
            limit=int(limit) if limit is not None else None,
            store_distribution=(
                _resolve_bool(store_distribution) if store_distribution is not None else None
            ),
            resolutions=resolutions,
        )


@dataclass
class MultiPrecomputeTask:
    vae_name: str
    vae_slug: str
    dataset_path: Path
    cache_dir: Path
    high_resolution: int
    model_resolution: int
    resize_long_side: int
    limit: int
    batch_size: int
    num_workers: int
    store_distribution: bool
    embeddings_dtype: torch.dtype
    weights_dtype: Optional[torch.dtype]
    load_from: Optional[str]
    hf_repo: Optional[str]
    hf_subfolder: Optional[str]
    hf_revision: Optional[str]
    hf_auth_token: Optional[str]
    vae_kind: Optional[str]
    display_resolution: str


@dataclass
class MultiPrecomputeConfig:
    dataset_root: Optional[Path]
    cache_root: Optional[Path]
    defaults: MultiPrecomputeDefaults
    models: List[MultiVaeConfig]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiPrecomputeConfig":
        dataset_root_raw = data.get("dataset_root") or data.get("ds_path")
        cache_root_raw = data.get("cache_root")

        defaults = MultiPrecomputeDefaults.from_dict(data.get("defaults", {}))

        models_section = data.get("models") or data.get("vaes") or {}
        if isinstance(models_section, dict):
            items = models_section.items()
        elif isinstance(models_section, list):
            entries: List[tuple[str, Dict[str, Any]]] = []
            for entry in models_section:
                if not isinstance(entry, dict) or "name" not in entry:
                    raise ValueError("List-style model entries must include a 'name' field")
                payload = {key: value for key, value in entry.items() if key != "name"}
                entries.append((str(entry["name"]), payload))
            items = entries
        else:
            raise TypeError("'models' section must be a mapping or list of mappings")

        models = [MultiVaeConfig.from_mapping(str(name), dict(payload)) for name, payload in items]

        dataset_root = Path(dataset_root_raw).expanduser() if dataset_root_raw else None
        cache_root = Path(cache_root_raw).expanduser() if cache_root_raw else None

        return cls(dataset_root=dataset_root, cache_root=cache_root, defaults=defaults, models=models)

    def generate_tasks(
        self,
        *,
        dataset_root_override: Optional[Path] = None,
        cache_override: Optional[Path] = None,
        batch_override: Optional[int] = None,
        workers_override: Optional[int] = None,
    ) -> List[MultiPrecomputeTask]:
        def _resolve(base: Path, override: Optional[Path]) -> Path:
            if override is None:
                return base
            return override.resolve() if override.is_absolute() else (base / override).resolve()

        base_dataset_root = dataset_root_override or self.dataset_root
        if base_dataset_root is None:
            raise RuntimeError("Dataset root must be provided either in the config or via --dataset-root")
        base_dataset_root = base_dataset_root.expanduser().resolve()
        print(f"[Embeddings] Base dataset root resolved to {base_dataset_root}")

        cache_root = cache_override.expanduser() if cache_override else self.cache_root
        if cache_root is None:
            cache_root = base_dataset_root / self.defaults.cache_subdir
        cache_root = cache_root.expanduser() if isinstance(cache_root, Path) else Path(cache_root).expanduser()
        cache_root = cache_root if cache_root.is_absolute() else (base_dataset_root / cache_root).resolve()
        cache_root = cache_root.resolve()
        print(f"[Embeddings] Cache root resolved to {cache_root}")

        tasks: List[MultiPrecomputeTask] = []

        for model in self.models:
            dataset_root = _resolve(base_dataset_root, model.dataset_root) if model.dataset_root else base_dataset_root
            dataset_path = _resolve(dataset_root, model.dataset_subdir) if model.dataset_subdir else dataset_root
            print(
                f"[Embeddings] Model '{model.name}' dataset path set to {dataset_path}"
            )

            model_cache_root = cache_root / model.slug
            if model.cache_subdir is not None:
                model_cache_root = _resolve(cache_root, model.cache_subdir)

            base_batch = batch_override or model.batch_size or self.defaults.batch_size
            workers = workers_override or model.num_workers or self.defaults.num_workers
            resize = (
                model.resize_long_side
                if model.resize_long_side is not None
                else self.defaults.resize_long_side
            )
            limit = model.limit if model.limit is not None else self.defaults.limit
            store_distribution = (
                model.store_distribution if model.store_distribution is not None else self.defaults.store_distribution
            )
            embeddings_dtype = model.embeddings_dtype or self.defaults.embeddings_dtype
            base_batch = max(1, int(base_batch))
            last_high_resolution: Optional[int] = None
            reductions_applied = 0

            for resolution in model.resolutions:
                if last_high_resolution is not None and resolution.high_resolution > last_high_resolution:
                    reductions_applied += 1

                if resolution.batch_size is not None:
                    effective_batch = max(1, int(resolution.batch_size))
                else:
                    effective_batch = max(1, base_batch // (2**reductions_applied))

                resize_long_side = (
                    resolution.resize_long_side
                    if resolution.resize_long_side is not None
                    else (resize if resize is not None else 0)
                )

                cache_dir = (model_cache_root / resolution.folder_name).resolve()

                resolution_dirname = f"{resolution.high_resolution}px"
                candidate = dataset_path / resolution_dirname
                if candidate.is_dir():
                    resolution_dataset_path = candidate.resolve()
                    print(
                        f"[Embeddings] Using resolution-specific dataset directory {resolution_dataset_path}"
                    )
                else:
                    resolution_dataset_path = dataset_path

                tasks.append(
                    MultiPrecomputeTask(
                        vae_name=model.name,
                        vae_slug=model.slug,
                        dataset_path=resolution_dataset_path,
                        cache_dir=cache_dir,
                        high_resolution=resolution.high_resolution,
                        model_resolution=resolution.model_resolution,
                        resize_long_side=int(resize_long_side),
                        limit=int(limit) if limit is not None else 0,
                        batch_size=effective_batch,
                        num_workers=max(0, int(workers)),
                        store_distribution=bool(store_distribution),
                        embeddings_dtype=embeddings_dtype,
                        weights_dtype=model.weights_dtype,
                        load_from=model.load_from,
                        hf_repo=model.hf_repo,
                        hf_subfolder=model.hf_subfolder,
                        hf_revision=model.hf_revision,
                        hf_auth_token=model.hf_auth_token,
                        vae_kind=model.vae_kind,
                        display_resolution=resolution.display,
                    )
                )
                print(
                    f"[Embeddings] Scheduled task: {model.name} @ {resolution.display}"
                    f" -> dataset {resolution_dataset_path}, cache {cache_dir}"
                )

                last_high_resolution = resolution.high_resolution

        return tasks


# --------------------------------------------------------------------------------------
# CLI execution helpers


def _log(message: str) -> None:
    print(f"[Embeddings] {message}")


def parse_args() -> argparse.Namespace:
    _log("Parsing command-line arguments...")
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
    print(f"[Embeddings] Loading VAE for task '{task.vae_name}'...")
    load_path = Path(task.load_from).expanduser() if task.load_from else None
    path_exists = load_path is not None and load_path.exists()

    hf_source = task.hf_repo or None
    if not path_exists and task.load_from and not task.hf_repo:
        hf_source = task.load_from

    kind = (task.vae_kind or "").strip().lower()

    if kind == "qwen":
        if path_exists:
            source = str(load_path)
            kwargs: Dict[str, Any] = {}
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
    _log(f"VAE loaded from: {display_source}")
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


def _execute_task(
    task: MultiPrecomputeTask,
    *,
    device: torch.device,
    overwrite: bool,
    skip_existing: bool,
    index: int,
    total: int,
    progress_position: int | None = None,
    progress_desc: str | None = None,
    log: Callable[[str], None] | None = None,
) -> None:
    from .dataset import ImageFolderDataset  # Local import to avoid circular dependency.

    logger = log or print

    logger(
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
        overwrite=overwrite,
        precompute_batch_size=task.batch_size,
        num_workers=task.num_workers,
        store_distribution=task.store_distribution,
        vae_names=(task.vae_name,),
        vae_cache_dirs=(task.cache_dir,),
    )

    cache = EmbeddingCache(embeddings_cfg, dataset_cfg, task.dataset_path)
    logger(f"[Embeddings] EmbeddingCache initialized for task '{task.vae_name}'")

    dataset = ImageFolderDataset(
        root=task.dataset_path,
        high_resolution=task.high_resolution,
        resize_long_side=task.resize_long_side,
        limit=task.limit,
        embedding_cache=None,
        model_resolution=task.model_resolution,
    )
    logger(
        f"[Embeddings] Dataset created: Collected {len(dataset.paths)} paths from {task.dataset_path}"
    )

    if skip_existing and not overwrite:
        try:
            cache.validate_dataset(dataset)
        except RuntimeError:
            pass
        else:
            logger(
                (
                    f"[Embeddings] Cache already complete for {task.vae_name}"
                    f" ({task.display_resolution}); skipping."
                )
            )
            return

    start = time.perf_counter()
    logger(f"[Embeddings] Starting VAE encoding for task '{task.vae_name}'...")

    _ensure_cuda_context(device)
    vae = _load_vae(task)
    to_kwargs = {"device": device}
    if task.weights_dtype is not None:
        to_kwargs["dtype"] = task.weights_dtype
    vae = vae.to(**to_kwargs)
    vae.eval()
    logger(f"[Embeddings] VAE moved to device {device} with dtype {to_kwargs.get('dtype', 'default')}")

    cache.ensure_populated(
        dataset,
        vae,
        device=device,
        encode_dtype=next(vae.parameters()).dtype,
        seed=0,
        accelerator=None,
        progress_position=progress_position,
        progress_desc=progress_desc,
        log=logger,
    )

    cache.validate_dataset(dataset)
    elapsed = time.perf_counter() - start

    logger(
        f"[Embeddings] Finished {task.vae_name} ({task.display_resolution}) in {elapsed:.1f}s"
    )

    del vae
    if device.type == "cuda":
        _ensure_cuda_context(device)
        torch.cuda.empty_cache()
    logger(f"[Embeddings] CUDA cache cleared for device {device}")


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

    tqdm.set_lock(threading.RLock())
    overall_progress = tqdm(
        total=total,
        desc="VAE tasks",
        position=len(devices),
        leave=True,
        dynamic_ncols=True,
        unit="task",
    )
    log = tqdm.write
    device_positions = {str(device): idx for idx, device in enumerate(devices)}

    def worker(device: torch.device) -> None:
        _ensure_cuda_context(device)
        position = device_positions[str(device)]
        while True:
            with lock:
                try:
                    index, task = next(enumerator)
                except StopIteration:
                    return
            desc = f"{task.vae_name} ({task.display_resolution})"
            log(
                f"[Embeddings] Device {device} starting task {task.vae_name}"
                f" at {task.display_resolution}"
            )
            try:
                _execute_task(
                    task,
                    device=device,
                    overwrite=overwrite,
                    skip_existing=skip_existing,
                    index=index,
                    total=total,
                    progress_position=position,
                    progress_desc=desc,
                    log=log,
                )
            except Exception:
                raise
            else:
                overall_progress.update(1)
                log(
                    f"[Embeddings] Device {device} finished task {task.vae_name}"
                    f" at {task.display_resolution}"
                )

    try:
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [executor.submit(worker, device) for device in devices]
            for future in futures:
                future.result()
    finally:
        overall_progress.close()


def _run_from_config(args: argparse.Namespace) -> None:
    _log("Loading configuration...")
    raw_config = load_config([args.config])
    multi_cfg = MultiPrecomputeConfig.from_dict(raw_config)
    _log("Configuration loaded successfully")

    dataset_override = args.dataset_root.expanduser() if args.dataset_root else None
    cache_override = Path(args.cache_subdir).expanduser() if args.cache_subdir else None
    batch_override = int(args.batch_size) if args.batch_size is not None else None
    workers_override = int(args.num_workers) if args.num_workers is not None else None
    _log(
        "Overrides -> "
        f"dataset_root={dataset_override}, cache_subdir={cache_override}, "
        f"batch_size={batch_override}, num_workers={workers_override}"
    )
    _log("Generating tasks from configuration...")
    tasks = multi_cfg.generate_tasks(
        dataset_root_override=dataset_override,
        cache_override=cache_override,
        batch_override=batch_override,
        workers_override=workers_override,
    )
    _log(f"Generated {len(tasks)} task(s)")

    if not tasks:
        raise RuntimeError("No VAE tasks were defined in the configuration file")

    default_device_list = multi_cfg.defaults.devices
    devices = _resolve_device_list(
        single_device=args.device,
        device_list=args.devices if args.devices is not None else default_device_list,
        default_device=multi_cfg.defaults.device,
    )
    _log(f"Resolved {len(devices)} device(s): {[str(d) for d in devices]}")
    _log(
        f"Prepared {len(tasks)} task(s) for multi-VAE precomputation using {len(devices)} device(s)"
    )

    overall_start = time.perf_counter()

    if len(devices) == 1:
        device = devices[0]
        _log(f"Running tasks sequentially on single device {device}")
        for index, task in enumerate(tasks, start=1):
            desc = f"{task.vae_name} ({task.display_resolution})"
            _log(f"Queueing task {index}/{len(tasks)}: {task.vae_name} at {task.display_resolution}")
            _execute_task(
                task,
                device=device,
                overwrite=args.overwrite,
                skip_existing=args.skip_existing,
                index=index,
                total=len(tasks),
                progress_desc=desc,
            )
    else:
        _log(f"Running tasks in parallel on {len(devices)} devices")
        _run_parallel_tasks(
            tasks,
            devices,
            overwrite=args.overwrite,
            skip_existing=args.skip_existing,
        )

    total_elapsed = time.perf_counter() - overall_start
    _log(f"Multi-VAE precomputation finished in {total_elapsed:.1f}s total")


def main() -> None:
    args = parse_args()
    _run_from_config(args)


if __name__ == "__main__":
    main()
