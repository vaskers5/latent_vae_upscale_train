"""Simplified configuration helpers for multi-VAE embedding precomputation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

from .config import _resolve_bool, _resolve_dtype, _slugify

__all__ = [
    "MultiPrecomputeConfig",
    "MultiPrecomputeDefaults",
    "MultiPrecomputeTask",
]


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
    variants_per_sample: int = 1
    batch_size: int = 16
    num_workers: int = 4
    resize_long_side: Optional[int] = None
    limit: Optional[int] = None
    store_distribution: bool = True
    embeddings_dtype: torch.dtype = torch.float16
    device: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiPrecomputeDefaults":
        cache_subdir = str(data.get("cache_subdir", "cache_embeddings"))
        variants = int(data.get("variants_per_sample", 1))
        batch = int(data.get("batch_size", 16))
        workers = int(data.get("num_workers", 4))
        resize = data.get("resize_long_side")
        limit = data.get("limit")
        store_distribution = data.get("store_distribution")
        dtype_value = data.get("embeddings_dtype")
        device = data.get("device")

        return cls(
            cache_subdir=cache_subdir,
            variants_per_sample=max(1, variants),
            batch_size=max(1, batch),
            num_workers=max(0, workers),
            resize_long_side=int(resize) if resize is not None else None,
            limit=int(limit) if limit is not None else None,
            store_distribution=_resolve_bool(store_distribution, default=True),
            embeddings_dtype=_resolve_dtype(dtype_value or "float16"),
            device=str(device) if device else None,
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
    variants_per_sample: Optional[int]
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
        variants = data.get("variants_per_sample")
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
            variants_per_sample=int(variants) if variants is not None else None,
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
    variants_per_sample: int
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
        variants_override: Optional[int] = None,
        batch_override: Optional[int] = None,
        workers_override: Optional[int] = None,
    ) -> List[MultiPrecomputeTask]:
        base_dataset_root = dataset_root_override or self.dataset_root
        if base_dataset_root is None:
            raise RuntimeError("Dataset root must be provided either in the config or via --dataset-root")
        base_dataset_root = base_dataset_root.expanduser().resolve()

        if cache_override is not None:
            cache_root = cache_override.expanduser()
        elif self.cache_root is not None:
            cache_root = self.cache_root.expanduser()
        else:
            cache_root = base_dataset_root / self.defaults.cache_subdir
        if not cache_root.is_absolute():
            cache_root = (base_dataset_root / cache_root).resolve()
        else:
            cache_root = cache_root.resolve()

        tasks: List[MultiPrecomputeTask] = []

        for model in self.models:
            dataset_root = base_dataset_root
            if model.dataset_root is not None:
                dataset_root = (
                    model.dataset_root.resolve()
                    if model.dataset_root.is_absolute()
                    else (base_dataset_root / model.dataset_root).resolve()
                )

            dataset_path = dataset_root
            if model.dataset_subdir is not None:
                dataset_path = (
                    model.dataset_subdir.resolve()
                    if model.dataset_subdir.is_absolute()
                    else (dataset_root / model.dataset_subdir).resolve()
                )

            model_cache_root = cache_root / model.slug
            if model.cache_subdir is not None:
                model_cache_root = (
                    model.cache_subdir.resolve()
                    if model.cache_subdir.is_absolute()
                    else (cache_root / model.cache_subdir).resolve()
                )

            variants = variants_override or model.variants_per_sample or self.defaults.variants_per_sample
            base_batch = batch_override or model.batch_size or self.defaults.batch_size
            workers = workers_override or model.num_workers or self.defaults.num_workers
            resize = model.resize_long_side if model.resize_long_side is not None else self.defaults.resize_long_side
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

                tasks.append(
                    MultiPrecomputeTask(
                        vae_name=model.name,
                        vae_slug=model.slug,
                        dataset_path=dataset_path,
                        cache_dir=cache_dir,
                        high_resolution=resolution.high_resolution,
                        model_resolution=resolution.model_resolution,
                        resize_long_side=int(resize_long_side),
                        limit=int(limit) if limit is not None else 0,
                        variants_per_sample=max(1, int(variants)),
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

                last_high_resolution = resolution.high_resolution

        return tasks
