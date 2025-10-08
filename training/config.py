"""Lightweight configuration objects used across the training scripts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

__all__ = [
    "_resolve_bool",
    "_resolve_dtype",
    "PathsConfig",
    "DatasetConfig",
    "OptimizerConfig",
    "SampleVaeConfig",
    "ModelConfig",
    "LossConfig",
    "LoggingConfig",
    "LatentUpscalerConfig",
    "EmbeddingsConfig",
    "TrainingConfig",
]


def _slugify(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in text.strip())
    return cleaned or None


def _resolve_dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
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
    key = str(value).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype value: {value!r}")
    return mapping[key]


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
        exp_name = data.get("exp_name")
        timestamp = datetime.now().strftime("%Y_%m_%d_%H")
        slug = _slugify(exp_name)
        folder = f"{timestamp}_{slug}".strip("_") if slug else timestamp
        run_dir = save_root / folder
        return cls(
            dataset_root=dataset_root,
            project=project,
            save_root=save_root,
            exp_name=exp_name,
            timestamp=timestamp,
            run_dir=run_dir,
            samples_dir=run_dir / str(data.get("generated_folder_name", "samples")),
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetConfig":
        section = data.get("dataset", {})
        return cls(
            high_resolution=int(section.get("high_resolution", data.get("high_resolution", 256))),
            model_resolution=int(section.get("model_resolution", data.get("model_resolution", 256))),
            resize_long_side=int(section.get("resize_long_side", data.get("resize_long_side", 0))),
            limit=int(section.get("limit", data.get("limit", 0))),
            num_workers=int(section.get("num_workers", data.get("num_workers", 4))),
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
    gradient_accumulation_steps: int
    scheduler: str

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
            use_decay=_resolve_bool(section.get("use_decay", data.get("use_decay", True)), default=True),
            weight_decay=float(section.get("weight_decay", data.get("weight_decay", 0.01))),
            gradient_accumulation_steps=int(
                section.get("gradient_accumulation_steps", data.get("gradient_accumulation_steps", 1))
            ),
            scheduler=str(section.get("scheduler", data.get("scheduler", "cosine"))).lower(),
        )


@dataclass
class SampleVaeConfig:
    load_from: Optional[str] = None
    hf_repo: Optional[str] = None
    hf_subfolder: Optional[str] = None
    hf_revision: Optional[str] = None
    hf_auth_token: Optional[str] = None
    vae_kind: Optional[str] = None
    weights_dtype: Optional[torch.dtype] = None

    @staticmethod
    def _clean_string(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], defaults: Dict[str, Any]) -> "SampleVaeConfig":
        load_from = cls._clean_string(data.get("load_from")) or cls._clean_string(defaults.get("load_from"))
        hf_repo = cls._clean_string(data.get("hf_repo")) or cls._clean_string(defaults.get("hf_repo"))
        hf_subfolder = cls._clean_string(data.get("hf_subfolder")) or cls._clean_string(defaults.get("hf_subfolder"))
        hf_revision = cls._clean_string(data.get("hf_revision")) or cls._clean_string(defaults.get("hf_revision"))
        hf_auth_token = cls._clean_string(data.get("hf_auth_token")) or cls._clean_string(defaults.get("hf_auth_token"))

        raw_kind = cls._clean_string(data.get("vae_kind")) or cls._clean_string(defaults.get("vae_kind"))
        vae_kind = raw_kind.lower() if raw_kind else None

        raw_dtype = data.get("weights_dtype")
        if raw_dtype is None:
            raw_dtype = defaults.get("weights_dtype")
        weights_dtype = _resolve_dtype(raw_dtype) if raw_dtype is not None else None

        return cls(
            load_from=load_from,
            hf_repo=hf_repo,
            hf_subfolder=hf_subfolder,
            hf_revision=hf_revision,
            hf_auth_token=hf_auth_token,
            vae_kind=vae_kind,
            weights_dtype=weights_dtype,
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SampleVaeConfig":
        weights = data.get("weights_dtype")
        dtype = _resolve_dtype(weights) if weights is not None else None
        return cls(
            load_from=cls._clean_string(data.get("load_from")),
            hf_repo=cls._clean_string(data.get("hf_repo")),
            hf_subfolder=cls._clean_string(data.get("hf_subfolder")),
            hf_revision=cls._clean_string(data.get("hf_revision")),
            hf_auth_token=cls._clean_string(data.get("hf_auth_token")),
            vae_kind=cls._clean_string(data.get("vae_kind")),
            weights_dtype=dtype,
        )


@dataclass
class ModelConfig:
    load_from: Optional[str]
    hf_repo: Optional[str]
    hf_subfolder: Optional[str]
    hf_revision: Optional[str]
    hf_auth_token: Optional[str]
    weights_dtype: torch.dtype
    vae_kind: str
    sample_vaes: Dict[str, SampleVaeConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        section = data.get("model", {})

        def _from_section(key: str) -> Optional[str]:
            value = section.get(key)
            if value is None:
                value = data.get(key)
            return str(value) if value else None

        load_from = _from_section("load_from")
        base_dtype = _resolve_dtype(section.get("weights_dtype", data.get("weights_dtype", "float32")))
        base_kind = str(section.get("vae_kind", data.get("vae_kind", "kl"))).strip().lower()

        sample_vaes_raw = section.get("sample_vaes") or data.get("sample_vaes") or {}
        sample_vaes: Dict[str, SampleVaeConfig] = {}
        if isinstance(sample_vaes_raw, Mapping):
            defaults = {
                "load_from": load_from,
                "hf_repo": section.get("hf_repo") or data.get("hf_repo"),
                "hf_subfolder": section.get("hf_subfolder") or data.get("hf_subfolder"),
                "hf_revision": section.get("hf_revision") or data.get("hf_revision"),
                "hf_auth_token": section.get("hf_auth_token") or data.get("hf_auth_token"),
                "vae_kind": base_kind,
                "weights_dtype": base_dtype,
            }
            for key, value in sample_vaes_raw.items():
                if isinstance(value, Mapping):
                    sample_vaes[str(key)] = SampleVaeConfig.from_dict(dict(value), defaults)

        return cls(
            load_from=load_from,
            hf_repo=_from_section("hf_repo"),
            hf_subfolder=_from_section("hf_subfolder"),
            hf_revision=_from_section("hf_revision"),
            hf_auth_token=_from_section("hf_auth_token"),
            weights_dtype=base_dtype,
            vae_kind=base_kind,
            sample_vaes=sample_vaes,
        )


@dataclass
class LossConfig:
    lpips_backbone: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LossConfig":
        section = data.get("loss", {})
        backbone = section.get("lpips_backbone", data.get("lpips_backbone", "vgg"))
        return cls(lpips_backbone=str(backbone))


@dataclass
class LoggingConfig:
    use_wandb: bool
    wandb_run_name: Optional[str]
    global_sample_interval: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any], timestamp: str) -> "LoggingConfig":
        section = data.get("logging", {})
        use_wandb = _resolve_bool(section.get("use_wandb", data.get("use_wandb", False)))
        run_name = section.get("wandb_run_name") or data.get("wandb_run_name") or timestamp
        sample_interval = int(section.get("global_sample_interval", data.get("GLOBAL_SAMPLE_INTERVAL", 500)))
        return cls(use_wandb=use_wandb, wandb_run_name=run_name, global_sample_interval=max(1, sample_interval))


@dataclass
class LatentUpscalerConfig:
    model_name: str
    model: Optional[str]
    window: Optional[int]
    depth: Optional[int]
    heads: Optional[int]
    mlp_ratio: Optional[float]
    liif_hidden: Optional[int]
    patch_size: Optional[int]
    blocks: Optional[int]
    nerf_blocks: Optional[int]
    groups: Optional[int]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LatentUpscalerConfig":
        section = data.get("latent_upscaler", {})
        name = section.get("model_name") or section.get("model") or data.get("latent_upscaler", "swin")
        patch_size = section.get("patch_size")
        return cls(
            model_name=str(name or "swin"),
            model=str(name or "swin"),
            window=_maybe_int(section.get("window") or patch_size),
            depth=_maybe_int(section.get("depth")),
            heads=_maybe_int(section.get("heads")),
            mlp_ratio=_maybe_float(section.get("mlp_ratio")),
            liif_hidden=_maybe_int(section.get("liif_hidden")),
            patch_size=_maybe_int(patch_size),
            blocks=_maybe_int(section.get("blocks") or section.get("nerf_blocks")),
            nerf_blocks=_maybe_int(section.get("nerf_blocks")),
            groups=_maybe_int(section.get("groups")),
        )


def _maybe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class EmbeddingsConfig:
    enabled: bool
    cache_dir: Path
    dtype: torch.dtype
    overwrite: bool
    precompute_batch_size: int
    num_workers: int
    store_distribution: bool
    vae_names: tuple[str, ...]
    vae_cache_dirs: tuple[Path, ...]

    @classmethod
    def from_dict(cls, data: Dict[str, Any], dataset_root: Path) -> "EmbeddingsConfig":
        section = data.get("embeddings", {})
        enabled = _resolve_bool(section.get("enabled", data.get("embeddings_enabled", False)))
        cache_dir_raw = section.get("cache_dir") or data.get("embeddings_cache_dir")
        cache_dir = Path(cache_dir_raw) if cache_dir_raw else dataset_root / "cache_embeddings"
        if not cache_dir.is_absolute():
            cache_dir = (dataset_root / cache_dir).resolve()
        dtype = _resolve_dtype(section.get("dtype", data.get("embeddings_dtype", "float16")))
        overwrite = _resolve_bool(section.get("overwrite", data.get("embeddings_overwrite", False)))
        precompute_batch_size = int(
            section.get("precompute_batch_size", data.get("embeddings_precompute_batch_size", 16))
        )
        num_workers = int(section.get("precompute_num_workers", data.get("embeddings_precompute_num_workers", 4)))
        store_distribution = _resolve_bool(
            section.get("store_distribution", data.get("embeddings_store_distribution", True)),
            default=True,
        )

        raw_names = section.get("vae_names") or data.get("vae_names") or data.get("embeddings_vae_names")
        vae_names = _coerce_names(raw_names)
        if not vae_names:
            fallback = cache_dir.name or "default"
            vae_names = [fallback]
        cache_dirs = [_resolve_cache_dir(cache_dir, name) for name in vae_names]

        return cls(
            enabled=enabled,
            cache_dir=cache_dir,
            dtype=dtype,
            overwrite=overwrite,
            precompute_batch_size=max(1, precompute_batch_size),
            num_workers=max(0, num_workers),
            store_distribution=store_distribution,
            vae_names=tuple(vae_names),
            vae_cache_dirs=tuple(cache_dirs),
        )


def _coerce_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        candidate = value.strip()
        return [candidate] if candidate else []
    names: list[str] = []
    if isinstance(value, (list, tuple, set)):
        for item in value:
            if item is None:
                continue
            candidate = str(item).strip()
            if candidate:
                names.append(candidate)
    return names


def _resolve_cache_dir(base: Path, name: str) -> Path:
    path = Path(name)
    if path.is_absolute():
        return path
    return (base / path).resolve()


@dataclass
class TrainingConfig:
    paths: PathsConfig
    dataset: DatasetConfig
    optimiser: OptimizerConfig
    model: ModelConfig
    losses: LossConfig
    logging: LoggingConfig
    latent_upscaler: LatentUpscalerConfig
    embeddings: EmbeddingsConfig
    seed: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        paths = PathsConfig.from_dict(data)
        dataset = DatasetConfig.from_dict(data)
        optimiser = OptimizerConfig.from_dict(data)
        model = ModelConfig.from_dict(data)
        losses = LossConfig.from_dict(data)
        logging = LoggingConfig.from_dict(data, timestamp=paths.timestamp)
        latent_upscaler = LatentUpscalerConfig.from_dict(data)
        embeddings = EmbeddingsConfig.from_dict(data, dataset_root=paths.dataset_root)
        seed = int(data.get("seed", int(datetime.now().strftime("%Y%m%d"))))
        return cls(
            paths=paths,
            dataset=dataset,
            optimiser=optimiser,
            model=model,
            losses=losses,
            logging=logging,
            latent_upscaler=latent_upscaler,
            embeddings=embeddings,
            seed=seed,
        )

