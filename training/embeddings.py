"""Embedding cache utilities for precomputing and loading VAE latents."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


from .config import DatasetConfig, EmbeddingsConfig

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from accelerate import Accelerator
    from .dataset import ImageFolderDataset


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
    """Container returned when loading a cached embedding."""

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
        self._pending_log_entries = []
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
            # Fallback for datasets outside the configured root – hash the absolute path.
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
        filename = f"{self._sanitize_fragment(slug_candidate)}_{self._sanitize_fragment(resolution)}_embed_cache_log"
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
                    parts = line.split("\t")
                    relative = Path(parts[0])
                    index.add(relative)
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
        # self.log(f"Saving embedding for {image_path} variant {variant_index}")
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
                f"Embedding cache version mismatch for '{record_path}'. Expected {_CACHE_VERSION}, found {version}."
            )
        params = TransformParams.from_dict(payload.get("params", {}))
        latents = payload["latents"]
        mean = payload.get("mean")
        logvar = payload.get("logvar")
        config = payload.get("dataset", {})
        return EmbeddingRecord(latents=latents, mean=mean, logvar=logvar, params=params, variant_index=0, config=config)

    def choose_record(self, image_path: Path, rng: Optional[random.Random] = None) -> EmbeddingRecord:
        if not self.has_variant(image_path, 0):
            raise FileNotFoundError(f"No cached embeddings available for '{image_path}'")
        self.log(f"Chose cached embedding for {image_path}")
        return self.load_record(image_path)

    def validate_dataset(self, dataset: "ImageFolderDataset") -> None:
        """Ensure every dataset image has the expected cached embedding variants."""

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
        else:
            self.log("Dataset validation passed: all embeddings available")

    # ------------------------------------------------------------------ preparation
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
            # A different process is responsible for generating the cache – return and wait for sync.
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
        # self.cache.log(f"Building tensor sample for {image_path}")
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
