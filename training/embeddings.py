"""Embedding cache utilities for precomputing and loading VAE latents."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DatasetConfig, EmbeddingsConfig

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from accelerate import Accelerator
    from .dataset import ImageFolderDataset


_CACHE_VERSION = 1


@dataclass(frozen=True)
class TransformParams:
    """Deterministic description of spatial augmentations applied to an image."""

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
    ) -> None:
        self.cfg = embeddings_config
        self.dataset_cfg = dataset_config
        self.dataset_root = dataset_root.resolve()
        self.cache_root = self.cfg.cache_dir.resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ storage helpers
    def _relative_path(self, image_path: Path) -> Path:
        try:
            return image_path.resolve().relative_to(self.dataset_root)
        except ValueError:
            # Fallback for datasets outside the configured root – hash the absolute path.
            digest = hashlib.sha256(str(image_path.resolve()).encode("utf-8")).hexdigest()[:16]
            safe_name = image_path.stem + f"_{digest}"
            return Path("__external__") / safe_name

    def _record_path(self, image_path: Path, variant_index: int) -> Path:
        relative = self._relative_path(image_path)
        filename = f"{relative.stem}_v{variant_index:02d}.pt"
        return self.cache_root / relative.parent / filename

    def available_variants(self, image_path: Path) -> List[int]:
        variants: List[int] = []
        for idx in range(self.cfg.variants_per_sample):
            if self._record_path(image_path, idx).exists():
                variants.append(idx)
        return variants

    def has_variant(self, image_path: Path, variant_index: int) -> bool:
        return self._record_path(image_path, variant_index).exists()

    def save_record(
        self,
        image_path: Path,
        variant_index: int,
        latents: torch.Tensor,
        params: TransformParams,
        *,
        mean: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
    ) -> None:
        record_path = self._record_path(image_path, variant_index)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "version": _CACHE_VERSION,
            "latents": latents.cpu(),
            "params": params.to_dict(),
            "variant": int(variant_index),
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

    def load_record(self, image_path: Path, variant_index: int) -> EmbeddingRecord:
        record_path = self._record_path(image_path, variant_index)
        if not record_path.exists():
            raise FileNotFoundError(f"Missing embedding cache for '{image_path}' variant {variant_index}")
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
        return EmbeddingRecord(latents=latents, mean=mean, logvar=logvar, params=params, variant_index=variant_index, config=config)

    def choose_record(self, image_path: Path, rng: Optional[random.Random] = None) -> EmbeddingRecord:
        variants = self.available_variants(image_path)
        if not variants:
            raise FileNotFoundError(f"No cached embeddings available for '{image_path}'")
        if rng is None:
            variant_index = random.choice(variants)
        else:
            variant_index = variants[rng.randrange(0, len(variants))]
        return self.load_record(image_path, variant_index)

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
    ) -> None:
        if not self.cfg.enabled:
            return

        missing: List[Tuple[Path, int]] = []
        for path in dataset.paths:
            for variant in range(self.cfg.variants_per_sample):
                present = self.has_variant(path, variant)
                if present and not self.cfg.overwrite:
                    continue
                missing.append((path, variant))

        main_process = accelerator is None or accelerator.is_main_process

        if not missing:
            if main_process:
                print("[Embeddings] Cache already populated")
            return

        if not main_process:
            # A different process is responsible for generating the cache – return and wait for sync.
            return

        print(f"[Embeddings] Populating cache for {len(missing)} variants ...")

        vae_prev_mode = vae.training
        vae = vae.to(device)
        vae.eval()

        from tqdm.auto import tqdm  # local import to avoid global dependency

        iterator: Iterable[Tuple[Path, int]]
        if accelerator is None or accelerator.is_main_process:
            iterator = tqdm(missing, desc="Precomputing latents", unit="img")
        else:
            iterator = missing

        with torch.no_grad():
            for image_path, variant_index in iterator:
                rng = self._build_rng(seed, image_path, variant_index)
                sample_tensor, params = dataset.build_tensor_sample(image_path, rng=rng, params=None)
                # sample_tensor: (C, H, W) in [-1, 1]
                input_tensor = sample_tensor.unsqueeze(0)
                if self.dataset_cfg.high_resolution != self.dataset_cfg.model_resolution:
                    input_tensor = F.interpolate(
                        input_tensor,
                        size=(self.dataset_cfg.model_resolution, self.dataset_cfg.model_resolution),
                        mode="bilinear",
                        align_corners=False,
                    )
                encode_input = input_tensor.to(device=device, dtype=encode_dtype)
                if self._is_video_vae(vae):
                    encode_input = encode_input.unsqueeze(2)
                encoding = vae.encode(encode_input)
                latent_mean = encoding.latent_dist.mean.detach()
                if self.cfg.store_distribution:
                    latent_logvar = encoding.latent_dist.logvar.detach()
                else:
                    latent_logvar = None
                latents = latent_mean
                if self._is_video_vae(vae):
                    latents = latents.squeeze(2)
                    if latent_logvar is not None:
                        latent_logvar = latent_logvar.squeeze(2)
                latents = latents.squeeze(0).to(self.cfg.dtype)
                mean_tensor = latent_mean.squeeze(0) if self.cfg.store_distribution else None
                logvar_tensor = latent_logvar.squeeze(0) if latent_logvar is not None else None
                self.save_record(
                    image_path,
                    variant_index,
                    latents,
                    params,
                    mean=mean_tensor,
                    logvar=logvar_tensor,
                )

        print("[Embeddings] Cache population complete")
        vae.train(vae_prev_mode)

    @staticmethod
    def _is_video_vae(module: nn.Module) -> bool:
        encoder = getattr(module, "encoder", None)
        conv_in = getattr(encoder, "conv_in", None) if encoder is not None else None
        weight = getattr(conv_in, "weight", None) if conv_in is not None else None
        return isinstance(weight, torch.nn.Parameter) and weight.ndimension() == 5

    @staticmethod
    def _build_rng(seed: int, image_path: Path, variant_index: int) -> random.Random:
        payload = f"{seed}:{image_path.as_posix()}:{variant_index}".encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()[:16]
        seed_value = int(digest, 16) & 0xFFFFFFFF
        return random.Random(seed_value)