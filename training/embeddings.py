"""Embedding cache utilities for precomputing and loading VAE latents."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import DatasetConfig, EmbeddingsConfig

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from accelerate import Accelerator
    from .dataset import ImageFolderDataset


_CACHE_VERSION = 1


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

    def validate_dataset(self, dataset: "ImageFolderDataset") -> None:
        """Ensure every dataset image has the expected cached embedding variants."""

        if not self.cfg.enabled:
            return

        missing: List[Tuple[Path, int]] = []
        for path in dataset.paths:
            for variant in range(self.cfg.variants_per_sample):
                if not self.has_variant(path, variant):
                    missing.append((path, variant))

        if missing:
            preview = "\n".join(
                f"- {path} (variant {variant})" for path, variant in missing[:10]
            )
            suffix = "" if len(missing) <= 10 else f"\n... and {len(missing) - 10} more"
            raise RuntimeError(
                "Embedding cache incomplete. Missing the following entries:\n"
                f"{preview}{suffix}"
            )

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

        dataset_loader = _EmbeddingGenerationDataset(
            dataset=dataset,
            pending=missing,
            cache=self,
            seed=seed,
        )
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
            from tqdm.auto import tqdm  # local import to avoid global dependency

            progress = tqdm(total=len(missing), desc="Precomputing latents", unit="img")

        with torch.no_grad():
            for batch in loader:
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
                    variant_index = batch["variants"][idx]
                    params = batch["params"][idx]
                    mean_tensor = latent_mean[idx] if self.cfg.store_distribution else None
                    logvar_tensor = latent_logvar[idx] if latent_logvar is not None else None
                    self.save_record(
                        image_path,
                        variant_index,
                        latents[idx],
                        params,
                        mean=mean_tensor,
                        logvar=logvar_tensor,
                    )

                if progress is not None:
                    progress.update(len(batch["paths"]))

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
    def _build_rng(seed: int, image_path: Path, variant_index: int) -> random.Random:
        payload = f"{seed}:{image_path.as_posix()}:{variant_index}".encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()[:16]
        seed_value = int(digest, 16) & 0xFFFFFFFF
        return random.Random(seed_value)


class _EmbeddingGenerationDataset(Dataset):
    """Dataset wrapper that yields tensors ready for VAE encoding."""

    def __init__(
        self,
        *,
        dataset: "ImageFolderDataset",
        pending: Sequence[Tuple[Path, int]],
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
        image_path, variant_index = self.pending[index]
        rng = self.cache._build_rng(self.seed, image_path, variant_index)
        sample_tensor, params = self.dataset.build_tensor_sample(image_path, rng=rng, params=None)
        model_input = self.dataset.prepare_model_input(sample_tensor).contiguous()
        return {
            "path": image_path,
            "variant": variant_index,
            "model_input": model_input,
            "params": params,
        }


def _collate_embedding_batches(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    model_inputs = torch.stack([item["model_input"] for item in batch], dim=0)
    return {
        "model_input": model_inputs,
        "params": [item["params"] for item in batch],
        "paths": [item["path"] for item in batch],
        "variants": [item["variant"] for item in batch],
    }
