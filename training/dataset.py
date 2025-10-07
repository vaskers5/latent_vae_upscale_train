"""Dataset utilities used by the VAE trainer."""

from __future__ import annotations

import glob
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from .embeddings import EmbeddingCache, TransformParams

__all__ = ["ImageFolderDataset", "UpscaleDataset"]


class ImageFolderDataset(Dataset):
    """Recursively reads images and returns centred crops normalised to [-1, 1]."""

    def __init__(
        self,
        root: Path,
        high_resolution: int,
        resize_long_side: int = 0,
        limit: int = 0,
        embedding_cache: Optional[EmbeddingCache] = None,
        model_resolution: int = 0,
    ) -> None:
        self.root = root
        self.high_resolution = high_resolution
        self.resize_long_side = resize_long_side
        self.paths: List[Path] = []
        self.embedding_cache = embedding_cache
        self.model_resolution = model_resolution
        self._needs_model_downsample = (
            self.model_resolution > 0 and self.model_resolution != self.high_resolution
        )

        exts = {".png", ".jpg", ".jpeg", ".webp"}
        for current_root, _dirs, files in os.walk(root):
            for name in files:
                if Path(name).suffix.lower() in exts:
                    self.paths.append(Path(current_root) / name)
        if limit:
            self.paths = self.paths[:limit]

        self.paths = self._filter_valid_images(self.paths)
        if not self.paths:
            raise RuntimeError(f"No valid images found under '{root}'")
        random.shuffle(self.paths)

    def set_embedding_cache(self, cache: Optional[EmbeddingCache]) -> None:
        self.embedding_cache = cache

    @staticmethod
    def _filter_valid_images(paths: Iterable[Path]) -> List[Path]:
        valid: List[Path] = []
        for path in paths:
            try:
                with Image.open(path) as img:
                    img.verify()
                valid.append(path)
            except (OSError, UnidentifiedImageError):
                continue
        return valid

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index % len(self.paths)]
        if self.embedding_cache is None or not self.embedding_cache.cfg.enabled:
            tensor, _params = self.build_tensor_sample(path)
            return {
                "image": tensor,
                "model_input": self.prepare_model_input(tensor),
            }

        record = self.embedding_cache.choose_record(path)
        tensor, params = self.build_tensor_sample(path, params=record.params)
        sample = {
            "image": tensor,
            "model_input": self.prepare_model_input(tensor),
            "latents": record.latents,
            "path": str(path),
            "variant_index": record.variant_index,
        }
        if record.mean is not None:
            sample["latent_mean"] = record.mean
        if record.logvar is not None:
            sample["latent_logvar"] = record.logvar
        return sample

    def build_tensor_sample(
        self,
        path: Path,
        rng: Optional[random.Random] = None,
        params: Optional[TransformParams] = None,
    ) -> Tuple[torch.Tensor, TransformParams]:
        with Image.open(path) as img:
            img = img.convert("RGB")
            transformed, used_params = self._apply_transforms(img, rng=rng, params=params)
        tensor = TF.to_tensor(transformed)
        tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return tensor, used_params

    def prepare_model_input(self, tensor: torch.Tensor) -> torch.Tensor:
        """Produce the tensor fed into the model, applying downsampling if needed."""

        return self._downsample_for_model(tensor)

    def _downsample_for_model(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self._needs_model_downsample:
            return tensor
        batched = tensor.unsqueeze(0)
        resized = F.interpolate(
            batched,
            size=(self.model_resolution, self.model_resolution),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0)

    def _resize_if_needed(self, img: Image.Image) -> Image.Image:
        if self.resize_long_side <= 0:
            return img
        width, height = img.size
        longest = max(width, height)
        if longest <= self.resize_long_side:
            return img
        scale = self.resize_long_side / float(longest)
        new_size = (int(round(width * scale)), int(round(height * scale)))
        return img.resize(new_size, Image.LANCZOS)

    def _apply_transforms(
        self,
        img: Image.Image,
        rng: Optional[random.Random] = None,
        params: Optional[TransformParams] = None,
    ) -> Tuple[Image.Image, TransformParams]:
        del rng, params

        if self.high_resolution <= 0:
            return img, TransformParams(flip=False, crop_x=0, crop_y=0)

        img = self._resize_if_needed(img)
        img = ImageOps.fit(
            img,
            (self.high_resolution, self.high_resolution),
            Image.LANCZOS,
            centering=(0.5, 0.5),
        )
        return img, TransformParams(flip=False, crop_x=0, crop_y=0)


class UpscaleDataset(Dataset):
    """Dataset that loads precomputed low/high resolution tensor pairs."""

    def __init__(self, cache_dir: str, low_res: int, high_res: int) -> None:
        self.pairs = []
        low_res_files = glob.glob(
            f"{cache_dir}/{low_res}px/**/*.pt", recursive=True
        )

        for low_path in low_res_files:
            high_path = low_path.replace(
                f"{os.sep}{low_res}px{os.sep}", f"{os.sep}{high_res}px{os.sep}"
            )
            if os.path.exists(high_path):
                self.pairs.append((low_path, high_path))

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _extract_latent(record: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Normalize stored latent payloads into tensors, preserving auxiliary metadata.

        Newer caches store dictionaries with a `latents` tensor along with mean/logvar,
        while older caches may persist the tensor directly. We gracefully handle both.
        """

        if isinstance(record, torch.Tensor):
            return record, {}

        if isinstance(record, dict):
            for key in ("latents", "tensor", "latent"):
                value = record.get(key)
                if isinstance(value, torch.Tensor):
                    meta = {k: v for k, v in record.items() if k != key}
                    return value, meta
            raise TypeError("Latent record dict does not contain a tensor payload.")

        if isinstance(record, (list, tuple)) and record:
            tensor_candidate = None
            meta: Dict[str, Any] = {}
            for item in record:
                if isinstance(item, torch.Tensor) and tensor_candidate is None:
                    tensor_candidate = item
                elif isinstance(item, dict):
                    meta.update(item)
            if tensor_candidate is not None:
                return tensor_candidate, meta

        raise TypeError(f"Unsupported latent record type: {type(record)}")

    def __getitem__(self, idx: int):
        low_path, high_path = self.pairs[idx]
        low_record = torch.load(low_path, map_location="cpu")
        high_record = torch.load(high_path, map_location="cpu")

        low_tensor, low_meta = self._extract_latent(low_record)
        high_tensor, high_meta = self._extract_latent(high_record)

        sample: Dict[str, Any] = {"low": low_tensor, "high": high_tensor}
        if low_meta:
            sample["low_meta"] = low_meta
        if high_meta:
            sample["high_meta"] = high_meta
        return sample
