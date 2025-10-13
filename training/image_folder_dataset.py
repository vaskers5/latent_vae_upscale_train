"""Image folder dataset helpers used by the VAE trainer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from .precompute_embeddings import EmbeddingCache, TransformParams

__all__ = ["ImageFolderDataset"]


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
        print(
            f"[Dataset] Initializing ImageFolderDataset with root={root}, high_resolution={high_resolution}, "
            f"resize_long_side={resize_long_side}, limit={limit}"
        )
        self.root = Path(root).expanduser().resolve()
        self.high_resolution = high_resolution
        self.paths: List[Path] = []
        self.embedding_cache = embedding_cache
        self.model_resolution = model_resolution
        self._needs_model_downsample = (
            self.model_resolution > 0 and self.model_resolution != self.high_resolution
        )
        self.paths = self._collect_valid_paths(limit=limit)
        print(f"[Dataset] Collected {len(self.paths)} valid paths")
        if not self.paths:
            raise RuntimeError(f"No valid images found under '{self.root}'")
        print(f"[Dataset] Total dataset length: {len(self.paths)}")

    def _collect_valid_paths(self, limit: int = 0) -> List[Path]:
        print(f"[Dataset] Collecting valid image paths from {self.root}...")
        return self._collect_from_disk(limit=limit)

    def _collect_from_disk(self, limit: int = 0) -> List[Path]:
        all_images: List[Path] = []
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        for root, _dirs, files in os.walk(self.root):
            for file in files:
                if os.path.splitext(file)[1].lower() in exts:
                    full_path = Path(os.path.abspath(os.path.join(root, file)))
                    all_images.append(full_path)
                    if limit > 0 and len(all_images) >= limit:
                        return all_images
        return all_images[:limit] if limit > 0 else all_images

    def set_embedding_cache(self, cache: Optional[EmbeddingCache]) -> None:
        self.embedding_cache = cache

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        print(f"[Dataset] Loading sample {index} from path: {path}")
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
        rng: Optional[Any] = None,
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
        return tensor

    def _apply_transforms(
        self,
        img: Image.Image,
        rng: Optional[Any] = None,
        params: Optional[TransformParams] = None,
    ) -> Tuple[Image.Image, TransformParams]:
        del rng, params
        return img, TransformParams(flip=False, crop_x=0, crop_y=0)
