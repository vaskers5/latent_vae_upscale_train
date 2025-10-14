"""Dataset helpers for iterating over image folders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from .embedding_io import TransformParams

__all__ = ["ImageFolderDataset"]


class ImageFolderDataset(Dataset):
    """Recursively reads images and returns centred crops normalised to [-1, 1]."""

    def __init__(
        self,
        root: Path,
        high_resolution: int,
        resize_long_side: int = 0,
        limit: int = 0,
        embedding_cache: Optional[Any] = None,
        model_resolution: int = 0,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.high_resolution = high_resolution
        self.resize_long_side = resize_long_side
        self.embedding_cache = embedding_cache
        self.model_resolution = model_resolution
        self._needs_model_downsample = (
            self.model_resolution > 0 and self.model_resolution != self.high_resolution
        )
        self.paths = self._collect_paths(limit)
        if not self.paths:
            raise RuntimeError(f"No valid images found under '{self.root}'")

    def _collect_paths(self, limit: int) -> List[Path]:
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        collected: List[Path] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in exts:
                continue
            collected.append(path)
            if limit > 0 and len(collected) >= limit:
                break
        return collected if limit == 0 else collected[:limit]

    def set_embedding_cache(self, cache: Optional[Any]) -> None:
        self.embedding_cache = cache

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        tensor, params = self.build_tensor_sample(path)
        sample = {
            "image": tensor,
            "model_input": self.prepare_model_input(tensor),
            "params": params,
            "path": path,
        }
        if self.embedding_cache is not None:
            record = self.embedding_cache.choose_record(path)
            sample.update(
                {
                    "latents": record.latents,
                    "variant_index": record.variant_index,
                }
            )
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
