"""Dataset utilities used by the VAE trainer."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from .embeddings import EmbeddingCache, TransformParams

__all__ = ["ImageFolderDataset"]


class ImageFolderDataset(Dataset):
    """Recursively reads images and returns random crops normalised to [-1, 1]."""

    def __init__(
        self,
        root: Path,
        high_resolution: int,
        resize_long_side: int = 0,
        limit: int = 0,
        horizontal_flip_prob: float = 0.0,
        embedding_cache: Optional[EmbeddingCache] = None,
    ) -> None:
        self.root = root
        self.high_resolution = high_resolution
        self.resize_long_side = resize_long_side
        self.horizontal_flip_prob = horizontal_flip_prob
        self.paths: List[Path] = []
        self.embedding_cache = embedding_cache

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
            return tensor

        record = self.embedding_cache.choose_record(path)
        tensor, params = self.build_tensor_sample(path, params=record.params)
        sample = {
            "image": tensor,
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
        if self.high_resolution <= 0:
            return img, TransformParams(flip=False, crop_x=0, crop_y=0)

        gen = rng if rng is not None else random

        img = self._resize_if_needed(img)

        if params is None:
            flip = False
            if self.horizontal_flip_prob > 0 and gen.random() < self.horizontal_flip_prob:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                flip = True
            img, crop_x, crop_y = self._random_crop(img, gen)
            return img, TransformParams(flip=flip, crop_x=crop_x, crop_y=crop_y)

        if params.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        img = self._ensure_minimum_size(img)
        width, height = img.size
        crop_x = max(0, min(width - self.high_resolution, params.crop_x)) if width > self.high_resolution else 0
        crop_y = max(0, min(height - self.high_resolution, params.crop_y)) if height > self.high_resolution else 0
        img = img.crop((crop_x, crop_y, crop_x + self.high_resolution, crop_y + self.high_resolution))
        return img, TransformParams(flip=params.flip, crop_x=crop_x, crop_y=crop_y)

    def _ensure_minimum_size(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        if width < self.high_resolution or height < self.high_resolution:
            img = img.resize((max(width, self.high_resolution), max(height, self.high_resolution)), Image.LANCZOS)
        return img

    def _random_crop(self, img: Image.Image, gen: random.Random) -> Tuple[Image.Image, int, int]:
        img = self._ensure_minimum_size(img)
        width, height = img.size
        if width == self.high_resolution and height == self.high_resolution:
            return img, 0, 0
        crop_x = gen.randint(0, width - self.high_resolution)
        crop_y = gen.randint(0, height - self.high_resolution)
        img = img.crop((crop_x, crop_y, crop_x + self.high_resolution, crop_y + self.high_resolution))
        return img, crop_x, crop_y
