"""Dataset utilities used by the VAE trainer."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable, List

from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

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
    ) -> None:
        self.root = root
        self.high_resolution = high_resolution
        self.resize_long_side = resize_long_side
        self.horizontal_flip_prob = horizontal_flip_prob
        self.paths: List[Path] = []

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

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.paths[index % len(self.paths)]
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = self._resize_if_needed(img)
            img = self._maybe_flip(img)
            img = self._crop_to_resolution(img)
            tensor = TF.to_tensor(img)
            tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            return tensor

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

    def _maybe_flip(self, img: Image.Image) -> Image.Image:
        if self.horizontal_flip_prob > 0 and random.random() < self.horizontal_flip_prob:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def _crop_to_resolution(self, img: Image.Image) -> Image.Image:
        if self.high_resolution <= 0:
            return img
        width, height = img.size
        if width < self.high_resolution or height < self.high_resolution:
            img = img.resize((max(width, self.high_resolution), max(height, self.high_resolution)), Image.LANCZOS)
            width, height = img.size
        if width == self.high_resolution and height == self.high_resolution:
            return img
        x = random.randint(0, width - self.high_resolution)
        y = random.randint(0, height - self.high_resolution)
        return img.crop((x, y, x + self.high_resolution, y + self.high_resolution))
