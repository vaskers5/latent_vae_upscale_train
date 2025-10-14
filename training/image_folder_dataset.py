"""Dataset helpers for iterating over image folders."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from tqdm import tqdm

__all__ = ["ImageFolderDataset"]


class ImageFolderDataset(Dataset):
    """Recursively reads images and returns centred crops normalised to [-1, 1]."""

    def __init__(
        self,
        root: Path,
    ) -> None:
        self.root = os.path.abspath(root)
        self.paths = self._collect_paths()
        if not self.paths:
            raise RuntimeError(f"No valid images found under '{self.root}'")

    def _collect_paths(self) -> List[Path]:
        exts = {"png", "jpg", "jpeg", "webp"}
        all_images = [os.path.join(self.root, p) for p in os.listdir(self.root) if p.split(".")[-1] in exts]
        return all_images

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        tensor = self.build_tensor_sample(path)
        sample = {
            "model_input": tensor,
            "path": str(path),
        }
        return sample

    def build_tensor_sample(
        self,
        path: Path,
    ) -> Tuple[torch.Tensor]:
        with Image.open(path) as img:
            img = img.convert("RGB")
        tensor = TF.to_tensor(img)
        tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return tensor
