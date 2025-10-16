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
    def __init__(
        self,
        paths: List[Path],
    ) -> None:
        self.paths = paths

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
