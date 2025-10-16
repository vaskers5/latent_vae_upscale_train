"""BasicSR dataset that exposes latent caches without relying on training helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import random

random.seed(42)

import torch
from torch.utils import data as data
from tqdm.auto import tqdm

from basicsr.utils.registry import DATASET_REGISTRY


@dataclass(frozen=True)
class _CacheSample:
    """Represents a matched low/high latent pair within a cache."""

    lq_path: Path
    gt_path: Path
    vae_name: str


@DATASET_REGISTRY.register()
class LatentCacheDataset(data.Dataset):
    """Expose cached latent pairs to the BasicSR dataloader infrastructure."""

    def __init__(self, opt: Dict[str, Any]):  # type: ignore[override]
        print("Initializing LatentCacheDataset")
        super().__init__()
        self.opt = dict(opt)

        self.scale = opt["scale"]
        self.phase = opt["phase"]

        # Normalization parameters
        self.mean = opt.get("mean", None)
        self.std = opt.get("std", None)
        self.number_of_samples = opt.get("number_of_samples", None)

        cache_dirs = self._normalize_cache_dirs(self.opt.get("cache_dirs", []))
        print(f"LatentCacheDataset using cache_dirs: {[str(path) for path in cache_dirs]}")

        if not cache_dirs:
            raise RuntimeError("No cache directories supplied to LatentCacheDataset.")

        self._low_res = int(self.opt.get("low_res"))
        self._high_res = int(self.opt.get("high_res"))
        self._vae_names: Sequence[str] = tuple(self.opt.get("vae_names") or [])

        self._samples: List[_CacheSample] = []
        self._gather_samples(cache_dirs)
        if not self._samples:
            raise RuntimeError("No latent pairs found in the provided cache directories.")

    @staticmethod
    def _normalize_cache_dirs(cache_dirs: Any) -> List[Path]:
        if isinstance(cache_dirs, (str, Path)):
            return [Path(cache_dirs)]
        normalized: List[Path] = []
        for entry in cache_dirs or []:
            normalized.append(Path(entry))
        return normalized

    def _gather_samples(self, cache_dirs: Sequence[Path]) -> None:
        tasks: List[Tuple[Path, str]] = []
        for idx, cache_dir in enumerate(cache_dirs):
            resolved_dir = cache_dir.expanduser().resolve()
            vae_name = self._vae_names[idx] if idx < len(self._vae_names) else str(resolved_dir)
            tasks.append((resolved_dir, vae_name))

        if not tasks:
            return

        if len(tasks) == 1:
            cache_dir, vae_name = tasks[0]
            self._samples.extend(self._collect_pairs(cache_dir, vae_name))
            return

        with tqdm(total=len(tasks), desc="Loading latent caches", unit="cache") as progress:
            for cache_dir, vae_name in tasks:
                samples = self._collect_pairs(cache_dir, vae_name)
                self._samples.extend(samples)
                progress.update(1)

    def _collect_pairs(self, cache_dir: Path, vae_name: str) -> List[_CacheSample]:
        low_dir = cache_dir / f"{self._low_res}px"
        high_dir = cache_dir / f"{self._high_res}px"

        if not low_dir.is_dir():
            raise FileNotFoundError(f"Low-resolution cache directory missing: {low_dir}")
        if not high_dir.is_dir():
            raise FileNotFoundError(f"High-resolution cache directory missing: {high_dir}")

        low_filenames = self._scan_pt_filenames(low_dir)
        high_filenames = self._scan_pt_filenames(high_dir)

        if len(low_filenames) <= len(high_filenames):
            high_lookup = set(high_filenames)
            matched_names = [name for name in low_filenames if name in high_lookup]
        else:
            low_lookup = set(low_filenames)
            matched_names = [name for name in high_filenames if name in low_lookup]

        matched_pairs = [
            _CacheSample(lq_path=low_dir / name, gt_path=high_dir / name, vae_name=vae_name)
            for name in matched_names
        ]
        matched_pairs = random.sample(matched_pairs, min(len(matched_pairs), self.number_of_samples)) if self.number_of_samples else matched_pairs
        print(
            f"LatentCacheDataset matched {len(matched_pairs)} pairs in '{cache_dir}' "
            f"(low_res={self._low_res}, high_res={self._high_res})"
        )
        return matched_pairs

    @staticmethod
    def _scan_pt_filenames(directory: Path) -> List[str]:
        entries = os.scandir(directory)
        try:
            filenames: List[str] = []
            for entry in entries:
                if entry.is_file() and entry.name.endswith(".pt"):
                    filenames.append(entry.name)
            return filenames
        finally:
            entries.close()

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        sample = self._samples[idx]
        # Load latent tensors from .pt files
        lq_data = torch.load(sample.lq_path, map_location="cpu")
        gt_data = torch.load(sample.gt_path, map_location="cpu")

        # Extract latents from the data dictionary
        if isinstance(lq_data, dict) and "latents" in lq_data:
            img_lq = lq_data["latents"]
        else:
            img_lq = lq_data

        if isinstance(gt_data, dict) and "latents" in gt_data:
            img_gt = gt_data["latents"]
        else:
            img_gt = gt_data

        # Ensure tensors are float32
        img_lq = img_lq.float()
        img_gt = img_gt.float()

        # # For training, we can apply some basic augmentations
        # if self.phase == "train":
        #     # Random horizontal flip (applied to both LQ and GT)
        #     if torch.rand(1) < 0.5:
        #         img_lq = torch.flip(img_lq, dims=[2])  # flip width dimension
        #         img_gt = torch.flip(img_gt, dims=[2])

        #     # Random vertical flip (applied to both LQ and GT)
        #     if torch.rand(1) < 0.5:
        #         img_lq = torch.flip(img_lq, dims=[1])  # flip height dimension
        #         img_gt = torch.flip(img_gt, dims=[1])

        # # Normalize if specified
        # if self.mean is not None and self.std is not None:
        #     # Apply normalization channel-wise
        #     for c in range(img_lq.shape[0]):
        #         img_lq[c] = (img_lq[c] - self.mean[c]) / self.std[c]
        #     for c in range(img_gt.shape[0]):
        #         img_gt[c] = (img_gt[c] - self.mean[c]) / self.std[c]
        return {
            "lq": img_lq,
            "gt": img_gt,
            "lq_path": str(sample.lq_path),
            "gt_path": str(sample.gt_path),
            "vae_name": sample.vae_name,
        }
