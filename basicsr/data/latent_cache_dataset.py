"""BasicSR dataset that wraps the project's latent cache datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch.utils import data as data
from tqdm.auto import tqdm

from basicsr.utils.registry import DATASET_REGISTRY
from training.dataset import UpscaleDataset


@dataclass
class _DatasetEntry:
    dataset: UpscaleDataset
    vae_name: str


@DATASET_REGISTRY.register()
class LatentCacheDataset(data.Dataset):
    """Expose cached latent pairs to the BasicSR dataloader infrastructure."""

    def __init__(self, opt: Dict[str, Any]):  # type: ignore[override]
        super().__init__()
        self.opt = dict(opt)
        cache_dirs: Sequence[str] = self.opt.get("cache_dirs", [])
        if isinstance(cache_dirs, str):
            cache_dirs = [cache_dirs]
        if not cache_dirs:
            raise ValueError("LatentCacheDataset requires at least one cache directory.")

        low_res = int(self.opt.get("low_res"))
        high_res = int(self.opt.get("high_res"))
        csv_path = self.opt.get("csv_path")
        vae_names: Sequence[str] = self.opt.get("vae_names") or []
        if vae_names and len(vae_names) != len(cache_dirs):
            raise ValueError("vae_names length must match cache_dirs length if provided.")

        self._datasets: List[_DatasetEntry] = []
        self._index: List[Tuple[int, int]] = []
        for idx, cache_dir in enumerate(tqdm(cache_dirs, desc="Loading latent caches", unit="cache")):
            vae_name = vae_names[idx] if vae_names else str(cache_dir)
            dataset = UpscaleDataset(cache_dir=cache_dir, low_res=low_res, high_res=high_res, csv_path=csv_path)
            entry_index = len(self._datasets)
            self._datasets.append(_DatasetEntry(dataset=dataset, vae_name=vae_name))
            self._index.extend((entry_index, sample_idx) for sample_idx in range(len(dataset)))

        if not self._index:
            raise RuntimeError("No latent pairs found in the provided cache directories.")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        dataset_idx, sample_idx = self._index[idx]
        entry = self._datasets[dataset_idx]
        sample = entry.dataset[sample_idx]

        lq = sample["low"].float()
        gt = sample["high"].float()
        lq_path = sample.get("low_path") or sample.get("lq_path") or ""
        gt_path = sample.get("high_path") or sample.get("gt_path") or ""

        return {
            "lq": lq,
            "gt": gt,
            "lq_path": lq_path,
            "gt_path": gt_path,
            "vae_name": entry.vae_name,
        }
