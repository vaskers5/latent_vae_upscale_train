"""Dataset utilities used by the VAE trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


__all__ = ["UpscaleDataset"]


class UpscaleDataset(Dataset):
    """Dataset that loads precomputed low/high resolution tensor pairs."""

    def __init__(self, cache_dir: str, low_res: int, high_res: int, csv_path: Optional[str] = None) -> None:
        print(
            f"[Dataset] Initializing UpscaleDataset with cache_dir={cache_dir}, low_res={low_res}, high_res={high_res}, csv_path={csv_path}"
        )
        self.pairs = []
        self._load_from_cache(cache_dir, low_res, high_res)
        print(f"[Dataset] Collected {len(self.pairs)} valid low/high pairs")

    def _load_from_cache(self, cache_dir: str, low_res: int, high_res: int) -> None:
        low_dir = Path(cache_dir) / f"{low_res}px"
        high_dir = Path(cache_dir) / f"{high_res}px"

        if not low_dir.is_dir():
            raise FileNotFoundError(f"Low-resolution cache directory missing: {low_dir}")
        if not high_dir.is_dir():
            raise FileNotFoundError(f"High-resolution cache directory missing: {high_dir}")

        low_res_files = list(
            tqdm(
                (p for p in low_dir.glob("*.pt")),
                desc="Loading low-res files",
                unit="file",
            )
        )
        print(f"[Dataset] Found {len(low_res_files)} low-res files")

        high_filenames = {
            p.name for p in high_dir.glob("*.pt")
        }

        self.pairs = [
            (low_path, high_dir / low_path.name)
            for low_path in low_res_files
            if low_path.name in high_filenames
        ]
        print(f"[Dataset] Validated {len(self.pairs)} low/high pairs")

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

        return {"lq": low_tensor, "gt": high_tensor, "lq_path": low_path, "gt_path": high_path}