"""Shared utilities for encoding and storing latent embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

__all__ = ["TransformParams", "save_record", "CACHE_VERSION"]

CACHE_VERSION = 1


def save_record(
    record_path: Path,
    *,
    latents: torch.Tensor,
    dataset_info: Dict[str, Any],
    mean: Optional[torch.Tensor] = None,
    logvar: Optional[torch.Tensor] = None,
) -> None:
    """Persist the embedding payload for a single image to disk."""

    record_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "version": CACHE_VERSION,
        "latents": latents.cpu(),
        "variant": 0,
        "dataset": dataset_info,
        "storage_dtype": str(latents.dtype).replace("torch.", ""),
    }
    if mean is not None:
        payload["mean"] = mean.cpu()
    if logvar is not None:
        payload["logvar"] = logvar.cpu()
    torch.save(payload, record_path)
