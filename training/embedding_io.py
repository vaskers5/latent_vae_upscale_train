"""Shared utilities for encoding and storing latent embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

__all__ = ["TransformParams", "save_record", "CACHE_VERSION"]

CACHE_VERSION = 1


@dataclass(frozen=True)
class TransformParams:
    """Metadata describing how an image tensor was prepared for the model."""

    flip: bool
    crop_x: int
    crop_y: int

    def to_dict(self) -> Dict[str, Any]:
        return {"flip": self.flip, "crop_x": int(self.crop_x), "crop_y": int(self.crop_y)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformParams":
        return cls(
            flip=bool(data.get("flip", False)),
            crop_x=int(data.get("crop_x", 0)),
            crop_y=int(data.get("crop_y", 0)),
        )


def save_record(
    record_path: Path,
    *,
    latents: torch.Tensor,
    params: TransformParams,
    dataset_info: Dict[str, Any],
    mean: Optional[torch.Tensor] = None,
    logvar: Optional[torch.Tensor] = None,
) -> None:
    """Persist the embedding payload for a single image to disk."""

    record_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "version": CACHE_VERSION,
        "latents": latents.cpu(),
        "params": params.to_dict(),
        "variant": 0,
        "dataset": dataset_info,
        "storage_dtype": str(latents.dtype).replace("torch.", ""),
    }
    if mean is not None:
        payload["mean"] = mean.cpu()
    if logvar is not None:
        payload["logvar"] = logvar.cpu()
    torch.save(payload, record_path)
