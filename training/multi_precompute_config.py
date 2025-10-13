"""Backward compatible shim re-exporting merged multi-precompute configuration helpers."""

from __future__ import annotations

from .precompute_embeddings import (
    MultiPrecomputeConfig,
    MultiPrecomputeDefaults,
    MultiPrecomputeTask,
)

__all__ = [
    "MultiPrecomputeConfig",
    "MultiPrecomputeDefaults",
    "MultiPrecomputeTask",
]
