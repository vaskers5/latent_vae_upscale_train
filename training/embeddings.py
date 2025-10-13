"""Backward compatible shim after merging embedding utilities into a single module."""

from __future__ import annotations

from .precompute_embeddings import EmbeddingCache, TransformParams

__all__ = ["EmbeddingCache", "TransformParams"]
