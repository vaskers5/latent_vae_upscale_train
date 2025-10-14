"""Backward compatible shim after merging embedding utilities into a single module."""

from __future__ import annotations

from .embedding_io import CACHE_VERSION, TransformParams, save_record

__all__ = ["TransformParams", "save_record", "CACHE_VERSION"]
