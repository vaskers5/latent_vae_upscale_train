#!/usr/bin/env python3
"""Deprecated shim kept for backward compatibility."""

from __future__ import annotations

import sys


def main() -> None:  # pragma: no cover - thin shim
    raise SystemExit(
        "connect_embeddings_with_images.py has been merged into create_test_dataset.py. "
        "Please invoke the unified CLI instead."
    )


if __name__ == "__main__":
    main()