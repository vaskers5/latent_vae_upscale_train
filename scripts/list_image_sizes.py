#!/usr/bin/env python3
"""
List image files in a dataset, validate them against size and pixel limits,
and optionally delete offending files.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import chain, repeat
from pathlib import Path
from typing import Iterator, Sequence

from PIL import Image
from tqdm.auto import tqdm

# Disable Pillow's own limit since we are implementing our own check.
Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTENSIONS: Sequence[str] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".webp",
    ".tiff",
)

# Default limit of 10 MB for file size
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024
# Default pixel limit, based on Pillow's default DecompressionBombWarning
# 93438718 pixels is ~9.6k x 9.6k resolution
MAX_IMAGE_PIXELS = 4000000
COMBINED_HEADER: Sequence[str] = ("stage", "path", "size_bytes", "width", "height", "over_limit")


def format_display_path(path: Path, root: Path, relative: bool) -> str:
    if relative:
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)
    return str(path.resolve())


def resolve_image_paths(root: Path, extensions: Sequence[str]) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in extensions
    ]


def _process_image(
    path_str: str,
    root_str: str,
    relative: bool,
    fail_on_error: bool,
    size_limit: int,
    pixel_limit: int,  # NEW
    delete: bool,      # NEW
) -> tuple[bool, str, int | None, int | None, int | None, str | None, bool]:
    path = Path(path_str)
    root = Path(root_str)
    display_path = format_display_path(path, root, relative)
    width, height = None, None

    # 1. Check file size first (it's cheap)
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        if fail_on_error:
            raise
        return (False, display_path, None, None, None, f"Could not read file size: {exc}", False)

    if size_bytes > size_limit:
        reason = f"File size {size_bytes} exceeds limit of {size_limit}"
        if delete:
            try:
                path.unlink()
                return (False, display_path, size_bytes, None, None, f"DELETED: {reason}", True)
            except OSError as exc:
                return (False, display_path, size_bytes, None, None, f"DELETE FAILED ({reason}): {exc}", True)
        # Return as skipped if not deleting
        return (True, display_path, size_bytes, None, None, None, True)

    # 2. Check image content (pixels, corruption)
    try:
        # Use 'with' to ensure the file handle is closed before a potential delete operation
        with Image.open(path) as image:
            width, height = image.size
            num_pixels = width * height

            if num_pixels > pixel_limit:
                reason = f"Pixel count {num_pixels} exceeds limit of {pixel_limit}"
                # The 'with' block will close the image, then we can delete below.
            else:
                image.convert("RGB")  # Fully load the image to check for internal errors
                return (True, display_path, size_bytes, int(width), int(height), None, False)

    except Exception as exc:  # Catches corrupt images, etc.
        reason = f"Could not read image: {exc}"
        if delete:
            try:
                path.unlink(missing_ok=True)
                return (False, display_path, size_bytes, None, None, f"DELETED corrupt file: {reason}", False)
            except OSError as unlink_exc:
                return (False, display_path, size_bytes, None, None, f"DELETE FAILED (corrupt): {unlink_exc}", False)
        # If not deleting, just report the error
        if fail_on_error:
            raise
        return (True, display_path, size_bytes, None, None, f"Error processing {path}: {exc}", False)

    # 3. This part is reached only if pixel count was too high. 'reason' is set.
    if delete:
        try:
            path.unlink()
            return (False, display_path, size_bytes, width, height, f"DELETED: {reason}", False)
        except OSError as exc:
            return (False, display_path, size_bytes, width, height, f"DELETE FAILED ({reason}): {exc}", False)

    # If not deleting, report as an invalid file with a reason
    return (True, display_path, size_bytes, width, height, f"INVALID: {reason}", False)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Walk a dataset directory, validate images, and optionally delete invalid ones."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Path to the root of the dataset containing image files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Optional path to a CSV file to save the combined results. Defaults to stdout.",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Output paths relative to the dataset root instead of absolute paths.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Override the list of file extensions to consider as images. "
            "Provide one or more extensions (e.g. .png .jpg)."
        ),
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Stop processing if an image cannot be opened.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress warnings and info messages about individual images.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar output.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes to use. Defaults to the number of CPU cores.",
    )
    parser.add_argument(
        "--size-limit",
        type=int,
        default=MAX_IMAGE_SIZE_BYTES,
        help=f"Max file size in bytes to process. Defaults to {MAX_IMAGE_SIZE_BYTES} (10 MB).",
    )
    # NEW ARGUMENTS
    parser.add_argument(
        "--pixel-limit",
        type=int,
        default=MAX_IMAGE_PIXELS,
        help=f"Max number of pixels (width * height). Defaults to {MAX_IMAGE_PIXELS}.",
    )
    parser.add_argument(
        "--delete-offending",
        action="store_true",
        help="WARNING: Permanently deletes images that are corrupt or exceed size/pixel limits.",
    )

    args = parser.parse_args(argv)
    if args.extensions:
        args.extensions = tuple(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.extensions)
    else:
        args.extensions = tuple(IMAGE_EXTENSIONS)
    if args.size_limit <= 0:
        parser.error("--size-limit must be a positive integer")
    if args.pixel_limit <= 0: # NEW
        parser.error("--pixel-limit must be a positive integer")
    return args


def run_stage_collect(
    root: Path,
    extensions: Sequence[str],
    relative: bool,
    logger: logging.Logger,
    show_progress: bool,
) -> tuple[tuple[str, ...], Iterator[tuple[str]]]:
    paths = resolve_image_paths(root, extensions)
    logger.info("Found %d image files matching %s", len(paths), ", ".join(extensions))

    def rows() -> Iterator[tuple[str]]:
        iterable = tqdm(
            paths,
            total=len(paths),
            unit="image",
            desc="Collecting image paths",
            disable=not show_progress,
        )
        for path in iterable:
            yield (format_display_path(path, root, relative),)

    return (("path",), rows())


def run_stage_sizes(
    root: Path,
    extensions: Sequence[str],
    relative: bool,
    size_limit: int,
    fail_on_error: bool,
    logger: logging.Logger,
    show_progress: bool,
) -> tuple[tuple[str, ...], Iterator[tuple[str, int, bool]]]:
    paths = resolve_image_paths(root, extensions)

    def rows() -> Iterator[tuple[str, int, bool]]:
        iterable = tqdm(
            paths,
            total=len(paths),
            unit="image",
            desc="Checking image sizes",
            disable=not show_progress,
        )
        for path in iterable:
            display_path = format_display_path(path, root, relative)
            try:
                size_bytes = path.stat().st_size
            except OSError as exc:
                message = f"Could not read file size for {path}: {exc}"
                if fail_on_error:
                    raise
                logger.warning("%s", message)
                continue

            over_limit = size_bytes > size_limit
            yield (display_path, size_bytes, over_limit)

    return (("path", "size_bytes", "over_limit"), rows())


def run_stage_validate(
    root: Path,
    extensions: Sequence[str],
    relative: bool,
    fail_on_error: bool,
    logger: logging.Logger,
    max_workers: int | None,
    show_progress: bool,
    size_limit: int,
    pixel_limit: int, # NEW
    delete: bool,     # NEW
) -> tuple[tuple[str, ...], Iterator[tuple[str, int, int | None, int | None]]]:
    return (
        ("path", "size_bytes", "width", "height"),
        iter_images(
            root=root,
            extensions=extensions,
            relative=relative,
            fail_on_error=fail_on_error,
            logger=logger,
            max_workers=max_workers,
            show_progress=show_progress,
            size_limit=size_limit,
            pixel_limit=pixel_limit, # NEW
            delete=delete,           # NEW
        ),
    )


def normalized_stage_rows(stage: str, rows: Iterator[tuple]) -> Iterator[tuple[object, ...]]:
    if stage == "collect":
        for (path,) in rows:
            yield (stage, path, None, None, None, None)
    elif stage == "sizes":
        for (path, size_bytes, over_limit) in rows:
            yield (stage, path, size_bytes, None, None, over_limit)
    elif stage == "validate":
        for (path, size_bytes, width, height) in rows:
            yield (stage, path, size_bytes, width, height, None)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown stage {stage}")


def iter_images(
    root: Path,
    extensions: Sequence[str],
    relative: bool,
    fail_on_error: bool,
    logger: logging.Logger,
    max_workers: int | None,
    show_progress: bool,
    size_limit: int = MAX_IMAGE_SIZE_BYTES,
    pixel_limit: int = MAX_IMAGE_PIXELS, # NEW
    delete: bool = False,                # NEW
) -> Iterator[tuple[str, int, int | None, int | None]]:
    paths = resolve_image_paths(root, extensions)
    if not paths:
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_iter = executor.map(
            _process_image,
            (str(path) for path in paths),
            repeat(str(root)),
            repeat(relative),
            repeat(fail_on_error),
            repeat(size_limit),
            repeat(pixel_limit), # NEW
            repeat(delete),      # NEW
            chunksize=32,
        )

        iterator = tqdm(
            results_iter,
            total=len(paths),
            unit="image",
            desc="Processing images",
            disable=not show_progress,
        )

        for (
            success,
            display_path,
            size_bytes,
            width,
            height,
            error_message,
            skipped_for_size,
        ) in iterator:
            if skipped_for_size and size_bytes is not None and not delete: # Don't log skips if we're deleting them
                logger.info(
                    "Skipping resolution read for %s (%.2f MB > %.2f MB limit)",
                    display_path,
                    size_bytes / (1024 * 1024),
                    size_limit / (1024 * 1024),
                )
            if error_message:
                logger.info("%s", error_message) # Changed to info to show DELETED messages
            if not success:
                continue
            yield (display_path, size_bytes, width, height)


def write_csv(
    rows: Iterator[tuple[object, ...]], header: Sequence[str], file_path: Path | None
) -> None:
    if file_path is None:
        writer = csv.writer(sys.stdout)
        if header:
            writer.writerow(header)
        writer.writerows(rows)
    else:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if header:
                writer.writerow(header)
            writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")
    logger = logging.getLogger("image_validator")

    # NEW: Safety confirmation prompt before deleting files
    if args.delete_offending:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=sys.stderr)
        print("!!! WARNING: --delete-offending is enabled.                !!!", file=sys.stderr)
        print("!!! This will PERMANENTLY DELETE images from your disk.    !!!", file=sys.stderr)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", file=sys.stderr)
        try:
            response = input(" > Type 'yes' to proceed with deletion: ")
            if response.lower() != 'yes':
                print("Aborted by user.", file=sys.stderr)
                return 1
        except (KeyboardInterrupt, EOFError):
            print("\nAborted by user.", file=sys.stderr)
            return 1

    try:
        show_progress = not args.no_progress and not args.quiet and sys.stderr.isatty()

        # NOTE: The original script's logic of running three separate stages is inefficient
        # as it walks the directory tree multiple times. This structure is preserved here,
        # but a more optimized script might combine these into a single pass.

        _, collect_rows = run_stage_collect(
            root=args.root,
            extensions=args.extensions,
            relative=args.relative,
            logger=logger,
            show_progress=show_progress,
        )
        _, sizes_rows = run_stage_sizes(
            root=args.root,
            extensions=args.extensions,
            relative=args.relative,
            size_limit=args.size_limit,
            fail_on_error=args.fail_on_error,
            logger=logger,
            show_progress=show_progress,
        )
        _, validate_rows = run_stage_validate(
            root=args.root,
            extensions=args.extensions,
            relative=args.relative,
            fail_on_error=args.fail_on_error,
            logger=logger,
            max_workers=args.workers,
            show_progress=show_progress,
            size_limit=args.size_limit,
            pixel_limit=args.pixel_limit,       # NEW
            delete=args.delete_offending,       # NEW
        )

        rows = chain(
            normalized_stage_rows("collect", collect_rows),
            normalized_stage_rows("sizes", sizes_rows),
            normalized_stage_rows("validate", validate_rows),
        )

        write_csv(rows, COMBINED_HEADER, args.output)
    except (FileNotFoundError, NotADirectoryError) as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - guard rail
        logger.exception("Unexpected error: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())