#!/usr/bin/env python3
"""List image files in a dataset along with their file size and resolution."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import chain, repeat
from pathlib import Path
from typing import Iterator, Sequence

import cv2
from tqdm.auto import tqdm

IMAGE_EXTENSIONS: Sequence[str] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".webp",
    ".tiff",
)

MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024
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
) -> tuple[bool, str, int | None, int | None, int | None, str | None, bool]:
    path = Path(path_str)
    root = Path(root_str)
    display_path = format_display_path(path, root, relative)

    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        if fail_on_error:
            raise
        return (False, display_path, None, None, None, f"Could not read file size for {path}: {exc}", False)

    if size_bytes > size_limit:
        return (True, display_path, size_bytes, None, None, None, True)

    try:
        image = cv2.imread(path_str, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("OpenCV could not read the image data.")
        height, width = image.shape[:2]
        return (True, display_path, size_bytes, int(width), int(height), None, False)
    except Exception as exc:  # pragma: no cover - guard rail
        if fail_on_error:
            raise
        return (
            True,
            display_path,
            size_bytes,
            None,
            None,
            f"Could not read image {path}: {exc}",
            False,
        )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Walk a dataset directory and output image file paths, file sizes, and resolutions. "
            "All three stages (collect, size check, validate) are executed sequentially. "
            "Skips decoding images larger than 10 MB."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="Path to the root of the dataset containing image files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help=(
            "Optional path to a CSV file to save the combined results. Defaults to stdout."
        ),
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
        choices=None,
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
        help="Suppress warnings about unreadable images.",
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
        help=(
            "Maximum file size in bytes to decode image resolutions. Files above this limit "
            "will be reported but skipped for decoding. Defaults to 10485760 (10 MB)."
        ),
    )

    args = parser.parse_args(argv)
    if args.extensions:
        args.extensions = tuple(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.extensions)
    else:
        args.extensions = tuple(IMAGE_EXTENSIONS)
    if args.size_limit <= 0:
        parser.error("--size-limit must be a positive integer")
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
        progress_bar = None
        iterable = paths
        try:
            if show_progress:
                progress_bar = tqdm(
                    paths,
                    total=len(paths),
                    unit="image",
                    desc="Collecting image paths",
                    leave=False,
                    disable=False,
                )
                iterable = progress_bar
            for path in iterable:
                yield (format_display_path(path, root, relative),)
        finally:
            if progress_bar is not None:
                progress_bar.close()

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
        progress_bar = None
        iterable = paths
        try:
            if show_progress:
                progress_bar = tqdm(
                    paths,
                    total=len(paths),
                    unit="image",
                    desc="Checking image sizes",
                    leave=False,
                    disable=False,
                )
                iterable = progress_bar
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
                if over_limit:
                    logger.info(
                        "File %s is %.2f MB (> %.2f MB limit)",
                        display_path,
                        size_bytes / (1024 * 1024),
                        size_limit / (1024 * 1024),
                    )
                yield (display_path, size_bytes, over_limit)
        finally:
            if progress_bar is not None:
                progress_bar.close()

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
            chunksize=32,
        )

        progress_bar = None
        if show_progress:
            progress_bar = tqdm(
                results_iter,
                total=len(paths),
                unit="image",
                desc="Processing images",
                leave=False,
                disable=False,
            )
            iterator = progress_bar
        else:
            iterator = results_iter

        try:
            for (
                success,
                display_path,
                size_bytes,
                width,
                height,
                error_message,
                skipped_for_size,
            ) in iterator:
                if skipped_for_size and size_bytes is not None:
                    logger.info(
                        "Skipping resolution read for %s (%.2f MB exceeds %.2f MB limit)",
                        display_path,
                        size_bytes / (1024 * 1024),
                        size_limit / (1024 * 1024),
                    )
                if error_message:
                    logger.warning("%s", error_message)
                if not success:
                    continue
                yield (display_path, size_bytes, width, height)
        finally:
            if progress_bar is not None:
                progress_bar.close()


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
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO)
    logger = logging.getLogger("image_sizes")

    try:
        show_progress = not args.no_progress and not args.quiet and sys.stderr.isatty()

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
