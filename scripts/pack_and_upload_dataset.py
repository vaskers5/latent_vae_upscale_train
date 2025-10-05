#!/usr/bin/env python3
"""Pack a large image dataset into compressed archives and upload to Hugging Face.

The script walks a dataset directory, batches files into `.tar.gz` archives with
maximum compression (lossless for pre-compressed image formats), and uploads the
resulting archives to a Hugging Face dataset repository.

Example usage:

    python pack_and_upload_dataset.py \
        --dataset-root ./workspace/d23/d23 \
        --output-dir ./workspace/d23_archives \
        --repo-id vaskers5/latent_vae_upscale_ds \
        --max-files-per-archive 50000 \
        --max-archive-bytes 3221225472 \
        --commit-message "Add compressed dataset shards"

Environment variables:
    HF_TOKEN (optional) - fallback Hugging Face token if --hf-token is omitted
"""

import argparse
import os
import shutil
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, NamedTuple, Optional, Sequence

from huggingface_hub import HfApi, HfFolder
from tqdm import tqdm


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".gif",
    ".tiff",
    ".tif",
}


class ArchivingConfig(NamedTuple):
    dataset_root: Path
    output_dir: Path
    repo_id: str
    repo_revision: str
    commit_message: str
    archive_prefix: str
    max_files_per_archive: int
    max_archive_bytes: Optional[int]
    compress_level: int
    include_extensions: Sequence[str]
    follow_symlinks: bool
    overwrite_output: bool
    make_private: bool
    hf_token: str


def parse_args() -> ArchivingConfig:
    parser = argparse.ArgumentParser(description="Pack dataset shards and upload to Hugging Face")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("./workspace/d23/d23"),
        help="Root directory containing the full dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./workspace/d23_archives"),
        help="Destination directory where tar.gz archives will be written",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face dataset repository id (e.g. username/dataset-name)",
    )
    parser.add_argument(
        "--repo-revision",
        default="main",
        help="Target branch or tag on the Hugging Face repo (default: main)",
    )
    parser.add_argument(
        "--commit-message",
        default="Add compressed dataset archives",
        help="Commit message used for the upload",
    )
    parser.add_argument(
        "--archive-prefix",
        default="dataset_part",
        help="Filename prefix for generated archives",
    )
    parser.add_argument(
        "--max-files-per-archive",
        type=int,
        default=50_000,
        help="Maximum number of files packed into a single archive",
    )
    parser.add_argument(
        "--max-archive-bytes",
        type=int,
        default=3 * 1024 * 1024 * 1024,
        help="Maximum (approximate) uncompressed bytes per archive; set 0 to disable",
    )
    parser.add_argument(
        "--compress-level",
        type=int,
        choices=range(1, 10),
        default=9,
        help="gzip compression level (1=fastest, 9=smallest)",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=None,
        help="Optional list of file extensions to include (defaults to common image types)",
    )
    parser.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="Follow symbolic links when discovering files",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete existing output directory before writing archives",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hugging Face dataset repository as private",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Explicit Hugging Face token (otherwise uses HF_TOKEN or cached login)",
    )

    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root {dataset_root} not found or is not a directory")

    extensions = [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in (args.extensions or IMAGE_EXTENSIONS)]

    token = (
        args.hf_token
        or os.getenv("HF_TOKEN")
        or HfFolder().get_token()
    )

    if not token:
        raise RuntimeError("No Hugging Face token provided. Use --hf-token or set HF_TOKEN.")

    max_archive_bytes = args.max_archive_bytes if args.max_archive_bytes > 0 else None

    return ArchivingConfig(
        dataset_root=dataset_root,
        output_dir=output_dir,
        repo_id=args.repo_id,
        repo_revision=args.repo_revision,
        commit_message=args.commit_message,
        archive_prefix=args.archive_prefix,
        max_files_per_archive=args.max_files_per_archive,
        max_archive_bytes=max_archive_bytes,
        compress_level=args.compress_level,
        include_extensions=extensions,
        follow_symlinks=args.follow_symlinks,
        overwrite_output=args.overwrite_output,
        make_private=args.private,
        hf_token=token,
    )


def discover_files(root: Path, extensions: Sequence[str], follow_symlinks: bool) -> List[Path]:
    candidates: List[Path] = []
    for path in root.rglob("*"):
        if not follow_symlinks and path.is_symlink():
            continue
        if path.is_file() and path.suffix.lower() in extensions:
            candidates.append(path)
    if not candidates:
        raise RuntimeError(f"No files with extensions {extensions} found under {root}")
    return candidates


def ensure_output_dir(config: ArchivingConfig) -> None:
    if config.output_dir.exists():
        if not config.overwrite_output and any(config.output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory {config.output_dir} exists and is not empty."
                " Use --overwrite-output to replace it."
            )
        if config.overwrite_output:
            shutil.rmtree(config.output_dir)
            config.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        config.output_dir.mkdir(parents=True, exist_ok=True)


def chunk_files(
    files: Sequence[Path],
    max_files: int,
    max_bytes: Optional[int],
) -> Iterable[Sequence[Path]]:
    if max_files <= 0:
        raise ValueError("max_files_per_archive must be positive")

    current: List[Path] = []
    current_bytes = 0

    for file_path in files:
        file_size = file_path.stat().st_size
        if current and (
            len(current) >= max_files
            or (max_bytes is not None and current_bytes + file_size > max_bytes)
        ):
            yield current
            current = []
            current_bytes = 0

        current.append(file_path)
        current_bytes += file_size

    if current:
        yield current


def _create_archive(dataset_root: str, archive_path: str, compress_level: int, batch: Sequence[str]) -> str:
    root_path = Path(dataset_root)
    with tarfile.open(archive_path, mode="w:gz", compresslevel=compress_level) as tar:
        for src in batch:
            src_path = Path(src)
            arcname = src_path.relative_to(root_path)
            tar.add(str(src_path), arcname=arcname.as_posix())
    return archive_path


def create_archives(config: ArchivingConfig, files: Sequence[Path]) -> List[Path]:
    archives: List[Path] = []
    file_progress = tqdm(total=len(files), desc="Archiving", unit="file")

    with ProcessPoolExecutor() as executor:
        futures: dict = {}
        for index, batch in enumerate(
            chunk_files(files, config.max_files_per_archive, config.max_archive_bytes),
            start=1,
        ):
            archive_path = config.output_dir / f"{config.archive_prefix}_{index:04d}.tar.gz"
            archives.append(archive_path)
            futures[
                executor.submit(
                    _create_archive,
                    str(config.dataset_root),
                    str(archive_path),
                    config.compress_level,
                    [str(path) for path in batch],
                )
            ] = len(batch)

        for future in as_completed(futures):
            file_progress.update(futures[future])
            future.result()

    file_progress.close()
    return archives


def upload_archives(config: ArchivingConfig, archives_dir: Path) -> None:
    api = HfApi(token=config.hf_token)
    api.create_repo(repo_id=config.repo_id, repo_type="dataset", private=config.make_private, exist_ok=True)

    tqdm.write("Uploading archives to Hugging Face…")
    api.upload_folder(
        folder_path=str(archives_dir),
        repo_id=config.repo_id,
        repo_type="dataset",
        revision=config.repo_revision,
        token=config.hf_token,
        commit_message=config.commit_message,
        allow_patterns=["*.tar.gz"],
    )


def main() -> None:
    config = parse_args()

    print(f"Dataset root: {config.dataset_root}")
    print(f"Output directory: {config.output_dir}")
    print(f"Target repository: {config.repo_id} ({'private' if config.make_private else 'public'})")

    ensure_output_dir(config)

    files = discover_files(config.dataset_root, config.include_extensions, config.follow_symlinks)
    print(f"Discovered {len(files)} files to archive.")

    archives = create_archives(config, files)
    print(f"Created {len(archives)} archives:")
    for archive in archives:
        size_mb = archive.stat().st_size / (1024 * 1024)
        print(f" - {archive.name} ({size_mb:.2f} MiB)")

    upload_archives(config, config.output_dir)
    print("Upload completed successfully.")


if __name__ == "__main__":
    main()