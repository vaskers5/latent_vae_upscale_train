#!/usr/bin/env python3
"""
Fast local unpacker for `.tar.gz` shards produced by your packer.

- Source: local folder only (no HF download logic).
- Parallel extraction (one process per shard).
- Optional system `tar` (+ pigz if available) for multi-threaded decompression.
- Idempotent via per-archive .ok markers.
- Optional deletion of shards after successful extraction.

Example:
python unpack_local_shards.py \
  --source-dir ./workspace/d23_archives \
  --output-dir ./workspace/d23_unpacked \
  --workers 8 \
  --use-system-tar \
  --delete-archives-after
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Fast parallel unpacker for local .tar.gz shards")
    p.add_argument("--source-dir", type=Path, required=True, help="Directory containing *.tar.gz shards")
    p.add_argument("--output-dir", type=Path, required=True, help="Where to extract the dataset")
    p.add_argument("--archive-prefix", type=str, default="dataset_part",
                   help="Shard filename prefix (e.g., dataset_part_0001.tar.gz)")
    p.add_argument("--allow-pattern", type=str, default="*.tar.gz",
                   help="Glob pattern to select archives (applied recursively)")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1),
                   help="Number of parallel extractions")
    p.add_argument("--overwrite", action="store_true",
                   help="If set, delete output-dir before extraction")
    p.add_argument("--delete-archives-after", action="store_true",
                   help="Delete each shard after successful extraction")
    p.add_argument("--use-system-tar", action="store_true",
                   help="Prefer external `tar` (with pigz if present) over Python tarfile")
    p.add_argument("--verify", action="store_true",
                   help="Quickly test archive structure before extraction (slower)")
    return p.parse_args()


def ensure_output_dir(path: Path, overwrite: bool):
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def discover_archives(source_dir: Path, prefix: str, allow_pattern: str) -> List[Path]:
    # Search recursively; require ...tar.gz suffix and starting with prefix
    candidates = sorted(
        p for p in source_dir.rglob(allow_pattern)
        if p.is_file() and p.name.startswith(prefix) and p.suffixes[-2:] == [".tar", ".gz"]
    )
    if not candidates:
        raise FileNotFoundError(
            f"No archives found under {source_dir} (prefix='{prefix}', pattern='{allow_pattern}')"
        )
    return candidates


def system_has(cmd: Sequence[str]) -> bool:
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False


def _extract_with_system_tar(archive: Path, out_dir: Path) -> None:
    pigz = shutil.which("pigz")
    if pigz:
        cmd = ["tar", "--use-compress-program", pigz, "-x", "-f", str(archive), "-C", str(out_dir)]
    else:
        cmd = ["tar", "-xzf", str(archive), "-C", str(out_dir)]
    subprocess.run(cmd, check=True)


def _extract_with_python_tar(archive: Path, out_dir: Path) -> None:
    with tarfile.open(archive, mode="r:gz") as tar:
        tar.extractall(path=out_dir)


def _quick_verify(archive: Path) -> None:
    with tarfile.open(archive, mode="r:gz") as tar:
        _ = next(iter(tar), None)


def _marker_path(archive: Path, out_dir: Path) -> Path:
    mark = out_dir / ".unpack_markers" / f"{archive.name}.ok"
    mark.parent.mkdir(parents=True, exist_ok=True)
    return mark


def _needs_unpack(archive: Path, out_dir: Path) -> bool:
    return not _marker_path(archive, out_dir).exists()


def _mark_done(archive: Path, out_dir: Path) -> None:
    _marker_path(archive, out_dir).write_text("ok", encoding="utf-8")


def _extract_one(args: Tuple[Path, Path, bool, bool]) -> Tuple[str, bool, Optional[str]]:
    """
    Worker for ProcessPoolExecutor.
    Returns: (archive_name, success, error_message)
    """
    archive, out_dir, use_system_tar, do_verify = args
    try:
        if do_verify:
            _quick_verify(archive)

        if use_system_tar and system_has(["tar", "--version"]):
            _extract_with_system_tar(archive, out_dir)
        else:
            _extract_with_python_tar(archive, out_dir)

        _mark_done(archive, out_dir)
        return (archive.name, True, None)
    except Exception as e:
        return (archive.name, False, str(e))


def main():
    args = parse_args()

    source_dir = args.source_dir.expanduser().resolve()
    if not source_dir.exists():
        print(f"Source dir not found: {source_dir}", file=sys.stderr)
        sys.exit(2)

    ensure_output_dir(args.output_dir, overwrite=args.overwrite)
    (args.output_dir / ".unpack_markers").mkdir(parents=True, exist_ok=True)

    archives = discover_archives(source_dir, args.archive_prefix, args.allow_pattern)
    total_bytes = sum(p.stat().st_size for p in archives)
    print(f"Found {len(archives)} shards (~{total_bytes/1024/1024:.1f} MiB) in {source_dir}")

    # Skip already extracted shards
    todo = [a for a in archives if _needs_unpack(a, args.output_dir)]
    skipped = len(archives) - len(todo)
    if skipped:
        print(f"Skipping {skipped} shard(s) already unpacked (marker present).")
    if not todo:
        print("Nothing to do. All shards already unpacked.")
        return

    work_items = [(a, args.output_dir, args.use_system_tar, args.verify) for a in todo]  # noqa: E231

    results: List[Tuple[str, bool, Optional[str]]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_extract_one, w): w[0].name for w in work_items}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting shards", unit="shard"):
            results.append(fut.result())

    ok = [r for r in results if r[1]]
    bad = [r for r in results if not r[1]]
    print(f"Done. {len(ok)} OK, {len(bad)} failed.")
    if bad:
        for name, _, err in bad:
            print(f" - {name}: {err}", file=sys.stderr)

    if args.delete_archives_after:
        deleted = 0
        ok_names = {name for name, success, _ in ok if success}
        for a in todo:
            if a.name in ok_names and a.exists():
                try:
                    a.unlink()
                    deleted += 1
                except Exception as e:
                    print(f"Warning: could not delete {a}: {e}", file=sys.stderr)
        if deleted:
            print(f"Deleted {deleted} shard(s) after successful extraction.")

    print("Unpack complete.")


if __name__ == "__main__":
    main()
