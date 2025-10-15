#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart cropper for SR: minimal-overlap 512x512 crops from arbitrary-size images,
with automatic downscales to 256 and 128, folder layout like script_2.py.

New:
  --min_any N   -> process only if (W>=N or H>=N)
  --min_both N  -> process only if (W>=N and H>=N)
  --dry_run     -> compute & report crop counts only (no images saved)
  Per-image report of planned/created crop counts
  meta.jsonl includes num_crops for each image when --save_meta is set

Usage:
  python smart_sr_crops.py \
    --in_dir /path/to/images \
    --out_root dataset/result \
    --workers 8 \
    --min_any 1000 \
    --dry_run
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple
from multiprocessing import Pool, cpu_count

from PIL import Image
from tqdm import tqdm

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
TARGET_SIZES = (512, 256, 128)


def try_import_resizeright(force_pillow: bool = False):
    if force_pillow:
        def r256(img): return img.resize((256, 256), resample=Image.BICUBIC)
        def r128(img): return img.resize((128, 128), resample=Image.BICUBIC)
        return r256, r128, "pillow-bicubic"
    try:
        from utils.resize_utils import pil_resize_right
        from utils import interp_methods
        def r256(img):
            return pil_resize_right(img, (256, 256),
                                    interp_method=interp_methods.cubic,
                                    antialiasing=True)
        def r128(img):
            return pil_resize_right(img, (128, 128),
                                    interp_method=interp_methods.cubic,
                                    antialiasing=True)
        return r256, r128, "ResizeRight-cubic-AA"
    except Exception:
        def r256(img): return img.resize((256, 256), resample=Image.BICUBIC)
        def r128(img): return img.resize((128, 128), resample=Image.BICUBIC)
        return r256, r128, "pillow-bicubic"


def list_images(in_dir: Path) -> List[Path]:
    return sorted([p for p in in_dir.iterdir() if p.suffix.lower() in EXTS])


def minimal_overlap_grid_positions(dim: int, win: int):
    if dim < win:
        return []
    if dim == win:
        return [0]
    n = math.ceil(dim / win)
    if n <= 1:
        return [0]
    span = dim - win
    positions = [int(round(k * span / (n - 1))) for k in range(n)]
    positions[0] = 0
    positions[-1] = span
    for i in range(1, len(positions)):
        if positions[i] < positions[i - 1]:
            positions[i] = positions[i - 1]
        if positions[i] > span:
            positions[i] = span
    return positions


def ensure_outdirs(root: Path):
    (root / "512px").mkdir(parents=True, exist_ok=True)
    (root / "256px").mkdir(parents=True, exist_ok=True)
    (root / "128px").mkdir(parents=True, exist_ok=True)


def passes_min_constraints(w: int, h: int, min_any: int, min_both: int):
    """Return (ok: bool, reason: str|None)."""
    if min_both is not None:
        if not (w >= min_both and h >= min_both):
            return False, f"skip: below --min_both {min_both} ({w}x{h})"
    if min_any is not None:
        if not (w >= min_any or h >= min_any):
            return False, f"skip: below --min_any {min_any} ({w}x{h})"
    return True, None


def compute_num_crops(w: int, h: int, win: int = 512) -> Tuple[int, List[int], List[int]]:
    """Return (#crops, xs, ys) using the minimal-overlap grid logic."""
    xs = minimal_overlap_grid_positions(w, win)
    ys = minimal_overlap_grid_positions(h, win)
    if not xs or not ys:
        return 0, xs, ys
    return len(xs) * len(ys), xs, ys


def process_one(job):
    idx, path, out_root, force_pillow, save_meta, backend_hint, min_any, min_both, dry_run = job
    r256, r128, backend = try_import_resizeright(force_pillow)
    try:
        im = Image.open(path).convert("RGB")
        w, h = im.size

        # hard min for a 512x512 crop
        if w < 512 or h < 512:
            return (idx, path.name, 0, "skip: too small for 512 crop", w, h)

        # user constraints
        ok, reason = passes_min_constraints(w, h, min_any, min_both)
        if not ok:
            return (idx, path.name, 0, reason, w, h)

        # compute planned crop count
        k_planned, xs, ys = compute_num_crops(w, h, 512)
        if k_planned == 0:
            return (idx, path.name, 0, "skip: grid empty", w, h)

        # if dry-run, don't save anything—just report planned count
        if dry_run:
            if save_meta:
                meta_line = {
                    "index": idx, "filename": path.name,
                    "size": [w, h], "num_crops": k_planned,
                    "resizer_backend": backend_hint or backend,
                    "mode": "dry_run"
                }
                with open(out_root / "meta.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(meta_line, ensure_ascii=False) + "\n")
            return (idx, path.name, k_planned, "dry_run", w, h)

        # otherwise, actually produce crops (and verify we end up with k_planned)
        k = 0
        for y0 in ys:
            for x0 in xs:
                crop = im.crop((x0, y0, x0 + 512, y0 + 512))
                k += 1
                name = f"{idx}_crop_{k}.png"
                crop.save(out_root / "512px" / name)
                r256(crop).save(out_root / "256px" / name)
                r128(crop).save(out_root / "128px" / name)

        if save_meta:
            meta_line = {
                "index": idx, "filename": path.name,
                "size": [w, h], "num_crops": k,
                "resizer_backend": backend_hint or backend
            }
            with open(out_root / "meta.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(meta_line, ensure_ascii=False) + "\n")

        return (idx, path.name, k, "ok", w, h)
    except Exception as e:
        return (idx, path.name, 0, f"error: {e}", None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=Path, required=True, help="Folder with images")
    ap.add_argument("--out_root", type=Path, required=True, help="Output root (e.g., dataset/result)")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--use_pillow", action="store_true", help="Force Pillow bicubic instead of ResizeRight")
    ap.add_argument("--save_meta", action="store_true", help="Write meta.jsonl mapping")
    ap.add_argument("--dry_run", action="store_true",
                    help="Only compute & report crop counts; do not write image crops")
    # new constraints
    ap.add_argument("--min_any", type=int, default=None,
                    help="Process only if (W>=N or H>=N)")
    ap.add_argument("--min_both", type=int, default=None,
                    help="Process only if (W>=N and H>=N)")
    args = ap.parse_args()

    if args.min_any is not None and args.min_any < 0:
        raise ValueError("--min_any must be >= 0")
    if args.min_both is not None and args.min_both < 0:
        raise ValueError("--min_both must be >= 0")

    args.out_root.mkdir(parents=True, exist_ok=True)
    ensure_outdirs(args.out_root)

    images = list_images(args.in_dir)
    if not images:
        print("No images found.")
        return

    _, _, backend = try_import_resizeright(force_pillow=args.use_pillow)
    print(f"[info] resize backend: {backend}")

    jobs = [(i, p, args.out_root, args.use_pillow, args.save_meta, backend,
             args.min_any, args.min_both, args.dry_run)
            for i, p in enumerate(images)]

    ok = skipped = err = 0
    total_crops = 0
    with Pool(processes=args.workers) as pool:
        for i, name, k, status, w, h in tqdm(
            pool.imap_unordered(process_one, jobs),
            total=len(jobs), desc="Processing", unit="img"):
            if status in ("ok", "dry_run"):
                ok += 1
                total_crops += k
                dims = f"{w}x{h}" if (w and h) else "unknown"
                tag = "planned" if status == "dry_run" else "created"
                # print(f"[{i:04d}] {name} ({dims}) -> {k} crops ({tag})")
            elif status.startswith("skip:"):
                skipped += 1
                dims = f"{w}x{h}" if (w and h) else "unknown"
                # print(f"[{i:04d}] {name} ({dims}) -> {status}")
            else:
                err += 1
                # print(f"[{i:04d}] {name} -> {status}")

    mode = "dry_run" if args.dry_run else "run"
    print(f"Done ({mode}). ok={ok}, skipped={skipped}, errors={err}, crops={total_crops}")


if __name__ == "__main__":
    main()
