#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel 4-crop pipeline for 2048x2048 2x2-collage images with ResizeRight.

For each input image (index i = 0..N-1):
  1) Load image (must be 2048x2048)
  2) Split into 4 quadrant crops of size 512x512 (TL, TR, BL, BR)
  3) For each 512 crop:
       - Save 512px/{i}_crop_{k}.png
       - Downscale to 256 with ResizeRight -> save 256px/{i}_crop_{k}.png
       - Downscale to 128 with ResizeRight -> save 128px/{i}_crop_{k}.png

Naming matches your requirement:
  128px/0_crop_1.png
  256px/0_crop_1.png
  512px/0_crop_1.png
  ... (and similarly for crop_2..4 and subsequent images)

Dependencies:
  - Pillow
  - tqdm
  - ResizeRight utilities available as:
        from utils.resize_utils import pil_resize_right
        from utils import interp_methods

Usage:
  python make_crops_with_resizeright.py \
    --in_dir /path/to/images \
    --out_root dataset/result_folder \
    --workers 8
"""

import argparse
from pathlib import Path
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
from PIL import Image
from tqdm import tqdm
from PIL import Image

TARGET_SIZES = (512, 256, 128)  # fixed order
EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def list_images(in_dir: Path) -> List[Path]:
    return sorted([p for p in in_dir.iterdir() if p.suffix.lower() in EXTS])


def tiles_2x2_512_boxes() -> List[Tuple[int, int, int, int]]:
    """Return four boxes (left, top, right, bottom) for 512x512 crops per quadrant."""
    # Full image is 2048x2048 made of 4 tiles 1024x1024.
    # We want centered 512x512 crop inside each tile.
    T = 1024
    # Each tile box:
    tile_boxes = [
        (0,    0,    T,    T),     # TL
        (T,    0,    2*T,  T),     # TR
        (0,    T,    T,    2*T),   # BL
        (T,    T,    2*T,  2*T),   # BR
    ]
    # Centered 512x512 inside each 1024x1024 tile:
    crop_boxes = []
    s = 512
    half = s // 2
    for (x1, y1, x2, y2) in tile_boxes:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        box = (cx - half, cy - half, cx - half + s, cy - half + s)
        crop_boxes.append(box)
    return crop_boxes


def process_one(job):
    """
    Worker: processes one image.
    Args:
      job = (idx, path, out_root)
    Returns:
      (idx, name, status_str)
    """
    idx, path, out_root = job
    try:
        im = Image.open(path).convert("RGB")
        if im.size != (2048, 2048):
            return (idx, path.name, f"skip: expected 2048x2048, got {im.size}")

        # Four 512x512 crops (same order always): 1..4
        for crop_id, box in enumerate(tiles_2x2_512_boxes(), start=1):
            hr_512 = im.crop(box)  # pure crop

            # Save 512
            out_512 = out_root / "512px" / f"{idx}_crop_{crop_id}.png"
            hr_512.save(out_512)

            # ResizeRight down to 256 and 128 (cubic + antialiasing=True)
            # gt_256 = pil_resize_right(
            #     hr_512, (256, 256),
            #     interp_method=interp_methods.cubic,
            #     antialiasing=True
            # )
            # lr_128 = pil_resize_right(
            #     hr_512, (128, 128),
            #     interp_method=interp_methods.cubic,
            #     antialiasing=True
            # )
            gt_256 = hr_512.resize((256, 256), resample=Image.BICUBIC)  # Fallback if needed
            lr_128 = hr_512.resize((128, 128), resample=Image.BICUBIC)  # Fallback if needed
            out_256 = out_root / "256px" / f"{idx}_crop_{crop_id}.png"
            out_128 = out_root / "128px" / f"{idx}_crop_{crop_id}.png"
            gt_256.save(out_256)
            lr_128.save(out_128)

        return (idx, path.name, "ok")
    except Exception as e:
        return (idx, path.name, f"error: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=Path, required=True, help="Folder with 2048x2048 collages")
    ap.add_argument("--out_root", type=Path, required=True, help="Output root (e.g., dataset/result_folder)")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    args = ap.parse_args()

    # Prepare folders
    args.out_root.mkdir(parents=True, exist_ok=True)
    for s in TARGET_SIZES:
        (args.out_root / f"{s}px").mkdir(parents=True, exist_ok=True)

    # Collect images and enumerate from 0 (as in your example)
    images = list_images(args.in_dir)
    if not images:
        print("No images found.")
        return

    jobs = [(i, p, args.out_root) for i, p in enumerate(images)]

    # Parallel loop with tqdm
    ok = skipped = err = 0
    with Pool(processes=args.workers) as pool:
        for i, name, status in tqdm(
            pool.imap_unordered(process_one, jobs),
            total=len(jobs),
            desc="Processing",
            unit="img",
        ):
            if status == "ok":
                ok += 1
            elif status.startswith("skip:"):
                skipped += 1
            else:
                err += 1

    print(f"Done. ok={ok}, skipped={skipped}, errors={err}")


if __name__ == "__main__":
    main()
