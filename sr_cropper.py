#!/usr/bin/env python3
"""
sr_chains.py — Build aligned SR chains 128->256->512 from 2048×2048 2×2 collages.

For each selected 512×512 HR crop (same FOV), we create:
  - hr_512: native 512×512 crop from source
  - gt_256: hr_512 downscaled by 2 (clean) — target for stage1 and input for stage2
  - lr_128: hr_512 downscaled by 4 + mild degradations — input for stage1
python sr_cropper.py \
  --in_dir images \
  --out_dir cropped_result \
  --stride 256 \
  --per_tile_limit 80 \
  --tile_guard 8 \
  --seed 42


This guarantees strict alignment 128->256->512 over exactly the same field of view.
"""

import argparse, io, os, random, json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image, ImageFilter

# ResizeRight imports
from utils.resize_utils import pil_resize_right
from utils import interp_methods

# ---------- utils ----------
def pil_load_rgb(path: Path) -> Image.Image:
    im = Image.open(path)
    return im.convert('RGB')

def im_to_np(im: Image.Image) -> np.ndarray:
    return np.asarray(im).astype(np.float32) / 255.0

def rgb_to_gray_np(img: np.ndarray) -> np.ndarray:
    return 0.2989*img[...,0] + 0.5870*img[...,1] + 0.1140*img[...,2]

def local_entropy(gray: np.ndarray, bins: int = 64) -> float:
    hist, _ = np.histogram((gray * (bins-1)).round(), bins=bins, range=(0, bins-1), density=True)
    hist = hist + 1e-9
    return float(-(hist * np.log2(hist)).sum())

def laplacian_var(gray: np.ndarray) -> float:
    k = np.array([[0, 1, 0],[1,-4, 1],[0, 1, 0]], dtype=np.float32)
    gpad = np.pad(gray, ((1,1),(1,1)), mode='reflect')
    resp = (k[0,0]*gpad[:-2,:-2] + k[0,1]*gpad[:-2,1:-1] + k[0,2]*gpad[:-2,2:] +
            k[1,0]*gpad[1:-1,:-2] + k[1,1]*gpad[1:-1,1:-1] + k[1,2]*gpad[1:-1,2:] +
            k[2,0]*gpad[2:,  :-2] + k[2,1]*gpad[2:,  1:-1] + k[2,2]*gpad[2:,  2:])
    return float(resp.var())

def saturation_std(rgb: np.ndarray) -> float:
    maxc = np.max(rgb, axis=-1); minc = np.min(rgb, axis=-1)
    v = maxc; s = np.zeros_like(v)
    nz = v > 1e-6
    s[nz] = (maxc[nz]-minc[nz])/(v[nz]+1e-6)
    return float(s.std())

def brightness_mean(gray: np.ndarray) -> float: return float(gray.mean())
def contrast_std(gray: np.ndarray) -> float: return float(gray.std())

def informativeness_score(rgb: np.ndarray):
    gray = rgb_to_gray_np(rgb)
    ent = local_entropy(gray)
    lap = laplacian_var(gray)
    sat = saturation_std(rgb)
    bri = brightness_mean(gray)
    con = contrast_std(gray)
    score = 0.45*(ent/6.0) + 0.45*min(lap*100.0, 2.0) + 0.10*min(sat*2.5, 1.5)
    if bri < 0.08 or bri > 0.92: score *= 0.3
    if con < 0.03: score *= 0.3
    return float(score), {"brightness": float(bri),"contrast": float(con),"lapvar": float(lap),"entropy": float(ent)}

def gaussian_blur_pil(im: Image.Image, sigma: float) -> Image.Image:
    return im.filter(ImageFilter.GaussianBlur(radius=sigma))

def jpeg_roundtrip(im: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=q, subsampling=2)
    buf.seek(0)
    out = Image.open(buf).convert('RGB')
    return out

def add_noise(im: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(im).astype(np.float32) / 255.0
    noise = np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((arr*255.0).astype('uint8'))

def nms(cands, iou_thr=0.35):
    def iou(a,b):
        ax,ay,sz=a['x'],a['y'],a['size']; bx,by,bsz=b['x'],b['y'],b['size']
        ax2,ay2=ax+sz,ay+sz; bx2,by2=bx+bsz,by+bsz
        iw=max(0,min(ax2,bx2)-max(ax,bx)); ih=max(0,min(ay2,by2)-max(ay,by))
        inter=iw*ih; union=sz*sz+bsz*bsz-inter
        return inter/union if union>0 else 0.0
    cands=sorted(cands, key=lambda d:d["score"], reverse=True)
    picked=[]
    for c in cands:
        if all(iou(c,p)<iou_thr for p in picked):
            picked.append(c)
    return picked

# ---------- main ----------
@dataclass
class Args:
    in_dir: Path
    out_dir: Path
    stride: int
    per_tile_limit: int
    tile_guard: int
    seed: int
    degrade: bool

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir', type=Path, required=True)
    p.add_argument('--out_dir', type=Path, required=True)
    p.add_argument('--stride', type=int, default=256, help='slide step for 512 window')
    p.add_argument('--per_tile_limit', type=int, default=80)
    p.add_argument('--tile_guard', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no_degrade', action='store_true', help='disable degradations for lr_128 (still downscale)')
    return p

def main():
    ns = build_parser().parse_args()
    args = Args(
        in_dir=ns.in_dir, out_dir=ns.out_dir, stride=ns.stride,
        per_tile_limit=ns.per_tile_limit, tile_guard=ns.tile_guard,
        seed=ns.seed, degrade=not ns.no_degrade
    )
    rng = random.Random(args.seed)
    (args.out_dir/'chains').mkdir(parents=True, exist_ok=True)
    meta = open(args.out_dir/'metadata.jsonl', 'w', encoding='utf-8')

    for path in sorted(args.in_dir.glob('*')):
        if path.suffix.lower() not in ('.png','.jpg','.jpeg','.webp','.bmp'): continue
        im = pil_load_rgb(path)
        if im.size!=(2048,2048):
            print(f"[WARN] Skip {path.name}: not 2048x2048"); continue

        T=1024; g=args.tile_guard
        tiles=[]
        for r in range(2):
            for c in range(2):
                tiles.append(((r,c),(c*T+g, r*T+g, (c+1)*T-g, (r+1)*T-g)))

        for (r,c),(x1,y1,x2,y2) in tiles:
            tile = im.crop((x1,y1,x2,y2))
            tile_np = im_to_np(tile)
            th,tw,_ = tile_np.shape
            cands=[]
            size=512
            for yy in range(0, th-size+1, args.stride):
                for xx in range(0, tw-size+1, args.stride):
                    patch = tile_np[yy:yy+size, xx:xx+size, :]
                    score, stats = informativeness_score(patch)
                    if stats["contrast"] < 0.02 or stats["lapvar"] < 0.0003: continue
                    if stats["brightness"] < 0.03 or stats["brightness"] > 0.97: continue
                    cands.append({"x":xx,"y":yy,"size":size,"score":score,"stats":stats})
            cands = nms(cands, iou_thr=0.35)[:args.per_tile_limit]

            for cand in cands:
                xx,yy = cand["x"], cand["y"]
                hr_512 = tile.crop((xx,yy,xx+512,yy+512))
                # Use ResizeRight for high-quality, consistent resizing
                gt_256 = pil_resize_right(hr_512, (256, 256), interp_method=interp_methods.cubic, antialiasing=True)
                lr_128 = pil_resize_right(hr_512, (128, 128), interp_method=interp_methods.cubic, antialiasing=True)
                if args.degrade:
                    sigma = rng.uniform(0.0, 1.2)
                    if sigma>1e-6: lr_128 = gaussian_blur_pil(lr_128, sigma)
                    if rng.random()<0.5:
                        q = rng.randint(40,90)
                        lr_128 = jpeg_roundtrip(lr_128, q)
                    lr_128 = add_noise(lr_128, rng.uniform(0.0,0.03))

                chain_id = f"{path.stem}_r{r}c{c}_x{xx}_y{yy}"
                cdir = args.out_dir/'chains'/chain_id
                cdir.mkdir(parents=True, exist_ok=True)
                hr_512.save(cdir/'hr_512.png')
                gt_256.save(cdir/'gt_256.png')
                lr_128.save(cdir/'lr_128.png')

                rec = {
                    "chain_id": chain_id,
                    "src": path.name,
                    "tile": [r,c],
                    "x": int(xx), "y": int(yy),
                    "score": float(cand["score"]),
                    "stats": cand["stats"],
                    "paths": {
                        "hr_512": f"chains/{chain_id}/hr_512.png",
                        "gt_256": f"chains/{chain_id}/gt_256.png",
                        "lr_128": f"chains/{chain_id}/lr_128.png",
                    }
                }
                meta.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"Processed {path.name}")

    meta.close()

if __name__ == "__main__":
    main()
