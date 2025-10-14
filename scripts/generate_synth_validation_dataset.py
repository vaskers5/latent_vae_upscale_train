#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paired-resolution generation with FLUX.1-Krea-dev.

For each pair (LOW, HIGH), the script:
    1) Generates ONCE at HIGH (to minimize noise),
    2) Decodes to HIGH-res PIL,
    3) Downscales to LOW with BICUBIC,
    4) VAE-encodes BOTH (HIGH and LOW) and stores the EmbeddingCache payload,
    5) Saves into this structure:

  flux_val/{LOW}_{HIGH}/
    images/
      {LOW}px/*.png
      {HIGH}px/*.png
    embeddings/
      {LOW}px/*.pt     (VAE latents)
      {HIGH}px/*.pt    (VAE latents)

Notes
- Packed (transformer) latents are optional; by default we do NOT save them since the requested layout lists only embeddings per size.
- If desired, toggle SAVE_PACKED_LATENTS=True to also store: embeddings/transformer/{HIGH}px/*.pt under each pair folder.
"""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image
import torchvision.transforms as T
from diffusers import FluxPipeline

from training.embedding_io import CACHE_VERSION, TransformParams


EMBEDDING_DTYPE = torch.float16
STORE_DISTRIBUTION = True
DEFAULT_RESIZE_LONG_SIDE = 0


def _build_embedding_payload(
    latents: torch.Tensor,
    params: TransformParams,
    *,
    high_resolution: int,
    model_resolution: int,
    resize_long_side: Optional[int] = None,
    mean: Optional[torch.Tensor] = None,
    logvar: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "version": CACHE_VERSION,
        "latents": latents.cpu().contiguous(),
        "params": params.to_dict(),
        "variant": 0,
        "dataset": {
            "high_resolution": int(high_resolution),
            "model_resolution": int(model_resolution),
            "resize_long_side": int(resize_long_side or 0),
        },
        "storage_dtype": str(latents.dtype).replace("torch.", ""),
    }
    if mean is not None:
        payload["mean"] = mean.cpu().contiguous()
    if logvar is not None:
        payload["logvar"] = logvar.cpu().contiguous()
    return payload

# ---------------------- Config ----------------------
# Define square pairs (LOW, HIGH)
PAIRS: List[Tuple[int, int]] = [
    (128, 256),
    (256, 512),
    # (512, 1024),
]

# Toggle to also save packed transformer latents for the HIGH size of each pair
SAVE_PACKED_LATENTS = False

@torch.inference_mode()
def main():
    # --- User-configurable paths ---
    parquet_path = "/data/kazanplova/midjourney300k.parquet"
    result_root = Path("./flux_val")

    # --- Model & generation config ---
    model_id = "black-forest-labs/FLUX.1-Krea-dev"
    num_samples = 100              # number of prompts to process
    guidance_scale = 4.5
    num_inference_steps = 12

    # --- Device & model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print(f"[Info] Loading model '{model_id}' onto '{device}' with dtype '{dtype}'.")
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to(device)

    # --- Prompts ---
    df = pd.read_parquet(parquet_path)
    if "prompt" not in df.columns:
        raise ValueError("Input Parquet must contain a 'prompt' column.")

    if num_samples > len(df):
        print(f"[Warning] Requested {num_samples} samples, but only {len(df)} are available. Using all.")
        num_samples = len(df)

    prompts = df["prompt"].sample(n=num_samples, random_state=42).tolist()
    if not prompts:
        raise ValueError("No prompts found in the Parquet file.")
    print(f"[Info] Loaded {len(prompts)} prompts.")

    # --- Image → tensor transform for VAE.encode ([-1, 1]) ---
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # --- Prepare pair-specific directories ---
    pair_dirs = {}
    for low, high in PAIRS:
        pair_name = f"{low}_{high}"
        base_dir = result_root / pair_name
        images_low = base_dir / "images" / f"{low}px"
        images_high = base_dir / "images" / f"{high}px"
        emb_low = base_dir / "embeddings" / f"{low}px"
        emb_high = base_dir / "embeddings" / f"{high}px"

        for d in [images_low, images_high, emb_low, emb_high]:
            d.mkdir(parents=True, exist_ok=True)

        tr_dir = None
        if SAVE_PACKED_LATENTS:
            tr_dir = base_dir / "embeddings" / "transformer" / f"{high}px"
            tr_dir.mkdir(parents=True, exist_ok=True)

        pair_dirs[(low, high)] = {
            "images_low": images_low,
            "images_high": images_high,
            "emb_low": emb_low,
            "emb_high": emb_high,
            "tr_high": tr_dir,
        }

    # --- Shared VAE bits ---
    vae = pipe.vae
    image_dtype = vae.dtype

    # --- Main loop ---
    for (low, high) in PAIRS:
        print(f"\n[Info] Pair {low}_{high}: generate at {high}px, downscale to {low}px")
        dirs = pair_dirs[(low, high)]  # shorthand

        for idx, prompt in enumerate(tqdm(prompts, desc=f"{low}_{high}"), start=1):
            try:
                # 1) Generate packed latents at HIGH only (minimize noise)
                out = pipe(
                    prompt=prompt,
                    width=high,
                    height=high,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    output_type="latent",
                )
                packed_latents = out.images  # [B, N_tokens, hidden]

                prompt_slug = re.sub(r"\W+", "_", str(prompt)).lower()[:50]

                if SAVE_PACKED_LATENTS and dirs["tr_high"] is not None:
                    torch.save(packed_latents[0].cpu(), (dirs["tr_high"] / f"{idx:06d}_{prompt_slug}.pt"))

                # 2) Decode to HIGH-res PIL
                latents_spatial = pipe._unpack_latents(
                    packed_latents,
                    height=high,
                    width=high,
                    vae_scale_factor=pipe.vae_scale_factor,
                )
                latents_for_decode = (latents_spatial / vae.config.scaling_factor) + vae.config.shift_factor
                image_tensor = vae.decode(latents_for_decode, return_dict=False)[0]
                high_pil = pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]

                # 3) Downscale HIGH → LOW with BICUBIC
                low_pil = high_pil.resize((low, low), Image.Resampling.BICUBIC)

                # 4) Save images
                high_png = dirs["images_high"] / f"{idx}.png"
                low_png = dirs["images_low"] / f"{idx}.png"
                high_pil.save(high_png)
                low_pil.save(low_png)

                # 5) VAE-encode both (normalize to [-1,1])
                high_tensor = transform(high_pil).unsqueeze(0).to(device, dtype=image_dtype)
                low_tensor = transform(low_pil).unsqueeze(0).to(device, dtype=image_dtype)

                high_encoding = vae.encode(high_tensor)
                low_encoding = vae.encode(low_tensor)

                high_mean = high_encoding.latent_dist.mean.detach()
                low_mean = low_encoding.latent_dist.mean.detach()
                high_logvar = high_encoding.latent_dist.logvar.detach() if STORE_DISTRIBUTION else None
                low_logvar = low_encoding.latent_dist.logvar.detach() if STORE_DISTRIBUTION else None

                high_latents = high_mean.to(dtype=EMBEDDING_DTYPE)
                low_latents = low_mean.to(dtype=EMBEDDING_DTYPE)

                params_high = TransformParams(flip=False, crop_x=0, crop_y=0)
                params_low = TransformParams(flip=False, crop_x=0, crop_y=0)

                high_record = _build_embedding_payload(
                    high_latents[0],
                    params_high,
                    high_resolution=high,
                    model_resolution=high,
                    resize_long_side=DEFAULT_RESIZE_LONG_SIDE,
                    mean=high_mean[0],
                    logvar=high_logvar[0] if high_logvar is not None else None,
                )
                low_record = _build_embedding_payload(
                    low_latents[0],
                    params_low,
                    high_resolution=high,
                    model_resolution=low,
                    resize_long_side=DEFAULT_RESIZE_LONG_SIDE,
                    mean=low_mean[0],
                    logvar=low_logvar[0] if low_logvar is not None else None,
                )

                torch.save(high_record, (dirs["emb_high"] / f"{idx}.pt"))
                torch.save(low_record, (dirs["emb_low"] / f"{idx}.pt"))

                if idx % 25 == 0:
                    print(f"[OK {low}_{high}] {idx}/{len(prompts)} — images + VAE latents saved")

            except Exception as e:
                print(f"[Error {low}_{high}] idx={idx}: {e}")
                continue


if __name__ == "__main__":
    main()
