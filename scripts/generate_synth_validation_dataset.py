#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate images from CSV prompts with FLUX.1-Krea-dev and save both images and VAE latents.
This updated script saves the denoised latents before they are passed to the VAE.
"""

from pathlib import Path

import torch
import pandas as pd
from tqdm.auto import tqdm
from diffusers import FluxPipeline

SUPPORTED_RESOLUTIONS = [
    (256, 256),
    (512, 512),
    (768, 768),
    (1024, 1024),
]

@torch.inference_mode()
def main():
    # --- Configuration ---
    parquet_path = "/data/kazanplova/midjourney300k.parquet"
    result_dir = Path("./flux_val")
    model_id = "black-forest-labs/FLUX.1-Krea-dev"
    num_samples = 300  # Number of prompts to process
    
    # --- Setup Directories ---
    result_dir.mkdir(exist_ok=True)
    img_dir = result_dir / "images"
    embeddings_dir = result_dir / "embeddings"
    img_dir.mkdir(exist_ok=True)
    embeddings_dir.mkdir(exist_ok=True)
    
    # --- Device & Model Loading ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    print(f"[Info] Loading model '{model_id}' onto '{device}' with dtype '{dtype}'.")
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to(device)

    # --- Load Prompts ---
    df = pd.read_parquet(parquet_path)
    if num_samples > len(df):
        print(f"[Warning] Requested {num_samples} samples, but only {len(df)} are available.")
        num_samples = len(df)
        
    prompts = df["prompt"].sample(n=num_samples, random_state=42).tolist()
    if not prompts:
        raise ValueError("No prompts found in the Parquet file.")
    print(f"[Info] Loaded {len(prompts)} prompts.")

    # --- Generation Loop ---
    for width, height in SUPPORTED_RESOLUTIONS:
        res_str = f"{width}px"
        print(f"\n[Info] Starting generation for resolution: {res_str}")

        # Create resolution-specific subdirectories
        img_res_dir = img_dir / res_str
        latent_res_dir = embeddings_dir / res_str
        img_res_dir.mkdir(exist_ok=True)
        latent_res_dir.mkdir(exist_ok=True)

        for idx, prompt in enumerate(tqdm(prompts, desc=f"Processing {res_str}"), start=1):
            # 1. Generate latents by stopping the pipeline before the VAE decoder
            # The `images` attribute contains the batched latent tensor. Do NOT remove the batch dimension.
            denoised_latents = pipe(
                prompt=prompt,
                width=width,
                height=height,
                guidance_scale=4.5,
                num_inference_steps=12,
                output_type="latent"
            ).images
            # 2. Save the packed (transformer) latents as-is if you want
            latent_path = latent_res_dir / f"{idx}.pt"
            torch.save(denoised_latents[0].cpu(), latent_path)

            # 3. UNPACK packed latents -> VAE spatial latents [B, 16, H/vae_scale, W/vae_scale]
            #    (private helper mirrors the reference implementation)
            latents_spatial = pipe._unpack_latents(
                denoised_latents,  # shape [B, N_tokens, hidden]
                height=height,
                width=width,
                vae_scale_factor=pipe.vae_scale_factor,  # usually 8
            )

            # 4. Apply VAE scale & shift BEFORE decoding (again: same as reference)
            latents_for_decode = (latents_spatial / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor

            # 5. Decode with the VAE and postprocess to PIL
            image_tensor = pipe.vae.decode(latents_for_decode, return_dict=False)[0]
            pil_img = pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]
            img_path = img_res_dir / f"{idx}.png"
            pil_img.save(img_path)



if __name__ == "__main__":
    main()