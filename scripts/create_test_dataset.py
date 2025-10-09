#!/usr/bin/env python3
"""
Build a 10K-image test dataset with embeddings.

This script samples a fixed number of images from the full dataset and copies them
into a dedicated test split folder, along with their corresponding embeddings.
"""

from pathlib import Path
import random
import shutil
import os
from tqdm.auto import tqdm

# --- Configuration ---
dataset_root = Path("./workspace/d23/d23")  # full dataset
cache_root = Path("./workspace/d23/d23/cache_vae_embeddings")  # embeddings cache directory
test_root = Path("./workspace/d23/test_dataset")  # output folder for the sampled test split
sample_count = 10_000
random_seed = 1337
overwrite_existing = False  # set to True to delete and recreate the test folder if it already exists

image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}
embedding_extension = ".pt"

def main():
    assert dataset_root.exists() and dataset_root.is_dir(), f"Dataset root {dataset_root} is missing"
    assert cache_root.exists() and cache_root.is_dir(), f"Cache root {cache_root} is missing"

    if test_root.exists():
        if overwrite_existing:
            shutil.rmtree(test_root)
        else:
            raise FileExistsError(f"Destination {test_root} already exists. Set overwrite_existing=True to rebuild it.")

    test_root.mkdir(parents=True, exist_ok=True)
    print(f"Dataset root: {dataset_root.resolve()}")
    print(f"Cache root: {cache_root.resolve()}")
    print(f"Test split destination: {test_root.resolve()}")
    print(f"Target sample size: {sample_count}")

    # Index all image files under the dataset root
    all_images = []
    for path, _, files in os.walk(dataset_root):
        for name in files:
            ext = Path(name).suffix.lower()
            if ext in image_extensions:
                all_images.append(Path(path) / name)

    total_images = len(all_images)
    print(f"Total images discovered: {total_images}")
    if total_images < sample_count:
        raise ValueError(f"Requested {sample_count} samples but only found {total_images} eligible images.")

    # Sample unique image paths using a reproducible RNG
    rng = random.Random(random_seed)
    sampled_paths = rng.sample(all_images, sample_count)
    print(f"Sampled {len(sampled_paths)} images.")

    # Copy each sampled image while recreating its relative directory tree
    for src_path in tqdm(sampled_paths):
        relative_path = src_path.relative_to(dataset_root)
        dest_path = test_root / relative_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)

        # Copy corresponding embedding
        embedding_src = cache_root / relative_path.with_suffix(embedding_extension)
        embedding_dest = test_root / "cache_vae_embeddings" / relative_path.with_suffix(embedding_extension)
        if embedding_src.exists():
            embedding_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(embedding_src, embedding_dest)
        else:
            print(f"Warning: Embedding not found for {src_path}")

    print("Finished copying sampled files and embeddings.")

    # Quick verification of the resulting test split
    copied_images = [p for p in test_root.rglob('*') if p.suffix.lower() in image_extensions]
    copied_embeddings = [p for p in test_root.rglob('*') if p.suffix.lower() == embedding_extension]
    print(f"Images inside test split: {len(copied_images)}")
    print(f"Embeddings inside test split: {len(copied_embeddings)}")
    print("Sample preview:")
    for preview_path in copied_images[:5]:
        print(f" - {preview_path.relative_to(test_root)}")

if __name__ == "__main__":
    main()</content>
<parameter name="filePath">/data/kazanplova/latent_vae_upscale_train/scripts/create_test_dataset.py