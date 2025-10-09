#!/usr/bin/env python3
"""
Clear embeddings that are not connected to images in the cleared dataset.

This script takes a pandas DataFrame from a CSV file (e.g., 'clear_images.csv')
containing image paths to keep, and deletes all embeddings in the cache directory
that do not correspond to these images.
"""

import os
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

# --- Configuration ---
dataset_root = os.path.abspath("unpacked_original_ds/full_dataset/")  # full dataset root
cache_root = Path("unpacked_original_ds/full_dataset/cache_vae_embeddings")  # embeddings cache directory
csv_file = "clear_images.csv"  # CSV file with paths to keep
embedding_extension = ".pt"

def resolve_to_str(p):
    return str(p.resolve())

def format_path(file_path):
#   file_path = str(file_path).replace(str(cache_root), "")
  pattern = r"\d+px/(.*)\.[^.]+$"
  
  match = re.search(pattern, file_path)
  
  if match:
    # If a match is found, return the captured group.
    return match.group(1)
  else:
    # If the path doesn't match the expected format, return it unchanged.
    return file_path


def main():
    # Load the CSV with paths to keep
    if not Path(csv_file).exists():
        raise FileNotFoundError(f"CSV file {csv_file} not found.")
    
    clear_df = pd.read_csv(csv_file)
    keep_paths = set(clear_df['path'])
    
    print(f"Loaded {len(keep_paths)} image paths to keep from {csv_file}.")
    
    # Collect all embedding files in the cache directory
    cache_root_resolved = cache_root.resolve()
    all_embeddings = [p for p in tqdm(cache_root.rglob(f"*{embedding_extension}"))]
    
    print(f"Found {len(all_embeddings)} embedding files in {cache_root}.")
    
    # Group embeddings by filename for fast lookup
    embeddings_by_name = defaultdict(list)
    for emb in tqdm(all_embeddings):
        name = "/" + format_path(str(emb))
        embeddings_by_name[name].append(os.path.abspath(emb))
    
    # Add available embeddings column to the dataframe
    def get_available_embeddings(img_path_str):
        name = img_path_str.replace(str(dataset_root), "").split(".")[0]
        available_embeddings = embeddings_by_name[name]
        return available_embeddings
    available_embeddings_for_path = []
    without_embeddings = []
    for img_path_str in tqdm(clear_df['path'].tolist()):
            available_embeddings_for_path.append(get_available_embeddings(img_path_str) )
            if not available_embeddings_for_path[-1]:
                without_embeddings.append(img_path_str)

    clear_df['available_embeddings'] = available_embeddings_for_path
    
    # Save the new CSV with the additional column
    new_csv_file = "clear_images_with_embeddings.csv"
    clear_df.to_csv(new_csv_file, index=False)
    print(f"Saved updated CSV with available embeddings to {new_csv_file}.")
    

if __name__ == "__main__":
    main()