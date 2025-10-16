#!/usr/bin/env python3
"""
Utilities for removing near-duplicate crops produced by ``scripts/crop_resize.py``.

Given a dataset directory that contains ``512px``, ``256px`` and ``128px`` subfolders,
this script computes SigLIP image embeddings for the 512px crops, finds the nearest
neighbour of every embedding with FAISS, and removes crops whose nearest neighbour is
closer than the configured distance threshold. Matching crops are deleted from all
resolution folders so that the dataset stays consistent.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import faiss  # type: ignore
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import SiglipModel, SiglipProcessor

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
RESOLUTION_FOLDERS = ("512px", "256px", "128px")
DEFAULT_SIGLIP_MODEL = "google/siglip-base-patch16-512"


@dataclass(frozen=True)
class DatasetStats:
    total_images: int
    duplicate_groups: int
    removed_images: int
    removed_files: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove near duplicate crops using SigLIP embeddings and FAISS."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Path to the folder that contains 512px/256px/128px sub-directories.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of images to encode per batch.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.96,
        help="L2 distance threshold for declaring two crops as duplicates.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_SIGLIP_MODEL,
        help="SigLIP vision model to use for embedding computation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to use (default: cuda if available, otherwise cpu).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Identify duplicates but do not delete any files.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing directory: {images_dir}")
    return sorted(
        [
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        ]
    )


def load_siglip(model_name: str, device: torch.device) -> Tuple[SiglipProcessor, SiglipModel]:
    processor = SiglipProcessor.from_pretrained(model_name)
    model = SiglipModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model


def compute_embeddings(
    image_paths: Sequence[Path],
    processor: SiglipProcessor,
    model: SiglipModel,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    embeddings: List[np.ndarray] = []
    progress = tqdm(total=len(image_paths), desc="Computing embeddings", unit="img")
    for offset in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[offset : offset + batch_size]
        images = []
        for path in batch_paths:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
        if not images:
            continue
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds  # (batch, dim)
        image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=-1)
        embeddings.append(image_embeds.cpu().numpy().astype("float32"))
        progress.update(len(batch_paths))
    progress.close()
    if not embeddings:
        return np.empty((0, 0), dtype="float32")
    return np.concatenate(embeddings, axis=0)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def find_duplicate_components(
    embeddings: np.ndarray, index: faiss.Index, threshold: float
) -> Dict[int, List[int]]:
    if embeddings.size == 0:
        return {}
    distances_sq, neighbours = index.search(embeddings, k=2)
    parent = list(range(len(embeddings)))
    size = [1] * len(embeddings)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        root_x, root_y = find(x), find(y)
        if root_x == root_y:
            return
        if size[root_x] < size[root_y]:
            root_x, root_y = root_y, root_x
        parent[root_y] = root_x
        size[root_x] += size[root_y]

    for idx in range(len(embeddings)):
        neighbour_idx = neighbours[idx, 1]
        if neighbour_idx < 0:
            continue
        dist = math.sqrt(distances_sq[idx, 1])
        if dist < threshold:
            union(idx, neighbour_idx)

    components: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(embeddings)):
        root = find(idx)
        components[root].append(idx)
    components = {root: members for root, members in components.items() if len(members) > 1}
    return components


def plan_deletions(
    components: Dict[int, List[int]], image_paths: Sequence[Path]
) -> List[int]:
    deletions: List[int] = []
    for members in components.values():
        members.sort(key=lambda idx: image_paths[idx].name)
        deletions.extend(members[1:])
    return sorted(set(deletions))


def delete_images(
    indices_to_remove: Iterable[int], image_paths: Sequence[Path], dataset_dir: Path
) -> int:
    removed_files = 0
    for index in indices_to_remove:
        filename = image_paths[index].name
        for folder in RESOLUTION_FOLDERS:
            target = dataset_dir / folder / filename
            if target.exists():
                target.unlink()
                removed_files += 1
    return removed_files


def run_cleanup(args: argparse.Namespace) -> DatasetStats:
    dataset_dir = args.dataset_dir.resolve()
    hi_res_dir = dataset_dir / "512px"
    image_paths = list_images(hi_res_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {hi_res_dir}")

    device = resolve_device(args.device)
    processor, model = load_siglip(args.model_name, device)
    embeddings = compute_embeddings(image_paths, processor, model, device, args.batch_size)
    if embeddings.size == 0:
        raise RuntimeError("No embeddings were computed.")

    index = build_faiss_index(embeddings)
    duplicate_components = find_duplicate_components(embeddings, index, args.distance_threshold)
    indices_to_remove = plan_deletions(duplicate_components, image_paths)

    removed_files = 0
    if indices_to_remove and not args.dry_run:
        removed_files = delete_images(indices_to_remove, image_paths, dataset_dir)

    stats = DatasetStats(
        total_images=len(image_paths),
        duplicate_groups=len(duplicate_components),
        removed_images=len(indices_to_remove),
        removed_files=removed_files,
    )
    return stats


def print_summary(stats: DatasetStats, dry_run: bool, threshold: float) -> None:
    action = "would remove" if dry_run else "removed"
    print("\nDataset cleaning summary")
    print(f"  Total 512px crops scanned : {stats.total_images}")
    print(f"  Duplicate groups found    : {stats.duplicate_groups}")
    print(f"  Images {action}           : {stats.removed_images}")
    if not dry_run:
        print(f"  Files deleted (all scales): {stats.removed_files}")
    print(f"  Distance threshold        : {threshold}")


def main() -> None:
    args = parse_args()
    stats = run_cleanup(args)
    print_summary(stats, args.dry_run, args.distance_threshold)


if __name__ == "__main__":
    main()
