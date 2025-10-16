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
import atexit
import math
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import faiss  # type: ignore
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import SiglipModel, SiglipProcessor
from torch.utils.data import DataLoader, Dataset

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
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes for the DataLoader (default: 0).",
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


class SiglipImageDataset(Dataset):
    """Torch dataset that loads crops and prepares pixel values for SigLIP."""

    def __init__(self, image_paths: Sequence[Path], processor: SiglipProcessor) -> None:
        self.image_paths = list(image_paths)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | Path]:
        path = self.image_paths[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
        processed = self.processor(images=image, return_tensors="pt")
        pixel_values = processed["pixel_values"].squeeze(0)
        return {"pixel_values": pixel_values, "path": path}


def collate_siglip_batch(batch: Sequence[Dict[str, torch.Tensor | Path]]) -> Dict[str, torch.Tensor | List[Path]]:
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    paths = [item["path"] for item in batch]
    return {"pixel_values": pixel_values, "paths": paths}


def resolve_processor_image_size(processor: SiglipProcessor) -> Tuple[int, int]:
    size = processor.image_processor.size
    if isinstance(size, dict):
        if "height" in size and "width" in size:
            return int(size["height"]), int(size["width"])
        if "shortest_edge" in size:
            edge = int(size["shortest_edge"])
            return edge, edge
    if isinstance(size, (list, tuple)):
        if len(size) == 2:
            return int(size[0]), int(size[1])
        if len(size) == 1:
            edge = int(size[0])
            return edge, edge
    if isinstance(size, int):
        return size, size
    raise ValueError(f"Cannot resolve image size from processor config: {size}")


def create_dataloader(
    dataset: SiglipImageDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_siglip_batch,
    )


def build_index_with_temp_storage(
    dataset: SiglipImageDataset,
    model: SiglipModel,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[faiss.Index, Path]:
    dataloader = create_dataloader(dataset, device, batch_size, num_workers)
    index: faiss.Index | None = None
    progress = tqdm(total=len(dataset), desc="Computing embeddings", unit="img")
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            image_embeds = outputs.image_embeds
        image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=-1)
        embeddings_np = image_embeds.cpu().numpy().astype("float32")
        if embeddings_np.size == 0:
            continue
        if index is None:
            index = faiss.IndexFlatL2(embeddings_np.shape[1])
        index.add(embeddings_np)
        progress.update(embeddings_np.shape[0])
    progress.close()
    if index is None:
        raise RuntimeError("No embeddings were computed.")
    temp_file = tempfile.NamedTemporaryFile(prefix="faiss_index_", suffix=".bin", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()
    faiss.write_index(index, str(temp_path))
    atexit.register(lambda: temp_path.exists() and temp_path.unlink())
    index = faiss.read_index(str(temp_path))
    return index, temp_path


def find_duplicate_components(
    index: faiss.Index,
    dataset: SiglipImageDataset,
    model: SiglipModel,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    threshold: float,
) -> Dict[int, List[int]]:
    total = len(dataset)
    parent = list(range(total))
    size = [1] * total
    dataloader = create_dataloader(dataset, device, batch_size, num_workers)
    progress = tqdm(total=total, desc="Searching duplicates", unit="img")
    offset = 0

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

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            image_embeds = outputs.image_embeds
        image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=-1)
        embeddings_np = image_embeds.cpu().numpy().astype("float32")
        if embeddings_np.size == 0:
            continue
        distances_sq, neighbours = index.search(embeddings_np, k=2)
        batch_size_actual = embeddings_np.shape[0]
        for local_idx in range(batch_size_actual):
            global_idx = offset + local_idx
            if global_idx >= total:
                break
            neighbour_idx = int(neighbours[local_idx, 1])
            if neighbour_idx < 0:
                continue
            dist = math.sqrt(float(distances_sq[local_idx, 1]))
            if dist < threshold:
                union(global_idx, neighbour_idx)
        offset += batch_size_actual
        progress.update(batch_size_actual)
    progress.close()

    components: Dict[int, List[int]] = defaultdict(list)
    for idx in range(total):
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
    target_height, target_width = resolve_processor_image_size(processor)
    print(f"[info] SigLIP input size: {target_height}x{target_width}")
    dataset = SiglipImageDataset(image_paths, processor)
    image_paths = dataset.image_paths
    index, index_path = build_index_with_temp_storage(
        dataset, model, device, args.batch_size, args.num_workers
    )
    print(f"[info] FAISS index stored at: {index_path}")
    duplicate_components = find_duplicate_components(
        index,
        dataset,
        model,
        device,
        args.batch_size,
        args.num_workers,
        args.distance_threshold,
    )
    indices_to_remove = plan_deletions(duplicate_components, image_paths)

    removed_files = 0
    if indices_to_remove and not args.dry_run:
        removed_files = delete_images(indices_to_remove, image_paths, dataset_dir)

    stats = DatasetStats(
        total_images=len(dataset),
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
