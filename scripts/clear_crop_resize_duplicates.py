#!/usr/bin/env python3
"""
Fast duplicate-removal tool for 3-resolution SR datasets with multi-GPU embeddings.

- Multi-GPU embedding via PyTorch Distributed (DDP) when launched with torchrun.
- Compute SigLIP embeddings ONCE (AMP on CUDA), normalize.
- Build in-memory FAISS index (optional GPU; rank-0 only, single GPU).
- Single batched k-NN (k=2) on the same embedding set.
- Union-find over pairs whose cosine similarity >= threshold.
- Keep one file per group (lexicographically smallest basename),
  delete the rest ACROSS 512px/256px/128px in parallel.
- Interactive confirmation (y/n) unless --yes or --dry-run.
- Also provides --no-index mode (direct similarity) for small datasets.

Author: optimized for speed & correctness.
"""

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import faiss  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from open_clip import create_model_from_pretrained
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
RESOLUTION_FOLDERS = ("512px", "256px", "128px")
DEFAULT_SIGLIP_MODEL = "hf-hub:timm/ViT-SO400M-14-SigLIP"

@dataclass(frozen=True)
class DatasetStats:
    total_images: int
    duplicate_groups: int
    removed_images: int   # count of 512px images removed
    removed_files: int    # count across 512/256/128 actually deleted

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Remove near-duplicate crops using SigLIP embeddings and FAISS (single-pass, fast, multi-GPU embeddings)."
    )
    p.add_argument("--dataset-dir", type=Path, required=True,
                   help="Folder containing 512px/256px/128px subdirs.")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Images per batch (embedding).")
    p.add_argument("--num-workers", type=int, default=max(0, (os.cpu_count() or 8) - 2),
                   help="DataLoader workers per process.")
    p.add_argument("--sim-threshold", type=float, default=0.985,
                   help="Cosine similarity threshold to mark duplicates (0..1).")
    p.add_argument("--model-name", type=str, default=DEFAULT_SIGLIP_MODEL,
                   help="SigLIP vision model to use for embeddings.")
    p.add_argument("--device", type=str, default=None,
                   help="Torch device (default: cuda if available else cpu). Ignored under DDP.")
    p.add_argument("--gpu-index", action="store_true",
                   help="Use FAISS-GPU for indexing/search on rank-0 (single GPU).")
    p.add_argument("--no-index", action="store_true",
                   help="Small-dataset mode: find duplicates without FAISS index (O(N^2) memory).")
    p.add_argument("--dry-run", action="store_true",
                   help="Identify duplicates but do not delete.")
    p.add_argument("--yes", action="store_true",
                   help="Do not prompt for confirmation; proceed.")
    p.add_argument("--delete-workers", type=int, default=max(1, (os.cpu_count() or 8)//2),
                   help="Processes for deletion.")
    return p.parse_args()

def is_distributed_env() -> Tuple[bool, int, int, int]:
    """Return (is_dist, rank, world_size, local_rank) based on env provided by torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
        return True, rank, world, local
    return False, 0, 1, 0

def setup_distributed(backend: str | None = None) -> Tuple[bool, int, int, int]:
    is_dist, rank, world, local = is_distributed_env()
    if not is_dist:
        return False, 0, 1, 0
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.cuda.set_device(local if torch.cuda.is_available() else 0)
    dist.init_process_group(backend=backend, init_method="env://")
    return True, rank, world, local

def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def list_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing directory: {images_dir}")
    paths: List[Path] = []
    for f in tqdm(os.listdir(images_dir), desc="Listing images", unit="file"):
        p = Path(images_dir, f)
        # accept files with valid extension (case-insensitive)
        if p.suffix.lower() in VALID_EXTENSIONS:
            paths.append(p)
    # stable order for reproducibility
    paths.sort(key=lambda x: x.name)
    return paths

def load_siglip(model_name: str, device: torch.device):
    model, preprocess = create_model_from_pretrained(model_name)
    model.to(device)
    model.eval()
    return preprocess, model

class SiglipImageDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], preprocess) -> None:
        self.image_paths = list(image_paths)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
        pixel_values = self.preprocess(image)
        # return global index to restore ordering after distributed gather
        return {"pixel_values": pixel_values, "path": path, "index": index}

def collate_siglip_batch(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)
    paths = [item["path"] for item in batch]
    indices = torch.tensor([item["index"] for item in batch], dtype=torch.long)
    return {"pixel_values": pixel_values, "paths": paths, "indices": indices}

def create_dataloader(
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    sampler: DistributedSampler | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if sampler is not None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
        collate_fn=collate_siglip_batch,
    )

# ---------- Embeddings (single GPU) ----------

def compute_embeddings_single(
    dataset: SiglipImageDataset,
    model,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, List[Path]]:
    dl = create_dataloader(dataset, device, batch_size, num_workers, sampler=None)

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype) if use_amp else nullcontext()

    embs: List[np.ndarray] = []
    paths_all: List[Path] = []

    progress = tqdm(total=len(dataset), desc="Embedding (1 pass)", unit="img")
    with torch.no_grad():
        for batch in dl:
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            with autocast_ctx:
                feats = model.encode_image(pixels)  # (B, D)
            feats = feats.float()
            feats = F.normalize(feats, dim=-1)
            embs.append(feats.cpu().numpy().astype("float32", copy=False))
            paths_all.extend(batch["paths"])
            progress.update(feats.shape[0])
    progress.close()

    if not embs:
        raise RuntimeError("No embeddings computed.")
    embeddings = np.concatenate(embs, axis=0)
    return embeddings, paths_all

# ---------- Embeddings (multi-GPU via DDP) ----------

def compute_embeddings_distributed(
    dataset: SiglipImageDataset,
    model,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
) -> Tuple[np.ndarray | None, List[Path] | None]:
    """
    Each rank encodes a shard, then we all_gather embeddings + indices.
    Rank-0 reconstructs the (N,D) matrix in original dataset order and returns it.
    Other ranks return (None, None).
    """
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    dl = create_dataloader(dataset, device, batch_size, num_workers, sampler=sampler)

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype) if use_amp else nullcontext()

    local_embs: List[torch.Tensor] = []
    local_indices: List[torch.Tensor] = []

    # Only show per-rank bars if you really want; default: main-only
    progress = tqdm(total=len(sampler), desc=f"Rank {rank} embedding", unit="img", disable=(rank != 0))

    with torch.no_grad():
        for batch in dl:
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            idxs = batch["indices"].to(device, non_blocking=True)
            with autocast_ctx:
                feats = model.encode_image(pixels)  # (B, D)
            feats = feats.float()
            feats = F.normalize(feats, dim=-1)
            local_embs.append(feats.detach())
            local_indices.append(idxs.detach())
            progress.update(feats.shape[0])

    progress.close()

    # Concatenate local shards
    if local_embs:
        local_embs_t = torch.cat(local_embs, dim=0).contiguous()     # (m_i, D) on GPU
        local_idx_t = torch.cat(local_indices, dim=0).long().contiguous()  # (m_i,)
    else:
        # empty shard
        local_embs_t = torch.empty((0, 1), dtype=torch.float32, device=device)
        local_idx_t = torch.empty((0,), dtype=torch.int64, device=device)

    # Gather sizes across ranks
    m_i = torch.tensor([local_embs_t.shape[0]], dtype=torch.int64, device=device)
    sizes = [torch.zeros_like(m_i) for _ in range(world_size)]
    dist.all_gather(sizes, m_i)
    sizes_cpu = torch.stack(sizes).cpu()
    total = int(sizes_cpu.sum().item())
    max_m = int(sizes_cpu.max().item())

    # If there are no samples, exit early
    if total == 0:
        if rank == 0:
            return np.empty((0, 1), dtype=np.float32), dataset.image_paths
        return None, None

    # Dimension D: ensure non-empty local tensor available somewhere
    # Broadcast D from the first rank having data.
    local_D = torch.tensor([local_embs_t.shape[1] if m_i.item() > 0 else -1],
                           dtype=torch.int64, device=device)
    Ds = [torch.zeros_like(local_D) for _ in range(world_size)]
    dist.all_gather(Ds, local_D)
    D_vals = [int(d.item()) for d in Ds if int(d.item()) > 0]
    if not D_vals:
        # Shouldn't happen since total > 0, but guard anyway
        if rank == 0:
            return np.empty((0, 1), dtype=np.float32), dataset.image_paths
        return None, None
    D = D_vals[0]

    # Pad local to (max_m, D) for all_gather (NCCL requires equal shapes)
    if m_i.item() == 0:
        pad_embs = torch.zeros((max_m, D), dtype=torch.float32, device=device)
        pad_idx  = torch.full((max_m,), -1, dtype=torch.int64, device=device)
    else:
        pad_embs = torch.zeros((max_m, D), dtype=torch.float32, device=device)
        pad_embs[:m_i.item()] = local_embs_t
        pad_idx = torch.full((max_m,), -1, dtype=torch.int64, device=device)
        pad_idx[:m_i.item()] = local_idx_t

    # Gather all padded shards (GPU tensors)
    embs_list = [torch.empty((max_m, D), dtype=torch.float32, device=device) for _ in range(world_size)]
    idx_list  = [torch.empty((max_m,), dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_gather(embs_list, pad_embs)
    dist.all_gather(idx_list,  pad_idx)

    # Rank-0 reconstructs (N, D) in original order using gathered indices
    if rank == 0:
        embeddings = np.empty((total, D), dtype=np.float32)
        for r in range(world_size):
            cnt = int(sizes_cpu[r].item())
            if cnt == 0:
                continue
            emb_r = embs_list[r][:cnt].cpu().numpy()
            idx_r = idx_list[r][:cnt].cpu().numpy()
            embeddings[idx_r] = emb_r
        # Paths are already available on rank-0 via dataset.image_paths (sorted)
        return embeddings, dataset.image_paths

    # Non-zero ranks return nothing
    return None, None

# ---------- Duplicate discovery ----------

def find_duplicates_with_index(
    embeddings: np.ndarray,
    sim_threshold: float,
    use_gpu_index: bool,
) -> Dict[int, List[int]]:
    """
    Build a (CPU or single-GPU) FAISS index once and run a single k=2 search (self-search).
    We then union i with its nearest non-self neighbor if similarity >= sim_threshold.
    """
    n, d = embeddings.shape
    index = faiss.IndexFlatIP(d)  # cosine via inner product (embeddings are normalized)
    if use_gpu_index:
        try:
            if faiss.get_num_gpus() > 0:
                # Use a single GPU (safer with DDP) — default to LOCAL_RANK if present, else 0
                gpu_id = int(os.environ.get("FAISS_GPU", os.environ.get("LOCAL_RANK", "0")))
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, gpu_id, index)
        except Exception:
            # Fall back silently to CPU index
            pass

    index.add(embeddings)                      # DB
    sims, nbrs = index.search(embeddings, k=2) # Query = DB (self at pos 0)

    parent = list(range(n))
    size = [1] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if size[rx] < size[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        size[rx] += size[ry]

    for i in range(n):
        j = int(nbrs[i, 1])
        if j < 0:
            continue
        sim = float(sims[i, 1])
        if sim >= sim_threshold:
            union(i, j)

    comps: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        comps.setdefault(r, []).append(i)
    comps = {r: members for r, members in comps.items() if len(members) > 1}
    return comps

def find_duplicates_no_index(embeddings: np.ndarray, sim_threshold: float) -> Dict[int, List[int]]:
    n = embeddings.shape[0]
    bytes_est = n * n * 4
    if bytes_est > 2_000_000_000:  # ~2GB guard
        raise RuntimeError(
            f"--no-index mode requires O(N^2) memory; estimated {bytes_est/1e9:.1f} GB. "
            "Use the FAISS index path for large datasets."
        )
    sims = embeddings @ embeddings.T  # cosine since normalized
    np.fill_diagonal(sims, -np.inf)   # ignore self
    nbrs = np.argmax(sims, axis=1)

    parent = list(range(n))
    size = [1] * n
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if size[rx] < size[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        size[rx] += size[ry]

    for i in range(n):
        j = int(nbrs[i])
        if j >= 0 and sims[i, j] >= sim_threshold:
            union(i, j)

    comps: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        comps.setdefault(r, []).append(i)
    comps = {r: members for r, members in comps.items() if len(members) > 1}
    return comps

def plan_deletions(components: Dict[int, List[int]], image_paths: Sequence[Path]) -> List[int]:
    deletions: List[int] = []
    for members in components.values():
        members_sorted = sorted(members, key=lambda idx: image_paths[idx].name)
        deletions.extend(members_sorted[1:])
    return sorted(set(deletions))

# ---------- Deletions (multiprocessing) ----------

def _delete_set(dataset_dir: str, filename: str) -> int:
    base = Path(dataset_dir)
    deleted = 0
    for folder in RESOLUTION_FOLDERS:
        target = base / folder / filename
        try:
            if target.exists():
                target.unlink()
                deleted += 1
        except Exception:
            pass
    return deleted

def delete_images_parallel(
    indices_to_remove: Iterable[int],
    image_paths: Sequence[Path],
    dataset_dir: Path,
    max_workers: int,
) -> int:
    filenames = [image_paths[i].name for i in indices_to_remove]
    removed_files = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_delete_set, str(dataset_dir), fn) for fn in filenames]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Deleting", unit="file-set"):
            removed_files += fut.result()
    return removed_files

# ---------- Orchestration ----------

def run_cleanup(args: argparse.Namespace) -> DatasetStats | None:
    # DDP setup (no-op if not launched with torchrun)
    is_dist, rank, world, local = setup_distributed()

    # Rank-specific device
    if is_dist and torch.cuda.is_available():
        device = torch.device(f"cuda:{local}")
    else:
        device = resolve_device(args.device)

    dataset_dir = args.dataset_dir.resolve()
    hi_res_dir = dataset_dir / "512px"
    image_paths = list_images(hi_res_dir)
    if not image_paths:
        if is_dist:
            dist.barrier()
            dist.destroy_process_group()
        raise RuntimeError(f"No images found in {hi_res_dir}")

    preprocess, model = load_siglip(args.model_name, device)
    dataset = SiglipImageDataset(image_paths, preprocess)

    # Embeddings (distributed or single)
    if is_dist:
        embeddings, image_paths_out = compute_embeddings_distributed(
            dataset, model, device, args.batch_size, args.num_workers, rank, world
        )
        # Only rank-0 continues; others exit
        if rank != 0:
            dist.barrier()
            dist.destroy_process_group()
            return None
        # Rank-0 post-processing
        embeddings = embeddings  # type: ignore
        image_paths = image_paths_out  # type: ignore
        # Clean up process group before FAISS to free GPU mem on other ranks
        dist.barrier()
        dist.destroy_process_group()
    else:
        embeddings, image_paths = compute_embeddings_single(
            dataset, model, device, args.batch_size, args.num_workers
        )

    # One search over the same embedding set (rank-0 or single process)
    if args.no_index:
        components = find_duplicates_no_index(embeddings, args.sim_threshold)  # type: ignore
    else:
        components = find_duplicates_with_index(embeddings, args.sim_threshold, args.gpu_index)  # type: ignore

    indices_to_remove = plan_deletions(components, image_paths)  # type: ignore

    # Interactive confirmation (unless --yes or --dry-run)
    removed_files = 0
    if indices_to_remove:
        to_remove_imgs = len(indices_to_remove)
        to_remove_files_est = 0
        for idx in indices_to_remove:
            fn = image_paths[idx].name  # type: ignore
            to_remove_files_est += sum((dataset_dir / f / fn).exists() for f in RESOLUTION_FOLDERS)

        if not (args.dry_run or args.yes):
            print("\nPlanned deletions")
            print(f"  Duplicate groups            : {len(components)}")
            print(f"  512px images to remove      : {to_remove_imgs}")
            print(f"  Files to remove (all scales): ~{to_remove_files_est}")
            ans = input("Proceed with deletion? [y/N]: ").strip().lower()
            if ans not in ("y", "yes"):
                print("Aborted by user.")
                return DatasetStats(
                    total_images=len(image_paths),            # type: ignore
                    duplicate_groups=len(components),
                    removed_images=0,
                    removed_files=0,
                )

        if not args.dry_run:
            removed_files = delete_images_parallel(
                indices_to_remove, image_paths, dataset_dir, max_workers=args.delete_workers  # type: ignore
            )

    stats = DatasetStats(
        total_images=len(image_paths),                 # type: ignore
        duplicate_groups=len(components),
        removed_images=len(indices_to_remove),
        removed_files=removed_files,
    )
    return stats

def print_summary(stats: DatasetStats, dry_run: bool, sim_threshold: float) -> None:
    action = "would remove" if dry_run else "removed"
    print("\nDataset cleaning summary")
    print(f"  Total 512px crops scanned : {stats.total_images}")
    print(f"  Duplicate groups found    : {stats.duplicate_groups}")
    print(f"  512px images {action}     : {stats.removed_images}")
    if not dry_run:
        print(f"  Files deleted (all scales): {stats.removed_files}")
    print(f"  Cosine similarity thr     : {sim_threshold}")

def main() -> None:
    args = parse_args()
    stats = run_cleanup(args)
    if stats is not None:
        print_summary(stats, args.dry_run, args.sim_threshold)

if __name__ == "__main__":
    main()
