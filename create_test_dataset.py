from __future__ import annotations

import ast
import shutil
import warnings
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


CSV_PATH = Path("clear_images_with_maniqa_scores.csv")
DESTINATION_ROOT = Path("upscale_df_quality_100k_per_metric_v3")
COUNT_PER_METRIC = 100000
SAMPLE_SIZE = 10  # set to 0 to disable additional sampling
MODEL_NAMES: Sequence[str] = ("flux_vae", "sd3_vae_anime_ft")
# Required resolutions for each model - all must be present for an image to be included
REQUIRED_RESOLUTIONS: Dict[str, List[str]] = {
    "flux_vae": ["128px", "256px", "512px"],
    "sd3_vae_anime_ft": ["128px", "256px", "512px"],
}
RESIZE_LONG_SIDE = 1024
IMAGE_WORKERS = 120
EMBEDDING_WORKERS = 40
DATASET_DIR_NAME = "dataset"
IMAGES_SUBDIR = "images"
EMBEDDINGS_SUBDIR = "train_embeddings"
OVERWRITE_EXISTING = False
REMOVE_DESTINATION_FIRST = True  # mirrors the explicit `rm -rf` step in the notebook
VERBOSE = True


# --- helpers copied from the notebook implementation ---
def _rel_after_cache_root(path: Path) -> Path:
    parts = path.parts
    if "cache_vae_embeddings" not in parts:
        raise ValueError(f"'cache_vae_embeddings' not in path: {path}")
    return Path(*parts[parts.index("cache_vae_embeddings") + 1 :])


def _new_size(width: int, height: int, long_side: int) -> Tuple[int, int]:
    longest = max(width, height)
    if longest <= long_side:
        return width, height
    scale = long_side / float(longest)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _should_include_model(path: Path, model_names: Optional[Sequence[str]]) -> bool:
    if not model_names:
        return True
    candidate = str(path)
    return any(name in candidate for name in model_names)


def _copy_and_resize_image(src: Path, dst: Path, long_side: int, overwrite: bool) -> Optional[str]:
    if dst.exists() and not overwrite:
        return "skipped"
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                category=UserWarning,
                message=r"Palette images with Transparency expressed in bytes.*",
            )
            with Image.open(src) as image:
                image = image.convert("RGB")
                new_size = _new_size(*image.size, long_side)
                if new_size != image.size:
                    image = image.resize(new_size, Image.LANCZOS)
                dst.parent.mkdir(parents=True, exist_ok=True)
                image.save(dst)
        return None
    except UserWarning:
        return "skipped: palette transparency (bytes)"
    except Exception as exc:  # pylint: disable=broad-except
        return str(exc)


def _copy_embedding_file(src: Path, dst: Path, overwrite: bool) -> Optional[str]:
    if dst.exists() and not overwrite:
        return "skipped"
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return None
    except Exception as exc:  # pylint: disable=broad-except
        return str(exc)


def _image_job_worker(job: Tuple[int, str, str, int, bool]) -> Tuple[int, str, str, Optional[str]]:
    row_index, src, dst, long_side, overwrite = job
    message = _copy_and_resize_image(Path(src), Path(dst), long_side, overwrite)
    return row_index, src, dst, message


def _embedding_job_worker(job: Tuple[int, int, str, str, bool]) -> Tuple[int, int, str, str, Optional[str]]:
    row_index, order, src, dst, overwrite = job
    message = _copy_embedding_file(Path(src), Path(dst), overwrite)
    return row_index, order, src, dst, message


def _extract_model_and_size(relative_path: Path) -> Tuple[str, str]:
    parts = list(relative_path.parts)
    model = parts[0] if parts else "unknown_model"
    size = next((part for part in parts[1:] if part.endswith("px")), "default")
    return model, size


def _has_complete_embeddings(
    embedding_list: List[str],
    model_names: Optional[Sequence[str]],
    required_resolutions: Dict[str, List[str]],
) -> bool:
    """Check if the embedding list contains all required model/resolution combinations."""
    if not model_names or not required_resolutions:
        return True
    
    # Track which (model, resolution) pairs we've found
    found_combinations = set()
    
    for embedding_src_str in embedding_list:
        embedding_src_path = Path(embedding_src_str)
        
        if not _should_include_model(embedding_src_path, model_names):
            continue
        
        try:
            relative_after_cache = _rel_after_cache_root(embedding_src_path)
            model_name, resolution = _extract_model_and_size(relative_after_cache)
            found_combinations.add((model_name, resolution))
        except ValueError:
            continue
    
    # Check if all required combinations are present
    for model in model_names:
        if model not in required_resolutions:
            continue
        for resolution in required_resolutions[model]:
            if (model, resolution) not in found_combinations:
                return False
    
    return True


def _destination_roots(destination_root: Path) -> Tuple[Path, Path, Path]:
    dataset_root = destination_root / DATASET_DIR_NAME
    image_root = dataset_root / IMAGES_SUBDIR
    embedding_root = dataset_root / EMBEDDINGS_SUBDIR
    image_root.mkdir(parents=True, exist_ok=True)
    embedding_root.mkdir(parents=True, exist_ok=True)
    return dataset_root, image_root, embedding_root


def _load_quality_dataframe(csv_path: Path, count_per_metric: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.assign(clip_hyper_avg=(df["clip_iqa"] + df["hyperiqa"]) / 2)

    top_clip = df.nlargest(count_per_metric, "clip_iqa", keep="first")
    top_hyper = df.nlargest(count_per_metric, "hyperiqa", keep="first")
    top_avg = df.nlargest(count_per_metric, "clip_hyper_avg", keep="first")

    combined = pd.concat([top_clip, top_hyper, top_avg], axis=0)
    combined.drop_duplicates(inplace=True)
    return combined


def _parse_embedding_column(raw_value: Any) -> List[str]:
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value]
    if isinstance(raw_value, str) and raw_value.strip():
        try:
            evaluated = ast.literal_eval(raw_value)
        except (SyntaxError, ValueError):
            return [raw_value]
        if isinstance(evaluated, (list, tuple, set)):
            return [str(item) for item in evaluated]
        if isinstance(evaluated, (str, Path)):
            return [str(evaluated)]
    return []


def build_dataset(
    *,
    quality_df: pd.DataFrame,
    parsed_embeddings: List[List[str]],
    destination_root: Path,
    overwrite: bool,
    verbose: bool,
    resize_long_side: int,
    model_names: Optional[Sequence[str]],
    required_resolutions: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, List[str], List[List[str]], Dict[str, Dict[str, int]], Path, Path, Path]:
    if resize_long_side <= 0:
        raise ValueError("resize_long_side must be positive")

    dest_root = destination_root.expanduser().resolve()
    dataset_root, image_root, embedding_root = _destination_roots(dest_root)

    # Filter to keep only images with complete embedding sets
    if verbose:
        print("Filtering images to keep only those with complete embedding sets...")
    
    filtered_indices = []
    for idx, embedding_list in enumerate(parsed_embeddings):
        if _has_complete_embeddings(embedding_list, model_names, required_resolutions):
            filtered_indices.append(idx)
    
    if verbose:
        print(f"Filtered: {len(quality_df)} -> {len(filtered_indices)} images with complete embeddings")
        dropped = len(quality_df) - len(filtered_indices)
        if dropped > 0:
            print(f"Dropped {dropped} images due to missing embeddings")
    
    # Update dataframe and embeddings to only include filtered rows
    quality_df = quality_df.iloc[filtered_indices].reset_index(drop=True)
    parsed_embeddings = [parsed_embeddings[i] for i in filtered_indices]

    image_stats = {"ok": 0, "skipped": 0, "failed": 0, "filtered_out": len(quality_df) - len(filtered_indices)}
    embedding_stats = {"ok": 0, "skipped": 0, "failed": 0}

    row_iterator = zip(quality_df["path"], parsed_embeddings)
    progress = tqdm(row_iterator, total=len(quality_df), desc="Preparing jobs", leave=False)

    new_image_paths: List[str] = []
    embedding_placeholders: List[List[Optional[str]]] = []
    image_jobs: List[Tuple[int, str, str, int, bool]] = []
    embedding_jobs: List[Tuple[int, int, str, str, bool]] = []

    for row_index, (image_src, embedding_list) in enumerate(progress):
        image_src_path = Path(image_src)
        image_dst_path = image_root / f"{row_index}.png"

        new_image_paths.append(str(image_dst_path))
        embedding_placeholders.append([])

        image_jobs.append(
            (
                row_index,
                str(image_src_path),
                str(image_dst_path),
                resize_long_side,
                overwrite,
            )
        )

        per_bucket_counter: Dict[Tuple[str, str], int] = defaultdict(int)

        for embedding_src_str in embedding_list:
            embedding_src_path = Path(embedding_src_str)
            if not _should_include_model(embedding_src_path, model_names):
                continue

            try:
                relative_after_cache = _rel_after_cache_root(embedding_src_path)
            except ValueError:
                embedding_stats["failed"] += 1
                if verbose:
                    print(f"Embedding path does not include cache root: {embedding_src_path}")
                continue

            model_name, resolution = _extract_model_and_size(relative_after_cache)
            suffix = "".join(embedding_src_path.suffixes) or ".pt"

            bucket_key = (model_name, resolution)
            order_within_bucket = per_bucket_counter[bucket_key]
            per_bucket_counter[bucket_key] += 1

            filename = f"{row_index}" if order_within_bucket == 0 else f"{row_index}_{order_within_bucket}"
            embedding_dst_path = (
                embedding_root / model_name / resolution / f"{filename}{suffix}"
            )

            order_in_row = len(embedding_placeholders[row_index])
            embedding_placeholders[row_index].append(None)
            embedding_jobs.append(
                (
                    row_index,
                    order_in_row,
                    str(embedding_src_path),
                    str(embedding_dst_path),
                    overwrite,
                )
            )

    if image_jobs:
        desired = IMAGE_WORKERS or cpu_count()
        worker_count = max(1, min(desired, cpu_count()))
        if verbose:
            print(
                f"Resizing {len(image_jobs)} images with {worker_count} worker(s) to max {resize_long_side}px"
            )
        with Pool(processes=worker_count) as pool, tqdm(
            total=len(image_jobs), desc="Resizing images", leave=False
        ) as bar:
            for row_index, src, dst, message in pool.imap_unordered(_image_job_worker, image_jobs):
                if message is None:
                    image_stats["ok"] += 1
                elif message.startswith("skipped"):
                    image_stats["skipped"] += 1
                else:
                    image_stats["failed"] += 1
                    if verbose:
                        print(f"Failed to process image {src} -> {dst}: {message}")
                bar.update(1)

    if embedding_jobs:
        desired = EMBEDDING_WORKERS or cpu_count()
        worker_count = max(1, min(desired, cpu_count()))
        if verbose:
            print(f"Copying {len(embedding_jobs)} embeddings with {worker_count} worker(s)")
        with Pool(processes=worker_count) as pool, tqdm(
            total=len(embedding_jobs), desc="Copying embeddings", leave=False
        ) as bar:
            for row_index, order, src, dst, message in pool.imap_unordered(
                _embedding_job_worker, embedding_jobs, chunksize=32
            ):
                if message is None:
                    embedding_stats["ok"] += 1
                    embedding_placeholders[row_index][order] = dst
                elif message == "skipped":
                    embedding_stats["skipped"] += 1
                    embedding_placeholders[row_index][order] = dst
                else:
                    embedding_stats["failed"] += 1
                    if verbose:
                        print(f"Failed to copy embedding {src} -> {dst}: {message}")
                bar.update(1)

    summaries = {"images": image_stats, "embeddings": embedding_stats}
    final_embedding_paths = [
        [path for path in row if path is not None] for row in embedding_placeholders
    ]
    return (
        quality_df,
        new_image_paths,
        final_embedding_paths,
        summaries,
        dataset_root,
        image_root,
        embedding_root,
    )


def main() -> None:
    csv_path = CSV_PATH.expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if VERBOSE:
        print(f"Loading metrics from {csv_path}")

    dest_root = DESTINATION_ROOT.expanduser().resolve()

    if dest_root.exists() and REMOVE_DESTINATION_FIRST:
        if VERBOSE:
            print(f"Removing existing destination {dest_root}")
        shutil.rmtree(dest_root)

    quality_df = _load_quality_dataframe(csv_path, COUNT_PER_METRIC)

    # if SAMPLE_SIZE:
    #     sample_n = min(len(quality_df), max(1, SAMPLE_SIZE))
    #     quality_df = quality_df.sample(n=sample_n, random_state=42)
    #     if VERBOSE:
    #         print(f"Down-sampled to {sample_n} rows for processing")

    if VERBOSE:
        print(f"Total unique rows after merging metrics: {len(quality_df)}")

    parsed_embeddings = quality_df["available_embeddings"].apply(_parse_embedding_column).tolist()

    (
        filtered_df,
        new_image_paths,
        new_embedding_paths,
        summaries,
        dataset_root,
        image_root,
        embedding_root,
    ) = build_dataset(
        quality_df=quality_df,
        parsed_embeddings=parsed_embeddings,
        destination_root=dest_root,
        overwrite=OVERWRITE_EXISTING,
        verbose=VERBOSE,
        resize_long_side=RESIZE_LONG_SIDE,
        model_names=MODEL_NAMES,
        required_resolutions=REQUIRED_RESOLUTIONS,
    )

    if VERBOSE:
        image_summary = summaries["images"]
        embedding_summary = summaries["embeddings"]
        print(
            f"Images -> ok: {image_summary['ok']}, skipped: {image_summary['skipped']}, failed: {image_summary['failed']}. "
            f"Output dir: {image_root}"
        )
        print(
            f"Embeddings -> ok: {embedding_summary['ok']}, skipped: {embedding_summary['skipped']}, failed: {embedding_summary['failed']}. "
            f"Output dir: {embedding_root}"
        )

    quality_df = filtered_df.copy()
    quality_df["path"] = new_image_paths
    quality_df["available_embeddings"] = new_embedding_paths

    output_csv = dataset_root / "quality_df_with_emb_paths.csv"
    quality_df.to_csv(output_csv, index=False)
    if VERBOSE:
        print(f"Saved updated DataFrame with embedding paths to {output_csv}")


if __name__ == "__main__":
    main()
