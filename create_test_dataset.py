from __future__ import annotations

import ast
import os
import shutil
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


CSV_PATH = Path("clear_images_with_maniqa_scores.csv")
DESTINATION_ROOT = Path("upscale_df_quality_10k_per_metric_v1")
COUNT_PER_METRIC = 10000
SAMPLE_SIZE = 10  # set to 0 to disable additional sampling
MODEL_NAMES: Sequence[str] = ("flux_vae", "sd3_vae_anime_ft")
RESIZE_LONG_SIDE = 1024
RESIZE_DIR_NAME = "resized_1024"
RESIZE_WORKERS = 120
EMBEDDING_WORKERS = 40
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


def _iter_paths_maybe_nested(items: Iterable[Any]) -> Iterator[Path]:
    for item in items:
        if isinstance(item, (list, tuple, set)):
            yield from _iter_paths_maybe_nested(item)
        elif isinstance(item, (Path, str)):
            yield Path(item)


def _filter_by_model(paths: Iterable[Path], model_names: Optional[Sequence[str]]) -> Iterator[Path]:
    if not model_names:
        yield from paths
        return
    for candidate in paths:
        candidate_str = str(candidate)
        if any(name in candidate_str for name in model_names):
            yield candidate


def _resize_task(job: Tuple[str, str, int]) -> Tuple[str, Optional[str]]:
    src, dst, long_side = job
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
                Path(dst).parent.mkdir(parents=True, exist_ok=True)
                image.save(dst)
        return src, None
    except UserWarning:
        return src, "skipped: palette transparency (bytes)"
    except Exception as exc:  # pylint: disable=broad-except
        return src, str(exc)


def _copy_embedding_task(job: Tuple[str, str, bool]) -> Tuple[str, Optional[str]]:
    src, dst, overwrite = job
    try:
        dst_path = Path(dst)
        if dst_path.exists() and not overwrite:
            return src, "skipped"
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_path)
        return src, None
    except Exception as exc:  # pylint: disable=broad-except
        return src, str(exc)


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


def process_images_and_embeddings(
    *,
    image_paths: Iterable[Any],
    embedding_paths: Iterable[Any],
    destination_root: Path,
    overwrite: bool,
    verbose: bool,
    resize_long_side: int,
    resize_dir_name: str,
    resize_workers: Optional[int],
    embedding_workers: Optional[int],
    model_names: Optional[Sequence[str]],
) -> Dict[str, str]:
    images = [Path(path) for path in _iter_paths_maybe_nested(image_paths)]
    flat_embeddings = list(_filter_by_model(_iter_paths_maybe_nested(embedding_paths), model_names))

    path_mapping: Dict[str, str] = {}

    if not images and not flat_embeddings:
        if verbose:
            print("Nothing to process.")
        return path_mapping

    if resize_long_side <= 0:
        raise ValueError("resize_long_side must be positive")

    dest_root = destination_root.expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    img_ok = img_err = img_skipped = 0
    if images:
        try:
            dataset_root = Path(os.path.commonpath([str(p.parent) for p in images]))
        except ValueError:
            dataset_root = images[0].parent

        resize_root = dest_root / resize_dir_name
        resize_root.mkdir(parents=True, exist_ok=True)

        jobs: List[Tuple[str, str, int]] = []
        for img_path in tqdm(images, desc="Preparing resize jobs", leave=False):
            dst = resize_root / img_path.relative_to(dataset_root)
            if dst.exists() and not overwrite:
                continue
            jobs.append((str(img_path), str(dst), int(resize_long_side)))

        if jobs:
            worker_count = max(1, min(resize_workers or cpu_count(), cpu_count()))
            if verbose:
                print(
                    f"Resizing {len(jobs)} images with {worker_count} worker(s) to max {resize_long_side}px"
                )
            with Pool(processes=worker_count) as pool, tqdm(total=len(jobs), desc="Resizing images", leave=False) as bar:
                for _, message in pool.imap_unordered(_resize_task, jobs):
                    if message is None:
                        img_ok += 1
                    elif message.startswith("skipped:"):
                        img_skipped += 1
                    else:
                        img_err += 1
                    bar.update(1)
        if verbose:
            print(
                f"Images -> ok: {img_ok}, skipped (palette byte transparency): {img_skipped}, errors: {img_err}. "
                f"Output dir: {resize_root}"
            )

    copied = skipped = failed = 0
    copy_jobs: List[Tuple[str, str, bool]] = []
    seen_destinations: Set[Path] = set()

    for emb_path in tqdm(flat_embeddings, desc="Preparing embedding copy jobs", leave=False):
        try:
            rel = _rel_after_cache_root(emb_path)
        except ValueError:
            failed += 1
            continue

        dst_path = dest_root / "cache_vae_embeddings" / rel
        src_key = str(emb_path)
        dst_value = str(dst_path)

        path_mapping[src_key] = dst_value

        if not overwrite and dst_path in seen_destinations:
            skipped += 1
            continue

        seen_destinations.add(dst_path)
        copy_jobs.append((str(emb_path), str(dst_path), overwrite))

    if copy_jobs:
        worker_count = max(1, min(embedding_workers or cpu_count(), cpu_count()))
        if verbose:
            print(f"Copying {len(copy_jobs)} embeddings with {worker_count} worker(s)")
        with Pool(processes=worker_count) as pool, tqdm(total=len(copy_jobs), desc="Copying embeddings", leave=False) as bar:
            for _, message in pool.imap_unordered(_copy_embedding_task, copy_jobs, chunksize=32):
                if message is None:
                    copied += 1
                elif message == "skipped":
                    skipped += 1
                else:
                    failed += 1
                bar.update(1)
        if verbose:
            print(f"Embeddings -> copied: {copied}, skipped: {skipped}, failed: {failed}.")

    return path_mapping


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

    path_mapping = process_images_and_embeddings(
        image_paths=quality_df["path"],
        embedding_paths=parsed_embeddings,
        destination_root=dest_root,
        overwrite=OVERWRITE_EXISTING,
        verbose=VERBOSE,
        resize_long_side=RESIZE_LONG_SIDE,
        resize_dir_name=RESIZE_DIR_NAME,
        resize_workers=RESIZE_WORKERS,
        embedding_workers=EMBEDDING_WORKERS,
        model_names=MODEL_NAMES,
    )

    quality_df = quality_df.copy()
    quality_df["available_embeddings"] = [
        [path_mapping.get(str(Path(emb)), emb) for emb in embeddings]
        for embeddings in parsed_embeddings
    ]

    output_csv = dest_root / "quality_df_with_emb_paths.csv"
    quality_df.to_csv(output_csv, index=False)
    if VERBOSE:
        print(f"Saved updated DataFrame with embedding paths to {output_csv}")


if __name__ == "__main__":
    main()
