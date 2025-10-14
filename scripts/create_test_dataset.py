from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm.auto import tqdm


@dataclass
class EmbeddingInfo:
    model: str
    resolution: str
    path: Path


def _parse_required(items: Sequence[str]) -> Dict[str, List[str]]:
    requirements: Dict[str, List[str]] = {}
    for entry in items:
        if ":" not in entry:
            raise ValueError(
                "Required resolution entries must use the form 'model:res1,res2'"
            )
        model, values = entry.split(":", 1)
        parts = [part.strip() for part in values.split(",") if part.strip()]
        if not parts:
            raise ValueError(f"No resolutions provided for '{model}'")
        requirements[model.strip()] = parts
    return requirements


def _normalise_models(models: Sequence[str] | None) -> Optional[List[str]]:
    if not models:
        return None
    return [model.strip() for model in models if model.strip()]


def _embedding_key(cache_root: Path, embedding_path: Path) -> Tuple[str, EmbeddingInfo] | None:
    try:
        relative = embedding_path.resolve().relative_to(cache_root)
    except ValueError:
        return None

    parts = list(relative.parts)
    if len(parts) < 3:
        return None

    model = parts[0]
    try:
        resolution_index = next(i for i, part in enumerate(parts[1:], start=1) if part.endswith("px"))
    except StopIteration:
        return None

    resolution = parts[resolution_index]
    remainder = Path(*parts[resolution_index + 1 :])
    if not remainder:
        return None

    key = str(remainder.with_suffix(""))
    return key, EmbeddingInfo(model=model, resolution=resolution, path=embedding_path.resolve())


def _build_embedding_index(
    cache_root: Path,
    *,
    models: Optional[Sequence[str]],
    extension: str,
) -> Dict[str, List[EmbeddingInfo]]:
    embeddings: Dict[str, List[EmbeddingInfo]] = {}
    model_filter = set(models) if models else None
    for file_path in tqdm(
        list(cache_root.rglob(f"*{extension}")),
        desc="Scanning embeddings",
        leave=False,
    ):
        result = _embedding_key(cache_root, file_path)
        if result is None:
            continue
        key, info = result
        if model_filter and info.model not in model_filter:
            continue
        embeddings.setdefault(key, []).append(info)
    return embeddings


def _path_key(dataset_root: Path, image_path: Path) -> str:
    relative = image_path.resolve().relative_to(dataset_root)
    return str(relative.with_suffix(""))


def _meets_requirements(
    infos: Iterable[EmbeddingInfo],
    requirements: Dict[str, List[str]],
) -> bool:
    if not requirements:
        return True
    found: Dict[str, set[str]] = {}
    for info in infos:
        found.setdefault(info.model, set()).add(info.resolution)
    for model, needed in requirements.items():
        present = found.get(model, set())
        if any(resolution not in present for resolution in needed):
            return False
    return True


def _filter_dataframe(
    df: pd.DataFrame,
    *,
    stage: Optional[str],
    max_bytes: Optional[int],
    max_dimension_sum: Optional[int],
) -> pd.DataFrame:
    result = df
    if stage:
        result = result[result.get("stage") == stage]
    columns = {"width", "height", "size_bytes"}
    missing = [column for column in columns if column not in result]
    if missing:
        raise KeyError(f"Missing required column(s) in CSV: {', '.join(missing)}")
    result = result.dropna(subset=list(columns))
    if max_bytes is not None:
        result = result[result["size_bytes"] <= max_bytes]
    if max_dimension_sum is not None:
        result = result[(result["width"] + result["height"]) <= max_dimension_sum]
    return result.reset_index(drop=True)


def _connect_embeddings(
    df: pd.DataFrame,
    *,
    dataset_root: Path,
    embedding_index: Dict[str, List[EmbeddingInfo]],
    requirements: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, List[List[str]], Dict[str, int]]:
    available_embeddings: List[List[str]] = []
    stats = {"matched": 0, "missing": 0, "filtered": 0}

    for image_path_str in tqdm(df["path"], desc="Matching embeddings", leave=False):
        image_path = Path(image_path_str)
        key = _path_key(dataset_root, image_path)
        candidates = embedding_index.get(key, [])
        if not candidates:
            available_embeddings.append([])
            stats["missing"] += 1
            continue
        if requirements and not _meets_requirements(candidates, requirements):
            available_embeddings.append([])
            stats["filtered"] += 1
            continue
        available_embeddings.append([str(info.path) for info in candidates])
        stats["matched"] += 1

    df = df.assign(available_embeddings=available_embeddings)
    df = df[df["available_embeddings"].map(bool)].reset_index(drop=True)
    return df, available_embeddings, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Connect images with cached embeddings and emit a test dataset CSV",
    )
    parser.add_argument("--input-csv", type=Path, required=True, help="Source CSV containing image metadata")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory containing the original images referenced by the CSV",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="Root directory of the cached embeddings (e.g. cache_vae_embeddings)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Destination CSV path; defaults to <input>-with-embeddings.csv",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional whitelist of model names to include (matches cache folder names)",
    )
    parser.add_argument(
        "--required",
        nargs="*",
        default=None,
        help="Resolution requirements per model in the form model:128px,256px",
    )
    parser.add_argument(
        "--stage",
        default=None,
        help="Optional stage value to filter the CSV (e.g. 'validate')",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="Discard rows whose size_bytes exceed this threshold",
    )
    parser.add_argument(
        "--max-dimension-sum",
        type=int,
        default=None,
        help="Discard rows whose width+height exceed this threshold",
    )
    parser.add_argument(
        "--extension",
        default=".pt",
        help="Embedding file extension to scan for (default: .pt)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"CSV file not found: {args.input_csv}")
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")
    if not args.cache_root.exists():
        raise FileNotFoundError(f"Embedding cache root not found: {args.cache_root}")

    models = _normalise_models(args.models)
    required = _parse_required(args.required) if args.required else {}
    if required and models:
        unknown = sorted(set(required).difference(models))
        if unknown:
            raise ValueError(
                f"Required resolutions provided for unknown model(s): {', '.join(unknown)}"
            )

    df = pd.read_csv(args.input_csv)
    df = _filter_dataframe(
        df,
        stage=args.stage,
        max_bytes=args.max_bytes,
        max_dimension_sum=args.max_dimension_sum,
    )
    if df.empty:
        raise RuntimeError("No rows remaining after filtering; cannot build dataset")

    embedding_index = _build_embedding_index(
        args.cache_root.expanduser().resolve(),
        models=models,
        extension=args.extension,
    )

    dataset_root = args.dataset_root.expanduser().resolve()
    df, _, stats = _connect_embeddings(
        df,
        dataset_root=dataset_root,
        embedding_index=embedding_index,
        requirements=required,
    )

    if df.empty:
        raise RuntimeError("No rows satisfy the embedding requirements")

    output_csv = args.output_csv
    if output_csv is None:
        output_csv = args.input_csv.with_name(f"{args.input_csv.stem}-with-embeddings.csv")
    output_csv = output_csv.expanduser().resolve()

    df.to_csv(output_csv, index=False)

    total = stats["matched"] + stats["missing"] + stats["filtered"]
    print(
        "Dataset prepared:"
        f" {stats['matched']} matched, {stats['filtered']} filtered, {stats['missing']} missing"
        f" (from {total} rows)."
    )
    print(f"Saved dataset CSV to {output_csv}")


if __name__ == "__main__":
    main()
