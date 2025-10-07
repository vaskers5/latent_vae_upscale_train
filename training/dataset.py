"""Dataset utilities used by the VAE trainer."""

from __future__ import annotations

import glob
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from .embeddings import EmbeddingCache, TransformParams

__all__ = ["ImageFolderDataset", "UpscaleDataset"]


CACHE_FILENAME = ".image_folder_cache.json"
CACHE_VERSION = 1


def _verify_image_path(path: str) -> bool:
    """Return True if the PIL reader can successfully verify the image."""

    try:
        with Image.open(path) as img:
            img.verify()
    except (OSError, UnidentifiedImageError):
        return False
    return True


class ImageFolderDataset(Dataset):
    """Recursively reads images and returns centred crops normalised to [-1, 1]."""

    def __init__(
        self,
        root: Path,
        high_resolution: int,
        resize_long_side: int = 0,
        limit: int = 0,
        embedding_cache: Optional[EmbeddingCache] = None,
        model_resolution: int = 0,
    ) -> None:
        self.root = Path(root)
        self.high_resolution = high_resolution
        self.resize_long_side = resize_long_side
        self.cache_path = self.root / CACHE_FILENAME
        self.paths: List[Path] = []
        self.embedding_cache = embedding_cache
        self.model_resolution = model_resolution
        self._needs_model_downsample = (
            self.model_resolution > 0 and self.model_resolution != self.high_resolution
        )

        self.paths = self._collect_valid_paths(limit=limit)
        if not self.paths:
            raise RuntimeError(f"No valid images found under '{self.root}'")
        random.shuffle(self.paths)

    def _collect_valid_paths(self, limit: int = 0) -> List[Path]:
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        candidates: List[Path] = []
        for current_root, _dirs, files in os.walk(self.root):
            for name in files:
                if Path(name).suffix.lower() in exts:
                    candidates.append(Path(current_root) / name)
        if limit:
            candidates = candidates[:limit]

        cache_data = self._load_cache()
        cached_files: Dict[str, Dict[str, Any]] = cache_data.setdefault("files", {})
        valid_paths: List[Path] = []
        to_verify: List[Tuple[Path, str, float, int]] = []
        seen_rel_paths = set()
        cache_updated = False

        for path in candidates:
            try:
                stat = path.stat()
            except OSError:
                rel = self._relative_key(path)
                if rel in cached_files:
                    del cached_files[rel]
                    cache_updated = True
                continue

            rel_path = self._relative_key(path)
            seen_rel_paths.add(rel_path)
            cache_entry = cached_files.get(rel_path)
            size = stat.st_size
            mtime = stat.st_mtime

            if (
                cache_entry
                and cache_entry.get("size") == size
                and cache_entry.get("mtime") == mtime
                and cache_entry.get("valid", False)
            ):
                valid_paths.append(path)
                continue

            if (
                cache_entry
                and cache_entry.get("size") == size
                and cache_entry.get("mtime") == mtime
                and not cache_entry.get("valid", False)
            ):
                # Known invalid image with unchanged metadata; skip verification.
                continue

            to_verify.append((path, rel_path, mtime, size))

        if to_verify:
            verification_map = self._verify_paths([item[0] for item in to_verify])
            for path, rel_path, mtime, size in to_verify:
                is_valid = verification_map.get(str(path), False)
                cached_files[rel_path] = {
                    "mtime": mtime,
                    "size": size,
                    "valid": is_valid,
                }
                cache_updated = True
                if is_valid:
                    valid_paths.append(path)

        removed_keys = [key for key in cached_files.keys() if key not in seen_rel_paths]
        if removed_keys:
            for key in removed_keys:
                del cached_files[key]
            cache_updated = True

        if cache_updated:
            cache_data["version"] = CACHE_VERSION
            self._save_cache(cache_data)

        return valid_paths

    def _verify_paths(self, paths: Iterable[Path]) -> Dict[str, bool]:
        path_list = list(paths)
        if not path_list:
            return {}

        workers = max(1, min(cpu_count(), len(path_list)))
        results: Dict[str, bool] = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for path, is_valid in zip(
                path_list, executor.map(_verify_image_path, (str(p) for p in path_list))
            ):
                results[str(path)] = is_valid
        return results

    def _load_cache(self) -> Dict[str, Any]:
        try:
            with self.cache_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {"version": CACHE_VERSION, "files": {}}

        if data.get("version") != CACHE_VERSION:
            return {"version": CACHE_VERSION, "files": {}}
        files = data.get("files")
        if not isinstance(files, dict):
            return {"version": CACHE_VERSION, "files": {}}
        return data

    def _save_cache(self, data: Dict[str, Any]) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as file:
                json.dump(data, file)
            os.replace(tmp_path, self.cache_path)
        except OSError:
            # Swallow cache write errors to avoid disrupting dataset usage.
            pass

    def _relative_key(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()

    def set_embedding_cache(self, cache: Optional[EmbeddingCache]) -> None:
        self.embedding_cache = cache

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index % len(self.paths)]
        if self.embedding_cache is None or not self.embedding_cache.cfg.enabled:
            tensor, _params = self.build_tensor_sample(path)
            return {
                "image": tensor,
                "model_input": self.prepare_model_input(tensor),
            }

        record = self.embedding_cache.choose_record(path)
        tensor, params = self.build_tensor_sample(path, params=record.params)
        sample = {
            "image": tensor,
            "model_input": self.prepare_model_input(tensor),
            "latents": record.latents,
            "path": str(path),
            "variant_index": record.variant_index,
        }
        if record.mean is not None:
            sample["latent_mean"] = record.mean
        if record.logvar is not None:
            sample["latent_logvar"] = record.logvar
        return sample

    def build_tensor_sample(
        self,
        path: Path,
        rng: Optional[random.Random] = None,
        params: Optional[TransformParams] = None,
    ) -> Tuple[torch.Tensor, TransformParams]:
        with Image.open(path) as img:
            img = img.convert("RGB")
            transformed, used_params = self._apply_transforms(img, rng=rng, params=params)
        tensor = TF.to_tensor(transformed)
        tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return tensor, used_params

    def prepare_model_input(self, tensor: torch.Tensor) -> torch.Tensor:
        """Produce the tensor fed into the model, applying downsampling if needed."""

        return self._downsample_for_model(tensor)

    def _downsample_for_model(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self._needs_model_downsample:
            return tensor
        batched = tensor.unsqueeze(0)
        resized = F.interpolate(
            batched,
            size=(self.model_resolution, self.model_resolution),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0)

    def _resize_if_needed(self, img: Image.Image) -> Image.Image:
        if self.resize_long_side <= 0:
            return img
        width, height = img.size
        longest = max(width, height)
        if longest <= self.resize_long_side:
            return img
        scale = self.resize_long_side / float(longest)
        new_size = (int(round(width * scale)), int(round(height * scale)))
        return img.resize(new_size, Image.LANCZOS)

    def _apply_transforms(
        self,
        img: Image.Image,
        rng: Optional[random.Random] = None,
        params: Optional[TransformParams] = None,
    ) -> Tuple[Image.Image, TransformParams]:
        del rng, params

        if self.high_resolution <= 0:
            return img, TransformParams(flip=False, crop_x=0, crop_y=0)

        img = self._resize_if_needed(img)
        img = ImageOps.fit(
            img,
            (self.high_resolution, self.high_resolution),
            Image.LANCZOS,
            centering=(0.5, 0.5),
        )
        return img, TransformParams(flip=False, crop_x=0, crop_y=0)


class UpscaleDataset(Dataset):
    """Dataset that loads precomputed low/high resolution tensor pairs."""

    def __init__(self, cache_dir: str, low_res: int, high_res: int) -> None:
        self.pairs = []
        low_res_files = glob.glob(
            f"{cache_dir}/{low_res}px/**/*.pt", recursive=True
        )

        for low_path in low_res_files:
            high_path = low_path.replace(
                f"{os.sep}{low_res}px{os.sep}", f"{os.sep}{high_res}px{os.sep}"
            )
            if os.path.exists(high_path):
                self.pairs.append((low_path, high_path))

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _extract_latent(record: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Normalize stored latent payloads into tensors, preserving auxiliary metadata.

        Newer caches store dictionaries with a `latents` tensor along with mean/logvar,
        while older caches may persist the tensor directly. We gracefully handle both.
        """

        if isinstance(record, torch.Tensor):
            return record, {}

        if isinstance(record, dict):
            for key in ("latents", "tensor", "latent"):
                value = record.get(key)
                if isinstance(value, torch.Tensor):
                    meta = {k: v for k, v in record.items() if k != key}
                    return value, meta
            raise TypeError("Latent record dict does not contain a tensor payload.")

        if isinstance(record, (list, tuple)) and record:
            tensor_candidate = None
            meta: Dict[str, Any] = {}
            for item in record:
                if isinstance(item, torch.Tensor) and tensor_candidate is None:
                    tensor_candidate = item
                elif isinstance(item, dict):
                    meta.update(item)
            if tensor_candidate is not None:
                return tensor_candidate, meta

        raise TypeError(f"Unsupported latent record type: {type(record)}")

    def __getitem__(self, idx: int):
        low_path, high_path = self.pairs[idx]
        low_record = torch.load(low_path, map_location="cpu")
        high_record = torch.load(high_path, map_location="cpu")

        low_tensor, low_meta = self._extract_latent(low_record)
        high_tensor, high_meta = self._extract_latent(high_record)

        sample: Dict[str, Any] = {"low": low_tensor, "high": high_tensor}
        if low_meta:
            sample["low_meta"] = low_meta
        if high_meta:
            sample["high_meta"] = high_meta
        return sample
