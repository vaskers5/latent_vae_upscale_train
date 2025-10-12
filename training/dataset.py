"""Dataset utilities used by the VAE trainer."""

from __future__ import annotations

import glob
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from .embeddings import EmbeddingCache, TransformParams
# ResizeRight imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.resize_utils import torch_resize_right, pil_resize_right
from utils import interp_methods

__all__ = ["ImageFolderDataset", "UpscaleDataset"]


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
        paths_csv: Optional[Path] = None,
    ) -> None:
        print(
            f"[Dataset] Initializing ImageFolderDataset with root={root}, high_resolution={high_resolution}, "
            f"resize_long_side={resize_long_side}, limit={limit}, paths_csv={paths_csv}"
        )
        self.root = Path(root).expanduser().resolve()
        self.high_resolution = high_resolution
        self.resize_long_side = resize_long_side
        self.paths: List[Path] = []
        self.embedding_cache = embedding_cache
        self.model_resolution = model_resolution
        self.paths_csv = Path(paths_csv).expanduser() if paths_csv else None
        self._needs_model_downsample = (
            self.model_resolution > 0 and self.model_resolution != self.high_resolution
        )

        self.paths = self._collect_valid_paths(limit=limit)
        print(f"[Dataset] Collected {len(self.paths)} valid paths")
        if not self.paths:
            raise RuntimeError(f"No valid images found under '{self.root}'")
        random.shuffle(self.paths)
        print(f"[Dataset] Paths shuffled; total dataset length: {len(self.paths)}")

    def _collect_valid_paths(self, limit: int = 0) -> List[Path]:
        if self.paths_csv:
            print(f"[Dataset] Loading image list from CSV manifest: {self.paths_csv}")
            return self._collect_from_csv(limit=limit)

        print(f"[Dataset] Collecting valid image paths from {self.root}...")
        return self._collect_from_disk(limit=limit)

    def _collect_from_disk(self, limit: int = 0) -> List[Path]:
        all_images: List[Path] = []
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        for root, _dirs, files in os.walk(self.root):
            for file in files:
                if os.path.splitext(file)[1].lower() in exts:
                    full_path = Path(os.path.abspath(os.path.join(root, file)))
                    all_images.append(full_path)
                    if limit > 0 and len(all_images) >= limit:
                        return all_images
        return all_images[:limit] if limit > 0 else all_images

    def _collect_from_csv(self, limit: int = 0) -> List[Path]:
        if not self.paths_csv:
            return []
        manifest_path = self.paths_csv
        if not manifest_path.exists():
            raise FileNotFoundError(f"CSV manifest not found: {manifest_path}")

        images: List[Path] = []
        df = pd.read_csv(manifest_path)
        if df.empty:
            raise RuntimeError(f"No valid image records found in CSV manifest '{manifest_path}'.")

        lowered = {col.lower(): col for col in df.columns}
        path_key = next((lowered[key] for key in ("path", "filepath", "image_path") if key in lowered), None)
        if path_key is None:
            raise KeyError(
                f"CSV manifest '{manifest_path}' must include a 'path' column (accepted aliases: filepath, image_path)."
            )

        path_series = df[path_key].dropna().astype(str).str.strip()
        if limit > 0:
            path_series = path_series.iloc[:limit]
        images = path_series.apply(Path).tolist()

        return images

    def set_embedding_cache(self, cache: Optional[EmbeddingCache]) -> None:
        self.embedding_cache = cache

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index % len(self.paths)]
        print(f"[Dataset] Loading sample {index} from path: {path}")
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
        # Use ResizeRight for high-quality downsampling
        resized = torch_resize_right(
            tensor,
            size=(self.model_resolution, self.model_resolution),
            interp_method=interp_methods.cubic,
            antialiasing=True,
        )
        return resized

    def _fit_image(self, img: Image.Image, target_size: int) -> Image.Image:
        """Fit image to target size with center crop and ResizeRight resize."""
        width, height = img.size
        
        # Calculate the scale to fit the image
        scale = max(target_size / width, target_size / height)
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        
        # Resize using ResizeRight
        if new_width != width or new_height != height:
            img = pil_resize_right(img, (new_width, new_height), 
                                 interp_method=interp_methods.lanczos3, 
                                 antialiasing=True)
        
        # Center crop
        left = (new_width - target_size) // 2
        top = (new_height - target_size) // 2
        right = left + target_size
        bottom = top + target_size
        
        return img.crop((left, top, right, bottom))
    
    def _resize_if_needed(self, img: Image.Image) -> Image.Image:
        if self.resize_long_side <= 0:
            return img
        width, height = img.size
        longest = max(width, height)
        if longest <= self.resize_long_side:
            return img
        scale = self.resize_long_side / float(longest)
        new_size = (int(round(width * scale)), int(round(height * scale)))
        # Use ResizeRight with Lanczos3 for high-quality resizing
        return pil_resize_right(img, new_size, interp_method=interp_methods.lanczos3, antialiasing=True)

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
        # Use our ResizeRight-based fit function instead of ImageOps.fit
        img = self._fit_image(img, self.high_resolution)
        return img, TransformParams(flip=False, crop_x=0, crop_y=0)


class UpscaleDataset(Dataset):
    """Dataset that loads precomputed low/high resolution tensor pairs."""

    def __init__(self, cache_dir: str, low_res: int, high_res: int, csv_path: Optional[str] = None) -> None:
        print(f"[Dataset] Initializing UpscaleDataset with cache_dir={cache_dir}, low_res={low_res}, high_res={high_res}, csv_path={csv_path}")
        self.pairs = []
        if csv_path:
            self._load_from_csv(csv_path, low_res, high_res)
        else:
            self._load_from_cache(cache_dir, low_res, high_res)
        print(f"[Dataset] Collected {len(self.pairs)} valid low/high pairs")

    def _load_from_cache(self, cache_dir: str, low_res: int, high_res: int) -> None:
        low_res_files = glob.glob(
            f"{cache_dir}/{low_res}px/**/*.pt", recursive=True
        )
        print(f"[Dataset] Found {len(low_res_files)} low-res files")

        for low_path in low_res_files:
            high_path = low_path.replace(
                f"{os.sep}{low_res}px{os.sep}", f"{os.sep}{high_res}px{os.sep}"
            )
            if os.path.exists(high_path):
                self.pairs.append((low_path, high_path))

    def _load_from_csv(self, csv_path: str, low_res: int, high_res: int) -> None:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            embeddings = eval(row['available_embeddings'])
            low_embs = [e for e in embeddings if f'/{low_res}px/' in e]
            high_embs = [e for e in embeddings if f'/{high_res}px/' in e]
            for low_emb in low_embs:
                high_emb = low_emb.replace(f'/{low_res}px/', f'/{high_res}px/')
                if high_emb in high_embs:
                    self.pairs.append((low_emb, high_emb))

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
