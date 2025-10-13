
from __future__ import annotations
import time
import pandas as pd
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import pyiqa

# ResizeRight imports
from utils.resize_utils import pil_resize_right
from utils import interp_methods

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ResizeRightTransform:
    """Custom transform using ResizeRight for high-quality resizing."""
    def __init__(self, size: int):
        self.size = size
    
    def __call__(self, img: Image.Image) -> Image.Image:
        return pil_resize_right(img, (self.size, self.size), 
                               interp_method=interp_methods.cubic, 
                               antialiasing=True)


def build_transform(resize_shorter_side: int) -> transforms.Compose:
    ops = []
    if resize_shorter_side and resize_shorter_side > 0:
        ops.append(ResizeRightTransform(resize_shorter_side))
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


class ImageDataset:
    def __init__(self, img_paths: Iterable[Path]) -> None:
        self.image_records = img_paths
        self.transform = build_transform(128)
    
    def __len__(self) -> int:
        return len(self.image_records)
    
    def __getitem__(self, idx: int) -> Path:
        img = Image.open(self.image_records[idx])
        img = img.convert("RGB")
        img = self.transform(img)
        return img

       
def main() -> None:
    df = pd.read_csv("unpacked_original_ds/full_dataset/clear_images_with_embeddings.csv")
    device = torch.device("cuda:0")
    print(f"Using device: {device}")
    metrics = {
        # "maniqa": pyiqa.create_metric("maniqa", device=device),
        "clip_iqa": pyiqa.create_metric("clipiqa+_vitL14_512", device=device),
        "hyperiqa": pyiqa.create_metric("hyperiqa", device=device),
    }
    
    for metric in metrics:
        metrics[metric].eval()
        metrics[metric].to(device)
        
    dataset = ImageDataset(df["path"].tolist())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=20)
    metric_results = {name: [] for name in metrics.keys()}
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch_tensor = batch.to(device)
            for metric_name, metric in metrics.items():
                batch_scores = metric(batch_tensor).detach().cpu().flatten().tolist()
                metric_results[metric_name].extend(batch_scores)

    for metric_name, metric_scores in metric_results.items():
        df[metric_name] = metric_scores
        
    df.to_csv("clear_images_with_maniqa_scores.csv", index=False)
        


if __name__ == "__main__":
    main()
