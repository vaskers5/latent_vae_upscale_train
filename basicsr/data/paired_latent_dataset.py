import torch
from torch.utils import data as data
from os import path as osp

from basicsr.data.data_util import paired_paths_from_folder
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PairedLatentDataset(data.Dataset):
    """Paired latent dataset for latent space super resolution.

    Read LQ (Low Quality, e.g. low resolution latent) and GT (Ground Truth, high resolution latent) pairs.
    This dataset is specifically designed for training models that work in latent space,
    such as upscaling latent representations from VAE encoders.

    The dataset expects:
    - LQ latents: shape [C, H, W] where typically C=16, H=W=16 for 128px images
    - GT latents: shape [C, H*scale, W*scale] where scale is the upscaling factor

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt latents.
        dataroot_lq (str): Data root path for lq latents.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        scale (int): Scale factor for upscaling (e.g., 2 for 16x16 -> 32x32).
        phase (str): 'train' or 'val'.
        mean (list): Mean values for normalization. Default: None.
        std (list): Std values for normalization. Default: None.
    """

    def __init__(self, opt):
        super(PairedLatentDataset, self).__init__()
        self.opt = opt

        self.gt_folder, self.lq_folder = opt["dataroot_gt"], opt["dataroot_lq"]
        if "filename_tmpl" in opt:
            self.filename_tmpl = opt["filename_tmpl"]
        else:
            self.filename_tmpl = "{}"

        self.scale = opt["scale"]
        self.phase = opt["phase"]

        # Normalization parameters
        self.mean = opt.get("mean", None)
        self.std = opt.get("std", None)

        # Generate paired paths
        self.paths = paired_paths_from_folder(
            [self.lq_folder, self.gt_folder], ["lq", "gt"], self.filename_tmpl
        )

    def __getitem__(self, index):
        # Load LQ and GT latent tensors
        lq_path = self.paths[index]["lq_path"]
        gt_path = self.paths[index]["gt_path"]

        # Load latent tensors from .pt files
        lq_data = torch.load(lq_path, map_location="cpu")
        gt_data = torch.load(gt_path, map_location="cpu")

        # Extract latents from the data dictionary
        if isinstance(lq_data, dict) and "latents" in lq_data:
            img_lq = lq_data["latents"]
        else:
            img_lq = lq_data

        if isinstance(gt_data, dict) and "latents" in gt_data:
            img_gt = gt_data["latents"]
        else:
            img_gt = gt_data

        # Ensure tensors are float32
        img_lq = img_lq.float()
        img_gt = img_gt.float()

        # For training, we can apply some basic augmentations
        if self.phase == "train":
            # Random horizontal flip (applied to both LQ and GT)
            if torch.rand(1) < 0.5:
                img_lq = torch.flip(img_lq, dims=[2])  # flip width dimension
                img_gt = torch.flip(img_gt, dims=[2])

            # Random vertical flip (applied to both LQ and GT)
            if torch.rand(1) < 0.5:
                img_lq = torch.flip(img_lq, dims=[1])  # flip height dimension
                img_gt = torch.flip(img_gt, dims=[1])

        # Normalize if specified
        if self.mean is not None and self.std is not None:
            # Apply normalization channel-wise
            for c in range(img_lq.shape[0]):
                img_lq[c] = (img_lq[c] - self.mean[c]) / self.std[c]
            for c in range(img_gt.shape[0]):
                img_gt[c] = (img_gt[c] - self.mean[c]) / self.std[c]

        return {"lq": img_lq, "gt": img_gt, "lq_path": lq_path, "gt_path": gt_path}

    def __len__(self):
        return len(self.paths)
