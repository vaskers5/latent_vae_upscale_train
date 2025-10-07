"""Simple trainer for latent upscaling using precomputed embeddings."""

from __future__ import annotations

import random
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .dataset import UpscaleDataset
from .helpers import LatentUpscaler

__all__ = ["VAETrainer", "run"]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _flatten_config(config: Any) -> Dict[str, Any]:
    if is_dataclass(config):
        raw = {}
        for key, value in asdict(config).items():
            raw[key] = _flatten_config(value)
        return raw
    if isinstance(config, dict):
        return {key: _flatten_config(value) for key, value in config.items()}
    if isinstance(config, (str, int, float, bool)):
        return config
    if hasattr(config, "as_posix"):
        return config.as_posix()
    return str(config)


class VAETrainer:
    """Minimal trainer that optimises the latent upscaler with MSE loss."""

    def __init__(self, config: TrainingConfig) -> None:
        self.cfg = config
        self.cfg.paths.ensure_directories()
        _seed_everything(self.cfg.seed)

        if not self.cfg.embeddings.enabled:
            raise ValueError("Latent training requires precomputed embeddings to be enabled in the config.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = UpscaleDataset(
            cache_dir=str(self.cfg.embeddings.cache_dir),
            low_res=self.cfg.dataset.model_resolution or self.cfg.dataset.high_resolution,
            high_res=self.cfg.dataset.high_resolution,
        )
        if len(self.dataset) == 0:
            raise RuntimeError("No latent pairs were found in the cache directory for the given resolutions.")
        if len(self.dataset) < self.cfg.optimiser.batch_size:
            raise RuntimeError(
                f"Found only {len(self.dataset)} latent pairs but batch_size is {self.cfg.optimiser.batch_size}."
            )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.cfg.optimiser.batch_size,
            shuffle=True,
            num_workers=self.cfg.dataset.num_workers,
            drop_last=True,
        )

        example = self.dataset[0]
        low_latents = example["low"].unsqueeze(0) if example["low"].dim() == 3 else example["low"]
        high_latents = example["high"].unsqueeze(0) if example["high"].dim() == 3 else example["high"]
        if low_latents.dim() != 4 or high_latents.dim() != 4:
            raise ValueError("Expected latent tensors to be 4D (B, C, H, W) or 3D (C, H, W).")

        channels = low_latents.shape[1]
        if high_latents.shape[-1] % low_latents.shape[-1] != 0:
            raise ValueError("High-resolution latents must be an integer multiple of the low-resolution size.")
        inferred_scale = high_latents.shape[-1] // low_latents.shape[-1]
        scale_factor = self.cfg.latent_upscaler.scale_factor or inferred_scale
        if scale_factor < 1:
            raise ValueError("Latent upscaler scale factor must be >= 1.")

        self.model = LatentUpscaler(
            channels=channels,
            scale_factor=scale_factor,
            hidden_multiplier=self.cfg.latent_upscaler.width_multiplier,
        ).to(self.device)
        self.param_dtype = next(self.model.parameters()).dtype

        self.criterion = nn.MSELoss()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.global_step = 0
        self.use_wandb = self.cfg.logging.use_wandb
        if self.use_wandb:
            wandb.init(
                project=self.cfg.paths.project,
                name=self.cfg.logging.wandb_run_name,
                config=_flatten_config(self.cfg),
            )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        lr = self.cfg.optimiser.base_learning_rate
        betas = (0.9, self.cfg.optimiser.beta2)
        eps = self.cfg.optimiser.eps
        weight_decay = self.cfg.optimiser.weight_decay if self.cfg.optimiser.use_decay else 0.0

        opt_type = self.cfg.optimiser.optimizer_type.lower()
        if "8bit" in opt_type:
            import bitsandbytes as bnb

            return bnb.optim.Adam8bit(self.model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        if opt_type == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        return torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        scheduler_type = (self.cfg.optimiser.scheduler or "").lower()
        if scheduler_type == "cosine":
            total_steps = self.cfg.optimiser.num_epochs * len(self.dataloader)
            if total_steps <= 0:
                return None
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.cfg.optimiser.min_learning_rate,
            )
        return None

    def train(self) -> None:
        num_epochs = self.cfg.optimiser.num_epochs
        clip_norm = self.cfg.optimiser.clip_grad_norm
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch in self.dataloader:
                low = batch["low"].to(self.device, dtype=self.param_dtype)
                high = batch["high"].to(self.device, dtype=self.param_dtype)

                self.optimizer.zero_grad(set_to_none=True)
                upscaled = self.model(low)
                loss = self.criterion(upscaled, high)
                loss.backward()
                if clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                loss_value = loss.detach().item()
                epoch_loss += loss_value
                self.global_step += 1
                if self.use_wandb:
                    wandb.log({"train/mse_loss": loss_value, "train/epoch": epoch}, step=self.global_step)

            epoch_loss /= max(1, len(self.dataloader))
            if self.use_wandb:
                wandb.log({"train/epoch_loss": epoch_loss, "train/epoch_index": epoch}, step=self.global_step)

        if self.use_wandb:
            wandb.finish()


def run(config: TrainingConfig) -> None:
    trainer = VAETrainer(config)
    trainer.train()
