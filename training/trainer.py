"""Simple trainer for latent upscaling using precomputed embeddings."""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainingConfig
from .dataset import UpscaleDataset
from .helpers import PixNerfUpscaler
from .sample_logging import SampleLogger
from .wandb_logger import WandbLogger

__all__ = ["VAETrainer", "run"]


def _create_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(logging.INFO)
    return logger


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
        # _seed_everything(self.cfg.seed)

        if not self.cfg.embeddings.enabled:
            raise ValueError("Latent training requires precomputed embeddings to be enabled in the config.")

        self.logger = _create_logger(self.__class__.__name__)
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
        if inferred_scale != 2:
            raise ValueError(f"PixNerfUpscaler supports 2x upscaling, but dataset scale factor is {inferred_scale}.")

        self.model = PixNerfUpscaler(
            channels=channels,
            patch_size=self.cfg.latent_upscaler.patch_size,
            hidden_dim_multiplier=self.cfg.latent_upscaler.hidden_dim_multiplier,
            nerf_blocks=self.cfg.latent_upscaler.nerf_blocks,
        ).to(self.device)
        self.param_dtype = next(self.model.parameters()).dtype

        self.criterion = nn.MSELoss()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.global_step = 0
        total_params = sum(param.numel() for param in self.model.parameters())
        trainable_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        self.logger.info(
            "Initialised trainer | device=%s | dataset_size=%d | batch_size=%d | epochs=%d",
            self.device,
            len(self.dataset),
            self.cfg.optimiser.batch_size,
            self.cfg.optimiser.num_epochs,
        )
        self.logger.info(
            "Model parameters | total=%d | trainable=%d | dtype=%s | patch_size=%d",
            total_params,
            trainable_params,
            self.param_dtype,
            self.cfg.latent_upscaler.patch_size,
        )
        self.logger.info(
            "Latent shapes | channels=%d | low_resolution=%d | high_resolution=%d | scale=%d",
            channels,
            low_latents.shape[-1],
            high_latents.shape[-1],
            inferred_scale,
        )

        self.wandb_logger = WandbLogger(
            project=self.cfg.paths.project,
            run_name=self.cfg.logging.wandb_run_name,
            enabled=self.cfg.logging.use_wandb,
            logger=self.logger,
        )
        metadata = {
            "metadata/total_parameters": total_params,
            "metadata/trainable_parameters": trainable_params,
            "metadata/device": str(self.device),
            "metadata/dataset_size": len(self.dataset),
            "metadata/batch_size": self.cfg.optimiser.batch_size,
            "metadata/num_epochs": self.cfg.optimiser.num_epochs,
            "metadata/optimizer": self.cfg.optimiser.optimizer_type,
            "metadata/scheduler": self.cfg.optimiser.scheduler or "none",
            "metadata/clip_grad_norm": self.cfg.optimiser.clip_grad_norm,
            "metadata/latent_channels": channels,
            "metadata/latent_scale": inferred_scale,
        }
        self.sample_logger: Optional[SampleLogger] = None
        if self.wandb_logger.start(config=_flatten_config(self.cfg), model=self.model, metadata=metadata):
            try:
                self.sample_logger = SampleLogger(self.cfg, dataset=self.dataset, wandb_logger=self.wandb_logger)
                self.sample_logger.maybe_log_samples(model=self.model, step=0, device=self.device)
                self.logger.info(
                    "Sample logger initialised | interval=%d | samples=%d",
                    self.sample_logger.sample_interval,
                    self.sample_logger.sample_count,
                )
            except Exception as exc:
                self.logger.warning("Failed to initialise sample logger: %s", exc)
                self.sample_logger = None

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

    def _current_lr(self) -> float:
        if not self.optimizer.param_groups:
            return 0.0
        return float(self.optimizer.param_groups[0].get("lr", 0.0))

    def _compute_grad_norm(self) -> float:
        total = 0.0
        for param in self.model.parameters():
            if param.grad is None:
                continue
            norm = param.grad.detach().data.norm(2)
            if torch.isnan(norm):
                return float("nan")
            total += float(norm.item()) ** 2
        return float(total**0.5)

    def train(self) -> None:
        num_epochs = self.cfg.optimiser.num_epochs
        clip_norm = self.cfg.optimiser.clip_grad_norm
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        num_batches = len(self.dataloader)
        final_epoch_loss: Optional[float] = None
        final_epoch_mae: Optional[float] = None
        best_epoch_loss: float = float("inf")
        best_epoch_index: Optional[int] = None
        for epoch in tqdm(range(1, num_epochs + 1), desc="Training epochs", unit="epoch"):
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            epoch_mae = 0.0
            grad_norm_accumulator = 0.0
            grad_norm_max = 0.0
            batch_time_accumulator = 0.0
            data_time_accumulator = 0.0
            max_loss = 0.0
            samples_seen = 0
            data_timer = time.perf_counter()

            for batch_idx, batch in enumerate(
                tqdm(self.dataloader, desc=f"Epoch {epoch} batches", unit="batch"),
                start=1,
            ):
                data_time = time.perf_counter() - data_timer
                low = batch["low"].to(self.device, dtype=self.param_dtype)
                high = batch["high"].to(self.device, dtype=self.param_dtype)
                samples_seen += int(low.shape[0])

                step_timer = time.perf_counter()
                self.optimizer.zero_grad(set_to_none=True)
                upscaled = self.model(low)
                loss = self.criterion(upscaled, high)
                loss.backward()

                grad_norm: Optional[float] = None
                if clip_norm > 0:
                    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
                    if self.cfg.logging.log_grad_norm:
                        grad_norm = float(grad_norm_tensor.detach().item())
                elif self.cfg.logging.log_grad_norm:
                    grad_norm = self._compute_grad_norm()

                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                step_time = time.perf_counter() - step_timer
                batch_time_accumulator += step_time
                data_time_accumulator += data_time
                loss_value = loss.detach().item()
                epoch_loss += loss_value
                max_loss = max(max_loss, loss_value)
                with torch.no_grad():
                    diff = upscaled - high
                    mae_value = diff.abs().mean().item()
                    epoch_mae += mae_value

                if grad_norm is not None and not math.isnan(grad_norm):
                    grad_norm_accumulator += grad_norm
                    grad_norm_max = max(grad_norm_max, grad_norm)

                self.global_step += 1
                current_lr = self._current_lr()
                data_timer = time.perf_counter()

                if self.wandb_logger.is_active:
                    metrics: Dict[str, Any] = {
                        "train/mse_loss": loss_value,
                        "train/lr": current_lr,
                        "train/epoch": epoch,
                        "train/batch_index": batch_idx,
                    }
                    if step_time > 0:
                        metrics["train/samples_per_sec_batch"] = low.shape[0] / step_time
                    if grad_norm is not None:
                        metrics["train/grad_norm"] = grad_norm
                    self.wandb_logger.log(metrics, step=self.global_step)
                    if self.sample_logger is not None:
                        try:
                            self.sample_logger.maybe_log_samples(
                                model=self.model, step=self.global_step, device=self.device
                            )
                        except Exception as sample_exc:
                            self.logger.warning("Sample logging failed at step %d: %s", self.global_step, sample_exc)
                            self.sample_logger = None

            epoch_time = time.perf_counter() - epoch_start
            avg_loss = epoch_loss / max(1, num_batches)
            final_epoch_loss = avg_loss
            avg_mae = epoch_mae / max(1, num_batches)
            final_epoch_mae = avg_mae
            avg_batch_time = batch_time_accumulator / max(1, num_batches)
            avg_data_time = data_time_accumulator / max(1, num_batches)
            throughput = samples_seen / epoch_time if epoch_time > 0 else 0.0
            avg_grad_norm = None
            if self.cfg.logging.log_grad_norm and num_batches > 0:
                avg_grad_norm = grad_norm_accumulator / max(1, num_batches)
            if avg_loss < best_epoch_loss:
                best_epoch_loss = avg_loss
                best_epoch_index = epoch

            peak_memory_mb: Optional[float] = None
            if self.device.type == "cuda":
                peak_memory = torch.cuda.max_memory_allocated(self.device)
                peak_memory_mb = peak_memory / (1024**2)
                torch.cuda.reset_peak_memory_stats(self.device)

            summary_parts = [
                f"Epoch {epoch}/{num_epochs}",
                f"loss={avg_loss:.6f}",
                f"max_loss={max_loss:.6f}",
                f"lr={self._current_lr():.6e}",
                f"epoch_time={epoch_time:.2f}s",
                f"batch_time={avg_batch_time:.3f}s",
                f"data_time={avg_data_time:.3f}s",
                f"samples_per_sec={throughput:.1f}",
            ]
            if final_epoch_mae is not None:
                summary_parts.append(f"latent_mae={avg_mae:.6f}")
            if avg_grad_norm is not None:
                summary_parts.append(f"avg_grad={avg_grad_norm:.4f}")
                summary_parts.append(f"max_grad={grad_norm_max:.4f}")
            if peak_memory_mb is not None:
                summary_parts.append(f"gpu_mem={peak_memory_mb:.1f}MB")
            self.logger.info(" | ".join(summary_parts))

            if self.wandb_logger.is_active:
                epoch_metrics: Dict[str, Any] = {
                    "train/epoch_loss": avg_loss,
                    "train/epoch_mae_latent": avg_mae,
                    "train/epoch_max_loss": max_loss,
                    "train/epoch_time": epoch_time,
                    "train/avg_batch_time": avg_batch_time,
                    "train/avg_data_time": avg_data_time,
                    "train/samples_per_sec": throughput,
                    "train/epoch_index": epoch,
                    "train/samples_seen_epoch": samples_seen,
                    "train/learning_rate_epoch_end": current_lr,
                }
                if avg_grad_norm is not None:
                    epoch_metrics["train/avg_grad_norm"] = avg_grad_norm
                    epoch_metrics["train/max_grad_norm"] = grad_norm_max
                if peak_memory_mb is not None:
                    epoch_metrics["train/peak_gpu_memory_mb"] = peak_memory_mb
                self.wandb_logger.log(epoch_metrics, step=self.global_step)

        if self.wandb_logger.is_active:
            summary: Dict[str, Any] = {
                "total_steps": self.global_step,
                "best_epoch_loss": best_epoch_loss,
            }
            if final_epoch_loss is not None:
                summary["final_loss"] = final_epoch_loss
            if final_epoch_mae is not None:
                summary["final_latent_mae"] = final_epoch_mae
            if best_epoch_index is not None:
                summary["best_epoch_index"] = best_epoch_index
            self.wandb_logger.log_summary(summary)
            self.wandb_logger.finish()


def run(config: TrainingConfig | Dict[str, Any]) -> None:
    if not isinstance(config, TrainingConfig):
        config = TrainingConfig.from_dict(config)
    trainer = VAETrainer(config)
    trainer.train()
