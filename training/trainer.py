"""Simple trainer for latent upscaling using precomputed embeddings."""

from __future__ import annotations

import logging
import math
import random
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainingConfig, _resolve_bool
from .dataset import UpscaleDataset
from .helpers import get_latent_upscaler
from .losses import LossManager
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
        _seed_everything(self.cfg.seed)

        if not self.cfg.embeddings.enabled:
            raise ValueError("Latent training requires precomputed embeddings to be enabled in the config.")

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg.optimiser.gradient_accumulation_steps
        )
        self.logger = _create_logger(self.__class__.__name__)
        self.logger.disabled = not self.accelerator.is_local_main_process
        self.device = self.accelerator.device

        vae_names = list(self.cfg.embeddings.vae_names)
        cache_dirs = list(self.cfg.embeddings.vae_cache_dirs)
        if not cache_dirs:
            raise RuntimeError("No embedding caches configured for training.")
        if len(cache_dirs) != len(vae_names):
            raise RuntimeError(
                f"Mismatch between configured VAE names ({len(vae_names)}) and cache directories ({len(cache_dirs)})."
            )

        self.vae_names = vae_names
        self.vae_metric_names = [name.replace(" ", "_").replace("/", "_").replace("\\", "_") for name in self.vae_names]
        self.datasets: List[UpscaleDataset] = []
        self.dataloaders: List[DataLoader] = []
        dataset_sizes: List[int] = []

        low_res = self.cfg.dataset.model_resolution or self.cfg.dataset.high_resolution
        high_res = self.cfg.dataset.high_resolution

        for name, cache_dir in zip(self.vae_names, cache_dirs):
            dataset = UpscaleDataset(
                cache_dir=str(cache_dir),
                low_res=low_res,
                high_res=high_res,
            )
            if len(dataset) == 0:
                raise RuntimeError(f"No latent pairs were found in cache '{cache_dir}' for VAE '{name}'.")
            if len(dataset) < self.cfg.optimiser.batch_size:
                raise RuntimeError(
                    f"VAE '{name}' provides only {len(dataset)} latent pairs but batch_size is {self.cfg.optimiser.batch_size}."
                )

            dataloader = DataLoader(
                dataset,
                batch_size=self.cfg.optimiser.batch_size,
                shuffle=True,
                num_workers=self.cfg.dataset.num_workers,
                drop_last=True,
            )

            self.datasets.append(dataset)
            self.dataloaders.append(dataloader)
            dataset_sizes.append(len(dataset))

        world_size = max(1, self.accelerator.num_processes)
        per_dataset_steps: List[int] = []
        for name, loader in zip(self.vae_names, self.dataloaders):
            total_batches = len(loader)
            per_process_batches = total_batches // world_size
            if per_process_batches <= 0:
                raise RuntimeError(
                    f"Dataset for VAE '{name}' does not provide enough batches ({total_batches}) for {world_size} processes."
                )
            per_dataset_steps.append(per_process_batches)

        self.steps_per_epoch = min(per_dataset_steps)
        if self.steps_per_epoch <= 0:
            raise RuntimeError("No training batches available – check the embedding caches, batch size, and world size.")
        self.total_batches_per_epoch = self.steps_per_epoch * len(self.dataloaders)
        total_dataset_size = sum(dataset_sizes)

        example = self.datasets[0][0]
        low_latents = example["low"].unsqueeze(0) if example["low"].dim() == 3 else example["low"]
        high_latents = example["high"].unsqueeze(0) if example["high"].dim() == 3 else example["high"]
        if low_latents.dim() != 4 or high_latents.dim() != 4:
            raise ValueError("Expected latent tensors to be 4D (B, C, H, W) or 3D (C, H, W).")

        channels = low_latents.shape[1]
        if high_latents.shape[-1] % low_latents.shape[-1] != 0:
            raise ValueError("High-resolution latents must be an integer multiple of the low-resolution size.")
        if high_latents.shape[-2] % low_latents.shape[-2] != 0:
            raise ValueError("High-resolution latents must be an integer multiple of the low-resolution height.")
        scale_w = high_latents.shape[-1] // low_latents.shape[-1]
        scale_h = high_latents.shape[-2] // low_latents.shape[-2]
        if scale_w != scale_h:
            raise ValueError("High-resolution latents must use the same integer scale factor for height and width.")
        inferred_scale = scale_h
        if inferred_scale != 2:
            raise ValueError(f"Latent upscalers in helpers.py currently support only 2x upscaling (found {inferred_scale}×).")

        for idx, dataset in enumerate(self.datasets[1:], start=1):
            other = dataset[0]
            other_low = other["low"].unsqueeze(0) if other["low"].dim() == 3 else other["low"]
            other_high = other["high"].unsqueeze(0) if other["high"].dim() == 3 else other["high"]
            if other_low.shape[1:] != low_latents.shape[1:] or other_high.shape[1:] != high_latents.shape[1:]:
                raise ValueError(
                    f"Dataset for VAE '{self.vae_names[idx]}' has latent shape mismatch compared to the first dataset."
                )

        upscaler_cfg = self.cfg.latent_upscaler
        upscaler_name = getattr(upscaler_cfg, "model_name", getattr(upscaler_cfg, "model", None)) or "swin"
        upscaler_kwargs: Dict[str, Any] = {}
        if upscaler_name.startswith("swin"):
            window = getattr(upscaler_cfg, "window", None)
            if window is None:
                window = getattr(upscaler_cfg, "patch_size", None)
            if window:
                upscaler_kwargs["window"] = int(window)
            depth = getattr(upscaler_cfg, "depth", None)
            if depth:
                upscaler_kwargs["depth"] = int(depth)
            heads = getattr(upscaler_cfg, "heads", None)
            if heads:
                upscaler_kwargs["heads"] = int(heads)
            mlp_ratio = getattr(upscaler_cfg, "mlp_ratio", None)
            if mlp_ratio:
                upscaler_kwargs["mlp_ratio"] = float(mlp_ratio)
            extra_opts = getattr(upscaler_cfg, "extra", {}) or {}
            embed_dim = extra_opts.get("embed_dim")
            if embed_dim is not None:
                upscaler_kwargs["embed_dim"] = int(embed_dim)
            stages = extra_opts.get("stages")
            if stages is not None:
                upscaler_kwargs["stages"] = int(stages)
            depths = extra_opts.get("depths")
            if isinstance(depths, str):
                tokens = [token.strip() for token in depths.split(",") if token.strip()]
                try:
                    depths = [int(token) for token in tokens]
                except ValueError:
                    depths = None
            if depths is not None:
                upscaler_kwargs["depths"] = depths
            heads_list = extra_opts.get("num_heads") or extra_opts.get("num_heads_list")
            if isinstance(heads_list, str):
                tokens = [token.strip() for token in heads_list.split(",") if token.strip()]
                try:
                    heads_list = [int(token) for token in tokens]
                except ValueError:
                    heads_list = None
            if heads_list is not None:
                upscaler_kwargs["num_heads"] = heads_list
            for key in ("drop_rate", "attn_drop_rate", "drop_path_rate", "img_range"):
                value = extra_opts.get(key)
                if value is not None:
                    upscaler_kwargs[key] = float(value)
            for key in ("img_size", "scale"):
                value = extra_opts.get(key)
                if value is not None:
                    upscaler_kwargs[key] = int(value)
            for key in ("ape", "patch_norm", "use_checkpoint"):
                value = extra_opts.get(key)
                if value is not None:
                    upscaler_kwargs[key] = _resolve_bool(value, default=False)
            for key in ("resi_connection", "upsampler"):
                value = extra_opts.get(key)
                if value is not None:
                    upscaler_kwargs[key] = str(value)
            if upscaler_name == "swin_liif":
                liif_hidden = getattr(upscaler_cfg, "liif_hidden", None)
                if liif_hidden:
                    upscaler_kwargs["liif_hidden"] = int(liif_hidden)
        elif upscaler_name.startswith("naf"):
            blocks = getattr(upscaler_cfg, "blocks", None) or getattr(upscaler_cfg, "nerf_blocks", None)
            if blocks:
                upscaler_kwargs["blocks"] = int(blocks)
            groups = getattr(upscaler_cfg, "groups", None)
            if groups:
                upscaler_kwargs["groups"] = int(groups)

        self.model = get_latent_upscaler(
            model_name=upscaler_name,
            channels=channels,
            **upscaler_kwargs,
        ).to(self.device)
        self.upscaler_name = upscaler_name
        self.param_dtype = next(self.model.parameters()).dtype
        if self.cfg.losses.components:
            self.criterion = LossManager(self.cfg.losses.components)
            loss_names = ", ".join(comp.get("name", comp.get("type", "unknown")) for comp in self.cfg.losses.components)
            self.logger.info("Configured custom losses: %s", loss_names)
        else:
            self.criterion = nn.MSELoss()
        self.grad_accum_steps = max(1, self.cfg.optimiser.gradient_accumulation_steps)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler(self.grad_accum_steps)

        prepare_targets: List[Any] = [self.model, self.optimizer, self.criterion, *self.dataloaders]
        scheduler_present = self.scheduler is not None
        if scheduler_present:
            prepare_targets.append(self.scheduler)

        prepared = self.accelerator.prepare(*prepare_targets)
        self.model = prepared[0]
        self.optimizer = prepared[1]
        self.criterion = prepared[2]
        loaders_start = 3
        loaders_end = loaders_start + len(self.dataloaders)
        self.dataloaders = list(prepared[loaders_start:loaders_end])
        if scheduler_present:
            self.scheduler = prepared[-1]

        self.optimizer.zero_grad(set_to_none=True)

        self.global_step = 0  # counts optimiser steps
        parameter_source = self.accelerator.unwrap_model(self.model)
        total_params = sum(param.numel() for param in parameter_source.parameters())
        trainable_params = sum(param.numel() for param in parameter_source.parameters() if param.requires_grad)

        self.logger.info(
            "Initialised trainer | device=%s | datasets=%d | total_pairs=%d | batch_size=%d | epochs=%d",
            self.device,
            len(self.datasets),
            total_dataset_size,
            self.cfg.optimiser.batch_size,
            self.cfg.optimiser.num_epochs,
        )
        self.logger.info(
            "Model parameters | total=%d | trainable=%d | dtype=%s | upscaler=%s",
            total_params,
            trainable_params,
            self.param_dtype,
            self.upscaler_name,
        )
        self.logger.info(
            "Latent shapes | channels=%d | low_resolution=%d | high_resolution=%d | scale=%d",
            channels,
            low_latents.shape[-1],
            high_latents.shape[-1],
            inferred_scale,
        )
        for name, cache_dir, size, loader in zip(self.vae_names, cache_dirs, dataset_sizes, self.dataloaders):
            self.logger.info(
                "Dataset[%s] | cache=%s | pairs=%d | batches_per_epoch=%d",
                name,
                cache_dir,
                size,
                len(loader),
            )

        self.wandb_logger = WandbLogger(
            project=self.cfg.paths.project,
            run_name=self.cfg.logging.wandb_run_name,
            enabled=self.cfg.logging.use_wandb and self.accelerator.is_main_process,
            logger=self.logger,
        )
        metadata = {
            "metadata/total_parameters": total_params,
            "metadata/trainable_parameters": trainable_params,
            "metadata/device": str(self.device),
            "metadata/datasets": len(self.datasets),
            "metadata/dataset_size_total": total_dataset_size,
            "metadata/batch_size": self.cfg.optimiser.batch_size,
            "metadata/num_epochs": self.cfg.optimiser.num_epochs,
            "metadata/optimizer": self.cfg.optimiser.optimizer_type,
            "metadata/scheduler": self.cfg.optimiser.scheduler or "none",
            "metadata/clip_grad_norm": self.cfg.optimiser.clip_grad_norm,
            "metadata/latent_channels": channels,
            "metadata/latent_scale": inferred_scale,
            "metadata/vae_names": ", ".join(self.vae_names),
        }
        for name, size, loader in zip(self.vae_names, dataset_sizes, self.dataloaders):
            metadata[f"metadata/dataset_size/{name}"] = size
            metadata[f"metadata/batches_per_epoch/{name}"] = len(loader)

        self.sample_logger: Optional[SampleLogger] = None
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if self.wandb_logger.start(config=_flatten_config(self.cfg), model=unwrapped_model, metadata=metadata):
            try:
                dataset_map = {name: dataset for name, dataset in zip(self.vae_names, self.datasets)}
                primary_dataset = self.datasets[0] if self.datasets else None
                self.sample_logger = SampleLogger(
                    self.cfg,
                    dataset=primary_dataset,
                    datasets=dataset_map,
                    wandb_logger=self.wandb_logger,
                )
                self.sample_logger.maybe_log_samples(model=unwrapped_model, step=0, device=self.device)
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

    def _build_scheduler(self, accum_steps: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        scheduler_type = (self.cfg.optimiser.scheduler or "").lower()
        if scheduler_type == "cosine":
            batches_per_epoch = getattr(self, "total_batches_per_epoch", 0)
            if batches_per_epoch <= 0:
                return None
            effective_batches = math.ceil(batches_per_epoch / max(1, accum_steps))
            total_steps = self.cfg.optimiser.num_epochs * effective_batches
            if total_steps <= 0:
                return None
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.cfg.optimiser.min_learning_rate,
            )
        return None

    def step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        low = batch["low"].to(self.device, dtype=self.param_dtype)
        high = batch["high"].to(self.device, dtype=self.param_dtype)

        upscaled = self.model(low)
        loss = self.criterion(upscaled, high)
        self.accelerator.backward(loss)

        gathered = self.accelerator.gather(loss.detach())
        loss_value = float(gathered.mean().item())

        metrics: Dict[str, float] = {"loss": loss_value}
        criterion_module = self.accelerator.unwrap_model(self.criterion)
        raw_metrics = getattr(criterion_module, "raw_metrics", None)
        if isinstance(raw_metrics, dict) and raw_metrics:
            for name, tensor in raw_metrics.items():
                metric = tensor.detach()
                if metric.dim() > 0:
                    metric = metric.mean()
                metric = metric.to(device=loss.device, dtype=loss.dtype)
                gathered_metric = self.accelerator.gather(metric)
                metrics[name] = float(gathered_metric.mean().item())

        return loss_value, metrics

    def _optimizer_step(self, clip_norm: float) -> None:
        if clip_norm > 0:
            self.accelerator.clip_grad_norm_(self.model.parameters(), clip_norm)
        step_fn = getattr(self.accelerator, "optimizer_step", None)
        if callable(step_fn):
            step_fn(self.optimizer)
        else:
            compat_step = getattr(self.accelerator, "step", None)
            if callable(compat_step):
                compat_step(self.optimizer)
            else:
                self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1

    def _log_step_metrics(self, metrics: Dict[str, float]) -> None:
        if not self.wandb_logger.is_active or not self.accelerator.is_main_process:
            return
        self.wandb_logger.log(metrics, step=self.global_step)
        if self.sample_logger is not None:
            try:
                model = self.accelerator.unwrap_model(self.model)
                self.sample_logger.maybe_log_samples(model=model, step=self.global_step, device=self.device)
            except Exception as sample_exc:
                self.logger.warning("Sample logging failed at step %d: %s", self.global_step, sample_exc)
                self.sample_logger = None

    def train(self) -> None:
        num_epochs = self.cfg.optimiser.num_epochs
        clip_norm = self.cfg.optimiser.clip_grad_norm

        final_epoch_loss: Optional[float] = None
        best_epoch_loss: float = float("inf")
        accum_total_loss = 0.0
        accum_loss_by_vae = [0.0 for _ in self.dataloaders]
        accum_counts_by_vae = [0 for _ in self.dataloaders]
        accum_counter = 0
        component_accum: Dict[str, float] = {}
        component_counts: Dict[str, int] = {}
        num_vaes = len(self.dataloaders)

        def flush_accumulated() -> None:
            nonlocal accum_total_loss, accum_loss_by_vae, accum_counts_by_vae, accum_counter, component_accum, component_counts
            if accum_counter == 0 or not self.accelerator.is_main_process:
                accum_total_loss = 0.0
                accum_loss_by_vae = [0.0 for _ in range(num_vaes)]
                accum_counts_by_vae = [0 for _ in range(num_vaes)]
                accum_counter = 0
                component_accum = {}
                component_counts = {}
                return

            effective_counter = accum_counter
            metrics: Dict[str, float] = {
                "train/loss": accum_total_loss / max(1, effective_counter),
            }
            for idx, count in enumerate(accum_counts_by_vae):
                if count > 0:
                    metrics[f"train/loss/{self.vae_metric_names[idx]}"] = accum_loss_by_vae[idx] / count

            for name, total in component_accum.items():
                count = component_counts.get(name, 0)
                if count > 0:
                    metrics[f"train/loss_components/{name}"] = total / count

            self._log_step_metrics(metrics)

            accum_total_loss = 0.0
            accum_loss_by_vae = [0.0 for _ in range(num_vaes)]
            accum_counts_by_vae = [0 for _ in range(num_vaes)]
            accum_counter = 0
            component_accum = {}
            component_counts = {}

        for epoch in tqdm(
            range(1, num_epochs + 1),
            desc="Training epochs",
            unit="epoch",
            disable=not self.accelerator.is_local_main_process,
        ):
            epoch_loss_totals = [0.0 for _ in self.dataloaders]

            for loader in self.dataloaders:
                sampler = getattr(loader, "sampler", None)
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)

            loader_iters = [iter(loader) for loader in self.dataloaders]

            for _ in tqdm(
                range(self.steps_per_epoch),
                desc=f"Epoch {epoch} steps",
                unit="step",
                disable=not self.accelerator.is_local_main_process,
            ):
                for vae_index, loader_iter in enumerate(loader_iters):
                    batch = next(loader_iter)
                    with self.accelerator.accumulate(self.model):
                        loss_value, step_metrics = self.step(batch)
                        epoch_loss_totals[vae_index] += loss_value
                        accum_total_loss += loss_value
                        accum_loss_by_vae[vae_index] += loss_value
                        accum_counts_by_vae[vae_index] += 1
                        accum_counter += 1

                        for name, value in step_metrics.items():
                            if name == "loss":
                                continue
                            component_accum[name] = component_accum.get(name, 0.0) + value
                            component_counts[name] = component_counts.get(name, 0) + 1

                        if self.accelerator.sync_gradients:
                            self._optimizer_step(clip_norm)
                            flush_accumulated()

            avg_losses = [total / max(1, self.steps_per_epoch) for total in epoch_loss_totals]
            combined_avg = sum(avg_losses) / len(avg_losses)
            final_epoch_loss = combined_avg
            if combined_avg < best_epoch_loss:
                best_epoch_loss = combined_avg

            per_vae_summary = " | ".join(f"{name}=%.6f" % avg for name, avg in zip(self.vae_names, avg_losses))
            self.logger.info("Epoch %d/%d | loss=%.6f | %s", epoch, num_epochs, combined_avg, per_vae_summary)

            if self.wandb_logger.is_active:
                metrics = {"train/epoch_loss": combined_avg}
                for suffix, avg in zip(self.vae_metric_names, avg_losses):
                    metrics[f"train/epoch_loss/{suffix}"] = avg
                self.wandb_logger.log(metrics, step=self.global_step)

        if self.wandb_logger.is_active:
            summary: Dict[str, Any] = {"best_epoch_loss": best_epoch_loss}
            if final_epoch_loss is not None:
                summary["final_loss"] = final_epoch_loss
            self.wandb_logger.log_summary(summary)
            self.wandb_logger.finish()
        self.accelerator.wait_for_everyone()


def run(config: TrainingConfig | Dict[str, Any]) -> None:
    if not isinstance(config, TrainingConfig):
        config = TrainingConfig.from_dict(config)
    trainer = VAETrainer(config)
    trainer.train()
