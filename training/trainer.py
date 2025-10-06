"""Core training loop and orchestration for VAE fine-tuning."""

from __future__ import annotations

import gc
import math
import random
import shutil
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import bitsandbytes as bnb
import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from diffusers import (
    AsymmetricAutoencoderKL,
    AutoencoderKL,
    AutoencoderKLQwenImage,
    AutoencoderKLWan,
)
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainingConfig
from .dataset import ImageFolderDataset
from .embeddings import EmbeddingCache
from .helpers import FocalFrequencyLoss, LatentUpscaler, MedianLossNormalizer

__all__ = ["VAETrainer", "run"]


class VAETrainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.cfg = config
        self.cfg.paths.ensure_directories()

        self.accelerator = Accelerator(
            mixed_precision=config.model.mixed_precision,
            gradient_accumulation_steps=config.optimiser.gradient_accumulation_steps,
        )
        self.device = self.accelerator.device
        self._seed_everything(config.seed)

        self.embedding_cache: Optional[EmbeddingCache] = None
        if self.cfg.embeddings.enabled:
            if self.cfg.model.full_training and not self.cfg.model.train_decoder_only:
                raise ValueError(
                    "Embedding cache is only supported when the encoder is frozen (train_decoder_only or latent upscaler mode)."
                )
            self.embedding_cache = EmbeddingCache(self.cfg.embeddings, self.cfg.dataset, self.cfg.paths.dataset_root)

        self.dataset = ImageFolderDataset(
            root=self.cfg.paths.dataset_root,
            high_resolution=self.cfg.dataset.high_resolution,
            resize_long_side=self.cfg.dataset.resize_long_side,
            limit=self.cfg.dataset.limit,
            horizontal_flip_prob=self.cfg.dataset.horizontal_flip_prob,
            embedding_cache=self.embedding_cache,
            model_resolution=self.cfg.dataset.model_resolution,
        )
        if len(self.dataset) < self.cfg.optimiser.batch_size:
            raise RuntimeError(
                f"Found only {len(self.dataset)} images but batch_size is {self.cfg.optimiser.batch_size}."
            )

        self.vae = self._load_vae()
        self.latent_upscaler: Optional[LatentUpscaler] = self._build_latent_upscaler(self.vae)
        self._freeze_parameters()

        if self.embedding_cache is not None:
            self.embedding_cache.validate_dataset(self.dataset)
            self.accelerator.wait_for_everyone()

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.cfg.optimiser.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True,
        )

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        prepare_items: List[Any] = [self.dataloader, self.vae]
        if self.latent_upscaler is not None:
            prepare_items.append(self.latent_upscaler)
        prepare_items.extend([self.optimizer, self.scheduler])
        prepared = self.accelerator.prepare(*prepare_items)
        iterator = iter(prepared)
        self.dataloader = next(iterator)
        self.vae = next(iterator)
        if self.latent_upscaler is not None:
            self.latent_upscaler = next(iterator)
        self.optimizer = next(iterator)
        self.scheduler = next(iterator)

        self.compile_enabled = False
        if self.cfg.model.use_torch_compile and hasattr(torch, "compile"):
            compile_kwargs: Dict[str, Any] = {}
            if self.cfg.model.torch_compile_backend:
                compile_kwargs["backend"] = self.cfg.model.torch_compile_backend
            if self.cfg.model.torch_compile_mode:
                compile_kwargs["mode"] = self.cfg.model.torch_compile_mode
            if self.cfg.model.torch_compile_fullgraph is not None:
                compile_kwargs["fullgraph"] = self.cfg.model.torch_compile_fullgraph
            try:
                self.vae = torch.compile(self.vae, **compile_kwargs)
                self.compile_enabled = True
            except Exception as exc:  # pragma: no cover - diagnostic only
                self.accelerator.print(f"[WARN] torch.compile failed for VAE: {exc}")
            if self.latent_upscaler is not None:
                self.accelerator.print("[INFO] Skipping torch.compile for latent upscaler (eager mode forced)")

        self._offload_unused_encoder()

        self.trainable_params = [p for p in self.vae.parameters() if p.requires_grad]
        if self.latent_upscaler is not None:
            self.trainable_params.extend(p for p in self.latent_upscaler.parameters() if p.requires_grad)

        self.loss_normalizer = MedianLossNormalizer(
            self.cfg.losses.ratios,
            self.cfg.losses.median_coeff_steps,
            device=self.device,
        )
        self.ffl_module: Optional[FocalFrequencyLoss] = None
        if self.cfg.losses.ratios.get("ffl", 0.0) > 0:
            self.ffl_module = FocalFrequencyLoss(
                alpha=self.cfg.losses.focal_frequency_alpha,
                patch_factor=self.cfg.losses.focal_frequency_patch_factor,
                ave_spectrum=self.cfg.losses.focal_frequency_ave_spectrum,
                log_weight=self.cfg.losses.focal_frequency_log_weight,
                normalize=self.cfg.losses.focal_frequency_normalize,
                eps=self.cfg.losses.focal_frequency_eps,
            )

        self.lpips_module: Optional[lpips.LPIPS] = None
        if self.cfg.losses.ratios.get("lpips", 0.0) > 0:
            self.lpips_module = lpips.LPIPS(net=self.cfg.losses.lpips_backbone, verbose=False)
            self.lpips_module = self.lpips_module.to(self.device).eval()
            for p in self.lpips_module.parameters():
                p.requires_grad_(False)

        self.ema_vae: Optional[AveragedModel] = None
        self.ema_latent_upscaler: Optional[AveragedModel] = None
        if self.cfg.ema.enabled:
            if self.cfg.ema.device:
                try:
                    ema_device = torch.device(self.cfg.ema.device)
                except (TypeError, RuntimeError):  # pragma: no cover - config fallback
                    self.accelerator.print(
                        f"[WARN] Invalid EMA device '{self.cfg.ema.device}', falling back to training device"
                    )
                    ema_device = self.device
            else:
                ema_device = self.device
            base_vae = self._unwrap_model(self.vae)
            self.ema_vae = AveragedModel(base_vae, avg_fn=self._ema_avg_fn, device=ema_device)
            if self.latent_upscaler is not None:
                base_up = self._unwrap_model(self.latent_upscaler)
                self.ema_latent_upscaler = AveragedModel(base_up, avg_fn=self._ema_avg_fn, device=ema_device)

        self.fixed_samples = self._sample_fixed_batch()

        if self.cfg.logging.use_wandb and self.accelerator.is_main_process:
            wandb.init(
                project=self.cfg.paths.project,
                name=self.cfg.logging.wandb_run_name,
                config=self._wandb_payload(),
            )
            wandb.config.update({"run_dir": str(self.cfg.paths.run_dir)}, allow_val_change=True)

        if self.cfg.logging.save_model and self.accelerator.is_main_process:
            self.accelerator.print("Generating reference samples before training...")
            self._generate_and_log_samples(step=0)
        self.accelerator.wait_for_everyone()

    # ------------------------------------------------------------------ helpers
    def _seed_everything(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False

    def _load_vae(self) -> nn.Module:
        cfg = self.cfg.model
        load_path = Path(cfg.load_from) if cfg.load_from else None
        path_exists = load_path is not None and load_path.exists()

        hf_source = cfg.hf_repo if cfg.hf_repo else None
        if not path_exists and cfg.load_from and not cfg.hf_repo:
            hf_source = cfg.load_from

        vae: nn.Module
        kind = (cfg.vae_kind or "").strip().lower()

        if kind == "qwen":
            if path_exists:
                source = str(load_path)
                kwargs: Dict[str, Any] = {}
            else:
                source = hf_source or "Qwen/Qwen-Image"
                kwargs = {}
                if cfg.hf_subfolder or not hf_source:
                    kwargs["subfolder"] = cfg.hf_subfolder or "vae"
                if cfg.hf_revision:
                    kwargs["revision"] = cfg.hf_revision
                if cfg.hf_auth_token:
                    kwargs["use_auth_token"] = cfg.hf_auth_token
            vae = AutoencoderKLQwenImage.from_pretrained(source, **kwargs)
        else:
            if path_exists:
                source = str(load_path)
                kwargs = {}
            else:
                source = hf_source or self.cfg.paths.project
                kwargs = {}
                if cfg.hf_subfolder:
                    kwargs["subfolder"] = cfg.hf_subfolder
                if cfg.hf_revision:
                    kwargs["revision"] = cfg.hf_revision
                if cfg.hf_auth_token:
                    kwargs["use_auth_token"] = cfg.hf_auth_token
            if kind == "wan":
                vae = AutoencoderKLWan.from_pretrained(source, **kwargs)
            else:
                if kind in {"kl", "autoencoderkl", "autoencoder_kl"}:
                    vae = AutoencoderKL.from_pretrained(source, **kwargs)
                elif kind in {"asymmetric_kl", "kl_asymmetric", "kl_asym", "asym_kl"}:
                    vae = AsymmetricAutoencoderKL.from_pretrained(source, **kwargs)
                else:
                    if self.cfg.dataset.model_resolution == self.cfg.dataset.high_resolution:
                        vae = AutoencoderKL.from_pretrained(source, **kwargs)
                    else:
                        vae = AsymmetricAutoencoderKL.from_pretrained(source, **kwargs)
        display_source = source if not path_exists else str(load_path)
        self.accelerator.print(f"[INFO] Loading VAE from: {display_source}")
        return vae.to(cfg.weights_dtype)

    def _determine_latent_channels(self, vae: nn.Module) -> Optional[int]:
        core = self._unwrap_model(vae)
        cfg = getattr(core, "config", None)
        latent_channels = getattr(cfg, "latent_channels", None)
        if latent_channels:
            return latent_channels
        decoder = getattr(core, "decoder", None)
        conv_in = getattr(decoder, "conv_in", None) if decoder is not None else None
        return getattr(conv_in, "in_channels", None)

    def _build_latent_upscaler(self, vae: nn.Module) -> Optional[LatentUpscaler]:
        cfg = self.cfg
        if not cfg.latent_upscaler.enabled:
            return None
        scale_factor = cfg.latent_upscaler.scale_factor
        if scale_factor is None:
            if cfg.dataset.high_resolution <= cfg.dataset.model_resolution:
                raise ValueError("Latent upscaler enabled but high_resolution <= model_resolution")
            if cfg.dataset.high_resolution % cfg.dataset.model_resolution != 0:
                raise ValueError("High resolution must be integer multiple of model resolution")
            scale_factor = cfg.dataset.high_resolution // cfg.dataset.model_resolution
        latent_channels = self._determine_latent_channels(vae)
        if latent_channels is None:
            raise RuntimeError("Unable to determine latent channels for upscaler")
        upscaler = LatentUpscaler(
            latent_channels,
            scale_factor=scale_factor,
            hidden_multiplier=cfg.latent_upscaler.width_multiplier,
        )
        return upscaler.to(self.device, dtype=self.cfg.model.weights_dtype)

    def _freeze_parameters(self) -> None:
        core = self._unwrap_model(self.vae)
        for param in core.parameters():
            param.requires_grad = False

        names: List[str] = []
        if self.latent_upscaler is not None:
            for name, param in self.latent_upscaler.named_parameters():
                param.requires_grad = True
                names.append(f"latent_upscaler.{name}")
        elif self.cfg.model.full_training and not self.cfg.model.train_decoder_only:
            for name, param in core.named_parameters():
                param.requires_grad = True
                names.append(name)
        else:
            decoder = getattr(core, "decoder", None)
            if decoder is not None:
                for name, param in decoder.named_parameters():
                    param.requires_grad = True
                    names.append(f"decoder.{name}")
            post_quant = getattr(core, "post_quant_conv", None)
            if post_quant is not None:
                for name, param in post_quant.named_parameters():
                    param.requires_grad = True
                    names.append(f"post_quant_conv.{name}")
        self.accelerator.print(f"[INFO] Unfrozen {len(names)} parameter tensors")

    def _offload_unused_encoder(self) -> None:
        if not self.cfg.embeddings.enabled:
            return

        core = self._unwrap_model(self.vae)
        modules = [("encoder", getattr(core, "encoder", None)), ("quant_conv", getattr(core, "quant_conv", None))]
        freed_any = False
        for name, module in modules:
            if module is None:
                continue
            module.to("cpu")
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)
            freed_any = True
            self.accelerator.print(f"[INFO] Offloaded VAE {name} to CPU")

        if freed_any and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _collect_param_groups(self) -> List[Dict[str, Any]]:
        modules: List[nn.Module] = [self.vae]
        if self.latent_upscaler is not None:
            modules.append(self.latent_upscaler)
        decay, no_decay = [], []
        tokens = ("bias", "norm", "rms", "layernorm")
        for module in modules:
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                (no_decay if any(tok in name.lower() for tok in tokens) else decay).append(param)
        return [
            {
                "params": decay,
                "weight_decay": self.cfg.optimiser.weight_decay if self.cfg.optimiser.use_decay else 0.0,
            },
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def _ema_avg_fn(self, averaged_param: torch.Tensor, model_param: torch.Tensor, _num_averaged: int) -> torch.Tensor:
        decay = self.cfg.ema.decay
        if averaged_param is None:
            return model_param.detach()
        return averaged_param * decay + model_param.detach() * (1.0 - decay)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        groups = self._collect_param_groups()
        cfg = self.cfg.optimiser
        name = cfg.optimizer_type.lower()
        if name == "adam8bit":
            return bnb.optim.AdamW8bit(groups, lr=cfg.base_learning_rate, betas=(0.9, cfg.beta2), eps=cfg.eps)
        if name == "adamw":
            return torch.optim.AdamW(groups, lr=cfg.base_learning_rate, betas=(0.9, cfg.beta2), eps=cfg.eps)
        if name == "adam":
            return torch.optim.Adam(groups, lr=cfg.base_learning_rate, betas=(0.9, cfg.beta2), eps=cfg.eps)
        if name in {"dadapt", "dadaptadam", "d_adapt_adam", "dadapt_adam"}:
            try:
                from dadaptation import DAdaptAdam
            except ImportError as exc:  # pragma: no cover - optional dependency guard
                raise ImportError(
                    "Optimizer type 'dadapt_adam' requested but the 'dadaptation' package is not installed. "
                    "Install it with 'pip install dadaptation'."
                ) from exc
            return DAdaptAdam(
                groups,
                lr=max(cfg.base_learning_rate, 1e-12),
                betas=(0.9, cfg.beta2),
                eps=cfg.eps,
                weight_decay=cfg.weight_decay if cfg.use_decay else 0.0,
                decouple=cfg.use_decay,
            )
        raise ValueError(f"Unsupported optimizer type: {self.cfg.optimiser.optimizer_type}")

    def _total_steps(self) -> int:
        batches_per_epoch = len(self.dataloader)
        effective_batches = batches_per_epoch / max(1, self.cfg.optimiser.gradient_accumulation_steps)
        total = int(math.ceil(effective_batches)) * self.cfg.optimiser.num_epochs
        self.accelerator.print(
            f"[INFO] batches/epoch={batches_per_epoch}, optimiser steps/epoch={int(math.ceil(effective_batches))}, total={total}"
        )
        return max(1, total)

    def _build_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        cfg = self.cfg.optimiser
        total_steps = self._total_steps()
        if cfg.scheduler == "cosine":
            eta_min = cfg.base_learning_rate * cfg.cosine_min_ratio
            return CosineAnnealingLR(self.optimizer, total_steps, eta_min=eta_min)

        warmup_steps = max(1, int(total_steps * cfg.warmup_percent))
        min_ratio = cfg.min_learning_rate / max(cfg.base_learning_rate, 1e-12)

        def schedule(step: int) -> float:
            step += 1
            if step <= warmup_steps:
                return min_ratio + (1.0 - min_ratio) * (step / warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(self.optimizer, schedule)

    def _prepare_batch(
        self, batch: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        dtype = next(self.vae.parameters()).dtype
        if torch.is_tensor(batch):
            tensor = batch.to(self.device)
            return tensor, tensor, None, None, None
        if isinstance(batch, dict):
            image = batch["image"].to(self.device)
            model_input = batch.get("model_input")
            if model_input is None:
                model_input = image
            else:
                model_input = model_input.to(self.device)
            latents = self._convert_optional_tensor(batch.get("latents"), dtype)
            mean = self._convert_optional_tensor(batch.get("latent_mean"), dtype)
            logvar = self._convert_optional_tensor(batch.get("latent_logvar"), dtype)
            return model_input, image, latents, mean, logvar
        raise TypeError(f"Unsupported batch type: {type(batch)!r}")

    def _convert_optional_tensor(self, value: Any, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if isinstance(value, list) and value and torch.is_tensor(value[0]):
            value = torch.stack(value)
        if torch.is_tensor(value):
            return value.to(self.device, dtype=dtype)
        return None

    def _update_ema(self, step: int) -> None:
        if self.ema_vae is None:
            return
        if step < self.cfg.ema.update_after_step:
            return
        if (step - self.cfg.ema.update_after_step) % self.cfg.ema.update_interval != 0:
            return
        base_vae = self._unwrap_model(self.vae)
        self.ema_vae.update_parameters(base_vae)
        if self.ema_latent_upscaler is not None and self.latent_upscaler is not None:
            base_up = self._unwrap_model(self.latent_upscaler)
            self.ema_latent_upscaler.update_parameters(base_up)

    def _sample_fixed_batch(self, count: int = 4) -> Dict[str, torch.Tensor]:
        indices = random.sample(range(len(self.dataset)), min(count, len(self.dataset)))
        high_tensors: List[torch.Tensor] = []
        model_inputs: List[torch.Tensor] = []
        for idx in indices:
            sample = self.dataset[idx]
            if isinstance(sample, dict):
                high_tensors.append(sample["image"])
                model_inputs.append(sample.get("model_input", sample["image"]))
            else:
                high_tensors.append(sample)
                model_inputs.append(sample)
        high_batch = torch.stack(high_tensors)
        model_batch = torch.stack(model_inputs)
        return {
            "image": high_batch.to(self.device),
            "model_input": model_batch.to(self.device),
        }

    def _wandb_payload(self) -> Dict[str, Any]:
        cfg = self.cfg
        return {
            "batch_size": cfg.optimiser.batch_size,
            "base_learning_rate": cfg.optimiser.base_learning_rate,
            "num_epochs": cfg.optimiser.num_epochs,
            "optimizer_type": cfg.optimiser.optimizer_type,
            "gradient_accumulation_steps": cfg.optimiser.gradient_accumulation_steps,
            "model_resolution": cfg.dataset.model_resolution,
            "high_resolution": cfg.dataset.high_resolution,
            "train_decoder_only": cfg.model.train_decoder_only,
            "full_training": cfg.model.full_training,
            "vae_kind": cfg.model.vae_kind,
            "latent_upscaler_enabled": cfg.latent_upscaler.enabled,
        }

    # ---------------------------------------------------------------- training
    def train(self) -> None:
        total_steps = self._total_steps()
        sample_interval = self._resolve_interval(
            self.cfg.logging.global_sample_interval,
            total_steps,
            self.cfg.logging.sample_interval_share,
        )
        save_interval = self.cfg.logging.global_save_interval

        progress = tqdm(total=total_steps, disable=not self.accelerator.is_local_main_process)
        best_loss = float("inf")
        best_step = -1
        global_step = 0

        for epoch in range(self.cfg.optimiser.num_epochs):
            self.vae.train()
            if self.latent_upscaler is not None:
                self.latent_upscaler.train()
            batch_losses: List[float] = []
            batch_grads: List[float] = []

            for batch in self.dataloader:
                with self.accelerator.accumulate(self.vae):
                    _, high_res, cached_latents, cached_mean, cached_logvar = self._prepare_batch(batch)
                    with self.accelerator.autocast():
                        reconstruction, encode_out = self._forward(
                            precomputed_latents=cached_latents,
                            cached_mean=cached_mean,
                            cached_logvar=cached_logvar,
                        )
                        reconstruction = self._match_spatial(reconstruction, high_res)
                        losses = self._compute_losses(reconstruction, high_res, encode_out)
                        total_loss, coeffs, medians = self.loss_normalizer.update(losses)
                    if not torch.isfinite(total_loss):
                        raise RuntimeError("Encountered NaN/Inf loss")

                    self.accelerator.backward(total_loss)

                    grad_norm = torch.tensor(0.0, device=self.device)
                    if self.accelerator.sync_gradients:
                        if self.cfg.optimiser.clip_grad_norm > 0:
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.trainable_params, self.cfg.optimiser.clip_grad_norm
                            )
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        self._update_ema(global_step)
                        progress.update(1)

                    if self.accelerator.is_main_process:
                        lr = self.optimizer.param_groups[0]["lr"]
                        batch_losses.append(float(total_loss.detach().item()))
                        batch_grads.append(float(grad_norm.detach().item()) if isinstance(grad_norm, torch.Tensor) else 0.0)
                        if self.cfg.logging.use_wandb and self.accelerator.sync_gradients:
                            log = {
                                "total_loss": float(total_loss.detach().item()),
                                "learning_rate": lr,
                                "epoch": epoch,
                            }
                            if self.cfg.logging.log_grad_norm:
                                log["grad_norm"] = batch_grads[-1]
                            for key, value in losses.items():
                                log[f"loss_{key}"] = float(value.detach().item())
                            for key, value in coeffs.items():
                                log[f"coeff_{key}"] = float(value)
                                log[f"median_{key}"] = float(medians.get(key, 0.0))
                            wandb.log(log, step=global_step)

                    if (
                        self.cfg.logging.save_model
                        and self.accelerator.is_main_process
                        and save_interval
                        and global_step > 0
                        and global_step % save_interval == 0
                    ):
                        self._save_checkpoint(self.cfg.paths.checkpoints_dir / f"step_{global_step:08d}")

                    if global_step > 0 and global_step % sample_interval == 0:
                        if self.accelerator.is_main_process:
                            self._generate_and_log_samples(global_step)
                        self.accelerator.wait_for_everyone()
                        avg_loss = float(np.mean(batch_losses[-sample_interval:])) if batch_losses else float("nan")
                        avg_grad = float(np.mean(batch_grads[-sample_interval:])) if batch_grads else 0.0
                        if self.accelerator.is_main_process:
                            self.accelerator.print(
                                f"Epoch {epoch} step {global_step}: loss={avg_loss:.6f}, grad_norm={avg_grad:.6f}, lr={lr:.6e}"
                            )
                            if (
                                self.cfg.logging.save_model
                                and math.isfinite(avg_loss)
                                and avg_loss < best_loss * self.cfg.logging.save_barrier
                            ):
                                best_loss = avg_loss
                                best_step = global_step
                                self._save_best_checkpoint(best_loss, best_step)
                            if self.cfg.logging.use_wandb:
                                wandb.log({"interm_loss": avg_loss, "interm_grad": avg_grad}, step=global_step)

            if self.accelerator.is_main_process:
                epoch_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
                self.accelerator.print(f"Epoch {epoch} finished, avg loss {epoch_loss:.6f}")
                if self.cfg.logging.use_wandb:
                    wandb.log({"epoch_loss": epoch_loss, "epoch": epoch + 1}, step=global_step)

        if self.accelerator.is_main_process:
            self.accelerator.print("Training complete – storing final model")
            if self.cfg.logging.save_model:
                self._save_final_model()
        self.accelerator.free_memory()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        self.accelerator.print("Training run complete")

    # ---------------------------------------------------------------- private
    def _forward(
        self,
        precomputed_latents: Optional[torch.Tensor] = None,
        cached_mean: Optional[torch.Tensor] = None,
        cached_logvar: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        dtype = next(self.vae.parameters()).dtype
        is_video = self._is_video_vae(self.vae)

        if precomputed_latents is None:
            raise RuntimeError(
                "Precomputed latents are required but missing. Generate embeddings before training and "
                "ensure the dataloader returns cached latents."
            )

        latents = precomputed_latents.to(self.device, dtype=dtype)
        mean = cached_mean.to(self.device, dtype=dtype) if cached_mean is not None else None
        logvar = cached_logvar.to(self.device, dtype=dtype) if cached_logvar is not None else None
        if is_video:
            latents = latents.unsqueeze(2)
            if mean is not None:
                mean = mean.unsqueeze(2)
            if logvar is not None:
                logvar = logvar.unsqueeze(2)
        encoding_result = self._build_cached_encoding(latents, mean, logvar)

        if self.latent_upscaler is not None:
            compiler_mod = getattr(torch, "compiler", None)
            mark_step = getattr(compiler_mod, "cudagraph_mark_step_begin", None)
            if callable(mark_step):
                mark_step()
            latents = self.latent_upscaler(latents.clone())
        latents = latents.to(dtype)
        if is_video:
            decoded = self.vae.decode(latents).sample.squeeze(2)
        else:
            decoded = self.vae.decode(latents).sample
        return decoded, {"encoding": encoding_result, "latents": latents}

    @staticmethod
    def _build_cached_encoding(
        latents: torch.Tensor,
        mean: Optional[torch.Tensor],
        logvar: Optional[torch.Tensor],
    ) -> SimpleNamespace:
        effective_mean = mean if mean is not None else latents.detach()
        if logvar is None:
            logvar = torch.zeros_like(effective_mean, device=effective_mean.device, dtype=effective_mean.dtype)
        dist = SimpleNamespace(mean=effective_mean, logvar=logvar)
        return SimpleNamespace(latent_dist=dist)

    def _match_spatial(self, tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if tensor.shape[-2:] == reference.shape[-2:]:
            return tensor
        return F.interpolate(tensor, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def _compute_losses(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        encode_result: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        rec = reconstruction.to(torch.float32)
        tgt = target.to(torch.float32)
        active = set(self.cfg.losses.active_losses)
        losses: Dict[str, torch.Tensor] = {}

        if "mae" in active:
            losses["mae"] = F.l1_loss(rec, tgt)
        if "mse" in active:
            losses["mse"] = F.mse_loss(rec, tgt)
        if "edge" in active:
            losses["edge"] = F.l1_loss(self._sobel(rec), self._sobel(tgt))

        if "ffl" in active:
            if self.ffl_module is None:
                losses["ffl"] = torch.zeros((), device=self.device, dtype=rec.dtype)
            else:
                losses["ffl"] = self.ffl_module(rec, tgt)

        if "lpips" in active:
            losses["lpips"] = self._lpips_loss(rec, tgt)

        if "kl" in active:
            if self.cfg.model.full_training and not self.cfg.model.train_decoder_only:
                dist = encode_result["encoding"].latent_dist
                mean = dist.mean
                logvar = dist.logvar
                losses["kl"] = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
            else:
                losses["kl"] = torch.zeros((), device=self.device, dtype=rec.dtype)
        return losses

    def _lpips_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.lpips_module is None:
            return torch.tensor(0.0, device=self.device)
        size = self.cfg.losses.lpips_eval_resolution
        pred = prediction
        tgt = target
        if size > 0 and prediction.shape[-2:] != (size, size):
            pred = F.interpolate(prediction, size=(size, size), mode="bilinear", align_corners=False)
        if size > 0 and target.shape[-2:] != (size, size):
            tgt = F.interpolate(target, size=(size, size), mode="bilinear", align_corners=False)
        return self.lpips_module(pred, tgt).mean()

    @staticmethod
    def _sobel(tensor: torch.Tensor) -> torch.Tensor:
        channels = tensor.shape[1]
        kernel_x = torch.tensor(
            [[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], device=tensor.device, dtype=tensor.dtype
        )
        kernel_y = torch.tensor(
            [[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], device=tensor.device, dtype=tensor.dtype
        )
        kernel_x = kernel_x.repeat(channels, 1, 1, 1)
        kernel_y = kernel_y.repeat(channels, 1, 1, 1)
        grad_x = F.conv2d(tensor, kernel_x, padding=1, groups=channels)
        grad_y = F.conv2d(tensor, kernel_y, padding=1, groups=channels)
        return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-12)

    @contextmanager
    def _evaluation_context(self, module: Optional[nn.Module]):
        if module is None:
            yield None
            return
        prev_mode = module.training
        first_param = next((p for p in module.parameters()), None)
        original_device = first_param.device if first_param is not None else self.device
        if original_device != self.device:
            module = module.to(self.device)
        module.eval()
        try:
            yield module
        finally:
            module.train(prev_mode)
            if original_device != self.device:
                module.to(original_device)

    def _generate_and_log_samples(self, step: int) -> None:
        try:
            vae_source = self.ema_vae.module if self.ema_vae is not None else self._unwrap_model(self.vae)
            if self.ema_latent_upscaler is not None:
                upscaler_source: Optional[nn.Module] = self.ema_latent_upscaler.module
            else:
                upscaler_source = self._unwrap_model(self.latent_upscaler)
            with self._evaluation_context(vae_source) as vae, self._evaluation_context(upscaler_source) as upscaler:
                if vae is None:
                    return
                with torch.no_grad():
                    samples = self.fixed_samples
                    high = samples["image"]
                    low = samples["model_input"]
                    first_param = next((p for p in vae.parameters()), None)
                    dtype = first_param.dtype if first_param is not None else torch.float32
                    low = low.to(dtype)
                    if self._is_video_vae(vae):
                        latents = vae.encode(low.unsqueeze(2)).latent_dist.mean
                        if upscaler is not None:
                            latents = upscaler(latents)
                        latents = latents.to(dtype)
                        recon = vae.decode(latents).sample.squeeze(2)
                    else:
                        latents = vae.encode(low).latent_dist.mean
                        if upscaler is not None:
                            latents = upscaler(latents)
                        latents = latents.to(dtype)
                        recon = vae.decode(latents).sample
                    recon = self._match_spatial(recon, high)

            self.cfg.paths.samples_dir.mkdir(parents=True, exist_ok=True)
            grid_path = self._save_sample_grid(high, recon, step)
            if self.cfg.logging.use_wandb and self.accelerator.is_main_process:
                wandb.log({"samples/pairs": wandb.Image(str(grid_path))}, step=step)
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    def _save_sample_grid(self, originals: torch.Tensor, decoded: torch.Tensor, step: int) -> Path:
        def to_pil(tensor: torch.Tensor):
            array = ((tensor.clamp(-1, 1) + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
            array = array.permute(1, 2, 0).cpu().numpy()
            from PIL import Image

            return Image.fromarray(array)

        num = decoded.shape[0]
        fig, axes = plt.subplots(2, num, figsize=(num * 2.5, 6))
        if num == 1:
            axes = np.array([[axes[0]], [axes[1]]])
        for idx in range(num):
            axes[0, idx].imshow(to_pil(originals[idx]))
            axes[0, idx].axis("off")
            axes[0, idx].set_title(f"Real {idx}")
            axes[1, idx].imshow(to_pil(decoded[idx]))
            axes[1, idx].axis("off")
            axes[1, idx].set_title(f"Decoded {idx}")
        plt.tight_layout()
        grid_path = self.cfg.paths.samples_dir / f"grid_{step}.png"
        fig.savefig(grid_path, dpi=150)
        plt.close(fig)
        to_pil(originals[0]).save(self.cfg.paths.samples_dir / "sample_real.jpg", quality=95)
        to_pil(decoded[0]).save(self.cfg.paths.samples_dir / "sample_decoded.jpg", quality=95)
        return grid_path

    def _save_checkpoint(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        model = self.accelerator.unwrap_model(self.vae)
        model.save_pretrained(str(directory))
        torch.save(self.optimizer.state_dict(), directory / "optimizer.pt")
        torch.save(self.scheduler.state_dict(), directory / "scheduler.pt")
        if self.latent_upscaler is not None:
            torch.save(
                {
                    "state_dict": self.accelerator.unwrap_model(self.latent_upscaler).state_dict(),
                    "scale_factor": self.cfg.dataset.high_resolution // self.cfg.dataset.model_resolution
                    if self.cfg.dataset.high_resolution > self.cfg.dataset.model_resolution
                    else self.cfg.latent_upscaler.scale_factor,
                    "width_multiplier": self.cfg.latent_upscaler.width_multiplier,
                },
                directory / "latent_upscaler.pt",
            )
        if self.ema_vae is not None:
            torch.save(
                {
                    "state_dict": self.ema_vae.module.state_dict(),
                    "decay": self.cfg.ema.decay,
                    "update_after_step": self.cfg.ema.update_after_step,
                    "update_interval": self.cfg.ema.update_interval,
                },
                directory / "ema_vae.pt",
            )
            if self.ema_latent_upscaler is not None:
                torch.save(
                    {
                        "state_dict": self.ema_latent_upscaler.module.state_dict(),
                        "decay": self.cfg.ema.decay,
                        "update_after_step": self.cfg.ema.update_after_step,
                        "update_interval": self.cfg.ema.update_interval,
                    },
                    directory / "ema_latent_upscaler.pt",
                )

    def _save_best_checkpoint(self, loss: float, step: int) -> None:
        if self.cfg.paths.best_dir.exists():
            shutil.rmtree(self.cfg.paths.best_dir)
        self.cfg.paths.best_dir.mkdir(parents=True, exist_ok=True)
        self._save_checkpoint(self.cfg.paths.best_dir)
        (self.cfg.paths.best_dir / "best_summary.txt").write_text(
            f"step={step}\nloss={loss}\n", encoding="utf-8"
        )
        self.accelerator.print(f"[CHECKPOINT] Updated best model at step {step} (loss {loss:.6f})")

    def _save_final_model(self) -> None:
        if self.cfg.paths.final_dir.exists():
            shutil.rmtree(self.cfg.paths.final_dir)
        self.cfg.paths.final_dir.mkdir(parents=True, exist_ok=True)
        self._save_checkpoint(self.cfg.paths.final_dir)

    def _unwrap_model(self, module: Optional[nn.Module]) -> Optional[nn.Module]:
        if module is None:
            return None
        if hasattr(self.accelerator, "unwrap_model"):
            try:
                module = self.accelerator.unwrap_model(module)
            except Exception:  # pragma: no cover - defensive
                pass
        if hasattr(module, "_orig_mod"):
            module = module._orig_mod
        return module

    def _is_video_vae(self, module: nn.Module) -> bool:
        encoder = getattr(module, "encoder", None)
        conv_in = getattr(encoder, "conv_in", None) if encoder is not None else None
        weight = getattr(conv_in, "weight", None) if conv_in is not None else None
        return isinstance(weight, torch.nn.Parameter) and weight.ndimension() == 5

    @staticmethod
    def _resolve_interval(configured: int, total_steps: int, share: int) -> int:
        if configured > 0:
            return configured
        return max(1, total_steps // max(1, share))


def run(config: Dict[str, Any]) -> None:
    training_config = TrainingConfig.from_dict(config)
    trainer = VAETrainer(training_config)
    trainer.train()
