"""Utilities for generating and logging sample reconstructions to Weights & Biases."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from diffusers import (
    AsymmetricAutoencoderKL,
    AutoencoderKL,
    AutoencoderKLQwenImage,
    AutoencoderKLWan,
)
import lpips
import wandb

from .config import TrainingConfig
from .wandb_logger import WandbLogger

__all__ = ["SampleLogger"]


def _to_pil_uint8(tensor: torch.Tensor) -> Image.Image:
    """Convert a tensor in [-1, 1] to a uint8 PIL image."""

    image = tensor.detach().cpu().clamp(-1.0, 1.0)
    image = (image + 1.0) * 0.5  # [-1, 1] -> [0, 1]
    image = image.mul(255.0).clamp(0.0, 255.0).byte()
    if image.dim() == 4:
        image = image.squeeze(0)
    return Image.fromarray(image.permute(1, 2, 0).numpy())


def _is_video_vae(module: torch.nn.Module) -> bool:
    encoder = getattr(module, "encoder", None)
    conv_in = getattr(encoder, "conv_in", None) if encoder is not None else None
    weight = getattr(conv_in, "weight", None) if conv_in is not None else None
    return isinstance(weight, torch.nn.Parameter) and weight.ndimension() == 5


class SampleLogger:
    """Generate qualitative samples using a VAE decoder and log them to wandb."""

    def __init__(
        self,
        cfg: TrainingConfig,
        *,
        dataset: Sequence,
        max_samples: int = 4,
        wandb_logger: Optional[WandbLogger] = None,
    ) -> None:
        self.cfg = cfg
        self._wandb = wandb_logger
        self.generated_folder = Path(cfg.paths.samples_dir)
        self.generated_folder.mkdir(parents=True, exist_ok=True)
        self.sample_interval = max(1, cfg.logging.global_sample_interval)
        self.sample_count = max(1, min(max_samples, len(dataset)))

        self.fixed_low, self.fixed_high = self._collect_fixed_latents(dataset)
        self._vae: Optional[torch.nn.Module] = None
        self._lpips: Optional[lpips.LPIPS] = None
        self._is_video_vae: bool = False

    # --------------------------------------------------------------------- helpers
    def _collect_fixed_latents(self, dataset: Sequence) -> Tuple[torch.Tensor, torch.Tensor]:
        lows: list[torch.Tensor] = []
        highs: list[torch.Tensor] = []

        for index in range(self.sample_count):
            sample = dataset[index]
            low = sample["low"].detach().clone()
            high = sample["high"].detach().clone()
            if low.dim() == 3:
                low = low.unsqueeze(0)
            if high.dim() == 3:
                high = high.unsqueeze(0)
            lows.append(low.squeeze(0))
            highs.append(high.squeeze(0))

        low_stack = torch.stack(lows, dim=0)
        high_stack = torch.stack(highs, dim=0)
        return low_stack, high_stack

    def _load_vae(self) -> torch.nn.Module:
        model_cfg = self.cfg.model
        load_path = Path(model_cfg.load_from).expanduser() if model_cfg.load_from else None
        path_exists = load_path is not None and load_path.exists()

        hf_source = model_cfg.hf_repo or None
        if not path_exists and model_cfg.load_from and not model_cfg.hf_repo:
            hf_source = model_cfg.load_from

        kind = (model_cfg.vae_kind or "").strip().lower()

        if kind == "qwen":
            if path_exists:
                source = str(load_path)
                kwargs = {}
            else:
                source = hf_source or "Qwen/Qwen-Image"
                kwargs = {}
                if model_cfg.hf_subfolder or not hf_source:
                    kwargs["subfolder"] = model_cfg.hf_subfolder or "vae"
                if model_cfg.hf_revision:
                    kwargs["revision"] = model_cfg.hf_revision
                if model_cfg.hf_auth_token:
                    kwargs["use_auth_token"] = model_cfg.hf_auth_token
            vae: torch.nn.Module = AutoencoderKLQwenImage.from_pretrained(source, **kwargs)
        else:
            if path_exists:
                source = str(load_path)
                kwargs = {}
            else:
                source = hf_source
                if not source:
                    raise RuntimeError(
                        "Sample logging requires a VAE source (either 'load_from' or 'hf_repo')."
                    )
                kwargs = {}
                if model_cfg.hf_subfolder:
                    kwargs["subfolder"] = model_cfg.hf_subfolder
                if model_cfg.hf_revision:
                    kwargs["revision"] = model_cfg.hf_revision
                if model_cfg.hf_auth_token:
                    kwargs["use_auth_token"] = model_cfg.hf_auth_token

            if kind == "wan":
                vae = AutoencoderKLWan.from_pretrained(source, **kwargs)
            elif kind in {"kl", "autoencoderkl", "autoencoder_kl"}:
                vae = AutoencoderKL.from_pretrained(source, **kwargs)
            elif kind in {"asymmetric_kl", "kl_asymmetric", "kl_asym", "asym_kl"}:
                vae = AsymmetricAutoencoderKL.from_pretrained(source, **kwargs)
            else:
                if self.cfg.dataset.model_resolution == self.cfg.dataset.high_resolution:
                    vae = AutoencoderKL.from_pretrained(source, **kwargs)
                else:
                    vae = AsymmetricAutoencoderKL.from_pretrained(source, **kwargs)

        vae = vae.to(dtype=self.cfg.model.weights_dtype)
        self._is_video_vae = _is_video_vae(vae)
        return vae.eval()

    def _ensure_models(self, device: torch.device) -> Tuple[torch.nn.Module, lpips.LPIPS]:
        if self._vae is None:
            self._vae = self._load_vae()
        if self._lpips is None:
            self._lpips = lpips.LPIPS(net=self.cfg.losses.lpips_backbone or "vgg").eval()

        self._vae = self._vae.to(device)
        self._lpips = self._lpips.to(device)
        return self._vae, self._lpips

    def _offload_models(self) -> None:
        if self._vae is not None:
            self._vae = self._vae.to("cpu")
        if self._lpips is not None:
            self._lpips = self._lpips.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _decode_latents(self, vae: torch.nn.Module, latents: torch.Tensor) -> torch.Tensor:
        decoder_dtype = next(vae.parameters()).dtype
        inputs = latents.to(dtype=decoder_dtype)
        if self._is_video_vae and inputs.dim() == 4:
            inputs = inputs.unsqueeze(2)
        decoded = vae.decode(inputs).sample
        if decoded.dim() == 5:
            decoded = decoded.squeeze(2)
        return decoded.to(torch.float32)

    # ----------------------------------------------------------------- main entry
    def maybe_log_samples(
        self,
        *,
        model: torch.nn.Module,
        step: int,
        device: torch.device,
    ) -> None:
        if step % self.sample_interval != 0:
            return

        vae, lpips_model = self._ensure_models(device)
        model_prev_mode = model.training
        model.eval()

        try:
            with torch.no_grad():
                model_dtype = next(model.parameters()).dtype
                low_latents = self.fixed_low.to(device=device, dtype=model_dtype)
                high_latents = self.fixed_high.to(device=device, dtype=model_dtype)

                predicted_latents = model(low_latents)

                real_images = self._decode_latents(vae, high_latents)
                decoded_images = self._decode_latents(vae, predicted_latents)

            if decoded_images.shape[-2:] != real_images.shape[-2:]:
                decoded_images = F.interpolate(
                    decoded_images,
                    size=real_images.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            real_cpu = real_images.detach().cpu()
            decoded_cpu = decoded_images.detach().cpu()

            # Save quick inspection samples
            first_real = _to_pil_uint8(real_cpu[0])
            first_decoded = _to_pil_uint8(decoded_cpu[0])
            first_real.save(self.generated_folder / "sample_real.jpg", quality=95)
            first_decoded.save(self.generated_folder / "sample_decoded.jpg", quality=95)

            sample_paths = []
            for index in range(decoded_cpu.shape[0]):
                sample_path = self.generated_folder / f"sample_{index}.jpg"
                _to_pil_uint8(decoded_cpu[index]).save(sample_path, quality=95)
                sample_paths.append(sample_path)

            # Build comparison grid
            n = decoded_cpu.shape[0]
            fig, axes = plt.subplots(2, n, figsize=(n * 2.5, 6))
            if n == 1:
                axes = np.array([[axes[0]], [axes[1]]])

            for idx in range(n):
                axes[0, idx].imshow(_to_pil_uint8(real_cpu[idx]))
                axes[0, idx].axis("off")
                axes[0, idx].set_title(f"Real {idx}")

                axes[1, idx].imshow(_to_pil_uint8(decoded_cpu[idx]))
                axes[1, idx].axis("off")
                axes[1, idx].set_title(f"Decoded {idx}")

            plt.tight_layout()
            pairs_path = self.generated_folder / "samples_pairs.jpg"
            fig.savefig(pairs_path, dpi=150)
            plt.close(fig)

            # LPIPS metric
            lpips_scores = []
            with torch.no_grad():
                real_lpips = real_images.to(device=device, dtype=torch.float32)
                rec_lpips = decoded_images.to(device=device, dtype=torch.float32)
                for idx in range(rec_lpips.shape[0]):
                    score = lpips_model(
                        rec_lpips[idx : idx + 1], real_lpips[idx : idx + 1]
                    ).mean().item()
                    lpips_scores.append(score)

            avg_lpips = float(np.mean(lpips_scores)) if lpips_scores else 0.0

            log_payload: Dict[str, Any] = {
                "samples/lpips_mean": avg_lpips,
                "samples/pairs": wandb.Image(
                    str(pairs_path), caption=f"{decoded_cpu.shape[0]} pairs (real vs decoded)"
                ),
            }
            for index, sample_path in enumerate(sample_paths):
                log_payload[f"samples/decoded_{index}"] = wandb.Image(str(sample_path))

            if self._wandb is not None and self._wandb.is_active:
                self._wandb.log(log_payload, step=step)
        finally:
            model.train(model_prev_mode)
            self._offload_models()
