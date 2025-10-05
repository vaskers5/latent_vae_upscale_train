"""Shared helper modules for the VAE trainer."""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["LatentUpscaler", "MedianLossNormalizer", "FocalFrequencyLoss"]


class LatentUpscaler(nn.Module):
    def __init__(self, channels: int, scale_factor: int, hidden_multiplier: int) -> None:
        super().__init__()
        if scale_factor < 2:
            raise ValueError("Latent upscaler requires scale_factor >= 2")
        hidden_channels = max(channels * hidden_multiplier, channels)
        self.scale_factor = scale_factor
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels * (scale_factor**2), kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.dim() == 5:  # video VAE (B, C, T, H, W)
            b, c, t, h, w = latents.shape
            latents_2d = latents.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            upsampled = self._forward_2d(latents_2d)
            _, c2, h2, w2 = upsampled.shape
            upsampled = upsampled.reshape(b, t, c2, h2, w2).permute(0, 2, 1, 3, 4)
            return upsampled
        if latents.dim() != 4:
            raise ValueError(f"Expected 4D or 5D latents, got shape {latents.shape}")
        return self._forward_2d(latents)

    def _forward_2d(self, latents: torch.Tensor) -> torch.Tensor:
        residual = F.interpolate(latents, scale_factor=self.scale_factor, mode="nearest")
        out = self.net(latents)
        return self.act(out + residual.to(out.dtype))


class MedianLossNormalizer:
    def __init__(self, desired_ratios: Dict[str, float], window_steps: int, device: Optional[torch.device] = None) -> None:
        total = sum(desired_ratios.values()) or 1.0
        self.target = {k: v / total for k, v in desired_ratios.items()}
        self.buffers = {k: deque(maxlen=window_steps) for k in desired_ratios.keys()}
        self.device = device

    def update(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:
        if not losses:
            zero = torch.zeros((), device=self.device) if self.device is not None else torch.tensor(0.0)
            return zero, {}, {}
        for key, value in losses.items():
            if key in self.buffers:
                self.buffers[key].append(float(value.detach().abs().cpu()))
        medians = {k: (np.median(buf) if buf else 1.0) for k, buf in self.buffers.items()}
        coeffs: Dict[str, float] = {}
        for key in losses.keys():
            denom = max(medians.get(key, 1.0), 1e-12)
            coeffs[key] = self.target.get(key, 0.0) / denom
        first_loss = next(iter(losses.values()))
        total_loss = torch.zeros((), device=first_loss.device, dtype=first_loss.dtype)
        for key, value in losses.items():
            total_loss = total_loss + coeffs[key] * value
        return total_loss, coeffs, medians


class FocalFrequencyLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        patch_factor: int = 1,
        ave_spectrum: bool = False,
        log_weight: bool = True,
        normalize: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.patch_factor = max(1, patch_factor)
        self.ave_spectrum = ave_spectrum
        self.log_weight = log_weight
        self.normalize = normalize
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = prediction.to(torch.float32)
        ref = target.to(torch.float32)
        pred_freq = torch.fft.rfft2(pred, dim=(-2, -1), norm="ortho")
        ref_freq = torch.fft.rfft2(ref, dim=(-2, -1), norm="ortho")
        diff = pred_freq - ref_freq
        magnitude = diff.abs()
        if self.patch_factor > 1:
            magnitude = magnitude[..., :: self.patch_factor, :: self.patch_factor]
        if self.ave_spectrum:
            weight = magnitude.mean(dim=(0, 1), keepdim=True)
        else:
            weight = magnitude
        if self.log_weight:
            weight = torch.log1p(weight)
        weight = torch.pow(weight + self.eps, self.alpha)
        loss_matrix = weight * (magnitude**2)
        if self.normalize:
            return loss_matrix.mean()
        return loss_matrix.sum() / loss_matrix.numel()
