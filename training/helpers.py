"""Shared helper modules for the VAE trainer."""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

__all__ = ["PixNerfUpscaler", "MedianLossNormalizer", "FocalFrequencyLoss"]


class NerfHead(nn.Module):
    def __init__(self, channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(channels)
        self.weight_net = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, channels * channels),
        )
        self.bias_net = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, channels),
        )

    def forward(self, positions: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        if positions.shape != patches.shape:
            raise ValueError(f"positions shape {positions.shape} must match patches shape {patches.shape}.")
        residual = patches
        normed = self.norm(patches)
        normed = normed.transpose(1, 2)
        summary = positions.mean(dim=1)
        weights = self.weight_net(summary).view(-1, self.channels, self.channels)
        bias = self.bias_net(summary).unsqueeze(-1)
        transformed = torch.bmm(weights, normed) + bias
        transformed = transformed.transpose(1, 2)
        return transformed + residual


class PixNerfUpscaler(nn.Module):
    def __init__(
        self,
        channels: int,
        patch_size: int = 4,
        hidden_dim_multiplier: int = 4,
        nerf_blocks: int = 2,
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        self.channels = channels
        self.patch_size = patch_size
        hidden_dim = max(channels * hidden_dim_multiplier, channels)
        self.pixel_positions = nn.Parameter(torch.randn(1, patch_size * patch_size, channels))
        self.patch_position_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, channels),
        )
        self.nerf = nn.ModuleList(
            [NerfHead(channels=channels, hidden_dim=hidden_dim) for _ in range(max(1, nerf_blocks))]
        )
        self.out_conv = nn.Conv2d(channels, channels * 4, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"PixNerfUpscaler expects 4D latents, got shape {x.shape}.")
        b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(f"Expected {self.channels} channels, received {c}.")
        ps = self.patch_size
        if h % ps != 0 or w % ps != 0:
            raise ValueError(f"Image dimensions ({h}, {w}) must be divisible by patch size {ps}.")

        patch_h = h // ps
        patch_w = w // ps

        patches = x.unfold(2, ps, ps).unfold(3, ps, ps)
        patches = patches.permute(0, 2, 3, 4, 5, 1).contiguous()
        patches = patches.view(b, patch_h * patch_w, ps * ps, c)
        patches = patches.view(b * patch_h * patch_w, ps * ps, c)

        device = x.device
        dtype = x.dtype
        coords_y = torch.arange(patch_h, device=device, dtype=dtype)
        coords_x = torch.arange(patch_w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing="ij")
        coords = torch.stack([grid_y, grid_x], dim=-1).view(-1, 2)
        if patch_h > 1:
            coords[:, 0] /= patch_h - 1
        else:
            coords[:, 0] = 0
        if patch_w > 1:
            coords[:, 1] /= patch_w - 1
        else:
            coords[:, 1] = 0
        coords = coords.unsqueeze(0).repeat(b, 1, 1).view(-1, 2)
        patch_pos = self.patch_position_encoder(coords)

        pixel_pos = self.pixel_positions.expand(patches.size(0), -1, -1).to(dtype=dtype, device=device)
        positions = pixel_pos + patch_pos.unsqueeze(1)
        patches = patches.to(dtype)

        processed = patches
        for block in self.nerf:
            processed = block(positions, processed)

        processed = processed.view(b, patch_h, patch_w, ps * ps, c)
        processed = processed.view(b, patch_h, patch_w, ps, ps, c)
        processed = processed.permute(0, 5, 1, 3, 2, 4).contiguous()
        merged = processed.view(b, c, h, w)

        out = self.out_conv(merged)
        out = self.pixel_shuffle(out)
        return out


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
