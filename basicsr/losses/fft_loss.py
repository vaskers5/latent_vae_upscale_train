from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from basicsr.losses.loss_util import weighted_loss
from basicsr.utils.registry import LOSS_REGISTRY

_REDUCTION_MODES = {"none", "mean", "sum"}


def _maybe_convert_dtype(tensor: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.dtype]]:
    """Convert unsupported FFT dtypes to float32 while keeping the original dtype."""

    if tensor.dtype in {torch.float16, torch.bfloat16}:
        return tensor.float(), tensor.dtype
    return tensor, None


@weighted_loss
def _fft_spectrum_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    norm: str,
    eps: float,
    use_log_amplitude: bool,
    alpha: float,
    normalize_weight: bool,
) -> torch.Tensor:
    """Element-wise L1 loss computed on magnitude spectra."""

    if pred.shape != target.shape:
        raise ValueError(
            f"FFTFrequencyLoss expects tensors with identical shapes, "
            f"received {tuple(pred.shape)} and {tuple(target.shape)}."
        )

    pred_cast, restore_dtype = _maybe_convert_dtype(pred)
    target_cast, _ = _maybe_convert_dtype(target)

    pred_spec = torch.fft.fft2(pred_cast, norm=norm)
    target_spec = torch.fft.fft2(target_cast, norm=norm)

    pred_mag = torch.abs(pred_spec)
    target_mag = torch.abs(target_spec)

    if use_log_amplitude:
        pred_mag = torch.log(pred_mag.clamp_min(eps))
        target_mag = torch.log(target_mag.clamp_min(eps))

    spectral_diff = torch.abs(pred_mag - target_mag)

    if alpha != 0:
        weight = _build_frequency_weight(
            spectral_diff,
            alpha=alpha,
            normalize=normalize_weight,
            eps=eps,
        )
        spectral_diff = spectral_diff * weight

    # Cast back to the original dtype if necessary (keeps higher precision for gradients)
    if restore_dtype is not None:
        spectral_diff = spectral_diff.to(restore_dtype)

    return spectral_diff


def _build_frequency_weight(
    tensor: torch.Tensor,
    *,
    alpha: float,
    normalize: bool,
    eps: float,
) -> torch.Tensor:
    """Create a radial frequency weighting map for the provided tensor."""

    b, c, h, w = tensor.shape
    device = tensor.device
    dtype = tensor.dtype

    fy = torch.fft.fftfreq(h, device=device, dtype=dtype)
    fx = torch.fft.fftfreq(w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(grid_x**2 + grid_y**2)
    weight = torch.pow(radius + eps, alpha)
    if normalize:
        weight = weight / weight.mean().clamp_min(eps)

    weight = weight.reshape(1, 1, h, w)
    if b > 1 or c > 1:
        weight = weight.expand(b, c, h, w)
    return weight


@LOSS_REGISTRY.register()
class FFTFrequencyLoss(nn.Module):
    """L1 loss on magnitude spectra with optional high-frequency emphasis.

    Args:
        loss_weight (float): Scale applied to the final loss value. Default: 1.0.
        reduction (str): Reduction to apply. One of {'none', 'mean', 'sum'}. Default: 'mean'.
        norm (str): Normalisation mode for FFT. Default: 'ortho'.
        use_log_amplitude (bool): Whether to compute the loss on log-magnitude spectra. Default: False.
        alpha (float): Frequency weighting exponent. Values > 0 emphasise high frequencies,
            values < 0 emphasise low frequencies. Default: 1.0.
        normalize_weight (bool): Normalise the frequency weighting map to unit mean. Default: True.
        eps (float): Numerical stability epsilon. Default: 1e-8.
    """

    def __init__(
        self,
        *,
        loss_weight: float = 1.0,
        reduction: str = "mean",
        norm: str = "ortho",
        use_log_amplitude: bool = False,
        alpha: float = 1.0,
        normalize_weight: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if reduction not in _REDUCTION_MODES:
            raise ValueError(f"Unsupported reduction mode '{reduction}'. Expected one of {_REDUCTION_MODES}.")

        if not isinstance(norm, str):
            raise TypeError(f"FFT normalization mode must be a string; received {type(norm)}.")

        self.loss_weight = float(loss_weight)
        self.reduction = reduction
        self.norm = norm
        self.use_log_amplitude = bool(use_log_amplitude)
        self.alpha = float(alpha)
        self.normalize_weight = bool(normalize_weight)
        self.eps = float(eps)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        loss = _fft_spectrum_l1_loss(
            pred,
            target,
            weight=weight,
            reduction=self.reduction,
            norm=self.norm,
            eps=self.eps,
            use_log_amplitude=self.use_log_amplitude,
            alpha=self.alpha,
            normalize_weight=self.normalize_weight,
        )
        return loss * self.loss_weight
