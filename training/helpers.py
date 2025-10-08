# helpers.py
# ------------------------------------------------------------
# Latent upscalers: Swin-lite, NAFNet-like, optional LIIF head,
# and a tiny frequency compensation branch.
# All operate on latent feature maps of shape (B, C, H, W)
# and output 2x upscaled latents (B, C, 2H, 2W) unless noted.
# ------------------------------------------------------------
from __future__ import annotations
from typing import Optional, Tuple, Literal, Dict, Any

import torch
from torch import nn
import torch.nn.functional as F

from .models import SwinIR


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return bool(value)


# ---------------------------
# Small utility components
# ---------------------------
class PixelShuffleUpsample(nn.Module):
    """Simple 2× upsample head that preserves channel count."""
    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.refine = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.shuffle(x)
        x = self.refine(x)
        return x


# ---------------------------
# NAFNet-style blocks
# ---------------------------
class NAFBlock(nn.Module):
    """
    Minimal NAF block: DWConv -> channel-mix (gated 1x1) -> residual.
    No explicit nonlinearity in the main path; gating uses sigmoid.
    """
    def __init__(self, c: int):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, 1, 1, groups=c)
        self.pw1 = nn.Conv2d(c, c * 2, 1)  # channel expansion and gating
        self.pw2 = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        a, b = self.pw1(y).chunk(2, dim=1)
        y = a * torch.sigmoid(b)
        y = self.pw2(y)
        return x + self.beta * y


class LatentNAFUpscaler(nn.Module):
    """
    Fast, strong baseline for latent SR.
    Returns (B, C, 2H, 2W).
    """
    def __init__(self, channels: int, blocks: int = 24, groups: int = 6):
        super().__init__()
        assert blocks % groups == 0, "blocks should be divisible by groups"
        per_group = blocks // groups

        self.head = nn.Conv2d(channels, channels, 3, 1, 1)
        self.groups = nn.ModuleList([
            nn.Sequential(*[NAFBlock(channels) for _ in range(per_group)])
            for _ in range(groups)
        ])
        self.body_skip_scale = nn.Parameter(torch.ones(len(self.groups)))
        self.tail = PixelShuffleUpsample(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.head(x)
        skip = y
        for i, g in enumerate(self.groups):
            y = y + self.body_skip_scale[i] * g(y)
        y = y + skip
        return self.tail(y)


# ---------------------------
# Swin-lite transformer blocks
# ---------------------------
class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 2.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, HW, C)
        return self.fc2(F.silu(self.fc1(x)))


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Right/bottom pad input so both spatial dims become divisible by `multiple`.
    Returns padded tensor and the (height, width) padding that was applied.
    """
    H, W = x.shape[-2:]
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, (pad_h, pad_w)


class WindowSelfAttention(nn.Module):
    """
    Lightweight windowed self-attention via MultiheadAttention over flattened windows.
    For simplicity, we tile the feature map with non-overlapping windows.
    """
    def __init__(self, dim: int, heads: int = 4, window: int = 8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.window = window
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        if H % self.window != 0 or W % self.window != 0:
            raise ValueError("LatentSwinUpscaler expects inputs divisible by the window size after padding.")

        nH = H // self.window
        nW = W // self.window

        # Rearrange into non-overlapping windows: (B * nH * nW, window^2, C)
        windows = (
            x.view(B, C, nH, self.window, nW, self.window)
            .permute(0, 2, 4, 3, 5, 1)
            .contiguous()
            .view(B * nH * nW, self.window * self.window, C)
        )

        normed = self.norm(windows)
        attn_out = self.attn(normed, normed, normed, need_weights=False)[0]
        windows = windows + attn_out  # residual inside the window

        # Restore original spatial layout
        y = (
            windows.view(B, nH, nW, self.window, self.window, C)
            .permute(0, 5, 1, 3, 2, 4)
            .contiguous()
            .view(B, C, H, W)
        )
        return y


class SwinBlockLite(nn.Module):
    def __init__(self, channels: int, heads: int = 4, window: int = 8, mlp_ratio: float = 2.0):
        super().__init__()
        self.attn = WindowSelfAttention(channels, heads=heads, window=window)
        self.norm = nn.LayerNorm(channels)
        self.mlp = MLP(channels, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BCHW -> BHWC for LayerNorm/MLP convenience
        y = self.attn(x)
        y = y.permute(0, 2, 3, 1).contiguous()  # BHWC
        y = y + self.mlp(self.norm(y))
        return y.permute(0, 3, 1, 2).contiguous()


class LatentSwinUpscaler(nn.Module):
    """
    Quality-first latent SR with lightweight windowed attention.
    Returns (B, C, 2H, 2W).
    """
    def __init__(self, channels: int, depth: int = 4, heads: int = 4, window: int = 8, mlp_ratio: float = 2.0):
        super().__init__()
        self.pad_window = window
        self.inp = nn.Conv2d(channels, channels, 3, 1, 1)
        self.blocks = nn.ModuleList([
            SwinBlockLite(channels, heads=heads, window=window, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.up = PixelShuffleUpsample(channels)

    def _pad_to_multiple(self, x: torch.Tensor, m: int) -> Tuple[torch.Tensor, Tuple[int,int]]:
        return _pad_to_multiple(x, m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = x.shape[-2:]
        y, (ph, pw) = self._pad_to_multiple(x, self.pad_window)
        y = self.inp(y)
        for blk in self.blocks:
            y = blk(y)
        y = self.up(y)
        # remove padding *after* upsample (scale=2)
        if ph or pw:
            target_h = orig_h * 2
            target_w = orig_w * 2
            y = y[..., :target_h, :target_w]
        return y


# ---------------------------
# Optional LIIF head (2× here)
# ---------------------------
class LIIFHead(nn.Module):
    """
    A minimal LIIF-style implicit decoder.
    For simplicity we implement fixed 2× sampling (on-grid) here,
    but the projection uses coordinate-aware MLP to encourage crisp details.
    """
    def __init__(self, feat_ch: int, hidden: int = 2 * 16):  # small hidden
        super().__init__()
        self.fc1 = nn.Linear(feat_ch + 2, hidden)  # +2 for relative coords
        self.fc2 = nn.Linear(hidden, feat_ch)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, C, H, W)
        B, C, H, W = feats.shape
        # build 2× target grid and local relative coords in [-1, 1]
        # target coords (2H, 2W)
        yy = torch.linspace(-1, 1, steps=2*H, device=feats.device)
        xx = torch.linspace(-1, 1, steps=2*W, device=feats.device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')  # (2H, 2W)
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 2H, 2W, 2)

        # sample features with bilinear (acts as local neighborhood fetch)
        up_feats = F.interpolate(feats, scale_factor=2, mode='bilinear', align_corners=False)  # (B,C,2H,2W)
        up_feats_perm = up_feats.permute(0, 2, 3, 1).contiguous()  # (B,2H,2W,C)

        inp = torch.cat([up_feats_perm, grid], dim=-1)  # (B,2H,2W,C+2)
        y = F.silu(self.fc1(inp))
        y = self.fc2(y)  # (B,2H,2W,C)
        return y.permute(0, 3, 1, 2).contiguous()  # (B,C,2H,2W)


class LatentSwinLIIFUpscaler(nn.Module):
    """
    Swin-lite backbone + LIIF implicit head (2× here).
    """
    def __init__(self, channels: int, depth: int = 4, heads: int = 4, window: int = 8, mlp_ratio: float = 2.0, liif_hidden: int = 32):
        super().__init__()
        self.window = window
        # Use backbone without the final pixel shuffle: tap features before tail.
        # To keep code simple, we re-create a copy of blocks and head here,
        # and use a LIIF head for the upscale, bypassing PixelShuffle.
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            *[SwinBlockLite(channels, heads=heads, window=window, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.liif = LIIFHead(channels, hidden=liif_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Perform the same padding logic as the Swin backbone before running LIIF
        orig_h, orig_w = x.shape[-2:]
        feats, (ph, pw) = _pad_to_multiple(x, self.window)
        feats = self.encoder(feats)
        y = self.liif(feats)
        if ph or pw:
            y = y[..., : (orig_h * 2), : (orig_w * 2)]
        return y


# ---------------------------
# Tiny frequency compensation
# ---------------------------
class FrequencyComp(nn.Module):
    """
    Tiny frequency-domain residual: predicts a correction in Fourier domain
    from mid-level features (spatial), adds it back after iFFT.
    """
    def __init__(self, channels: int, hidden: int = 32):
        super().__init__()
        self.spatial_to_freq = nn.Conv2d(channels, 2 * hidden, 1)  # real/imag features
        self.mix = nn.Conv2d(2 * hidden, 2 * channels, 1)         # to real/imag residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        X = torch.fft.rfft2(x, norm="ortho")  # complex (B,C,H, W//2+1)
        # pack real/imag to channels for convs
        X_ri = torch.stack([X.real, X.imag], dim=2).reshape(B, 2*C, X.shape[-2], X.shape[-1])  # (B,2C,H,Wc)
        R = self.mix(F.silu(self.spatial_to_freq(x)))  # (B,2C,H,W)
        # interpolate R to X's last-dim size (Wc = W//2+1)
        if R.shape[-1] != X_ri.shape[-1]:
            R = F.interpolate(R, size=(X_ri.shape[-2], X_ri.shape[-1]), mode='bilinear', align_corners=False)
        # add residual in RI space then convert back to complex
        X_corr_ri = X_ri + R
        X_corr = torch.complex(X_corr_ri[:, 0:C], X_corr_ri[:, C:2*C])
        x_rec = torch.fft.irfft2(X_corr, s=(H, W), norm="ortho")
        return x + x_rec - x  # just return corrected component (x_rec) to be added by caller


class LatentNAFUpscalerFreq(nn.Module):
    """
    NAF backbone + frequency compensation before pixel-shuffle tail.
    """
    def __init__(self, channels: int, blocks: int = 20):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            *[NAFBlock(channels) for _ in range(blocks)]
        )
        self.freq = FrequencyComp(channels, hidden=32)
        self.tail = PixelShuffleUpsample(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.enc(x)
        y = self.freq(y) + y
        return self.tail(y)


# ---------------------------
# Model factory
# ---------------------------
def get_latent_upscaler(model_name: str, channels: int, **kwargs) -> nn.Module:
    """
    Build a latent upscaler by name.
    model_name in {
        'swin',                 # SwinIR (official implementation)
        'swinir',               # alias for SwinIR
        'swin_lite',            # LatentSwinUpscaler (lightweight)
        'naf',                  # LatentNAFUpscaler
        'swin_liif',            # LatentSwinLIIFUpscaler (2× here)
        'naf_freq',             # LatentNAFUpscalerFreq
    }
    """
    name = model_name.lower()
    if name in {'swin', 'swinir'}:
        window = int(kwargs.get('window', 8))
        patch_size = int(kwargs.get('patch_size', 1))
        mlp_ratio = float(kwargs.get('mlp_ratio', 4.0))
        embed_dim = int(kwargs.get('embed_dim', max(64, channels * 4)))

        stages = int(kwargs.get('stages', 4))
        depths_raw = kwargs.get('depths')
        if depths_raw is None:
            base_depth = int(kwargs.get('depth', 6))
            depth_values = tuple(base_depth for _ in range(max(1, stages)))
        elif isinstance(depths_raw, (list, tuple)):
            depth_values = tuple(int(d) for d in depths_raw)
            stages = len(depth_values)
        else:
            depth_values = tuple(int(depths_raw) for _ in range(max(1, stages)))

        heads_raw = kwargs.get('num_heads') or kwargs.get('heads')
        if heads_raw is None:
            base_heads = int(kwargs.get('heads', 6))
            num_heads = tuple(base_heads for _ in range(len(depth_values)))
        elif isinstance(heads_raw, (list, tuple)):
            num_heads = tuple(int(h) for h in heads_raw)
        else:
            num_heads = tuple(int(heads_raw) for _ in range(len(depth_values)))
        if len(num_heads) != len(depth_values):
            num_heads = tuple(num_heads[0] for _ in range(len(depth_values)))

        drop_rate = float(kwargs.get('drop_rate', 0.0))
        attn_drop_rate = float(kwargs.get('attn_drop_rate', 0.0))
        drop_path_rate = float(kwargs.get('drop_path_rate', 0.1))
        ape = _as_bool(kwargs.get('ape', False), default=False)
        patch_norm = _as_bool(kwargs.get('patch_norm', True), default=True)
        use_checkpoint = _as_bool(kwargs.get('use_checkpoint', False), default=False)
        resi_connection = str(kwargs.get('resi_connection', '1conv'))
        upsampler = str(kwargs.get('upsampler', 'pixelshuffle'))
        img_range = float(kwargs.get('img_range', 1.0))
        scale = int(kwargs.get('scale', 2))
        img_size = int(kwargs.get('img_size', window * patch_size))

        return SwinIR(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=channels,
            embed_dim=embed_dim,
            depths=depth_values,
            num_heads=num_heads,
            window_size=window,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            upscale=scale,
            img_range=img_range,
            upsampler=upsampler,
            resi_connection=resi_connection,
        )
    if name in {'swin_lite', 'swinlite'}:
        return LatentSwinUpscaler(channels=channels,
                                  depth=kwargs.get('depth', 4),
                                  heads=kwargs.get('heads', 4),
                                  window=kwargs.get('window', 8),
                                  mlp_ratio=kwargs.get('mlp_ratio', 2.0))
    if name == 'naf':
        return LatentNAFUpscaler(channels=channels,
                                 blocks=kwargs.get('blocks', 24),
                                 groups=kwargs.get('groups', 6))
    if name == 'swin_liif':
        return LatentSwinLIIFUpscaler(channels=channels,
                                      depth=kwargs.get('depth', 4),
                                      heads=kwargs.get('heads', 4),
                                      window=kwargs.get('window', 8),
                                      mlp_ratio=kwargs.get('mlp_ratio', 2.0),
                                      liif_hidden=kwargs.get('liif_hidden', 32))
    if name == 'naf_freq':
        return LatentNAFUpscalerFreq(channels=channels,
                                     blocks=kwargs.get('blocks', 20))
    raise ValueError(f"Unknown model_name '{model_name}'. Valid: swin | naf | swin_liif | naf_freq")
