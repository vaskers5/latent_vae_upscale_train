"""Utilities for generating and logging sample reconstructions to Weights & Biases."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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

from .config import SampleVaeConfig, TrainingConfig
from .wandb_logger import WandbLogger

__all__ = ["SampleLogger"]


_DEFAULT_SAMPLE_VAE_SOURCES: Dict[str, Dict[str, Any]] = {
    "flux_vae": {
        "hf_repo": "wolfgangblack/flux_vae",
        "vae_kind": "kl",
        "weights_dtype": "float32",
    },
    "sd3_vae_anime_ft": {
        "hf_repo": "Disty0/sd3_vae_anime_ft",
        "vae_kind": "kl",
        "weights_dtype": "float32",
    },
}


def _to_pil_uint8(tensor: torch.Tensor) -> Image.Image:
    """Convert a tensor in [-1, 1] to a uint8 PIL image."""

    image = tensor.detach().cpu().clamp(-1.0, 1.0)
    image = (image + 1.0) * 0.5  # [-1, 1] -> [0, 1]
    image = image.mul(255.0).clamp(0.0, 255.0).byte()
    if image.dim() == 4:
        image = image.squeeze(0)
    return Image.fromarray(image.permute(1, 2, 0).numpy())


def _concat_side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    """Concatenate two images with matching heights horizontally."""

    if left.mode != right.mode:
        right = right.convert(left.mode)
    if left.height != right.height:
        raise ValueError("Images must share the same height to concatenate.")
    combined = Image.new(left.mode, (left.width + right.width, left.height))
    combined.paste(left, (0, 0))
    combined.paste(right, (left.width, 0))
    return combined


def _is_video_vae(module: torch.nn.Module) -> bool:
    encoder = getattr(module, "encoder", None)
    conv_in = getattr(encoder, "conv_in", None) if encoder is not None else None
    weight = getattr(conv_in, "weight", None) if conv_in is not None else None
    return isinstance(weight, torch.nn.Parameter) and weight.ndimension() == 5


@dataclass
class SampleGroup:
    name: str
    slug: str
    low_latents: torch.Tensor
    high_latents: torch.Tensor


@dataclass
class GroupLogResult:
    name: str
    slug: str
    real_cpu: torch.Tensor
    decoded_cpu: torch.Tensor
    pair_paths: List[Path]
    lpips_scores: List[float]
    primary_pair_paths: Optional[List[Path]] = None


class SampleLogger:
    """Generate qualitative samples using a VAE decoder and log them to wandb."""

    def __init__(
        self,
        cfg: TrainingConfig,
        *,
        dataset: Optional[Sequence] = None,
        datasets: Optional[Mapping[str, Sequence]] = None,
        max_samples: int = 4,
        wandb_logger: Optional[WandbLogger] = None,
    ) -> None:
        self.cfg = cfg
        self._wandb = wandb_logger
        self.generated_folder = Path(cfg.paths.samples_dir)
        self.generated_folder.mkdir(parents=True, exist_ok=True)
        self.sample_interval = max(1, cfg.logging.global_sample_interval)

        dataset_items: List[Tuple[str, Sequence]] = []
        seen: set[str] = set()

        if datasets:
            for name, ds in datasets.items():
                str_name = str(name)
                if str_name in seen:
                    continue
                dataset_items.append((str_name, ds))
                seen.add(str_name)

        if dataset is not None:
            default_name = self._default_group_name()
            if default_name not in seen:
                dataset_items.insert(0, (default_name, dataset))
                seen.add(default_name)

        if not dataset_items:
            raise ValueError("SampleLogger requires at least one dataset to visualise samples.")

        lengths = [len(ds) for _name, ds in dataset_items]
        if not lengths or any(length <= 0 for length in lengths):
            raise ValueError("All datasets used for sample logging must contain at least one sample.")

        base_count = min(lengths)
        self.sample_count = max(1, min(max_samples, base_count))

        self._configured_sample_vaes: Dict[str, SampleVaeConfig] = {}
        for name, spec in self.cfg.model.sample_vaes.items():
            self._register_sample_spec(self._configured_sample_vaes, name, spec)

        self._auto_sample_vaes: Dict[str, SampleVaeConfig] = {}

        self.groups: List[SampleGroup] = []
        for name, ds in dataset_items:
            low_stack, high_stack = self._collect_fixed_latents(ds, self.sample_count)
            slug = self._slugify_name(name)
            self.groups.append(
                SampleGroup(
                    name=name,
                    slug=slug,
                    low_latents=low_stack,
                    high_latents=high_stack,
                )
            )
            auto_spec = self._default_sample_spec(name)
            if auto_spec is not None:
                self._register_sample_spec(self._auto_sample_vaes, name, auto_spec)

        self._vae_cache: Dict[str, torch.nn.Module] = {}
        self._vae_video_flags: Dict[str, bool] = {}
        self._lpips: Optional[lpips.LPIPS] = None

    # --------------------------------------------------------------------- helpers
    def _default_group_name(self) -> str:
        names = list(getattr(self.cfg.embeddings, "vae_names", ()))
        if names:
            return str(names[0])
        return "vae"

    @staticmethod
    def _slugify_name(name: str) -> str:
        slug = name.replace("\\", "_").replace("/", "_").replace(" ", "_")
        return slug or "vae"

    def _register_sample_spec(
        self,
        registry: Dict[str, SampleVaeConfig],
        name: str,
        spec: SampleVaeConfig,
    ) -> None:
        variants = {
            name,
            name.lower(),
            self._slugify_name(name),
            self._slugify_name(name).lower(),
        }
        for key in variants:
            registry[key] = spec

    def _default_sample_spec(self, name: str) -> Optional[SampleVaeConfig]:
        key = name.strip().lower()
        mapping = _DEFAULT_SAMPLE_VAE_SOURCES.get(key)
        if mapping is None:
            slug = self._slugify_name(key)
            mapping = _DEFAULT_SAMPLE_VAE_SOURCES.get(slug)
        if mapping is None:
            return None
        return SampleVaeConfig.from_mapping(mapping)

    def _resolve_sample_spec(self, name: str) -> Optional[SampleVaeConfig]:
        variants = [
            name,
            name.lower(),
            self._slugify_name(name),
            self._slugify_name(name).lower(),
        ]
        for key in variants:
            spec = self._configured_sample_vaes.get(key)
            if spec is not None:
                return spec
        for key in variants:
            spec = self._auto_sample_vaes.get(key)
            if spec is not None:
                return spec
        return None

    def _collect_fixed_latents(self, dataset: Sequence, count: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lows: List[torch.Tensor] = []
        highs: List[torch.Tensor] = []

        for index in range(count):
            sample = dataset[index % len(dataset)]
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

    def _build_vae(self, group_name: str) -> Tuple[torch.nn.Module, bool]:
        model_cfg = self.cfg.model
        spec = self._resolve_sample_spec(group_name)

        load_from_value = spec.load_from if spec and spec.load_from else model_cfg.load_from
        load_path = Path(load_from_value).expanduser() if load_from_value else None
        path_exists = load_path is not None and load_path.exists()

        hf_source = spec.hf_repo if spec and spec.hf_repo else model_cfg.hf_repo
        if not path_exists and load_from_value and not hf_source:
            hf_source = load_from_value

        hf_subfolder = spec.hf_subfolder if spec and spec.hf_subfolder else model_cfg.hf_subfolder
        hf_revision = spec.hf_revision if spec and spec.hf_revision else model_cfg.hf_revision
        hf_auth_token = spec.hf_auth_token if spec and spec.hf_auth_token else model_cfg.hf_auth_token
        kind = (spec.vae_kind if spec and spec.vae_kind else model_cfg.vae_kind or "").strip().lower()
        weights_dtype = spec.weights_dtype if spec and spec.weights_dtype else model_cfg.weights_dtype

        if kind == "qwen":
            if path_exists:
                source = str(load_path)
                kwargs: Dict[str, Any] = {}
            else:
                source = hf_source or "Qwen/Qwen-Image"
                kwargs = {}
                if hf_subfolder or not hf_source:
                    kwargs["subfolder"] = hf_subfolder or "vae"
                if hf_revision:
                    kwargs["revision"] = hf_revision
                if hf_auth_token:
                    kwargs["use_auth_token"] = hf_auth_token
            vae: torch.nn.Module = AutoencoderKLQwenImage.from_pretrained(source, **kwargs)
        else:
            if path_exists:
                source = str(load_path)
                kwargs = {}
            else:
                source = hf_source
                if not source:
                    raise RuntimeError(
                        f"Sample logging requires a VAE source for '{group_name}' (either 'load_from' or 'hf_repo')."
                    )
                kwargs = {}
                if hf_subfolder:
                    kwargs["subfolder"] = hf_subfolder
                if hf_revision:
                    kwargs["revision"] = hf_revision
                if hf_auth_token:
                    kwargs["use_auth_token"] = hf_auth_token

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

        vae = vae.to(dtype=weights_dtype).eval()
        is_video = _is_video_vae(vae)
        return vae, is_video

    def _ensure_vae(self, group_name: str, device: torch.device) -> Tuple[torch.nn.Module, bool]:
        if group_name not in self._vae_cache:
            vae, is_video = self._build_vae(group_name)
            self._vae_cache[group_name] = vae
            self._vae_video_flags[group_name] = is_video
        vae = self._vae_cache[group_name].to(device)
        return vae, self._vae_video_flags[group_name]

    def _ensure_lpips(self, device: torch.device) -> lpips.LPIPS:
        if self._lpips is None:
            self._lpips = lpips.LPIPS(net=self.cfg.losses.lpips_backbone or "vgg").eval()
        self._lpips = self._lpips.to(device)
        return self._lpips

    def _offload_models(self) -> None:
        for name, vae in self._vae_cache.items():
            self._vae_cache[name] = vae.to("cpu")
        if self._lpips is not None:
            self._lpips = self._lpips.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _decode_latents(
        self,
        group_name: str,
        vae: torch.nn.Module,
        latents: torch.Tensor,
        *,
        treat_as_video: bool,
    ) -> torch.Tensor:
        decoder_dtype = next(vae.parameters()).dtype
        inputs = latents.to(dtype=decoder_dtype)
        if treat_as_video and inputs.dim() == 4:
            inputs = inputs.unsqueeze(2)
        try:
            decoded = vae.decode(inputs).sample
        except RuntimeError as exc:
            message = str(exc)
            if "expected input" in message.lower() and "to have" in message.lower():
                raise RuntimeError(
                    f"VAE decode failed for group '{group_name}'. The cached latents have a channel configuration "
                    "incompatible with the loaded decoder. Configure 'model.sample_vaes' (or extend the built-in "
                    "defaults) so that the correct VAE weights are used. Original error: {message}"
                ) from exc
            raise
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

        if not self.groups:
            return

        lpips_model = self._ensure_lpips(device)
        model_prev_mode = model.training
        model.eval()

        results: List[GroupLogResult] = []
        try:
            with torch.no_grad():
                model_dtype = next(model.parameters()).dtype

                for idx, group in enumerate(self.groups):
                    low_latents = group.low_latents.to(device=device, dtype=model_dtype)
                    high_latents = group.high_latents.to(device=device, dtype=model_dtype)
                    predicted_latents = model(low_latents)

                    vae, is_video = self._ensure_vae(group.name, device)
                    real_images = self._decode_latents(group.name, vae, high_latents, treat_as_video=is_video)
                    decoded_images = self._decode_latents(group.name, vae, predicted_latents, treat_as_video=is_video)

                    if decoded_images.shape[-2:] != real_images.shape[-2:]:
                        decoded_images = F.interpolate(
                            decoded_images,
                            size=real_images.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )

                    lpips_scores: List[float] = []
                    real_lpips = real_images.to(device=device, dtype=torch.float32)
                    rec_lpips = decoded_images.to(device=device, dtype=torch.float32)
                    for sample_idx in range(rec_lpips.shape[0]):
                        score = (
                            lpips_model(
                                rec_lpips[sample_idx : sample_idx + 1],
                                real_lpips[sample_idx : sample_idx + 1],
                            )
                            .mean()
                            .item()
                        )
                        lpips_scores.append(score)

                    real_cpu = real_images.detach().cpu()
                    decoded_cpu = decoded_images.detach().cpu()

                    group_dir = self.generated_folder / group.slug
                    group_dir.mkdir(parents=True, exist_ok=True)

                    pair_paths: List[Path] = []
                    primary_pair_paths: List[Path] = []

                    for sample_idx in range(decoded_cpu.shape[0]):
                        real_image = _to_pil_uint8(real_cpu[sample_idx])
                        decoded_image = _to_pil_uint8(decoded_cpu[sample_idx])
                        pair_image = _concat_side_by_side(real_image, decoded_image)

                        pair_path = group_dir / f"pair_{sample_idx}.jpg"
                        pair_image.save(pair_path, quality=95)

                        pair_paths.append(pair_path)

                        if idx == 0:
                            base_pair = self.generated_folder / f"pair_{sample_idx}.jpg"
                            pair_image.save(base_pair, quality=95)
                            primary_pair_paths.append(base_pair)

                    if idx == 0:
                        first_real = _to_pil_uint8(real_cpu[0])
                        first_decoded = _to_pil_uint8(decoded_cpu[0])
                        _concat_side_by_side(first_real, first_decoded).save(
                            self.generated_folder / "sample_pair.jpg",
                            quality=95,
                        )

                    results.append(
                        GroupLogResult(
                            name=group.name,
                            slug=group.slug,
                            real_cpu=real_cpu,
                            decoded_cpu=decoded_cpu,
                            pair_paths=pair_paths,
                            lpips_scores=lpips_scores,
                            primary_pair_paths=primary_pair_paths if idx == 0 else None,
                        )
                    )
        finally:
            model.train(model_prev_mode)
            self._offload_models()

        if not results:
            return

        rows = len(results)
        cols = self.sample_count
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 2.5))

        def _ensure_axes_array(ax):
            if rows == 1 and cols == 1:
                return np.array([[ax]])
            if rows == 1:
                return np.array([ax])
            if cols == 1:
                return np.array([[item] for item in ax])
            return ax

        axes = _ensure_axes_array(axes)

        for row_idx, result in enumerate(results):
            for sample_idx in range(self.sample_count):
                pair_axis = axes[row_idx, sample_idx]
                pair_image = _concat_side_by_side(
                    _to_pil_uint8(result.real_cpu[sample_idx]),
                    _to_pil_uint8(result.decoded_cpu[sample_idx]),
                )

                pair_axis.imshow(pair_image)
                pair_axis.axis("off")

                if row_idx == 0:
                    pair_axis.set_title(f"Real vs Decoded {sample_idx}")

            axes[row_idx, 0].set_ylabel(result.name, rotation=90, fontsize=10, labelpad=8)

        plt.tight_layout()
        pairs_path = self.generated_folder / "samples_pairs.jpg"
        fig.savefig(pairs_path, dpi=150)
        plt.close(fig)

        overall_scores = [score for result in results for score in result.lpips_scores]
        overall_avg = float(np.mean(overall_scores)) if overall_scores else 0.0

        log_payload: Dict[str, Any] = {
            "samples/lpips_mean": overall_avg,
            "samples/pairs": wandb.Image(
                str(pairs_path),
                caption=f"{rows} VAEs × {self.sample_count} samples (real + decoded)",
            ),
        }

        for result in results:
            if result.lpips_scores:
                log_payload[f"samples/lpips_mean/{result.slug}"] = float(np.mean(result.lpips_scores))
            for index, pair_path in enumerate(result.pair_paths):
                log_payload[f"samples/{result.slug}/pair_{index}"] = wandb.Image(str(pair_path))
            if result.primary_pair_paths:
                for index, pair_path in enumerate(result.primary_pair_paths):
                    log_payload[f"samples/pair_{index}"] = wandb.Image(str(pair_path))

        if self._wandb is not None and self._wandb.is_active:
            self._wandb.log(log_payload, step=step)
        plt.close(fig)

        overall_scores = [score for result in results for score in result.lpips_scores]
        overall_avg = float(np.mean(overall_scores)) if overall_scores else 0.0

        log_payload: Dict[str, Any] = {
            "samples/lpips_mean": overall_avg,
            "samples/pairs": wandb.Image(
                str(pairs_path),
                caption=f"{rows} VAEs × {self.sample_count} samples (real + decoded)",
            ),
        }

        for result in results:
            if result.lpips_scores:
                log_payload[f"samples/lpips_mean/{result.slug}"] = float(np.mean(result.lpips_scores))
            for index, pair_path in enumerate(result.pair_paths):
                log_payload[f"samples/{result.slug}/pair_{index}"] = wandb.Image(str(pair_path))
            if result.primary_pair_paths:
                for index, pair_path in enumerate(result.primary_pair_paths):
                    log_payload[f"samples/pair_{index}"] = wandb.Image(str(pair_path))

        if self._wandb is not None and self._wandb.is_active:
            self._wandb.log(log_payload, step=step)
