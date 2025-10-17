from copy import deepcopy
import hashlib
import cv2
import inspect
import numpy as np
import os
import torch
import warnings
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from os import path as osp
from torch.nn import functional as F
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# --- Matplotlib Integration ---
# Use the 'Agg' backend for non-interactive environments (servers, etc.)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from basicsr.archs import build_network

# --- End Matplotlib Integration ---
from basicsr.utils import get_root_logger, imwrite
from basicsr.utils.registry import MODEL_REGISTRY
from .swinir_model import SwinIRModel

# Try to import VAE for decoding (optional)
try:
    from diffusers import (
    AsymmetricAutoencoderKL,
    AutoencoderKL,
    AutoencoderKLQwenImage,
    AutoencoderKLWan,
    )

    VAE_AVAILABLE = True
except ImportError:
    AsymmetricAutoencoderKL = None  # type: ignore[assignment]
    AutoencoderKLQwenImage = None  # type: ignore[assignment]
    AutoencoderKLWan = None  # type: ignore[assignment]
    VAE_AVAILABLE = False
    print("Warning: diffusers not available. VAE decoding will be skipped.")

# Try to import wandb
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Wandb logging will be skipped.")


_DEFAULT_VAE_SOURCES: Dict[str, Dict[str, Any]] = {
    "flux_vae": {
        "hf_repo": "wolfgangblack/flux_vae",
        "vae_kind": "kl",
        "latents_scaled": False,
    },
    "sdxl_vae": {
        "hf_repo": "stabilityai/sdxl-vae",
        "vae_kind": "kl",
        "latents_scaled": False,
    },
}


@dataclass(frozen=True)
class _VaeSpec:
    """Configuration required to instantiate a VAE."""

    name: str
    load_from: Optional[Path]
    hf_repo: Optional[str]
    hf_subfolder: Optional[str]
    hf_revision: Optional[str]
    hf_auth_token: Optional[str]
    vae_kind: str
    weights_dtype: Optional[torch.dtype]
    latents_scaled: bool

    @property
    def cache_key(self) -> str:
        return _slugify_name(self.name)


def _slugify_name(name: str) -> str:
    slug = name.replace("\\", "_").replace("/", "_").replace(" ", "_")
    return slug or "vae"


def _resolve_dtype(value: Any) -> Optional[torch.dtype]:
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        norm = value.strip().lower().replace(" ", "")
        if not norm:
            return None
        if not norm.startswith("torch."):
            norm = f"torch.{norm}"
        attr = norm.split(".")[-1]
        if not hasattr(torch, attr):
            raise ValueError(f"Unsupported torch dtype specification: {value}")
        dtype = getattr(torch, attr)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Resolved value is not a torch.dtype: {value}")
        return dtype
    raise TypeError(f"Expected dtype string or torch.dtype, received: {value!r}")


def _candidate_keys(name: str) -> Sequence[str]:
    slug = _slugify_name(name)
    return (name, name.lower(), slug, slug.lower())


def _build_spec_from_mapping(name: str, mapping: Any) -> _VaeSpec:
    if isinstance(mapping, _VaeSpec):
        return mapping
    if mapping is None:
        mapping = {}
    if isinstance(mapping, str):
        mapping = {"hf_repo": mapping}
    if not isinstance(mapping, Mapping):
        raise TypeError(
            f"VAE specification for '{name}' must be provided as a mapping or string; received {type(mapping)}."
        )

    load_from_value = mapping.get("load_from")
    load_from = Path(str(load_from_value)).expanduser() if load_from_value else None
    hf_repo = mapping.get("hf_repo")
    hf_repo = str(hf_repo) if hf_repo else None
    hf_subfolder = mapping.get("hf_subfolder")
    hf_subfolder = str(hf_subfolder) if hf_subfolder else None
    hf_revision = mapping.get("hf_revision")
    hf_revision = str(hf_revision) if hf_revision else None
    hf_auth_token = mapping.get("hf_auth_token")
    hf_auth_token = str(hf_auth_token) if hf_auth_token else None
    vae_kind = str(mapping.get("vae_kind", "kl")).strip().lower() or "kl"
    weights_dtype = _resolve_dtype(mapping.get("weights_dtype"))

    raw_scaled = mapping.get("latents_scaled")
    if raw_scaled is None:
        raw_scaled = mapping.get("latents_are_scaled")
    if isinstance(raw_scaled, str):
        norm = raw_scaled.strip().lower()
        latents_scaled = norm in {"1", "true", "yes", "y", "on"}
    elif raw_scaled is None:
        latents_scaled = False
    else:
        latents_scaled = bool(raw_scaled)

    return _VaeSpec(
        name=name,
        load_from=load_from,
        hf_repo=hf_repo,
        hf_subfolder=hf_subfolder,
        hf_revision=hf_revision,
        hf_auth_token=hf_auth_token,
        vae_kind=vae_kind,
        weights_dtype=weights_dtype,
        latents_scaled=latents_scaled,
    )



@MODEL_REGISTRY.register()
class SwinIRLatentModel(SwinIRModel):
    """SwinIR model for latent space super resolution with custom visualization.
    This version uses Matplotlib to replicate the visualization style from the
    provided reference script.
    """

    def __init__(self, opt):
        # Initialize loss_configs before calling super().__init__()
        self.loss_configs = {}
        self._vae_specs_primary: Dict[str, _VaeSpec] = {}
        self._vae_aliases: Dict[str, _VaeSpec] = {}
        self._ordered_vae_keys: List[str] = []
        self._vae_cache: Dict[str, torch.nn.Module] = {}
        self._current_vae_names: Optional[List[str]] = None
        self._default_train_vae_name: Optional[str] = None
        self._default_val_vae_name: Optional[str] = None
        self._val_decode_mem_cache: Dict[str, torch.Tensor] = {}
        self._val_cache_dir: Optional[str] = None
        super().__init__(opt)
        if VAE_AVAILABLE:
            self._setup_vae_registry()
        else:
            logger = get_root_logger()
            logger.warning(
                "diffusers is not available; pixel-space decoding for validation will be skipped."
            )

    # ------------------------------------------------------------------ VAE helpers
    def _setup_vae_registry(self) -> None:
        """Populate VAE specifications from defaults and user configuration."""

        self._vae_specs_primary.clear()
        self._vae_aliases.clear()
        self._ordered_vae_keys.clear()
        self._vae_cache.clear()

        raw_sources = self.opt.get("vae_sources") or {}
        if raw_sources and not isinstance(raw_sources, Mapping):
            raise TypeError(
                "Option 'vae_sources' must be a mapping from names to configuration dictionaries."
            )

        logger = get_root_logger()

        for name, mapping in _DEFAULT_VAE_SOURCES.items():
            try:
                spec = _build_spec_from_mapping(name, mapping)
            except Exception as exc:
                logger.warning(f"Skipping default VAE '{name}': {exc}")
                continue
            self._register_vae_spec(spec, allow_override=False)

        if raw_sources:
            for name, mapping in raw_sources.items():
                try:
                    spec = _build_spec_from_mapping(name, mapping)
                except Exception as exc:
                    logger.warning(f"Skipping configured VAE '{name}': {exc}")
                    continue
                self._register_vae_spec(spec, allow_override=True)

        self._default_train_vae_name = self._resolve_dataset_default_name("train")
        self._default_val_vae_name = self._resolve_dataset_default_name("val")

        if self._ordered_vae_keys:
            resolved = [self._vae_specs_primary[key].name for key in self._ordered_vae_keys]
            logger.info("Configured VAE decoders: %s", ", ".join(resolved))
        else:
            logger.warning("No VAE configurations found. Pixel-space decoding will be skipped.")

    def _register_vae_spec(self, spec: _VaeSpec, *, allow_override: bool) -> None:
        primary_key = spec.cache_key
        if allow_override or primary_key not in self._vae_specs_primary:
            self._vae_specs_primary[primary_key] = spec
            if primary_key not in self._ordered_vae_keys:
                self._ordered_vae_keys.append(primary_key)

        def _register_alias(key: str) -> None:
            if allow_override or key not in self._vae_aliases:
                self._vae_aliases[key] = spec

        for key in _candidate_keys(spec.name):
            _register_alias(key)
        if spec.hf_repo:
            for key in _candidate_keys(spec.hf_repo):
                _register_alias(key)
        if spec.load_from:
            for key in _candidate_keys(str(spec.load_from)):
                _register_alias(key)

    def _resolve_dataset_default_name(self, phase: str) -> Optional[str]:
        datasets_cfg = self.opt.get("datasets") or {}
        if not isinstance(datasets_cfg, Mapping):
            return None

        for key, cfg in datasets_cfg.items():
            if key.split("_")[0] != phase:
                continue
            names = cfg.get("vae_names")
            if isinstance(names, str) and names.strip():
                return names.strip()
            if isinstance(names, Sequence) and not isinstance(names, str):
                for entry in names:
                    if isinstance(entry, str) and entry.strip():
                        return entry.strip()
        return None

    def _resolve_vae_spec(self, name: Optional[str]) -> Optional[_VaeSpec]:
        if not name:
            return None
        key = str(name).strip()
        if not key:
            return None

        spec = self._vae_aliases.get(key)
        if spec is None:
            slug = _slugify_name(key)
            spec = self._vae_aliases.get(slug)

        if spec is not None:
            return spec

        # Attempt to auto-configure unknown names.
        candidate_path = Path(key).expanduser()
        mapping: Dict[str, Any]
        if candidate_path.exists():
            mapping = {"load_from": str(candidate_path)}
        else:
            mapping = {"hf_repo": key}

        try:
            spec = _build_spec_from_mapping(key, mapping)
        except Exception:
            return None

        self._register_vae_spec(spec, allow_override=False)
        logger = get_root_logger()
        origin = "local path" if candidate_path.exists() else "Hugging Face repo"
        logger.info("Auto-configured VAE '%s' from %s '%s'.", spec.name, origin, key)
        return spec

    def _ensure_vae(self, name: Optional[str]) -> Optional[Tuple[torch.nn.Module, _VaeSpec]]:
        if not VAE_AVAILABLE:
            return None

        spec = self._resolve_vae_spec(name)
        if spec is None:
            fallback_names = [
                self._default_train_vae_name,
                self._default_val_vae_name,
            ]
            for fallback in fallback_names:
                spec = self._resolve_vae_spec(fallback)
                if spec is not None:
                    break
        if spec is None and self._ordered_vae_keys:
            spec = self._vae_specs_primary[self._ordered_vae_keys[0]]
        if spec is None:
            return None

        cache_key = spec.cache_key
        vae = self._vae_cache.get(cache_key)
        if vae is None:
            vae = self._instantiate_vae(spec)
            self._vae_cache[cache_key] = vae

        params = next(vae.parameters(), None)
        if params is not None and params.device != self.device:
            vae = vae.to(self.device)
            self._vae_cache[cache_key] = vae

        return vae, spec

    def _instantiate_vae(self, spec: _VaeSpec) -> torch.nn.Module:
        kwargs: Dict[str, Any] = {}
        source: Optional[str]

        if spec.load_from is not None and spec.load_from.exists():
            source = str(spec.load_from)
        else:
            source = spec.hf_repo
            if source is None:
                raise RuntimeError(
                    f"VAE '{spec.name}' must define 'hf_repo' or a valid 'load_from' path."
                )
            if spec.hf_subfolder:
                kwargs["subfolder"] = spec.hf_subfolder
            if spec.hf_revision:
                kwargs["revision"] = spec.hf_revision
            if spec.hf_auth_token:
                kwargs["use_auth_token"] = spec.hf_auth_token

        if spec.weights_dtype is not None:
            kwargs["torch_dtype"] = spec.weights_dtype

        kind = spec.vae_kind
        logger = get_root_logger()
        logger.info("Loading VAE '%s' (kind=%s) from %s", spec.name, kind, source)

        if kind == "qwen":
            if AutoencoderKLQwenImage is None:
                raise RuntimeError("AutoencoderKLQwenImage is not available in this diffusers version.")
            vae = AutoencoderKLQwenImage.from_pretrained(source, **kwargs)
        elif kind == "wan":
            if AutoencoderKLWan is None:
                raise RuntimeError("AutoencoderKLWan is not available in this diffusers version.")
            vae = AutoencoderKLWan.from_pretrained(source, **kwargs)
        elif kind in {"kl", "autoencoderkl", "autoencoder_kl"}:
            vae = AutoencoderKL.from_pretrained(source, **kwargs)
        elif kind in {"asymmetric_kl", "kl_asymmetric", "kl_asym", "asym_kl"}:
            if AsymmetricAutoencoderKL is None:
                raise RuntimeError("AsymmetricAutoencoderKL is not available in this diffusers version.")
            vae = AsymmetricAutoencoderKL.from_pretrained(source, **kwargs)
        else:
            try:
                vae = AutoencoderKL.from_pretrained(source, **kwargs)
                logger.warning(
                    "VAE kind '%s' is unrecognised. Loaded AutoencoderKL for '%s'.", kind, spec.name
                )
            except Exception:
                if AsymmetricAutoencoderKL is None:
                    raise
                vae = AsymmetricAutoencoderKL.from_pretrained(source, **kwargs)
                logger.warning(
                    "VAE kind '%s' is unrecognised. Loaded AsymmetricAutoencoderKL for '%s'.",
                    kind,
                    spec.name,
                )

        if spec.weights_dtype is not None:
            vae = vae.to(dtype=spec.weights_dtype)
        vae = vae.to(self.device).eval()
        return vae

    def _has_any_vae(self) -> bool:
        return VAE_AVAILABLE and bool(self._vae_specs_primary or self._vae_cache)

    def _prepare_batch_vae_names(
        self,
        provided: Optional[Sequence[str]],
        batch_size: int,
    ) -> List[Optional[str]]:
        def _expand(candidate) -> Optional[List[str]]:
            if candidate is None:
                return None
            if isinstance(candidate, str):
                return [candidate] * batch_size
            if isinstance(candidate, Sequence) and not isinstance(candidate, str):
                entries = [str(item) for item in candidate]
                if len(entries) == batch_size:
                    return entries
                if len(entries) == 1:
                    return entries * batch_size
            return None

        for candidate in (provided, self._current_vae_names):
            expanded = _expand(candidate)
            if expanded is not None:
                return expanded

        for default_name in (self._default_val_vae_name, self._default_train_vae_name):
            if default_name:
                return [default_name] * batch_size

        if self._ordered_vae_keys:
            spec = self._vae_specs_primary[self._ordered_vae_keys[0]]
            return [spec.name] * batch_size

        return [None] * batch_size

    # ------------------------------------------------------------------ overrides
    def feed_data(self, data):
        super().feed_data(data)
        names = data.get("vae_name")
        if names is None:
            self._current_vae_names = None
        elif isinstance(names, (list, tuple)):
            self._current_vae_names = [str(entry) for entry in names]
        else:
            self._current_vae_names = [str(names)]

        if VAE_AVAILABLE and self._current_vae_names:
            for entry in set(self._current_vae_names):
                self._resolve_vae_spec(entry)

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)
        
    def init_training_settings(self):
        """Override to handle space-aware loss initialization"""
        self.net_g.train()
        train_opt = self.opt["train"]
        self.ema_decay = train_opt.get("ema_decay", 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f"Use Exponential Moving Average with decay: {self.ema_decay}")
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt["network_g"]).to(self.device)
            # load pretrained model
            load_path = self.opt["path"].get("pretrain_network_g", None)
            if load_path is not None:
                self.load_network(
                    self.net_g_ema,
                    load_path,
                    self.opt["path"].get("strict_load_g", True),
                    "params_ema",
                )
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # Initialize space-aware losses
        self._init_space_aware_losses(train_opt)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def _init_space_aware_losses(self, train_opt):
        """Initialize losses with space parameter support"""
        from basicsr.losses import build_loss

        # Initialize loss criteria
        self.loss_criteria = {}

        # Process all loss configurations
        for loss_name, loss_opt in train_opt.items():
            if "_opt" in loss_name and isinstance(loss_opt, dict):
                loss_space = loss_opt.get("space", "latent")  # Default to latent space

                # Store the loss configuration
                self.loss_configs[loss_name] = {
                    "config": loss_opt,
                    "space": loss_space,
                    "weight": float(loss_opt.get("loss_weight", 1.0)),
                }

                # Build the loss criterion (without space parameter)
                loss_config_copy = loss_opt.copy()
                loss_config_copy.pop("space", None)  # Remove space parameter
                loss_config_copy.pop(
                    "loss_weight", None
                )  # Remove weight (handled separately)

                self.loss_criteria[loss_name] = build_loss(loss_config_copy).to(
                    self.device
                )

                logger = get_root_logger()
                logger.info(
                    f'Initialized {loss_name} in {loss_space} space with weight {loss_opt.get("loss_weight", 1.0)}'
                )

        if not self.loss_criteria:
            raise ValueError(
                "No losses configured. Please add at least one loss with _opt suffix."
            )

    @staticmethod
    def _evaluate_loss(criterion, prediction, target=None):
        """Call a loss criterion, handling single-input losses gracefully."""

        forward = getattr(criterion, "forward", None)
        if forward is None:
            return criterion(prediction, target) if target is not None else criterion(prediction)

        signature = inspect.signature(forward)
        params = list(signature.parameters.values())
        required = [
            param
            for param in params
            if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and param.default is inspect._empty
        ]
        if target is not None and (len(required) >= 2 or any(param.name == "target" for param in params)):
            return criterion(prediction, target)
        return criterion(prediction)

    def optimize_parameters(self, current_iter):
        """Override to handle space-aware loss calculation"""
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()

        # Calculate losses in their respective spaces
        for loss_name, loss_config in self.loss_configs.items():
            loss_space = loss_config["space"]
            loss_weight = loss_config["weight"]
            loss_criterion = self.loss_criteria[loss_name]

            if loss_space == "latent":
                # Calculate loss directly on latent tensors
                loss_value = self._evaluate_loss(loss_criterion, self.output, self.gt)

                # Handle losses that return multiple values (like perceptual loss)
                if isinstance(loss_value, (tuple, list)):
                    # For perceptual loss: (l_percep, l_style)
                    for i, val in enumerate(loss_value):
                        if val is not None:
                            weighted_val = val * loss_weight
                            l_total += weighted_val
                            loss_dict[f"{loss_name}_{i}"] = val
                else:
                    # Single loss value
                    loss_dict[loss_name] = loss_value
                    weighted_loss = loss_value * loss_weight
                    l_total += weighted_loss

            elif loss_space == "pixel":
                # Calculate loss in pixel space after VAE decoding
                decoded_pred = self.decode_latents(self.output, vae_names=self._current_vae_names)
                decoded_gt = self.decode_latents(self.gt, vae_names=self._current_vae_names)
                if decoded_pred is None or decoded_gt is None:
                    print(
                        f"Warning: Unable to decode latents, skipping pixel space loss: {loss_name}"
                    )
                    continue

                loss_value = self._evaluate_loss(loss_criterion, decoded_pred, decoded_gt)

                # Handle losses that return multiple values (like perceptual loss)
                if isinstance(loss_value, (tuple, list)):
                    # For perceptual loss: (l_percep, l_style)
                    for i, val in enumerate(loss_value):
                        if val is not None:
                            weighted_val = val * loss_weight
                            l_total += weighted_val
                            loss_dict[f"{loss_name}_{i}"] = val
                else:
                    # Single loss value
                    loss_dict[loss_name] = loss_value
                    weighted_loss = loss_value * loss_weight
                    l_total += weighted_loss

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def decode_latents(
        self,
        latents: Optional[torch.Tensor],
        *,
        vae_names: Optional[Sequence[str]] = None,
    ) -> Optional[torch.Tensor]:
        """Decode latents to RGB images using a configured VAE."""

        if latents is None or not VAE_AVAILABLE:
            return None

        if latents.dim() == 3:
            latents = latents.unsqueeze(0)
            squeeze_result = True
        else:
            squeeze_result = False

        if latents.dim() != 4:
            raise ValueError(
                f"Expected latents to have shape [B, C, H, W]; received tensor with shape {tuple(latents.shape)}."
            )

        batch_size = latents.shape[0]
        name_list = self._prepare_batch_vae_names(vae_names, batch_size)
        if all(name is None for name in name_list):
            return None

        grouped_indices: Dict[Optional[str], List[int]] = defaultdict(list)
        for idx, name in enumerate(name_list):
            grouped_indices[name].append(idx)

        decoded_outputs: List[Optional[torch.Tensor]] = [None] * batch_size

        for name, indices in grouped_indices.items():
            ensured = self._ensure_vae(name)
            if ensured is None:
                return None
            vae, spec = ensured

            params = next(vae.parameters())
            vae_device = params.device
            vae_dtype = params.dtype

            latent_chunk = latents[indices].to(device=vae_device, dtype=vae_dtype)

            with torch.no_grad():
                scaling = getattr(getattr(vae, "config", None), "scaling_factor", 1.0) or 1.0
                if spec.latents_scaled:
                    latent_chunk = latent_chunk / scaling
                decoded = vae.decode(latent_chunk).sample
                decoded = torch.clamp(decoded, -1.0, 1.0)

            decoded = decoded.to(device=self.device, dtype=torch.float32)

            for idx, tensor in zip(indices, decoded):
                decoded_outputs[idx] = tensor

        if any(entry is None for entry in decoded_outputs):
            return None

        stacked = torch.stack(decoded_outputs, dim=0)
        if squeeze_result:
            stacked = stacked[0]
        return stacked

    # ------------------------------------------------------------------ Validation helpers & caching
    def _get_validation_root_dir(self, dataset_name: str) -> str:
        """Resolve and create the visualization root for a dataset."""
        root = osp.join(self.opt["path"]["visualization"], dataset_name)
        os.makedirs(root, exist_ok=True)
        return root

    def _get_cache_dir(self, dataset_name: str) -> str:
        """Return the cache directory for decoded tensors."""
        root = self._get_validation_root_dir(dataset_name)
        cache_dir = osp.join(root, "cache_folder")
        os.makedirs(cache_dir, exist_ok=True)
        self._val_cache_dir = cache_dir
        return cache_dir

    @staticmethod
    def _hash_str(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    def _decoded_cache_key(
        self,
        *,
        img_name: str,
        role: str,
        vae_key: str,
        spatial_hw: Tuple[int, int],
    ) -> str:
        base = f"{img_name}__{role}__{vae_key}__{spatial_hw[0]}x{spatial_hw[1]}"
        return f"{base}__{self._hash_str(base)}"

    def _pixel_metrics_requested(self) -> bool:
        metrics = (self.opt.get("val") or {}).get("metrics") or {}
        for cfg in metrics.values():
            if isinstance(cfg, Mapping) and str(cfg.get("space", "latent")).lower() == "pixel":
                return True
        return False

    def _decode_with_cache(
        self,
        latents: Optional[torch.Tensor],
        *,
        img_name: str,
        role: str,
        dataset_name: str,
    ) -> Optional[torch.Tensor]:
        """Decode latents to RGB tensors, caching LQ/GT results on disk."""
        if latents is None or not VAE_AVAILABLE:
            return None

        squeeze = False
        if latents.dim() == 3:
            latents = latents.unsqueeze(0)
            squeeze = True

        if latents.dim() != 4:
            raise ValueError(
                f"Expected latents with shape [B, C, H, W]; received {tuple(latents.shape)}."
            )

        batch_size = latents.shape[0]
        name_list = self._prepare_batch_vae_names(self._current_vae_names, batch_size)
        ensured = self._ensure_vae(name_list[0] if name_list else None)
        if ensured is None:
            return None
        _, spec = ensured
        vae_key = spec.cache_key
        spatial_hw = (latents.shape[-2], latents.shape[-1])

        cache_key = self._decoded_cache_key(
            img_name=img_name,
            role=role,
            vae_key=vae_key,
            spatial_hw=spatial_hw,
        )

        if cache_key in self._val_decode_mem_cache:
            cached = self._val_decode_mem_cache[cache_key]
            return cached if not squeeze else cached

        disk_path = None
        if role in {"lq", "gt"} and self._val_cache_dir:
            disk_path = osp.join(self._val_cache_dir, f"{cache_key}.pt")
            if osp.exists(disk_path):
                try:
                    tensor = torch.load(disk_path, map_location="cpu")
                    if isinstance(tensor, torch.Tensor):
                        self._val_decode_mem_cache[cache_key] = tensor
                        return tensor if not squeeze else tensor
                except Exception:
                    pass  # fall back to re-decode on failure

        decoded = self.decode_latents(latents, vae_names=self._current_vae_names)
        if decoded is None:
            return None

        if decoded.dim() == 4 and decoded.size(0) == 1:
            decoded = decoded.squeeze(0)

        decoded_cpu = decoded.detach().to("cpu", dtype=torch.float16).contiguous()

        self._val_decode_mem_cache[cache_key] = decoded_cpu

        if disk_path is not None:
            try:
                torch.save(decoded_cpu, disk_path)
            except Exception:
                pass

        return decoded_cpu if not squeeze else decoded_cpu

    def calculate_metric_in_space(
        self,
        pred_latent,
        gt_latent,
        metric_name,
        metric_opt,
        *,
        decoded_pred: Optional[torch.Tensor] = None,
        decoded_gt: Optional[torch.Tensor] = None,
    ):
        """Calculate metric in specified space (latent or pixel)"""
        metric_space = metric_opt.get("space", "latent")

        if metric_space == "latent":
            # Calculate metrics directly on latent tensors
            if metric_opt["type"] == "L1Loss":
                return F.l1_loss(pred_latent, gt_latent).item()
            elif metric_opt["type"] == "MSELoss":
                return F.mse_loss(pred_latent, gt_latent).item()
            else:
                return 0.0

        elif metric_space == "pixel":
            if decoded_pred is None or decoded_gt is None:
                # Decode latents to images once for all pixel-space metrics
                decoded_pred = self.decode_latents(pred_latent, vae_names=self._current_vae_names)
                decoded_gt = self.decode_latents(gt_latent, vae_names=self._current_vae_names)
                if decoded_pred is None or decoded_gt is None:
                    print(
                        f"Warning: Unable to decode latents, skipping pixel space metric: {metric_name}"
                    )
                    return 0.0
                if decoded_pred.dim() == 4 and decoded_pred.size(0) == 1:
                    decoded_pred = decoded_pred.squeeze(0)
                if decoded_gt.dim() == 4 and decoded_gt.size(0) == 1:
                    decoded_gt = decoded_gt.squeeze(0)
                decoded_pred = decoded_pred.detach().to("cpu", dtype=torch.float32)
                decoded_gt = decoded_gt.detach().to("cpu", dtype=torch.float32)
            else:
                decoded_pred = decoded_pred.detach().to("cpu", dtype=torch.float32)
                decoded_gt = decoded_gt.detach().to("cpu", dtype=torch.float32)

            # Handle L1Loss directly on tensors
            if metric_opt["type"] == "L1Loss":
                return F.l1_loss(decoded_pred, decoded_gt).item()

            # Handle metrics that use the basicsr `calculate_metric` dispatcher
            elif metric_opt["type"] in [
                "calculate_psnr",
                "calculate_ssim",
                "calculate_psnr_pt",
                "calculate_ssim_pt",
            ]:
                from basicsr.metrics import calculate_metric

                metric_data = {}

                # Check if it's a PyTorch metric (expects tensors in range [0, 1])
                if metric_opt["type"].endswith("_pt"):
                    # Convert decoded tensors from [-1, 1] to [0, 1]
                    pred_img_01 = (decoded_pred + 1.0) / 2.0
                    gt_img_01 = (decoded_gt + 1.0) / 2.0
                    # Use the required keys: 'img' and 'img2'
                    metric_data = dict(img=pred_img_01, img2=gt_img_01)
                # Otherwise, it's a NumPy metric (expects ndarray in range [0, 255])
                else:
                    # Use the existing helper to convert [-1, 1] tensor to [0, 255] numpy array
                    pred_img_np = self.tensor_to_numpy_image(decoded_pred)
                    gt_img_np = self.tensor_to_numpy_image(decoded_gt)
                    # Use the required keys: 'img' and 'img2'
                    metric_data = dict(img=pred_img_np, img2=gt_img_np)

                # Call the generic metric calculator
                result = calculate_metric(metric_data, metric_opt)

                # Ensure the result is a standard Python float for logging
                if isinstance(result, torch.Tensor):
                    return result.item()
                else:
                    # NumPy metrics already return floats, so just return
                    return result
            else:
                # Handle unknown pixel-space metric types gracefully
                return 0.0

        return 0.0

    def tensor_to_numpy_image(self, tensor):
        """Convert tensor to a numpy image array for plotting.
        Input shape: [C, H, W] or [B, C, H, W] in [-1, 1] range.
        Output shape: [H, W, C] in [0, 255] uint8 range (RGB).
        """
        image = tensor.detach().cpu().clamp(-1.0, 1.0)
        image = (image + 1.0) * 0.5  # to [0, 1]
        image = image.mul(255.0).clamp(0.0, 255.0).byte()
        if image.dim() == 4:
            image = image.squeeze(0)
        return image.permute(1, 2, 0).numpy()

    def _visualize_latent_channels(self, latents, title, ax, max_channels=4):
        """Visualize first few channels of latents on a matplotlib axis."""
        channels_to_show = min(max_channels, latents.shape[1])
        latents_cpu = latents.detach().cpu()

        if channels_to_show == 1:
            im = ax.imshow(latents_cpu[0, 0].numpy(), cmap="viridis")
            ax.set_title(f"{title}\nCh 0")
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Activation Value", rotation=270, labelpad=15)
        else:
            h, w = latents_cpu.shape[-2:]
            combined = torch.zeros(h * 2, w * 2)
            for i in range(min(4, channels_to_show)):
                row, col = i // 2, i % 2
                combined[row * h : (row + 1) * h, col * w : (col + 1) * w] = (
                    latents_cpu[0, i]
                )

            im = ax.imshow(combined.numpy(), cmap="viridis")
            ax.set_title(f"{title}\nCh 0-{channels_to_show-1}")
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Activation Value", rotation=270, labelpad=15)

        ax.axis("off")

    def _save_comparison_plot(
        self,
        save_path,
        img_name,
        lq_latent,
        pred_latent,
        gt_latent,
        decoded_lq,
        decoded_pred,
        decoded_gt,
        log_to_wandb=False,
        current_iter=None,
    ):
        """Creates and saves the full 2x4 visualization plot.

        Args:
            log_to_wandb: If True, also log the figure to wandb
            current_iter: Current iteration number for wandb logging
        """
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))

        # --- Row 1: Latent Tensors ---
        # Col 1: Low-quality (input) latents
        lh, lw = lq_latent.shape[-2:]
        self._visualize_latent_channels(
            lq_latent, f"Input Latents\n{lh}x{lw} spatial, 16 channels", axes[0, 0]
        )

        # Col 2: Predicted high-quality latents
        ph, pw = pred_latent.shape[-2:]
        self._visualize_latent_channels(
            pred_latent,
            f"Predicted Latents\n{ph}x{pw} spatial, 16 channels",
            axes[0, 1],
        )

        # Col 3: Ground-truth high-quality latents
        gh, gw = gt_latent.shape[-2:]
        self._visualize_latent_channels(
            gt_latent,
            f"Ground Truth Latents\n{gh}x{gw} spatial, 16 channels",
            axes[0, 2],
        )

        # Col 4: Difference between predicted and ground-truth latents
        diff_latents = torch.abs(gt_latent - pred_latent)
        diff_mean = diff_latents.mean(dim=1, keepdim=True)
        im = axes[0, 3].imshow(diff_mean[0, 0].cpu().numpy(), cmap="hot")
        axes[0, 3].set_title(
            f"Latent Difference |GT - Pred|\nAveraged across 16 channels"
        )
        axes[0, 3].axis("off")
        cbar = plt.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)
        cbar.set_label("Difference Magnitude", rotation=270, labelpad=15)

        # --- Row 2: Decoded Images ---
        if (
            decoded_lq is not None
            and decoded_pred is not None
            and decoded_gt is not None
        ):
            # Col 1: Decoded LQ
            axes[1, 0].imshow(self.tensor_to_numpy_image(decoded_lq))
            axes[1, 0].set_title(
                f"Decoded from Input Latents\n{decoded_lq.shape[-2]}x{decoded_lq.shape[-1]} image"
            )
            axes[1, 0].axis("off")

            # Col 2: Decoded Prediction
            axes[1, 1].imshow(self.tensor_to_numpy_image(decoded_pred))
            axes[1, 1].set_title(
                f"Decoded from Predicted Latents\n{decoded_pred.shape[-2]}x{decoded_pred.shape[-1]} image"
            )
            axes[1, 1].axis("off")

            # Col 3: Decoded Ground Truth
            axes[1, 2].imshow(self.tensor_to_numpy_image(decoded_gt))
            axes[1, 2].set_title(
                f"Decoded from Ground Truth Latents\n{decoded_gt.shape[-2]}x{decoded_gt.shape[-1]} image"
            )
            axes[1, 2].axis("off")

            # Col 4: Image Difference
            diff_decoded = torch.abs(decoded_gt - decoded_pred)
            axes[1, 3].imshow(self.tensor_to_numpy_image(diff_decoded))
            axes[1, 3].set_title(
                f"Image Difference |GT - Pred|\n{diff_decoded.shape[-2]}x{diff_decoded.shape[-1]} image"
            )
            axes[1, 3].axis("off")
        else:
            for i in range(4):
                axes[1, i].text(
                    0.5,
                    0.5,
                    "VAE not available\nCannot decode images",
                    ha="center",
                    va="center",
                )
                axes[1, i].axis("off")

        # --- Final Touches & Saving ---
        # Calculate metrics for title
        mse = F.mse_loss(pred_latent, gt_latent).item()
        mae = F.l1_loss(pred_latent, gt_latent).item()
        fig.suptitle(
            f"Latent Upscaler Visualization: {img_name}\n"
            f"Latent MSE: {mse:.6f}, Latent MAE: {mae:.6f}",
            fontsize=16,
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.85, hspace=0.4, wspace=0.3)

        # Save to file
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

        # Log to wandb if requested
        if log_to_wandb and WANDB_AVAILABLE and wandb.run is not None:
            try:
                # Log the matplotlib figure directly
                wandb.log(
                    {
                        f"validation/{img_name}": wandb.Image(fig),
                        "iteration": current_iter,
                    }
                )
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f"Failed to log image to wandb: {e}")

        plt.close(fig)  # Close figure to free memory

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Validation with caching for decoded tensors and separated metric/plot flows."""
        dataset_name = dataloader.dataset.opt["name"]
        with_metrics = self.opt["val"]["metrics"] is not None
        use_pbar = self.opt["val"].get("pbar", False)

        log_to_wandb = (
            self.opt.get("logger", {}).get("wandb", {}).get("project") is not None
        )
        max_wandb_images = (
            self.opt.get("logger", {}).get("wandb", {}).get("max_val_images", 8)
        )
        wandb_image_count = 0
        need_pixel_metrics = with_metrics and self._pixel_metrics_requested()

        if (save_img or need_pixel_metrics) and VAE_AVAILABLE:
            self._get_cache_dir(dataset_name)
        else:
            self._val_cache_dir = None
        self._val_decode_mem_cache.clear()

        evaluated_images = 0
        if with_metrics:
            if not hasattr(self, "metric_results"):
                self.metric_results = {
                    name: 0 for name in self.opt["val"]["metrics"].keys()
                }
            self._initialize_best_metric_results(dataset_name)
            for metric in self.metric_results.keys():
                self.metric_results[metric] = 0

        pbar = tqdm(total=len(dataloader), unit="image") if use_pbar else None

        # Limit number of images to log to wandb (to avoid clutter)
        max_wandb_images = (
            self.opt.get("logger", {}).get("wandb", {}).get("max_val_images", 8)
        )
        wandb_image_count = 0

        for idx, val_data in enumerate(tqdm(dataloader)):
            img_name = osp.splitext(osp.basename(val_data["lq_path"][0]))[0]
            self.feed_data(val_data)
            self.test()

            lq_latent = self.lq.detach()
            pred_latent = self.output.detach()
            gt_latent = self.gt.detach() if hasattr(self, "gt") else None

            decoded_lq_cpu: Optional[torch.Tensor] = None
            decoded_pred_cpu: Optional[torch.Tensor] = None
            decoded_gt_cpu: Optional[torch.Tensor] = None

            if VAE_AVAILABLE:
                if save_img and gt_latent is not None:
                    decoded_lq_cpu = self._decode_with_cache(
                        lq_latent, img_name=img_name, role="lq", dataset_name=dataset_name
                    )
                    decoded_gt_cpu = self._decode_with_cache(
                        gt_latent, img_name=img_name, role="gt", dataset_name=dataset_name
                    )
                    decoded_pred_cpu = self._decode_with_cache(
                        pred_latent, img_name=img_name, role="pred", dataset_name=dataset_name
                    )
                if need_pixel_metrics and gt_latent is not None:
                    if decoded_gt_cpu is None:
                        decoded_gt_cpu = self._decode_with_cache(
                            gt_latent, img_name=img_name, role="gt", dataset_name=dataset_name
                        )
                    if decoded_pred_cpu is None:
                        decoded_pred_cpu = self._decode_with_cache(
                            pred_latent, img_name=img_name, role="pred", dataset_name=dataset_name
                        )

            # --- Visualization ---
            if save_img and gt_latent is not None:
                # Determine save path
                if self.opt["is_train"]:
                    save_path = osp.join(
                        self.opt["path"]["visualization"],
                        img_name,
                        f"{img_name}_{current_iter}.png",
                    )
                else:
                    suffix = self.opt["val"]["suffix"] or self.opt["name"]
                    save_path = osp.join(
                        self.opt["path"]["visualization"],
                        dataset_name,
                        f"{img_name}_{suffix}.png",
                    )
                os.makedirs(osp.dirname(save_path), exist_ok=True)

                # Decode all latents to images if a VAE is available
                decoded_lq = self.decode_latents(lq_latent, vae_names=self._current_vae_names)
                decoded_pred = self.decode_latents(pred_latent, vae_names=self._current_vae_names)
                decoded_gt = self.decode_latents(gt_latent, vae_names=self._current_vae_names)

                # Decide whether to log this image to wandb
                should_log_wandb = log_to_wandb and wandb_image_count < max_wandb_images
                if should_log_wandb:
                    wandb_image_count += 1

                # Generate and save the plot
                self._save_comparison_plot(
                    save_path,
                    img_name,
                    lq_latent.cpu(),
                    pred_latent.cpu(),
                    gt_latent.cpu(),
                    decoded_lq_cpu.float() if decoded_lq_cpu is not None else None,
                    decoded_pred_cpu.float() if decoded_pred_cpu is not None else None,
                    decoded_gt_cpu.float() if decoded_gt_cpu is not None else None,
                    log_to_wandb=should_log_wandb,
                    current_iter=current_iter,
                )

            # --- Metrics Calculation ---
            if with_metrics and gt_latent is not None:
                for name, opt_ in self.opt["val"]["metrics"].items():
                    metric_value = self.calculate_metric_in_space(
                        pred_latent,
                        gt_latent,
                        name,
                        opt_,
                        decoded_pred=decoded_pred_cpu,
                        decoded_gt=decoded_gt_cpu,
                    )
                    self.metric_results[name] += metric_value
            evaluated_images += 1

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f"Test {img_name}")

            decoded_lq_cpu = None
            decoded_pred_cpu = None
            decoded_gt_cpu = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if pbar:
            pbar.close()

        if with_metrics and evaluated_images > 0:
            # Loop through the metrics that were calculated
            for metric_name in self.metric_results.keys():
                # Average the metric value over the validation set
                self.metric_results[metric_name] /= evaluated_images
                # Update the best result for THIS specific metric using its correct name
                self._update_best_metric_result(
                    dataset_name,
                    metric_name,
                    self.metric_results[metric_name],
                    current_iter,
                )
            # After all metrics have been updated, log their final values
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
