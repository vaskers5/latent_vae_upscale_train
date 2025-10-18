import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import hashlib
import os
from os import path as osp
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from collections import OrderedDict

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.nn import functional as F
from tqdm import tqdm

# --- Matplotlib (headless) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from basicsr.archs import build_network
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from .swinir_model import SwinIRModel

try:
    from diffusers import AutoencoderKL, AsymmetricAutoencoderKL  # type: ignore
    VAE_AVAILABLE = True
except ImportError:
    AutoencoderKL = AsymmetricAutoencoderKL = None  # type: ignore
    VAE_AVAILABLE = False

try:
    import wandb  # type: ignore
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore
    WANDB_AVAILABLE = False


def _resolve_dtype(value: Any) -> Optional[torch.dtype]:
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if not s:
            return None
        if not s.startswith("torch."):
            s = f"torch.{s}"
        attr = s.split(".")[-1]
        if hasattr(torch, attr) and isinstance(getattr(torch, attr), torch.dtype):
            return getattr(torch, attr)
    raise ValueError(f"Unsupported torch dtype spec: {value!r}")


@MODEL_REGISTRY.register()
class SwinIRLatentModel(SwinIRModel):
    """Streamlined SwinIR model with optional pixel-space decode via a single VAE."""

    def __init__(self, opt):
        # runtime state
        self.loss_configs: Dict[str, Dict[str, Any]] = {}
        self.loss_criteria: Dict[str, torch.nn.Module] = {}

        # VAE (single, lazy-loaded)
        self._vae = None
        self._vae_scaling: float = 1.0
        self._vae_latents_scaled: bool = False
        self._vae_dtype: Optional[torch.dtype] = None
        self._val_decode_mem_cache: Dict[str, torch.Tensor] = {}
        self._val_cache_dir: Optional[str] = None

        # read VAE opts once; defaults handled in _get_vae()
        self._vae_opt: Dict[str, Any] = {}
        self._vae_name: str = ""
        self._vae_cache_namespace: str = "default"
        self._initialize_vae_config(opt)

        super().__init__(opt)

        if not VAE_AVAILABLE:
            get_root_logger().warning(
                "diffusers is not available; pixel-space decoding and pixel metrics will be skipped."
            )

    def _initialize_vae_config(self, opt: Dict[str, Any]) -> None:
        """Resolve the VAE configuration from explicit opts or shared sources."""
        explicit_cfg = opt.get("vae")
        if isinstance(explicit_cfg, dict) and explicit_cfg:
            self._vae_opt = dict(explicit_cfg)
            self._vae_name = str(self._vae_opt.get("vae_name") or "")
        else:
            self._vae_opt = {}
            self._vae_name = ""

        sources = opt.get("vae_sources") or {}
        if not self._vae_opt and isinstance(sources, dict) and sources:
            dataset_vae_names: List[str] = []
            for dataset_cfg in (opt.get("datasets") or {}).values():
                names = dataset_cfg.get("vae_names")
                if isinstance(names, str):
                    dataset_vae_names.append(names)
                elif isinstance(names, (list, tuple)):
                    dataset_vae_names.extend([str(name) for name in names if name])

            for candidate in dataset_vae_names:
                cfg = sources.get(candidate)
                if isinstance(cfg, dict) and cfg:
                    self._vae_opt = dict(cfg)
                    self._vae_opt.setdefault("vae_name", candidate)
                    self._vae_name = candidate
                    break

            if not self._vae_opt:
                for candidate, cfg in sources.items():
                    if isinstance(cfg, dict) and cfg:
                        self._vae_opt = dict(cfg)
                        self._vae_opt.setdefault("vae_name", candidate)
                        self._vae_name = str(candidate)
                        break

        if not self._vae_opt:
            self._vae_opt = {}
        if not self._vae_name:
            self._vae_name = str(self._vae_opt.get("vae_name") or "")
        self._vae_cache_namespace = self._build_vae_cache_namespace(self._vae_opt)

    @staticmethod
    def _build_vae_cache_namespace(cfg: Dict[str, Any]) -> str:
        """Create a stable cache namespace for the active VAE configuration."""
        if not cfg:
            return "default"

        parts: List[str] = []
        for key in ("vae_name", "load_from", "hf_repo", "hf_revision", "vae_kind", "weights_dtype", "latents_scaled"):
            value = cfg.get(key)
            if value is not None:
                parts.append(str(value))

        if not parts:
            parts.append("default")

        raw = "::".join(parts)
        safe = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in raw)
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        return f"{safe}__{digest}"

    # ------------------------------ torch.compile helpers ------------------------------
    def _parse_compile_settings(self) -> Optional[Dict[str, Any]]:
        cfg = self.opt.get("compile", False)
        if not cfg:
            return None

        if isinstance(cfg, dict):
            enabled = cfg.get("enabled", True)
            if not enabled:
                return None
            mode = cfg.get("mode", "max-autotune")
            fullgraph = bool(cfg.get("fullgraph", False))
            dynamic = bool(cfg.get("dynamic", True))
            backend = cfg.get("backend", None)
            apply_raw = cfg.get("apply_to_validation", False)
            if isinstance(apply_raw, str):
                lowered = apply_raw.strip().lower()
                if lowered in {"1", "true", "yes", "on"}:
                    apply_to_validation = True
                elif lowered in {"0", "false", "no", "off"}:
                    apply_to_validation = False
                else:
                    apply_to_validation = bool(lowered)
            else:
                apply_to_validation = bool(apply_raw)
        else:
            if not bool(cfg):
                return None
            mode = "max-autotune"
            fullgraph = False
            dynamic = True
            backend = None
            apply_to_validation = False

        if not hasattr(torch, "compile"):
            get_root_logger().warning("torch.compile requested but this PyTorch build lacks torch.compile; running eager.")
            return None

        return {
            "mode": mode,
            "fullgraph": fullgraph,
            "dynamic": dynamic,
            "backend": backend,
            "apply_to_validation": apply_to_validation,
        }

    def _compile_module_if_requested(self, module: torch.nn.Module, name: str,
                                     settings: Optional[Dict[str, Any]]) -> torch.nn.Module:
        if not settings:
            return module

        logger = get_root_logger()
        if isinstance(module, DistributedDataParallel):
            logger.warning("Skip torch.compile for %s: DistributedDataParallel is not supported.", name)
            return module

        top_level = module
        wrapper: Optional[DataParallel] = None
        bare = module
        if isinstance(module, DataParallel):
            device_ids = getattr(module, "device_ids", None)
            if device_ids and len(device_ids) > 1:
                logger.warning(
                    "Skip torch.compile for %s: DataParallel with multiple devices is not supported; running eager.",
                    name)
                return module
            wrapper = module
            bare = module.module

        was_training = bare.training
        try:
            torch._dynamo.config.suppress_errors = True  # type: ignore[attr-defined]
            compiled = torch.compile(
                bare,
                mode=settings["mode"],
                fullgraph=settings["fullgraph"],
                dynamic=settings["dynamic"],
                backend=settings["backend"])
        except Exception as exc:  # pragma: no cover - best effort logging only
            logger.warning("torch.compile failed for %s; running in eager mode. %s", name, exc)
            return top_level

        compiled.train(was_training)
        if wrapper is not None:
            wrapper.module = compiled
            return wrapper
        return compiled

    # ------------------------------ Basic lifecycle ------------------------------
    def feed_data(self, data):
        # just use parent (no per-sample VAE names anymore)
        super().feed_data(data)

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Standardized loader; removes 'module.' prefixes and supports params_ema fallback."""
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        logger.info(f'Loading {net.__class__.__name__} from {load_path} [key={param_key}].')
        for k, v in list(load_net.items()):
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    def init_training_settings(self):
        """EMA, losses, optimizers, schedulers."""
        train_opt = self.opt["train"]
        compile_settings = self._parse_compile_settings()
        compile_args: Optional[Dict[str, Any]] = None
        compile_for_validation = False
        if compile_settings:
            compile_args = {
                "mode": compile_settings["mode"],
                "fullgraph": compile_settings["fullgraph"],
                "dynamic": compile_settings["dynamic"],
                "backend": compile_settings["backend"],
            }
            compile_for_validation = bool(compile_settings.get("apply_to_validation", False))
            try:
                torch.backends.cudnn.benchmark = True
            except AttributeError:
                pass
            cuda_backend = getattr(torch.backends, "cuda", None)
            if cuda_backend is not None and torch.cuda.is_available():
                matmul_backend = getattr(cuda_backend, "matmul", None)
                if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
                    matmul_backend.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                try:
                    torch.set_float32_matmul_precision("high")
                except (TypeError, AttributeError):
                    pass

        self.net_g = self._compile_module_if_requested(self.net_g, "net_g", compile_args)
        self.net_g.train()

        # EMA
        self.ema_decay = train_opt.get("ema_decay", 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f"Use EMA with decay: {self.ema_decay}")
            self.net_g_ema = build_network(self.opt["network_g"]).to(self.device)
            load_path = self.opt["path"].get("pretrain_network_g")
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt["path"].get("strict_load_g", True),
                                  "params_ema")
            else:
                self.model_ema(0)
            ema_compile_args = compile_args if compile_for_validation else None
            self.net_g_ema = self._compile_module_if_requested(self.net_g_ema, "net_g_ema", ema_compile_args)
            if compile_args and not compile_for_validation:
                logger.info("Torch.compile disabled for EMA network; validation runs in eager mode.")
            self.net_g_ema.eval()

        # Losses / optim / sched
        self._init_space_aware_losses(train_opt)
        self.setup_optimizers()
        self.setup_schedulers()

    def _init_space_aware_losses(self, train_opt: Dict[str, Any]):
        """Build all losses; support 'space' (latent|pixel) and 'loss_weight'."""
        from basicsr.losses import build_loss
        self.loss_configs.clear()
        self.loss_criteria.clear()

        for name, cfg in train_opt.items():
            if not (isinstance(cfg, dict) and name.endswith("_opt")):
                continue
            space = str(cfg.get("space", "latent")).lower()
            weight = float(cfg.get("loss_weight", 1.0))
            build_cfg = cfg.copy()
            build_cfg.pop("space", None)
            build_cfg.pop("loss_weight", None)
            self.loss_criteria[name] = build_loss(build_cfg).to(self.device)
            self.loss_configs[name] = {"space": space, "weight": weight}
            get_root_logger().info(f"Initialized {name} in {space} space (w={weight}).")

        if not self.loss_criteria:
            raise ValueError("No losses configured. Add at least one '*_opt' entry in train options.")

    # ------------------------------ Train step -----------------------------------
    @staticmethod
    def _call_loss(criterion, pred, tgt=None):
        try:
            return criterion(pred, tgt) if tgt is not None else criterion(pred)
        except TypeError:
            return criterion(pred)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        # decode once if any pixel-space loss exists
        need_pixel = any(v["space"] == "pixel" for v in self.loss_configs.values())
        decoded_pred = self.decode_latents(self.output) if need_pixel else None
        decoded_gt = self.decode_latents(self.gt) if (need_pixel and hasattr(self, "gt")) else None

        l_total = 0.0
        loss_dict = OrderedDict()

        for loss_name, cfg in self.loss_configs.items():
            crit = self.loss_criteria[loss_name]
            w = cfg["weight"]
            if cfg["space"] == "latent":
                val = self._call_loss(crit, self.output, getattr(self, "gt", None))
            else:  # pixel
                if decoded_pred is None or decoded_gt is None:
                    get_root_logger().warning(f"Skip pixel loss {loss_name}: cannot decode.")
                    continue
                val = self._call_loss(crit, decoded_pred, decoded_gt)

            if isinstance(val, (tuple, list)):
                for i, v in enumerate(val):
                    if v is not None:
                        l_total = l_total + v * w
                        loss_dict[f"{loss_name}_{i}"] = v
            else:
                loss_dict[loss_name] = val
                l_total = l_total + val * w

        l_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if getattr(self, "ema_decay", 0) > 0:
            self.model_ema(decay=self.ema_decay)

    # ------------------------------ VAE decode -----------------------------------
    def _get_vae(self):
        """Lazy-load a single VAE according to opt['vae'] (or sensible defaults)."""
        if not VAE_AVAILABLE:
            return None
        if self._vae is not None:
            return self._vae

        cfg = dict(self._vae_opt or {})
        if self._vae_name and "vae_name" not in cfg:
            cfg["vae_name"] = self._vae_name
        source = cfg.get("load_from") or cfg.get("hf_repo") or "stabilityai/sdxl-vae"
        kind = str(cfg.get("vae_kind", "kl")).lower()
        dtype = _resolve_dtype(cfg.get("weights_dtype")) if cfg.get("weights_dtype") else None

        kwargs: Dict[str, Any] = {}
        if isinstance(source, (str, Path)):
            source = str(Path(source).expanduser())
        if cfg.get("hf_subfolder"):
            kwargs["subfolder"] = cfg["hf_subfolder"]
        if cfg.get("hf_revision"):
            kwargs["revision"] = cfg["hf_revision"]
        if cfg.get("hf_auth_token"):
            kwargs["use_auth_token"] = cfg["hf_auth_token"]
        if dtype is not None:
            kwargs["torch_dtype"] = dtype

        logger = get_root_logger()
        vae_name = self._vae_name or cfg.get("vae_name") or "unnamed"
        logger.info(f"Loading VAE(name={vae_name}, kind={kind}) from {source}")

        if kind in {"asymmetric_kl", "kl_asymmetric", "asym_kl"} and AsymmetricAutoencoderKL is not None:
            vae = AsymmetricAutoencoderKL.from_pretrained(source, **kwargs)
        else:
            vae = AutoencoderKL.from_pretrained(source, **kwargs)  # default

        if dtype is not None:
            vae = vae.to(dtype=dtype)

        vae = vae.to(self.device).eval()
        self._vae = vae
        self._vae_dtype = dtype
        self._vae_scaling = float(getattr(getattr(vae, "config", None), "scaling_factor", 1.0) or 1.0)
        self._vae_latents_scaled = bool(cfg.get("latents_scaled", False))
        self._vae_cache_namespace = self._build_vae_cache_namespace(cfg)
        return self._vae

    def decode_latents(self, latents: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Decode [B,C,H,W] (or [C,H,W]) latents to [-1,1] RGB; returns float32 on model device."""
        if latents is None or not VAE_AVAILABLE:
            return None

        squeeze = False
        if latents.dim() == 3:
            latents, squeeze = latents.unsqueeze(0), True
        if latents.dim() != 4:
            raise ValueError(f"Expected latents [B,C,H,W], got {tuple(latents.shape)}")

        vae = self._get_vae()
        if vae is None:
            return None

        with torch.inference_mode():
            z = latents.to(device=self.device, dtype=getattr(self._vae, "dtype", None) or latents.dtype)
            if self._vae_latents_scaled:
                z = z / self._vae_scaling
            decoded = vae.decode(z).sample
            decoded = torch.clamp(decoded, -1.0, 1.0).to(self.device, dtype=torch.float32)

        return decoded[0] if squeeze and decoded.dim() == 4 and decoded.size(0) == 1 else decoded

    # ------------------------------ Validation + caching -------------------------
    def _get_validation_root_dir(self, dataset_name: str) -> str:
        root = osp.join(self.opt["path"]["visualization"], dataset_name)
        os.makedirs(root, exist_ok=True)
        return root

    def _get_cache_dir(self, dataset_name: str) -> str:
        root = self._get_validation_root_dir(dataset_name)
        cache_root_name = (self.opt.get("val") or {}).get("cache_dir_name") or "cache_folder"
        cache_root = osp.join(root, cache_root_name)
        namespace = getattr(self, "_vae_cache_namespace", "default")
        cache_dir = osp.join(cache_root, namespace)
        os.makedirs(cache_dir, exist_ok=True)
        self._val_cache_dir = cache_dir
        return cache_dir

    @staticmethod
    def _hash_str(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    def _decoded_cache_key(self, *, img_name: str, role: str, spatial_hw: Tuple[int, int]) -> str:
        vae_tag = type(self._vae).__name__ if self._vae is not None else "no_vae"
        namespace = getattr(self, "_vae_cache_namespace", "default")
        base = f"{namespace}__{img_name}__{role}__{vae_tag}__{spatial_hw[0]}x{spatial_hw[1]}"
        return f"{base}__{self._hash_str(base)}"

    def _pixel_metrics_requested(self) -> bool:
        metrics = (self.opt.get("val") or {}).get("metrics") or {}
        for cfg in metrics.values():
            if isinstance(cfg, dict) and str(cfg.get("space", "latent")).lower() == "pixel":
                return True
        return False

    def _decode_with_cache(self, latents: Optional[torch.Tensor], *, img_name: str, role: str,
                           dataset_name: str) -> Optional[torch.Tensor]:
        if latents is None or not VAE_AVAILABLE:
            return None

        # ensure 4D
        squeeze = False
        if latents.dim() == 3:
            latents, squeeze = latents.unsqueeze(0), True
        if latents.dim() != 4:
            raise ValueError(f"Expected latents [B,C,H,W], got {tuple(latents.shape)}")

        H, W = latents.shape[-2:]
        key = self._decoded_cache_key(img_name=img_name, role=role, spatial_hw=(H, W))

        if key in self._val_decode_mem_cache:
            cached = self._val_decode_mem_cache[key]
            return cached if not squeeze else cached

        disk_path = None
        if role in {"lq", "gt"} and self._val_cache_dir:
            disk_path = osp.join(self._val_cache_dir, f"{key}.pt")
            if osp.exists(disk_path):
                try:
                    t = torch.load(disk_path, map_location="cpu")
                    if isinstance(t, torch.Tensor):
                        self._val_decode_mem_cache[key] = t
                        return t if not squeeze else t
                except Exception:
                    pass  # fall through

        decoded = self.decode_latents(latents)
        if decoded is None:
            return None
        if decoded.dim() == 4 and decoded.size(0) == 1:
            decoded = decoded.squeeze(0)

        decoded_cpu = decoded.detach().to("cpu", dtype=torch.float16).contiguous()
        self._val_decode_mem_cache[key] = decoded_cpu

        if disk_path is not None:
            try:
                torch.save(decoded_cpu, disk_path)
            except Exception:
                pass

        return decoded_cpu if not squeeze else decoded_cpu

    # ------------------------------ Metrics & viz --------------------------------
    def calculate_metric_in_space(self, pred_latent, gt_latent, metric_name, metric_opt,
                                  *, decoded_pred: Optional[torch.Tensor] = None,
                                  decoded_gt: Optional[torch.Tensor] = None):
        """Compute metric in 'latent' or 'pixel' space."""
        space = metric_opt.get("space", "latent")

        if space == "latent":
            if metric_opt["type"] == "L1Loss":
                return F.l1_loss(pred_latent, gt_latent).item()
            if metric_opt["type"] == "MSELoss":
                return F.mse_loss(pred_latent, gt_latent).item()
            return 0.0

        # pixel space
        if decoded_pred is None or decoded_gt is None:
            decoded_pred = self.decode_latents(pred_latent)
            decoded_gt = self.decode_latents(gt_latent)
            if decoded_pred is None or decoded_gt is None:
                print(f"Warning: Unable to decode latents; skip metric: {metric_name}")
                return 0.0
            if decoded_pred.dim() == 4 and decoded_pred.size(0) == 1:
                decoded_pred = decoded_pred.squeeze(0)
            if decoded_gt.dim() == 4 and decoded_gt.size(0) == 1:
                decoded_gt = decoded_gt.squeeze(0)
            decoded_pred = decoded_pred.detach().to("cpu", dtype=torch.float32)
            decoded_gt = decoded_gt.detach().to("cpu", dtype=torch.float32)

        if decoded_pred.dim() == 3:
            decoded_pred = decoded_pred.unsqueeze(0)
        if decoded_gt.dim() == 3:
            decoded_gt = decoded_gt.unsqueeze(0)

        if metric_opt["type"] == "L1Loss":
            return F.l1_loss(decoded_pred, decoded_gt).item()

        elif metric_opt["type"] in ["calculate_psnr", "calculate_ssim",
                                    "calculate_psnr_pt", "calculate_ssim_pt"]:
            from basicsr.metrics import calculate_metric
            if metric_opt["type"].endswith("_pt"):
                # expects tensors in [0,1]
                pred01 = (decoded_pred + 1.0) / 2.0
                gt01 = (decoded_gt + 1.0) / 2.0
                data = dict(img=pred01, img2=gt01)
            else:
                # expects numpy arrays in [0,255]
                pred_np = self.tensor_to_numpy_image(decoded_pred)
                gt_np = self.tensor_to_numpy_image(decoded_gt)
                data = dict(img=pred_np, img2=gt_np)
            res = calculate_metric(data, metric_opt)
            return res.item() if isinstance(res, torch.Tensor) else res

        return 0.0

    def tensor_to_numpy_image(self, tensor):
        """[C,H,W] or [B,C,H,W] in [-1,1] -> [H,W,C] uint8 RGB."""
        img = tensor.detach().cpu().clamp(-1, 1)
        img = (img + 1.0) * 0.5
        img = img.mul(255.0).clamp(0, 255).byte()
        if img.dim() == 4:
            img = img.squeeze(0)
        return img.permute(1, 2, 0).numpy()

    def _visualize_latent_channels(self, latents, title, ax, max_channels=4):
        """Show first channels of latents; grid up to 2x2."""
        ch = min(max_channels, latents.shape[1])
        lat = latents.detach().cpu()
        if ch == 1:
            im = ax.imshow(lat[0, 0].numpy(), cmap="viridis")
            ax.set_title(f"{title}\nCh 0")
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Activation", rotation=270, labelpad=15)
        else:
            h, w = lat.shape[-2:]
            grid = torch.zeros(h * 2, w * 2)
            for i in range(min(4, ch)):
                r, c = divmod(i, 2)
                grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = lat[0, i]
            im = ax.imshow(grid.numpy(), cmap="viridis")
            ax.set_title(f"{title}\nCh 0-{ch-1}")
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Activation", rotation=270, labelpad=15)
        ax.axis("off")
    
    def _save_comparison_plot(self, save_path, img_name, lq_latent, pred_latent, gt_latent,
                              decoded_lq, decoded_pred, decoded_gt, log_to_wandb=False,
                              current_iter=None):
        """2x4 panel: latent input/pred/gt/diff and decoded input/pred/gt/diff."""
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))

        # row 1: latents
        self._visualize_latent_channels(
            lq_latent, f"Input Latents\n{lq_latent.shape[-2]}x{lq_latent.shape[-1]}, C={lq_latent.shape[1]}", axes[0, 0]
        )
        self._visualize_latent_channels(
            pred_latent, f"Pred Latents\n{pred_latent.shape[-2]}x{pred_latent.shape[-1]}, C={pred_latent.shape[1]}",
            axes[0, 1]
        )
        self._visualize_latent_channels(
            gt_latent, f"GT Latents\n{gt_latent.shape[-2]}x{gt_latent.shape[-1]}, C={gt_latent.shape[1]}",
            axes[0, 2]
        )

        diff = torch.abs(gt_latent - pred_latent).mean(dim=1, keepdim=True)
        im = axes[0, 3].imshow(diff[0, 0].cpu().numpy(), cmap="hot")
        axes[0, 3].set_title("Latent |GT - Pred| (mean over C)")
        axes[0, 3].axis("off")
        cbar = plt.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)
        cbar.set_label("Diff", rotation=270, labelpad=15)

        # row 2: decoded
        if decoded_lq is not None and decoded_pred is not None and decoded_gt is not None:
            axes[1, 0].imshow(self.tensor_to_numpy_image(decoded_lq))
            axes[1, 0].set_title(f"Decoded LQ\n{decoded_lq.shape[-2]}x{decoded_lq.shape[-1]}")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(self.tensor_to_numpy_image(decoded_pred))
            axes[1, 1].set_title(f"Decoded Pred\n{decoded_pred.shape[-2]}x{decoded_pred.shape[-1]}")
            axes[1, 1].axis("off")

            axes[1, 2].imshow(self.tensor_to_numpy_image(decoded_gt))
            axes[1, 2].set_title(f"Decoded GT\n{decoded_gt.shape[-2]}x{decoded_gt.shape[-1]}")
            axes[1, 2].axis("off")

            diff_dec = torch.abs(decoded_gt - decoded_pred)
            axes[1, 3].imshow(self.tensor_to_numpy_image(diff_dec))
            axes[1, 3].set_title(f"Decoded |GT - Pred|\n{diff_dec.shape[-2]}x{diff_dec.shape[-1]}")
            axes[1, 3].axis("off")
        else:
            for i in range(4):
                axes[1, i].text(0.5, 0.5, "VAE not available\nCannot decode", ha="center", va="center")
                axes[1, i].axis("off")

        mse = F.mse_loss(pred_latent, gt_latent).item()
        mae = F.l1_loss(pred_latent, gt_latent).item()
        fig.suptitle(f"Latent Upscaler: {img_name} | MSE: {mse:.6f}, MAE: {mae:.6f}", fontsize=16)

        plt.tight_layout()
        plt.subplots_adjust(top=0.85, hspace=0.4, wspace=0.3)
        # plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if log_to_wandb and WANDB_AVAILABLE and wandb.run is not None:
            try:
                wandb.log({f"validation/{img_name}": wandb.Image(fig), "iteration": current_iter})
            except Exception as e:
                get_root_logger().warning(f"wandb log failed: {e}")

        plt.close(fig)

    # ------------------------------ Validation -----------------------------------
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Validation with optional pixel-space metrics and cached decode."""
        dataset_name = dataloader.dataset.opt["name"]
        val_opt = self.opt.get("val") or {}
        with_metrics = val_opt.get("metrics") is not None
        use_pbar = bool(val_opt.get("pbar", False))

        log_to_wandb = (self.opt.get("logger", {}).get("wandb", {}).get("project") is not None)
        max_wandb_images = int(self.opt.get("logger", {}).get("wandb", {}).get("max_val_images", 8))
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
                self.metric_results = {name: 0 for name in val_opt["metrics"].keys()}
            self._initialize_best_metric_results(dataset_name)
            for k in list(self.metric_results.keys()):
                self.metric_results[k] = 0

        total_images = 0
        if use_pbar:
            try:
                total_images = len(dataloader.dataset)  # type: ignore[arg-type]
            except (TypeError, AttributeError):
                try:
                    total_images = len(dataloader)
                except TypeError:
                    total_images = 0

        pbar = tqdm(total=total_images, unit="image", disable=(not use_pbar))

        for batch_idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.test()

            lq_lat = self.lq.detach()
            pred_lat = self.output.detach()
            gt_lat = self.gt.detach() if hasattr(self, "gt") else None

            if lq_lat.dim() == 3: lq_lat = lq_lat.unsqueeze(0)
            if pred_lat.dim() == 3: pred_lat = pred_lat.unsqueeze(0)
            if torch.is_tensor(gt_lat) and gt_lat.dim() == 3: gt_lat = gt_lat.unsqueeze(0)

            B = lq_lat.shape[0]
            batch_lq_paths = val_data.get("lq_path")
            lq_paths = list(batch_lq_paths) if isinstance(batch_lq_paths, (list, tuple)) else [batch_lq_paths]

            for i in range(B):
                img_path = lq_paths[i] if i < len(lq_paths) else None
                img_name = osp.splitext(osp.basename(img_path))[0] if isinstance(img_path, str) else f"sample_{batch_idx}_{i}"

                lq_i = lq_lat[i:i+1]
                pred_i = pred_lat[i:i+1]
                gt_i = gt_lat[i:i+1] if torch.is_tensor(gt_lat) else None

                dec_lq = dec_pred = dec_gt = None
                if VAE_AVAILABLE and (save_img or need_pixel_metrics) and gt_i is not None:
                    dec_lq = self._decode_with_cache(lq_i, img_name=img_name, role="lq", dataset_name=dataset_name)
                    dec_gt = self._decode_with_cache(gt_i, img_name=img_name, role="gt", dataset_name=dataset_name)
                    dec_pred = self._decode_with_cache(pred_i, img_name=img_name, role="pred", dataset_name=dataset_name)

                # Visualization
                if save_img and gt_i is not None:
                    if self.opt["is_train"]:
                        save_path = osp.join(self.opt["path"]["visualization"], img_name, f"{img_name}_{current_iter}.png")
                    else:
                        suffix = val_opt.get("suffix") or self.opt["name"]
                        save_path = osp.join(self.opt["path"]["visualization"], dataset_name, f"{img_name}_{suffix}.png")
                    os.makedirs(osp.dirname(save_path), exist_ok=True)

                    should_log = log_to_wandb and (wandb_image_count < max_wandb_images)
                    if should_log:
                        wandb_image_count += 1

                    self._save_comparison_plot(
                        save_path, img_name,
                        lq_i.cpu(), pred_i.cpu(), gt_i.cpu(),
                        None if dec_lq is None else dec_lq.float(),
                        None if dec_pred is None else dec_pred.float(),
                        None if dec_gt is None else dec_gt.float(),
                        log_to_wandb=should_log,
                        current_iter=current_iter
                    )

                # Metrics
                if with_metrics and gt_i is not None:
                    for name, opt_ in val_opt["metrics"].items():
                        v = self.calculate_metric_in_space(
                            pred_i, gt_i, name, opt_, decoded_pred=dec_pred, decoded_gt=dec_gt
                        )
                        self.metric_results[name] += v

                evaluated_images += 1
                if use_pbar:
                    pbar.update(1)
                    pbar.set_description(f"Val {img_name}")

        if use_pbar:
            pbar.close()

        if with_metrics and evaluated_images > 0:
            for k in self.metric_results.keys():
                self.metric_results[k] /= evaluated_images
                self._update_best_metric_result(dataset_name, k, self.metric_results[k], current_iter)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
