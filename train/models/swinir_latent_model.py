import cv2
import numpy as np
import os
import torch
import warnings
from collections import OrderedDict
from os import path as osp
from torch.nn import functional as F
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# --- Matplotlib Integration ---
# Use the 'Agg' backend for non-interactive environments (servers, etc.)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train.archs import build_network

# --- End Matplotlib Integration ---
from train.utils import get_root_logger, imwrite
from train.utils.registry import MODEL_REGISTRY
from .swinir_model import SwinIRModel

# Try to import VAE for decoding (optional)
try:
    from diffusers import AutoencoderKL

    VAE_AVAILABLE = True
except ImportError:
    VAE_AVAILABLE = False
    print("Warning: diffusers not available. VAE decoding will be skipped.")

# Try to import wandb
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Wandb logging will be skipped.")


@MODEL_REGISTRY.register()
class SwinIRLatentModel(SwinIRModel):
    """SwinIR model for latent space super resolution with custom visualization.
    This version uses Matplotlib to replicate the visualization style from the
    provided reference script.
    """

    def __init__(self, opt):
        # Initialize loss_configs before calling super().__init__()
        self.loss_configs = {}
        super().__init__(opt)
        self.vae = None
        if VAE_AVAILABLE:
            self._load_vae()

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
        from train.losses import build_loss

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
                loss_value = loss_criterion(self.output, self.gt)

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
                if self.vae is not None:
                    # Decode latents to images
                    decoded_pred = self.decode_latents(self.output)
                    decoded_gt = self.decode_latents(self.gt)

                    loss_value = loss_criterion(decoded_pred, decoded_gt)

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
                else:
                    print(
                        f"Warning: VAE not available, skipping pixel space loss: {loss_name}"
                    )
                    continue

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def _load_vae(self):
        """Load Flux VAE for decoding latents to images"""
        try:
            print("Loading Flux VAE for latent decoding...")
            self.vae = (
                AutoencoderKL.from_pretrained("wolfgangblack/flux_vae")
                .to(self.device)
                .eval()
            )
            print("✓ Flux VAE loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load Flux VAE: {e}")
            self.vae = None

    def decode_latents(self, latents):
        """Decode latents to RGB images using VAE"""
        if self.vae is None:
            return None

        with torch.no_grad():
            # Unscale latents (Flux VAE scaling factor)
            latents = latents / self.vae.config.scaling_factor
            # Decode to images
            images = self.vae.decode(latents).sample
            # Clamp to [-1, 1] range
            images = torch.clamp(images, -1.0, 1.0)

        return images

    def calculate_metric_in_space(
        self, pred_latent, gt_latent, metric_name, metric_opt
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
            if not self.vae:
                print(
                    f"Warning: VAE not available, skipping pixel space metric: {metric_name}"
                )
                return 0.0

            # Decode latents to images once for all pixel-space metrics
            decoded_pred = self.decode_latents(pred_latent)
            decoded_gt = self.decode_latents(gt_latent)

            # Handle L1Loss directly on tensors
            if metric_opt["type"] == "L1Loss":
                return F.l1_loss(decoded_pred, decoded_gt).item()

            # Handle metrics that use the train `calculate_metric` dispatcher
            elif metric_opt["type"] in [
                "calculate_psnr",
                "calculate_ssim",
                "calculate_psnr_pt",
                "calculate_ssim_pt",
            ]:
                from train.metrics import calculate_metric

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
        """Custom validation for latent space with detailed Matplotlib visualization."""
        dataset_name = dataloader.dataset.opt["name"]
        with_metrics = self.opt["val"]["metrics"] is not None
        use_pbar = self.opt["val"].get("pbar", False)

        # Check if wandb logging is enabled
        log_to_wandb = (
            self.opt.get("logger", {}).get("wandb", {}).get("project") is not None
        )

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

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data["lq_path"][0]))[0]
            self.feed_data(val_data)
            self.test()

            lq_latent = self.lq.detach()
            pred_latent = self.output.detach()
            gt_latent = self.gt.detach() if hasattr(self, "gt") else None

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

                # Decode all latents to images if VAE is available
                decoded_lq, decoded_pred, decoded_gt = None, None, None
                if self.vae:
                    decoded_lq = self.decode_latents(lq_latent)
                    decoded_pred = self.decode_latents(pred_latent)
                    decoded_gt = self.decode_latents(gt_latent)

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
                    decoded_lq.cpu() if decoded_lq is not None else None,
                    decoded_pred.cpu() if decoded_pred is not None else None,
                    decoded_gt.cpu() if decoded_gt is not None else None,
                    log_to_wandb=should_log_wandb,
                    current_iter=current_iter,
                )

            # --- Metrics Calculation ---
            if with_metrics and gt_latent is not None:
                for name, opt_ in self.opt["val"]["metrics"].items():
                    metric_value = self.calculate_metric_in_space(
                        pred_latent, gt_latent, name, opt_
                    )
                    self.metric_results[name] += metric_value

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f"Test {img_name}")

        if pbar:
            pbar.close()

        if with_metrics:
            # Loop through the metrics that were calculated
            for metric_name in self.metric_results.keys():
                # Average the metric value over the validation set
                self.metric_results[metric_name] /= idx + 1
                # Update the best result for THIS specific metric using its correct name
                self._update_best_metric_result(
                    dataset_name,
                    metric_name,
                    self.metric_results[metric_name],
                    current_iter,
                )
            # After all metrics have been updated, log their final values
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
