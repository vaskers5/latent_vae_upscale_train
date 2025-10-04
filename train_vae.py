# -*- coding: utf-8 -*-
import os
import math
import re
import torch
import numpy as np
import random
import gc
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
from diffusers import AutoencoderKL, AsymmetricAutoencoderKL
# QWEN: импорт класса
from diffusers import AutoencoderKLQwenImage
from diffusers import AutoencoderKLWan

from accelerate import Accelerator
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import bitsandbytes as bnb
import wandb
import lpips
from collections import deque
from contextlib import nullcontext
from pytorch_memlab import profile


# --------------------------- Параметры ---------------------------
ds_path            = "./workspace/d23/d23/"
project            = "simple_vae2x_1024"
batch_size         = 6
base_learning_rate = 6e-6
min_learning_rate  = 9e-7
num_epochs         = 1000
sample_interval_share = 200
use_wandb          = True
save_model         = True
use_decay          = True
optimizer_type     = "adam8bit"
dtype              = torch.float32  # torch.float32, torch.float16, torch.bfloat16
GLOBAL_SAMPLE_INTERVAL = 50
model_resolution   = 256
high_resolution    = 512
limit              = 0
save_barrier       = 1.003
warmup_percent     = 0.01
percentile_clipping = 95
beta2              = 0.97
eps                = 1e-6
clip_grad_norm     = 1.0
mixed_precision    = "no"
gradient_accumulation_steps = 4
generated_folder   = "samples"
save_as            = "simple_vae2x_nightly"
num_workers        = 0
device = None

# --- Режимы обучения ---
# QWEN: учим только декодер
train_decoder_only = True
full_training      = False  # если True — учим весь VAE и добавляем KL (ниже)
kl_ratio           = 0.00

# Доли лоссов
loss_ratios = {
    "lpips": 0.75,
    "edge":  0.05,
    "mse":   0.10,
    "mae":   0.10,
    "kl":    0.00,  # активируем при full_training=True
}
median_coeff_steps = 256

lpips_backbone = "vgg"  # backbone for LPIPS metric
lpips_eval_resolution = model_resolution  # downsample inputs for LPIPS to save VRAM; set to 0 to disable

resize_long_side = 1280  # ресайз длинной стороны исходных картинок

# QWEN: конфиг загрузки модели
vae_kind      = "kl"  # "qwen" или "kl" (обычный)

Path(generated_folder).mkdir(parents=True, exist_ok=True)

accelerator = Accelerator(
    mixed_precision=mixed_precision,
    gradient_accumulation_steps=gradient_accumulation_steps
)
device = accelerator.device

# reproducibility
seed = int(datetime.now().strftime("%Y%m%d"))
torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
torch.backends.cudnn.benchmark = False

# --------------------------- WandB ---------------------------
if use_wandb and accelerator.is_main_process:
    wandb.init(project=project, config={
        "batch_size": batch_size,
        "base_learning_rate": base_learning_rate,
        "num_epochs": num_epochs,
        "optimizer_type": optimizer_type,
        "model_resolution": model_resolution,
        "high_resolution": high_resolution,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "train_decoder_only": train_decoder_only,
        "full_training": full_training,
        "kl_ratio": kl_ratio,
        "vae_kind": vae_kind,
    })

# --------------------------- VAE ---------------------------
def get_core_model(model):
    m = model
    # если модель уже обёрнута torch.compile
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m

def is_video_vae(model) -> bool:
    # WAN/Qwen — это видео-VAEs
    if vae_kind in ("wan", "qwen"):
        return True
    # fallback по структуре (если понадобится)
    try:
        core = get_core_model(model)
        enc = getattr(core, "encoder", None)
        conv_in = getattr(enc, "conv_in", None)
        w = getattr(conv_in, "weight", None)
        if isinstance(w, torch.nn.Parameter):
            return w.ndim == 5
    except Exception:
        pass
    return False

# загрузка
if vae_kind == "qwen":
    vae = AutoencoderKLQwenImage.from_pretrained("Qwen/Qwen-Image", subfolder="vae")
else:
    if vae_kind == "wan":
        vae = AutoencoderKLWan.from_pretrained(project)
    else:
        # старое поведение (пример)
        if model_resolution==high_resolution:
            vae = AutoencoderKL.from_pretrained(project)
        else:
            vae = AsymmetricAutoencoderKL.from_pretrained(project)

vae = vae.to(dtype)

# torch.compile (опционально)
if hasattr(torch, "compile"):
    try:
        vae = torch.compile(vae)
    except Exception as e:
        print(f"[WARN] torch.compile failed: {e}")

# --------------------------- Freeze/Unfreeze ---------------------------
core = get_core_model(vae)

for p in core.parameters():
    p.requires_grad = False

unfrozen_param_names = []

if full_training and not train_decoder_only:
    for name, p in core.named_parameters():
        p.requires_grad = True
        unfrozen_param_names.append(name)
        loss_ratios["kl"] = float(kl_ratio)
        trainable_module = core
else:
# учим только декодер + post_quant_conv на "ядре" модели
    if hasattr(core, "decoder"):
        for name, p in core.decoder.named_parameters():
            p.requires_grad = True
            unfrozen_param_names.append(f"decoder.{name}")
    if hasattr(core, "post_quant_conv"):
        for name, p in core.post_quant_conv.named_parameters():
            p.requires_grad = True
            unfrozen_param_names.append(f"post_quant_conv.{name}")
            trainable_module = core.decoder if hasattr(core, "decoder") else core

print(f"[INFO] Разморожено параметров: {len(unfrozen_param_names)}. Первые 200 имён:")
for nm in unfrozen_param_names[:200]:
    print(" ", nm)

# --------------------------- Датасет ---------------------------
class PngFolderDataset(Dataset):
    def __init__(self, root_dir, min_exts=('.png',), resolution=1024, limit=0):
        self.root_dir = root_dir
        self.resolution = resolution
        self.paths = []
        for root, _, files in os.walk(root_dir):
            for fname in files:
                if fname.lower().endswith(tuple(ext.lower() for ext in min_exts)):
                    self.paths.append(os.path.join(root, fname))
        if limit:
            self.paths = self.paths[:limit]
        valid = []
        for p in self.paths:
            try:
                with Image.open(p) as im:
                    im.verify()
                valid.append(p)
            except (OSError, UnidentifiedImageError):
                continue
        self.paths = valid
        if len(self.paths) == 0:
            raise RuntimeError(f"No valid PNG images found under {root_dir}")
        random.shuffle(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx % len(self.paths)]
        with Image.open(p) as img:
            img = img.convert("RGB")
            if not resize_long_side or resize_long_side <= 0:
                return img
            w, h = img.size
            long = max(w, h)
            if long <= resize_long_side:
                return img
            scale = resize_long_side / float(long)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            return img.resize((new_w, new_h), Image.LANCZOS)

def random_crop(img, sz):
    w, h = img.size
    if w < sz or h < sz:
        img = img.resize((max(sz, w), max(sz, h)), Image.LANCZOS)
    x = random.randint(0, max(1, img.width - sz))
    y = random.randint(0, max(1, img.height - sz))
    return img.crop((x, y, x + sz, y + sz))

tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = PngFolderDataset(ds_path, min_exts=('.png',), resolution=high_resolution, limit=limit)
if len(dataset) < batch_size:
    raise RuntimeError(f"Not enough valid images ({len(dataset)}) to form a batch of size {batch_size}")

def collate_fn(batch):
    imgs = []
    for img in batch:
        img = random_crop(img, high_resolution)
        imgs.append(tfm(img))
    return torch.stack(imgs)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True
)

# --------------------------- Оптимизатор ---------------------------
def get_param_groups(module, weight_decay=0.001):
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "ln_1.weight", "ln_f.weight"]
    decay_params, no_decay_params = [], []
    for n, p in vae.named_parameters():  # глобально по vae, с фильтром requires_grad
        if not p.requires_grad:
            continue
        if any(nd in n for nd in no_decay):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

def get_param_groups(module, weight_decay=0.001):
    no_decay_tokens = ("bias", "norm", "rms", "layernorm")
    decay_params, no_decay_params = [], []
    for n, p in module.named_parameters():
        if not p.requires_grad:
            continue
        n_l = n.lower()
        if any(t in n_l for t in no_decay_tokens):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

def create_optimizer(name, param_groups):
    if name == "adam8bit":
        return bnb.optim.AdamW8bit(param_groups, lr=base_learning_rate, betas=(0.9, beta2), eps=eps)
    raise ValueError(name)

param_groups = get_param_groups(get_core_model(vae), weight_decay=0.001)
optimizer = create_optimizer(optimizer_type, param_groups)

# --------------------------- LR schedule ---------------------------
batches_per_epoch = len(dataloader)
steps_per_epoch = int(math.ceil(batches_per_epoch / float(gradient_accumulation_steps)))
total_steps = steps_per_epoch * num_epochs
print(f"[INFO] {batches_per_epoch} batches per epoch, {steps_per_epoch} optimizer steps per epoch, {total_steps} total steps sample interval {total_steps // sample_interval_share}")

def lr_lambda(step):
    if not use_decay:
        return 1.0
    x = float(step) / float(max(1, total_steps))
    warmup = float(warmup_percent)
    min_ratio = float(min_learning_rate) / float(base_learning_rate)
    if x < warmup:
        return min_ratio + (1.0 - min_ratio) * (x / warmup)
    decay_ratio = (x - warmup) / (1.0 - warmup)
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * decay_ratio))

scheduler = LambdaLR(optimizer, lr_lambda)

# Подготовка
dataloader, vae, optimizer, scheduler = accelerator.prepare(dataloader, vae, optimizer, scheduler)
trainable_params = [p for p in vae.parameters() if p.requires_grad]

# --------------------------- LPIPS и вспомогательные ---------------------------
_lpips_net = None
def _get_lpips():
    global _lpips_net
    if _lpips_net is None:
        net = lpips.LPIPS(net=lpips_backbone, verbose=False)
        net = net.to(accelerator.device).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        _lpips_net = net
    return _lpips_net

def _prepare_for_lpips(pred, target):
    if not lpips_eval_resolution or lpips_eval_resolution <= 0:
        return pred, target
    size = (lpips_eval_resolution, lpips_eval_resolution)
    if pred.shape[-2:] != size:
        pred = F.interpolate(pred, size=size, mode="bilinear", align_corners=False)
    if target.shape[-2:] != size:
        target = F.interpolate(target, size=size, mode="bilinear", align_corners=False)
    return pred, target

def compute_lpips_loss(pred, target):
    pred_ds, target_ds = _prepare_for_lpips(pred, target)
    return _get_lpips()(pred_ds, target_ds)


_sobel_kx = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], dtype=torch.float32)
_sobel_ky = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], dtype=torch.float32)
def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    C = x.shape[1]
    kx = _sobel_kx.to(x.device, x.dtype).repeat(C, 1, 1, 1)
    ky = _sobel_ky.to(x.device, x.dtype).repeat(C, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=C)
    gy = F.conv2d(x, ky, padding=1, groups=C)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)

class MedianLossNormalizer:
    def __init__(self, desired_ratios: dict, window_steps: int):
        s = sum(desired_ratios.values())
        self.ratios = {k: (v / s) if s > 0 else 0.0 for k, v in desired_ratios.items()}
        self.buffers = {k: deque(maxlen=window_steps) for k in self.ratios.keys()}
        self.window = window_steps

    def update_and_total(self, abs_losses: dict):
        for k, v in abs_losses.items():
            if k in self.buffers:
                self.buffers[k].append(float(v.detach().abs().cpu()))
        meds = {k: (np.median(self.buffers[k]) if len(self.buffers[k]) > 0 else 1.0) for k in self.buffers}
        coeffs = {k: (self.ratios[k] / max(meds[k], 1e-12)) for k in self.ratios}
        total = sum(coeffs[k] * abs_losses[k] for k in abs_losses if k in coeffs)
        return total, coeffs, meds

if full_training and not train_decoder_only:
    loss_ratios["kl"] = float(kl_ratio)
normalizer = MedianLossNormalizer(loss_ratios, median_coeff_steps)

# --------------------------- Сэмплы ---------------------------
@torch.no_grad()
def get_fixed_samples(n=3):
    idx = random.sample(range(len(dataset)), min(n, len(dataset)))
    pil_imgs = [dataset[i] for i in idx]
    tensors = []
    for img in pil_imgs:
        img = random_crop(img, high_resolution)
        tensors.append(tfm(img))
    return torch.stack(tensors).to(accelerator.device, dtype)

fixed_samples = get_fixed_samples()

@torch.no_grad()
def _to_pil_uint8(img_tensor: torch.Tensor) -> Image.Image:
    arr = ((img_tensor.float().clamp(-1, 1) + 1.0) * 127.5).clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray(arr)

@profile
@torch.no_grad()
@profile
@torch.no_grad()
def generate_and_save_samples(step=None):
    try:
        temp_vae = accelerator.unwrap_model(vae).eval()
        with torch.no_grad():
            orig_high = fixed_samples
            orig_low = F.interpolate(
                orig_high,
                size=(model_resolution, model_resolution),
                mode="bilinear",
                align_corners=False
            )
            model_dtype = next(temp_vae.parameters()).dtype
            orig_low = orig_low.to(dtype=model_dtype)

            if is_video_vae(temp_vae):
                x_in = orig_low.unsqueeze(2)           # [B,3,1,H,W]
                enc = temp_vae.encode(x_in)
                latents_mean = enc.latent_dist.mean
                dec = temp_vae.decode(latents_mean).sample  # [B,3,1,H,W]
                rec = dec.squeeze(2)                   # [B,3,H,W]
            else:
                enc = temp_vae.encode(orig_low)
                latents_mean = enc.latent_dist.mean
                rec = temp_vae.decode(latents_mean).sample

        if rec.shape[-2:] != orig_high.shape[-2:]:
            rec = F.interpolate(
                rec, size=orig_high.shape[-2:], 
                mode="bilinear", align_corners=False
            )

        # Сохраняем первые две для quick-check
        first_real = _to_pil_uint8(orig_high[0])
        first_dec  = _to_pil_uint8(rec[0])
        first_real.save(f"{generated_folder}/sample_real.jpg", quality=95)
        first_dec.save(f"{generated_folder}/sample_decoded.jpg", quality=95)

        # Сохраняем все по отдельности (как раньше)
        for i in range(rec.shape[0]):
            _to_pil_uint8(rec[i]).save(f"{generated_folder}/sample_{i}.jpg", quality=95)

        # === NEW: создаём парный грид (real vs decoded) ===
        n = rec.shape[0]
        fig, axes = plt.subplots(2, n, figsize=(n * 2.5, 6))  # 2 строки (real/decoded)
        
        if n == 1:
            axes = np.array([[axes[0]], [axes[1]]])  # фиксим для случая batch_size=1

        for idx in range(n):
            real_img = _to_pil_uint8(orig_high[idx])
            rec_img  = _to_pil_uint8(rec[idx])

            axes[0, idx].imshow(real_img)
            axes[0, idx].axis("off")
            axes[0, idx].set_title(f"Real {idx}")

            axes[1, idx].imshow(rec_img)
            axes[1, idx].axis("off")
            axes[1, idx].set_title(f"Decoded {idx}")

        plt.tight_layout()
        fig.savefig(f"{generated_folder}/samples_pairs.jpg", dpi=150)
        plt.close(fig)

        # === LPIPS ===
        lpips_scores = []
        for i in range(rec.shape[0]):
            orig_full = orig_high[i:i+1].to(torch.float32)
            rec_full  = rec[i:i+1].to(torch.float32)
            if rec_full.shape[-2:] != orig_full.shape[-2:]:
                rec_full = F.interpolate(
                    rec_full, size=orig_full.shape[-2:], 
                    mode="bilinear", align_corners=False
                )
            lpips_val = compute_lpips_loss(rec_full, orig_full).mean().item()
            lpips_scores.append(lpips_val)
        avg_lpips = float(np.mean(lpips_scores))

        if use_wandb and accelerator.is_main_process:
            wandb.log({"lpips_mean": avg_lpips}, step=step)
            wandb.log({
                "samples/pairs": wandb.Image(f"{generated_folder}/samples_pairs.jpg", caption=f"{n} pairs (real vs decoded)")
            }, step=step)

    finally:
        gc.collect()
        torch.cuda.empty_cache()


if accelerator.is_main_process and save_model:
    print("Генерация сэмплов до старта обучения...")
    generate_and_save_samples(0)

accelerator.wait_for_everyone()

# --------------------------- Тренировка ---------------------------
@profile
def run_training():
    progress = tqdm(total=total_steps, disable=not accelerator.is_local_main_process)
    global_step = 0
    min_loss = float("inf")
    sample_interval = GLOBAL_SAMPLE_INTERVAL

    for epoch in range(num_epochs):
        vae.train()
        batch_losses, batch_grads = [], []
        track_losses = {k: [] for k in loss_ratios.keys()}

        for imgs in dataloader:
            with accelerator.accumulate(vae):
                imgs = imgs.to(accelerator.device)

                if high_resolution != model_resolution:
                    imgs_low = F.interpolate(imgs, size=(model_resolution, model_resolution), mode="bilinear", align_corners=False)
                else:
                    imgs_low = imgs

                model_dtype = next(vae.parameters()).dtype
                imgs_low_model = imgs_low.to(dtype=model_dtype) if imgs_low.dtype != model_dtype else imgs_low

                encode_input = imgs_low_model.unsqueeze(2) if is_video_vae(vae) else imgs_low_model
                encode_ctx = torch.no_grad() if train_decoder_only else nullcontext()
                with encode_ctx:
                    enc = vae.encode(encode_input)
                latents_dist = enc.latent_dist
                latents = latents_dist.mean if train_decoder_only else latents_dist.sample()
                if train_decoder_only:
                    latents = latents.detach()

                if is_video_vae(vae):
                    dec = vae.decode(latents).sample
                    rec = dec.squeeze(2)
                else:
                    rec = vae.decode(latents).sample

                if train_decoder_only:
                    del enc

                if rec.shape[-2:] != imgs.shape[-2:]:
                    rec = F.interpolate(rec, size=imgs.shape[-2:], mode="bilinear", align_corners=False)

                rec_f32 = rec.to(torch.float32)
                imgs_f32 = imgs.to(torch.float32)

                lpips_loss = compute_lpips_loss(rec_f32, imgs_f32).mean()
                abs_losses = {
                    "mae":   F.l1_loss(rec_f32, imgs_f32),
                    "mse":   F.mse_loss(rec_f32, imgs_f32),
                    "lpips": lpips_loss,
                    "edge":  F.l1_loss(sobel_edges(rec_f32), sobel_edges(imgs_f32)),
                }

                if full_training and not train_decoder_only:
                    mean = enc.latent_dist.mean
                    logvar = enc.latent_dist.logvar
                    kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
                    abs_losses["kl"] = kl
                else:
                    abs_losses["kl"] = torch.tensor(0.0, device=accelerator.device, dtype=torch.float32)

                total_loss, coeffs, meds = normalizer.update_and_total(abs_losses)

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    raise RuntimeError("NaN/Inf loss")

                accelerator.backward(total_loss)

                grad_norm = torch.tensor(0.0, device=accelerator.device)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(trainable_params, clip_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    progress.update(1)

                if accelerator.is_main_process:
                    try:
                        current_lr = optimizer.param_groups[0]["lr"]
                    except Exception:
                        current_lr = scheduler.get_last_lr()[0]

                    batch_losses.append(total_loss.detach().item())
                    batch_grads.append(float(grad_norm.detach().cpu().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm))
                    for k, v in abs_losses.items():
                        track_losses[k].append(float(v.detach().item()))

                    if use_wandb and accelerator.sync_gradients:
                        log_dict = {
                            "total_loss": float(total_loss.detach().item()),
                            "learning_rate": current_lr,
                            "epoch": epoch,
                            "grad_norm": batch_grads[-1],
                        }
                        for k, v in abs_losses.items():
                            log_dict[f"loss_{k}"] = float(v.detach().item())
                        for k in coeffs:
                            log_dict[f"coeff_{k}"] = float(coeffs[k])
                            log_dict[f"median_{k}"] = float(meds[k])
                        wandb.log(log_dict, step=global_step)

                if global_step > 0 and global_step % sample_interval == 0:
                    if accelerator.is_main_process:
                        generate_and_save_samples(global_step)
                    accelerator.wait_for_everyone()

                    n_micro = sample_interval * gradient_accumulation_steps
                    if batch_losses:
                        avg_loss = float(np.mean(batch_losses[-n_micro:])) if len(batch_losses) >= n_micro else float(np.mean(batch_losses))
                    else:
                        avg_loss = float("nan")
                    if batch_grads:
                        avg_grad = float(np.mean(batch_grads[-n_micro:])) if len(batch_grads) >= 1 else float(np.mean(batch_grads))
                    else:
                        avg_grad = 0.0

                    if accelerator.is_main_process:
                        print(f"Epoch {epoch} step {global_step} loss: {avg_loss:.6f}, grad_norm: {avg_grad:.6f}, lr: {current_lr:.9f}")
                        if save_model and avg_loss < min_loss * save_barrier:
                            min_loss = avg_loss
                            accelerator.unwrap_model(vae).save_pretrained(save_as)
                        if use_wandb:
                            wandb.log({"interm_loss": avg_loss, "interm_grad": avg_grad}, step=global_step)

        if accelerator.is_main_process:
            epoch_avg = float(np.mean(batch_losses)) if batch_losses else float("nan")
            print(f"Epoch {epoch} done, avg loss {epoch_avg:.6f}")
            if use_wandb:
                wandb.log({"epoch_loss": epoch_avg, "epoch": epoch + 1}, step=global_step)

    if accelerator.is_main_process:
        print("Training finished – saving final model")
        if save_model:
            accelerator.unwrap_model(vae).save_pretrained(save_as)

    accelerator.free_memory()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    print("Готово!")

run_training()
