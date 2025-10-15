#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate 1024×1024 SDXL images for the validation set prompts.

The script reads prompts from `captioned_validation.csv` (using the `sdxl`
caption by default) and renders them with a list of SDXL checkpoints. Each
checkpoint can choose its own scheduler, guidance scale, and negative prompt.
Outputs are written under `sdxl_validation/<model-slug>/` as PNG files plus a
small JSON sidecar with the generation metadata. Launch via `accelerate launch`
to fan prompts across multiple GPUs.

"""

from __future__ import annotations
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from accelerate import Accelerator
from diffusers import (
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
)
from tqdm.auto import tqdm


IMAGE_SIZE = 1024


def _default_dtype(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type == "cuda" else torch.float32


def _slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


@dataclass
class ModelConfig:
    name: str
    model_id: str
    scheduler: str
    num_inference_steps: int
    guidance_scale: float
    negative_prompt: Optional[str] = None
    scheduler_kwargs: Dict[str, object] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return _slugify(self.name)


MODEL_CONFIGS: List[ModelConfig] = [
    ModelConfig(
        name="RealVisXL V5.0",
        model_id="SG161222/RealVisXL_V5.0",
        scheduler="dpmpp-sde-karras",
        num_inference_steps=40,
        guidance_scale=5.5,
        negative_prompt=(
            "bad hands, bad anatomy, ugly, deformed, (face asymmetry, eyes asymmetry, "
            "deformed eyes, deformed mouth, open mouth)"
        ),
        scheduler_kwargs={"use_karras_sigmas": True},
    ),
    ModelConfig(
        name="Juggernaut XL v9",
        model_id="RunDiffusion/Juggernaut-XL-v9",
        scheduler="dpmpp-2m-karras",
        num_inference_steps=35,
        guidance_scale=4.0,
        negative_prompt=None,
        scheduler_kwargs={"use_karras_sigmas": True, "algorithm_type": "dpmsolver++"},
    ),
    ModelConfig(
        name="Obsession IllustriousXL v10",
        model_id="John6666/obsession-illustriousxl-v10-sdxl",
        scheduler="dpmpp-2m-karras",
        num_inference_steps=25,
        guidance_scale=6.0,
        negative_prompt=None,
        scheduler_kwargs={"use_karras_sigmas": True, "algorithm_type": "dpmsolver++"},
    ),
]


class PromptLoadError(RuntimeError):
    """Raised when a CSV row is missing the desired caption."""


CSV_PATH = Path("captioned_validation.csv")
PROMPT_KEY = "sdxl"
OUTPUT_DIR = Path("sdxl_validation")
LIMIT: Optional[int] = None
SEED: Optional[int] = None
DEVICE_OVERRIDE: Optional[str] = None  # e.g. "cuda:0" to override accelerator selection
MODEL_SLUGS: Optional[Sequence[str]] = None  # e.g. ["realvisxl-v5-0"]
SKIP_EXISTING = False
SAVE_METADATA = False
DISABLE_XFORMERS = False


def load_prompts(csv_path: Path, prompt_key: str, limit: Optional[int]) -> List[Tuple[str, str]]:
    prompts: List[Tuple[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "gpt_caption" not in reader.fieldnames or "img_path" not in reader.fieldnames:
            raise PromptLoadError("CSV must contain 'img_path' and 'gpt_caption' columns.")

        for row_idx, row in enumerate(reader):
            try:
                captions = json.loads(row["gpt_caption"])
            except json.JSONDecodeError as exc:
                raise PromptLoadError(f"Row {row_idx}: unable to parse `gpt_caption`.") from exc

            caption = captions.get(prompt_key)
            if not caption:
                raise PromptLoadError(f"Row {row_idx}: prompt key '{prompt_key}' not found.")

            image_id = Path(row["img_path"]).stem
            prompts.append((image_id, caption))

            if limit is not None and len(prompts) >= limit:
                break

    if not prompts:
        raise PromptLoadError("No prompts were loaded from the CSV.")

    return prompts


def _create_scheduler(pipe: DiffusionPipeline, name: str, kwargs: Dict[str, object]):
    name = name.lower()
    config = pipe.scheduler.config
    if name == "dpmpp-sde-karras":
        return DPMSolverSDEScheduler.from_config(config, **kwargs)
    if name == "dpmpp-2m-karras":
        return DPMSolverMultistepScheduler.from_config(config, **kwargs)
    if name == "euler-a":
        return EulerAncestralDiscreteScheduler.from_config(config, **kwargs)
    raise ValueError(f"Unsupported scheduler '{name}'.")


def _prepare_pipeline(
    config: ModelConfig,
    device: torch.device,
    dtype: torch.dtype,
    disable_xformers: bool,
) -> DiffusionPipeline:
    pipe = DiffusionPipeline.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )
    pipe.scheduler = _create_scheduler(pipe, config.scheduler, config.scheduler_kwargs)
    pipe.to(device)
    if device.type == "cuda" and not disable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as exc:  # pragma: no cover - best-effort hook
            print(f"[warn] xFormers not enabled: {exc}")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _generator(device: torch.device, seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    gen = torch.Generator(device=str(device))
    gen.manual_seed(seed)
    return gen


def generate_for_model(
    accelerator: Accelerator,
    pipe: DiffusionPipeline,
    config: ModelConfig,
    prompts: Sequence[Tuple[str, str]],
    indices: Sequence[int],
    output_dir: Path,
    *,
    device: torch.device,
    base_seed: Optional[int],
    skip_existing: bool,
    save_metadata: bool,
) -> None:
    model_dir = output_dir / config.slug
    if accelerator.is_main_process:
        model_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    if not indices:
        return

    for global_idx in tqdm(
        indices,
        desc=config.slug,
        total=len(indices),
        disable=not accelerator.is_main_process,
    ):
        image_id, prompt = prompts[global_idx]
        target_path = model_dir / f"{image_id}.png"
        if skip_existing and target_path.exists():
            continue

        seed = None if base_seed is None else base_seed + global_idx
        generator = _generator(device, seed)
        result = pipe(
            prompt=prompt,
            negative_prompt=config.negative_prompt,
            width=IMAGE_SIZE,
            height=IMAGE_SIZE,
            guidance_scale=config.guidance_scale,
            num_inference_steps=config.num_inference_steps,
            generator=generator,
        )
        image = result.images[0]
        image.save(target_path)

        if save_metadata:
            meta = {
                "model_name": config.name,
                "model_id": config.model_id,
                "scheduler": config.scheduler,
                "num_inference_steps": config.num_inference_steps,
                "guidance_scale": config.guidance_scale,
                "negative_prompt": config.negative_prompt,
                "prompt": prompt,
                "seed": seed,
                "width": IMAGE_SIZE,
                "height": IMAGE_SIZE,
            }
            with target_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
                json.dump(meta, handle, indent=2)


def main() -> None:
    accelerator = Accelerator()

    device = torch.device(DEVICE_OVERRIDE) if DEVICE_OVERRIDE else accelerator.device
    dtype = _default_dtype(device)

    prompts = load_prompts(CSV_PATH, PROMPT_KEY, LIMIT)
    accelerator.print(f"[info] Loaded {len(prompts)} prompts using key '{PROMPT_KEY}'.")

    selected_slugs = set(MODEL_SLUGS or [])
    configs = [cfg for cfg in MODEL_CONFIGS if not selected_slugs or cfg.slug in selected_slugs]

    missing = selected_slugs - {cfg.slug for cfg in configs}
    if missing:
        raise ValueError(f"Unknown model slugs requested: {', '.join(sorted(missing))}.")

    if not configs:
        raise ValueError("No model configurations selected.")

    if accelerator.is_main_process:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    local_indices = list(range(accelerator.process_index, len(prompts), accelerator.num_processes))
    accelerator.print(
        f"[info] num_processes={accelerator.num_processes}, "
        f"rank={accelerator.process_index}, prompts_assigned={len(local_indices)}."
    )

    for config in configs:
        accelerator.print(f"[info] Loading {config.name} ({config.model_id})")
        pipe = _prepare_pipeline(
            config,
            device=device,
            dtype=dtype,
            disable_xformers=DISABLE_XFORMERS,
        )
        try:
            generate_for_model(
                accelerator,
                pipe,
                config,
                prompts,
                local_indices,
                OUTPUT_DIR,
                device=device,
                base_seed=SEED,
                skip_existing=SKIP_EXISTING,
                save_metadata=SAVE_METADATA,
            )
        finally:
            pipe.to("cpu")
            del pipe
            if device.type == "cuda":
                torch.cuda.empty_cache()
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
