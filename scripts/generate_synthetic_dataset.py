import torch
import argparse
from diffusers import StableDiffusionPipeline
from PIL import Image
from pathlib import Path
import torchvision.transforms as T
from huggingface_hub import login
import re

# Import the Accelerator class
from accelerate import Accelerator

# A dictionary to map user-friendly names to Hugging Face model IDs.
MODEL_MAPPING = {
    "sd1.5": "runwayml/stable-diffusion-v1-5",
    "flux": "black-forest-labs/FLUX.1-schnell",
    "sd3": "stabilityai/stable-diffusion-3-medium",
}


def setup_directories(
    base_path: Path, val_set_name: str, model_name: str, high_res: int, low_res: int
):
    """Creates the necessary directory structure for storing the validation data."""
    model_path = base_path / val_set_name / model_name
    paths = {
        "high_res_img": model_path / "images" / str(high_res),
        "low_res_img": model_path / "images" / str(low_res),
        "high_res_emb": model_path / "embeddings" / str(high_res),
        "low_res_emb": model_path / "embeddings" / str(low_res),
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def generate_and_process_pair(
    pipe: StableDiffusionPipeline,
    prompt: str,
    output_paths: dict,
    high_res_size: int,
    low_res_size: int,
    accelerator: Accelerator,  # Pass the accelerator for its device
):
    """
    Generates a high-res image, creates a low-res version, encodes both to
    the latent space, and saves all outputs to the specified directories.
    """
    # Use the accelerator's logging for process-aware printing
    accelerator.print(f"\nProcessing prompt: '{prompt}'")

    # --- 1. Generate High-Resolution Image ---
    # The generator seed should be consistent across devices for reproducibility
    generator = torch.Generator(device="cpu").manual_seed(42)
    high_res_image_pil = pipe(
        prompt=prompt,
        height=high_res_size,
        width=high_res_size,
        num_inference_steps=50,
        generator=generator,
    ).images[0]
    accelerator.print(f"Generated {high_res_size}x{high_res_size} image.")

    # --- 2. Create Low-Resolution Image via Downscaling ---
    low_res_image_pil = high_res_image_pil.resize(
        (low_res_size, low_res_size), Image.Resampling.BICUBIC
    )
    accelerator.print(f"Downscaled image to {low_res_size}x{low_res_size}.")

    # --- 3. Prepare Images for VAE Encoding ---
    transform = T.Compose(
        [T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    )

    image_dtype = pipe.vae.dtype
    high_res_tensor = (
        transform(high_res_image_pil)
        .unsqueeze(0)
        .to(accelerator.device, dtype=image_dtype)
    )
    low_res_tensor = (
        transform(low_res_image_pil)
        .unsqueeze(0)
        .to(accelerator.device, dtype=image_dtype)
    )

    # --- 4. Encode Images to Latent Space ---
    accelerator.print("Encoding images into latents...")
    with torch.no_grad():
        vae = pipe.vae
        high_res_latents = vae.encode(high_res_tensor).latent_dist.sample()
        low_res_latents = vae.encode(low_res_tensor).latent_dist.sample()

        scaling_factor = vae.config.scaling_factor
        high_res_latents = high_res_latents * scaling_factor
        low_res_latents = low_res_latents * scaling_factor
    accelerator.print("Encoding complete.")

    # --- 5. Save all artifacts ---
    prompt_slug = re.sub(r"\W+", "_", prompt).lower()[:50]

    high_res_image_pil.save(output_paths["high_res_img"] / f"{prompt_slug}.png")
    low_res_image_pil.save(output_paths["low_res_img"] / f"{prompt_slug}.png")
    accelerator.print(f"Saved images to: {output_paths['high_res_img'].parent}")

    torch.save(
        high_res_latents.cpu(), output_paths["high_res_emb"] / f"{prompt_slug}.pt"
    )
    torch.save(low_res_latents.cpu(), output_paths["low_res_emb"] / f"{prompt_slug}.pt")
    accelerator.print(f"Saved latents to: {output_paths['high_res_emb'].parent}")


def main():
    """Main function to parse arguments and run the data generation process."""
    parser = argparse.ArgumentParser(
        description="Generate validation data for latent super-resolution."
    )
    # (the rest of the arguments are the same)
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        required=True,
        help="A list of short model names to use for generation (e.g., sd1.5).",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        required=True,
        help="A list of prompts to generate images from.",
    )
    parser.add_argument(
        "--validation_set_name",
        type=str,
        default="Synth_val_1",
        help="Name of the top-level folder for the validation set.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./validation_data",
        help="The base directory where the validation set will be saved.",
    )
    parser.add_argument(
        "--high_res_size",
        type=int,
        default=512,
        help="The height and width of the high-resolution images.",
    )
    parser.add_argument(
        "--downscale_factor",
        type=int,
        default=2,
        help="The factor by which to downscale the high-res image.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Your Hugging Face Hub token for logging in.",
    )
    args = parser.parse_args()

    # Initialize the accelerator
    accelerator = Accelerator()

    if args.hf_token and accelerator.is_main_process:
        login(args.hf_token)

    # Use accelerator.device instead of manual device selection
    device = accelerator.device
    accelerator.print(f"Using device: {device}")

    # Let Accelerate handle the distribution of prompts across processes
    with accelerator.split_between_processes(args.prompts) as prompts:

        low_res_size = args.high_res_size // args.downscale_factor
        base_output_path = Path(args.output_path)

        for model_name in args.model_names:
            if model_name not in MODEL_MAPPING:
                accelerator.print(
                    f"Warning: Model '{model_name}' not in MODEL_MAPPING. Skipping."
                )
                continue

            model_id = MODEL_MAPPING[model_name]
            accelerator.print(f"\n--- Loading model: {model_name} ({model_id}) ---")

            pipe = None
            try:
                # Set torch_dtype based on device type
                torch_dtype = torch.float16 if "cuda" in str(device) else torch.float32

                pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                )

                # Disable the safety checker to prevent black images for safe prompts
                pipe.safety_checker = lambda images, clip_input: (
                    images,
                    [False] * len(images),
                )

                # Move the pipeline to the device assigned by Accelerator
                pipe = pipe.to(device)

                # Only the main process should set up directories to avoid race conditions
                if accelerator.is_main_process:
                    output_paths = setup_directories(
                        base_output_path,
                        args.validation_set_name,
                        model_name,
                        args.high_res_size,
                        low_res_size,
                    )

                # Wait for the main process to finish creating directories before others proceed
                accelerator.wait_for_everyone()

                # Re-define output_paths for all processes after they are created
                output_paths = setup_directories(
                    base_output_path,
                    args.validation_set_name,
                    model_name,
                    args.high_res_size,
                    low_res_size,
                )

                # Each process now iterates over its own subset of prompts
                for prompt in prompts:
                    generate_and_process_pair(
                        pipe=pipe,
                        prompt=prompt,
                        output_paths=output_paths,
                        high_res_size=args.high_res_size,
                        low_res_size=low_res_size,
                        accelerator=accelerator,  # Pass accelerator instance
                    )

            except Exception as e:
                accelerator.print(
                    f"An error occurred while processing model {model_name}: {e}"
                )
            finally:
                if pipe:
                    del pipe
                # Clear cache on the specific device
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if hasattr(torch, "mps") and torch.mps.is_available():
                    torch.mps.empty_cache()

    accelerator.print("\nValidation data generation complete.")


if __name__ == "__main__":
    main()
