#!/bin/bash

# ======================================================================================
# Bash script to run synthetic validation data generation for multiple models & settings
#
# USAGE:
#   ./run_generate_synthetic_dataset.sh cpu
#   ./run_generate_synthetic_dataset.sh gpu
#   ./run_generate_synthetic_dataset.sh multi-gpu
#
# Models: flux, sd3
# Resolutions: 256, 512
# ======================================================================================

# --- Configuration ---

PYTHON_SCRIPT="generate_synthetic_dataset.py"
OUTPUT_PATH="./my_validation_sets"
VAL_SET_NAME_PREFIX="Synth_val"
DOWNSCALE_FACTOR=2
# HF_TOKEN should be set as environment variable or use `huggingface-cli login`
# HF_TOKEN="token_here" # Prefer using env vars or `huggingface-cli login`

# Models and resolutions
# MODEL_NAMES=("flux" "sd3")
# HIGH_RES_SIZES=(256 512)
MODEL_NAMES=("sd1.5")
HIGH_RES_SIZES=(256)


# Prompts
PROMPTS=(
  "Ultra-real portrait of an elderly woman, 85mm lens, f/1.8, Rembrandt lighting, soft gray seamless backdrop, sharp skin texture and fine wrinkles, clear catchlights, neutral Portra-style color grading, shallow depth of field, studio quality, high dynamic range"

  "Epic fantasy vista: a dragon banking over a cliff-top stone castle at golden hour, 24mm wide angle, sunbeams through mist, volumetric light, dramatic cumulus sky, windswept banners, cinematic composition, photoreal creature texturing, crisp scale detail, high contrast"

  "Extreme macro of a honey bee gathering pollen on a vibrant flower, 100mm macro, focus stacking look, morning backlight, dewdrops sparkling, razor-thin DOF, micro-textures on wings and hairs, natural color, clean bokeh"

  "Futuristic megacity at night in the rain, rain-slick streets with neon signage reflected in puddles, aerial perspective, subtle fog, flying vehicles with motion streaks, anamorphic lens flares, towering glass and concrete, photoreal cyberpunk mood, high detail"

  "Editorial fashion portrait in studio, androgynous model in sculptural couture, beauty-dish key with soft rim light, 135mm lens, f/2, precise skin tone, sharp eyes, minimal background, high-end retouching aesthetic, magazine cover quality"

  "Luxury product shot of a stainless-steel chronograph watch on dark slate, 3-point lighting with controlled reflections, high-speed splash of water freezing mid-air, macro sharpness on bezel and dial, crisp specular highlights, pristine, advertising grade"

  "Architectural interior, minimalist Scandinavian living room, 16mm tilt-shift look, natural window light, balanced exposure, clean lines, soft textiles and light wood, perfectly verticals, subtle shadows, catalog quality"

  "Wildlife close-up of a snow leopard on a rocky ridge during snowfall, 400mm telephoto, eye-level perspective, tack-sharp whiskers and eyes, shallow DOF background, natural muted palette, crisp snowflakes, documentary realism"

  "Culinary still life: artisan ramen bowl with rich broth, steam rising, 50mm lens, moody side light, 3/4 top angle on rustic wooden table, styled props (chopsticks, napkin, scallions), saturated yet natural colors, editorial food photography"

  "Street documentary at dusk in a narrow lantern-lit alley, 35mm lens, candid passerby with umbrella, puddle reflections, soft rain, mixed tungsten and ambient light, gritty textures, authentic urban atmosphere, high ISO filmic grain"
)

# --- Run Functions ---

run_cpu() {
  echo "========================================="
  echo "🚀 Starting generation on CPU..."
  echo "========================================="
  for model in "${MODEL_NAMES[@]}"; do
    for res in "${HIGH_RES_SIZES[@]}"; do
      echo ">>> Running for model: $model | res: $res"
      python "$PYTHON_SCRIPT" \
        --model_names "$model" \
        --prompts "${PROMPTS[@]}" \
        --validation_set_name "${VAL_SET_NAME_PREFIX}_${model}_${res}" \
        --output_path "$OUTPUT_PATH" \
        --high_res_size "$res" \
        --downscale_factor "$DOWNSCALE_FACTOR" \
        --hf_token "$HF_TOKEN"
    done
  done
}

run_gpu() {
  echo "========================================="
  echo "🚀 Starting generation on single GPU..."
  echo "========================================="
  for model in "${MODEL_NAMES[@]}"; do
    for res in "${HIGH_RES_SIZES[@]}"; do
      echo ">>> Running for model: $model | res: $res"
      python "$PYTHON_SCRIPT" \
        --model_names "$model" \
        --prompts "${PROMPTS[@]}" \
        --validation_set_name "${VAL_SET_NAME_PREFIX}_${model}_${res}" \
        --output_path "$OUTPUT_PATH" \
        --high_res_size "$res" \
        --downscale_factor "$DOWNSCALE_FACTOR" \
        --hf_token "$HF_TOKEN"
    done
  done
}

run_multi_gpu() {
  echo "==========================================="
  echo "🚀 Starting generation on MULTIPLE GPUs..."
  echo "==========================================="
  for model in "${MODEL_NAMES[@]}"; do
    for res in "${HIGH_RES_SIZES[@]}"; do
      echo ">>> Running for model: $model | res: $res"
      accelerate launch "$PYTHON_SCRIPT" \
        --model_names "$model" \
        --prompts "${PROMPTS[@]}" \
        --validation_set_name "${VAL_SET_NAME_PREFIX}_${model}_${res}" \
        --output_path "$OUTPUT_PATH" \
        --high_res_size "$res" \
        --downscale_factor "$DOWNSCALE_FACTOR" \
        --hf_token "$HF_TOKEN"
    done
  done
}

# --- Main ---

if [ -z "$1" ]; then
  echo "Usage: $0 {cpu|gpu|multi-gpu}"
  exit 1
fi

case "$1" in
  cpu) run_cpu ;;
  gpu) run_gpu ;;
  multi-gpu) run_multi_gpu ;;
  *) echo "Error: Invalid argument '$1'. Use {cpu|gpu|multi-gpu}" ; exit 1 ;;
esac

echo "✅ All runs completed successfully."
