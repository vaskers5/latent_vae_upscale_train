from training.models.swinir import SwinIR
import torch

def load_pixel_sr_model(device, sr_checkpoint_path):
    """Load pretrained SwinIR for pixel super-resolution"""
    print(f"Loading pixel SR model from {sr_checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(sr_checkpoint_path, map_location=device, weights_only=False)

    # Create SwinIR model using the original network_swinir.py initialization
    # Use img_size=48 to match the DIV2K_s48w8 checkpoint
    model = SwinIR(
        upscale=2,  # 2x upscaling
        in_chans=16,  # RGB input
        # img_size=48,  # Training patch size (matches DIV2K_s48w8)
        window_size=8,  # Window size
        img_range=1.0,  # Image range
        depths=[6, 6, 6, 6, 6, 6],  # Depths for each stage (6 stages)
        embed_dim=180,  # Embedding dimension
        num_heads=[6, 6, 6, 6, 6, 6],  # Number of attention heads (6 stages)
        mlp_ratio=2,  # MLP ratio
        upsampler="pixelshuffle",  # Upsampler
        resi_connection="1conv",  # Residual connection
    ).to(device)

    # # Load weights using the standard param_key_g = 'params'
    # param_key_g = "params"
    # if param_key_g in checkpoint:
    #     model.load_state_dict(checkpoint[param_key_g], strict=True)
    # elif "params_ema" in checkpoint:
    #     model.load_state_dict(checkpoint["params_ema"], strict=True)
    # else:
    #     model.load_state_dict(checkpoint, strict=True)

    model.eval()
    print("✓ Loaded pixel SR model")
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr_checkpoint_path = "pretrained_weights/pytorch_model.pth"
    model = load_pixel_sr_model(device, sr_checkpoint_path)
    test_sample = "unpacked_original_ds/full_dataset/cache_vae_embeddings/ostris_vae/128px/pexels_photo_japnf/00/006724614bd65ba8d4921c9f0c4b322f.pt"
    with torch.no_grad():
        output = model(torch.load(test_sample)['latents'].to(device))
    print(f"Input shape: {input_tensor.shape}, Output shape: {output.shape}")
    # Example usage: model(input_tensor)