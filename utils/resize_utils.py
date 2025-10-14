"""
Wrapper utilities for ResizeRight library to replace PIL and torch interpolation.

This module provides convenient functions to use ResizeRight for both PIL Images
and PyTorch tensors, ensuring high-quality, consistent resizing across the codebase.
"""

from typing import Tuple, Optional, Union
import numpy as np
import torch
from PIL import Image

from .resize_right import resize
from . import interp_methods


def pil_resize_right(
    img: Image.Image,
    size: Tuple[int, int],
    interp_method=interp_methods.cubic,
    antialiasing: bool = True,
) -> Image.Image:
    """
    Resize a PIL Image using ResizeRight.
    
    Args:
        img: Input PIL Image
        size: Target size as (width, height)
        interp_method: Interpolation method (cubic, linear, lanczos2, lanczos3, box)
        antialiasing: Whether to apply antialiasing for downsampling
        
    Returns:
        Resized PIL Image
    """
    # Convert PIL to numpy array
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # ResizeRight expects (H, W, C) for numpy
    target_h, target_w = size[1], size[0]
    
    # Perform resize
    resized_array = resize(
        img_array,
        out_shape=(target_h, target_w),
        interp_method=interp_method,
        antialiasing=antialiasing,
    )
    
    # Convert back to PIL
    resized_array = np.clip(resized_array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(resized_array)


def torch_resize_right(
    tensor: torch.Tensor,
    size: Union[int, Tuple[int, int]],
    interp_method=interp_methods.cubic,
    antialiasing: bool = True,
) -> torch.Tensor:
    """
    Resize a PyTorch tensor using ResizeRight.
    
    Args:
        tensor: Input tensor of shape (C, H, W) or (B, C, H, W)
        size: Target size as int (for square) or (height, width)
        interp_method: Interpolation method (cubic, linear, lanczos2, lanczos3, box)
        antialiasing: Whether to apply antialiasing for downsampling
        
    Returns:
        Resized tensor with same number of dimensions as input
    """
    if isinstance(size, int):
        target_h = target_w = size
    else:
        target_h, target_w = size
    
    # Handle both 3D and 4D tensors
    is_3d = tensor.ndim == 3
    if is_3d:
        tensor = tensor.unsqueeze(0)  # Add batch dimension
    
    # ResizeRight expects (B, C, H, W) for torch
    resized = resize(
        tensor,
        out_shape=(target_h, target_w),
        interp_method=interp_method,
        antialiasing=antialiasing,
    )
    
    if is_3d:
        resized = resized.squeeze(0)  # Remove batch dimension
    
    return resized


def get_interp_method(method_name: str):
    """
    Get interpolation method by name.
    
    Args:
        method_name: One of 'cubic', 'linear', 'lanczos2', 'lanczos3', 'box'
        
    Returns:
        Interpolation method function
    """
    methods = {
        'cubic': interp_methods.cubic,
        'bicubic': interp_methods.cubic,
        'linear': interp_methods.linear,
        'bilinear': interp_methods.linear,
        'lanczos': interp_methods.lanczos3,
        'lanczos2': interp_methods.lanczos2,
        'lanczos3': interp_methods.lanczos3,
        'box': interp_methods.box,
    }
    return methods.get(method_name.lower(), interp_methods.cubic)
