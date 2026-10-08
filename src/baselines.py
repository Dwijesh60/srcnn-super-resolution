"""
Traditional Interpolation Baselines:
- Nearest Neighbor
- Bilinear
- Bicubic
"""

import numpy as np
import pandas as pd
from PIL import Image
from src.metrics import compute_mse, compute_psnr, compute_ssim

def downsample_upsample_interpolation(image_np, scale_factor=2, method="bicubic"):
    """
    Simulates resolution degradation and classical interpolation reconstruction:
    1. Downsamples input high-resolution image by scale_factor using bicubic filter.
    2. Upsamples the low-resolution intermediate image back to target size using the specified method.
    """
    # Convert numpy [0.0, 1.0] to uint8 PIL Image
    img_uint8 = (np.clip(image_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    
    orig_w, orig_h = pil_img.size
    lr_w, lr_h = max(1, orig_w // scale_factor), max(1, orig_h // scale_factor)
    
    # Degradation: downsample using bicubic (standard SISR degradation model)
    lr_img = pil_img.resize((lr_w, lr_h), Image.Resampling.BICUBIC)
    
    # Upsample using chosen interpolation
    interp_map = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC
    }
    resampling_mode = interp_map.get(method.lower(), Image.Resampling.BICUBIC)
    sr_img = lr_img.resize((orig_w, orig_h), resampling_mode)
    
    # Return normalized numpy array [0.0, 1.0]
    return np.array(sr_img, dtype=np.float32) / 255.0

def evaluate_baseline_method(hr_images, method="bicubic", scale_factor=2):
    """
    Evaluates a single baseline interpolation method across a collection of images.
    Returns average MSE, PSNR, and SSIM.
    """
    mse_list = []
    psnr_list = []
    ssim_list = []
    
    for hr in hr_images:
        sr = downsample_upsample_interpolation(hr, scale_factor=scale_factor, method=method)
        mse_list.append(compute_mse(hr, sr))
        psnr_list.append(compute_psnr(hr, sr))
        ssim_list.append(compute_ssim(hr, sr))
        
    return {
        "Method": method.capitalize(),
        "MSE": float(np.mean(mse_list)),
        "PSNR (dB)": float(np.mean(psnr_list)),
        "SSIM": float(np.mean(ssim_list))
    }

def evaluate_all_baselines(hr_images, scale_factor=2):
    """
    Evaluates Nearest Neighbor, Bilinear, and Bicubic baselines.
    Returns a pandas DataFrame summary.
    """
    results = []
    for method in ["nearest", "bilinear", "bicubic"]:
        res = evaluate_baseline_method(hr_images, method=method, scale_factor=scale_factor)
        results.append(res)
    df = pd.DataFrame(results)
    return df
