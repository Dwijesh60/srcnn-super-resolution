"""
Inference and demonstration module for single-image super-resolution.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from src.metrics import compute_psnr, compute_ssim, compute_mse

def super_resolve_image(
    image_input,
    model,
    scale_factor=2,
    target_size=(224, 224),
    show_plot=True,
    save_path=None
):
    """
    Demonstrates single-image super-resolution:
    1. Loads or accepts an image
    2. Resizes target HR to target_size
    3. Simulates degradation (downsampling by scale_factor, upsampling bicubic)
    4. Passes through trained SRCNN
    5. Displays side-by-side comparison with quantitative metrics
    """
    # Accept either file path or PIL Image / numpy array
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray((np.clip(image_input, 0, 1) * 255).astype(np.uint8))
    else:
        raise ValueError("Unsupported input format for inference.")
        
    # Center crop if aspect ratio is not square
    w, h = pil_img.size
    crop_size = min(w, h, 800)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    cropped = pil_img.crop((left, top, left + crop_size, top + crop_size))
    
    # Ground Truth HR
    hr_img = cropped.resize(target_size, Image.Resampling.BICUBIC)
    hr_np = np.array(hr_img, dtype=np.float32) / 255.0
    
    # Degraded LR input
    lr_w, lr_h = max(1, target_size[0] // scale_factor), max(1, target_size[1] // scale_factor)
    lr_down = hr_img.resize((lr_w, lr_h), Image.Resampling.BICUBIC)
    lr_up = lr_down.resize(target_size, Image.Resampling.BICUBIC)
    lr_np = np.array(lr_up, dtype=np.float32) / 255.0
    
    # Model Prediction
    batch_in = np.expand_dims(lr_np, axis=0)
    pred_batch = model.predict(batch_in, verbose=0)
    sr_np = np.clip(pred_batch[0], 0.0, 1.0)
    
    # Compute metrics
    bicubic_psnr = compute_psnr(hr_np, lr_np)
    bicubic_ssim = compute_ssim(hr_np, lr_np)
    
    srcnn_psnr = compute_psnr(hr_np, sr_np)
    srcnn_ssim = compute_ssim(hr_np, sr_np)
    psnr_gain = srcnn_psnr - bicubic_psnr
    
    metrics = {
        "bicubic_psnr": bicubic_psnr,
        "bicubic_ssim": bicubic_ssim,
        "srcnn_psnr": srcnn_psnr,
        "srcnn_ssim": srcnn_ssim,
        "psnr_gain_db": psnr_gain
    }
    
    if show_plot or save_path:
        fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), dpi=300)
        
        # 1. LR input
        axes[0].imshow(lr_np)
        axes[0].set_title(f"Degraded Input\n({scale_factor}x Downsampled)", fontsize=11, pad=10)
        axes[0].axis("off")
        
        # 2. Bicubic
        axes[1].imshow(lr_np)
        axes[1].set_title(f"Bicubic Baseline\nPSNR: {bicubic_psnr:.2f} dB | SSIM: {bicubic_ssim:.4f}", fontsize=11, pad=10)
        axes[1].axis("off")
        
        # 3. SRCNN
        axes[2].imshow(sr_np)
        axes[2].set_title(f"SRCNN Enhanced\nPSNR: {srcnn_psnr:.2f} dB ({psnr_gain:+.2f} dB) | SSIM: {srcnn_ssim:.4f}", fontsize=11, fontweight="bold", color="darkgreen", pad=10)
        axes[2].axis("off")
        
        # 4. Ground Truth
        axes[3].imshow(hr_np)
        axes[3].set_title(f"Ground Truth Reference\n({target_size[0]}x{target_size[1]})", fontsize=11, pad=10)
        axes[3].axis("off")
        
        plt.suptitle("Single Image Super-Resolution Live Demonstration", fontsize=15, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[+] Saved demo visualization to: {save_path}")
            
        if show_plot:
            plt.show()
            
    return sr_np, metrics
