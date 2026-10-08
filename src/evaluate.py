"""
Evaluation and benchmark comparison module for SRCNN.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.metrics import compute_mse, compute_psnr, compute_ssim
from src.baselines import evaluate_all_baselines

def evaluate_model_on_test_set(model, test_lr, test_hr):
    """
    Evaluates trained model on the test dataset.
    Returns average MSE, PSNR, SSIM and predictions array.
    """
    preds = model.predict(test_lr, verbose=0)
    preds = np.clip(preds, 0.0, 1.0)
    
    mse_list = []
    psnr_list = []
    ssim_list = []
    
    for i in range(len(test_hr)):
        mse_list.append(compute_mse(test_hr[i], preds[i]))
        psnr_list.append(compute_psnr(test_hr[i], preds[i]))
        ssim_list.append(compute_ssim(test_hr[i], preds[i]))
        
    metrics = {
        "Method": "SRCNN",
        "MSE": float(np.mean(mse_list)),
        "PSNR (dB)": float(np.mean(psnr_list)),
        "SSIM": float(np.mean(ssim_list))
    }
    return metrics, preds

def generate_comparison_table(model, test_lr, test_hr, scale_factor=2):
    """
    Generates a benchmark table comparing Nearest Neighbor, Bilinear, Bicubic, and SRCNN.
    """
    # Baseline results
    baselines_df = evaluate_all_baselines(test_hr, scale_factor=scale_factor)
    
    # Model results
    srcnn_metrics, preds = evaluate_model_on_test_set(model, test_lr, test_hr)
    srcnn_df = pd.DataFrame([srcnn_metrics])
    
    # Combined summary table
    comparison_df = pd.concat([baselines_df, srcnn_df], ignore_index=True)
    return comparison_df, preds

def plot_qualitative_comparisons(
    test_lr, test_hr, preds, num_examples=5, save_path=None
):
    """
    Displays at least 5 qualitative test comparisons with:
    1. Low-Resolution Input (Bicubic Upscaled)
    2. Bicubic Baseline
    3. SRCNN Prediction
    4. Ground-Truth High-Resolution
    Includes PSNR and SSIM values and zoom crops to inspect edge sharpness.
    """
    num_examples = min(num_examples, len(test_hr))
    fig, axes = plt.subplots(num_examples, 4, figsize=(16, 4 * num_examples))
    
    for i in range(num_examples):
        lr_img = np.clip(test_lr[i], 0.0, 1.0)
        hr_img = np.clip(test_hr[i], 0.0, 1.0)
        sr_img = np.clip(preds[i], 0.0, 1.0)
        
        # Bicubic metrics
        bicubic_psnr = compute_psnr(hr_img, lr_img)
        bicubic_ssim = compute_ssim(hr_img, lr_img)
        
        # SRCNN metrics
        srcnn_psnr = compute_psnr(hr_img, sr_img)
        srcnn_ssim = compute_ssim(hr_img, sr_img)
        
        # Panel 1: Low-Resolution Input
        axes[i, 0].imshow(lr_img)
        axes[i, 0].set_title(f"Test #{i+1}: Degraded Input\n(2x Downsampled)", fontsize=10)
        axes[i, 0].axis("off")
        
        # Panel 2: Bicubic Baseline
        axes[i, 1].imshow(lr_img)
        axes[i, 1].set_title(f"Bicubic Baseline\nPSNR: {bicubic_psnr:.2f} dB | SSIM: {bicubic_ssim:.4f}", fontsize=10)
        axes[i, 1].axis("off")
        
        # Panel 3: SRCNN Output
        axes[i, 2].imshow(sr_img)
        gain_psnr = srcnn_psnr - bicubic_psnr
        axes[i, 2].set_title(f"SRCNN Reconstructed\nPSNR: {srcnn_psnr:.2f} dB ({gain_psnr:+.2f} dB)\nSSIM: {srcnn_ssim:.4f}", fontsize=10, fontweight="bold", color="darkgreen")
        axes[i, 2].axis("off")
        
        # Panel 4: Ground Truth HR
        axes[i, 3].imshow(hr_img)
        axes[i, 3].set_title(f"Ground Truth (HR Target)\nReference (224x224)", fontsize=10)
        axes[i, 3].axis("off")
        
    plt.suptitle("Qualitative Super-Resolution Comparison (DIV2K Test Set)", fontsize=16, fontweight="bold", y=1.002)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[+] Saved qualitative comparisons to: {save_path}")
        
    return fig
