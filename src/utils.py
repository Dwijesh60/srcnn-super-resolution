"""
Utility functions for reproducibility, system hardware detection, and plotting.
"""

import os
import random
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def set_random_seed(seed=42):
    """Set random seeds across Python, NumPy, and TensorFlow for strict reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_system_info():
    """Detect and return Python, TensorFlow, and hardware device execution status."""
    gpus = tf.config.list_physical_devices("GPU")
    device_name = f"GPU ({gpus[0].name})" if gpus else "CPU (Host Processor)"
    
    info = {
        "python_version": sys.version.split()[0],
        "tensorflow_version": tf.__version__,
        "gpu_available": len(gpus) > 0,
        "selected_device": device_name
    }
    return info

def print_system_info():
    """Print clean environment and hardware detection summary."""
    info = get_system_info()
    print("=" * 60)
    print("ENVIRONMENT & HARDWARE CONFIGURATION")
    print("=" * 60)
    print(f"  Python Version      : {info['python_version']}")
    print(f"  TensorFlow Version  : {info['tensorflow_version']}")
    print(f"  GPU Available       : {info['gpu_available']}")
    print(f"  Active Compute Device: {info['selected_device']}")
    print("=" * 60)

def plot_training_curves(history, save_path=None):
    """Plot and save Loss vs Epoch and PSNR vs Epoch training curves."""
    epochs = range(1, len(history.history["loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, history.history["loss"], "b-o", label="Training Loss (MSE)", linewidth=2, markersize=4)
    if "val_loss" in history.history:
        ax1.plot(epochs, history.history["val_loss"], "r--s", label="Validation Loss (MSE)", linewidth=2, markersize=4)
    ax1.set_title("Training & Validation Loss (MSE)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Mean Squared Error", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    # PSNR plot
    psnr_key = [k for k in history.history.keys() if "psnr" in k and not k.startswith("val_")]
    val_psnr_key = [k for k in history.history.keys() if "val_" in k and "psnr" in k]
    
    if psnr_key:
        ax2.plot(epochs, history.history[psnr_key[0]], "g-o", label="Training PSNR", linewidth=2, markersize=4)
    if val_psnr_key:
        ax2.plot(epochs, history.history[val_psnr_key[0]], "m--^", label="Validation PSNR", linewidth=2, markersize=4)
    ax2.set_title("Peak Signal-to-Noise Ratio (PSNR)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("PSNR (dB)", fontsize=11)
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[+] Saved training curves to: {save_path}")
    return fig

def plot_lr_ablation(lr_histories, save_path=None):
    """Plot validation PSNR and Loss comparison across different learning rates."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {"1e-2": "#e74c3c", "1e-3": "#2ecc71", "1e-4": "#3498db"}
    
    for lr_label, hist in lr_histories.items():
        color = colors.get(lr_label, None)
        epochs = range(1, len(hist["val_loss"]) + 1)
        ax1.plot(epochs, hist["val_loss"], marker="o", label=f"LR = {lr_label}", linewidth=2, color=color)
        
        val_psnr_key = [k for k in hist.keys() if "val_" in k and "psnr" in k]
        if val_psnr_key:
            ax2.plot(epochs, hist[val_psnr_key[0]], marker="s", label=f"LR = {lr_label}", linewidth=2, color=color)
            
    ax1.set_title("Validation Loss Comparison Across Learning Rates", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Validation MSE", fontsize=11)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)
    
    ax2.set_title("Validation PSNR Comparison Across Learning Rates", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Validation PSNR (dB)", fontsize=11)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[+] Saved LR comparison curves to: {save_path}")
    return fig
