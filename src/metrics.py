"""
Evaluation metrics for Single Image Super-Resolution:
- MSE: Mean Squared Error
- PSNR: Peak Signal-to-Noise Ratio (dB)
- SSIM: Structural Similarity Index Measure
"""

import numpy as np
import tensorflow as tf
from skimage.metrics import structural_similarity as sk_ssim

def compute_mse(y_true, y_pred):
    """
    Compute Mean Squared Error between ground truth and predicted image.
    Both images expected in range [0.0, 1.0].
    """
    y_true_np = np.asarray(y_true, dtype=np.float32)
    y_pred_np = np.clip(np.asarray(y_pred, dtype=np.float32), 0.0, 1.0)
    return float(np.mean((y_true_np - y_pred_np) ** 2))

def compute_psnr(y_true, y_pred, max_val=1.0):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    Formula: 10 * log10(MAX_I^2 / MSE) = 20 * log10(MAX_I / sqrt(MSE))
    Handles MSE = 0 edge case gracefully.
    """
    mse = compute_mse(y_true, y_pred)
    if mse <= 1e-10:
        return 99.0  # Cap at 99 dB for identical images
    return float(20.0 * np.log10(max_val) - 10.0 * np.log10(mse))

def compute_ssim(y_true, y_pred, max_val=1.0):
    """
    Compute Structural Similarity Index (SSIM) on RGB image pairs.
    Images should be [H, W, 3] in range [0.0, 1.0].
    """
    y_true_np = np.clip(np.asarray(y_true, dtype=np.float32), 0.0, 1.0)
    y_pred_np = np.clip(np.asarray(y_pred, dtype=np.float32), 0.0, 1.0)
    
    # If single image [H, W, 3]
    if y_true_np.ndim == 3:
        return float(sk_ssim(y_true_np, y_pred_np, channel_axis=2, data_range=max_val))
    # If batch [B, H, W, 3]
    elif y_true_np.ndim == 4:
        scores = [
            sk_ssim(y_true_np[i], y_pred_np[i], channel_axis=2, data_range=max_val)
            for i in range(y_true_np.shape[0])
        ]
        return float(np.mean(scores))
    else:
        raise ValueError(f"Unsupported image shape for SSIM: {y_true_np.shape}")

@tf.keras.utils.register_keras_serializable(package="SRCNN")
def psnr_metric(y_true, y_pred):
    """
    Keras-compatible PSNR metric wrapper.
    Clips predictions to [0.0, 1.0] before computing PSNR.
    """
    clipped_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.image.psnr(y_true, clipped_pred, max_val=1.0)

@tf.keras.utils.register_keras_serializable(package="SRCNN")
def ssim_metric(y_true, y_pred):
    """
    Keras-compatible SSIM metric wrapper.
    Clips predictions to [0.0, 1.0] before computing SSIM.
    """
    clipped_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.image.ssim(y_true, clipped_pred, max_val=1.0)
