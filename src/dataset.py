"""
Dataset pipeline for DIV2K Super-Resolution:
- Loads 100 images
- Crops center 800x800 and resizes HR target to 224x224
- Generates degraded LR inputs (bicubic downsampled by scale factor, then bicubic upsampled to 224x224)
- Provides 60 Train / 20 Validation / 20 Test split
- Builds high-performance tf.data.Dataset pipelines
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from src import config

def preprocess_single_image(image_path, target_size=(224, 224), scale_factor=2):
    """
    Load an image, center crop to 800x800 (if large enough), resize HR to target_size,
    and generate degraded LR (bicubic downsampled, bicubic upsampled back to target_size).
    Returns (lr_image, hr_image) as float32 numpy arrays in [0.0, 1.0].
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    crop_size = min(w, h, config.CROP_SIZE)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    cropped = img.crop((left, top, left + crop_size, top + crop_size))
    
    # High-Resolution Target
    hr_img = cropped.resize(target_size, Image.Resampling.BICUBIC)
    hr_np = np.array(hr_img, dtype=np.float32) / 255.0
    
    # Degraded Low-Resolution Input (downsampled by scale_factor, then upsampled back)
    lr_w, lr_h = max(1, target_size[0] // scale_factor), max(1, target_size[1] // scale_factor)
    lr_down = hr_img.resize((lr_w, lr_h), Image.Resampling.BICUBIC)
    lr_up = lr_down.resize(target_size, Image.Resampling.BICUBIC)
    lr_np = np.array(lr_up, dtype=np.float32) / 255.0
    
    return lr_np, hr_np

def load_all_div2k_data(data_dir=None, target_size=(224, 224), scale_factor=2):
    """
    Loads all 100 images from data directory, preprocesses paired LR/HR images,
    and splits into train (60), val (20), test (20).
    """
    if data_dir is None:
        data_dir = config.DATA_DIR
        
    img_files = sorted([
        f for f in Path(data_dir).glob("*.png")
    ])
    
    if len(img_files) == 0:
        raise FileNotFoundError(f"No PNG images found in {data_dir}. Run scripts/download_dataset.py first.")
        
    img_files = img_files[:config.TOTAL_SAMPLES]
    lr_list = []
    hr_list = []
    
    for f in img_files:
        lr_np, hr_np = preprocess_single_image(f, target_size=target_size, scale_factor=scale_factor)
        lr_list.append(lr_np)
        hr_list.append(hr_np)
        
    lr_arr = np.array(lr_list, dtype=np.float32)
    hr_arr = np.array(hr_list, dtype=np.float32)
    
    train_lr = lr_arr[:config.TRAIN_SAMPLES]
    train_hr = hr_arr[:config.TRAIN_SAMPLES]
    
    val_lr = lr_arr[config.TRAIN_SAMPLES : config.TRAIN_SAMPLES + config.VAL_SAMPLES]
    val_hr = hr_arr[config.TRAIN_SAMPLES : config.TRAIN_SAMPLES + config.VAL_SAMPLES]
    
    test_lr = lr_arr[config.TRAIN_SAMPLES + config.VAL_SAMPLES : config.TRAIN_SAMPLES + config.VAL_SAMPLES + config.TEST_SAMPLES]
    test_hr = hr_arr[config.TRAIN_SAMPLES + config.VAL_SAMPLES : config.TRAIN_SAMPLES + config.VAL_SAMPLES + config.TEST_SAMPLES]
    
    return (train_lr, train_hr), (val_lr, val_hr), (test_lr, test_hr)

def create_tf_datasets(batch_size=8, seed=42):
    """
    Builds optimized tf.data.Dataset pipelines for train, val, and test.
    """
    (train_lr, train_hr), (val_lr, val_hr), (test_lr, test_hr) = load_all_div2k_data()
    
    train_ds = tf.data.Dataset.from_tensor_slices((train_lr, train_hr))
    train_ds = train_ds.shuffle(buffer_size=len(train_lr), seed=seed)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((val_lr, val_hr))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    test_ds = tf.data.Dataset.from_tensor_slices((test_lr, test_hr))
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds, (train_lr, train_hr), (val_lr, val_hr), (test_lr, test_hr)
