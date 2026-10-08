"""
Configuration parameters for the SRCNN Super-Resolution Mini-Project.
Follows the Stanford CS229 Project specifications:
- DIV2K dataset: 100 images (60 Train, 20 Validation, 20 Test)
- Center crop ~800x800, target HR 224x224
- Degradation: Downsample 2x, upsample Bicubic back to 224x224
- 3-Layer SRCNN (9-1-5) architecture
- Adam optimizer, initial LR = 1e-3, MSE loss, PSNR/SSIM evaluation
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "DIV2K_100"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"
DEMO_DIR = PROJECT_ROOT / "demo"

# Create directories if they do not exist
for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR, DOCS_DIR, DEMO_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# File Paths
CHECKPOINT_PATH = MODELS_DIR / "best_srcnn.keras"
EXPERIMENT_RESULTS_CSV = RESULTS_DIR / "experiment_results.csv"

# Global Random Seed
SEED = 42

# Image and Preprocessing Hyperparameters
CROP_SIZE = 800
IMAGE_SIZE = (224, 224)
TARGET_HEIGHT = 224
TARGET_WIDTH = 224
CHANNELS = 3
SCALE_FACTOR = 2

# Dataset Split
TOTAL_SAMPLES = 100
TRAIN_SAMPLES = 60
VAL_SAMPLES = 20
TEST_SAMPLES = 20

# Training Hyperparameters
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-3

# Hyperparameter search learning rates for Experiment C
LR_EXPERIMENTS = [1e-2, 1e-3, 1e-4]
LR_EXP_EPOCHS = 10
