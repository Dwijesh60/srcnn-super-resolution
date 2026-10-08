"""
CLI script to train the SRCNN model.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from src import config
from src.utils import set_random_seed, print_system_info
from src.dataset import create_tf_datasets
from src.model import build_srcnn
from src.train import train_srcnn

def main():
    parser = argparse.ArgumentParser(description="Train SRCNN Super-Resolution Model")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    args = parser.parse_args()

    set_random_seed(config.SEED)
    print_system_info()
    
    print(f"[*] Loading DIV2K dataset from: {config.DATA_DIR}")
    train_ds, val_ds, test_ds, _, _, _ = create_tf_datasets(batch_size=args.batch_size)
    
    print(f"[*] Building SRCNN model...")
    model = build_srcnn()
    model.summary()
    
    print(f"[*] Starting training for {args.epochs} epochs with LR={args.lr}...")
    model, history = train_srcnn(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_path=config.CHECKPOINT_PATH
    )
    print(f"[+] Model training complete. Checkpoint saved to: {config.CHECKPOINT_PATH}")

if __name__ == "__main__":
    main()
