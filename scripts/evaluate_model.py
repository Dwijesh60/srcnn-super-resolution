"""
CLI script to evaluate saved SRCNN against traditional interpolation baselines.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import tensorflow as tf
from src import config
from src.utils import set_random_seed, print_system_info
from src.dataset import create_tf_datasets
from src.evaluate import generate_comparison_table, plot_qualitative_comparisons

def main():
    parser = argparse.ArgumentParser(description="Evaluate SRCNN against Baselines")
    parser.add_argument("--model-path", type=str, default=str(config.CHECKPOINT_PATH), help="Path to saved model")
    args = parser.parse_args()

    set_random_seed(config.SEED)
    print_system_info()
    
    print(f"[*] Loading DIV2K dataset...")
    _, _, _, _, _, (test_lr, test_hr) = create_tf_datasets()
    
    print(f"[*] Loading model from: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path, compile=False)
    
    print(f"[*] Evaluating on test set ({len(test_hr)} samples)...")
    comparison_df, preds = generate_comparison_table(model, test_lr, test_hr, scale_factor=config.SCALE_FACTOR)
    
    print("\n" + "=" * 60)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print("=" * 60)
    
    qual_path = config.RESULTS_DIR / "qualitative_comparison.png"
    plot_qualitative_comparisons(test_lr, test_hr, preds, num_examples=5, save_path=qual_path)
    print(f"[+] Qualitative comparison plot saved to: {qual_path}")

if __name__ == "__main__":
    main()
