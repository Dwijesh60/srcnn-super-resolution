"""
Master script to execute all experiments, train SRCNN, evaluate baselines,
generate comparison figures, and export experiment_results.csv.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import time
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from src import config
from src.utils import set_random_seed, print_system_info, plot_training_curves, plot_lr_ablation
from src.dataset import create_tf_datasets
from src.model import build_srcnn
from src.train import train_srcnn
from src.evaluate import generate_comparison_table, evaluate_model_on_test_set, plot_qualitative_comparisons
from src.inference import super_resolve_image

def main():
    print("=" * 70)
    print("SRCNN IMAGE SUPER-RESOLUTION EXPERIMENTAL PIPELINE")
    print("=" * 70)
    
    set_random_seed(config.SEED)
    print_system_info()
    
    print("\n[Phase 1] Loading and Preprocessing DIV2K Dataset...")
    train_ds, val_ds, test_ds, (train_lr, train_hr), (val_lr, val_hr), (test_lr, test_hr) = create_tf_datasets(
        batch_size=config.BATCH_SIZE, seed=config.SEED
    )
    print(f"  Train set: {len(train_lr)} pairs | Validation set: {len(val_lr)} pairs | Test set: {len(test_lr)} pairs")
    
    # ----------------------------------------------------
    # EXPERIMENT A: Baseline Interpolation Methods
    # ----------------------------------------------------
    print("\n[Phase 2] EXPERIMENT A: Traditional Interpolation Baselines...")
    from src.baselines import evaluate_all_baselines
    baselines_df = evaluate_all_baselines(test_hr, scale_factor=config.SCALE_FACTOR)
    print(baselines_df.to_string(index=False))
    
    # ----------------------------------------------------
    # EXPERIMENT B: Standard SRCNN Training (lr = 1e-3)
    # ----------------------------------------------------
    print(f"\n[Phase 3] EXPERIMENT B: Training SRCNN (LR={config.LEARNING_RATE}, Epochs={config.EPOCHS})...")
    srcnn_model = build_srcnn(input_shape=(224, 224, 3))
    srcnn_model.summary()
    
    t0 = time.time()
    trained_model, history = train_srcnn(
        model=srcnn_model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=config.EPOCHS,
        learning_rate=config.LEARNING_RATE,
        checkpoint_path=config.CHECKPOINT_PATH,
        verbose=1
    )
    train_time = time.time() - t0
    print(f"[+] SRCNN Training complete in {train_time:.2f}s")
    
    # Plot and save training history
    curves_path = config.RESULTS_DIR / "training_curves.png"
    plot_training_curves(history, save_path=curves_path)
    plt.close()
    
    # Evaluate best model on test set
    print("\n[Phase 4] Evaluating Best SRCNN on Test Set...")
    best_model = tf.keras.models.load_model(config.CHECKPOINT_PATH, compile=False)
    test_metrics, test_preds = evaluate_model_on_test_set(best_model, test_lr, test_hr)
    
    # Combined Benchmark Table
    comparison_df = pd.concat([baselines_df, pd.DataFrame([test_metrics])], ignore_index=True)
    print("\nFINAL BENCHMARK COMPARISON ON TEST SET:")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print("=" * 60)
    
    # ----------------------------------------------------
    # EXPERIMENT C: Learning Rate Ablation (1e-2, 1e-3, 1e-4)
    # ----------------------------------------------------
    print("\n[Phase 5] EXPERIMENT C: Learning Rate Comparison Ablation...")
    lr_results = []
    lr_histories = {}
    
    for lr in config.LR_EXPERIMENTS:
        lr_label = f"{lr:.0e}"
        print(f"  --> Running ablation with LR = {lr_label} for {config.LR_EXP_EPOCHS} epochs...")
        m_lr = build_srcnn(input_shape=(224, 224, 3), name=f"SRCNN_LR_{lr_label}")
        _, hist = train_srcnn(
            model=m_lr,
            train_dataset=train_ds,
            val_dataset=val_ds,
            epochs=config.LR_EXP_EPOCHS,
            learning_rate=lr,
            checkpoint_path=None,
            verbose=0
        )
        lr_histories[lr_label] = hist.history
        
        # Best val loss & PSNR
        val_losses = hist.history["val_loss"]
        best_val_loss = float(min(val_losses))
        
        psnr_keys = [k for k in hist.history.keys() if "val_" in k and "psnr" in k]
        best_val_psnr = float(max(hist.history[psnr_keys[0]])) if psnr_keys else 0.0
        
        # Test evaluation
        test_eval, _ = evaluate_model_on_test_set(m_lr, test_lr, test_hr)
        
        lr_results.append({
            "Experiment": f"SRCNN (LR={lr_label})",
            "Learning Rate": lr,
            "Best Val Loss": best_val_loss,
            "Best Val PSNR (dB)": best_val_psnr,
            "Test MSE": test_eval["MSE"],
            "Test PSNR (dB)": test_eval["PSNR (dB)"],
            "Test SSIM": test_eval["SSIM"],
            "Epochs": config.LR_EXP_EPOCHS
        })
        
    lr_df = pd.DataFrame(lr_results)
    print("\nLEARNING RATE ABLATION RESULTS:")
    print(lr_df.to_string(index=False))
    
    # Plot LR ablation
    lr_plot_path = config.RESULTS_DIR / "lr_comparison.png"
    plot_lr_ablation(lr_histories, save_path=lr_plot_path)
    plt.close()
    
    # ----------------------------------------------------
    # EXPERIMENT D: Residual SRCNN (Learning Detail Residuals)
    # ----------------------------------------------------
    print("\n[Phase 5b] EXPERIMENT D: Training Residual SRCNN (Res-SRCNN)...")
    from src.model import build_res_srcnn
    res_srcnn = build_res_srcnn(input_shape=(224, 224, 3))
    res_srcnn.summary()
    
    res_model, res_history = train_srcnn(
        model=res_srcnn,
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=15,
        learning_rate=config.LEARNING_RATE,
        checkpoint_path=config.CHECKPOINT_PATH,
        verbose=1
    )
    res_metrics, res_preds = evaluate_model_on_test_set(res_model, test_lr, test_hr)
    res_metrics["Method"] = "Residual SRCNN (Best)"
    print(f"\n[+] Residual SRCNN Test PSNR: {res_metrics['PSNR (dB)']:.2f} dB, SSIM: {res_metrics['SSIM']:.4f}")
    
    # Combined Benchmark Table
    comparison_df = pd.concat([baselines_df, pd.DataFrame([test_metrics]), pd.DataFrame([res_metrics])], ignore_index=True)
    print("\nFINAL BENCHMARK COMPARISON ON TEST SET:")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print("=" * 60)
    
    # ----------------------------------------------------
    # Save Master Experiment Results CSV
    # ----------------------------------------------------
    best_val_loss_b = float(min(history.history["val_loss"]))
    val_psnr_k = [k for k in history.history.keys() if "val_" in k and "psnr" in k]
    best_val_psnr_b = float(max(history.history[val_psnr_k[0]])) if val_psnr_k else 0.0
    
    best_val_loss_res = float(min(res_history.history["val_loss"]))
    val_psnr_res_k = [k for k in res_history.history.keys() if "val_" in k and "psnr" in k]
    best_val_psnr_res = float(max(res_history.history[val_psnr_res_k[0]])) if val_psnr_res_k else 0.0

    master_records = [
        {
            "Experiment": "Baseline: Nearest Neighbor",
            "Learning Rate": "N/A",
            "Best Val Loss": "N/A",
            "Best Val PSNR (dB)": "N/A",
            "Test MSE": baselines_df[baselines_df["Method"] == "Nearest"]["MSE"].values[0],
            "Test PSNR (dB)": baselines_df[baselines_df["Method"] == "Nearest"]["PSNR (dB)"].values[0],
            "Test SSIM": baselines_df[baselines_df["Method"] == "Nearest"]["SSIM"].values[0],
            "Epochs": 0
        },
        {
            "Experiment": "Baseline: Bilinear",
            "Learning Rate": "N/A",
            "Best Val Loss": "N/A",
            "Best Val PSNR (dB)": "N/A",
            "Test MSE": baselines_df[baselines_df["Method"] == "Bilinear"]["MSE"].values[0],
            "Test PSNR (dB)": baselines_df[baselines_df["Method"] == "Bilinear"]["PSNR (dB)"].values[0],
            "Test SSIM": baselines_df[baselines_df["Method"] == "Bilinear"]["SSIM"].values[0],
            "Epochs": 0
        },
        {
            "Experiment": "Baseline: Bicubic",
            "Learning Rate": "N/A",
            "Best Val Loss": "N/A",
            "Best Val PSNR (dB)": "N/A",
            "Test MSE": baselines_df[baselines_df["Method"] == "Bicubic"]["MSE"].values[0],
            "Test PSNR (dB)": baselines_df[baselines_df["Method"] == "Bicubic"]["PSNR (dB)"].values[0],
            "Test SSIM": baselines_df[baselines_df["Method"] == "Bicubic"]["SSIM"].values[0],
            "Epochs": 0
        },
        {
            "Experiment": "Standard SRCNN (Feedforward)",
            "Learning Rate": config.LEARNING_RATE,
            "Best Val Loss": best_val_loss_b,
            "Best Val PSNR (dB)": best_val_psnr_b,
            "Test MSE": test_metrics["MSE"],
            "Test PSNR (dB)": test_metrics["PSNR (dB)"],
            "Test SSIM": test_metrics["SSIM"],
            "Epochs": config.EPOCHS
        }
    ]
    # Add LR ablation rows
    for r in lr_results:
        master_records.append(r)
        
    master_records.append({
        "Experiment": "Residual SRCNN (Res-SRCNN)",
        "Learning Rate": config.LEARNING_RATE,
        "Best Val Loss": best_val_loss_res,
        "Best Val PSNR (dB)": best_val_psnr_res,
        "Test MSE": res_metrics["MSE"],
        "Test PSNR (dB)": res_metrics["PSNR (dB)"],
        "Test SSIM": res_metrics["SSIM"],
        "Epochs": 15
    })
        
    master_df = pd.DataFrame(master_records)
    master_df.to_csv(config.EXPERIMENT_RESULTS_CSV, index=False)
    print(f"\n[+] Saved complete experiment results to: {config.EXPERIMENT_RESULTS_CSV}")
    
    # ----------------------------------------------------
    # Qualitative Comparisons (Using Best Model)
    # ----------------------------------------------------
    print("\n[Phase 6] Generating Qualitative Comparisons (5 test images using Best Res-SRCNN)...")
    qual_path = config.RESULTS_DIR / "qualitative_comparison.png"
    plot_qualitative_comparisons(test_lr, test_hr, res_preds, num_examples=5, save_path=qual_path)
    plt.close()
    
    # ----------------------------------------------------
    # Single Image Demo (Using Best Model)
    # ----------------------------------------------------
    print("\n[Phase 7] Running Single Image Demo...")
    demo_img_path = config.DEMO_DIR / "test_image.png"
    demo_save_path = config.RESULTS_DIR / "demo_output.png"
    super_resolve_image(
        image_input=demo_img_path,
        model=res_model,
        scale_factor=config.SCALE_FACTOR,
        show_plot=False,
        save_path=demo_save_path
    )
    plt.close()
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Best SRCNN Model Saved: {config.CHECKPOINT_PATH}")
    print(f"Results CSV: {config.EXPERIMENT_RESULTS_CSV}")
    print(f"Training Curves: {curves_path}")
    print(f"LR Comparison: {lr_plot_path}")
    print(f"Qualitative Comparisons: {qual_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
