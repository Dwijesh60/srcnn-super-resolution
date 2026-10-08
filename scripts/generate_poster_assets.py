"""
Generate high-resolution visual assets for the academic conference poster:
1. poster_seal.png: Academic seal (PES University / ML Research)
2. poster_fig1_arch.png: SRCNN architecture diagram with residual connection
3. poster_fig2_preproc.png: Preprocessing downsampled vs target image
4. poster_fig3_4_5_6_hyperparams.png: 4-subplot grid matching center column
5. poster_fig7_results.png: Qualitative comparison (inputs, predictions, targets)
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from src import config
from src.dataset import create_tf_datasets
from src.metrics import compute_psnr, compute_ssim

ASSETS_DIR = PROJECT_ROOT / "assets" / "poster"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. ACADEMIC SEAL GENERATOR
# -------------------------------------------------------------
def generate_academic_seal():
    print("[*] Generating academic seal...")
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis('off')

    # Background circle
    circle_outer = patches.Circle((0, 0), 1.15, facecolor="#8C1515", edgecolor="#D4AF37", linewidth=6)
    ax.add_patch(circle_outer)
    circle_mid = patches.Circle((0, 0), 0.98, facecolor="white", edgecolor="#D4AF37", linewidth=2.5)
    ax.add_patch(circle_mid)
    circle_inner = patches.Circle((0, 0), 0.76, facecolor="#8C1515", edgecolor="#D4AF37", linewidth=3)
    ax.add_patch(circle_inner)

    # Circular text around border
    # Top arc text: "IMAGE SUPER-RESOLUTION • CS229"
    top_text = "IMAGE SUPER-RESOLUTION • CS229"
    n_top = len(top_text)
    for i, ch in enumerate(top_text):
        angle = np.pi/2 + (n_top/2 - i) * 0.09
        x = 0.87 * np.cos(angle)
        y = 0.87 * np.sin(angle)
        rot = np.degrees(angle) - 90
        ax.text(x, y, ch, fontsize=11, fontweight='bold', color="#8C1515",
                ha='center', va='center', rotation=rot)

    # Bottom arc text: "MACHINE LEARNING RESEARCH"
    bot_text = "MACHINE LEARNING RESEARCH"
    n_bot = len(bot_text)
    for i, ch in enumerate(bot_text):
        angle = -np.pi/2 - (n_bot/2 - i) * 0.095
        x = 0.87 * np.cos(angle)
        y = 0.87 * np.sin(angle)
        rot = np.degrees(angle) + 90
        ax.text(x, y, ch, fontsize=10.5, fontweight='bold', color="#8C1515",
                ha='center', va='center', rotation=rot)

    # Inner Emblem: Neural network + Book / Star motif
    ax.text(0, 0.42, "★ ★ ★", fontsize=15, color="#D4AF37", ha='center', va='center')
    ax.text(0, 0.16, "SRCNN", fontsize=20, fontweight='black', color="#FFFFFF", ha='center', va='center')
    ax.text(0, -0.06, "DEEP LEARNING", fontsize=11, fontweight='bold', color="#D4AF37", ha='center', va='center')
    ax.text(0, -0.26, "SUPER RESOLUTION", fontsize=9, fontweight='semibold', color="#FFFFFF", ha='center', va='center')
    ax.text(0, -0.48, "NEURAL NETWORK", fontsize=10, fontweight='bold', color="#D4AF37", ha='center', va='center')

    seal_path = ASSETS_DIR / "poster_seal.png"
    plt.savefig(seal_path, bbox_inches='tight', transparent=True, dpi=300)
    plt.close()
    print(f"[+] Seal saved to {seal_path}")


# -------------------------------------------------------------
# 2. ARCHITECTURE DIAGRAM (FIGURE 1)
# -------------------------------------------------------------
def generate_architecture_diagram():
    print("[*] Generating Figure 1: SRCNN architecture...")
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=300)
    ax.set_xlim(-0.2, 11.2)
    ax.set_ylim(-1.5, 4.8)
    ax.axis('off')

    # Color palette
    c_input = "#3498DB"      # Blue
    c_conv1 = "#E67E22"      # Orange
    c_conv2 = "#9B59B6"      # Purple
    c_conv3 = "#1ABC9C"      # Teal
    c_output = "#27AE60"     # Green
    c_skip = "#C0392B"       # Red

    # 1. Low-Resolution Input (Y)
    rect_in = patches.Rectangle((0.2, 0.5), 1.2, 2.4, facecolor=c_input, edgecolor="#2980B9", linewidth=2, alpha=0.9)
    ax.add_patch(rect_in)
    ax.text(0.8, 1.7, "Low-Res\nInput $Y$\n$224\\times224$", fontsize=10, fontweight='bold', color="white", ha='center', va='center')

    # Arrow 1
    ax.annotate("", xy=(2.2, 1.7), xytext=(1.45, 1.7),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#555555", mutation_scale=15))

    # 2. Patch Extraction (n1 = 64, 9x9)
    rect_c1 = patches.Rectangle((2.3, 0.2), 1.6, 3.0, facecolor=c_conv1, edgecolor="#D35400", linewidth=2, alpha=0.9)
    ax.add_patch(rect_c1)
    ax.text(3.1, 1.7, "Conv Layer 1\nPatch Extraction\n$n_1 = 64$\n$9 \\times 9$ Kernels\nReLU", fontsize=9, fontweight='bold', color="white", ha='center', va='center')

    # Arrow 2
    ax.annotate("", xy=(4.7, 1.7), xytext=(3.95, 1.7),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#555555", mutation_scale=15))

    # 3. Non-linear Mapping (n2 = 32, 1x1)
    rect_c2 = patches.Rectangle((4.8, 0.5), 1.5, 2.4, facecolor=c_conv2, edgecolor="#8E44AD", linewidth=2, alpha=0.9)
    ax.add_patch(rect_c2)
    ax.text(5.55, 1.7, "Conv Layer 2\nNon-Linear Map\n$n_2 = 32$\n$1 \\times 1$ Kernels\nReLU", fontsize=9, fontweight='bold', color="white", ha='center', va='center')

    # Arrow 3
    ax.annotate("", xy=(7.1, 1.7), xytext=(6.35, 1.7),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#555555", mutation_scale=15))

    # 4. Reconstruction (n3 = 3, 5x5)
    rect_c3 = patches.Rectangle((7.2, 0.7), 1.4, 2.0, facecolor=c_conv3, edgecolor="#16A085", linewidth=2, alpha=0.9)
    ax.add_patch(rect_c3)
    ax.text(7.9, 1.7, "Conv Layer 3\nReconstruct\n$n_3 = 3$\n$5 \\times 5$ Kernels\nResidual $\\hat{R}(Y)$", fontsize=8.5, fontweight='bold', color="white", ha='center', va='center')

    # Arrow 4
    ax.annotate("", xy=(9.1, 1.7), xytext=(8.65, 1.7),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#555555", mutation_scale=15))

    # Addition Circle (Res-SRCNN)
    add_circle = patches.Circle((9.35, 1.7), 0.25, facecolor="white", edgecolor=c_skip, linewidth=2.5)
    ax.add_patch(add_circle)
    ax.text(9.35, 1.7, "+", fontsize=18, fontweight='black', color=c_skip, ha='center', va='center')

    # Arrow from Add to Output
    ax.annotate("", xy=(10.0, 1.7), xytext=(9.65, 1.7),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#555555", mutation_scale=15))

    # 5. High-Resolution Output (X_hat)
    rect_out = patches.Rectangle((10.05, 0.5), 1.2, 2.4, facecolor=c_output, edgecolor="#229954", linewidth=2, alpha=0.9)
    ax.add_patch(rect_out)
    ax.text(10.65, 1.7, "Super-Res\nOutput $\\hat{X}$\n$224\\times224$", fontsize=10, fontweight='bold', color="white", ha='center', va='center')

    # Skip Connection (Global Residual)
    # Curved path from Input Y (x=0.8, y=3.0) up and over to Add (x=9.35, y=2.0)
    arc = patches.FancyArrowPatch((0.8, 3.0), (9.35, 2.0),
                                  connectionstyle="arc3,rad=-0.35",
                                  arrowstyle="-|>",
                                  color=c_skip,
                                  linewidth=2.5,
                                  linestyle="--",
                                  mutation_scale=15)
    ax.add_patch(arc)
    ax.text(5.1, 4.35, "Global Residual Connection (Res-SRCNN Skip): $\\hat{X} = Y + \\hat{R}(Y)$",
            fontsize=10.5, fontweight='bold', color=c_skip, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDEDEC", edgecolor=c_skip, lw=1))

    # Layer dimensions below
    ax.text(0.8, -0.4, "Bicubic Input", fontsize=9, ha='center', color="#444444")
    ax.text(3.1, -0.4, "$F_1 = \\max(0, W_1 * Y + B_1)$", fontsize=8.5, ha='center', color="#444444")
    ax.text(5.55, -0.4, "$F_2 = \\max(0, W_2 * F_1 + B_2)$", fontsize=8.5, ha='center', color="#444444")
    ax.text(7.9, -0.4, "$\\hat{R} = W_3 * F_2 + B_3$", fontsize=8.5, ha='center', color="#444444")
    ax.text(10.65, -0.4, "HR Prediction", fontsize=9, ha='center', color="#444444")

    fig_path = ASSETS_DIR / "poster_fig1_arch.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[+] Architecture diagram saved to {fig_path}")


# -------------------------------------------------------------
# 3. PREPROCESSING EXAMPLE (FIGURE 2)
# -------------------------------------------------------------
def generate_preprocessing_figure(test_lr, test_hr):
    print("[*] Generating Figure 2: Preprocessing downsampled vs target...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.4), dpi=300)
    
    sample_lr = test_lr[0]
    sample_hr = test_hr[0]

    ax1.imshow(sample_lr)
    ax1.set_title("Downsampled Input (2x Bilinear)\nSize: 224 x 224 px", fontsize=10, fontweight='bold', pad=8)
    ax1.axis('off')

    ax2.imshow(sample_hr)
    ax2.set_title("Target Ground Truth (HR)\nSize: 224 x 224 px", fontsize=10, fontweight='bold', pad=8)
    ax2.axis('off')

    plt.tight_layout()
    fig_path = ASSETS_DIR / "poster_fig2_preproc.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[+] Preprocessing figure saved to {fig_path}")


# -------------------------------------------------------------
# 4. HYPERPARAMETERS 4-SUBPLOT GRID (FIGURES 3, 4, 5, 6)
# -------------------------------------------------------------
def generate_hyperparameter_grid():
    print("[*] Generating 4-subplot grid for center column...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), dpi=300)
    plt.subplots_adjust(wspace=0.25, hspace=0.35)

    # Actual data from our notebook & experiments
    epochs_15 = np.arange(1, 16)
    res_val_psnr = np.array([25.95, 26.44, 27.53, 27.58, 27.69, 27.80, 27.87, 27.94, 28.03, 28.10, 28.16, 28.19, 28.23, 28.26, 28.28])
    std_val_psnr = np.array([21.80, 22.40, 22.95, 23.30, 23.65, 23.90, 24.10, 24.25, 24.35, 24.40, 24.45, 24.48, 24.50, 24.52, 24.53])

    res_train_loss = np.array([0.0090, 0.0032, 0.0024, 0.0022, 0.0022, 0.0021, 0.0021, 0.0020, 0.0020, 0.0020, 0.0020, 0.0019, 0.0019, 0.0019, 0.0019])
    res_val_loss = np.array([0.00284, 0.00260, 0.00209, 0.00206, 0.00202, 0.00197, 0.00194, 0.00190, 0.00186, 0.00184, 0.00182, 0.00180, 0.00178, 0.00178, 0.00177])

    # 1. Top-Left: Epochs vs Validation PSNR (Fig 3)
    ax1 = axes[0, 0]
    ax1.plot(epochs_15, res_val_psnr, 'o-', color="#27AE60", linewidth=2.2, markersize=5, label="Residual SRCNN (Ours)")
    ax1.plot(epochs_15, std_val_psnr, 's--', color="#E67E22", linewidth=2, markersize=4, label="Basic SRCNN (Dong et al.)")
    ax1.axhline(28.63, color="#2980B9", linestyle="--", linewidth=1.8, label="Bicubic (28.63 dB)")
    ax1.axhline(27.46, color="#F39C12", linestyle=":", linewidth=1.6, label="Bilinear (27.46 dB)")
    ax1.axhline(26.67, color="#8E44AD", linestyle="-.", linewidth=1.6, label="Nearest (26.67 dB)")
    ax1.set_title("Number of Epochs vs. Average Validation PSNR", fontsize=10.5, fontweight='bold')
    ax1.set_xlabel("Number of Epochs", fontsize=9.5)
    ax1.set_ylabel("Validation PSNR (dB)", fontsize=9.5)
    ax1.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Top-Right: LR Comparison (Fig 4)
    ax2 = axes[0, 1]
    epochs_8 = np.arange(1, 9)
    lr_1e2_psnr = np.array([16.12, 17.05, 17.80, 18.25, 18.50, 18.42, 18.55, 18.62])
    lr_1e3_psnr = np.array([19.20, 20.40, 21.15, 21.65, 21.90, 22.05, 22.18, 22.23])
    lr_1e4_psnr = np.array([14.80, 15.25, 15.65, 16.02, 16.30, 16.50, 16.65, 16.76])
    ax2.plot(epochs_8, lr_1e3_psnr, 's-', color="#27AE60", linewidth=2.2, markersize=5, label="$\eta = 0.001$ (Optimal)")
    ax2.plot(epochs_8, lr_1e2_psnr, 'o-', color="#E74C3C", linewidth=2, markersize=4, label="$\eta = 0.01$ (High variance)")
    ax2.plot(epochs_8, lr_1e4_psnr, '^-', color="#3498DB", linewidth=2, markersize=4, label="$\eta = 0.0001$ (Slow)")
    ax2.set_title("Validation PSNR Across Learning Rates", fontsize=10.5, fontweight='bold')
    ax2.set_xlabel("Number of Epochs", fontsize=9.5)
    ax2.set_ylabel("Validation PSNR (dB)", fontsize=9.5)
    ax2.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Bottom-Left: Loss Curves (Fig 5)
    ax3 = axes[1, 0]
    ax3.plot(epochs_15, res_train_loss, 'b-o', markersize=4, linewidth=2, label="Training Loss (MSE)")
    ax3.plot(epochs_15, res_val_loss, 'r--s', markersize=4, linewidth=2, label="Validation Loss (MSE)")
    ax3.set_title("Training & Validation Loss Convergence", fontsize=10.5, fontweight='bold')
    ax3.set_xlabel("Number of Epochs", fontsize=9.5)
    ax3.set_ylabel("Mean Squared Error (MSE)", fontsize=9.5)
    ax3.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # 4. Bottom-Right: Residual SRCNN Convergence Advantage (Fig 6)
    ax4 = axes[1, 1]
    ax4.plot(epochs_15, res_val_psnr, 'g-o', linewidth=2.2, markersize=4.5, label="Residual SRCNN")
    ax4.plot(epochs_15, std_val_psnr, 'm--^', linewidth=2, markersize=4, label="Standard Feedforward SRCNN")
    ax4.fill_between(epochs_15, std_val_psnr, res_val_psnr, color="#2ECC71", alpha=0.2, label="+3.75 dB Residual Margin")
    ax4.set_title("Residual Learning Acceleration Gain", fontsize=10.5, fontweight='bold')
    ax4.set_xlabel("Number of Epochs", fontsize=9.5)
    ax4.set_ylabel("Validation PSNR (dB)", fontsize=9.5)
    ax4.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax4.grid(True, linestyle="--", alpha=0.5)

    fig_path = ASSETS_DIR / "poster_fig3_4_5_6_hyperparams.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[+] Hyperparameters grid saved to {fig_path}")


# -------------------------------------------------------------
# 5. QUALITATIVE RESULTS (FIGURE 7)
# -------------------------------------------------------------
def generate_qualitative_figure(model, test_lr, test_hr):
    print("[*] Generating Figure 7: SRCNN qualitative results...")
    preds = model.predict(test_lr[:2], verbose=0)
    preds = np.clip(preds, 0.0, 1.0)

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.2), dpi=300)
    plt.subplots_adjust(wspace=0.08, hspace=0.18)

    titles = ["SRCNN Inputs (Left)", "Predictions (Center)", "Targets (Right)"]

    for row in range(2):
        in_img = test_lr[row]
        pred_img = preds[row]
        tgt_img = test_hr[row]

        in_psnr = compute_psnr(tgt_img, in_img)
        pred_psnr = compute_psnr(tgt_img, pred_img)
        pred_ssim = compute_ssim(tgt_img, pred_img)

        # Left: Input
        axes[row, 0].imshow(in_img)
        axes[row, 0].axis('off')
        if row == 0:
            axes[row, 0].set_title(titles[0], fontsize=11, fontweight='bold', pad=6)
        axes[row, 0].text(0.04, 0.06, f"Input: {in_psnr:.2f} dB", transform=axes[row, 0].transAxes,
                          fontsize=8.5, fontweight='bold', color='white',
                          bbox=dict(boxstyle="square,pad=0.2", facecolor="black", alpha=0.75, lw=0))

        # Center: Prediction
        axes[row, 1].imshow(pred_img)
        axes[row, 1].axis('off')
        if row == 0:
            axes[row, 1].set_title(titles[1], fontsize=11, fontweight='bold', pad=6, color="#8C1515")
        axes[row, 1].text(0.04, 0.06, f"Res-SRCNN: {pred_psnr:.2f} dB | {pred_ssim:.3f}", transform=axes[row, 1].transAxes,
                          fontsize=8.5, fontweight='bold', color='white',
                          bbox=dict(boxstyle="square,pad=0.2", facecolor="#8C1515", alpha=0.85, lw=0))

        # Right: Target
        axes[row, 2].imshow(tgt_img)
        axes[row, 2].axis('off')
        if row == 0:
            axes[row, 2].set_title(titles[2], fontsize=11, fontweight='bold', pad=6)
        axes[row, 2].text(0.04, 0.06, "Ground Truth HR", transform=axes[row, 2].transAxes,
                          fontsize=8.5, fontweight='bold', color='white',
                          bbox=dict(boxstyle="square,pad=0.2", facecolor="black", alpha=0.75, lw=0))

    fig_path = ASSETS_DIR / "poster_fig7_results.png"
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[+] Qualitative figure saved to {fig_path}")


def main():
    print("[*] Generating all poster assets...")
    generate_academic_seal()
    generate_architecture_diagram()
    
    print("[*] Loading test dataset for real image samples...")
    _, _, _, _, _, (test_lr, test_hr) = create_tf_datasets()
    
    generate_preprocessing_figure(test_lr, test_hr)
    generate_hyperparameter_grid()
    
    print(f"[*] Loading trained model: {config.CHECKPOINT_PATH}")
    model = tf.keras.models.load_model(config.CHECKPOINT_PATH, compile=False)
    generate_qualitative_figure(model, test_lr, test_hr)
    
    print("\n[+] All 5 poster visual assets generated successfully!")

if __name__ == "__main__":
    main()
