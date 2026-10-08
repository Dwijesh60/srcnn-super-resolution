# Image Super-Resolution Via a Convolutional Neural Network (SRCNN)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![Gradio UI](https://img.shields.io/badge/Gradio-Demo%20UI-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Machine Learning Mini-Project**  
> Reproducing, evaluating, and extending the Super-Resolution Convolutional Neural Network (SRCNN) on the DIV2K benchmark against classical mathematical interpolation baselines. Inspired by the Stanford University CS229 Project (*Garber, Grossman, Johnson-Yu, 2020*) and the foundational work of *Chao Dong et al. (IEEE TPAMI 2016)*.

---

## 👥 Authors & Project Information
- **Team Members:**
  - **Dwijesh krishna G** — SRN: `PES2UG24CS163`
  - **Chiranthan Shankar** — SRN: `PES2UG24CS137`
- **Course:** Machine Learning Mini-Project  
- **Repository:** [https://github.com/Dwijesh60/srcnn-super-resolution](https://github.com/Dwijesh60/srcnn-super-resolution)

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Problem Statement](#-problem-statement)
3. [Dataset & Preprocessing](#-dataset--preprocessing)
4. [SRCNN Architecture](#-srcnn-architecture)
5. [Methodology & Experiments](#-methodology--experiments)
6. [Quantitative Benchmark Results](#-quantitative-benchmark-results)
7. [Visual Comparison](#-visual-comparison)
8. [Repository Structure](#-repository-structure)
9. [Installation & Setup](#-installation--setup)
10. [Running the PyTorch Pipeline & Gradio Demo](#-running-the-pytorch-pipeline--gradio-demo)
11. [Running the TensorFlow / Modular Pipeline](#-running-the-tensorflow--modular-pipeline)
12. [How to Run in Google Colab](#-how-to-run-in-google-colab)
13. [Academic Documentation & Defense Materials](#-academic-documentation--defense-materials)
14. [References](#-references)

---

## 📖 Project Overview
Single-Image Super-Resolution (SISR) is the computer vision task of recovering high-resolution (HR) photographic detail from a degraded low-resolution (LR) observation. While classical interpolation techniques (Nearest Neighbor, Bilinear, and Bicubic) apply static mathematical formulas that blur sharp transitions, deep learning models learn non-linear spatial representations directly from data.

This project delivers a **dual-framework implementation**:
1. **PyTorch Framework:** Clean modular scripts for training, evaluation, and an interactive **Gradio Web App** for live real-time super-resolution.
2. **TensorFlow / Keras Framework:** A modular pipeline with residual detail learning (**Res-SRCNN**), automated ablation suites, and academic research poster generation.

---

## 🎯 Problem Statement
In physical imaging systems, sensor limitations, focal degradation, compression, and transmission bandwidth constraints irreversibly discard high-frequency spatial frequencies. The forward degradation process is formulated as:

$$Y = (X \otimes k) \downarrow_s + \epsilon$$

where:
- $X$ is the ground-truth high-resolution image,
- $k$ is the blur kernel point-spread function,
- $\downarrow_s$ represents downsampling by scale factor $s=2$,
- $\epsilon$ denotes additive sensor noise,
- $Y$ is the degraded low-resolution observation.

Because downsampling is a many-to-one projection, SISR is a mathematically **ill-posed inverse problem**. Our objective is to learn a parameterized non-linear mapping function $F(Y; \Theta) \to X$ that reconstructs high-fidelity spatial details.

---

## 📊 Dataset & Preprocessing
We evaluate on the **DIV2K (Diverse 2K Resolution)** benchmark dataset following the Stanford CS229 protocol:
- **100 Authentic DIV2K Images:**
  - **Train Set:** 60 images (60%)
  - **Validation Set:** 20 images (20%)
  - **Test Set:** 20 images (20%)
- **Systematic Pipeline:**
  1. **Center Crop:** Source images cropped to approximately $800 \times 800$.
  2. **Ground Truth HR Target:** Resized to $224 \times 224 \times 3$.
  3. **Degradation:** Downsampled by scale factor $s=2$ (to $112 \times 112$) using bicubic filtering.
  4. **Upscaling:** Upsampled back to $224 \times 224$ via bicubic interpolation to create paired network input tensors.
  5. **Normalization:** Intensities scaled to $[0.0, 1.0]$.
  6. **Data Leakage Prevention:** Validation and test partitions remain strictly isolated from gradient updates and hyperparameter tuning.

---

## 🧠 SRCNN Architecture
SRCNN formulates super-resolution through three sequential convolutional operations:

```
Low-Resolution Bicubic Input (224x224x3)
                    │
                    ▼
┌───────────────────────────────────────┐
│ Conv2D: 64 filters, 9x9, ReLU, 'same' │  <-- Layer 1: Patch Extraction & Representation
└───────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│ Conv2D: 32 filters, 1x1, ReLU, 'same' │  <-- Layer 2: Non-Linear Feature Mapping
└───────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│ Conv2D: 3 filters, 5x5, Linear, 'same'│  <-- Layer 3: High-Resolution Reconstruction
└───────────────────────────────────────┘
                    │
                    ▼
   Super-Resolved Output (224x224x3)
```

1. **Layer 1 (Patch Extraction & Representation):**
   - Kernel: $9 \times 9$, 64 filters, ReLU activation.
   - Computes 64 feature maps extracting overlapping edge and texture primitives (15,616 parameters).
2. **Layer 2 (Non-Linear Mapping):**
   - Kernel: $1 \times 1$, 32 filters, ReLU activation.
   - Non-linearly maps 64-dimensional feature representations to 32 channels (2,080 parameters).
3. **Layer 3 (Reconstruction):**
   - Kernel: $5 \times 5$, 3 filters, Linear activation.
   - Aggregates local feature maps into continuous RGB pixel values (2,403 parameters).
- **Total Parameters:** 20,099 parameters (~78.5 KB).

---

## 🔬 Methodology & Experiments
- **Loss Function:** Mean Squared Error (MSE), directly minimizing pixel error and maximizing PSNR:
  $$\mathcal{L}(\Theta) = \frac{1}{N} \sum_{i=1}^N \| F(Y_i; \Theta) - X_i \|_2^2$$
- **Metrics:**
  - **PSNR (Peak Signal-to-Noise Ratio):** $\text{PSNR} = 10 \cdot \log_{10}(1.0 / \text{MSE})$
  - **SSIM (Structural Similarity Index Measure):** Evaluates luminance, contrast, and structural fidelity.
- **Optimization:** Adam optimizer ($\beta_1=0.9, \beta_2=0.999$), batch size 8.
- **Experimental Program:**
  - **Experiment A:** Classical interpolation baselines (Nearest, Bilinear, Bicubic).
  - **Experiment B:** Standard Feedforward SRCNN.
  - **Experiment C:** Learning Rate Ablation ($\text{LR} \in \{10^{-2}, 10^{-3}, 10^{-4}\}$).
  - **Experiment D:** Residual SRCNN (Res-SRCNN) learning high-frequency detail residuals ($Output = Input + \mathcal{F}(Input)$).

---

## 📈 Quantitative Benchmark Results
Evaluated on the held-out DIV2K test set (20 images):

| Method / Model | Test MSE | Test PSNR (dB) | Test SSIM | Parameters |
| :--- | :---: | :---: | :---: | :---: |
| Nearest Neighbor | 0.002628 | 26.67 dB | 0.8535 | Hand-crafted (0) |
| Bilinear Interpolation | 0.002283 | 27.46 dB | 0.8504 | Hand-crafted (0) |
| Bicubic Interpolation | 0.001790 | 28.63 dB | 0.8815 | Hand-crafted (0) |
| Standard SRCNN (PyTorch) | 0.002150 | 26.67 dB | 0.8610 | 20,099 |
| **Residual SRCNN (Keras Best)** | **0.001512** | **29.44 dB** | **0.9002** | **20,099** |

### Key Findings:
1. **Bicubic Outperforms Nearest & Bilinear:** +1.96 dB gain over Nearest Neighbor.
2. **Residual SRCNN Decisively Outperforms Bicubic:** +0.81 dB PSNR gain and +0.0187 SSIM improvement over Bicubic.
3. **Optimal Learning Rate:** $\text{LR} = 0.001$ demonstrated the highest validation stability and convergence speed.

---

## 🖼️ Visual Comparison
- **Degraded Input:** Soft edges, loss of high-frequency texture.
- **Bicubic Baseline:** Smooths pixelation but leaves blurred boundaries.
- **SRCNN / Res-SRCNN:** Reconstructs sharp edge transitions, eliminates color fringe halos, and enhances contrast.
- Visual comparison plots are preserved in `results/qualitative_comparison.png`, `results/demo_output.png`, and `results/test_sample_1_zoomed.png`.

---

## 📁 Repository Structure
```
srcnn-super-resolution/
│
├── README.md                          # Comprehensive project documentation
├── requirements.txt                   # Production Python dependencies
├── .gitignore                         # Configured for checkpoints and datasets
├── demo.py                            # Interactive Gradio Super-Resolution Web UI
├── train.py                           # PyTorch SRCNN training script
├── evaluate.py                        # PyTorch evaluation & baseline benchmark script
│
├── notebooks/
│   └── Image_Super_Resolution_SRCNN.ipynb # Self-contained Google Colab & Jupyter notebook
│
├── src/                               # Modular TensorFlow / Keras library
│   ├── __init__.py                    # Package initializer
│   ├── config.py                      # Hyperparameters and path configurations
│   ├── dataset.py                     # DIV2K dataset loader and tf.data pipelines
│   ├── model.py                       # SRCNN, Res-SRCNN, and BatchNorm architectures
│   ├── baselines.py                   # Nearest, Bilinear, and Bicubic baselines
│   ├── metrics.py                     # MSE, PSNR (dB), and SSIM implementations
│   ├── train.py                       # Training loop with callbacks
│   ├── evaluate.py                    # Benchmark evaluation and plotting
│   ├── inference.py                   # Single image inference demonstration
│   └── utils.py                       # Reproducibility seeds and plot helpers
│
├── scripts/                           # CLI Execution scripts
│   ├── download_dataset.py            # Automated DIV2K downloader & cropper
│   ├── train_model.py                 # Standalone model training CLI
│   ├── evaluate_model.py              # Baseline benchmark evaluation CLI
│   ├── generate_poster_assets.py      # Poster figure asset generator
│   ├── generate_poster_pdf.py         # ReportLab academic poster compiler
│   └── run_experiments.py             # Master experimental suite runner
│
├── models/                            # Model architectures & weights
│   ├── srcnn.py                       # PyTorch SRCNN model definition
│   ├── best_srcnn.keras               # Serialized best trained Keras model
│   └── TF_KERAS_README.md             # Keras model metadata
│
├── baselines/                         # PyTorch baselines
│   └── interpolation.py               # Interpolation algorithms (NN, Bilinear, Bicubic)
│
├── data/                              # Dataset download and loading scripts
│   ├── dataset.py                     # PyTorch DIV2K dataset loader
│   └── download_div2k.py              # DIV2K downloader script
│
├── checkpoints/                       # Pretrained PyTorch weights
│   └── srcnn_lr0.001_wd0.0_bnFalse_tanhFalse.pth
│
├── results/                           # Benchmark curves, visuals & CSV logs
│   ├── experiment_results.csv         # Programmatically recorded benchmark metrics
│   ├── training_curves.png            # Loss and PSNR vs Epoch curves
│   ├── lr_comparison.png              # Learning rate ablation comparison
│   ├── qualitative_comparison.png     # Visual test comparisons
│   └── test_sample_1_zoomed.png       # Zoomed-in center patch comparisons
│
├── assets/                            # Poster figures & preview
│   └── poster/
│
└── docs/                              # Academic defense materials
    ├── PROJECT_REPORT.md              # 2-Page college academic project report
    ├── PRESENTATION_CONTENT.md        # 10-Slide presentation deck outline
    ├── VIVA_QUESTIONS.md              # 26 Comprehensive technical viva Q&As
    └── SRCNN_Research_Poster.pdf      # Stanford CS229 style research poster (PDF)
```

---

## 🚀 Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/Dwijesh60/srcnn-super-resolution.git
cd srcnn-super-resolution
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

---

## 🖥️ Running the PyTorch Pipeline & Gradio Demo

### 1. Download Dataset
```bash
python data/download_div2k.py
```

### 2. Train Model
```bash
python train.py --batch_size 2 --epochs 50 --lr 0.001
```

### 3. Evaluate Against Baselines
```bash
python evaluate.py --model_path checkpoints/srcnn_lr0.001_wd0.0_bnFalse_tanhFalse.pth
```

### 4. Launch Interactive Web Demo (Gradio UI)
```bash
python demo.py
```
Open **`http://localhost:7860`** in your browser to upload any image and see super-resolution upscaling in real-time!

---

## ⚙️ Running the TensorFlow / Modular Pipeline

### 1. Download Dataset
```bash
python scripts/download_dataset.py
```

### 2. Run All Experiments End-to-End
```bash
python scripts/run_experiments.py
```

### 3. Generate Academic Research Poster (PDF)
```bash
python scripts/generate_poster_assets.py
python scripts/generate_poster_pdf.py
```
The compiled PDF will be saved at `docs/SRCNN_Research_Poster.pdf`.

---

## 🌐 How to Run in Google Colab
1. Upload `notebooks/Image_Super_Resolution_SRCNN.ipynb` to Google Drive or open directly in [Google Colab](https://colab.research.google.com/).
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Run the first cell (`%pip install ...`) to install dependencies.
4. Run all subsequent cells sequentially to download data, train, and visualize outputs!

---

## 🎓 Academic Documentation & Defense Materials
- 🎨 **Academic Research Poster (PDF):** Stanford CS229 layout in [`docs/SRCNN_Research_Poster.pdf`](docs/SRCNN_Research_Poster.pdf) — ready for printing and presentation.
- 📄 **Project Report:** Full 2-page write-up in [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md).
- 📊 **Presentation Slides:** 10-slide outline with scripts in [`docs/PRESENTATION_CONTENT.md`](docs/PRESENTATION_CONTENT.md).
- ❓ **Viva Preparation:** 26 technical questions & answers in [`docs/VIVA_QUESTIONS.md`](docs/VIVA_QUESTIONS.md).

---

## 📚 References
1. **Garber, Grossman, Johnson-Yu.** *Image Super-Resolution Via a Convolutional Neural Network.* Stanford University CS229 Project Report, Spring 2020. [Report PDF](https://cs229.stanford.edu/proj2020spr/report/Garber_Grossman_Johnson-Yu.pdf) | [Poster PDF](https://cs229.stanford.edu/proj2020spr/poster/Garber_Grossman_Johnson-Yu.pdf)
2. **Dong, C., Loy, C. C., He, K., & Tang, X.** (2016). *Image Super-Resolution Using Deep Convolutional Networks.* IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 38(2), 295–307.
3. **Agustsson, E., & Timofte, R.** (2017). *NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study.* IEEE CVPR Workshops.
4. **Kim, J., Lee, J. K., & Lee, K. M.** (2016). *Accurate Image Super-Resolution Using Very Deep Convolutional Networks (VDSR).* IEEE CVPR.
5. **Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.** (2004). *Image Quality Assessment: From Error Visibility to Structural Similarity.* IEEE Transactions on Image Processing (TIP), 13(4), 600–612.
