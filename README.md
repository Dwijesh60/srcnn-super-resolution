# Image Super-Resolution Via SRCNN

This repository contains the end-to-end implementation of the paper *"Image Super-Resolution Via a Convolutional Neural Network"* for the Machine Learning Mini-Project.

## 1. Setup & Installation

Create a virtual environment and install the required packages:

```bash
pip install -r requirements.txt
```

## 2. Dataset Preparation

We use a 100-image subset from the DIV2K dataset. The script below downloads the validation set (100 images) and prepares them:

```bash
python data/download_div2k.py
```

## 3. Training the Model

The training script automatically splits the 100 images into train (60), val (20), and test (20).

**Standard Training:**
```bash
python train.py --batch_size 2 --epochs 50 --lr 0.001
```

**Hyperparameter Tuning & Ablations:**
- Vary learning rate: `--lr 0.01` or `--lr 0.0001`
- Add L2 Regularization (Weight decay): `--weight_decay 1e-4`
- Add BatchNorm: `--use_bn`
- Add Tanh Activation: `--use_tanh`

Example:
```bash
python train.py --lr 0.001 --use_bn --weight_decay 1e-4
```

## 4. Evaluation

To evaluate the trained model on the test set and compare it against interpolation baselines (Nearest Neighbor, Bilinear, Bicubic):

```bash
python evaluate.py --model_path checkpoints/srcnn_lr0.001_wd0.0_bnFalse_tanhFalse.pth
```
This script will output PSNR metrics and save visual comparisons in the `results/` directory.

## 5. Demo UI

Launch an interactive Gradio app to upload and super-resolve your own images:

```bash
python demo.py
```
Open the provided URL (e.g., `http://localhost:7860`) in your browser.

## 6. Project Writeup

The 2-page project summary report detailing the approach, dataset, architecture, metrics, and ablation studies can be found in `writeup/summary.md`.
