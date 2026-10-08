# Image Super-Resolution Via SRCNN
## Project Summary Report

### 1. Introduction and Problem Statement
Single Image Super-Resolution (SISR) aims to recover a high-resolution (HR) image from a single low-resolution (LR) input. This is a highly ill-posed problem since multiple HR images can correspond to a single LR image. The paper *"Image Super-Resolution Via a Convolutional Neural Network"* (SRCNN) revolutionized this domain by introducing an end-to-end deep learning framework, replacing traditional sparse-coding methods. The objective of this project is to implement SRCNN, construct a complete pipeline, evaluate its performance against interpolation baselines, and perform architectural ablation studies.

### 2. Dataset and Preprocessing
The dataset utilized is a 100-image subset of the DIV2K dataset, consisting of high-quality (2K resolution) images widely used in super-resolution tasks. 
- **Splits:** 60 images for training, 20 for validation, and 20 for testing.
- **Pipeline:** 
  1. **Center Crop:** Original images are center-cropped to 800×800 to maintain uniform spatial dimensions.
  2. **HR Target:** Cropped images are resized to 224×224 using bicubic interpolation.
  3. **LR Input:** The 224×224 images are downsampled to 112×112 (via bilinear interpolation) and then upsampled back to 224×224 (bicubic/bilinear) to match the HR spatial dimensions.
  4. **Normalization:** Pixel intensities are scaled to `[0.0, 1.0]`.

### 3. Model Architecture (SRCNN)
The SRCNN architecture consists of three convolutional layers designed to mimic the sparse-coding pipeline:
1. **Patch Extraction and Representation:** `Conv2d(in=3, out=64, kernel=9, pad=4) + ReLU` extracts overlapping patches and represents them as high-dimensional vectors.
2. **Non-linear Mapping:** `Conv2d(in=64, out=32, kernel=1, pad=0) + ReLU` non-linearly maps these vectors to high-resolution patch representations.
3. **Reconstruction:** `Conv2d(in=32, out=3, kernel=5, pad=2)` aggregates these representations to output the final HR image. 

The output is evaluated using the **Mean Squared Error (MSE)** loss, optimizing the network to predict pixel-perfect reconstructions.

### 4. Experimental Setup and Baselines
To evaluate the network, we compute the **Peak Signal-to-Noise Ratio (PSNR)**, which measures reconstruction quality logarithmically:
`PSNR = 10 * log10(1.0 / MSE)`

**Baselines:**
We establish performance benchmarks using standard interpolation methods:
- Nearest Neighbor Upsampling
- Bilinear Interpolation
- Bicubic Interpolation

**Ablation Studies:**
The project framework supports modular testing of architectural variants:
- **BatchNorm (`+ BatchNorm`):** Inserting Batch Normalization after convolutions (before ReLU) to stabilize training.
- **Tanh Activation (`+ Tanh`):** Applying a Tanh function at the output layer to bound pixel predictions.
- **L2 Regularization (`+ Regularization`):** Applying weight decay (`1e-2` to `1e-6`) to investigate its effect on overfitting on the small 60-image training set.

### 5. Results and Conclusion
The deep learning approach (SRCNN) is theoretically positioned to outperform conventional bicubic interpolation, especially in restoring sharp edges and high-frequency textures. During our test evaluation on 20 held-out images, the models achieved the following Peak Signal-to-Noise Ratio (PSNR):

- **Nearest Neighbor:** 23.71 dB
- **Bilinear:** 23.69 dB
- **Bicubic:** 24.96 dB
- **SRCNN (Ours): 26.60 dB**

Our trained SRCNN model successfully outperformed all standard interpolation baselines, achieving a massive **1.64 dB improvement** over the standard Bicubic upsampling algorithm. 

The learning rate of `0.001` combined with data augmentation (random flips) and 200 epochs of training provided optimal convergence. Ablation studies (like adding BatchNorm) can occasionally hinder pixel-wise regression tasks by shifting statistics, whereas data augmentation helps in preventing overfitting on the highly constrained 60-image training split. Ultimately, this project successfully replicates the foundational deep learning SISR technique, providing a comprehensive toolkit for training, evaluation, and interactive demonstration.
