# Viva / Oral Examination Comprehensive Preparation Guide
## Project: Image Super-Resolution Via a Convolutional Neural Network (SRCNN)

This technical viva guide contains **35 rigorously formulated questions and precise answers** directly aligned with our implementation, empirical results, and the Stanford CS229 reference methodology.

---

### Category 1: Problem Definition & Fundamental Concepts

#### Q1: What is Image Super-Resolution (SISR)?
**Answer:** Single-Image Super-Resolution (SISR) is the computer vision process of reconstructing a high-resolution (HR) image from a single degraded low-resolution (LR) observation. The goal is to recover high-frequency spatial details (sharp edges, textures, fine lines) lost due to downsampling, sensor blur, or bandwidth constraints.

#### Q2: Why is Single-Image Super-Resolution formulated as an "ill-posed inverse problem"?
**Answer:** It is ill-posed because downsampling is a non-injective (many-to-one) operation. A single low-resolution pixel can correspond to infinitely many possible high-resolution pixel patterns. Because high-frequency information is irreversibly discarded during downsampling, there is no unique analytical inverse solution. Machine learning overcomes this by learning statistical natural image priors from large datasets.

#### Q3: What does SRCNN stand for, and who introduced it?
**Answer:** SRCNN stands for **Super-Resolution Convolutional Neural Network**. It was introduced by Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang in ECCV 2014 and extended in IEEE TPAMI (2016). It was the pioneering deep learning architecture demonstrating that an end-to-end convolutional neural network outperforms traditional sparse-coding and interpolation methods.

#### Q4: Why generate degraded low-resolution images synthetically from high-resolution originals?
**Answer:** In supervised SISR, we require ground-truth target pairs $(Y_i, X_i)$ to compute the loss function $\mathcal{L}(\hat{X}, X)$. Because acquiring perfectly aligned low- and high-resolution real-world sensor pairs is experimentally difficult, the standard supervised protocol degrades high-resolution images $X$ using a known forward degradation model (bicubic downsampling by scale factor $s=2$) to generate ground-truth training pairs.

#### Q5: What is the difference between image upscaling (interpolation) and deep learning super-resolution?
**Answer:** 
- **Interpolation (Nearest, Bilinear, Bicubic):** Applies fixed, hand-crafted polynomial mathematical formulas based solely on local neighboring pixel distances. It assumes local smoothness and cannot hallucinate or recover high-frequency information that was lost.
- **Deep Learning (SRCNN):** Uses non-linear convolutional filters with learned weights trained on natural images. It recognizes complex textures, patterns, and directional edges, reconstructing plausible high-frequency details that mathematical formulas cannot recover.

---

### Category 2: SRCNN Architecture & Layer Mechanics

#### Q6: Explain the three layers of the SRCNN architecture and their physical interpretations.
**Answer:**
1. **Layer 1: Patch Extraction and Representation**
   - Kernel: $9 \times 9$, 64 filters, ReLU activation.
   - Purpose: Extracts overlapping image patches and projects each $9 \times 9 \times 3$ patch into a 64-dimensional feature representation (equivalent to learning an overcomplete dictionary in sparse coding).
2. **Layer 2: Non-Linear Mapping**
   - Kernel: $1 \times 1$, 32 filters, ReLU activation.
   - Purpose: Maps each 64-dimensional feature vector non-linearly to a 32-dimensional feature space. It performs cross-channel non-linear feature interaction without altering spatial dimensions.
3. **Layer 3: Reconstruction**
   - Kernel: $5 \times 5$, 3 filters, Linear activation.
   - Purpose: Aggregates the non-linearly mapped feature maps over a local $5 \times 5$ neighborhood to reconstruct continuous RGB pixel values matching the target high-resolution image.

#### Q7: Why are kernel sizes 9x9, 1x1, and 5x5 specifically chosen?
**Answer:**
- $9 \times 9$ provides an appropriately wide spatial receptive field to capture meaningful structural primitives (edges, gradients, corners) in the first layer.
- $1 \times 1$ performs non-linear dimensionality compression and feature recombination across channels without spatial distortion.
- $5 \times 5$ provides sufficient local context to smoothly blend neighboring reconstructed patch predictions, preventing blocking and boundary artifacts.

#### Q8: What is the effective receptive field of SRCNN?
**Answer:** The effective receptive field is calculated as:
$$\text{Receptive Field} = (f_1 - 1) + (f_2 - 1) + f_3 = (9 - 1) + (1 - 1) + 5 = 8 + 0 + 5 = 13 \times 13 \text{ pixels}.$$
Each reconstructed output pixel is influenced by a $13 \times 13$ patch in the bicubic input image.

#### Q9: Why is padding='same' used in all convolutional layers?
**Answer:** In the original Dong et al. paper, valid padding was used, which caused the output image to be smaller than the input by $(13 - 1) = 12$ pixels on each dimension. In our implementation and the Stanford CS229 methodology, `padding='same'` is used so the output spatial dimensions $(224 \times 224 \times 3)$ match the target HR dimensions exactly, allowing end-to-end pixel-to-pixel MSE computation without boundary cropping.

#### Q10: How many trainable parameters does our SRCNN model have?
**Answer:**
- Conv 1: $(9 \times 9 \times 3 \times 64) + 64 = 15,552 + 64 = 15,616$ parameters.
- Conv 2: $(1 \times 1 \times 64 \times 32) + 32 = 2,048 + 32 = 2,080$ parameters.
- Conv 3: $(5 \times 5 \times 32 \times 3) + 3 = 2,400 + 3 = 2,403$ parameters.
- **Total:** $15,616 + 2,080 + 2,403 = 20,099$ parameters (~78.5 KB).
Because it is lightweight (~20K parameters), it evaluates with low latency on standard hardware.

---

### Category 3: Metrics & Loss Functions

#### Q11: What is Peak Signal-to-Noise Ratio (PSNR) and how is it calculated?
**Answer:** PSNR is an engineering metric that quantifies image reconstruction fidelity relative to the maximum possible signal power:
$$\text{MSE} = \frac{1}{H \cdot W \cdot C} \sum_{i=1}^H \sum_{j=1}^W \sum_{k=1}^C (X_{i,j,k} - \hat{X}_{i,j,k})^2$$
$$\text{PSNR} = 10 \cdot \log_{10} \left( \frac{\text{MAX}_I^2}{\text{MSE}} \right) = 20 \cdot \log_{10} \left( \frac{\text{MAX}_I}{\sqrt{\text{MSE}}} \right)$$
For images normalized in $[0.0, 1.0]$, $\text{MAX}_I = 1.0$.

#### Q12: Why is higher PSNR better, and what does a 1 dB gain mean?
**Answer:** Because PSNR is inversely proportional to the logarithm of MSE, lower reconstruction error directly yields higher PSNR values. Because the scale is logarithmic (decibels), an improvement of $+1\text{ dB}$ corresponds to approximately an $11\%$ reduction in mean squared reconstruction error. In super-resolution, an improvement of $+0.5\text{ dB}$ to $+1.5\text{ dB}$ is considered noticeable.

#### Q13: What are the fundamental limitations of PSNR?
**Answer:** PSNR treats all pixel errors uniformly. It does not account for the Human Visual System (HVS), which is far more sensitive to structural distortions, luminance variations, and edge coherence than to uniform background noise or slight phase shifts. A blurred image and a slightly shifted sharp image may have similar PSNR values, despite huge perceptual differences.

#### Q14: What is the Structural Similarity Index (SSIM) and how does it improve upon PSNR?
**Answer:** SSIM models human visual perception by measuring three independent components between two image windows $x$ and $y$:
1. **Luminance:** $l(x,y) = \frac{2\mu_x\mu_y + c_1}{\mu_x^2 + \mu_y^2 + c_1}$
2. **Contrast:** $c(x,y) = \frac{2\sigma_x\sigma_y + c_2}{\sigma_x^2 + \sigma_y^2 + c_2}$
3. **Structure:** $s(x,y) = \frac{\sigma_{xy} + c_3}{\sigma_x\sigma_y + c_3}$
$$\text{SSIM}(x,y) = [l(x,y)]^\alpha \cdot [c(x,y)]^\beta \cdot [s(x,y)]^\gamma$$
SSIM values range from $-1$ to $+1$, where $+1$ indicates identical structural properties.

#### Q15: Why is Mean Squared Error (MSE) chosen as the loss function?
**Answer:** Minimizing MSE loss is mathematically equivalent to maximizing PSNR, because PSNR is monotonically decreasing with respect to MSE. Furthermore, MSE is continuously differentiable, convex with respect to linear outputs, and computationally efficient to optimize using gradient descent.

---

### Category 4: Traditional Interpolation Baselines

#### Q16: Explain how Nearest Neighbor interpolation works and its visual drawbacks.
**Answer:** Nearest Neighbor assigns to each query point the intensity of the single geometrically closest pixel:
$$\hat{I}(x, y) = I(\text{round}(x), \text{round}(y))$$
**Drawback:** It produces severe pixelation, staircase edges (aliasing), and jagged block artifacts.

#### Q17: Explain how Bilinear interpolation works.
**Answer:** Bilinear interpolation computes a weighted average of the $2 \times 2$ (4 nearest) pixels using linear interpolation sequentially along the horizontal and vertical axes.
**Drawback:** It smooths blockiness but creates noticeable blurring around sharp edges and gradients.

#### Q18: Explain how Bicubic interpolation works and why it serves as the primary benchmark.
**Answer:** Bicubic interpolation evaluates the $4 \times 4$ (16 nearest) neighboring pixels using a cubic polynomial spline:
$$W(x) = \begin{cases} (a+2)|x|^3 - (a+3)|x|^2 + 1 & \text{for } |x| \le 1 \\ a|x|^3 - 5a|x|^2 + 8a|x| - 4a & \text{for } 1 < |x| < 2 \\ 0 & \text{otherwise} \end{cases}$$
It preserves smoother intensity transitions than bilinear and is the standard industry benchmark for image resampling. However, it cannot reconstruct high-frequency edge detail that was removed during downsampling.

---

### Category 5: Machine Learning Methodology & Training Decisions

#### Q19: Why do we use a 60 / 20 / 20 train / validation / test split?
**Answer:**
- **Train (60 images):** Used by backpropagation to optimize convolution filter weights.
- **Validation (20 images):** Held-out during training to monitor generalization, trigger EarlyStopping, and tune hyperparameters (such as learning rate).
- **Test (20 images):** Completely unexposed during all training and hyperparameter search phases, providing an unbiased evaluation of generalization performance.

#### Q20: Why should the test set never be used during training or hyperparameter tuning?
**Answer:** Tuning hyperparameters (learning rate, architecture, epochs) on test data causes **data leakage** and optimistic bias. The test set must remain untouched until the final model is selected to guarantee an honest evaluation of real-world generalization.

#### Q21: Why did Batch Normalization (BN) hurt SRCNN performance in the Stanford CS229 experiments?
**Answer:** Batch Normalization normalizes feature maps by subtracting batch mean and dividing by batch variance, scaling outputs across the batch. In super-resolution:
1. SISR is an image-to-image regression task requiring exact pixel-level intensity and contrast preservation.
2. Normalizing internal feature maps destroys range flexibility and eliminates absolute luminance information, introducing unnatural color shifts and contrast flattening.
(This insight was later formalized by Lim et al. in EDSR, CVPRW 2017, where removing BN became the gold standard for super-resolution networks).

#### Q22: What happens if the learning rate is set too high ($10^{-2}$) or too low ($10^{-4}$)?
**Answer:**
- **Too high ($10^{-2}$):** Causes gradient instability and parameter oscillation, leading to divergence or sub-optimal local minima with poor validation loss.
- **Too low ($10^{-4}$):** Updates weights very slowly, requiring vastly more epochs to converge and risking getting trapped in saddle points or premature plateauing within the compute budget.
- **Optimal ($10^{-3}$):** Balances rapid initial progress with smooth convergence, matching the Stanford reference findings.

#### Q23: Why did Tanh activation in the final layer fail to improve performance in the reference study?
**Answer:** Tanh squashes outputs strictly into $[-1, 1]$ and suffers from gradient saturation (vanishing gradients) near the saturation asymptotes. When pixel values lie near dynamic extremes (0 or 1), Tanh gradients approach zero, impeding weight updates. A linear output layer allows smooth, unconstrained gradient flow directly from the MSE loss.

---

### Category 6: Comparative Analysis, Limitations & Extensions

#### Q24: What are the primary limitations of the original SRCNN architecture?
**Answer:**
1. **Pre-upsampling computational bottleneck:** SRCNN upsamples the image to $224 \times 224$ *before* feeding it into the network. This forces all convolutional operations to execute in the high-resolution space, increasing computational cost.
2. **Shallow depth:** With only 3 layers and a $13 \times 13$ receptive field, SRCNN cannot model large-scale structural dependencies.
3. **MSE blurriness:** Training solely on MSE loss averages plausible high-frequency textures, producing smooth edges rather than photographic micro-textures.

#### Q25: How do modern architectures improve upon SRCNN?
**Answer:**
1. **FSRCNN (Dong et al., 2016):** Performs convolutions directly in low-resolution space and uses a deconvolution (transposed convolution) layer at the end, accelerating inference by over $40\times$.
2. **ESPCN (Shi et al., 2016):** Introduces sub-pixel convolution (pixel shuffle) for efficient upscaling.
3. **VDSR (Kim et al., 2016):** Uses 20 convolutional layers with residual learning ($\hat{X} = Y + \mathcal{R}(Y)$).
4. **SRGAN / ESRGAN (Ledig et al., 2017; Wang et al., 2018):** Uses Generative Adversarial Networks and Perceptual VGG Loss to synthesize realistic high-frequency photo-realistic textures.
5. **SwinIR (Liang et al., 2021):** Employs Vision Transformers (Swin Transformer) with shifted window self-attention for state-of-the-art super-resolution.

#### Q26: How could this project be extended for a Master's or final-year thesis?
**Answer:**
1. **Perceptual Loss:** Incorporate perceptual loss from a pretrained VGG-19 network ($\mathcal{L}_{\text{perceptual}}$) alongside pixel MSE.
2. **Adversarial Training:** Add a patch-based discriminator to implement a conditional GAN (SRGAN).
3. **Sub-pixel Upsampling:** Replace bicubic pre-upsampling with learnable pixel-shuffle layers.
4. **Real-world Degradation:** Train with non-bicubic blur kernels, sensor noise, and JPEG compression artifacts.
