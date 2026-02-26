# 3. Methodology

## 3.1. Physics-Prior Driven Heteroscedastic Noise Modeling
In the clinical practice of Digital Subtraction Angiography (DSA), the raw acquisition is inherently corrupted by signal-dependent artifacts. To bridge the domain gap between deterministic image restoration and stochastic clinical reality, we formulate the noise synthesis as a **heteroscedastic process**. This model explicitly accounts for the dual-source nature of X-ray imaging: quantum mottle (Poisson-distributed shot noise) and electronic stationary noise (Gaussian-distributed read noise). 

Given a noise-free reconstructed background $I_{bg} \in \mathbb{R}^{B \times 3 \times H \times W}$, the synthesized observation $I_{syn}$ is generated via:

$$I_{syn} = I_{bg} + \left( \epsilon \cdot \sqrt{\alpha \cdot I_{bg} + \beta} \right) \odot M_{soft}$$

where $\alpha$ and $\beta$ represent the predicted gain and stationary noise coefficients, respectively. The term $\epsilon \sim \mathcal{N}(0, 1)$ denotes a **monochromatic standard normal distribution** generated as a single-channel map and broadcasted across RGB channels to ensure grayscale fidelity. $M_{soft}$ is a spatially-smoothed vascular mask that constrains the noise injection to the anatomically relevant regions, mimicking the physical interaction between contrast agents and X-ray attenuation.



## 3.2. Structural Guidance via Adaptive Noise Module (ANM)
To adaptively estimate the noise parameters based on local anatomical context, we propose the **Adaptive Noise Module (ANM)**. The module extracts high-level latent features $\mathcal{F}$ from the network bottleneck and embeds spatial dependencies using **Coordinate Attention (CoordAtt)** to preserve fine-grained vascular structures.

### 3.2.1. Mask-Guided Dual-Path Feature Aggregation (MG-DPFA)
To isolate noise-relevant features from complex anatomical backgrounds, we introduce a **Mask-Guided Dual-Path (MG-DPFA)** strategy. The dilated vascular mask $M_{dil}$ serves as a spatial prior to guide the feature pooling, ensuring that the noise estimation is conditioned on the region of interest (ROI):

$$f_{mean} = \frac{\sum (\text{CoordAtt}(\mathcal{F}) \odot M_{dil})}{\sum M_{dil} + \text{eps}}, \quad f_{max} = \max(\text{CoordAtt}(\mathcal{F}) \odot M_{dil})$$

The resulting descriptor $f_{combined} = [f_{mean} \parallel f_{max}]$ is mapped through decoupled linear heads to regress the physical parameters. This aggregation prevents the "zero-noise collapse" in low-signal-to-noise ratio (SNR) scenarios, ensuring the model remains sensitive to subtle vessel variations.



## 3.3. Statistical Moment Alignment with Saliency Weighting
Since noise is a stochastic variable, deterministic pixel-wise losses (e.g., $L_1$ or $L_2$) are insufficient for capturing the underlying distribution. We implement a **Statistical Moment Alignment Loss** ($\mathcal{L}_{stat}$) that operates on the local statistical manifold to ensure clinical realism.

### 3.3.1. Weighted Log-Variance Loss
The loss enforces consistency between the local moments of the synthesized output $I_{syn}$ and the clinical target $I_{tar}$. We estimate the local variance $\sigma^2$ using an average pooling operator $\phi$ with an $11 \times 11$ sliding window:

$$\text{Var}_{loc}(I) = \phi(I^2) - (\phi(I))^2$$

To address the spatial imbalance caused by the sparsity of the vasculature, a **Spatial Saliency Weight Map** $W$ is derived from the dilated mask. The optimization objective is formulated as:

$$\mathcal{L}_{stat} = \mathbb{E} \left[ W \cdot | \log(\sigma^2_{syn} + \epsilon) - \log(\sigma^2_{tar} + \epsilon) | \right] + \lambda \mathbb{E} \left[ W \cdot | \mu_{syn} - \mu_{tar} | \right]$$

where $\mu$ denotes the local mean. The use of the **logarithmic domain** acts as a Variance Stabilizing Transformation (VST), ensuring balanced sensitivity across varying intensity levels in clinical DSA sequences.



## 3.4. Unbiased Distributional Evaluation
To rigorously validate the stochastic realism without introducing training bias, we employ **Fréchet Inception Distance (FID)** and **Kernel Inception Distance (KID)**. Unlike standard metrics, KID provides an unbiased estimator of the squared Maximum Mean Discrepancy (MMD) between synthesized and real clinical residuals. This evaluation protocol quantifies the alignment of high-order statistical manifolds, proving that the ANM synthesis replicates the complex textures and spatial correlations of real-world X-ray sensors.

---

# Appendix: Writing Strategies & Reviewer Insights for "Artificial Intelligence in Medicine"

### 1. Clinical-Physics Justification (临床物理辩护)
**AIIM Insight:** This journal values the "Medicine" in its title. 
**Suggestion:** Frame the $\alpha I + \beta$ model not just as a loss function, but as a **Digital Twin** component. Explain that this modeling allows the AI to simulate different "exposure doses," which is highly relevant to **Low-Dose Imaging** research—a hot topic in AIIM.

### 2. Handling of "Monochromatic" Grayscale (强调灰度注入的物理意义)
**AIIM Insight:** Medical imaging experts are sensitive to unphysical artifacts like color noise.
**Suggestion:** Explicitly state: "By broadcasting a single-channel noise map, we prevent unphysical chromatic dispersion, ensuring the output aligns with the monochromatic nature of energy-integrating detectors used in digital radiography."

### 3. Log-Domain Stability (解释对数域转换的必要性)
**AIIM Insight:** Numerical stability in clinical data is crucial.
**Suggestion:** Describe the Log-transformation as a method to handle the **dynamic range** of DSA images. Clinical images often have high contrast (bone vs. vessel), and the log-domain loss ensures the AI doesn't ignore noise in the darker, more critical vascular regions.

### 4. ROI-Centric Learning (以病灶/血管为中心的权重逻辑)
**AIIM Insight:** General AI often fails on small, critical structures.
**Suggestion:** Emphasize the $10:1$ saliency ratio as an "Anatomical Importance Weighting." This shows the reviewers that your model is designed with the clinical priority of preserving vascular integrity over background reconstruction.

### 5. Evaluating Beyond Pixels (超越像素对齐的评估)
**AIIM Insight:** SSIM/PSNR often correlate poorly with clinical diagnostic quality.
**Suggestion:** Use FID/KID to argue that your model captures the **texture and 'graininess'** of real X-ray images, which is essential for radiologist acceptance and the training of other downstream diagnostic AI models.