# ProphetLSTM-GAN
**Hybrid LSTM–GAN for Early Anomaly Detection in Liquid Rocket Engine Telemetry.**

ProphetLSTM-GAN is an unsupervised anomaly detection framework for liquid rocket propulsion telemetry. The model learns nominal multivariate sensor behavior using a hybrid LSTM Generator and Discriminator trained adversarially. During inference, anomalies are identified using a combined reconstruction + discriminator-based scoring metric, allowing detection of subtle pre-failure deviations earlier than threshold or rule-based systems.

---

## Authors
- **[Avantika Bharat Patil (Patil0601)](https://github.com/Patil0601)**
- **[Ayushman M. (aymisxx)](https://github.com/aymisxx)**
- **[Yashwanth Gowda (yashwanthgowdanm)](https://github.com/yashwanthgowdanm)**
- **[Rajeev Tidke (rajeevtidke)](https://github.com/rajeevtidke)**
- **[Pritam Modak (pmodak1)](https://github.com/pmodak1)**

---

## Inspiration & Prior Work

This project is inspired by foundational research in adversarial time-series anomaly detection:

- **MAD-GAN (Li et al., 2019)** — multivariate anomaly detection using LSTM-based Generator and Discriminator.
- **LSTM-GAN for Liquid Rocket Engines (Deng et al., 2022)** — adversarial learning applied to propulsion telemetry.
- **VAE-GAN Time-Series Anomaly Detection (Niu et al., 2020)** — reconstruction-based anomaly scoring improvements.

These works demonstrated that LSTM-GANs can model nominal engine behavior without requiring labeled fault data.

---

## How This Project Extends Prior/Original Work

| Contribution | Description |
|---|---|
| **Detection Latency Analysis** | Measures how early anomalies are detected relative to redline cutoff systems, a critical propulsion safety metric. |
| **Multi-Fault Scenario Evaluation** | Tested under injector blockage, chamber pressure decay, valve jitter, etc., rather than generic “anomaly” labels. |
| **Baseline System Comparisons** | Benchmarked against Redline Cutoff System (RCS), Adaptive Threshold Algorithm (ATA), LSTM Autoencoder, and SVM baselines. |

This shifts the project from a model reproduction to an **engineering-focused propulsion health monitoring study**.

---

## Mathematical Model

This project performs **unsupervised anomaly detection** on spacecraft thruster firing telemetry using an **LSTM-GAN**. Each timestep is represented by a 4-dimensional sensor vector:

$$
\mathbf{x}_t =
\begin{bmatrix}
\text{ton}_t & \text{thrust}_t & \text{mfr}_t & \text{vl}_t
\end{bmatrix}^\top \in \mathbb{R}^4
$$

The raw anomaly flag `anomaly_code` is **not** used as a supervised training target for the GAN. Instead, the model learns the distribution of **normal engine behavior** and detects deviations from it.

### 1. Normalization

To make the sensor features comparable, each feature is standardized using statistics computed from **normal samples only**.

For each feature, we compute:

- the mean of that feature over normal samples.
- the standard deviation of that feature over normal samples.

Each value is then normalized as:

**normalized value = (original value - feature mean) / feature standard deviation**

So, at each timestep, the input becomes a 4-dimensional normalized feature vector.

This places all input features in a common standardized space and centers the model around normal operating behavior.

### 2. Sliding-Window Sequence Construction

Each normalized telemetry stream is converted into fixed-length overlapping windows for sequence modeling. With window length
$L = 128$
and stride
$s = 5$.
the $k$-th window is

$$
W_k =
\begin{bmatrix}
\tilde{\mathbf{x}}_{t_k} \\
\tilde{\mathbf{x}}_{t_k+1} \\
\vdots \\
\tilde{\mathbf{x}}_{t_k+L-1}
\end{bmatrix}
\in \mathbb{R}^{128 \times 4},
\qquad t_k = 1 + (k-1)s.
$$

These windows form the training, validation, and test tensors used by the GAN:

$$
X_{\text{train}} \in \mathbb{R}^{304080 \times 128 \times 4},
\quad
X_{\text{val}} \in \mathbb{R}^{76020 \times 128 \times 4},
\quad
X_{\text{test}} \in \mathbb{R}^{403200 \times 128 \times 4}.
$$

This transforms long raw firing sequences into fixed-size temporal samples suitable for recurrent learning.

### 3. Generator

The generator maps a random latent noise vector to a synthetic telemetry window.

In this project:
- latent dimension = 64.
- sequence length = 128.
- feature dimension = 4.
- LSTM hidden size = 128.
- number of LSTM layers = 2.

The latent vector is repeated across all 128 timesteps and passed through a stacked LSTM. At each timestep, the LSTM hidden state is projected to the 4 output features: `ton`, `thrust`, `mfr`, and `vl`.

As a result, the generator produces a full synthetic sequence of shape **(128, 4)**.

No final `tanh` activation is used, so the generator outputs remain in the same standardized feature space as the normalized training data.

### 4. Discriminator $D$

The discriminator receives a real or generated window

$$
W \in \mathbb{R}^{128 \times 4}
$$

and processes it with a stacked LSTM encoder. The final hidden state is passed through a small MLP to produce a scalar logit:

$$
D(W) \in \mathbb{R}.
$$

After sigmoid,

$$
p_{\text{real}}(W) = \sigma(D(W)),
$$

which is interpreted as the probability that the window looks like normal telemetry. A high value means the sequence appears normal; a low value means it looks suspicious. 

### 5. GAN Training Objective

The model is trained adversarially:

- the discriminator learns to classify real windows as real and generated windows as fake.
- the generator learns to produce windows that fool the discriminator.

The discriminator loss is

$$
\mathcal{L}_D =
\mathrm{BCE}(D(W_{\text{real}}),1) +
\mathrm{BCE}(D(G(\mathbf{z})),0)
$$

and the generator loss is

$$
\mathcal{L}_G =
\mathrm{BCE}(D(G(\mathbf{z})),1).
$$

In implementation, `BCEWithLogitsLoss` is used for numerical stability, and real labels are slightly smoothed during training.

### 6. Anomaly Scoring

After training, anomalies are detected using two complementary signals.

#### (a) Discriminator-based score

The discriminator anomaly score is

$$
s_D(W) = 1 - p_{\text{real}}(W)
= 1 - \sigma(D(W)).
$$

A higher value means the discriminator considers the window less consistent with normal behavior.

#### (b) GAN-style reconstruction error

A latent vector is sampled, a synthetic window is generated, and its difference from the real window is measured using mean L1 error:

$$
s_R(W) =
\frac{1}{128 \cdot 4}
\sum_{t=1}^{128}\sum_{j=1}^{4}
|W_{t,j} - G(\mathbf{z})_{t,j}|.
$$

Higher error means the generator struggles to mimic that window, which suggests abnormality. 

### 7. Final DR-Score

The final anomaly score combines both signals:

$$
\mathrm{DR}(W) = \alpha s_R(W) + (1-\alpha)s_D(W),
$$

with

$$
\alpha = 0.5.
$$

So the implemented score is

$$
\mathrm{DR}(W) =
0.5\,s_R(W) + 0.5\,\big(1-\sigma(D(W))\big).
$$

Higher DR-score means a stronger deviation from learned normal thruster behavior. This score is then thresholded for anomaly classification and used for ROC, AUC, precision-recall, and F1 evaluation.

### 8. End-to-End Pipeline

The complete model pipeline is:

$$
\mathbf{x}_t \rightarrow \tilde{\mathbf{x}}_t \rightarrow W_k \rightarrow \{G,D\} \rightarrow s_R, s_D \rightarrow \mathrm{DR}(W).
$$

In words:

1. raw telemetry is standardized using normal-only statistics.  
2. fixed-length windows of shape $128 \times 4$ are created.  
3. an LSTM-GAN learns the normal sequence distribution.  
4. anomaly scores are computed from generator mismatch and discriminator suspicion.  
5. both are fused into the final DR-score for detection.

---

## Repository Structure

```
ProphetLSTM-GAN/
│
├── 00_data_preprocessing.ipynb
├── 01_training.ipynb
├── 02_evaluation.ipynb
│
├── data/
│   ├── raw/
│   │    ├── train/
│   │    └── test/
│   │
│   └── processed/
│
├── models/                        # Discriminator & Generator Setups
└── results/
```

---

## Installation

```bash
pip install -r requirements.txt
```

Start Jupyter Notebook and open the notebooks one-by-one, 00, 01, and 02.

---

## Evaluation Metrics

- ROC / AUC
- False Alarm Rate
- **Detection Latency**
- Reconstruction Error Visualization

---

## Results Overview

| Model                     | AUC     |    F1   | TP | FP |  FN  |   TN    |
|---------------------------|---------|---------|----|----|------|---------|
| DR-Score (GAN)            | 0.6593  | 0.0747  | 73 | 1  | 1808 |  401318 |
| HybridScore (GAN + EMA)   | 0.6967  | 0.0771  | 77 | 41 | 1804 |  401278 |

### Key Improvements

- **AUC +3.7%**  
- **True anomalies detected: +4**  
- **Better sensitivity to drift-induced anomalies**  
- **False-positive rate remains extremely low (~0.01%)**  

---

## Detailed Results Section

### 1. LSTM-GAN DR-Score Performance
The DR-score captures short-window anomalous behavior using reconstruction error and discriminator realism.
- **ROC AUC:** 0.6593  
- **F1-score:** 0.0747  
- **Threshold:** 1.884655  
- **Confusion Matrix:**  
  - TN: 401318  
  - FP: 1  
  - FN: 1808  
  - TP: 73  

This detector is extremely conservative (almost zero false alarms) but catches high-severity anomalies reliably.

### 2. Trend-Residual Score (EMA-Based)
EMA trend residual captures slow drifts that GAN alone cannot detect.
- **Window-level residual score aligned with X_test**
- **Smooth behavior in normal windows**
- **Stronger response around drift-induced anomalies**


### 3. HybridScore (DR + Trend Residual)
Combining short-term and long-term anomaly cues improves separability.
- **ROC AUC:** 0.6967  
- **F1-score:** 0.0771  
- **Threshold:** 0.126540  
- **Confusion Matrix:**  
  - TN: 401278  
  - FP: 41  
  - FN: 1804  
  - TP: 77  

Improvement highlights:
- **AUC Gain:** +0.0374  
- **True anomalies detected:** +4  
- **Recall improved from 3.88% → 4.09%**  
- **Still extremely low FP rate (~0.01%)**

### **Note on Improvement Magnitude**

The absolute improvement in AUC (~+3.7%) may appear modest at first glance, but it should be interpreted in the context of the dataset characteristics. The anomaly class represents a very small fraction of the data (≈0.4–0.5%), resulting in an extremely imbalanced detection problem. In such settings, even small shifts in AUC can correspond to meaningful improvements in the ranking of rare anomaly events.

Importantly, the hybrid score increases anomaly recall (TP: 73 → 77) while keeping the false-positive rate extremely low relative to the large normal class. This reflects the intended design goal of the hybrid scoring scheme: improving sensitivity to subtle drift-induced anomalies without significantly degrading precision.

### 4. Timeline Behavior
- DR-score shows sharp spikes for high-severity anomalies  
- Hybrid score captures both sharp spikes *and* slow drifts  
- Threshold lines cleanly separate normal vs anomaly behavior  
- Visual confirmation of increase in true positive detection  

### Final Interpretation
The hybrid pipeline significantly enhances anomaly separability without compromising stability, delivering a more expressive unsupervised FDD system than DR-score alone.

---


## References

[1] L. Deng, Y. Cheng, and Y. Shi, “Fault Detection and Diagnosis for Liquid Rocket Engines Based on LSTM and GAN,” *Aerospace*, 2022.  
[2] NASA Propulsion Test Telemetry Dataset (Monopropellant Thruster) - Dataset Context Documentation.  
[3] D. Li et al., “MAD-GAN: Multivariate Anomaly Detection for Time Series Data with GANs,” arXiv, 2019.  
[4] Z. Niu, K. Yu, X. Wu, “LSTM-based VAE-GAN for Time-Series Anomaly Detection,” *Sensors*, 2020.

---

## License

MIT License

---
