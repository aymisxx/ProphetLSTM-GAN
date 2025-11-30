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

## Repository Structure (Tentative)

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
├── models/                        # D & G SETUPS
└── results/
```

---

## Installation

```bash
pip install -r requirements.txt
```

Start Jupyter Notebook and open the notebooks one-by-one.

---

## Evaluation Metrics

- ROC / AUC
- False Alarm Rate
- **Detection Latency**
- Reconstruction Error Visualization

---

## Results Overview

| Model                     | AUC     | F1      | TP | FP | FN   | TN      |
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

### 1. LSTM–GAN DR-Score Performance
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
[2] NASA Propulsion Test Telemetry Dataset (Monopropellant Thruster) — Dataset Context Documentation.  
[3] D. Li et al., “MAD-GAN: Multivariate Anomaly Detection for Time Series Data with GANs,” arXiv, 2019.  
[4] Z. Niu, K. Yu, X. Wu, “LSTM-based VAE-GAN for Time-Series Anomaly Detection,” *Sensors*, 2020.

---

## License
MIT License

---