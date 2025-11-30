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

---

## Training Example

```bash
python -m src.train --data ./data/processed --epochs 50 --batch 64
```

---

## Inference Example

```bash
python -m src.infer --data ./data/test --ckpt ./checkpoints/best.pt
```

---

## Evaluation Metrics

- ROC / AUC
- False Alarm Rate
- **Detection Latency**
- Reconstruction Error Visualization

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

# Condition: UNDER CONSTRUCTION.

