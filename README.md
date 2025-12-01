# 🔒 Network Traffic Anomaly Detection for Embedded Systems

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

A comprehensive **multi-paradigm machine learning** approach for binary classification of network traffic in embedded/IoT systems. This project implements innovative techniques combining **Supervised Learning**, **Unsupervised Anomaly Detection**, and **Deep Learning** to detect malicious network activities.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Innovative Approach](#-innovative-approach)
- [Models Implemented](#-models-implemented)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Key Insights](#-key-insights)
- [Future Work](#-future-work)

---

## 🎯 Overview

This project addresses the challenge of detecting **malicious network traffic** in embedded system environments. The dataset presents a significant class imbalance problem (90% Normal vs 10% Malicious), requiring innovative approaches beyond traditional machine learning.

### Key Challenges Addressed:
- **Class Imbalance**: 90:10 ratio between Normal and Malicious traffic
- **Feature Similarity**: Features have similar distributions across classes
- **Limited Discriminative Power**: Traditional ML models struggle with this dataset

---

## 📊 Dataset

**File**: `embedded_system_network_security_dataset.csv`

| Attribute | Description |
|-----------|-------------|
| **Samples** | 1,000 network traffic records |
| **Features** | 17 original features |
| **Classes** | Binary (Normal: 0, Malicious: 1) |
| **Imbalance** | 90% Normal, 10% Malicious |

### Original Features:
- `packet_size` - Size of network packet
- `inter_arrival_time` - Time between consecutive packets
- `spectral_entropy` - Entropy measure of packet content
- `frequency_band_energy` - Energy distribution across frequency bands
- `protocol_type` - TCP/UDP/ICMP
- `src_ip`, `dst_ip` - Source and destination IP addresses
- `dst_port` - Destination port number
- `tcp_flags` - TCP flag indicators (SYN, ACK, FIN, RST, PSH)
- `packet_count_5s` - Packet count in 5-second window

### Engineered Features (Domain Knowledge):
| Feature | Security Interpretation |
|---------|------------------------|
| `traffic_intensity` | packet_size / inter_arrival_time |
| `packet_rate` | packet_count_5s / inter_arrival_time |
| `burst_indicator` | High packet count detection |
| `entropy_energy_ratio` | Spectral entropy / frequency energy |
| `syn_without_ack` | SYN flood attack indicator |
| `port_range` | Port categorization (well-known, registered, dynamic) |

---

## 🧠 Innovative Approach

### Multi-Paradigm Learning Strategy

| Paradigm | Approach | Rationale |
|----------|----------|-----------|
| **Supervised** | Stacking Ensemble | Meta-learning combining multiple base learners |
| **Unsupervised** | Anomaly Detection | Learn normal patterns, detect deviations |
| **Deep Learning** | AutoEncoder | Reconstruction-based anomaly scoring |
| **Hybrid** | Weighted Ensemble | Combine all paradigms with performance-based weights |

### What Makes This Different?

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRADITIONAL APPROACH                         │
│  Dataset → Single Model → Threshold 0.5 → Classification       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    OUR INNOVATIVE APPROACH                      │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐       │
│  │  Supervised  │   │ Unsupervised │   │Deep Learning │       │
│  │   Stacking   │   │   Anomaly    │   │  AutoEncoder │       │
│  │   Ensemble   │   │  Detection   │   │              │       │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘       │
│         │                  │                  │                │
│         └────────────┬─────┴──────────────────┘                │
│                      ▼                                         │
│         ┌────────────────────────┐                             │
│         │   Hybrid Weighted      │                             │
│         │      Ensemble          │                             │
│         │  (Performance-based)   │                             │
│         └────────────────────────┘                             │
│                      │                                         │
│                      ▼                                         │
│         ┌────────────────────────┐                             │
│         │  Adaptive Threshold    │                             │
│         │    Optimization        │                             │
│         └────────────────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Models Implemented

### 1. Supervised Learning - Stacking Ensemble
```
Base Learners:
├── Random Forest
├── XGBoost  
├── LightGBM
├── Extra Trees
└── MLP Classifier

Meta-Learner: Logistic Regression
```

### 2. Unsupervised Anomaly Detection
- **Isolation Forest** - Isolates anomalies using random partitioning
- **One-Class SVM** - Learns boundary around normal data
- **Local Outlier Factor (LOF)** - Density-based anomaly detection

### 3. Deep Learning
- **AutoEncoder** - Learns compressed representation of normal traffic
  - Architecture: Input → 32 → 16 → 8 → 16 → 32 → Output
  - Anomaly Score: Reconstruction Error (MSE)
  - Training: Only on normal samples (semi-supervised)

### 4. Hybrid Ensemble
- **Hard Voting** - Majority vote from all models
- **Soft Voting** - Average probability with threshold optimization
- **Weighted Ensemble** - Performance-based weighted combination

---

## 📈 Results

### Final Model Performance

| Model | F1-Score | ROC-AUC | MCC | Recall |
|-------|----------|---------|-----|--------|
| **Hybrid (Weighted Ensemble)** | **0.250** | **0.594** | **0.161** | **0.700** |
| Deep Learning (AutoEncoder) | 0.233 | 0.609 | 0.141 | 0.250 |
| Anomaly (One-Class SVM) | 0.098 | 0.526 | -0.045 | 0.150 |
| Anomaly (LOF) | 0.080 | 0.508 | -0.047 | 0.100 |
| Anomaly (Isolation Forest) | 0.039 | 0.468 | -0.097 | 0.050 |

### Key Findings:
- **Hybrid Weighted Ensemble** achieves highest F1-Score (0.250) with 70% recall
- **AutoEncoder** shows best ROC-AUC (0.609), indicating good discrimination ability
- Traditional supervised methods fail (F1=0) due to class imbalance and feature similarity

---

## 🛠 Installation

### Prerequisites
```bash
Python 3.8+
pip or conda
```

### Install Dependencies
```bash
pip install pandas numpy scikit-learn
pip install xgboost lightgbm catboost
pip install torch torchvision
pip install imbalanced-learn
pip install matplotlib seaborn
```

### Or use requirements (if available)
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Run the Analysis
Open `main.ipynb` in Jupyter Notebook or VS Code and run all cells sequentially.

### 2. Load Saved Models
```python
import pickle
import torch

# Load sklearn models
with open('model_stacking_ensemble.pkl', 'rb') as f:
    stacking_model = pickle.load(f)

with open('model_isolation_forest.pkl', 'rb') as f:
    iso_forest = pickle.load(f)

# Load AutoEncoder
class AutoEncoder(nn.Module):
    # ... (define architecture)
    pass

autoencoder = AutoEncoder(input_dim=17)
autoencoder.load_state_dict(torch.load('model_autoencoder.pth'))
```

### 3. Make Predictions
```python
# Preprocess new data
from sklearn.preprocessing import StandardScaler
scaler = pickle.load(open('scaler_final.pkl', 'rb'))
X_new_scaled = scaler.transform(X_new)

# Predict
predictions = stacking_model.predict(X_new_scaled)
```

---

## 📁 Project Structure

```
tubes_eas_sains_data/
│
├── 📓 main.ipynb                          # Main analysis notebook
├── 📊 embedded_system_network_security_dataset.csv  # Dataset
├── 📖 README.md                           # Project documentation
│
├── 🤖 Models/
│   ├── model_stacking_ensemble.pkl        # Stacking Ensemble model
│   ├── model_isolation_forest.pkl         # Isolation Forest model
│   ├── model_one_class_svm.pkl            # One-Class SVM model
│   ├── model_lof.pkl                      # Local Outlier Factor model
│   ├── model_autoencoder.pth              # AutoEncoder (PyTorch)
│   └── model_autoencoder_engineered.pth   # AutoEncoder with engineered features
│
├── 🔧 Preprocessing/
│   └── scaler_final.pkl                   # StandardScaler
│
├── 📊 Visualizations/
│   ├── model_comparison_advanced.png      # Model comparison charts
│   ├── feature_importance_advanced.png    # Feature importance analysis
│   ├── feature_engineering_comparison.png # Original vs Engineered features
│   └── final_summary_dashboard.png        # Final results dashboard
│
└── 📋 project_summary.json                # Project metadata
```

---

## 💡 Key Insights

### 1. Dataset Characteristics
- Features have **very similar distributions** between Normal and Malicious classes
- Statistical tests show most features are **not significantly different** (p-value > 0.05)
- Only `packet_count_5s` shows statistical significance (p = 0.0003)

### 2. Why Traditional ML Fails
- Standard classifiers predict all samples as "Normal" (majority class)
- Default threshold (0.5) is inappropriate for imbalanced data
- Features lack discriminative power

### 3. Solution: Multi-Paradigm Approach
- **Semi-supervised learning**: Train anomaly detectors on normal data only
- **Threshold optimization**: Find optimal decision boundary
- **Ensemble methods**: Combine diverse models for robust predictions

### 4. Feature Importance (Top 5)
1. `packet_rate` - Highest sensitivity in AutoEncoder
2. `dst_port` - Critical for identifying attack targets
3. `burst_indicator` - DoS/DDoS detection indicator
4. `protocol_type` - Protocol-based anomaly patterns
5. `entropy_energy_ratio` - Signal characteristic anomalies

---

## 🔮 Future Work

1. **Variational AutoEncoder (VAE)** - Generative anomaly detection
2. **Attention Mechanism** - Sequence-aware traffic analysis
3. **Online Learning** - Adaptive threshold for streaming data
4. **Graph Neural Networks** - Network topology-aware detection
5. **Federated Learning** - Privacy-preserving distributed training

---

## 📚 References

- [Isolation Forest Paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
- [AutoEncoder for Anomaly Detection](https://arxiv.org/abs/2003.05912)
- [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813)
- [Stacking Ensemble Methods](https://www.sciencedirect.com/science/article/pii/S0957417421009234)

---

## 👤 Author

**Network Traffic Anomaly Detection Project**  
Tugas Besar - Sains Data  
2025

---

## 📄 License

This project is for educational purposes.

---

<p align="center">
  <b>🛡️ Securing Embedded Systems Through Intelligent Network Analysis 🛡️</b>
</p>
