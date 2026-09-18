# 🧠 SysSense AI

### AI-Powered Operating System Resource Monitoring, Prediction & Optimization Platform

SysSense AI goes beyond traditional task managers by continuously monitoring your system, storing historical data, predicting future resource usage using machine learning, detecting abnormal behavior, and recommending optimizations — before your system becomes slow.

---

## 🏗️ System Architecture

```
Operating System
        │
        ▼
System Monitoring (psutil)
        │
        ▼
SQLite Database
        │
        ▼
FastAPI Backend
        │
        ├──────────────┐
        ▼              ▼
Machine Learning    Analytics
        │              │
        └──────┬───────┘
               ▼
React Dashboard
```

## 📊 OS Concepts Demonstrated

- **Process Management** — Process monitoring, PCB-equivalent data: PID, state, CPU time
- **CPU Utilization** — Usage patterns, burst behavior, scheduling analysis
- **Memory Management** — RAM tracking, working set analysis, per-process memory
- **Resource Allocation** — Dynamic resource monitoring and allocation tracking
- **Disk Management** — Storage usage monitoring and trend analysis
- **System Performance Monitoring** — Real-time and historical performance metrics
- **Multitasking** — Concurrent process tracking and resource contention detection

## 🤖 AI/ML Concepts Used

- **Time-Series Prediction** — Forecasting CPU/RAM usage 30 seconds ahead
- **Regression** — Random Forest Regressor for resource prediction
- **Anomaly Detection** — Isolation Forest for detecting unusual system behavior
- **Predictive Analytics** — Historical pattern analysis for future load prediction
- **Recommendation System** — Rule-based + ML-driven optimization suggestions

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- npm 9+

### 1. Generate Training Dataset & Train Models (one-time)
```bash
cd ml
pip install pandas numpy scikit-learn joblib
python generate_dataset.py
python train.py
python anomaly_detection.py
```
> This creates the public dataset and trains the ML models once.
> The models are saved as .pkl files in `ml/models/`.

### 2. Start the Data Collector (Terminal 1)
```bash
cd backend
pip install -r requirements.txt
python collector.py
```
> Leave this running to accumulate live data for the dashboard.

### 3. Start the API Server (Terminal 2)
```bash
cd backend
uvicorn main:app --reload
```
> API docs: http://localhost:8000/docs

### 4. Start the Frontend (Terminal 3)
```bash
cd frontend
npm install
npm run dev
```
> Dashboard: http://localhost:5173

---

## 📁 Project Structure

```
SysSense-AI/
├── backend/                    # FastAPI Backend
│   ├── main.py                 # App entry point with health & recommendations
│   ├── collector.py            # Background data collector (psutil)
│   ├── database.py             # SQLite storage layer
│   ├── routes/
│   │   ├── metrics.py          # /metrics/* endpoints
│   │   ├── processes.py        # /processes endpoint
│   │   ├── alerts.py           # /alerts endpoint + process anomalies
│   │   └── predict.py          # /predict endpoint + ML + anomaly detection
│   ├── services/
│   │   ├── health_score.py     # Weighted health score (0-100)
│   │   └── recommendations.py  # Rule-based + ML + anomaly recommendations
│   └── requirements.txt
│
├── frontend/                   # React + Vite Dashboard
│   └── src/
│       ├── pages/              # Dashboard, Processes, Analytics, Prediction, Alerts
│       ├── components/         # Sidebar, MetricCard, LiveChart, HealthScore, ProcessTable
│       └── services/api.js     # Axios API wrapper
│
├── ml/                         # Machine Learning Pipeline
│   ├── generate_dataset.py     # Synthetic dataset generator (public dataset)
│   ├── train.py                # Random Forest training on public dataset
│   ├── dl_model.py             # LSTM deep learning model (PyTorch)
│   ├── preprocessing.py        # Feature engineering
│   ├── evaluation.py           # MAE/RMSE/R² + baseline comparison
│   ├── anomaly_detection.py    # Isolation Forest + process anomalies
│   ├── predict.py              # Model loading & prediction (RF + LSTM)
│   └── data/                   # Generated/downloaded datasets
│       └── system_metrics_dataset.csv
│
└── files_1/                    # Original scaffold (reference)
```

---

## 📦 Project Modules

### Module 1 – System Monitoring
Collects CPU, RAM, Disk, Network, and per-process metrics every few seconds using `psutil`.

### Module 2 – Database
Stores every reading in SQLite with tables for metrics, process snapshots, and alert history.

### Module 3 – AI Prediction Engine
Trained on a **public dataset** (generated via `generate_dataset.py`). The trained model predicts CPU and RAM usage 30 seconds ahead. Saved as `model.pkl`.

### Module 4 – Live Prediction
Current system metrics collected via psutil are passed to the trained model for real-time predictions.

### Module 5 – Anomaly Detection
Uses Isolation Forest to detect unusual resource consumption patterns. Also detects per-process anomalies (e.g., "Chrome is consuming unusually high CPU").

### Module 6 – Recommendation Engine
Provides intelligent suggestions combining rule-based logic, ML feature importances, and anomaly detection results.

### Module 7 – Analytics Dashboard
Displays live CPU, RAM, Disk, Network, running processes, trend graphs, prediction graphs, alerts, and recommendations.

### Module 8 – System Health Score
Generates a weighted health score (0-100) considering CPU, RAM, Disk, and active alerts.

### Module 9 – LSTM Deep Learning Prediction
Uses a 2-layer stacked LSTM model built with PyTorch to predict CPU and RAM usage 30 seconds ahead. The model uses a sliding window of 10 consecutive timesteps to capture temporal dependencies and sequential system patterns.

### Module 10 – UI Theme Toggle
Supports user-configurable **Dark/Light theme switching** across the dashboard, including the sidebar, cards, backgrounds, borders, and text.

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/metrics/current` | GET | Live CPU/RAM/disk/network snapshot |
| `/metrics/history` | GET | Historical readings for charts |
| `/metrics/since?hours=1` | GET | Metrics from last N hours |
| `/metrics/aggregated` | GET | Aggregated averages in time buckets |
| `/processes` | GET | Top processes (sortable, searchable) |
| `/processes/summary` | GET | Process count by status |
| `/alerts` | GET | Threshold + process anomaly alerts |
| `/alerts/history` | GET | Historical alert log |
| `/predict` | GET | ML prediction + anomaly detection |
| `/predict/reload` | GET | Force-reload ML models |
| `/health` | GET | Weighted health score (0-100) |
| `/recommendations` | GET | Optimization suggestions |
| `/system-info` | GET | Static hardware/OS info |

---

## 🤖 Machine Learning

### Dataset
Instead of waiting months to collect training data, we use a **synthetic system performance dataset** that mimics patterns from public datasets (Google Cluster Trace, Azure VM Performance Dataset). The model is trained **once** on this dataset.

### Prediction Model — Machine Learning
- **Algorithm:** Random Forest Regressor
- **Predicts:** CPU and RAM usage 30 seconds ahead
- **Features:** Current values, lag features, moving averages, rate of change, time features
- **Evaluation:** Compared against naive baseline (predicted = current)

### Prediction Model — Deep Learning
- **Algorithm:** 2-Layer Stacked LSTM (Long Short-Term Memory)
- **Framework:** PyTorch
- **Architecture:** LSTM(128) → LSTM(64) → Dense(32) → Output
- **Input:** Sliding window of 10 consecutive timesteps
- **Advantage:** Captures temporal dependencies and sequential patterns that tree-based models treat as independent samples
- **Comparison:** Evaluated against both Random Forest and naive baseline

### Anomaly Detection
- **Algorithm:** Isolation Forest
- **Detects:** Unusual system behavior patterns, resource spikes, per-process anomalies
- **Example:** "Warning: Chrome is consuming unusually high CPU"

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React, Vanilla CSS, Axios, Recharts |
| Backend | FastAPI, Python |
| Database | SQLite |
| Machine Learning | Pandas, NumPy, Scikit-learn, Joblib |
| Deep Learning | PyTorch (LSTM) |
| OS Monitoring | psutil |

---

## 🔄 Workflow

```
Public Dataset
        │
        ▼
Train ML Model (one-time)
        │
        ▼
model.pkl + lstm_model.pth
        │
        ▼
Live System Monitoring (psutil)
        │
        ▼
FastAPI Backend
        │
        ▼
ML + DL Prediction + Anomaly Detection
        │
        ▼
Dashboard + Alerts + Recommendations
```

---

## 🔮 Future Enhancements

- **GPU Monitoring** — Track GPU usage and temperature for ML/gaming workloads
- **Multi-Node Support** — Monitor multiple machines from a single dashboard
- **Docker Container Metrics** — Per-container CPU/RAM tracking
- **Email/Slack Alerts** — Push notifications when anomalies are detected
- **Model Retraining Pipeline** — Auto-retrain models as more live data accumulates

---

## 📄 License

This project is developed for academic purposes as part of an Operating Systems course project.

---

## 📚 References

1. *An Effective Workload Prediction with RNN-LSTM for Efficient Resource Autoscaling in Private Cloud Environments*, IJAACI, 2025. DOI: 10.54216/IJAACI.070105
2. *Elastic Cloud Resource Allocation using Short-Term LSTM-based Workload Prediction*, SPIE Proceedings, 2025. DOI: 10.1117/12.3060861
3. *Application-Oriented Cloud Workload Prediction: A Survey and New Perspectives*, Tsinghua Science & Technology (IEEE), 2025.
4. *Cloud Resource Prediction using Hybrid GRU-LSTM Deep Learning Model*, ResearchGate, 2025.
5. *TFEGRU: Time-Frequency Enhanced GRU with Attention for Cloud Workload Prediction*, IEEE Computer Society, 2024.
6. *Deep Learning Advancements in Anomaly Detection: A Comprehensive Survey*, arXiv, 2025.
7. *A Comprehensive Survey on Anomaly Detection Using Deep Learning*, ResearchGate, 2026.
