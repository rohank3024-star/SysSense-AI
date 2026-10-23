# 🧠 SysSense AI

### AI-Powered Operating System Resource Monitoring, Prediction, and Anomaly Detection System

SysSense AI is an intelligent operating-system monitoring application that tracks live CPU, RAM, disk, network, and process activity. It combines real-time collection with models trained on a time-series dataset to forecast short-term CPU and RAM usage, identify experimental anomaly signals, and present system-rule alerts through an interactive dashboard.

The project is designed to demonstrate Operating System concepts together with Machine Learning and Deep Learning in a practical, faculty-presentable application. Live resource values are collected from the running machine; ML/DL forecasts are clearly separated from those live measurements and are evaluated after their target time has passed.

---

## 🏗️ System Architecture

```
System Metrics (psutil)
        ↓
Collector Service (5-second cadence)
        ↓
SQLite Database (metrics + forecast log)
        ↓
FastAPI Backend
        ↓
ML / DL Forecasting + Anomaly Model + Rule Engine
        ↓
React Dashboard
        ↓
Live Monitoring, Forecasts, Alerts, and Evaluation
```

The backend records current system metrics every five seconds. Forecasts use a 30-second horizon (six collector samples), and each forecast is stored so that it can later be compared with the actual observed CPU/RAM value.

---

## 💻 OS Concepts Demonstrated

- Process Monitoring
- CPU Utilization Tracking
- Memory (RAM) Utilization Tracking
- Disk Usage and Disk I/O Monitoring
- Network I/O Monitoring
- Thread and Process Count Tracking
- System Resource Alerting
- Process Threshold Detection

---

## 🤖 AI / ML Concepts Used

- Supervised Regression (Random Forest)
- Deep Learning Regression (LSTM)
- Isolation Forest Anomaly Detection
- Time-Series Feature Engineering
- Chronological Train/Test Splitting
- TimeSeries Cross-Validation with an Embargo Gap
- Forecast Baseline Comparison
- Post-Forecast Live Evaluation (MAE, RMSE, R², and ±5 percentage-point accuracy)

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone <repository-url>
cd OS_Project
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install backend and ML dependencies

```bash
cd backend
pip install -r requirements.txt
cd ..
```

### 4. Prepare the dataset and train the models

The models are trained from the supplied/generated dataset, as required for the academic project. Run these commands from the project root after dependencies are installed:

```bash
python ml/generate_dataset.py
python ml/train.py
python ml/anomaly_detection.py
python ml/dl_model.py
```

Training creates model artifacts and reports in `ml/models/` and `ml/*_evaluation_report.md`.

### 5. Start the backend

```bash
cd backend
uvicorn main:app --reload
```

Backend API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 6. Start the frontend

```bash
cd ../frontend
npm install
npm run dev
```

Open:


Open the dashboard at [http://localhost:5173](http://localhost:5173).


---

## 📂 Project Structure

```
OS_Project/
│
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── collector.py            # Live metric collector (5-second cadence)
│   ├── database.py             # SQLite metrics and forecast-evaluation storage
│   ├── services/
│   │   └── live_metrics.py     # Cached, interval-based live CPU sampler
│   ├── routes/
│   │   ├── metrics.py          # Metrics and history APIs
│   │   ├── predict.py          # Forecast, anomaly, and evaluation APIs
│   │   └── alerts.py           # System threshold alert APIs
│   └── requirements.txt
│
├── frontend/
│   └── src/
│       ├── pages/
│       │   ├── Dashboard.jsx
│       │   ├── Prediction.jsx
│       │   └── Alerts.jsx
│       └── services/api.js
│
├── ml/
│   ├── dataset/
│   │   └── system_metrics_dataset.csv
│   ├── models/
│   ├── preprocessing.py
│   ├── train.py
│   ├── dl_model.py
│   ├── anomaly_detection.py
│   ├── evaluation_report.md
│   ├── dl_evaluation_report.md
│   └── anomaly_evaluation_report.md
│
├── docs/
│   └── REVIEW_3_PREPARATION.md
│
└── README.md
```

---

## ⚙️ Project Modules

### Module 1: System Metrics Collection

Collects live CPU, RAM, disk, network, thread, and process data using `psutil`.

Metrics are saved every five seconds, matching the feature cadence used by the deployed forecasting pipeline.

### Module 2: Database Storage

Stores metric history in SQLite for graphs and monitoring. A separate forecast log stores each prediction, its expected target time, and the later observed value for live validation.

### Module 3: Random Forest Prediction

Random Forest models are trained on the system-metrics dataset using the 24-feature collector-compatible schema. They forecast CPU and RAM usage approximately 30 seconds ahead and are compared with a persistence baseline.

### Module 4: FastAPI Live Prediction and Evaluation API

FastAPI loads the saved Random Forest, LSTM, and anomaly artifacts and exposes live prediction endpoints. Every fresh forecast is recorded with its expected target time; after the horizon has elapsed, it is matched with the actual collected value to report live MAE, RMSE, R², matched forecasts, and the share within ±5 percentage points.

### Module 5: Anomaly Detection and Process Signals

An Isolation Forest model produces an experimental anomaly signal from collector-compatible features. It is intentionally displayed separately from system threshold alerts and process threshold signals, which are transparent rule-based checks.

### Module 6: Recommendation Engine

Generates actionable, rule-based optimization suggestions from current CPU, RAM, disk, and top-process conditions. Recommendations are not presented as predictions made by the ML models.

### Module 7: React Dashboard and Alerts

The React dashboard displays live resource values, historical trends, forecasts, forecast evaluation, system threshold alerts, and process signals. Live measurements and experimental predictions are explicitly labeled differently.

### Module 8: System Health Score

Calculates a weighted 0–100 health score from current CPU, RAM, disk, and active system alerts. It is a transparent rule-based indicator, not an ML accuracy score.

### Module 9: Deep Learning LSTM Forecasting

Two stacked LSTM regression models use a 10-timestep window of the same 24 collector-compatible features to forecast CPU and RAM approximately 30 seconds ahead. Training uses chronological train/validation/test partitions with embargo gaps and validation-only early stopping.

---

## 🌐 API Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Backend health and model availability |
| `GET /metrics/current` | Current live CPU, RAM, disk, network, and process metrics |
| `GET /metrics/history` | Stored monitoring history for charts |
| `GET /metrics/processes` | Current process resource information |
| `GET /predict` | Dataset-trained RF/LSTM forecasts, persistence baseline, and anomaly result |
| `GET /predict/evaluation` | Live forecast-versus-actual evaluation results |
| `GET /alerts` | System threshold alerts and process threshold signals |
| `GET /alerts/history` | Stored alert history |

Interactive API documentation is available at:

```
http://127.0.0.1:8000/docs
```

---

## 🧪 Machine Learning

### Preprocessing and Feature Alignment

The forecasting models use 24 collector-compatible features: current CPU/RAM/disk usage, lag values, moving averages, changes, rolling standard deviations, network rates, process/thread counts, and time features.

The deployed backend constructs only these supported features. It does not silently insert zero-filled features for inputs that are unavailable on the live machine.

### Dataset

Dataset file:

```
ml/dataset/system_metrics_dataset.csv
```

The dataset is transformed into the same feature schema that the live collector can produce. Data is kept in time order; the target represents the metric six samples (about 30 seconds) in the future.

### Prediction Model — Random Forest

The Random Forest model uses a chronological 80/20 train/test split, a six-sample embargo gap at the boundary, and time-series cross-validation during training.

Held-out test results from the current dataset:

| Target | MAE | RMSE | R² | Improvement over persistence MAE |
|---|---:|---:|---:|---:|
| CPU | 23.02 percentage points | 26.87 | -0.005 | 26.47% |
| RAM | 22.67 percentage points | 26.20 | -0.012 | 25.30% |

The model improves MAE against the persistence baseline on this dataset, but the near-zero/negative R² means it should be presented as an experimental short-horizon forecast rather than a high-accuracy predictor.

### Deep Learning — LSTM

The LSTM uses the same 24-feature contract with a 10-timestep input sequence. It uses chronological train/validation/test partitions and validation-only early stopping.

Held-out test results from the current dataset:

| Target | MAE | RMSE | R² |
|---|---:|---:|---:|
| CPU | 22.99 percentage points | 26.81 | -0.001 |
| RAM | 22.55 percentage points | 26.07 | -0.002 |

Like the Random Forest model, the LSTM should be described honestly as experimental on the present dataset. The dashboard exposes both models so their performance can be compared using later live observations.

### Anomaly Detection

Isolation Forest is trained with 10 collector-compatible features and evaluated on a chronological held-out split. Its current held-out results are:

| Metric | Result |
|---|---:|
| Accuracy | 98.15% |
| Balanced accuracy | 58.88% |
| Precision | 25.00% |
| Recall | 18.52% |
| F1 score | 21.28% |

The dataset is highly imbalanced. Therefore, the overall accuracy alone is not evidence of strong anomaly detection; balanced accuracy, precision, recall, and F1 are included for an honest assessment.

### Live Evaluation During Demonstration

For a faculty presentation, start the backend and allow it to collect data. After at least 30 seconds, open the Prediction page or call `/predict/evaluation`. Present the number of matched forecasts and the live error metrics, rather than claiming that a forecast is accurate before the actual future value is available.

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Uvicorn |
| System Monitoring | psutil |
| Database | SQLite |
| Machine Learning | scikit-learn |
| Deep Learning | PyTorch |
| Frontend | React, Vite |
| Visualization | Recharts |

---

## 🔄 Workflow

```
Dataset → Feature Engineering → Train RF / LSTM / Isolation Forest
                                      ↓
Live System Metrics → SQLite → FastAPI
                                      ↓
      Forecasts + Baseline + Anomaly Signal + Threshold Rules
                                      ↓
               React Dashboard and Live Evaluation
```

---

## 🚀 Future Enhancements

- Train on a longer, more varied real-system dataset.
- Add labelled anomaly scenarios to improve anomaly recall and precision.
- Retrain models periodically using approved collected data.
- Add user-configurable alert thresholds.
- Add authentication and role-based dashboard access.
- Add notification channels such as email or desktop alerts.
- Store metrics in PostgreSQL for larger deployments.
- Containerize the system with Docker.

---

## 📄 License

This project is developed for academic and educational purposes.

---

## 📚 References

- Python psutil Documentation
- FastAPI Documentation
- Scikit-learn Documentation
- PyTorch Documentation
- React Documentation
- Operating System Concepts — Silberschatz, Galvin, and Gagne
