# 🧠 SysSense AI

### AI-Powered Operating System Monitoring, Prediction & Optimization Platform

SysSense AI goes beyond traditional task managers by continuously monitoring your system, storing historical data, predicting future resource usage using machine learning, detecting abnormal behavior, and recommending optimizations — before your system becomes slow.

---

## 🏗️ Architecture

```
                    Windows Operating System
                              │
                         psutil Library
                              │
               ┌──────────────┴──────────────┐
               │                             │
         Live Monitoring               Background Collector
               │                             │
               └──────────────┬──────────────┘
                              │
                        SQLite Database
                              │
                      FastAPI Backend
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
      React Dashboard     ML Prediction     Alert Engine
            │                 │                 │
            └─────────────────┴─────────────────┘
                              │
                          User Interface
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- npm 9+

### 1. Start the Data Collector (Terminal 1)
```bash
cd backend
pip install -r requirements.txt
python collector.py
```
> Leave this running to accumulate data for ML training.

### 2. Start the API Server (Terminal 2)
```bash
cd backend
uvicorn main:app --reload
```
> API docs: http://localhost:8000/docs

### 3. Start the Frontend (Terminal 3)
```bash
cd frontend
npm install
npm run dev
```
> Dashboard: http://localhost:5173

### 4. Train ML Models (once enough data is collected)
```bash
cd ml
python train.py
python anomaly_detection.py
```

---

## 📁 Project Structure

```
SysSense-AI/
├── backend/                    # FastAPI Backend
│   ├── main.py                 # App entry point with health & recommendations
│   ├── collector.py            # Background data collector
│   ├── database.py             # SQLite storage layer
│   ├── routes/
│   │   ├── metrics.py          # /metrics/* endpoints
│   │   ├── processes.py        # /processes endpoint
│   │   ├── alerts.py           # /alerts endpoint
│   │   └── predict.py          # /predict endpoint + ML integration
│   ├── services/
│   │   ├── health_score.py     # Weighted health score (0-100)
│   │   └── recommendations.py  # Rule-based + ML recommendations
│   └── requirements.txt
│
├── frontend/                   # React + Vite Dashboard
│   └── src/
│       ├── pages/              # Dashboard, Processes, Analytics, Prediction, Alerts
│       ├── components/         # Sidebar, MetricCard, LiveChart, HealthScore, ProcessTable
│       └── services/api.js     # Axios API wrapper
│
├── ml/                         # Machine Learning Pipeline
│   ├── train.py                # Random Forest training
│   ├── preprocessing.py        # Feature engineering
│   ├── evaluation.py           # MAE/RMSE/R² + baseline comparison
│   ├── anomaly_detection.py    # Isolation Forest
│   └── predict.py              # Model loading & prediction
│
└── files_1/                    # Original scaffold (reference)
```

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
| `/alerts` | GET | Rule-based threshold alerts |
| `/alerts/history` | GET | Historical alert log |
| `/predict` | GET | ML prediction (or naive baseline) |
| `/health` | GET | Weighted health score (0-100) |
| `/recommendations` | GET | Optimization suggestions |
| `/system-info` | GET | Static hardware/OS info |

---

## 🤖 Machine Learning

### Prediction Model
- **Algorithm:** Random Forest Regressor
- **Predicts:** CPU and RAM usage 30 seconds ahead
- **Features:** Current values, lag features, moving averages, rate of change, time features
- **Evaluation:** Compared against naive baseline (predicted = current)

### Anomaly Detection
- **Algorithm:** Isolation Forest
- **Detects:** Unusual system behavior patterns, resource spikes, potential malware activity
- **Connects to:** OS security concepts (intrusion detection via resource patterns)

---

## 📊 OS Concepts Demonstrated

- **Process Management** — PCB-equivalent data: PID, state, CPU time
- **Memory Management** — RAM tracking, working set analysis
- **CPU Scheduling** — Usage patterns, burst behavior
- **Resource Monitoring** — System calls through psutil (wraps WinAPI on Windows)
- **Security** — Anomaly detection via resource pattern analysis

---

## 👥 Team

| Member | Responsibilities |
|---|---|
| Member A | Systems + ML: collector, SQLite, FastAPI, ML pipeline |
| Member B | Frontend: React dashboard, charts, API integration, testing |
