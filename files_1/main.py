"""
SysSense FastAPI backend.

Endpoints:
    GET /metrics/current    - live snapshot (CPU/RAM/disk right now)
    GET /metrics/history    - logged history for dashboard charts
    GET /processes          - top processes sorted by CPU or memory
    GET /alerts             - threshold-based alerts (rule-based, not ML)
    GET /predict            - naive baseline for now; swap in your trained
                               ML model's output here once it's ready (Day 20+)

Run with:
    uvicorn main:app --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import psutil

from database import init_db, get_recent_metrics

app = FastAPI(title="SysSense API")

# Allow the React dev server to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

# Tune these based on what actually feels "high" on your machine
CPU_ALERT_THRESHOLD = 90
RAM_ALERT_THRESHOLD = 95


@app.get("/metrics/current")
def current_metrics():
    """Live snapshot - independent of the collector, sampled on request."""
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.5),
        "ram_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent,
    }


@app.get("/metrics/history")
def metrics_history(limit: int = 100):
    """Historical readings written by collector.py - feed this to Recharts."""
    return get_recent_metrics(limit)


@app.get("/processes")
def top_processes(sort_by: str = "cpu", limit: int = 15):
    procs = []
    for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "status"]):
        try:
            procs.append(p.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    key = "cpu_percent" if sort_by == "cpu" else "memory_percent"
    procs.sort(key=lambda x: x.get(key) or 0, reverse=True)
    return procs[:limit]


@app.get("/alerts")
def check_alerts():
    """Rule-based alerts. Keep this separate from /predict in your report -
    this is NOT the ML part, just threshold logic."""
    cpu = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory().percent
    alerts = []
    if cpu > CPU_ALERT_THRESHOLD:
        alerts.append({"level": "warning", "message": f"High CPU usage: {cpu}%"})
    if ram > RAM_ALERT_THRESHOLD:
        alerts.append({"level": "critical", "message": f"Memory critical: {ram}%"})
    return {"alerts": alerts, "cpu": cpu, "ram": ram}


@app.get("/predict")
def predict_placeholder():
    """
    TEMPORARY naive baseline: predicted value = current value.
    Once you train the Random Forest model (Day 20+, once there's enough
    logged history), replace the body of this function with:
        1. load the saved model (joblib)
        2. build the feature row from recent metrics
        3. return model.predict(...)
    Keep this naive version around - you need it to prove your model
    actually beats it, which is the key result for your report.
    """
    return {
        "note": "naive baseline - replace with trained model output",
        "predicted_cpu_30s": psutil.cpu_percent(interval=0.5),
        "predicted_ram_30s": psutil.virtual_memory().percent,
    }
