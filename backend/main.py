"""
SysSense AI — FastAPI Backend

The main application entry point. Mounts all route modules and provides
the health score and recommendations endpoints.

Run with:
    uvicorn main:app --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import psutil
import platform

from database import init_db
from routes.metrics import router as metrics_router
from routes.processes import router as processes_router
from routes.alerts import router as alerts_router
from routes.predict import router as predict_router
from services.health_score import calculate_health_score
from services.recommendations import generate_recommendations

DISK_PATH = "C:\\" if platform.system() == "Windows" else "/"

app = FastAPI(
    title="SysSense AI API",
    description="AI-Powered Operating System Monitoring, Prediction & Optimization",
    version="1.0.0",
)

# Allow the React dev server to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database tables
init_db()

# Mount routers
app.include_router(metrics_router)
app.include_router(processes_router)
app.include_router(alerts_router)
app.include_router(predict_router)


@app.get("/", tags=["Root"])
def root():
    return {
        "name": "SysSense AI API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_score():
    """
    Weighted health score (0-100) based on current CPU, RAM, disk.
    Displayed as the main indicator on the dashboard.
    """
    cpu = psutil.cpu_percent(interval=0)
    ram = psutil.virtual_memory().percent
    disk = psutil.disk_usage(DISK_PATH).percent

    # Count active alerts
    from routes.alerts import CPU_WARNING, RAM_WARNING, DISK_WARNING
    active_alerts = sum([
        1 if cpu > CPU_WARNING else 0,
        1 if ram > RAM_WARNING else 0,
        1 if disk > DISK_WARNING else 0,
    ])

    return calculate_health_score(cpu, ram, disk, active_alerts)


@app.get("/recommendations", tags=["Recommendations"])
def recommendations():
    """
    Actionable optimization recommendations based on current system state.
    Phase 1: Rule-based.
    Phase 2: Augmented with ML feature importances.
    """
    cpu = psutil.cpu_percent(interval=0)
    ram = psutil.virtual_memory().percent
    disk = psutil.disk_usage(DISK_PATH).percent

    return generate_recommendations(cpu, ram, disk)


@app.get("/system-info", tags=["System"])
def system_info():
    """Static system information — hardware, OS details."""
    import time
    boot = psutil.boot_time()
    return {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "disk_total_gb": round(psutil.disk_usage(DISK_PATH).total / (1024 ** 3), 2),
        "boot_time": boot,
        "uptime_seconds": round(time.time() - boot),
    }
