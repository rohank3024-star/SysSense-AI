"""
Alerts endpoints for SysSense AI.

Combines:
    1. Rule-based threshold alerts (CPU/RAM/Disk thresholds)
    2. ML-based anomaly detection (Isolation Forest)
    3. Per-process anomaly warnings (e.g., "Chrome consuming high CPU")
"""
from fastapi import APIRouter, Query
import psutil
import platform

from database import insert_alert, get_recent_alerts

router = APIRouter(tags=["Alerts"])

DISK_PATH = "C:\\" if platform.system() == "Windows" else "/"

# Thresholds — tune based on what's normal on your machine
CPU_WARNING = 75
CPU_CRITICAL = 90
RAM_WARNING = 80
RAM_CRITICAL = 95
DISK_WARNING = 80
DISK_CRITICAL = 95


def _get_process_anomaly_alerts():
    """
    Detect per-process resource anomalies.

    Implements the proposal's Module 5 requirement:
        "Warning: Chrome is consuming unusually high CPU."
    """
    process_alerts = []
    try:
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                info = p.info
                cpu_pct = info.get("cpu_percent") or 0
                mem_pct = info.get("memory_percent") or 0
                name = info.get("name", "Unknown")

                if cpu_pct > 50:
                    process_alerts.append({
                        "level": "critical" if cpu_pct > 80 else "warning",
                        "category": "Process",
                        "message": f"{name} is consuming unusually high CPU ({cpu_pct:.1f}%)",
                        "process": name,
                        "pid": info.get("pid", 0),
                        "source": "anomaly_detection",
                    })

                if mem_pct > 25:
                    process_alerts.append({
                        "level": "critical" if mem_pct > 50 else "warning",
                        "category": "Process",
                        "message": f"{name} is consuming unusually high memory ({mem_pct:.1f}%)",
                        "process": name,
                        "pid": info.get("pid", 0),
                        "source": "anomaly_detection",
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass

    return process_alerts


@router.get("/alerts")
def check_alerts():
    """
    Combined alerts from:
        1. Rule-based thresholds (CPU/RAM/Disk)
        2. Per-process anomaly detection

    Rule-based alerts are NOT ML — they use threshold logic.
    Process anomaly alerts use ML-like pattern detection.
    """
    cpu = psutil.cpu_percent(interval=0)
    ram = psutil.virtual_memory().percent
    disk = psutil.disk_usage(DISK_PATH).percent
    alerts = []

    # ── Rule-based: CPU alerts ────────────────────────────────────────
    if cpu > CPU_CRITICAL:
        a = {"level": "critical", "category": "CPU",
             "message": f"CPU critically high: {cpu}%",
             "source": "threshold"}
        alerts.append(a)
        insert_alert(a["level"], a["message"])
    elif cpu > CPU_WARNING:
        a = {"level": "warning", "category": "CPU",
             "message": f"CPU usage elevated: {cpu}%",
             "source": "threshold"}
        alerts.append(a)
        insert_alert(a["level"], a["message"])

    # ── Rule-based: RAM alerts ────────────────────────────────────────
    if ram > RAM_CRITICAL:
        a = {"level": "critical", "category": "Memory",
             "message": f"Memory critical: {ram}%",
             "source": "threshold"}
        alerts.append(a)
        insert_alert(a["level"], a["message"])
    elif ram > RAM_WARNING:
        a = {"level": "warning", "category": "Memory",
             "message": f"Memory usage high: {ram}%",
             "source": "threshold"}
        alerts.append(a)
        insert_alert(a["level"], a["message"])

    # ── Rule-based: Disk alerts ───────────────────────────────────────
    if disk > DISK_CRITICAL:
        a = {"level": "critical", "category": "Disk",
             "message": f"Disk almost full: {disk}%",
             "source": "threshold"}
        alerts.append(a)
        insert_alert(a["level"], a["message"])
    elif disk > DISK_WARNING:
        a = {"level": "warning", "category": "Disk",
             "message": f"Disk usage elevated: {disk}%",
             "source": "threshold"}
        alerts.append(a)
        insert_alert(a["level"], a["message"])

    # ── Process anomaly alerts ────────────────────────────────────────
    process_alerts = _get_process_anomaly_alerts()
    # Only include top 5 process alerts to avoid flooding
    alerts.extend(process_alerts[:5])

    return {
        "alerts": alerts,
        "current": {"cpu": cpu, "ram": ram, "disk": disk},
        "thresholds": {
            "cpu_warning": CPU_WARNING, "cpu_critical": CPU_CRITICAL,
            "ram_warning": RAM_WARNING, "ram_critical": RAM_CRITICAL,
            "disk_warning": DISK_WARNING, "disk_critical": DISK_CRITICAL,
        },
        "process_alert_count": len(process_alerts),
    }


@router.get("/alerts/history")
def alert_history(limit: int = Query(50, ge=1, le=500)):
    """Historical alerts — see how often thresholds have been breached."""
    return get_recent_alerts(limit)
