"""
Alerts endpoints for SysSense AI.

Rule-based threshold alerts — explicitly NOT the ML part.
Keep this distinction clear in your report.
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


@router.get("/alerts")
def check_alerts():
    """
    Rule-based alerts.
    This is NOT ML — just threshold logic. Be clear about this in your report.
    """
    cpu = psutil.cpu_percent(interval=0)
    ram = psutil.virtual_memory().percent
    disk = psutil.disk_usage(DISK_PATH).percent
    alerts = []

    # CPU alerts
    if cpu > CPU_CRITICAL:
        a = {"level": "critical", "category": "CPU",
             "message": f"CPU critically high: {cpu}%"}
        alerts.append(a)
        insert_alert(a["level"], a["message"])
    elif cpu > CPU_WARNING:
        a = {"level": "warning", "category": "CPU",
             "message": f"CPU usage elevated: {cpu}%"}
        alerts.append(a)
        insert_alert(a["level"], a["message"])

    # RAM alerts
    if ram > RAM_CRITICAL:
        a = {"level": "critical", "category": "Memory",
             "message": f"Memory critical: {ram}%"}
        alerts.append(a)
        insert_alert(a["level"], a["message"])
    elif ram > RAM_WARNING:
        a = {"level": "warning", "category": "Memory",
             "message": f"Memory usage high: {ram}%"}
        alerts.append(a)
        insert_alert(a["level"], a["message"])

    # Disk alerts
    if disk > DISK_CRITICAL:
        a = {"level": "critical", "category": "Disk",
             "message": f"Disk almost full: {disk}%"}
        alerts.append(a)
        insert_alert(a["level"], a["message"])
    elif disk > DISK_WARNING:
        a = {"level": "warning", "category": "Disk",
             "message": f"Disk usage elevated: {disk}%"}
        alerts.append(a)
        insert_alert(a["level"], a["message"])

    return {
        "alerts": alerts,
        "current": {"cpu": cpu, "ram": ram, "disk": disk},
        "thresholds": {
            "cpu_warning": CPU_WARNING, "cpu_critical": CPU_CRITICAL,
            "ram_warning": RAM_WARNING, "ram_critical": RAM_CRITICAL,
            "disk_warning": DISK_WARNING, "disk_critical": DISK_CRITICAL,
        },
    }


@router.get("/alerts/history")
def alert_history(limit: int = Query(50, ge=1, le=500)):
    """Historical alerts — see how often thresholds have been breached."""
    return get_recent_alerts(limit)
