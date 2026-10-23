"""Rule-based alerts for live OS telemetry.

This endpoint intentionally serves transparent threshold rules only. The
separate Isolation Forest signal comes from ``/predict`` so a rule alert is
never mislabeled as a machine-learning anomaly.
"""
import platform

import psutil
from fastapi import APIRouter, Query

from database import get_recent_alerts, insert_alert
from services.live_metrics import get_live_cpu_percent


router = APIRouter(tags=["Alerts"])

DISK_PATH = "C:\\" if platform.system() == "Windows" else "/"
IDLE_PROCESS_NAMES = {"system idle process", "idle"}

# Tune these documented rules for the demonstration computer if necessary.
CPU_WARNING = 75
CPU_CRITICAL = 90
RAM_WARNING = 80
RAM_CRITICAL = 95
DISK_WARNING = 80
DISK_CRITICAL = 95
PROCESS_CPU_WARNING = 50
PROCESS_MEMORY_WARNING = 25


def _get_process_threshold_alerts():
    """Return fixed-rule process signals, excluding Windows idle time."""
    process_alerts = []
    try:
        for process in psutil.process_iter(
            ["pid", "name", "cpu_percent", "memory_percent"]
        ):
            try:
                info = process.info
                pid = info.get("pid", 0)
                name = info.get("name") or "Unknown"
                if pid == 0 or name.casefold() in IDLE_PROCESS_NAMES:
                    # Windows exposes idle time as a pseudo-process. Flagging it
                    # as high CPU would be a false positive.
                    continue

                cpu_percent = float(info.get("cpu_percent") or 0)
                memory_percent = float(info.get("memory_percent") or 0)
                if cpu_percent > PROCESS_CPU_WARNING:
                    process_alerts.append({
                        "level": "critical" if cpu_percent > 80 else "warning",
                        "category": "Process",
                        "message": (
                            f"{name} exceeds the high-CPU rule "
                            f"({cpu_percent:.1f}% sampled process CPU)"
                        ),
                        "process": name,
                        "pid": pid,
                        "source": "process_threshold",
                    })
                if memory_percent > PROCESS_MEMORY_WARNING:
                    process_alerts.append({
                        "level": "critical" if memory_percent > 50 else "warning",
                        "category": "Process",
                        "message": (
                            f"{name} exceeds the high-memory rule "
                            f"({memory_percent:.1f}% of RAM)"
                        ),
                        "process": name,
                        "pid": pid,
                        "source": "process_threshold",
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
                continue
    except Exception:
        pass
    return process_alerts


def _add_system_alert(alerts, level, category, message):
    alert = {
        "level": level,
        "category": category,
        "message": message,
        "source": "threshold",
    }
    alerts.append(alert)
    insert_alert(level, message)


@router.get("/alerts")
def check_alerts():
    """Return live CPU/RAM/disk rules and high-resource process rules."""
    cpu = get_live_cpu_percent()
    ram = psutil.virtual_memory().percent
    disk = psutil.disk_usage(DISK_PATH).percent
    alerts = []

    if cpu > CPU_CRITICAL:
        _add_system_alert(alerts, "critical", "CPU", f"CPU critically high: {cpu}%")
    elif cpu > CPU_WARNING:
        _add_system_alert(alerts, "warning", "CPU", f"CPU usage elevated: {cpu}%")

    if ram > RAM_CRITICAL:
        _add_system_alert(alerts, "critical", "Memory", f"Memory critical: {ram}%")
    elif ram > RAM_WARNING:
        _add_system_alert(alerts, "warning", "Memory", f"Memory usage high: {ram}%")

    if disk > DISK_CRITICAL:
        _add_system_alert(alerts, "critical", "Disk", f"Disk almost full: {disk}%")
    elif disk > DISK_WARNING:
        _add_system_alert(alerts, "warning", "Disk", f"Disk usage elevated: {disk}%")

    process_alerts = _get_process_threshold_alerts()
    alerts.extend(process_alerts[:5])

    return {
        "alerts": alerts,
        "current": {"cpu": cpu, "ram": ram, "disk": disk},
        "thresholds": {
            "cpu_warning": CPU_WARNING,
            "cpu_critical": CPU_CRITICAL,
            "ram_warning": RAM_WARNING,
            "ram_critical": RAM_CRITICAL,
            "disk_warning": DISK_WARNING,
            "disk_critical": DISK_CRITICAL,
            "process_cpu_warning": PROCESS_CPU_WARNING,
            "process_memory_warning": PROCESS_MEMORY_WARNING,
        },
        "process_alert_count": len(process_alerts),
    }


@router.get("/alerts/history")
def alert_history(limit: int = Query(50, ge=1, le=500)):
    """Historical system threshold entries."""
    return get_recent_alerts(limit)
