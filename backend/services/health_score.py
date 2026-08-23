"""
Health Score Calculator for SysSense AI.

Formula:
    score = 100 - (w_cpu × cpu + w_ram × ram + w_disk × disk + alert_penalty)

Labels:
    90–100  Excellent
    70–89   Good
    50–69   Warning
    0–49    Critical
"""


def calculate_health_score(cpu_percent: float, ram_percent: float,
                           disk_percent: float, active_alerts: int = 0) -> dict:
    """
    Compute weighted health score from current metrics.

    Returns dict with score (0-100), label, and color hex code.
    """
    # Weights (must sum to ≤ 1.0 before alert penalty)
    w_cpu = 0.40
    w_ram = 0.35
    w_disk = 0.15

    # Each active alert costs 5 points, capped at 20
    alert_penalty = min(active_alerts * 5, 20)

    raw = (w_cpu * cpu_percent) + (w_ram * ram_percent) + (w_disk * disk_percent)
    score = max(0, min(100, 100 - raw - alert_penalty))
    score = round(score, 1)

    if score >= 90:
        label, color = "Excellent", "#00e676"
    elif score >= 70:
        label, color = "Good", "#66bb6a"
    elif score >= 50:
        label, color = "Warning", "#ffa726"
    else:
        label, color = "Critical", "#ef5350"

    return {
        "score": score,
        "label": label,
        "color": color,
        "breakdown": {
            "cpu_impact": round(w_cpu * cpu_percent, 1),
            "ram_impact": round(w_ram * ram_percent, 1),
            "disk_impact": round(w_disk * disk_percent, 1),
            "alert_penalty": alert_penalty,
        },
    }
