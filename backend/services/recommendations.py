"""
Recommendation Engine for SysSense AI.

Phase 1: Rule-based recommendations.
Phase 2 (post-ML): Augmented with feature_importances_ from the trained
                    Random Forest for explainable, ML-driven suggestions.
"""
import psutil
import platform

DISK_PATH = "C:\\" if platform.system() == "Windows" else "/"


def _get_top_consumers(sort_by="cpu", top_n=5):
    """Get top resource-consuming processes."""
    procs = []
    for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
        try:
            procs.append(p.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    key = "cpu_percent" if sort_by == "cpu" else "memory_percent"
    procs.sort(key=lambda x: x.get(key) or 0, reverse=True)
    return procs[:top_n]


def generate_recommendations(cpu: float, ram: float, disk: float,
                             feature_importances: dict = None,
                             anomaly_data: dict = None,
                             process_anomalies: list = None) -> list:
    """
    Generate actionable recommendations based on current system state.

    Args:
        cpu:  Current CPU usage percent
        ram:  Current RAM usage percent
        disk: Current disk usage percent
        feature_importances: Optional dict from ML model's feature_importances_
                             for explainable recommendations (Phase 2)

    Returns:
        List of recommendation dicts with priority, category, message, detail.
    """
    recs = []

    # ── CPU Rules ─────────────────────────────────────────────────────
    if cpu > 90:
        top_cpu = _get_top_consumers("cpu", 3)
        names = ", ".join(p["name"] for p in top_cpu if p["name"])
        recs.append({
            "priority": "critical",
            "category": "CPU",
            "message": f"CPU usage critically high at {cpu}%",
            "detail": f"Top consumers: {names}. Consider closing unnecessary "
                      f"applications or background processes.",
            "icon": "cpu",
        })
    elif cpu > 75:
        top_cpu = _get_top_consumers("cpu", 3)
        names = ", ".join(p["name"] for p in top_cpu if p["name"])
        recs.append({
            "priority": "warning",
            "category": "CPU",
            "message": f"CPU usage elevated at {cpu}%",
            "detail": f"Top consumers: {names}. Monitor these processes.",
            "icon": "cpu",
        })

    # ── RAM Rules ─────────────────────────────────────────────────────
    if ram > 95:
        top_mem = _get_top_consumers("memory", 3)
        names = ", ".join(p["name"] for p in top_mem if p["name"])
        recs.append({
            "priority": "critical",
            "category": "Memory",
            "message": f"Memory critically low (usage: {ram}%)",
            "detail": f"Top consumers: {names}. Close memory-heavy apps like "
                      f"browser tabs, IDEs, or VMs.",
            "icon": "memory",
        })
    elif ram > 80:
        top_mem = _get_top_consumers("memory", 3)
        names = ", ".join(p["name"] for p in top_mem if p["name"])
        recs.append({
            "priority": "warning",
            "category": "Memory",
            "message": f"Memory usage high at {ram}%",
            "detail": f"Top consumers: {names}. Consider closing unused applications.",
            "icon": "memory",
        })

    # ── Disk Rules ────────────────────────────────────────────────────
    if disk > 95:
        recs.append({
            "priority": "critical",
            "category": "Disk",
            "message": f"Disk almost full ({disk}% used)",
            "detail": "Run Disk Cleanup, empty Recycle Bin, or uninstall unused programs.",
            "icon": "disk",
        })
    elif disk > 80:
        recs.append({
            "priority": "warning",
            "category": "Disk",
            "message": f"Disk usage elevated at {disk}%",
            "detail": "Consider cleaning temporary files and old downloads.",
            "icon": "disk",
        })

    # ── Browser-specific Rule ─────────────────────────────────────────
    top_mem = _get_top_consumers("memory", 10)
    browser_names = {"chrome.exe", "firefox.exe", "msedge.exe",
                     "chrome", "firefox", "Safari"}
    browser_mem = sum(
        (p.get("memory_percent") or 0) for p in top_mem
        if p.get("name", "").lower() in {n.lower() for n in browser_names}
    )
    if browser_mem > 30:
        recs.append({
            "priority": "warning",
            "category": "Browser",
            "message": f"Browser consuming {browser_mem:.1f}% of RAM",
            "detail": "Close unused browser tabs or use a lighter browser.",
            "icon": "browser",
        })

    # ── ML-Driven Recommendations (Phase 2) ──────────────────────────
    if feature_importances:
        top_feature = max(feature_importances, key=feature_importances.get)
        importance = feature_importances[top_feature]
        if importance > 0.3:
            recs.append({
                "priority": "info",
                "category": "AI Insight",
                "message": f"ML model identifies '{top_feature}' as the primary "
                           f"driver of predicted load ({importance:.0%} importance)",
                "detail": "This metric is most likely to cause future resource "
                          "pressure. Consider proactive optimization.",
                "icon": "ai",
            })

    # ── Anomaly-Driven Recommendations ────────────────────────────────
    if anomaly_data and anomaly_data.get("is_anomaly"):
        severity = anomaly_data.get("severity", "mild")
        details = anomaly_data.get("details", [])
        detail_text = "; ".join(details) if details else "Unusual system behavior detected."
        recs.append({
            "priority": "critical" if severity == "critical" else "warning",
            "category": "Anomaly Detection",
            "message": f"ML anomaly detected ({severity} severity)",
            "detail": f"{detail_text} Consider investigating running processes and "
                      f"closing unnecessary applications.",
            "icon": "anomaly",
        })

    # ── Process-specific anomaly recommendations ──────────────────────
    if process_anomalies:
        for pa in process_anomalies[:3]:  # limit to top 3
            recs.append({
                "priority": pa.get("severity", "warning"),
                "category": "Process Anomaly",
                "message": pa.get("message", "Process consuming excessive resources"),
                "detail": f"Consider closing {pa.get('process', 'this process')} "
                          f"or reducing its workload.",
                "icon": "process",
            })

    # ── All-clear ─────────────────────────────────────────────────────
    if not recs:
        recs.append({
            "priority": "info",
            "category": "System",
            "message": "System running smoothly",
            "detail": "All metrics are within normal ranges. No action needed.",
            "icon": "check",
        })

    return recs

