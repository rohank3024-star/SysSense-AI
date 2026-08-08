"""
Metrics endpoints for SysSense AI.

Uses a lightweight cache to avoid multiple blocking psutil calls
when the dashboard fires 4 API requests in parallel.
"""
from fastapi import APIRouter, Query
from database import get_recent_metrics, get_metrics_since, get_metrics_aggregated, get_metrics_count
import psutil
import platform
import time
import threading

router = APIRouter(prefix="/metrics", tags=["Metrics"])

DISK_PATH = "C:\\" if platform.system() == "Windows" else "/"

# ── Simple cache to prevent parallel blocking calls ─────────────────
_cache = {"data": None, "timestamp": 0}
_cache_lock = threading.Lock()
CACHE_TTL = 1.5  # seconds


def _get_live_snapshot():
    """Get a cached live snapshot (avoids blocking when dashboard polls in parallel)."""
    now = time.time()
    with _cache_lock:
        if _cache["data"] and (now - _cache["timestamp"]) < CACHE_TTL:
            return _cache["data"]

    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(DISK_PATH)
    net = psutil.net_io_counters()
    boot = psutil.boot_time()
    uptime_seconds = time.time() - boot

    # cpu_percent with interval=0 uses the delta since last call (non-blocking)
    # The first call returns 0.0 but subsequent calls are fast and accurate.
    cpu = psutil.cpu_percent(interval=0)

    data = {
        "cpu_percent": cpu,
        "cpu_count": psutil.cpu_count(),
        "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        "ram_percent": vm.percent,
        "ram_total_gb": round(vm.total / (1024 ** 3), 2),
        "ram_used_gb": round(vm.used / (1024 ** 3), 2),
        "ram_available_gb": round(vm.available / (1024 ** 3), 2),
        "disk_percent": disk.percent,
        "disk_total_gb": round(disk.total / (1024 ** 3), 2),
        "disk_used_gb": round(disk.used / (1024 ** 3), 2),
        "disk_free_gb": round(disk.free / (1024 ** 3), 2),
        "net_sent_mb": round(net.bytes_sent / (1024 ** 2), 2),
        "net_recv_mb": round(net.bytes_recv / (1024 ** 2), 2),
        "process_count": len(psutil.pids()),
        "thread_count": 0,  # computed lazily below
        "uptime_seconds": round(uptime_seconds),
        "boot_time": boot,
    }

    # Thread count can be slow — compute in a best-effort way
    try:
        data["thread_count"] = sum(
            p.info["num_threads"]
            for p in psutil.process_iter(["num_threads"])
            if p.info.get("num_threads")
        )
    except Exception:
        data["thread_count"] = 0

    with _cache_lock:
        _cache["data"] = data
        _cache["timestamp"] = time.time()

    return data


# Prime the CPU percent counter on import so the first request isn't 0
psutil.cpu_percent(interval=0)


@router.get("/current")
def current_metrics():
    """Live snapshot - uses a short cache to avoid blocking parallel requests."""
    return _get_live_snapshot()


@router.get("/history")
def metrics_history(limit: int = Query(100, ge=1, le=5000)):
    """Historical readings written by collector.py - feed to Recharts."""
    return get_recent_metrics(limit)


@router.get("/since")
def metrics_since(hours: float = Query(1, ge=0.1, le=168)):
    """Metrics from the last N hours."""
    return get_metrics_since(hours)


@router.get("/aggregated")
def metrics_aggregated(
    hours: int = Query(24, ge=1, le=168),
    bucket_minutes: int = Query(60, ge=1, le=1440),
):
    """Aggregated metrics in time buckets for analytics charts."""
    return get_metrics_aggregated(hours, bucket_minutes)


@router.get("/count")
def metrics_count():
    """Total number of collected data points."""
    return {"count": get_metrics_count()}
