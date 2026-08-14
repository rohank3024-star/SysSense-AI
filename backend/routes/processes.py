"""
Process management endpoints for SysSense AI.

Optimized for Windows where process iteration can be slow.
Uses only psutil.process_iter attrs (no separate memory_info() calls)
and a cache to prevent repeated slow iterations.
"""
from fastapi import APIRouter, Query
import psutil
import time
import threading

router = APIRouter(tags=["Processes"])

# Cache for process list
_proc_cache = {"data": [], "summary": {}, "timestamp": 0}
_proc_lock = threading.Lock()
PROC_CACHE_TTL = 4  # seconds


def _refresh_process_list():
    """Refresh the cached process list if stale."""
    now = time.time()
    with _proc_lock:
        if (now - _proc_cache["timestamp"]) < PROC_CACHE_TTL:
            return

    procs = []
    statuses = {}
    total = 0

    # Only request attrs available from process_iter (no per-process syscalls)
    for p in psutil.process_iter(["pid", "name", "cpu_percent",
                                   "memory_percent", "status",
                                   "num_threads"]):
        try:
            info = p.info
            procs.append({
                "pid": info["pid"],
                "name": info["name"],
                "cpu_percent": info["cpu_percent"] or 0,
                "memory_percent": round(info["memory_percent"] or 0, 2),
                "memory_rss_mb": 0,  # skip slow memory_info() call
                "status": info["status"],
                "threads": info["num_threads"] or 0,
            })

            s = info.get("status", "unknown")
            statuses[s] = statuses.get(s, 0) + 1
            total += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    with _proc_lock:
        _proc_cache["data"] = procs
        _proc_cache["summary"] = {"total": total, "by_status": statuses}
        _proc_cache["timestamp"] = time.time()


@router.get("/processes")
def top_processes(
    sort_by: str = Query("cpu", pattern="^(cpu|memory)$"),
    limit: int = Query(25, ge=1, le=100),
    search: str = Query("", description="Filter by process name"),
):
    """
    Running processes with PID, name, CPU%, memory%, status.
    Supports sorting, limiting, and name search.
    """
    _refresh_process_list()

    with _proc_lock:
        procs = list(_proc_cache["data"])

    if search:
        procs = [p for p in procs if search.lower() in (p.get("name") or "").lower()]

    key = "cpu_percent" if sort_by == "cpu" else "memory_percent"
    procs.sort(key=lambda x: x.get(key) or 0, reverse=True)
    return procs[:limit]


@router.get("/processes/summary")
def process_summary():
    """Summary stats about running processes."""
    _refresh_process_list()
    with _proc_lock:
        return dict(_proc_cache["summary"])
