"""Small shared helpers for accurate live system measurements."""
import threading
import time

import psutil


_cpu_lock = threading.Lock()
_cpu_cache = {"value": None, "timestamp": 0.0}
CPU_CACHE_SECONDS = 0.5


def get_live_cpu_percent():
    """Return a short sampled CPU value without the first-call zero problem.

    ``psutil.cpu_percent(interval=0)`` returns 0.0 on a thread's first call.
    FastAPI can serve requests on different worker threads, so priming it at
    module import does not reliably prevent that value. A short positive
    interval is a real measurement; the half-second cache prevents parallel
    dashboard endpoints from repeatedly blocking to take the same sample.
    """
    now = time.monotonic()
    with _cpu_lock:
        if (
            _cpu_cache["value"] is not None
            and now - _cpu_cache["timestamp"] < CPU_CACHE_SECONDS
        ):
            return _cpu_cache["value"]

        value = float(psutil.cpu_percent(interval=0.1))
        _cpu_cache["value"] = value
        _cpu_cache["timestamp"] = time.monotonic()
        return value
