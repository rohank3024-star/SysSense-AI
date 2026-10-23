"""
Background collector for SysSense AI.

Run this in its own terminal and leave it running from Day 1 — the
ML model needs a good volume of historical data to train on.

Usage:
    python collector.py
"""
import time
import platform
import sqlite3
import psutil
from database import init_db, insert_metric, insert_process_snapshot

COLLECTION_INTERVAL_SECONDS = 5    # align live samples with the training horizon
PROCESS_SNAPSHOT_EVERY = 10        # log top processes every N cycles
PROCESS_SNAPSHOT_TOP_N = 10        # number of top processes to snapshot

DISK_PATH = "C:\\" if platform.system() == "Windows" else "/"


def collect_once():
    """Collect one snapshot of system metrics and save to DB."""
    cpu = psutil.cpu_percent(interval=1)   # blocks ~1s to sample accurately
    ram = psutil.virtual_memory().percent
    disk = psutil.disk_usage(DISK_PATH).percent
    net = psutil.net_io_counters()
    sent_mb = net.bytes_sent / (1024 * 1024)
    recv_mb = net.bytes_recv / (1024 * 1024)
    thread_count = sum(
        p.info["num_threads"]
        for p in psutil.process_iter(["num_threads"])
        if p.info["num_threads"]
    )
    process_count = len(psutil.pids())

    insert_metric(cpu, ram, disk, sent_mb, recv_mb, thread_count, process_count)
    return cpu, ram, disk


def snapshot_top_processes():
    """Log the top N processes by CPU% for memory-leak detection later."""
    procs = []
    for p in psutil.process_iter(["pid", "name", "cpu_percent",
                                   "memory_percent", "status"]):
        try:
            info = p.info
            mem_info = p.memory_info()
            procs.append({
                "pid": info["pid"],
                "name": info["name"],
                "cpu_percent": info["cpu_percent"] or 0,
                "memory_percent": info["memory_percent"] or 0,
                "memory_rss_mb": mem_info.rss / (1024 * 1024),
                "status": info["status"],
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    procs.sort(key=lambda x: x["cpu_percent"], reverse=True)
    for proc in procs[:PROCESS_SNAPSHOT_TOP_N]:
        insert_process_snapshot(
            proc["pid"], proc["name"], proc["cpu_percent"],
            proc["memory_percent"], proc["memory_rss_mb"], proc["status"],
        )


def main():
    init_db()
    print(f"SysSense collector started. Logging every ~{COLLECTION_INTERVAL_SECONDS}s.")
    print("Press Ctrl+C to stop.\n")
    cycle = 0
    try:
        while True:
            cycle_started = time.monotonic()
            try:
                cpu, ram, disk = collect_once()
                cycle += 1
                marker = " [+procs]" if cycle % PROCESS_SNAPSHOT_EVERY == 0 else ""
                print(f"CPU: {cpu:5.1f}%  RAM: {ram:5.1f}%  Disk: {disk:5.1f}%{marker}")

                if cycle % PROCESS_SNAPSHOT_EVERY == 0:
                    snapshot_top_processes()
            except sqlite3.OperationalError as e:
                print(f"[Warning] Database busy, retrying next cycle: {e}")

            # ``cpu_percent(interval=1)`` is part of collection time. Sleep
            # only for the remainder so consecutive database rows stay close
            # to five seconds apart and six samples correspond to ~30 seconds.
            remaining = COLLECTION_INTERVAL_SECONDS - (time.monotonic() - cycle_started)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nCollector stopped.")


if __name__ == "__main__":
    main()
