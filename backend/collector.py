"""
Background collector for SysSense AI.

Run this in its own terminal and leave it running from Day 1 — the
ML model needs a good volume of historical data to train on.

Usage:
    python collector.py
"""
import time
import platform
import psutil
from database import init_db, insert_metric, insert_process_snapshot

POLL_INTERVAL_SECONDS = 3          # time between log writes
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
    print(f"SysSense collector started. Logging every ~{POLL_INTERVAL_SECONDS + 1}s.")
    print("Press Ctrl+C to stop.\n")
    cycle = 0
    try:
        while True:
            cpu, ram, disk = collect_once()
            cycle += 1
            marker = " [+procs]" if cycle % PROCESS_SNAPSHOT_EVERY == 0 else ""
            print(f"CPU: {cpu:5.1f}%  RAM: {ram:5.1f}%  Disk: {disk:5.1f}%{marker}")

            if cycle % PROCESS_SNAPSHOT_EVERY == 0:
                snapshot_top_processes()

            time.sleep(POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nCollector stopped.")


if __name__ == "__main__":
    main()
