"""
Background collector for SysSense.

Continuously polls system metrics and stores them in the database.

Usage:
    python collector.py
"""
import time
import psutil
from database import init_db, insert_metric

POLL_INTERVAL_SECONDS = 3  # time between log writes


def collect_once():
    cpu = psutil.cpu_percent(interval=1)  # blocks ~1s to sample accurately
    ram = psutil.virtual_memory().percent
    disk = psutil.disk_usage("/").percent
    net = psutil.net_io_counters()
    sent_mb = net.bytes_sent / (1024 * 1024)
    recv_mb = net.bytes_recv / (1024 * 1024)

    insert_metric(cpu, ram, disk, sent_mb, recv_mb)
    return cpu, ram, disk


def main():
    init_db()
    print(f"SysSense collector started. Logging every ~{POLL_INTERVAL_SECONDS + 1}s. Ctrl+C to stop.")
    try:
        while True:
            cpu, ram, disk = collect_once()
            print(f"CPU: {cpu:5.1f}%  RAM: {ram:5.1f}%  Disk: {disk:5.1f}%")
            time.sleep(POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nCollector stopped.")


if __name__ == "__main__":
    main()
