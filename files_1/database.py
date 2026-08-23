"""
SQLite storage layer for SysSense.
Handles the metrics table used by both collector.py (writes) and
main.py (reads for the dashboard + future ML training).
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime

DB_PATH = "sysense.db"


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cpu_percent REAL,
                ram_percent REAL,
                disk_percent REAL,
                net_sent_mb REAL,
                net_recv_mb REAL
            )
        """)


def insert_metric(cpu, ram, disk, net_sent_mb, net_recv_mb):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO metrics
               (timestamp, cpu_percent, ram_percent, disk_percent, net_sent_mb, net_recv_mb)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(), cpu, ram, disk, net_sent_mb, net_recv_mb),
        )


def get_recent_metrics(limit=100):
    """Most recent `limit` readings, oldest first (good for chart x-axis order)."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM metrics ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


def get_all_metrics_for_training():
    """Full history, oldest first. Use this in your ML training script (Day 20+)."""
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM metrics ORDER BY id ASC").fetchall()
    return [dict(row) for row in rows]
