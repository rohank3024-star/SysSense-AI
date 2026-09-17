"""
SQLite storage layer for SysSense AI.

Tables:
    metrics           – time-series CPU/RAM/disk/network readings (written by collector)
    process_snapshots – per-process stats logged periodically (for memory-leak detection)
    alerts_log        – historical alert records
"""
import sqlite3
import os
import platform
from contextlib import contextmanager
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sysense.db")


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── Schema ────────────────────────────────────────────────────────────

def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT NOT NULL,
                cpu_percent    REAL,
                ram_percent    REAL,
                disk_percent   REAL,
                net_sent_mb    REAL,
                net_recv_mb    REAL,
                thread_count   INTEGER,
                process_count  INTEGER
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS process_snapshots (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT NOT NULL,
                pid            INTEGER,
                name           TEXT,
                cpu_percent    REAL,
                memory_percent REAL,
                memory_rss_mb  REAL,
                status         TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts_log (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT NOT NULL,
                level          TEXT NOT NULL,
                message        TEXT NOT NULL
            )
        """)

        # Index for faster time-range queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_timestamp
            ON metrics (timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_process_snapshots_timestamp
            ON process_snapshots (timestamp)
        """)


# ── Writes ────────────────────────────────────────────────────────────

def insert_metric(cpu, ram, disk, net_sent_mb, net_recv_mb,
                  thread_count=0, process_count=0):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO metrics
               (timestamp, cpu_percent, ram_percent, disk_percent,
                net_sent_mb, net_recv_mb, thread_count, process_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(), cpu, ram, disk,
             net_sent_mb, net_recv_mb, thread_count, process_count),
        )


def insert_process_snapshot(pid, name, cpu_pct, mem_pct, mem_rss_mb, status):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO process_snapshots
               (timestamp, pid, name, cpu_percent, memory_percent,
                memory_rss_mb, status)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(), pid, name, cpu_pct,
             mem_pct, mem_rss_mb, status),
        )


def insert_alert(level, message):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO alerts_log (timestamp, level, message) VALUES (?, ?, ?)",
            (datetime.now().isoformat(), level, message),
        )


# ── Reads ─────────────────────────────────────────────────────────────

def get_recent_metrics(limit=100):
    """Most recent `limit` readings, oldest first (good for chart x-axis)."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM metrics ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


def get_metrics_since(hours=1):
    """Metrics from the last N hours, oldest first."""
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM metrics WHERE timestamp >= ? ORDER BY id ASC",
            (cutoff,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_metrics_aggregated(hours=24, bucket_minutes=60):
    """
    Aggregate metrics into time buckets for analytics charts.
    Returns avg CPU/RAM/disk per bucket.
    """
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT
                 strftime('%%Y-%%m-%%dT%%H:', timestamp) ||
                 printf('%%02d', (CAST(strftime('%%M', timestamp) AS INT) / ?) * ?) AS bucket,
                 AVG(cpu_percent) AS avg_cpu,
                 AVG(ram_percent) AS avg_ram,
                 AVG(disk_percent) AS avg_disk,
                 COUNT(*) AS sample_count
               FROM metrics
               WHERE timestamp >= ?
               GROUP BY bucket
               ORDER BY bucket ASC""",
            (bucket_minutes, bucket_minutes, cutoff),
        ).fetchall()
    return [dict(row) for row in rows]


def get_all_metrics_for_training():
    """Full history, oldest first. Used by ML training scripts."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM metrics ORDER BY id ASC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_process_history(pid=None, hours=1):
    """Process snapshots for memory-leak detection."""
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    with get_connection() as conn:
        if pid:
            rows = conn.execute(
                """SELECT * FROM process_snapshots
                   WHERE pid = ? AND timestamp >= ?
                   ORDER BY id ASC""",
                (pid, cutoff),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM process_snapshots
                   WHERE timestamp >= ?
                   ORDER BY id ASC""",
                (cutoff,),
            ).fetchall()
    return [dict(row) for row in rows]


def get_recent_alerts(limit=50):
    """Most recent alerts."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM alerts_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


def get_metrics_count():
    """Total number of collected metrics rows."""
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM metrics").fetchone()
    return dict(row)["cnt"]
