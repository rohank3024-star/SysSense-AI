"""
SQLite storage layer for SysSense AI.

Tables:
    metrics           – time-series CPU/RAM/disk/network readings (written by collector)
    process_snapshots – per-process stats logged periodically (for memory-leak detection)
    alerts_log        – historical alert records
"""
import math
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

        # A forecast is stored against the collector row used to build its
        # feature vector.  This lets the API compare the forecast with a real
        # later collector reading instead of claiming an unmeasured accuracy.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS forecast_log (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                source_metric_id   INTEGER NOT NULL,
                source_timestamp   TEXT NOT NULL,
                created_at         TEXT NOT NULL,
                target_timestamp   TEXT NOT NULL,
                model_name         TEXT NOT NULL,
                predicted_cpu      REAL,
                predicted_ram      REAL,
                actual_cpu         REAL,
                actual_ram         REAL,
                actual_timestamp   TEXT,
                status             TEXT NOT NULL DEFAULT 'pending',
                matched_at         TEXT,
                UNIQUE(source_metric_id, model_name)
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
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_forecast_log_status_target
            ON forecast_log (status, target_timestamp)
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
                 strftime('%Y-%m-%dT%H:', timestamp) ||
                 printf('%02d', (CAST(strftime('%M', timestamp) AS INT) / ?) * ?) AS bucket,
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


# ── Live forecast validation ──────────────────────────────────────────────

def insert_forecast(source_metric_id, source_timestamp, target_timestamp,
                    model_name, predicted_cpu, predicted_ram):
    """Store one model forecast for later comparison with live telemetry.

    The unique constraint intentionally allows only one forecast per model for
    a collector row.  The React UI polls frequently, whereas the collector
    produces a new source row on its own cadence; without this constraint a
    dashboard refresh would create duplicate evaluation samples.
    """
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT OR IGNORE INTO forecast_log
               (source_metric_id, source_timestamp, created_at, target_timestamp,
                model_name, predicted_cpu, predicted_ram)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                int(source_metric_id),
                str(source_timestamp),
                datetime.now().isoformat(),
                str(target_timestamp),
                str(model_name),
                float(predicted_cpu),
                float(predicted_ram),
            ),
        )
    return cursor.rowcount > 0


def match_pending_forecasts(grace_seconds=20):
    """Match due forecasts to the first collector reading after their target.

    A forecast made from collector sample *t* is evaluated against the first
    real reading in [t + horizon, t + horizon + grace].  It is never matched
    to a reading before the target timestamp.  Missing readings are eventually
    marked expired rather than silently treated as correct.
    """
    now = datetime.now()
    matched = 0
    expired = 0

    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, target_timestamp FROM forecast_log
               WHERE status = 'pending' ORDER BY target_timestamp ASC"""
        ).fetchall()

        for row in rows:
            try:
                target = datetime.fromisoformat(row["target_timestamp"])
            except (TypeError, ValueError):
                conn.execute(
                    "UPDATE forecast_log SET status = 'expired' WHERE id = ?",
                    (row["id"],),
                )
                expired += 1
                continue

            if target > now:
                continue

            deadline = target + timedelta(seconds=grace_seconds)
            actual = conn.execute(
                """SELECT timestamp, cpu_percent, ram_percent
                   FROM metrics
                   WHERE timestamp >= ? AND timestamp <= ?
                   ORDER BY timestamp ASC LIMIT 1""",
                (target.isoformat(), deadline.isoformat()),
            ).fetchone()

            if actual is not None:
                conn.execute(
                    """UPDATE forecast_log
                       SET actual_cpu = ?, actual_ram = ?, actual_timestamp = ?,
                           status = 'matched', matched_at = ?
                       WHERE id = ?""",
                    (
                        actual["cpu_percent"],
                        actual["ram_percent"],
                        actual["timestamp"],
                        now.isoformat(),
                        row["id"],
                    ),
                )
                matched += 1
            elif now > deadline:
                conn.execute(
                    "UPDATE forecast_log SET status = 'expired' WHERE id = ?",
                    (row["id"],),
                )
                expired += 1

    return {"matched_now": matched, "expired_now": expired}


def _forecast_metric_summary(rows, prediction_key, actual_key):
    """Return regression metrics for one target from matched forecast rows."""
    pairs = [
        (float(row[prediction_key]), float(row[actual_key]))
        for row in rows
        if row[prediction_key] is not None and row[actual_key] is not None
    ]
    if not pairs:
        return None

    predictions = [pair[0] for pair in pairs]
    actuals = [pair[1] for pair in pairs]
    errors = [prediction - actual for prediction, actual in pairs]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error ** 2 for error in errors) / len(errors))
    mean_actual = sum(actuals) / len(actuals)
    total_sum_squares = sum((actual - mean_actual) ** 2 for actual in actuals)
    residual_sum_squares = sum(error ** 2 for error in errors)

    # R² is undefined for one sample or a constant actual series.
    r2 = None
    if len(actuals) >= 2 and total_sum_squares > 0:
        r2 = 1 - (residual_sum_squares / total_sum_squares)

    return {
        "count": len(pairs),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4) if r2 is not None else None,
        "within_5pp_rate": round(
            100 * sum(abs(error) <= 5 for error in errors) / len(errors), 2
        ),
    }


def get_forecast_evaluation():
    """Return measured live forecast error grouped by model and target."""
    update_counts = match_pending_forecasts()

    with get_connection() as conn:
        count_rows = conn.execute(
            """SELECT status, COUNT(*) AS count
               FROM forecast_log GROUP BY status"""
        ).fetchall()
        matched_rows = conn.execute(
            """SELECT model_name, predicted_cpu, predicted_ram,
                      actual_cpu, actual_ram
               FROM forecast_log
               WHERE status = 'matched'
               ORDER BY id ASC"""
        ).fetchall()

    counts = {row["status"]: row["count"] for row in count_rows}
    grouped = {}
    for row in matched_rows:
        grouped.setdefault(row["model_name"], []).append(row)

    models = {}
    for model_name, rows in grouped.items():
        metrics = {
            "cpu": _forecast_metric_summary(rows, "predicted_cpu", "actual_cpu"),
            "ram": _forecast_metric_summary(rows, "predicted_ram", "actual_ram"),
        }
        models[model_name] = {
            key: value for key, value in metrics.items() if value is not None
        }

    return {
        "matched_count": counts.get("matched", 0),
        "pending_count": counts.get("pending", 0),
        "expired_count": counts.get("expired", 0),
        "updated": update_counts,
        "summary": {"models": models},
    }
