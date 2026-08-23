"""
SysSense AI — Dataset Loader & Preprocessor

Loads the 'IT System Performance & Resource Metrics' dataset from Kaggle,
a real-world system performance monitoring dataset (updated late 2025).

Dataset Source:
    Kaggle — IT System Performance & Resource Metrics
    URL: https://www.kaggle.com/datasets (IT System Performance & Resource Metrics)
    File: Big_data_dataset.csv

The dataset includes:
    - CPU utilization (%)
    - Memory usage (%)
    - Disk I/O rates
    - Network latency (ms)
    - Process and thread counts
    - Context switches, cache miss rate
    - System temperature and power consumption
    - Uptime
    - Anomaly status labels (0 = normal, 1 = anomaly)

This dataset is used to train the ML models ONCE. During live demos,
the application collects real metrics via psutil and uses the trained
model for predictions.

Usage:
    python generate_dataset.py
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
KAGGLE_DATASET_FILE = os.path.join(DATA_DIR, "Big_data_dataset.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "system_metrics_dataset.csv")

# ── Configuration ─────────────────────────────────────────────────────
INTERVAL_SECONDS = 5           # simulated polling interval for timestamps
SEED = 42


def load_kaggle_dataset():
    """
    Load the Kaggle IT System Performance & Resource Metrics dataset.

    Returns:
        pandas DataFrame with raw columns from the Kaggle CSV.
    """
    if not os.path.exists(KAGGLE_DATASET_FILE):
        print(f"[!] Kaggle dataset not found at {KAGGLE_DATASET_FILE}")
        print("    Download from Kaggle: IT System Performance & Resource Metrics")
        print("    Place the CSV file as: ml/data/Big_data_dataset.csv")
        return None

    df = pd.read_csv(KAGGLE_DATASET_FILE)
    print(f"Loaded {len(df)} rows from Kaggle dataset")
    print(f"  Columns: {list(df.columns)}")
    return df


def map_columns(df):
    """
    Map Kaggle dataset columns to SysSense AI naming conventions.

    Mappings:
        cpu_utilization   → cpu_percent
        memory_usage      → ram_percent
        disk_io           → disk_percent  (scaled from 0-50 to 0-100)
        network_latency   → network_latency_ms
        process_count     → process_count  (direct)
        thread_count      → thread_count   (direct)
        context_switches  → context_switches (kept as-is)
        cache_miss_rate   → cache_miss_rate  (kept as-is)
        temperature       → temperature      (kept as-is)
        power_consumption → power_consumption (kept as-is)
        uptime            → uptime            (kept as-is)
        status            → anomaly_label     (0=normal, 1=anomaly)
    """
    mapped = pd.DataFrame()

    # Generate sequential IDs
    mapped["id"] = range(1, len(df) + 1)

    # Generate synthetic timestamps (dataset has no timestamp column)
    start = datetime(2025, 6, 1, 0, 0, 0)
    timestamps = [start + timedelta(seconds=i * INTERVAL_SECONDS)
                  for i in range(len(df))]
    mapped["timestamp"] = [t.isoformat() for t in timestamps]

    # Core metrics — direct mapping
    mapped["cpu_percent"] = df["cpu_utilization"].round(1)
    mapped["ram_percent"] = df["memory_usage"].round(1)

    # Disk I/O: original range is 1–50, scale to percentage (0–100)
    mapped["disk_percent"] = (df["disk_io"] * 2).clip(0, 100).round(1)

    # Network: use latency as network metric
    # Simulate sent/recv MB from network latency for compatibility
    # Higher latency → more network activity (rough proxy)
    np.random.seed(SEED)
    net_base = np.cumsum(df["network_latency"].values / 200.0)
    mapped["net_sent_mb"] = (net_base + np.random.normal(0, 0.2, len(df))).round(2)
    mapped["net_recv_mb"] = (net_base * 2.5 + np.random.normal(0, 0.3, len(df))).round(2)
    mapped["net_sent_mb"] = mapped["net_sent_mb"].clip(lower=0)
    mapped["net_recv_mb"] = mapped["net_recv_mb"].clip(lower=0)

    # Direct mappings
    mapped["thread_count"] = df["thread_count"].astype(int)
    mapped["process_count"] = df["process_count"].astype(int)

    # Extra features from the Kaggle dataset (bonus columns)
    mapped["network_latency_ms"] = df["network_latency"].round(2)
    mapped["context_switches"] = df["context_switches"].astype(int)
    mapped["cache_miss_rate"] = df["cache_miss_rate"].round(4)
    mapped["temperature"] = df["temperature"].round(1)
    mapped["power_consumption"] = df["power_consumption"].round(2)
    mapped["uptime"] = df["uptime"].round(2)

    # Anomaly labels (ground truth)
    mapped["anomaly_label"] = df["status"].astype(int)

    return mapped


def generate_dataset():
    """
    Load the Kaggle dataset and produce the mapped DataFrame.

    This replaces the old synthetic generation approach.
    """
    raw = load_kaggle_dataset()
    if raw is None:
        return None

    mapped = map_columns(raw)
    return mapped


def main():
    print("=" * 60)
    print("   SysSense AI — Dataset Loader")
    print("   Source: Kaggle IT System Performance & Resource Metrics")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    df = generate_dataset()
    if df is None:
        return

    print(f"\nProcessed {len(df)} rows")
    print(f"Time range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"\nMetric Ranges:")
    for col in ["cpu_percent", "ram_percent", "disk_percent"]:
        print(f"  {col:15s}  min={df[col].min():5.1f}  "
              f"max={df[col].max():5.1f}  mean={df[col].mean():5.1f}")

    # Anomaly statistics from ground-truth labels
    anomalies = df["anomaly_label"].sum()
    print(f"\nGround-truth anomalous rows: {anomalies} ({anomalies/len(df)*100:.1f}%)")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[OK] Mapped dataset saved to {OUTPUT_FILE}")
    print(f"  File size: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")

    print("\nSample rows:")
    print(df.head(3).to_string(index=False))
    print("\n" + "=" * 60)
    print("   Dataset ready! Run 'python train.py' next.")
    print("=" * 60)


if __name__ == "__main__":
    main()
