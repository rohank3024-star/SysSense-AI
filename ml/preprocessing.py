"""
SysSense AI — Data Preprocessing Module

Loads raw metrics from SQLite, engineers features, and prepares
train/test splits for ML model training.

Features engineered:
    - Current CPU, RAM, Disk
    - Lag features (previous values at t-1, t-3, t-5)
    - Moving averages (5-point, 10-point)
    - Rate of change (delta between consecutive readings)
    - Hour of day, minute of hour
    - Network activity rates

Labels:
    - CPU value ~30 seconds later (~7-8 rows ahead at 4s intervals)
    - RAM value ~30 seconds later
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Allow importing database.py from sibling directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from database import get_all_metrics_for_training


def load_raw_data():
    """Load all metrics from SQLite into a pandas DataFrame."""
    data = get_all_metrics_for_training()
    if not data:
        print("No data found in database. Run collector.py first!")
        return None

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("id").reset_index(drop=True)
    print(f"Loaded {len(df)} rows spanning "
          f"{df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


def engineer_features(df, prediction_horizon=8):
    """
    Engineer features from raw metrics.

    Args:
        df: Raw metrics DataFrame
        prediction_horizon: Number of rows ahead to predict
                            (8 rows × ~4s = ~32 seconds ahead)

    Returns:
        DataFrame with features and labels, ready for training.
    """
    feat = pd.DataFrame()

    # ── Current values ────────────────────────────────────────────────
    feat["cpu_current"] = df["cpu_percent"]
    feat["ram_current"] = df["ram_percent"]
    feat["disk_current"] = df["disk_percent"]

    # ── Lag features ──────────────────────────────────────────────────
    for lag in [1, 3, 5]:
        feat[f"cpu_lag{lag}"] = df["cpu_percent"].shift(lag)
        feat[f"ram_lag{lag}"] = df["ram_percent"].shift(lag)

    # ── Moving averages ───────────────────────────────────────────────
    feat["cpu_ma5"] = df["cpu_percent"].rolling(5).mean()
    feat["cpu_ma10"] = df["cpu_percent"].rolling(10).mean()
    feat["ram_ma5"] = df["ram_percent"].rolling(5).mean()
    feat["ram_ma10"] = df["ram_percent"].rolling(10).mean()

    # ── Rate of change (delta) ────────────────────────────────────────
    feat["cpu_delta"] = df["cpu_percent"].diff()
    feat["ram_delta"] = df["ram_percent"].diff()
    feat["disk_delta"] = df["disk_percent"].diff()

    # ── Volatility (rolling std) ──────────────────────────────────────
    feat["cpu_std5"] = df["cpu_percent"].rolling(5).std()
    feat["ram_std5"] = df["ram_percent"].rolling(5).std()

    # ── Network rates ─────────────────────────────────────────────────
    if "net_sent_mb" in df.columns:
        feat["net_sent_rate"] = df["net_sent_mb"].diff().fillna(0).clip(lower=0)
        feat["net_recv_rate"] = df["net_recv_mb"].diff().fillna(0).clip(lower=0)

    # ── Time features ─────────────────────────────────────────────────
    feat["hour"] = df["timestamp"].dt.hour
    feat["minute"] = df["timestamp"].dt.minute

    # ── Labels: values ~30 seconds in the future ──────────────────────
    feat["label_cpu_30s"] = df["cpu_percent"].shift(-prediction_horizon)
    feat["label_ram_30s"] = df["ram_percent"].shift(-prediction_horizon)

    # Drop rows with NaN (from shifting and rolling)
    feat = feat.dropna().reset_index(drop=True)
    print(f"After feature engineering: {len(feat)} usable rows")

    return feat


def prepare_train_test(df, test_ratio=0.2):
    """
    Split into train/test chronologically (NOT randomly — time series!).

    Returns:
        X_train, X_test, y_train_cpu, y_test_cpu, y_train_ram, y_test_ram, feature_names
    """
    feature_cols = [c for c in df.columns if not c.startswith("label_")]
    label_cpu = "label_cpu_30s"
    label_ram = "label_ram_30s"

    split_idx = int(len(df) * (1 - test_ratio))

    X_train = df[feature_cols].iloc[:split_idx].values
    X_test = df[feature_cols].iloc[split_idx:].values
    y_train_cpu = df[label_cpu].iloc[:split_idx].values
    y_test_cpu = df[label_cpu].iloc[split_idx:].values
    y_train_ram = df[label_ram].iloc[:split_idx].values
    y_test_ram = df[label_ram].iloc[split_idx:].values

    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"Features: {feature_cols}")

    return (X_train, X_test, y_train_cpu, y_test_cpu,
            y_train_ram, y_test_ram, feature_cols)


if __name__ == "__main__":
    print("=== SysSense AI — Preprocessing ===\n")
    df = load_raw_data()
    if df is not None:
        feat = engineer_features(df)
        print(f"\nFeature columns: {list(feat.columns)}")
        print(f"\nSample row:\n{feat.iloc[0].to_dict()}")

        result = prepare_train_test(feat)
        X_train, X_test = result[0], result[1]
        print(f"\nReady for training. X_train shape: {X_train.shape}")
