"""Dataset loading and collector-compatible forecasting features.

The project trains on the Kaggle-derived system-metrics dataset as required.
The feature contract deliberately contains only data that ``backend/collector``
stores, so a deployed model never receives invented zero values for missing
training features.
"""
import os
import sys

import numpy as np
import pandas as pd


# Allow optional local-database training from the backend package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PUBLIC_DATASET_PATH = os.path.join(DATA_DIR, "system_metrics_dataset.csv")
KAGGLE_DATASET_PATH = os.path.join(DATA_DIR, "Big_data_dataset.csv")

FEATURE_PROFILE = "collector_compatible_v1"
FORECAST_HORIZON_SAMPLES = 6
COLLECTOR_INTERVAL_SECONDS = 5
COLLECTOR_FEATURE_NAMES = [
    "cpu_current", "ram_current", "disk_current",
    "cpu_lag1", "ram_lag1", "cpu_lag3", "ram_lag3",
    "cpu_lag5", "ram_lag5",
    "cpu_ma5", "cpu_ma10", "ram_ma5", "ram_ma10",
    "cpu_delta", "ram_delta", "disk_delta",
    "cpu_std5", "ram_std5",
    "net_sent_rate", "net_recv_rate",
    "thread_count", "process_count",
    "hour", "minute",
]


def get_feature_contract():
    """Return the versioned schema shared by training and live inference."""
    return {
        "profile": FEATURE_PROFILE,
        "feature_names": list(COLLECTOR_FEATURE_NAMES),
        "horizon_samples": FORECAST_HORIZON_SAMPLES,
        "assumed_sample_interval_seconds": COLLECTOR_INTERVAL_SECONDS,
        "nominal_horizon_seconds": FORECAST_HORIZON_SAMPLES * COLLECTOR_INTERVAL_SECONDS,
        "source": "collector metrics table",
    }


def load_public_dataset():
    """Load the mapped Kaggle dataset, generating it from the raw CSV if needed."""
    if os.path.exists(PUBLIC_DATASET_PATH):
        frame = pd.read_csv(PUBLIC_DATASET_PATH)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame = frame.sort_values("id").reset_index(drop=True)
        print(f"Loaded {len(frame)} rows from mapped Kaggle dataset")
        print(f"  Time range: {frame['timestamp'].min()} to {frame['timestamp'].max()}")
        return frame

    if os.path.exists(KAGGLE_DATASET_PATH):
        print("Mapped dataset not found. Loading and mapping the raw Kaggle CSV...")
        from generate_dataset import generate_dataset

        frame = generate_dataset()
        if frame is not None:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"])
            return frame.sort_values("id").reset_index(drop=True)

    print("No dataset found. Place Big_data_dataset.csv in ml/data/ and run generate_dataset.py.")
    return None


def load_raw_data():
    """Load locally collected data when an explicit local-training run is wanted."""
    try:
        from database import get_all_metrics_for_training

        data = get_all_metrics_for_training()
        if not data:
            print("No data found in the local database.")
            return None
        frame = pd.DataFrame(data)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame = frame.sort_values("id").reset_index(drop=True)
        print(f"Loaded {len(frame)} rows from local SQLite database")
        print(f"  Time range: {frame['timestamp'].min()} to {frame['timestamp'].max()}")
        return frame
    except Exception as exc:
        print(f"Could not load local metrics: {exc}")
        return None


def load_data(prefer_public=True):
    """Use the public dataset by default, with local DB as a fallback."""
    if prefer_public:
        frame = load_public_dataset()
        if frame is not None:
            return frame
        print("Falling back to local SQLite data...")
    return load_raw_data()


def _counter_rate(frame, column):
    """Compute a non-negative MB/s rate from a cumulative counter column."""
    elapsed = frame["timestamp"].diff().dt.total_seconds().replace(0, np.nan)
    return (frame[column].diff() / elapsed).replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)


def engineer_features(frame, prediction_horizon=FORECAST_HORIZON_SAMPLES):
    """Create the versioned collector-compatible supervised data set.

    ``prediction_horizon`` is six samples by default. With the documented
    five-second collector cadence it is a nominal 30-second forecast horizon.
    """
    required_columns = {
        "timestamp", "cpu_percent", "ram_percent", "disk_percent",
        "net_sent_mb", "net_recv_mb", "thread_count", "process_count",
    }
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing collector-compatible columns: {missing}")

    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    feature_frame = pd.DataFrame(index=frame.index)

    feature_frame["cpu_current"] = frame["cpu_percent"]
    feature_frame["ram_current"] = frame["ram_percent"]
    feature_frame["disk_current"] = frame["disk_percent"]

    for lag in (1, 3, 5):
        feature_frame[f"cpu_lag{lag}"] = frame["cpu_percent"].shift(lag)
        feature_frame[f"ram_lag{lag}"] = frame["ram_percent"].shift(lag)

    feature_frame["cpu_ma5"] = frame["cpu_percent"].rolling(5).mean()
    feature_frame["cpu_ma10"] = frame["cpu_percent"].rolling(10).mean()
    feature_frame["ram_ma5"] = frame["ram_percent"].rolling(5).mean()
    feature_frame["ram_ma10"] = frame["ram_percent"].rolling(10).mean()
    feature_frame["cpu_delta"] = frame["cpu_percent"].diff()
    feature_frame["ram_delta"] = frame["ram_percent"].diff()
    feature_frame["disk_delta"] = frame["disk_percent"].diff()
    feature_frame["cpu_std5"] = frame["cpu_percent"].rolling(5).std()
    feature_frame["ram_std5"] = frame["ram_percent"].rolling(5).std()
    feature_frame["net_sent_rate"] = _counter_rate(frame, "net_sent_mb")
    feature_frame["net_recv_rate"] = _counter_rate(frame, "net_recv_mb")
    feature_frame["thread_count"] = frame["thread_count"]
    feature_frame["process_count"] = frame["process_count"]
    feature_frame["hour"] = frame["timestamp"].dt.hour
    feature_frame["minute"] = frame["timestamp"].dt.minute

    feature_frame["label_cpu_30s"] = frame["cpu_percent"].shift(-prediction_horizon)
    feature_frame["label_ram_30s"] = frame["ram_percent"].shift(-prediction_horizon)
    feature_frame = feature_frame.dropna().reset_index(drop=True)

    # Guard against accidental training/inference order drift.
    ordered_columns = COLLECTOR_FEATURE_NAMES + ["label_cpu_30s", "label_ram_30s"]
    feature_frame = feature_frame[ordered_columns]
    print(f"After feature engineering: {len(feature_frame)} usable rows")
    print(f"Feature profile: {FEATURE_PROFILE} ({len(COLLECTOR_FEATURE_NAMES)} live-compatible inputs)")
    return feature_frame


def prepare_train_test(feature_frame, test_ratio=0.2):
    """Create a chronological split with a six-sample embargo at its boundary."""
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")

    split_index = int(len(feature_frame) * (1 - test_ratio))
    train_end = split_index - FORECAST_HORIZON_SAMPLES
    if train_end <= 0:
        raise ValueError("Not enough rows for the requested split and forecast embargo")
    feature_names = list(COLLECTOR_FEATURE_NAMES)
    X_train = feature_frame[feature_names].iloc[:train_end].values
    X_test = feature_frame[feature_names].iloc[split_index:].values
    y_train_cpu = feature_frame["label_cpu_30s"].iloc[:train_end].values
    y_test_cpu = feature_frame["label_cpu_30s"].iloc[split_index:].values
    y_train_ram = feature_frame["label_ram_30s"].iloc[:train_end].values
    y_test_ram = feature_frame["label_ram_30s"].iloc[split_index:].values

    print(f"Train: {len(X_train)} rows | Gap: {FORECAST_HORIZON_SAMPLES} rows | Test: {len(X_test)} rows")
    print(f"Features: {feature_names}")
    return (
        X_train, X_test, y_train_cpu, y_test_cpu,
        y_train_ram, y_test_ram, feature_names,
    )


if __name__ == "__main__":
    print("=== SysSense AI - Collector-Compatible Preprocessing ===\n")
    dataset = load_data(prefer_public=True)
    if dataset is not None:
        features = engineer_features(dataset)
        print(f"\nFeature columns: {list(features.columns)}")
        prepared = prepare_train_test(features)
        print(f"\nReady for training. X_train shape: {prepared[0].shape}")
