"""Collector-compatible Isolation Forest anomaly detection.

The model is trained from the project dataset, but only with values that are
also available from the live collector. Evaluation is chronological and keeps
the held-out metrics separate from the final deployment fit.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

try:
    from .preprocessing import load_data
except ImportError:
    from preprocessing import load_data


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "anomaly_evaluation_report.md")
ANOMALY_FEATURE_PROFILE = "collector_compatible_anomaly_v1"
ANOMALY_FEATURE_NAMES = [
    "cpu_percent", "ram_percent", "disk_percent",
    "net_sent_rate", "net_recv_rate",
    "thread_count", "process_count",
    "cpu_ram_product", "cpu_delta", "ram_delta",
]


def _counter_rate(frame, column):
    elapsed = frame["timestamp"].diff().dt.total_seconds().replace(0, np.nan)
    return (frame[column].diff() / elapsed).replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)


def build_anomaly_features(frame):
    """Return exactly the input schema available from ``backend/collector.py``."""
    required_columns = {
        "timestamp", "cpu_percent", "ram_percent", "disk_percent",
        "net_sent_mb", "net_recv_mb", "thread_count", "process_count",
    }
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing live-compatible anomaly inputs: {missing}")

    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    features = pd.DataFrame(index=frame.index)
    features["cpu_percent"] = frame["cpu_percent"]
    features["ram_percent"] = frame["ram_percent"]
    features["disk_percent"] = frame["disk_percent"]
    features["net_sent_rate"] = _counter_rate(frame, "net_sent_mb")
    features["net_recv_rate"] = _counter_rate(frame, "net_recv_mb")
    features["thread_count"] = frame["thread_count"]
    features["process_count"] = frame["process_count"]
    features["cpu_ram_product"] = frame["cpu_percent"] * frame["ram_percent"] / 100
    features["cpu_delta"] = frame["cpu_percent"].diff().fillna(0)
    features["ram_delta"] = frame["ram_percent"].diff().fillna(0)
    # Keep the original row index until labels are selected. If a future
    # dataset has missing telemetry and a row is dropped here, its anomaly
    # label must be dropped with it rather than shifted onto another sample.
    return features[ANOMALY_FEATURE_NAMES].dropna()


def _classification_metrics(actual, predicted):
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    return {
        "accuracy": round(float(accuracy_score(actual, predicted)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
        "precision": round(float(precision_score(actual, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(actual, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(actual, predicted, zero_division=0)), 4),
        "confusion_matrix": matrix.tolist(),
        "test_rows": int(len(actual)),
        "ground_truth_anomalies": int(np.sum(actual)),
        "predicted_anomalies": int(np.sum(predicted)),
        "all_normal_accuracy": round(float(accuracy_score(actual, np.zeros_like(actual))), 4),
    }


def _write_report(metrics, train_rows, contamination):
    matrix = metrics["confusion_matrix"]
    report = f"""# SysSense AI - Anomaly Evaluation Report

## Model and split

- Model: Isolation Forest (200 trees, `n_jobs=1`, `random_state=42`)
- Input profile: `{ANOMALY_FEATURE_PROFILE}`
- Training rows: {train_rows}; chronological held-out rows: {metrics['test_rows']}
- Contamination setting: {contamination}

## Held-out classification metrics

| Metric | Value |
|---|---:|
| Accuracy | {metrics['accuracy']:.2%} |
| Balanced accuracy | {metrics['balanced_accuracy']:.2%} |
| Precision | {metrics['precision']:.2%} |
| Recall | {metrics['recall']:.2%} |
| F1 | {metrics['f1']:.2%} |
| All-normal baseline accuracy | {metrics['all_normal_accuracy']:.2%} |

Confusion matrix: TN={matrix[0][0]}, FP={matrix[0][1]}, FN={matrix[1][0]}, TP={matrix[1][1]}.

Accuracy is included for completeness, but this data is imbalanced. Present
precision, recall, F1, balanced accuracy, and the confusion matrix together;
do not describe accuracy alone as anomaly-detection quality.
"""
    with open(REPORT_PATH, "w", encoding="utf-8") as report_file:
        report_file.write(report)


def train_anomaly_detector(contamination=0.01, test_ratio=0.2):
    """Evaluate chronologically, then fit a final deployment model on all rows."""
    print("Loading the dataset for collector-compatible anomaly training...")
    frame = load_data(prefer_public=True)
    if frame is None or len(frame) < 100:
        print("Not enough data for anomaly training.")
        return None
    if "anomaly_label" not in frame.columns:
        print("The dataset has no anomaly_label column; evaluation cannot be performed.")
        return None

    features = build_anomaly_features(frame)
    labels = frame.loc[features.index, "anomaly_label"].astype(int).to_numpy()
    features = features.reset_index(drop=True)
    split_index = int(len(features) * (1 - test_ratio))
    train_features = features.iloc[:split_index]
    test_features = features.iloc[split_index:]
    train_labels = labels[:split_index]
    test_labels = labels[split_index:]

    evaluation_model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=1,
    )
    evaluation_model.fit(train_features.values)
    predicted_labels = (evaluation_model.predict(test_features.values) == -1).astype(int)
    metrics = _classification_metrics(test_labels, predicted_labels)

    print("\nChronological held-out anomaly evaluation:")
    for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "all_normal_accuracy"):
        print(f"  {key}: {metrics[key]:.4f}")
    print(f"  Confusion matrix: {metrics['confusion_matrix']}")

    # Refit on all available dataset rows after holding out the evaluation.
    deployment_model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=1,
    )
    deployment_model.fit(features.values)
    return deployment_model, list(features.columns), metrics, split_index


def detect_anomalies(model, feature_names, current_metrics):
    """Evaluate one fully-specified live-compatible metrics record."""
    missing = [name for name in feature_names if name not in current_metrics]
    if missing:
        raise ValueError(f"Missing anomaly features: {missing}")
    matrix = np.array([[current_metrics[name] for name in feature_names]], dtype=float)
    prediction = model.predict(matrix)[0]
    score = float(model.score_samples(matrix)[0])
    return {
        "is_anomaly": bool(prediction == -1),
        "anomaly_score": round(score, 4),
    }


def main():
    print("=" * 60)
    print("SysSense AI - Collector-Compatible Anomaly Training")
    print("Dataset: Kaggle IT System Performance & Resource Metrics")
    print("=" * 60)
    result = train_anomaly_detector()
    if result is None:
        return

    model, feature_names, metrics, train_rows = result
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "anomaly_model.pkl"))
    joblib.dump({
        "feature_names": feature_names,
        "feature_profile": ANOMALY_FEATURE_PROFILE,
        "evaluation_protocol": "chronological_80_20_holdout_then_refit_on_all_rows",
        "eval_metrics": metrics,
        "trained_on": "kaggle_it_system_performance",
        "contamination": 0.01,
        "n_estimators": 200,
    }, os.path.join(MODELS_DIR, "anomaly_metadata.pkl"))
    _write_report(metrics, train_rows, 0.01)
    print("[OK] Anomaly model, metadata, and evaluation report saved.")


if __name__ == "__main__":
    main()
