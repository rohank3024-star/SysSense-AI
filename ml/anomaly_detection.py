"""
SysSense AI — Anomaly Detection Module

Uses Isolation Forest to detect abnormal system behavior patterns:
    - Sudden CPU/RAM spikes
    - Sustained high load over time
    - Unusual metric combinations (e.g., high disk I/O + high CPU)
    - Per-process anomalies (e.g., Chrome consuming unusual CPU)

Trained on the Kaggle IT System Performance & Resource Metrics dataset
which includes ground-truth anomaly labels in the 'status' column.

During demonstrations, live metrics from psutil are checked against
the trained model.

Usage:
    python anomaly_detection.py
"""
import os
import sys
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import load_data

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def train_anomaly_detector(contamination=0.01):
    """
    Train an Isolation Forest on usage patterns from the Kaggle dataset.

    The Kaggle dataset includes ground-truth anomaly labels in the
    'anomaly_label' column (0 = normal, 1 = anomaly, ~0.97% anomalies).
    We use contamination=0.01 to match the actual anomaly rate.

    Args:
        contamination: Expected proportion of anomalies (0.01 = ~1%)

    Returns:
        Trained IsolationForest model, feature names, and evaluation metrics
    """
    print("Loading Kaggle dataset for anomaly detection training...")
    df = load_data(prefer_public=True)
    if df is None or len(df) < 50:
        print("Not enough data for anomaly detection. Check dataset.")
        return None

    # Use features most relevant for anomaly detection
    features = df[["cpu_percent", "ram_percent", "disk_percent"]].copy()

    if "network_latency_ms" in df.columns:
        features["network_latency"] = df["network_latency_ms"]

    if "context_switches" in df.columns:
        features["context_switches"] = df["context_switches"]

    if "temperature" in df.columns:
        features["temperature"] = df["temperature"]

    if "net_sent_mb" in df.columns and "net_recv_mb" in df.columns:
        features["net_sent_rate"] = df["net_sent_mb"].diff().fillna(0).clip(lower=0)
        features["net_recv_rate"] = df["net_recv_mb"].diff().fillna(0).clip(lower=0)

    # Add derived features
    features["cpu_ram_product"] = features["cpu_percent"] * features["ram_percent"] / 100
    features["cpu_delta"] = df["cpu_percent"].diff().fillna(0)
    features["ram_delta"] = df["ram_percent"].diff().fillna(0)

    features = features.dropna()
    print(f"Training Isolation Forest on {len(features)} samples...")

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(features.values)

    # Report training results
    predictions = model.predict(features.values)
    n_anomalies = (predictions == -1).sum()
    print(f"Detected {n_anomalies} anomalies in training data "
          f"({n_anomalies/len(features)*100:.1f}%)")

    # Evaluate against ground-truth labels if available
    eval_metrics = None
    if "anomaly_label" in df.columns:
        # Align indices after dropna
        ground_truth = df["anomaly_label"].iloc[:len(features)].values
        # Convert IF predictions: -1 (anomaly) → 1, 1 (normal) → 0
        predicted_labels = (predictions == -1).astype(int)

        print(f"\n--- Evaluation against Ground-Truth Labels ---")
        print(f"Ground-truth anomalies: {ground_truth.sum()}")
        print(f"Predicted anomalies:    {predicted_labels.sum()}")

        print("\nClassification Report:")
        report = classification_report(
            ground_truth, predicted_labels,
            target_names=["Normal", "Anomaly"],
            zero_division=0,
        )
        print(report)

        cm = confusion_matrix(ground_truth, predicted_labels)
        print(f"Confusion Matrix:")
        print(f"  TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
        print(f"  FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")

        eval_metrics = {
            "ground_truth_anomalies": int(ground_truth.sum()),
            "predicted_anomalies": int(predicted_labels.sum()),
            "confusion_matrix": cm.tolist(),
        }

    return model, list(features.columns), eval_metrics


def detect_anomalies(model, feature_names, current_metrics):
    """
    Check if the current metrics are anomalous.

    Args:
        model: Trained IsolationForest
        feature_names: List of feature column names (same order as training)
        current_metrics: dict with keys matching feature_names

    Returns:
        dict with is_anomaly bool, anomaly_score, and details
    """
    X = np.array([[current_metrics.get(f, 0) for f in feature_names]])

    prediction = model.predict(X)[0]  # 1 = normal, -1 = anomaly
    score = model.score_samples(X)[0]  # lower = more anomalous

    is_anomaly = prediction == -1

    result = {
        "is_anomaly": bool(is_anomaly),
        "anomaly_score": round(float(score), 4),
        "severity": "normal",
        "details": [],
    }

    if is_anomaly:
        # Determine severity based on score
        if score < -0.3:
            result["severity"] = "critical"
        elif score < -0.2:
            result["severity"] = "warning"
        else:
            result["severity"] = "mild"

        # Identify which metrics are unusual
        cpu = current_metrics.get("cpu_percent", 0)
        ram = current_metrics.get("ram_percent", 0)
        cpu_delta = abs(current_metrics.get("cpu_delta", 0))
        ram_delta = abs(current_metrics.get("ram_delta", 0))

        if cpu > 80:
            result["details"].append(f"CPU unusually high ({cpu}%)")
        if ram > 90:
            result["details"].append(f"RAM unusually high ({ram}%)")
        if cpu_delta > 30:
            result["details"].append(f"CPU spike detected (delta {cpu_delta:.1f}%)")
        if ram_delta > 15:
            result["details"].append(f"RAM spike detected (delta {ram_delta:.1f}%)")
        if cpu > 70 and ram > 85:
            result["details"].append(
                "Combined high CPU + RAM may indicate resource exhaustion"
            )

        if not result["details"]:
            result["details"].append(
                "Unusual metric combination detected by ML model"
            )

    return result


def detect_process_anomalies(processes, cpu_threshold=50, mem_threshold=20):
    """
    Detect per-process anomalies — processes consuming unusual resources.

    This is the feature described in the proposal:
        "Warning: Chrome is consuming unusually high CPU."

    Args:
        processes: list of dicts with 'name', 'cpu_percent', 'memory_percent'
        cpu_threshold: CPU% above which a process is flagged
        mem_threshold: Memory% above which a process is flagged

    Returns:
        list of anomaly dicts
    """
    anomalies = []

    for proc in processes:
        name = proc.get("name", "Unknown")
        cpu_pct = proc.get("cpu_percent", 0) or 0
        mem_pct = proc.get("memory_percent", 0) or 0

        if cpu_pct > cpu_threshold:
            anomalies.append({
                "process": name,
                "pid": proc.get("pid", 0),
                "type": "high_cpu",
                "severity": "critical" if cpu_pct > 80 else "warning",
                "message": f"{name} is consuming unusually high CPU ({cpu_pct:.1f}%)",
                "value": round(cpu_pct, 1),
            })

        if mem_pct > mem_threshold:
            anomalies.append({
                "process": name,
                "pid": proc.get("pid", 0),
                "type": "high_memory",
                "severity": "critical" if mem_pct > 40 else "warning",
                "message": f"{name} is consuming unusually high memory ({mem_pct:.1f}%)",
                "value": round(mem_pct, 1),
            })

    return anomalies


def main():
    """Train and save the anomaly detection model."""
    print("=" * 60)
    print("   SysSense AI -- Anomaly Detection Training")
    print("   Dataset: Kaggle IT System Performance & Resource Metrics")
    print("=" * 60)

    result = train_anomaly_detector()
    if result is None:
        return

    model, feature_names, eval_metrics = result

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "anomaly_model.pkl")
    meta_path = os.path.join(MODELS_DIR, "anomaly_metadata.pkl")

    joblib.dump(model, model_path)
    joblib.dump({
        "feature_names": feature_names,
        "eval_metrics": eval_metrics,
        "trained_on": "kaggle_it_system_performance",
    }, meta_path)

    print(f"\n[OK] Anomaly model saved to {model_path}")
    print(f"[OK] Metadata saved to {meta_path}")

    # Demo: check some data points from the dataset
    print("\nDemo: checking last 5 data points from dataset...")
    df = load_data(prefer_public=True)
    if df is not None:
        for i in range(-5, 0):
            row = df.iloc[i]
            metrics = {
                "cpu_percent": row["cpu_percent"],
                "ram_percent": row["ram_percent"],
                "disk_percent": row["disk_percent"],
                "cpu_ram_product": row["cpu_percent"] * row["ram_percent"] / 100,
                "cpu_delta": 0,
                "ram_delta": 0,
            }
            if "network_latency_ms" in row.index:
                metrics["network_latency"] = row["network_latency_ms"]
            if "context_switches" in row.index:
                metrics["context_switches"] = row["context_switches"]
            if "temperature" in row.index:
                metrics["temperature"] = row["temperature"]
            if "net_sent_mb" in row.index:
                metrics["net_sent_rate"] = 0
                metrics["net_recv_rate"] = 0

            det = detect_anomalies(model, feature_names, metrics)
            gt_label = "ANOMALY" if row.get("anomaly_label", 0) == 1 else "Normal"
            pred_status = "ANOMALY" if det["is_anomaly"] else "Normal"
            print(f"  Row {len(df)+i}: CPU={row['cpu_percent']:5.1f}% "
                  f"RAM={row['ram_percent']:5.1f}% -> "
                  f"GT={gt_label:7s} Pred={pred_status:7s} "
                  f"(score: {det['anomaly_score']:.4f})")

    print("\n" + "=" * 60)
    print("   Anomaly detection training complete!")
    print("   Dataset: Kaggle IT System Performance & Resource Metrics")
    print("=" * 60)


if __name__ == "__main__":
    main()
