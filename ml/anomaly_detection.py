"""
SysSense AI — Anomaly Detection Module

Uses Isolation Forest to detect abnormal system behavior patterns:
    - Sudden CPU/RAM spikes
    - Sustained high load over time
    - Unusual metric combinations (e.g., high disk I/O + high CPU = possible malware)

This is the module that makes the project stand out from typical student projects.
In your report, connect this to OS security concepts (intrusion/malware detection
via resource pattern analysis).

Usage:
    python anomaly_detection.py
"""
import os
import sys
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from preprocessing import load_raw_data, engineer_features


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def train_anomaly_detector(contamination=0.05):
    """
    Train an Isolation Forest on 'normal' usage patterns.

    Args:
        contamination: Expected proportion of anomalies (0.05 = 5%)
                       Lower = more conservative (fewer false alarms)

    Returns:
        Trained IsolationForest model
    """
    print("Loading data for anomaly detection training...")
    df = load_raw_data()
    if df is None or len(df) < 50:
        print("Not enough data for anomaly detection. Collect more first.")
        return None

    # Use a subset of features most relevant for anomaly detection
    features = df[["cpu_percent", "ram_percent", "disk_percent"]].copy()

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

    return model, list(features.columns)


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
            result["details"].append(f"CPU spike detected (Δ{cpu_delta:.1f}%)")
        if ram_delta > 15:
            result["details"].append(f"RAM spike detected (Δ{ram_delta:.1f}%)")
        if cpu > 70 and ram > 85:
            result["details"].append(
                "Combined high CPU + RAM may indicate resource exhaustion"
            )

        if not result["details"]:
            result["details"].append(
                "Unusual metric combination detected by ML model"
            )

    return result


def main():
    """Train and save the anomaly detection model."""
    print("=" * 60)
    print("   SysSense AI — Anomaly Detection Training")
    print("=" * 60)

    result = train_anomaly_detector()
    if result is None:
        return

    model, feature_names = result

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "anomaly_model.pkl")
    meta_path = os.path.join(MODELS_DIR, "anomaly_metadata.pkl")

    joblib.dump(model, model_path)
    joblib.dump({"feature_names": feature_names}, meta_path)

    print(f"\n✓ Anomaly model saved to {model_path}")
    print(f"✓ Metadata saved to {meta_path}")

    # Demo: check the last few data points
    print("\nDemo: checking last 5 data points...")
    from preprocessing import load_raw_data
    df = load_raw_data()
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
            if "net_sent_mb" in row:
                metrics["net_sent_rate"] = 0
                metrics["net_recv_rate"] = 0
            result = detect_anomalies(model, feature_names, metrics)
            status = "🔴 ANOMALY" if result["is_anomaly"] else "🟢 Normal"
            print(f"  Row {len(df)+i}: CPU={row['cpu_percent']:5.1f}% "
                  f"RAM={row['ram_percent']:5.1f}% → {status} "
                  f"(score: {result['anomaly_score']:.4f})")


if __name__ == "__main__":
    main()
