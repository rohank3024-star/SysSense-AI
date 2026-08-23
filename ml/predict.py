"""
SysSense AI — Prediction Module

Loads saved models and makes predictions from current system state.
Used by the FastAPI backend's /predict endpoint.
"""
import os
import numpy as np
import joblib

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def load_models():
    """Load all saved models. Returns dict with model objects or None."""
    models = {
        "cpu_model": None,
        "ram_model": None,
        "anomaly_model": None,
        "metadata": None,
        "anomaly_metadata": None,
    }

    try:
        cpu_path = os.path.join(MODELS_DIR, "cpu_model.pkl")
        ram_path = os.path.join(MODELS_DIR, "ram_model.pkl")
        meta_path = os.path.join(MODELS_DIR, "model_metadata.pkl")

        if os.path.exists(cpu_path):
            models["cpu_model"] = joblib.load(cpu_path)
        if os.path.exists(ram_path):
            models["ram_model"] = joblib.load(ram_path)
        if os.path.exists(meta_path):
            models["metadata"] = joblib.load(meta_path)
    except Exception as e:
        print(f"Error loading prediction models: {e}")

    try:
        anomaly_path = os.path.join(MODELS_DIR, "anomaly_model.pkl")
        anomaly_meta_path = os.path.join(MODELS_DIR, "anomaly_metadata.pkl")

        if os.path.exists(anomaly_path):
            models["anomaly_model"] = joblib.load(anomaly_path)
        if os.path.exists(anomaly_meta_path):
            models["anomaly_metadata"] = joblib.load(anomaly_meta_path)
    except Exception as e:
        print(f"Error loading anomaly model: {e}")

    return models


def predict_future(models, feature_row):
    """
    Predict CPU and RAM 30 seconds ahead.

    Args:
        models: dict from load_models()
        feature_row: dict of feature name → value

    Returns:
        dict with predictions
    """
    if not models.get("cpu_model") or not models.get("ram_model"):
        return None

    metadata = models.get("metadata", {})
    feature_names = metadata.get("feature_names") if metadata else None

    if not feature_names:
        feature_names = sorted(feature_row.keys())

    X = np.array([[feature_row.get(f, 0) for f in feature_names]])

    cpu_pred = float(models["cpu_model"].predict(X)[0])
    ram_pred = float(models["ram_model"].predict(X)[0])

    # Clamp to valid range
    cpu_pred = max(0.0, min(100.0, cpu_pred))
    ram_pred = max(0.0, min(100.0, ram_pred))

    result = {
        "predicted_cpu_30s": round(cpu_pred, 2),
        "predicted_ram_30s": round(ram_pred, 2),
    }

    # Add feature importances if available
    if hasattr(models["cpu_model"], "feature_importances_"):
        result["feature_importances"] = dict(zip(
            feature_names,
            [round(float(v), 4) for v in models["cpu_model"].feature_importances_],
        ))

    return result
