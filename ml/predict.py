"""Safe standalone inference helpers for SysSense model artifacts.

The FastAPI deployment route owns production inference. These helpers are kept
for notebooks or scripts and enforce the same feature contract: no missing
model input is silently replaced with a fabricated zero.
"""
import os

import joblib
import numpy as np


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def load_models():
    """Load Random Forest and anomaly artifacts for standalone use."""
    models = {
        "cpu_model": None,
        "ram_model": None,
        "anomaly_model": None,
        "metadata": {},
        "anomaly_metadata": {},
    }
    try:
        cpu_path = os.path.join(MODELS_DIR, "cpu_model.pkl")
        ram_path = os.path.join(MODELS_DIR, "ram_model.pkl")
        metadata_path = os.path.join(MODELS_DIR, "model_metadata.pkl")
        if all(os.path.exists(path) for path in (cpu_path, ram_path, metadata_path)):
            models["cpu_model"] = joblib.load(cpu_path)
            models["ram_model"] = joblib.load(ram_path)
            models["metadata"] = joblib.load(metadata_path)
    except Exception as exc:
        print(f"Error loading Random Forest models: {exc}")

    try:
        anomaly_path = os.path.join(MODELS_DIR, "anomaly_model.pkl")
        anomaly_metadata_path = os.path.join(MODELS_DIR, "anomaly_metadata.pkl")
        if all(os.path.exists(path) for path in (anomaly_path, anomaly_metadata_path)):
            models["anomaly_model"] = joblib.load(anomaly_path)
            models["anomaly_metadata"] = joblib.load(anomaly_metadata_path)
    except Exception as exc:
        print(f"Error loading anomaly model: {exc}")
    return models


def _missing_features(feature_names, row):
    return [name for name in feature_names if name not in row]


def predict_future(models, feature_row):
    """Predict CPU/RAM only when every required feature is available.

    Returns an availability status rather than substituting unavailable inputs.
    """
    cpu_model = models.get("cpu_model")
    ram_model = models.get("ram_model")
    metadata = models.get("metadata") or {}
    feature_names = list(metadata.get("feature_names", []))
    if not (cpu_model and ram_model and feature_names):
        return {"available": False, "reason": "Random Forest artifacts are not loaded."}

    missing = _missing_features(feature_names, feature_row)
    if missing:
        return {
            "available": False,
            "reason": "Required live features are unavailable; prediction was not made.",
            "missing_features": missing,
        }

    matrix = np.array([[feature_row[name] for name in feature_names]], dtype=float)
    cpu_prediction = max(0.0, min(100.0, float(cpu_model.predict(matrix)[0])))
    ram_prediction = max(0.0, min(100.0, float(ram_model.predict(matrix)[0])))
    result = {
        "available": True,
        "predicted_cpu_30s": round(cpu_prediction, 2),
        "predicted_ram_30s": round(ram_prediction, 2),
    }
    if hasattr(cpu_model, "feature_importances_"):
        result["feature_importances"] = dict(zip(
            feature_names, [round(float(value), 4) for value in cpu_model.feature_importances_],
        ))
    return result


def load_dl_models():
    """Load LSTM artifacts for standalone use."""
    models = {
        "lstm_cpu": None,
        "lstm_ram": None,
        "scaler": None,
        "dl_metadata": {},
    }
    try:
        import torch
        try:
            from .dl_model import LSTMPredictor
        except ImportError:
            from dl_model import LSTMPredictor
    except ImportError:
        return models

    paths = {
        "cpu": os.path.join(MODELS_DIR, "lstm_cpu_model.pth"),
        "ram": os.path.join(MODELS_DIR, "lstm_ram_model.pth"),
        "scaler": os.path.join(MODELS_DIR, "lstm_scaler.pkl"),
        "metadata": os.path.join(MODELS_DIR, "lstm_metadata.pkl"),
    }
    if not all(os.path.exists(path) for path in paths.values()):
        return models

    try:
        metadata = joblib.load(paths["metadata"])
        models["dl_metadata"] = metadata
        models["scaler"] = joblib.load(paths["scaler"])
        for target in ("cpu", "ram"):
            checkpoint = torch.load(paths[target], map_location="cpu", weights_only=True)
            model = LSTMPredictor(
                input_size=checkpoint["input_size"],
                hidden1=checkpoint["hidden1"],
                hidden2=checkpoint["hidden2"],
                dense_size=checkpoint["dense_size"],
                dropout=checkpoint["dropout"],
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            models[f"lstm_{target}"] = model
    except Exception as exc:
        print(f"Error loading LSTM models: {exc}")
    return models


def predict_future_dl(models, feature_history):
    """Predict from a complete collector-compatible LSTM feature window."""
    cpu_model = models.get("lstm_cpu")
    ram_model = models.get("lstm_ram")
    metadata = models.get("dl_metadata") or {}
    feature_names = list(metadata.get("feature_names", []))
    window_size = int(metadata.get("window_size", 10))
    if not (cpu_model and ram_model and feature_names):
        return {"available": False, "reason": "LSTM artifacts are not loaded."}
    if len(feature_history) < window_size:
        return {
            "available": False,
            "reason": f"Need {window_size} complete feature rows for the LSTM.",
        }

    window_rows = feature_history[-window_size:]
    missing = sorted({
        name for row in window_rows for name in _missing_features(feature_names, row)
    })
    if missing:
        return {
            "available": False,
            "reason": "Required live features are unavailable; prediction was not made.",
            "missing_features": missing,
        }

    import torch

    window = np.array(
        [[row[name] for name in feature_names] for row in window_rows],
        dtype="float32",
    )
    scaler = models.get("scaler")
    if scaler is not None:
        window = scaler.transform(window)
    tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        cpu_prediction = float(cpu_model(tensor).item())
        ram_prediction = float(ram_model(tensor).item())
    return {
        "available": True,
        "predicted_cpu_30s": round(max(0.0, min(100.0, cpu_prediction)), 2),
        "predicted_ram_30s": round(max(0.0, min(100.0, ram_prediction)), 2),
        "model": "lstm",
    }

