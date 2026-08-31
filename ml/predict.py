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


# ══════════════════════════════════════════════════════════════════════════
# Deep Learning (LSTM) Model Support
# ══════════════════════════════════════════════════════════════════════════

def load_dl_models():
    """
    Load saved LSTM models for CPU and RAM prediction.

    Returns dict with model objects, scaler, and metadata, or None values
    if PyTorch is not installed or models are not found.
    """
    dl_models = {
        "lstm_cpu": None,
        "lstm_ram": None,
        "scaler": None,
        "dl_metadata": None,
    }

    try:
        import torch
        from dl_model import LSTMPredictor
    except ImportError:
        return dl_models

    try:
        cpu_path = os.path.join(MODELS_DIR, "lstm_cpu_model.pth")
        ram_path = os.path.join(MODELS_DIR, "lstm_ram_model.pth")
        scaler_path = os.path.join(MODELS_DIR, "lstm_scaler.pkl")
        meta_path = os.path.join(MODELS_DIR, "lstm_metadata.pkl")

        if not all(os.path.exists(p) for p in [cpu_path, ram_path, scaler_path, meta_path]):
            return dl_models

        dl_meta = joblib.load(meta_path)
        dl_models["dl_metadata"] = dl_meta
        dl_models["scaler"] = joblib.load(scaler_path)

        input_size = dl_meta.get("input_size", 22)
        device = torch.device("cpu")

        # Load CPU LSTM
        cpu_ckpt = torch.load(cpu_path, map_location=device, weights_only=True)
        cpu_model = LSTMPredictor(
            input_size=cpu_ckpt["input_size"],
            hidden1=cpu_ckpt["hidden1"],
            hidden2=cpu_ckpt["hidden2"],
            dense_size=cpu_ckpt["dense_size"],
            dropout=cpu_ckpt["dropout"],
        )
        cpu_model.load_state_dict(cpu_ckpt["model_state_dict"])
        cpu_model.eval()
        dl_models["lstm_cpu"] = cpu_model

        # Load RAM LSTM
        ram_ckpt = torch.load(ram_path, map_location=device, weights_only=True)
        ram_model = LSTMPredictor(
            input_size=ram_ckpt["input_size"],
            hidden1=ram_ckpt["hidden1"],
            hidden2=ram_ckpt["hidden2"],
            dense_size=ram_ckpt["dense_size"],
            dropout=ram_ckpt["dropout"],
        )
        ram_model.load_state_dict(ram_ckpt["model_state_dict"])
        ram_model.eval()
        dl_models["lstm_ram"] = ram_model

    except Exception as e:
        print(f"Error loading LSTM models: {e}")

    return dl_models


def predict_future_dl(dl_models, feature_history):
    """
    Predict CPU and RAM 30 seconds ahead using LSTM.

    Args:
        dl_models: dict from load_dl_models()
        feature_history: list of dicts (most recent window_size feature rows)

    Returns:
        dict with LSTM predictions or None
    """
    if not dl_models.get("lstm_cpu") or not dl_models.get("lstm_ram"):
        return None

    try:
        import torch

        metadata = dl_models.get("dl_metadata", {})
        feature_names = metadata.get("feature_names", [])
        window_size = metadata.get("window_size", 10)
        scaler = dl_models.get("scaler")

        if len(feature_history) < window_size:
            return None

        # Build window from feature history
        window = []
        for row in feature_history[-window_size:]:
            window.append([row.get(f, 0) for f in feature_names])

        window = np.array(window)

        # Scale features
        if scaler:
            window = scaler.transform(window)

        X = torch.FloatTensor(window).unsqueeze(0)  # (1, window_size, features)

        with torch.no_grad():
            cpu_pred = float(dl_models["lstm_cpu"](X).item())
            ram_pred = float(dl_models["lstm_ram"](X).item())

        cpu_pred = max(0.0, min(100.0, cpu_pred))
        ram_pred = max(0.0, min(100.0, ram_pred))

        return {
            "predicted_cpu_30s": round(cpu_pred, 2),
            "predicted_ram_30s": round(ram_pred, 2),
            "model": "lstm",
        }

    except Exception as e:
        return {"error": str(e)}

