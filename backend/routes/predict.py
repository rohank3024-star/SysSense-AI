"""
Prediction endpoints for SysSense AI.

Starts with a naive baseline (predict next = current).
After ML training, this loads the trained model and returns real predictions.
"""
from fastapi import APIRouter
import psutil
import os
import platform

router = APIRouter(tags=["Prediction"])

DISK_PATH = "C:\\" if platform.system() == "Windows" else "/"

# Path to trained model files
ML_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml", "models")

# Try to load trained models at import time
_cpu_model = None
_ram_model = None
_model_loaded = False

def _try_load_models():
    global _cpu_model, _ram_model, _model_loaded
    try:
        import joblib
        cpu_path = os.path.join(ML_MODELS_DIR, "cpu_model.pkl")
        ram_path = os.path.join(ML_MODELS_DIR, "ram_model.pkl")
        if os.path.exists(cpu_path) and os.path.exists(ram_path):
            _cpu_model = joblib.load(cpu_path)
            _ram_model = joblib.load(ram_path)
            _model_loaded = True
            print(f"[OK] ML models loaded from {ML_MODELS_DIR}")
        else:
            print(f"[INFO] ML models not found at {ML_MODELS_DIR} - using naive baseline")
    except ImportError:
        print("[INFO] joblib not installed - using naive baseline")
    except Exception as e:
        print(f"[WARN] Failed to load ML models: {e} - using naive baseline")

_try_load_models()


def _build_feature_row():
    """
    Build a feature row for prediction from the most recent metrics.
    This must match the feature engineering done during training.
    """
    from database import get_recent_metrics
    import numpy as np

    recent = get_recent_metrics(20)
    if len(recent) < 5:
        return None

    latest = recent[-1]
    cpu_vals = [r["cpu_percent"] for r in recent if r.get("cpu_percent") is not None]
    ram_vals = [r["ram_percent"] for r in recent if r.get("ram_percent") is not None]

    from datetime import datetime
    try:
        ts = datetime.fromisoformat(latest["timestamp"])
        hour = ts.hour
        minute = ts.minute
    except (ValueError, KeyError):
        hour = 12
        minute = 0

    features = {
        "cpu_current": latest.get("cpu_percent", 0),
        "ram_current": latest.get("ram_percent", 0),
        "disk_current": latest.get("disk_percent", 0),
        "cpu_lag1": cpu_vals[-2] if len(cpu_vals) >= 2 else cpu_vals[-1],
        "cpu_lag3": cpu_vals[-4] if len(cpu_vals) >= 4 else cpu_vals[-1],
        "ram_lag1": ram_vals[-2] if len(ram_vals) >= 2 else ram_vals[-1],
        "ram_lag3": ram_vals[-4] if len(ram_vals) >= 4 else ram_vals[-1],
        "cpu_ma5": float(np.mean(cpu_vals[-5:])) if len(cpu_vals) >= 5 else float(np.mean(cpu_vals)),
        "ram_ma5": float(np.mean(ram_vals[-5:])) if len(ram_vals) >= 5 else float(np.mean(ram_vals)),
        "cpu_delta": cpu_vals[-1] - cpu_vals[-2] if len(cpu_vals) >= 2 else 0,
        "ram_delta": ram_vals[-1] - ram_vals[-2] if len(ram_vals) >= 2 else 0,
        "hour": hour,
        "minute": minute,
    }
    return features


@router.get("/predict")
def predict():
    """
    Returns predicted CPU and RAM usage 30 seconds ahead.

    If trained ML models exist, uses them. Otherwise falls back to
    the naive baseline (predicted = current). Keep both around —
    the comparison is the key result for your report.
    """
    cpu = psutil.cpu_percent(interval=0)
    ram = psutil.virtual_memory().percent
    disk = psutil.disk_usage(DISK_PATH).percent

    naive_cpu = cpu
    naive_ram = ram

    result = {
        "current": {"cpu": cpu, "ram": ram, "disk": disk},
        "naive_baseline": {"predicted_cpu_30s": naive_cpu, "predicted_ram_30s": naive_ram},
        "model_used": "naive_baseline",
        "confidence": None,
        "feature_importances": None,
    }

    if _model_loaded and _cpu_model and _ram_model:
        features = _build_feature_row()
        if features:
            import numpy as np
            feature_names = sorted(features.keys())
            X = np.array([[features[f] for f in feature_names]])

            try:
                pred_cpu = float(_cpu_model.predict(X)[0])
                pred_ram = float(_ram_model.predict(X)[0])

                # Clamp predictions to valid range
                pred_cpu = max(0, min(100, pred_cpu))
                pred_ram = max(0, min(100, pred_ram))

                result["ml_prediction"] = {
                    "predicted_cpu_30s": round(pred_cpu, 2),
                    "predicted_ram_30s": round(pred_ram, 2),
                }
                result["model_used"] = "random_forest"

                # Feature importances for explainability
                if hasattr(_cpu_model, "feature_importances_"):
                    importances = dict(zip(
                        feature_names,
                        [round(float(v), 4) for v in _cpu_model.feature_importances_],
                    ))
                    result["feature_importances"] = importances

            except Exception as e:
                result["ml_error"] = str(e)

    return result


@router.get("/predict/reload")
def reload_models():
    """Force-reload ML models (after retraining)."""
    _try_load_models()
    return {"model_loaded": _model_loaded}
