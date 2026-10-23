"""Prediction and measured live-evaluation endpoints for SysSense AI.

The forecasting models are trained on the bundled dataset. At runtime they
receive only features that the collector can actually obtain on this machine.
Every distinct collector snapshot is logged once per model and later compared
with a real collector reading, so the dashboard can show measured live error
instead of an unsupported "accuracy" percentage.
"""
from datetime import datetime, timedelta
import os
import platform
import sys

import psutil
from fastapi import APIRouter

from database import get_forecast_evaluation, get_recent_metrics, insert_forecast
from services.live_metrics import get_live_cpu_percent


router = APIRouter(tags=["Prediction"])

DISK_PATH = "C:\\" if platform.system() == "Windows" else "/"
ML_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ml", "models"
)

FORECAST_HORIZON_SECONDS = 30
MAX_FORECAST_SOURCE_AGE_SECONDS = 15
MIN_HISTORY_ROWS = 10
IDLE_PROCESS_NAMES = {"system idle process", "idle"}
PROCESS_CPU_WARNING = 50
PROCESS_MEMORY_WARNING = 25


# Loaded artefacts and their feature contracts.
_cpu_model = None
_ram_model = None
_model_feature_names = []
_model_metadata = {}
_model_loaded = False

_anomaly_model = None
_anomaly_feature_names = []
_anomaly_metadata = {}
_anomaly_loaded = False

_lstm_cpu_model = None
_lstm_ram_model = None
_lstm_scaler = None
_lstm_feature_names = []
_lstm_metadata = {}
_lstm_window_size = 10
_lstm_loaded = False


def _try_load_models():
    """Load model artefacts and reset stale state on a reload."""
    global _cpu_model, _ram_model, _model_feature_names, _model_metadata, _model_loaded
    global _anomaly_model, _anomaly_feature_names, _anomaly_metadata, _anomaly_loaded
    global _lstm_cpu_model, _lstm_ram_model, _lstm_scaler, _lstm_feature_names
    global _lstm_metadata, _lstm_window_size, _lstm_loaded

    _cpu_model = _ram_model = None
    _model_feature_names = []
    _model_metadata = {}
    _model_loaded = False
    _anomaly_model = None
    _anomaly_feature_names = []
    _anomaly_metadata = {}
    _anomaly_loaded = False
    _lstm_cpu_model = _lstm_ram_model = _lstm_scaler = None
    _lstm_feature_names = []
    _lstm_metadata = {}
    _lstm_window_size = 10
    _lstm_loaded = False

    try:
        import joblib

        cpu_path = os.path.join(ML_MODELS_DIR, "cpu_model.pkl")
        ram_path = os.path.join(ML_MODELS_DIR, "ram_model.pkl")
        meta_path = os.path.join(ML_MODELS_DIR, "model_metadata.pkl")
        if all(os.path.exists(path) for path in (cpu_path, ram_path, meta_path)):
            _cpu_model = joblib.load(cpu_path)
            _ram_model = joblib.load(ram_path)
            _model_metadata = joblib.load(meta_path)
            _model_feature_names = list(_model_metadata.get("feature_names", []))

            # A single forecast does not benefit from joblib worker processes;
            # one worker avoids Windows process-spawn failures.
            if hasattr(_cpu_model, "n_jobs"):
                _cpu_model.n_jobs = 1
            if hasattr(_ram_model, "n_jobs"):
                _ram_model.n_jobs = 1

            _model_loaded = bool(_model_feature_names)
            if _model_loaded:
                print(f"[OK] Random Forest models loaded from {ML_MODELS_DIR}")
            else:
                print("[WARN] Random Forest metadata has no feature contract")
        else:
            print("[INFO] Random Forest artefacts not found - using baseline")
    except Exception as exc:
        print(f"[WARN] Failed to load Random Forest models: {exc}")

    try:
        import joblib

        anomaly_path = os.path.join(ML_MODELS_DIR, "anomaly_model.pkl")
        anomaly_meta_path = os.path.join(ML_MODELS_DIR, "anomaly_metadata.pkl")
        if all(os.path.exists(path) for path in (anomaly_path, anomaly_meta_path)):
            _anomaly_model = joblib.load(anomaly_path)
            _anomaly_metadata = joblib.load(anomaly_meta_path)
            _anomaly_feature_names = list(_anomaly_metadata.get("feature_names", []))
            _anomaly_loaded = bool(_anomaly_feature_names)
            if _anomaly_loaded:
                print("[OK] Isolation Forest anomaly model loaded")
        else:
            print("[INFO] Anomaly model not found - anomaly signal disabled")
    except Exception as exc:
        print(f"[WARN] Failed to load anomaly model: {exc}")

    try:
        import joblib
        import torch

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from ml.dl_model import LSTMPredictor

        cpu_path = os.path.join(ML_MODELS_DIR, "lstm_cpu_model.pth")
        ram_path = os.path.join(ML_MODELS_DIR, "lstm_ram_model.pth")
        scaler_path = os.path.join(ML_MODELS_DIR, "lstm_scaler.pkl")
        meta_path = os.path.join(ML_MODELS_DIR, "lstm_metadata.pkl")
        if all(os.path.exists(path) for path in (cpu_path, ram_path, scaler_path, meta_path)):
            _lstm_metadata = joblib.load(meta_path)
            _lstm_feature_names = list(_lstm_metadata.get("feature_names", []))
            _lstm_window_size = int(_lstm_metadata.get("window_size", 10))
            _lstm_scaler = joblib.load(scaler_path)
            device = torch.device("cpu")

            cpu_checkpoint = torch.load(cpu_path, map_location=device, weights_only=True)
            _lstm_cpu_model = LSTMPredictor(
                input_size=cpu_checkpoint["input_size"],
                hidden1=cpu_checkpoint["hidden1"],
                hidden2=cpu_checkpoint["hidden2"],
                dense_size=cpu_checkpoint["dense_size"],
                dropout=cpu_checkpoint["dropout"],
            )
            _lstm_cpu_model.load_state_dict(cpu_checkpoint["model_state_dict"])
            _lstm_cpu_model.eval()

            ram_checkpoint = torch.load(ram_path, map_location=device, weights_only=True)
            _lstm_ram_model = LSTMPredictor(
                input_size=ram_checkpoint["input_size"],
                hidden1=ram_checkpoint["hidden1"],
                hidden2=ram_checkpoint["hidden2"],
                dense_size=ram_checkpoint["dense_size"],
                dropout=ram_checkpoint["dropout"],
            )
            _lstm_ram_model.load_state_dict(ram_checkpoint["model_state_dict"])
            _lstm_ram_model.eval()
            _lstm_loaded = bool(_lstm_feature_names)
            if _lstm_loaded:
                print("[OK] LSTM models loaded")
        else:
            print("[INFO] LSTM artefacts not found - DL forecast disabled")
    except ImportError:
        print("[INFO] PyTorch not installed - LSTM forecast disabled")
    except Exception as exc:
        print(f"[WARN] Failed to load LSTM models: {exc}")


_try_load_models()


def _as_float(value, default=0.0):
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _parse_timestamp(value):
    """Parse a database ISO timestamp into a naive local datetime."""
    if not value:
        return None
    try:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if timestamp.tzinfo is not None:
            timestamp = timestamp.astimezone().replace(tzinfo=None)
        return timestamp
    except (TypeError, ValueError):
        return None


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _sample_std(values):
    """Match pandas rolling(...).std(), which uses sample deviation."""
    if len(values) < 2:
        return 0.0
    average = _mean(values)
    return (sum((value - average) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _build_collector_feature_rows(recent):
    """Build only features that the local collector records for every sample."""
    rows = []
    cpu_values = [_as_float(item.get("cpu_percent")) for item in recent]
    ram_values = [_as_float(item.get("ram_percent")) for item in recent]
    disk_values = [_as_float(item.get("disk_percent")) for item in recent]
    sent_values = [_as_float(item.get("net_sent_mb")) for item in recent]
    recv_values = [_as_float(item.get("net_recv_mb")) for item in recent]

    for index, item in enumerate(recent):
        cpu_current = cpu_values[index]
        ram_current = ram_values[index]
        disk_current = disk_values[index]
        timestamp = _parse_timestamp(item.get("timestamp"))
        cpu_window5 = cpu_values[max(0, index - 4):index + 1]
        cpu_window10 = cpu_values[max(0, index - 9):index + 1]
        ram_window5 = ram_values[max(0, index - 4):index + 1]
        ram_window10 = ram_values[max(0, index - 9):index + 1]
        previous_cpu = cpu_values[index - 1] if index else cpu_current
        previous_ram = ram_values[index - 1] if index else ram_current
        previous_disk = disk_values[index - 1] if index else disk_current
        net_sent_rate = 0.0
        net_recv_rate = 0.0

        if index:
            previous_timestamp = _parse_timestamp(recent[index - 1].get("timestamp"))
            if timestamp and previous_timestamp:
                elapsed = (timestamp - previous_timestamp).total_seconds()
                if elapsed > 0:
                    net_sent_rate = max(0.0, (sent_values[index] - sent_values[index - 1]) / elapsed)
                    net_recv_rate = max(0.0, (recv_values[index] - recv_values[index - 1]) / elapsed)

        rows.append({
            "cpu_current": cpu_current,
            "ram_current": ram_current,
            "disk_current": disk_current,
            "cpu_lag1": cpu_values[index - 1] if index >= 1 else cpu_current,
            "ram_lag1": ram_values[index - 1] if index >= 1 else ram_current,
            "cpu_lag3": cpu_values[index - 3] if index >= 3 else cpu_current,
            "ram_lag3": ram_values[index - 3] if index >= 3 else ram_current,
            "cpu_lag5": cpu_values[index - 5] if index >= 5 else cpu_current,
            "ram_lag5": ram_values[index - 5] if index >= 5 else ram_current,
            "cpu_ma5": _mean(cpu_window5),
            "cpu_ma10": _mean(cpu_window10),
            "ram_ma5": _mean(ram_window5),
            "ram_ma10": _mean(ram_window10),
            "cpu_delta": cpu_current - previous_cpu,
            "ram_delta": ram_current - previous_ram,
            "disk_delta": disk_current - previous_disk,
            "cpu_std5": _sample_std(cpu_window5),
            "ram_std5": _sample_std(ram_window5),
            "net_sent_rate": net_sent_rate,
            "net_recv_rate": net_recv_rate,
            "thread_count": _as_float(item.get("thread_count")),
            "process_count": _as_float(item.get("process_count")),
            "hour": timestamp.hour if timestamp else 0,
            "minute": timestamp.minute if timestamp else 0,
        })
    return rows


def _build_feature_context():
    """Return rows, latest feature vector, and its collector source row."""
    recent = get_recent_metrics(max(20, _lstm_window_size + 10))
    if len(recent) < MIN_HISTORY_ROWS:
        return None
    rows = _build_collector_feature_rows(recent)
    return {
        "feature_rows": rows,
        "latest_features": rows[-1],
        "source_metric": recent[-1],
    }


def _missing_feature_names(feature_names, feature_row):
    return [name for name in feature_names if name not in feature_row]


def _predict_random_forest(feature_row):
    """Generate an RF forecast only when every model input is real."""
    if not (_model_loaded and _cpu_model and _ram_model):
        return None, {"available": False, "reason": "Random Forest model is not loaded."}

    missing = _missing_feature_names(_model_feature_names, feature_row)
    if missing:
        return None, {
            "available": False,
            "reason": "The loaded model requires features the live collector does not record. Retrain with the collector-compatible feature profile.",
            "missing_features": missing,
        }

    try:
        import numpy as np

        matrix = np.array([[feature_row[name] for name in _model_feature_names]], dtype=float)
        cpu_prediction = max(0.0, min(100.0, float(_cpu_model.predict(matrix)[0])))
        ram_prediction = max(0.0, min(100.0, float(_ram_model.predict(matrix)[0])))
        return {
            "predicted_cpu_30s": round(cpu_prediction, 2),
            "predicted_ram_30s": round(ram_prediction, 2),
        }, {"available": True}
    except Exception as exc:
        return None, {"available": False, "reason": str(exc)}


def _predict_lstm(feature_rows):
    """Generate an LSTM forecast from collector-compatible feature history."""
    if not (_lstm_loaded and _lstm_cpu_model and _lstm_ram_model):
        return None, {"available": False, "reason": "LSTM model is not loaded."}
    if len(feature_rows) < _lstm_window_size:
        return None, {
            "available": False,
            "reason": f"Need {_lstm_window_size} collector samples for the LSTM window.",
        }

    missing = _missing_feature_names(_lstm_feature_names, feature_rows[-1])
    if missing:
        return None, {
            "available": False,
            "reason": "The loaded LSTM requires features the live collector does not record. Retrain with the collector-compatible feature profile.",
            "missing_features": missing,
        }

    try:
        import numpy as np
        import torch

        window = np.array(
            [[row[name] for name in _lstm_feature_names]
             for row in feature_rows[-_lstm_window_size:]],
            dtype="float32",
        )
        if _lstm_scaler is not None:
            window = _lstm_scaler.transform(window)
        tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            cpu_prediction = float(_lstm_cpu_model(tensor).item())
            ram_prediction = float(_lstm_ram_model(tensor).item())
        return {
            "predicted_cpu_30s": round(max(0.0, min(100.0, cpu_prediction)), 2),
            "predicted_ram_30s": round(max(0.0, min(100.0, ram_prediction)), 2),
            "model": "lstm",
        }, {"available": True}
    except Exception as exc:
        return None, {"available": False, "reason": str(exc)}


def _anomaly_input(feature_row):
    return {
        "cpu_percent": feature_row["cpu_current"],
        "ram_percent": feature_row["ram_current"],
        "disk_percent": feature_row["disk_current"],
        "net_sent_rate": feature_row["net_sent_rate"],
        "net_recv_rate": feature_row["net_recv_rate"],
        "thread_count": feature_row["thread_count"],
        "process_count": feature_row["process_count"],
        "cpu_ram_product": feature_row["cpu_current"] * feature_row["ram_current"] / 100,
        "cpu_delta": feature_row["cpu_delta"],
        "ram_delta": feature_row["ram_delta"],
    }


def _get_anomaly_status(feature_row):
    """Run Isolation Forest only with an exactly matched feature schema."""
    if not (_anomaly_loaded and _anomaly_model):
        return {"available": False, "is_anomaly": False, "reason": "Anomaly model is not loaded."}

    metrics = _anomaly_input(feature_row)
    missing = _missing_feature_names(_anomaly_feature_names, metrics)
    if missing:
        return {
            "available": False,
            "is_anomaly": False,
            "reason": "The loaded anomaly model requires unavailable live features. Retrain it with the collector-compatible profile.",
            "missing_features": missing,
        }

    try:
        import numpy as np

        matrix = np.array([[metrics[name] for name in _anomaly_feature_names]], dtype=float)
        prediction = _anomaly_model.predict(matrix)[0]
        score = float(_anomaly_model.score_samples(matrix)[0])
        is_anomaly = bool(prediction == -1)
        cpu = metrics["cpu_percent"]
        ram = metrics["ram_percent"]
        details = []
        if is_anomaly:
            if cpu > 80:
                details.append(f"CPU is high ({cpu:.1f}%)")
            if ram > 90:
                details.append(f"RAM is high ({ram:.1f}%)")
            if abs(metrics["cpu_delta"]) > 30:
                details.append(f"CPU changed sharply ({metrics['cpu_delta']:+.1f} percentage points)")
            if abs(metrics["ram_delta"]) > 15:
                details.append(f"RAM changed sharply ({metrics['ram_delta']:+.1f} percentage points)")
            if not details:
                details.append("The model flagged an unusual collector-metric combination")

        if not is_anomaly:
            severity = "normal"
        elif score < -0.3:
            severity = "critical"
        elif score < -0.2:
            severity = "warning"
        else:
            severity = "mild"

        return {
            "available": True,
            "is_anomaly": is_anomaly,
            "anomaly_score": round(score, 4),
            "severity": severity,
            "details": details,
            "current_metrics": {
                "cpu": round(cpu, 1),
                "ram": round(ram, 1),
                "disk": round(metrics["disk_percent"], 1),
            },
        }
    except Exception as exc:
        return {"available": False, "is_anomaly": False, "reason": str(exc)}


def _get_process_threshold_signals():
    """Return rule-based high-resource process signals, not ML anomalies."""
    signals = []
    try:
        for process in psutil.process_iter(
            ["pid", "name", "cpu_percent", "memory_percent"]
        ):
            try:
                info = process.info
                pid = info.get("pid", 0)
                name = info.get("name") or "Unknown"
                if pid == 0 or name.casefold() in IDLE_PROCESS_NAMES:
                    # Windows reports idle time as a process CPU percentage.
                    continue

                cpu_percent = _as_float(info.get("cpu_percent"))
                memory_percent = _as_float(info.get("memory_percent"))
                if cpu_percent > PROCESS_CPU_WARNING:
                    signals.append({
                        "process": name,
                        "pid": pid,
                        "type": "high_cpu",
                        "source": "process_threshold",
                        "severity": "critical" if cpu_percent > 80 else "warning",
                        "message": f"{name} exceeds the high-CPU rule ({cpu_percent:.1f}% sampled process CPU)",
                        "value": round(cpu_percent, 1),
                    })
                if memory_percent > PROCESS_MEMORY_WARNING:
                    signals.append({
                        "process": name,
                        "pid": pid,
                        "type": "high_memory",
                        "source": "process_threshold",
                        "severity": "critical" if memory_percent > 40 else "warning",
                        "message": f"{name} exceeds the high-memory rule ({memory_percent:.1f}% of RAM)",
                        "value": round(memory_percent, 1),
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass
    return signals


def _record_forecasts(context, random_forest_prediction, lstm_prediction):
    """Persist forecasts when their collector source is fresh enough to test."""
    source = context.get("source_metric") if context else None
    if not source or source.get("id") is None:
        return {"logged_models": [], "reason": "No collector source sample is available."}

    source_time = _parse_timestamp(source.get("timestamp"))
    if source_time is None:
        return {"logged_models": [], "reason": "Collector timestamp is invalid."}

    age = (datetime.now() - source_time).total_seconds()
    if age < 0 or age > MAX_FORECAST_SOURCE_AGE_SECONDS:
        return {
            "logged_models": [],
            "reason": "Collector sample is too old to create a valid live forecast.",
            "source_age_seconds": round(age, 1),
        }

    target_time = source_time + timedelta(seconds=FORECAST_HORIZON_SECONDS)
    # Persistence is the meaningful live baseline: predict that the collector
    # values at the source time will remain unchanged for the horizon. Logging
    # it alongside ML/DL forecasts makes the local evaluation table a real
    # comparison instead of an isolated model-error display.
    baseline_prediction = {
        "predicted_cpu_30s": _as_float(source.get("cpu_percent")),
        "predicted_ram_30s": _as_float(source.get("ram_percent")),
    }
    predictions = [("naive_baseline", baseline_prediction)]
    if random_forest_prediction:
        predictions.append(("random_forest", random_forest_prediction))
    if lstm_prediction:
        predictions.append(("lstm", lstm_prediction))

    logged_models = []
    for model_name, values in predictions:
        if insert_forecast(
            source["id"], source_time.isoformat(), target_time.isoformat(), model_name,
            values["predicted_cpu_30s"], values["predicted_ram_30s"],
        ):
            logged_models.append(model_name)

    return {
        "logged_models": logged_models,
        "source_timestamp": source_time.isoformat(),
        "target_timestamp": target_time.isoformat(),
        "source_age_seconds": round(age, 1),
    }


@router.get("/predict/evaluation")
def prediction_evaluation():
    """Measured error for completed live forecasts, grouped by model."""
    return get_forecast_evaluation()


@router.get("/predict")
def predict():
    """Return live telemetry plus dataset-trained forecast estimates."""
    live_cpu = get_live_cpu_percent()
    live_ram = psutil.virtual_memory().percent
    live_disk = psutil.disk_usage(DISK_PATH).percent
    context = _build_feature_context()

    result = {
        "current": {"cpu": live_cpu, "ram": live_ram, "disk": live_disk},
        "naive_baseline": {
            "predicted_cpu_30s": live_cpu,
            "predicted_ram_30s": live_ram,
        },
        "model_used": "naive_baseline",
        "feature_importances": None,
        "anomaly": None,
        "process_anomalies": _get_process_threshold_signals(),
        "forecast_horizon_seconds": FORECAST_HORIZON_SECONDS,
        "model_training_source": _model_metadata.get(
            "trained_on", "dataset metadata unavailable"
        ),
    }

    if context is None:
        result["ml_status"] = {
            "available": False,
            "reason": f"Need at least {MIN_HISTORY_ROWS} collector readings before forecasting.",
        }
        result["dl_status"] = dict(result["ml_status"])
        result["anomaly"] = {
            "available": False,
            "is_anomaly": False,
            "reason": "Need collector history before evaluating the anomaly model.",
        }
        return result

    latest_features = context["latest_features"]
    random_forest_prediction, ml_status = _predict_random_forest(latest_features)
    result["ml_status"] = ml_status
    if random_forest_prediction:
        result["ml_prediction"] = random_forest_prediction
        result["model_used"] = "random_forest"
        if hasattr(_cpu_model, "feature_importances_"):
            result["feature_importances"] = dict(zip(
                _model_feature_names,
                [round(float(value), 4) for value in _cpu_model.feature_importances_],
            ))

    lstm_prediction, dl_status = _predict_lstm(context["feature_rows"])
    result["dl_status"] = dl_status
    if lstm_prediction:
        result["dl_prediction"] = lstm_prediction

    result["anomaly"] = _get_anomaly_status(latest_features)
    result["forecast_validation"] = _record_forecasts(
        context, random_forest_prediction, lstm_prediction
    )
    source = context["source_metric"]
    result["forecast_basis"] = {
        "source_metric_id": source.get("id"),
        "source_timestamp": source.get("timestamp"),
        "source_metrics": {
            "cpu": round(_as_float(source.get("cpu_percent")), 2),
            "ram": round(_as_float(source.get("ram_percent")), 2),
            "disk": round(_as_float(source.get("disk_percent")), 2),
        },
        "feature_profile": _model_metadata.get("feature_profile", "legacy_or_unspecified"),
    }
    return result


@router.get("/predict/reload")
def reload_models():
    """Force-reload models after a documented retraining run."""
    _try_load_models()
    return {
        "model_loaded": _model_loaded,
        "model_feature_names": _model_feature_names,
        "anomaly_loaded": _anomaly_loaded,
        "anomaly_feature_names": _anomaly_feature_names,
        "lstm_loaded": _lstm_loaded,
        "lstm_feature_names": _lstm_feature_names,
    }
