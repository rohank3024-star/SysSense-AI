"""
Prediction endpoints for SysSense AI.

Uses ML models trained on a PUBLIC DATASET for predictions.
Current system metrics (collected via psutil) are passed to the
trained model for live predictions.

Includes:
    - CPU/RAM prediction 30 seconds ahead
    - Anomaly detection using Isolation Forest
    - Per-process anomaly warnings
"""
from fastapi import APIRouter
import psutil
import os
import sys
import platform

router = APIRouter(tags=["Prediction"])

DISK_PATH = "C:\\" if platform.system() == "Windows" else "/"

# Path to trained model files
ML_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ml", "models")

# Try to load trained models at import time
_cpu_model = None
_ram_model = None
_anomaly_model = None
_anomaly_feature_names = None
_model_loaded = False
_anomaly_loaded = False

# Deep Learning (LSTM) models
_lstm_cpu_model = None
_lstm_ram_model = None
_lstm_scaler = None
_lstm_feature_names = None
_lstm_window_size = 10
_lstm_loaded = False


def _try_load_models():
    global _cpu_model, _ram_model, _model_loaded
    global _anomaly_model, _anomaly_feature_names, _anomaly_loaded
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

    try:
        import joblib
        anomaly_path = os.path.join(ML_MODELS_DIR, "anomaly_model.pkl")
        anomaly_meta_path = os.path.join(ML_MODELS_DIR, "anomaly_metadata.pkl")
        if os.path.exists(anomaly_path) and os.path.exists(anomaly_meta_path):
            _anomaly_model = joblib.load(anomaly_path)
            meta = joblib.load(anomaly_meta_path)
            _anomaly_feature_names = meta.get("feature_names", [])
            _anomaly_loaded = True
            print(f"[OK] Anomaly detection model loaded")
        else:
            print(f"[INFO] Anomaly model not found - anomaly detection disabled")
    except Exception as e:
        print(f"[WARN] Failed to load anomaly model: {e}")

    # Load LSTM deep learning models
    global _lstm_cpu_model, _lstm_ram_model, _lstm_scaler
    global _lstm_feature_names, _lstm_window_size, _lstm_loaded
    try:
        import torch
        import joblib
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from ml.dl_model import LSTMPredictor

        lstm_cpu_path = os.path.join(ML_MODELS_DIR, "lstm_cpu_model.pth")
        lstm_ram_path = os.path.join(ML_MODELS_DIR, "lstm_ram_model.pth")
        lstm_scaler_path = os.path.join(ML_MODELS_DIR, "lstm_scaler.pkl")
        lstm_meta_path = os.path.join(ML_MODELS_DIR, "lstm_metadata.pkl")

        if all(os.path.exists(p) for p in [lstm_cpu_path, lstm_ram_path, lstm_scaler_path, lstm_meta_path]):
            device = torch.device("cpu")
            lstm_meta = joblib.load(lstm_meta_path)
            _lstm_feature_names = lstm_meta.get("feature_names", [])
            _lstm_window_size = lstm_meta.get("window_size", 10)
            _lstm_scaler = joblib.load(lstm_scaler_path)

            cpu_ckpt = torch.load(lstm_cpu_path, map_location=device, weights_only=True)
            _lstm_cpu_model = LSTMPredictor(
                input_size=cpu_ckpt["input_size"],
                hidden1=cpu_ckpt["hidden1"],
                hidden2=cpu_ckpt["hidden2"],
                dense_size=cpu_ckpt["dense_size"],
                dropout=cpu_ckpt["dropout"],
            )
            _lstm_cpu_model.load_state_dict(cpu_ckpt["model_state_dict"])
            _lstm_cpu_model.eval()

            ram_ckpt = torch.load(lstm_ram_path, map_location=device, weights_only=True)
            _lstm_ram_model = LSTMPredictor(
                input_size=ram_ckpt["input_size"],
                hidden1=ram_ckpt["hidden1"],
                hidden2=ram_ckpt["hidden2"],
                dense_size=ram_ckpt["dense_size"],
                dropout=ram_ckpt["dropout"],
            )
            _lstm_ram_model.load_state_dict(ram_ckpt["model_state_dict"])
            _lstm_ram_model.eval()

            _lstm_loaded = True
            print(f"[OK] LSTM deep learning models loaded")
        else:
            print(f"[INFO] LSTM models not found - DL prediction disabled")
    except ImportError:
        print(f"[INFO] PyTorch not installed - LSTM prediction disabled")
    except Exception as e:
        print(f"[WARN] Failed to load LSTM models: {e}")

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
        "cpu_lag5": cpu_vals[-6] if len(cpu_vals) >= 6 else cpu_vals[-1],
        "ram_lag1": ram_vals[-2] if len(ram_vals) >= 2 else ram_vals[-1],
        "ram_lag3": ram_vals[-4] if len(ram_vals) >= 4 else ram_vals[-1],
        "ram_lag5": ram_vals[-6] if len(ram_vals) >= 6 else ram_vals[-1],
        "cpu_ma5": float(np.mean(cpu_vals[-5:])) if len(cpu_vals) >= 5 else float(np.mean(cpu_vals)),
        "cpu_ma10": float(np.mean(cpu_vals[-10:])) if len(cpu_vals) >= 10 else float(np.mean(cpu_vals)),
        "ram_ma5": float(np.mean(ram_vals[-5:])) if len(ram_vals) >= 5 else float(np.mean(ram_vals)),
        "ram_ma10": float(np.mean(ram_vals[-10:])) if len(ram_vals) >= 10 else float(np.mean(ram_vals)),
        "cpu_delta": cpu_vals[-1] - cpu_vals[-2] if len(cpu_vals) >= 2 else 0,
        "ram_delta": ram_vals[-1] - ram_vals[-2] if len(ram_vals) >= 2 else 0,
        "disk_delta": 0,
        "cpu_std5": float(np.std(cpu_vals[-5:])) if len(cpu_vals) >= 5 else 0,
        "ram_std5": float(np.std(ram_vals[-5:])) if len(ram_vals) >= 5 else 0,
        "net_sent_rate": 0,
        "net_recv_rate": 0,
        "network_latency": 0,
        "context_switches": 0,
        "cache_miss_rate": 0,
        "temperature": 0,
        "power_consumption": 0,
        "hour": hour,
        "minute": minute,
    }
    return features


def _get_anomaly_status(cpu, ram, disk):
    """Run anomaly detection on current metrics."""
    if not _anomaly_loaded or not _anomaly_model:
        return None

    import numpy as np

    metrics = {
        "cpu_percent": cpu,
        "ram_percent": ram,
        "disk_percent": disk,
        "net_sent_rate": 0,
        "net_recv_rate": 0,
        "cpu_ram_product": cpu * ram / 100,
        "cpu_delta": 0,
        "ram_delta": 0,
    }

    X = np.array([[metrics.get(f, 0) for f in _anomaly_feature_names]])

    try:
        prediction = _anomaly_model.predict(X)[0]
        score = _anomaly_model.score_samples(X)[0]

        is_anomaly = prediction == -1
        severity = "normal"
        details = []

        if is_anomaly:
            if score < -0.3:
                severity = "critical"
            elif score < -0.2:
                severity = "warning"
            else:
                severity = "mild"

            if cpu > 80:
                details.append(f"CPU unusually high ({cpu:.1f}%)")
            if ram > 90:
                details.append(f"RAM unusually high ({ram:.1f}%)")
            if cpu > 70 and ram > 85:
                details.append("Combined high CPU + RAM may indicate resource exhaustion")
            if not details:
                details.append("Unusual metric combination detected by ML model")

        return {
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": round(float(score), 4),
            "severity": severity,
            "details": details,
            "current_metrics": {
                "cpu": round(float(cpu), 1),
                "ram": round(float(ram), 1),
                "disk": round(float(disk), 1),
            },
        }
    except Exception as e:
        return {"is_anomaly": False, "error": str(e)}


def _get_process_anomalies():
    """Detect per-process anomalies (e.g., Chrome using high CPU)."""
    anomalies = []
    try:
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                info = p.info
                cpu_pct = info.get("cpu_percent") or 0
                mem_pct = info.get("memory_percent") or 0
                name = info.get("name", "Unknown")

                if cpu_pct > 50:
                    anomalies.append({
                        "process": name,
                        "pid": info.get("pid", 0),
                        "type": "high_cpu",
                        "severity": "critical" if cpu_pct > 80 else "warning",
                        "message": f"{name} is consuming unusually high CPU ({cpu_pct:.1f}%)",
                        "value": round(cpu_pct, 1),
                    })

                if mem_pct > 20:
                    anomalies.append({
                        "process": name,
                        "pid": info.get("pid", 0),
                        "type": "high_memory",
                        "severity": "critical" if mem_pct > 40 else "warning",
                        "message": f"{name} is consuming unusually high memory ({mem_pct:.1f}%)",
                        "value": round(mem_pct, 1),
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass

    return anomalies


@router.get("/predict")
def predict():
    """
    Returns predicted CPU and RAM usage 30 seconds ahead.

    Uses ML models trained on a public dataset. If trained models exist,
    uses them. Otherwise falls back to the naive baseline (predicted = current).

    Also includes anomaly detection results and per-process anomaly warnings.
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
        "anomaly": None,
        "process_anomalies": [],
    }

    if _model_loaded and _cpu_model and _ram_model:
        features = _build_feature_row()
        if features:
            import numpy as np
            # IMPORTANT:
            # These features MUST be in the exact same order used during training.
            feature_names = [
                "cpu_current",
                "ram_current",
                "disk_current",
                "cpu_lag1",
                "ram_lag1",
                "cpu_lag3",
                "ram_lag3",
                "cpu_lag5",
                "ram_lag5",
                "cpu_ma5",
                "cpu_ma10",
                "ram_ma5",
                "ram_ma10",
                "cpu_delta",
                "ram_delta",
                "disk_delta",
                "cpu_std5",
                "ram_std5",
                "net_sent_rate",
                "net_recv_rate",
                "network_latency",
                "context_switches",
                "cache_miss_rate",
                "temperature",
                "power_consumption",
                "hour",
                "minute",
            ]

            X = np.array([[features.get(f, 0) for f in feature_names]])

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

    # Anomaly detection
    anomaly_result = _get_anomaly_status(cpu, ram, disk)
    if anomaly_result:
        result["anomaly"] = anomaly_result

    # Per-process anomalies
    result["process_anomalies"] = _get_process_anomalies()

    # ── Deep Learning (LSTM) prediction ───────────────────────────────
    if _lstm_loaded and _lstm_cpu_model and _lstm_ram_model:
        try:
            import torch
            import numpy as np
            import statistics
            from datetime import datetime
            from database import get_recent_metrics

            # We need enough history for:
            # - 10 timestep LSTM window
            # - lag-5 features
            # - MA10 features
            recent = get_recent_metrics(max(_lstm_window_size + 10, 20))

            if len(recent) >= _lstm_window_size:

                def get_value(item, key, default=0.0):
                    value = item.get(key)
                    if value is None:
                        return default
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return default

                def get_timestamp(item):
                    ts = item.get("timestamp")

                    if not ts:
                        return None

                    try:
                        ts = str(ts).replace("Z", "+00:00")
                        return datetime.fromisoformat(ts)
                    except (ValueError, TypeError):
                        return None

                # Extract historical values
                cpu_values = [
                    get_value(m, "cpu_percent")
                    for m in recent
                ]

                ram_values = [
                    get_value(m, "ram_percent")
                    for m in recent
                ]

                disk_values = [
                    get_value(m, "disk_percent")
                    for m in recent
                ]

                sent_values = [
                    get_value(m, "net_sent_mb")
                    for m in recent
                ]

                recv_values = [
                    get_value(m, "net_recv_mb")
                    for m in recent
                ]

                # Build feature rows
                feature_rows = []

                for i in range(len(recent)):

                    cpu_current = cpu_values[i]
                    ram_current = ram_values[i]
                    disk_current = disk_values[i]

                    # ── Lag features ──────────────────────────────
                    cpu_lag1 = cpu_values[i - 1] if i >= 1 else cpu_current
                    ram_lag1 = ram_values[i - 1] if i >= 1 else ram_current

                    cpu_lag3 = cpu_values[i - 3] if i >= 3 else cpu_current
                    ram_lag3 = ram_values[i - 3] if i >= 3 else ram_current

                    cpu_lag5 = cpu_values[i - 5] if i >= 5 else cpu_current
                    ram_lag5 = ram_values[i - 5] if i >= 5 else ram_current

                    # ── Moving averages ───────────────────────────
                    cpu_ma5_values = cpu_values[max(0, i - 4):i + 1]
                    cpu_ma10_values = cpu_values[max(0, i - 9):i + 1]

                    ram_ma5_values = ram_values[max(0, i - 4):i + 1]
                    ram_ma10_values = ram_values[max(0, i - 9):i + 1]

                    cpu_ma5 = sum(cpu_ma5_values) / len(cpu_ma5_values)
                    cpu_ma10 = sum(cpu_ma10_values) / len(cpu_ma10_values)

                    ram_ma5 = sum(ram_ma5_values) / len(ram_ma5_values)
                    ram_ma10 = sum(ram_ma10_values) / len(ram_ma10_values)

                    # ── Delta features ────────────────────────────
                    previous_cpu = cpu_values[i - 1] if i >= 1 else cpu_current
                    previous_ram = ram_values[i - 1] if i >= 1 else ram_current
                    previous_disk = disk_values[i - 1] if i >= 1 else disk_current

                    cpu_delta = cpu_current - previous_cpu
                    ram_delta = ram_current - previous_ram
                    disk_delta = disk_current - previous_disk

                    # ── Rolling standard deviation ────────────────
                    cpu_std_values = cpu_values[max(0, i - 4):i + 1]
                    ram_std_values = ram_values[max(0, i - 4):i + 1]

                    if len(cpu_std_values) > 1:
                        cpu_std5 = statistics.stdev(cpu_std_values)
                    else:
                        cpu_std5 = 0.0

                    if len(ram_std_values) > 1:
                        ram_std5 = statistics.stdev(ram_std_values)
                    else:
                        ram_std5 = 0.0

                    # ── Network rates ─────────────────────────────
                    net_sent_rate = 0.0
                    net_recv_rate = 0.0

                    if i >= 1:
                        previous_timestamp = get_timestamp(recent[i - 1])
                        current_timestamp = get_timestamp(recent[i])

                        if previous_timestamp and current_timestamp:
                            delta_seconds = (
                                current_timestamp - previous_timestamp
                            ).total_seconds()

                            if delta_seconds > 0:
                                net_sent_rate = (
                                    sent_values[i] - sent_values[i - 1]
                                ) / delta_seconds

                                net_recv_rate = (
                                    recv_values[i] - recv_values[i - 1]
                                ) / delta_seconds

                    # Prevent negative rates if counters reset
                    net_sent_rate = max(0.0, net_sent_rate)
                    net_recv_rate = max(0.0, net_recv_rate)

                    # ── Timestamp features ────────────────────────
                    timestamp = get_timestamp(recent[i])

                    if timestamp:
                        hour = timestamp.hour
                        minute = timestamp.minute
                    else:
                        hour = 0
                        minute = 0

                    # These features are not currently stored by the
                    # backend metrics database, so use 0 as fallback.
                    network_latency = 0.0
                    context_switches = 0.0
                    cache_miss_rate = 0.0
                    temperature = 0.0
                    power_consumption = 0.0

                    # ── Complete 27-feature row ────────────────────
                    row = {
                        "cpu_current": cpu_current,
                        "ram_current": ram_current,
                        "disk_current": disk_current,

                        "cpu_lag1": cpu_lag1,
                        "ram_lag1": ram_lag1,

                        "cpu_lag3": cpu_lag3,
                        "ram_lag3": ram_lag3,

                        "cpu_lag5": cpu_lag5,
                        "ram_lag5": ram_lag5,

                        "cpu_ma5": cpu_ma5,
                        "cpu_ma10": cpu_ma10,

                        "ram_ma5": ram_ma5,
                        "ram_ma10": ram_ma10,

                        "cpu_delta": cpu_delta,
                        "ram_delta": ram_delta,
                        "disk_delta": disk_delta,

                        "cpu_std5": cpu_std5,
                        "ram_std5": ram_std5,

                        "net_sent_rate": net_sent_rate,
                        "net_recv_rate": net_recv_rate,

                        "network_latency": network_latency,
                        "context_switches": context_switches,
                        "cache_miss_rate": cache_miss_rate,
                        "temperature": temperature,
                        "power_consumption": power_consumption,

                        "hour": hour,
                        "minute": minute,
                    }

                    # Preserve the exact feature order stored in
                    # LSTM metadata.
                    feature_rows.append(
                        [row.get(f, 0.0) for f in _lstm_feature_names]
                    )

                # Use only the final 10 timesteps for the LSTM
                window_rows = feature_rows[-_lstm_window_size:]

                window = np.array(
                    window_rows,
                    dtype=np.float32
                )

                # Apply the scaler fitted during LSTM training
                if _lstm_scaler:
                    window = _lstm_scaler.transform(window)

                # Shape:
                # (batch_size=1, sequence_length=10, features=27)
                X = torch.FloatTensor(window).unsqueeze(0)

                with torch.no_grad():
                    dl_cpu = float(_lstm_cpu_model(X).item())
                    dl_ram = float(_lstm_ram_model(X).item())

                # Keep predictions within valid percentage range
                dl_cpu = max(0, min(100, dl_cpu))
                dl_ram = max(0, min(100, dl_ram))

                result["dl_prediction"] = {
                    "predicted_cpu_30s": round(dl_cpu, 2),
                    "predicted_ram_30s": round(dl_ram, 2),
                    "model": "lstm",
                }

        except Exception as e:
            result["dl_error"] = str(e)

    return result


@router.get("/predict/reload")
def reload_models():
    """Force-reload all ML and DL models (after retraining)."""
    _try_load_models()
    return {
        "model_loaded": _model_loaded,
        "anomaly_loaded": _anomaly_loaded,
        "lstm_loaded": _lstm_loaded,
    }
