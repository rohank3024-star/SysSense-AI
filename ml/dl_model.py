"""LSTM forecasting trained on the project dataset.

The LSTM uses the same collector-compatible feature contract as the Random
Forest. Training, validation, and final testing are chronological and kept
separate, so early stopping never uses the final test partition.
"""
import copy
import os
import sys

import joblib
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[!] PyTorch not installed. Install with: pip install torch")

try:
    from .preprocessing import (
        COLLECTOR_FEATURE_NAMES,
        FORECAST_HORIZON_SAMPLES,
        engineer_features,
        get_feature_contract,
        load_data,
    )
except ImportError:
    from preprocessing import (
        COLLECTOR_FEATURE_NAMES,
        FORECAST_HORIZON_SAMPLES,
        engineer_features,
        get_feature_contract,
        load_data,
    )


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "dl_evaluation_report.md")

WINDOW_SIZE = 10
HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 64
DENSE_SIZE = 32
DROPOUT = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 40
PATIENCE = 6
RANDOM_SEED = 42


class TimeSeriesDataset(Dataset):
    """Windows where the label belongs to the final observed row.

    Each feature row already has a target six samples in the future. Therefore
    using the target of the final row preserves the documented 30-second
    horizon instead of accidentally shifting it one more sample ahead.
    """
    def __init__(self, features, labels, window_size=WINDOW_SIZE):
        if len(features) < window_size:
            raise ValueError("Not enough feature rows for an LSTM window")
        self.X = []
        self.y = []
        for end in range(window_size - 1, len(features)):
            self.X.append(features[end - window_size + 1:end + 1])
            self.y.append(labels[end])
        self.X = torch.tensor(np.asarray(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class LSTMPredictor(nn.Module):
    """Two-layer stacked LSTM for CPU or RAM regression."""
    def __init__(self, input_size, hidden1=HIDDEN_SIZE_1, hidden2=HIDDEN_SIZE_2,
                 dense_size=DENSE_SIZE, dropout=DROPOUT):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=hidden1, hidden_size=hidden2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden2, dense_size)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_size, 1)

    def forward(self, inputs):
        output, _ = self.lstm1(inputs)
        output = self.dropout1(output)
        output, _ = self.lstm2(output)
        output = self.dropout2(output[:, -1, :])
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout3(output)
        return self.fc2(output).squeeze(-1)


def train_lstm(train_loader, validation_loader, input_size, target_name):
    """Train with early stopping based only on the validation partition."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMPredictor(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    print(f"\nTraining LSTM for {target_name} on {device}...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(inputs)
        train_loss /= len(train_loader.dataset)

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                validation_loss += criterion(model(inputs), targets).item() * len(inputs)
        validation_loss /= len(validation_loader.dataset)
        scheduler.step(validation_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1:02d}/{EPOCHS} | "
                f"train MSE {train_loss:.4f} | validation MSE {validation_loss:.4f}"
            )

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch + 1}; best validation MSE {best_loss:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_lstm(model, loader, device):
    """Return final-test predictions and labels without updating the model."""
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            predictions.extend(model(inputs.to(device)).cpu().numpy())
            actuals.extend(targets.numpy())
    return np.asarray(predictions), np.asarray(actuals)


def compute_metrics(actuals, predictions):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = float(mean_absolute_error(actuals, predictions))
    rmse = float(np.sqrt(mean_squared_error(actuals, predictions)))
    r2 = float(r2_score(actuals, predictions))
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


def _format_metrics(metrics):
    return f"MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}"


def generate_dl_report(cpu_lstm, cpu_rf, cpu_naive, ram_lstm, ram_rf, ram_naive):
    """Write a transparent comparison report without a regression accuracy claim."""
    report = f"""# SysSense AI - Deep Learning Evaluation Report

## Evaluation protocol

- Dataset: Kaggle IT System Performance & Resource Metrics (mapped project CSV)
- Inputs: collector-compatible feature profile, {len(COLLECTOR_FEATURE_NAMES)} features
- Split: chronological 70% training, 10% validation, 20% final test, with a
  {FORECAST_HORIZON_SAMPLES}-sample embargo at split boundaries
- LSTM input: {WINDOW_SIZE} consecutive collector-compatible rows
- Early stopping: validation partition only; final test was not used for training

CPU/RAM forecasting is regression, so MAE, RMSE, and R2 are reported rather
than a classification-style accuracy percentage. These benchmark metrics do
not by themselves prove live-machine forecast quality; use the dashboard's
live forecast evaluation for that separate evidence.

## CPU prediction (nominal 30-second horizon)

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| LSTM | {cpu_lstm['mae']:.4f} | {cpu_lstm['rmse']:.4f} | {cpu_lstm['r2']:.4f} |
| Random Forest* | {cpu_rf['mae']:.4f} | {cpu_rf['rmse']:.4f} | {cpu_rf['r2']:.4f} |
| Persistence baseline | {cpu_naive['mae']:.4f} | {cpu_naive['rmse']:.4f} | {cpu_naive['r2']:.4f} |

## RAM prediction (nominal 30-second horizon)

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| LSTM | {ram_lstm['mae']:.4f} | {ram_lstm['rmse']:.4f} | {ram_lstm['r2']:.4f} |
| Random Forest* | {ram_rf['mae']:.4f} | {ram_rf['rmse']:.4f} | {ram_rf['r2']:.4f} |
| Persistence baseline | {ram_naive['mae']:.4f} | {ram_naive['rmse']:.4f} | {ram_naive['r2']:.4f} |

\*Random Forest values come from its separately saved chronological evaluation;
the LSTM final-test window removes its first nine rows, so this is not a
strictly identical-row head-to-head comparison.
"""
    with open(REPORT_PATH, "w", encoding="utf-8") as report_file:
        report_file.write(report)
    print(f"[OK] DL evaluation report saved to {REPORT_PATH}")


def _make_loaders(features, labels_cpu, labels_ram):
    """Build chronological train/validation/test loaders with embargoes."""
    from sklearn.preprocessing import StandardScaler

    total_rows = len(features)
    validation_start = int(total_rows * 0.7)
    test_start = int(total_rows * 0.8)
    train_end = validation_start - FORECAST_HORIZON_SAMPLES
    validation_end = test_start - FORECAST_HORIZON_SAMPLES
    if min(train_end, validation_end - validation_start, total_rows - test_start) < WINDOW_SIZE:
        raise ValueError("Not enough rows for chronological LSTM train/validation/test splits")

    scaler = StandardScaler()
    train_features = scaler.fit_transform(features[:train_end])
    validation_features = scaler.transform(features[validation_start:validation_end])
    test_features = scaler.transform(features[test_start:])

    partitions = {
        "train": (train_features, labels_cpu[:train_end], labels_ram[:train_end]),
        "validation": (
            validation_features,
            labels_cpu[validation_start:validation_end],
            labels_ram[validation_start:validation_end],
        ),
        "test": (test_features, labels_cpu[test_start:], labels_ram[test_start:]),
    }
    datasets = {}
    for name, (partition_features, cpu_labels, ram_labels) in partitions.items():
        datasets[name] = {
            "cpu": TimeSeriesDataset(partition_features, cpu_labels),
            "ram": TimeSeriesDataset(partition_features, ram_labels),
        }

    loaders = {
        name: {
            target: DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=(name == "train"),
            )
            for target, dataset in targets.items()
        }
        for name, targets in datasets.items()
    }
    return loaders, datasets, scaler, {
        "train_rows": train_end,
        "validation_rows": validation_end - validation_start,
        "test_rows": total_rows - test_start,
        "embargo_rows": FORECAST_HORIZON_SAMPLES,
    }


def _load_rf_metrics():
    defaults = {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.pkl")
    if not os.path.exists(metadata_path):
        return defaults, defaults.copy()
    metadata = joblib.load(metadata_path)
    return (
        metadata.get("cpu_metrics", defaults),
        metadata.get("ram_metrics", defaults),
    )


def main():
    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch is required. Install it with: pip install torch")
        return

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print("=" * 60)
    print("SysSense AI - LSTM Training (dataset source, live-compatible inputs)")
    print("=" * 60)

    frame = load_data(prefer_public=True)
    if frame is None or len(frame) < 100:
        print("[ERROR] Not enough dataset rows for LSTM training.")
        return
    feature_frame = engineer_features(frame)
    features = feature_frame[COLLECTOR_FEATURE_NAMES].values
    labels_cpu = feature_frame["label_cpu_30s"].values
    labels_ram = feature_frame["label_ram_30s"].values

    loaders, datasets, scaler, split_info = _make_loaders(features, labels_cpu, labels_ram)
    print(
        f"Train {split_info['train_rows']} | validation {split_info['validation_rows']} | "
        f"test {split_info['test_rows']} | embargo {split_info['embargo_rows']}"
    )

    input_size = len(COLLECTOR_FEATURE_NAMES)
    cpu_model = train_lstm(loaders["train"]["cpu"], loaders["validation"]["cpu"], input_size, "CPU")
    ram_model = train_lstm(loaders["train"]["ram"], loaders["validation"]["ram"], input_size, "RAM")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cpu_predictions, cpu_actuals = evaluate_lstm(cpu_model, loaders["test"]["cpu"], device)
    ram_predictions, ram_actuals = evaluate_lstm(ram_model, loaders["test"]["ram"], device)
    cpu_metrics = compute_metrics(cpu_actuals, cpu_predictions)
    ram_metrics = compute_metrics(ram_actuals, ram_predictions)

    cpu_index = COLLECTOR_FEATURE_NAMES.index("cpu_current")
    ram_index = COLLECTOR_FEATURE_NAMES.index("ram_current")
    cpu_naive_scaled = datasets["test"]["cpu"].X[:, -1, cpu_index].numpy()
    ram_naive_scaled = datasets["test"]["ram"].X[:, -1, ram_index].numpy()
    cpu_naive = cpu_naive_scaled * scaler.scale_[cpu_index] + scaler.mean_[cpu_index]
    ram_naive = ram_naive_scaled * scaler.scale_[ram_index] + scaler.mean_[ram_index]
    cpu_naive_metrics = compute_metrics(cpu_actuals, cpu_naive)
    ram_naive_metrics = compute_metrics(ram_actuals, ram_naive)

    print("\nFinal held-out metrics:")
    print(f"  LSTM CPU: {_format_metrics(cpu_metrics)}")
    print(f"  LSTM RAM: {_format_metrics(ram_metrics)}")
    print(f"  Persistence CPU: {_format_metrics(cpu_naive_metrics)}")
    print(f"  Persistence RAM: {_format_metrics(ram_naive_metrics)}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    cpu_path = os.path.join(MODELS_DIR, "lstm_cpu_model.pth")
    ram_path = os.path.join(MODELS_DIR, "lstm_ram_model.pth")
    scaler_path = os.path.join(MODELS_DIR, "lstm_scaler.pkl")
    metadata_path = os.path.join(MODELS_DIR, "lstm_metadata.pkl")
    checkpoint = {
        "input_size": input_size,
        "hidden1": HIDDEN_SIZE_1,
        "hidden2": HIDDEN_SIZE_2,
        "dense_size": DENSE_SIZE,
        "dropout": DROPOUT,
    }
    torch.save({"model_state_dict": cpu_model.state_dict(), **checkpoint}, cpu_path)
    torch.save({"model_state_dict": ram_model.state_dict(), **checkpoint}, ram_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump({
        "feature_names": list(COLLECTOR_FEATURE_NAMES),
        "feature_profile": get_feature_contract()["profile"],
        "feature_contract": get_feature_contract(),
        "window_size": WINDOW_SIZE,
        "input_size": input_size,
        "cpu_metrics": cpu_metrics,
        "ram_metrics": ram_metrics,
        "naive_cpu_metrics": cpu_naive_metrics,
        "naive_ram_metrics": ram_naive_metrics,
        "split_info": split_info,
        "split_strategy": "chronological_70_10_20_with_6_sample_embargo",
        "trained_on": "kaggle_it_system_performance",
    }, metadata_path)

    rf_cpu, rf_ram = _load_rf_metrics()
    generate_dl_report(
        cpu_metrics, rf_cpu, cpu_naive_metrics,
        ram_metrics, rf_ram, ram_naive_metrics,
    )
    print("[OK] LSTM models, scaler, metadata, and report saved.")


if __name__ == "__main__":
    main()
