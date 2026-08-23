"""
SysSense AI — Model Training Module

Trains RandomForestRegressor models for CPU and RAM prediction.
Compares against the naive baseline (predicted = current value).
Saves models to ml/models/ for use by the FastAPI backend.

Usage:
    python train.py
"""
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from preprocessing import load_raw_data, engineer_features, prepare_train_test
from evaluation import evaluate_model, compare_with_baseline, generate_report


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def train_model(X_train, y_train, model_name="CPU"):
    """
    Train a RandomForestRegressor with hyperparameter search.
    """
    print(f"\n{'='*50}")
    print(f"Training {model_name} prediction model...")
    print(f"{'='*50}")

    # Hyperparameter grid (kept reasonable for speed)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    # GridSearchCV with 3-fold cross-validation
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring="neg_mean_absolute_error",
        verbose=1, n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"\nBest params for {model_name}: {grid_search.best_params_}")
    print(f"Best CV MAE: {-grid_search.best_score_:.4f}")

    return best_model


def main():
    print("=" * 60)
    print("   SysSense AI — Model Training Pipeline")
    print("=" * 60)

    # ── Step 1: Load data ─────────────────────────────────────────────
    df = load_raw_data()
    if df is None or len(df) < 50:
        print("\n⚠  Not enough data to train a meaningful model.")
        print("   Run collector.py for longer and try again.")
        print(f"   Current rows: {len(df) if df is not None else 0}")
        print("   Recommended minimum: 500+ rows (several hours of data)")
        return

    # ── Step 2: Feature engineering ───────────────────────────────────
    feat = engineer_features(df)
    if len(feat) < 30:
        print("\n⚠  Not enough usable rows after feature engineering.")
        return

    # ── Step 3: Train/test split ──────────────────────────────────────
    (X_train, X_test, y_train_cpu, y_test_cpu,
     y_train_ram, y_test_ram, feature_names) = prepare_train_test(feat)

    # ── Step 4: Train CPU model ───────────────────────────────────────
    cpu_model = train_model(X_train, y_train_cpu, "CPU")

    # ── Step 5: Train RAM model ───────────────────────────────────────
    ram_model = train_model(X_train, y_train_ram, "RAM")

    # ── Step 6: Evaluate ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("   Evaluation Results")
    print("=" * 60)

    cpu_metrics = evaluate_model(cpu_model, X_test, y_test_cpu, "CPU")
    ram_metrics = evaluate_model(ram_model, X_test, y_test_ram, "RAM")

    # Naive baseline comparison
    # Naive = predict current value (first feature column is cpu_current or ram_current)
    naive_cpu = X_test[:, 0]  # cpu_current
    naive_ram = X_test[:, feature_names.index("ram_current")]

    cpu_comparison = compare_with_baseline(
        cpu_model, X_test, y_test_cpu, naive_cpu, "CPU"
    )
    ram_comparison = compare_with_baseline(
        ram_model, X_test, y_test_ram, naive_ram, "RAM"
    )

    # ── Step 7: Feature importance ────────────────────────────────────
    print("\n" + "-" * 40)
    print("CPU Feature Importances (Top 10):")
    print("-" * 40)
    importances = sorted(
        zip(feature_names, cpu_model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    for name, imp in importances[:10]:
        print(f"  {name:20s} {imp:.4f}")

    # ── Step 8: Save models ───────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)

    cpu_path = os.path.join(MODELS_DIR, "cpu_model.pkl")
    ram_path = os.path.join(MODELS_DIR, "ram_model.pkl")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.pkl")

    joblib.dump(cpu_model, cpu_path)
    joblib.dump(ram_model, ram_path)
    joblib.dump({
        "feature_names": feature_names,
        "cpu_metrics": cpu_metrics,
        "ram_metrics": ram_metrics,
        "cpu_comparison": cpu_comparison,
        "ram_comparison": ram_comparison,
    }, meta_path)

    print(f"\n✓ CPU model saved to {cpu_path}")
    print(f"✓ RAM model saved to {ram_path}")
    print(f"✓ Metadata saved to {meta_path}")

    # ── Step 9: Generate report ───────────────────────────────────────
    generate_report(cpu_metrics, ram_metrics, cpu_comparison, ram_comparison,
                    feature_names, cpu_model.feature_importances_)

    print("\n" + "=" * 60)
    print("   Training complete! Models ready for backend integration.")
    print("=" * 60)


if __name__ == "__main__":
    main()
