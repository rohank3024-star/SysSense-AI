"""Evaluation helpers for SysSense regression models.

CPU and RAM forecasts are regression outputs. This module deliberately uses
MAE, RMSE, and R2 rather than a classification-style accuracy percentage.
"""
import os

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test, label=""):
    """Evaluate a fitted regressor on a held-out test partition."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n{label} model performance:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2:   {r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def compare_with_baseline(model, X_test, y_test, naive_predictions, label=""):
    """Compare a regressor with a persistence (current-value) baseline."""
    ml_pred = model.predict(X_test)
    ml_mae = mean_absolute_error(y_test, ml_pred)
    naive_mae = mean_absolute_error(y_test, naive_predictions)
    improvement = ((naive_mae - ml_mae) / naive_mae) * 100 if naive_mae else 0.0
    ml_rmse = np.sqrt(mean_squared_error(y_test, ml_pred))
    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_predictions))
    ml_r2 = r2_score(y_test, ml_pred)
    naive_r2 = r2_score(y_test, naive_predictions)

    print(f"\n{'-' * 50}\n{label} - model versus persistence baseline\n{'-' * 50}")
    print(f"  MAE: model {ml_mae:.4f}, baseline {naive_mae:.4f}, lower by {improvement:.1f}%")
    print(f"  RMSE: model {ml_rmse:.4f}, baseline {naive_rmse:.4f}")
    print(f"  R2: model {ml_r2:.4f}, baseline {naive_r2:.4f}")
    return {
        "ml_mae": ml_mae,
        "naive_mae": naive_mae,
        "improvement_percent": round(improvement, 2),
        "ml_rmse": ml_rmse,
        "naive_rmse": naive_rmse,
        "ml_r2": ml_r2,
        "naive_r2": naive_r2,
    }


def generate_report(cpu_metrics, ram_metrics, cpu_comparison, ram_comparison,
                    feature_names, feature_importances):
    """Write the dataset hold-out report with defensible interpretation."""
    report_path = os.path.join(os.path.dirname(__file__), "evaluation_report.md")
    sorted_features = sorted(
        zip(feature_names, feature_importances), key=lambda item: item[1], reverse=True,
    )

    report = f"""# SysSense AI - Random Forest Evaluation

## Evaluation protocol

- Training source: mapped Kaggle IT System Performance & Resource Metrics dataset
- Input schema: {len(feature_names)} collector-compatible features
- Forecast target: six 5-second samples ahead (nominal 30 seconds)
- Split: chronological 80/20 holdout with a six-sample embargo
- Model selection: time-series cross-validation with the same embargo

CPU/RAM forecasting is regression. It must be assessed with MAE, RMSE, and
R2, rather than classification accuracy. The model was trained on a dataset;
the dashboard separately records forecasts and compares them with live
collector readings after the horizon has elapsed.

## CPU prediction (nominal 30-second horizon)

| Metric | Random Forest | Persistence baseline | Difference |
|--------|--------------:|---------------------:|-----------:|
| MAE    | {cpu_metrics['mae']:.4f} | {cpu_comparison['naive_mae']:.4f} | {cpu_comparison['improvement_percent']:.1f}% lower MAE |
| RMSE   | {cpu_metrics['rmse']:.4f} | {cpu_comparison['naive_rmse']:.4f} | - |
| R2     | {cpu_metrics['r2']:.4f} | {cpu_comparison['naive_r2']:.4f} | - |

## RAM prediction (nominal 30-second horizon)

| Metric | Random Forest | Persistence baseline | Difference |
|--------|--------------:|---------------------:|-----------:|
| MAE    | {ram_metrics['mae']:.4f} | {ram_comparison['naive_mae']:.4f} | {ram_comparison['improvement_percent']:.1f}% lower MAE |
| RMSE   | {ram_metrics['rmse']:.4f} | {ram_comparison['naive_rmse']:.4f} | - |
| R2     | {ram_metrics['r2']:.4f} | {ram_comparison['naive_r2']:.4f} | - |

## Top CPU feature importances

| Rank | Feature | Importance |
|------|---------|------------|
"""
    for rank, (name, importance) in enumerate(sorted_features[:10], start=1):
        report += f"| {rank} | {name} | {importance:.4f} |\n"

    report += """
## Interpretation for presentation

The Random Forest has lower held-out MAE than the persistence baseline on this
dataset. Its negative R2 values mean the task remains difficult and its
dataset result should not be described as a guaranteed live-machine accuracy.
Use the dashboard's **Live Forecast Evaluation** table after sufficient
forecasts have matured to report measured local performance.
"""
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(report)
    print(f"\n[OK] Evaluation report saved to {report_path}")
