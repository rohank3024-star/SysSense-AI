"""
SysSense AI — Model Evaluation Module

Computes MAE, RMSE, R² score and generates comparison
reports between ML predictions and the naive baseline.
"""
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test, label=""):
    """
    Evaluate a trained model on the test set.

    Returns dict with MAE, RMSE, R².
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n{label} Model Performance:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")

    return {"mae": mae, "rmse": rmse, "r2": r2}


def compare_with_baseline(model, X_test, y_test, naive_predictions, label=""):
    """
    Compare ML model against naive baseline (predict current = future).

    This comparison is THE key result for your report.
    """
    ml_pred = model.predict(X_test)

    ml_mae = mean_absolute_error(y_test, ml_pred)
    naive_mae = mean_absolute_error(y_test, naive_predictions)
    improvement = ((naive_mae - ml_mae) / naive_mae) * 100 if naive_mae > 0 else 0

    ml_rmse = np.sqrt(mean_squared_error(y_test, ml_pred))
    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_predictions))

    ml_r2 = r2_score(y_test, ml_pred)
    naive_r2 = r2_score(y_test, naive_predictions)

    print(f"\n{'-' * 50}")
    print(f"{label} -- ML vs Naive Baseline Comparison")
    print(f"{'-' * 50}")
    print(f"  {'Metric':<10} {'ML Model':>12} {'Naive':>12} {'Improvement':>14}")
    print(f"  {'-'*48}")
    print(f"  {'MAE':<10} {ml_mae:>12.4f} {naive_mae:>12.4f} {improvement:>13.1f}%")
    print(f"  {'RMSE':<10} {ml_rmse:>12.4f} {naive_rmse:>12.4f}")
    print(f"  {'R2':<10} {ml_r2:>12.4f} {naive_r2:>12.4f}")

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
    """
    Generate a markdown evaluation report.
    Save to ml/evaluation_report.md
    """
    report_path = os.path.join(os.path.dirname(__file__), "evaluation_report.md")

    # Sort feature importances
    sorted_features = sorted(
        zip(feature_names, feature_importances),
        key=lambda x: x[1], reverse=True,
    )

    report = f"""# SysSense AI — ML Evaluation Report

## Model: Random Forest Regressor

### CPU Prediction (30s ahead)

| Metric | ML Model | Naive Baseline | Improvement |
|--------|----------|----------------|-------------|
| MAE    | {cpu_metrics['mae']:.4f} | {cpu_comparison['naive_mae']:.4f} | {cpu_comparison['improvement_percent']:.1f}% |
| RMSE   | {cpu_metrics['rmse']:.4f} | {cpu_comparison['naive_rmse']:.4f} | — |
| R²     | {cpu_metrics['r2']:.4f} | {cpu_comparison['naive_r2']:.4f} | — |

### RAM Prediction (30s ahead)

| Metric | ML Model | Naive Baseline | Improvement |
|--------|----------|----------------|-------------|
| MAE    | {ram_metrics['mae']:.4f} | {ram_comparison['naive_mae']:.4f} | {ram_comparison['improvement_percent']:.1f}% |
| RMSE   | {ram_metrics['rmse']:.4f} | {ram_comparison['naive_rmse']:.4f} | — |
| R²     | {ram_metrics['r2']:.4f} | {ram_comparison['naive_r2']:.4f} | — |

### Top Feature Importances (CPU Model)

| Rank | Feature | Importance |
|------|---------|------------|
"""
    for i, (name, imp) in enumerate(sorted_features[:10], 1):
        report += f"| {i} | {name} | {imp:.4f} |\n"

    report += """
### Key Takeaways

1. The ML model should outperform the naive baseline, proving that historical patterns contain predictive signal.
2. Feature importances reveal which metrics most influence future CPU/RAM load.
3. Moving averages and lag features capture momentum and trends that raw current values miss.

### Resume Bullet

> Trained a Random Forest model to forecast CPU/RAM usage 30s ahead, outperforming a naive baseline by {cpu_improvement}% (CPU) and {ram_improvement}% (RAM) in MAE.
""".format(
        cpu_improvement=cpu_comparison['improvement_percent'],
        ram_improvement=ram_comparison['improvement_percent'],
    )

    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n[OK] Evaluation report saved to {report_path}")
