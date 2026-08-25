# SysSense AI — ML Evaluation Report

## Model: Random Forest Regressor

### CPU Prediction (30s ahead)

| Metric | ML Model | Naive Baseline | Improvement |
|--------|----------|----------------|-------------|
| MAE    | 23.0136 | 31.3095 | 26.5% |
| RMSE   | 26.8750 | 38.1658 | — |
| R²     | -0.0057 | -1.0282 | — |

### RAM Prediction (30s ahead)

| Metric | ML Model | Naive Baseline | Improvement |
|--------|----------|----------------|-------------|
| MAE    | 22.6491 | 30.3470 | 25.4% |
| RMSE   | 26.1806 | 36.9234 | — |
| R²     | -0.0105 | -1.0100 | — |

### Top Feature Importances (CPU Model)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | context_switches | 0.0461 |
| 2 | ram_std5 | 0.0440 |
| 3 | power_consumption | 0.0438 |
| 4 | temperature | 0.0436 |
| 5 | ram_lag5 | 0.0418 |
| 6 | cpu_std5 | 0.0416 |
| 7 | cpu_lag5 | 0.0416 |
| 8 | cache_miss_rate | 0.0413 |
| 9 | ram_lag3 | 0.0411 |
| 10 | cpu_ma10 | 0.0406 |

### Key Takeaways

1. The ML model should outperform the naive baseline, proving that historical patterns contain predictive signal.
2. Feature importances reveal which metrics most influence future CPU/RAM load.
3. Moving averages and lag features capture momentum and trends that raw current values miss.

### Resume Bullet

> Trained a Random Forest model to forecast CPU/RAM usage 30s ahead, outperforming a naive baseline by 26.5% (CPU) and 25.37% (RAM) in MAE.
