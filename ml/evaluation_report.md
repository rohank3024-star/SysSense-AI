# SysSense AI - Random Forest Evaluation

## Evaluation protocol

- Training source: mapped Kaggle IT System Performance & Resource Metrics dataset
- Input schema: 24 collector-compatible features
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
| MAE    | 23.0233 | 31.3095 | 26.5% lower MAE |
| RMSE   | 26.8685 | 38.1658 | - |
| R2     | -0.0052 | -1.0282 | - |

## RAM prediction (nominal 30-second horizon)

| Metric | Random Forest | Persistence baseline | Difference |
|--------|--------------:|---------------------:|-----------:|
| MAE    | 22.6682 | 30.3470 | 25.3% lower MAE |
| RMSE   | 26.2040 | 36.9234 | - |
| R2     | -0.0123 | -1.0100 | - |

## Top CPU feature importances

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | ram_std5 | 0.0520 |
| 2 | ram_lag5 | 0.0485 |
| 3 | disk_delta | 0.0476 |
| 4 | cpu_lag5 | 0.0472 |
| 5 | ram_lag3 | 0.0461 |
| 6 | cpu_ma10 | 0.0457 |
| 7 | ram_ma5 | 0.0448 |
| 8 | cpu_std5 | 0.0447 |
| 9 | process_count | 0.0446 |
| 10 | cpu_lag1 | 0.0439 |

## Interpretation for presentation

The Random Forest has lower held-out MAE than the persistence baseline on this
dataset. Its negative R2 values mean the task remains difficult and its
dataset result should not be described as a guaranteed live-machine accuracy.
Use the dashboard's **Live Forecast Evaluation** table after sufficient
forecasts have matured to report measured local performance.
