# SysSense AI - Deep Learning Evaluation Report

## Evaluation protocol

- Dataset: Kaggle IT System Performance & Resource Metrics (mapped project CSV)
- Inputs: collector-compatible feature profile, 24 features
- Split: chronological 70% training, 10% validation, 20% final test, with a
  6-sample embargo at split boundaries
- LSTM input: 10 consecutive collector-compatible rows
- Early stopping: validation partition only; final test was not used for training

CPU/RAM forecasting is regression, so MAE, RMSE, and R2 are reported rather
than a classification-style accuracy percentage. These benchmark metrics do
not by themselves prove live-machine forecast quality; use the dashboard's
live forecast evaluation for that separate evidence.

## CPU prediction (nominal 30-second horizon)

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| LSTM | 22.9939 | 26.8115 | -0.0008 |
| Random Forest* | 23.0233 | 26.8685 | -0.0052 |
| Persistence baseline | 31.2902 | 38.1449 | -1.0258 |

## RAM prediction (nominal 30-second horizon)

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| LSTM | 22.5530 | 26.0690 | -0.0022 |
| Random Forest* | 22.6682 | 26.2040 | -0.0123 |
| Persistence baseline | 30.3418 | 36.9204 | -1.0103 |

\*Random Forest values come from its separately saved chronological evaluation;
the LSTM final-test window removes its first nine rows, so this is not a
strictly identical-row head-to-head comparison.
