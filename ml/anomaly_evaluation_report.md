# SysSense AI - Anomaly Evaluation Report

## Model and split

- Model: Isolation Forest (200 trees, `n_jobs=1`, `random_state=42`)
- Input profile: `collector_compatible_anomaly_v1`
- Training rows: 8000; chronological held-out rows: 2000
- Contamination setting: 0.01

## Held-out classification metrics

| Metric | Value |
|---|---:|
| Accuracy | 98.15% |
| Balanced accuracy | 58.88% |
| Precision | 25.00% |
| Recall | 18.52% |
| F1 | 21.28% |
| All-normal baseline accuracy | 98.65% |

Confusion matrix: TN=1958, FP=15, FN=22, TP=5.

Accuracy is included for completeness, but this data is imbalanced. Present
precision, recall, F1, balanced accuracy, and the confusion matrix together;
do not describe accuracy alone as anomaly-detection quality.
