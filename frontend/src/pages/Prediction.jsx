import { useState, useEffect, useCallback } from 'react';
import LiveChart from '../components/LiveChart';
import { getPrediction, getPredictionEvaluation, getMetricsHistory } from '../services/api';

const MODEL_LABELS = {
  random_forest: 'Random Forest',
  lstm: 'LSTM',
  naive_baseline: 'Naive Baseline',
};

const METRIC_LABELS = {
  cpu: 'CPU',
  ram: 'RAM',
  cpu_percent: 'CPU',
  ram_percent: 'RAM',
};

const isObject = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);

const formatMetric = (value, digits = 2) => {
  if (value === null || value === undefined || value === '') return '—';
  const numericValue = Number(value);
  return Number.isFinite(numericValue) ? numericValue.toFixed(digits) : '—';
};

const formatCount = (value) => {
  if (value === null || value === undefined || value === '') return '—';
  const numericValue = Number(value);
  return Number.isFinite(numericValue) ? numericValue.toLocaleString() : '—';
};

const formatLocalTime = (value) => {
  if (!value) return 'unknown time';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleTimeString();
};

const labelFromKey = (key) => String(key)
  .replace(/_/g, ' ')
  .replace(/\b\w/g, (letter) => letter.toUpperCase());

const getEvaluationRows = (summary) => {
  if (!isObject(summary)) return [];

  const modelGroups = isObject(summary.models) ? summary.models : summary;

  return Object.entries(modelGroups).flatMap(([modelKey, modelMetrics]) => {
    if (!isObject(modelMetrics)) return [];

    const metricGroups = isObject(modelMetrics.metrics)
      ? modelMetrics.metrics
      : modelMetrics;

    return Object.entries(metricGroups).flatMap(([metricKey, metricValues]) => {
      if (!isObject(metricValues)) return [];

      const hasErrorMetric = ['mae', 'rmse', 'r2'].some((key) => metricValues[key] !== undefined);
      if (!hasErrorMetric) return [];

      return [{
        model: MODEL_LABELS[modelKey] || labelFromKey(modelKey),
        metric: METRIC_LABELS[metricKey] || labelFromKey(metricKey),
        mae: metricValues.mae,
        rmse: metricValues.rmse,
        r2: metricValues.r2,
        samples: metricValues.count ?? metricValues.samples ?? metricValues.matched_count,
      }];
    });
  });
};

export default function Prediction() {
  const [prediction, setPrediction] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    try {
      const predRes = await getPrediction();
      setPrediction(predRes.data);
      setLoading(false);
    } catch (err) {
      console.warn('predict failed:', err.message);
      if (loading) setLoading(false);
    }

    try {
      const histRes = await getMetricsHistory(30);
      setHistory(histRes.data);
    } catch (err) {
      console.warn('history failed:', err.message);
    }

    // The endpoint is optional while older backend versions are in use.
    // Its failure must never hide the forecast or live-telemetry sections.
    try {
      const evaluationRes = await getPredictionEvaluation();
      setEvaluation(evaluationRes.data);
    } catch {
      setEvaluation(null);
    }
  }, [loading]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const getPredClass = (current, predicted) => {
    if (!current || !predicted) return 'stable';
    const diff = predicted - current;
    if (diff > 3) return 'up';
    if (diff < -3) return 'down';
    return 'stable';
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <span>Loading predictions...</span>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="loading-container" style={{ flexDirection: 'column', gap: '16px' }}>
        <div style={{ fontSize: '2rem' }}>🤖</div>
        <span style={{ color: '#94a3b8' }}>Could not connect to prediction API</span>
      </div>
    );
  }

  const isMLModel = prediction?.model_used === 'random_forest';
  const mlPred = prediction?.ml_prediction;
  const dlPred = prediction?.dl_prediction;
  const naivePred = prediction?.naive_baseline;
  const current = prediction?.current;

  const cpuPred = isMLModel && mlPred
    ? mlPred.predicted_cpu_30s
    : naivePred?.predicted_cpu_30s;

  const ramPred = isMLModel && mlPred
    ? mlPred.predicted_ram_30s
    : naivePred?.predicted_ram_30s;

  const lstmCpuPred = dlPred?.predicted_cpu_30s;
  const lstmRamPred = dlPred?.predicted_ram_30s;
  const rfEstimateLabel = isMLModel ? 'RF estimate • 30s' : 'Baseline • current value';

  const evaluationSummary = evaluation?.summary ?? evaluation?.models ?? evaluation;
  const evaluationRows = getEvaluationRows(evaluationSummary);
  const matchedCount = evaluation?.matched_count ?? evaluation?.matched ?? evaluation?.validated_count;
  const pendingCount = evaluation?.pending_count ?? evaluation?.pending ?? evaluation?.unmatched_count;
  const hasEvaluation = Boolean(evaluation) && (
    evaluationRows.length > 0 || matchedCount !== undefined || pendingCount !== undefined
  );

  return (
    <div>
      <div className="page-header animate-in">
        <h2>Dataset-Trained Forecasts</h2>
        <p>Experimental CPU and RAM estimates alongside separately labelled live OS telemetry</p>
      </div>

      <div className="card animate-in" style={{ marginBottom: '18px', display: 'flex', alignItems: 'center', gap: '12px' }}>
        <span className={`model-badge ${isMLModel ? 'ml' : 'naive'}`}>
          {isMLModel ? '🤖 Dataset-Trained RF Active' : '📊 Current-Value Baseline'}
        </span>
        <span style={{ fontSize: '0.82rem', color: '#94a3b8' }}>
          {isMLModel
            ? 'Random Forest and LSTM were trained on an external dataset. Their 30-second values are estimates, not a live accuracy claim; compare completed forecasts with later live readings below.'
            : 'The trained model is unavailable, so the dashboard is showing the current-value baseline (forecast = current reading).'}
          {prediction?.forecast_basis?.source_timestamp && (
            <><br />Forecast features use the latest collector sample at {formatLocalTime(prediction.forecast_basis.source_timestamp)}.</>
          )}
        </span>
      </div>

      <div className="prediction-grid animate-in animate-in-delay-1">

        {/* CPU Prediction */}
        <div className="card prediction-card">
          <div className="prediction-label">CPU Forecast (Experimental)</div>

          <div className="prediction-values">
            <div>
              <div className="prediction-current">
                {current?.cpu?.toFixed(1) || '—'}%
              </div>
              <div style={{ fontSize: '0.7rem', color: '#64748b' }}>
                Current
              </div>
            </div>

            <div className="prediction-arrow">→</div>

            <div>
              <div className={`prediction-future ${getPredClass(current?.cpu, cpuPred)}`}>
                {cpuPred?.toFixed(1) || '—'}%
              </div>
              <div style={{ fontSize: '0.7rem', color: '#64748b' }}>
                {rfEstimateLabel}
              </div>
            </div>
          </div>

          <div className="progress-bar" style={{ marginTop: '16px' }}>
            <div
              className="progress-bar-fill"
              style={{
                width: `${cpuPred || 0}%`,
                background: 'var(--gradient-cpu)'
              }}
            />
          </div>

          <div style={{
            marginTop: '12px',
            paddingTop: '10px',
            borderTop: '1px solid rgba(255,255,255,0.05)',
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: '0.75rem'
          }}>
            <span style={{ color: '#94a3b8' }}>
              🧠 LSTM estimate • 30s
            </span>
            <span style={{ color: '#06b6d4', fontWeight: 600 }}>
              {lstmCpuPred?.toFixed(1) || '—'}%
            </span>
          </div>
        </div>


        {/* RAM Prediction */}
        <div className="card prediction-card">
          <div className="prediction-label">RAM Forecast (Experimental)</div>

          <div className="prediction-values">
            <div>
              <div className="prediction-current">
                {current?.ram?.toFixed(1) || '—'}%
              </div>
              <div style={{ fontSize: '0.7rem', color: '#64748b' }}>
                Current
              </div>
            </div>

            <div className="prediction-arrow">→</div>

            <div>
              <div className={`prediction-future ${getPredClass(current?.ram, ramPred)}`}>
                {ramPred?.toFixed(1) || '—'}%
              </div>
              <div style={{ fontSize: '0.7rem', color: '#64748b' }}>
                {rfEstimateLabel}
              </div>
            </div>
          </div>

          <div className="progress-bar" style={{ marginTop: '16px' }}>
            <div
              className="progress-bar-fill"
              style={{
                width: `${ramPred || 0}%`,
                background: 'var(--gradient-ram)'
              }}
            />
          </div>

          <div style={{
            marginTop: '12px',
            paddingTop: '10px',
            borderTop: '1px solid rgba(255,255,255,0.05)',
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: '0.75rem'
          }}>
            <span style={{ color: '#94a3b8' }}>
              🧠 LSTM estimate • 30s
            </span>
            <span style={{ color: '#8b5cf6', fontWeight: 600 }}>
              {lstmRamPred?.toFixed(1) || '—'}%
            </span>
          </div>
        </div>


        {/* Disk is intentionally live telemetry, not a forecast. */}
        <div className="card prediction-card">
          <div className="prediction-label">Disk Usage (Live)</div>

          <div className="prediction-values" style={{ justifyContent: 'center' }}>
            <div>
              <div className="prediction-current">
                {current?.disk?.toFixed(1) || '—'}%
              </div>
              <div style={{ fontSize: '0.7rem', color: '#64748b' }}>
                Current OS telemetry
              </div>
            </div>
          </div>

          <div className="progress-bar" style={{ marginTop: '16px' }}>
            <div
              className="progress-bar-fill"
              style={{
                width: `${current?.disk || 0}%`,
                background: 'var(--gradient-disk)'
              }}
            />
          </div>
          <p style={{ fontSize: '0.72rem', color: '#64748b', margin: '12px 0 0' }}>
            Disk is shown as current usage only; it is not forecast by the ML or DL models.
          </p>
        </div>

      </div>

      {hasEvaluation && (
        <div className="card animate-in animate-in-delay-2" style={{ marginBottom: '18px' }}>
          <div className="chart-header">
            <span className="chart-title">Live Forecast Evaluation</span>
            <span style={{ fontSize: '0.72rem', color: '#64748b' }}>
              {matchedCount !== undefined && <span>{formatCount(matchedCount)} matched</span>}
              {matchedCount !== undefined && pendingCount !== undefined && <span> · </span>}
              {pendingCount !== undefined && <span>{formatCount(pendingCount)} awaiting observation</span>}
            </span>
          </div>
          <p style={{ fontSize: '0.82rem', color: '#94a3b8', marginBottom: '16px' }}>
            Each completed forecast is compared with the live value observed 30 seconds later. Lower MAE and RMSE are better; R² closer to 1 is better. These are evaluation metrics, not classification accuracy percentages.
          </p>
          {evaluationRows.length > 0 ? (
            <table className="process-table">
              <thead>
                <tr><th>Model</th><th>Metric</th><th>MAE</th><th>RMSE</th><th>R²</th><th>Samples</th></tr>
              </thead>
              <tbody>
                {evaluationRows.map((row) => (
                  <tr key={`${row.model}-${row.metric}`}>
                    <td className="process-name">{row.model}</td>
                    <td>{row.metric}</td>
                    <td>{formatMetric(row.mae)}</td>
                    <td>{formatMetric(row.rmse)}</td>
                    <td>{formatMetric(row.r2, 3)}</td>
                    <td>{formatCount(row.samples ?? matchedCount)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="empty-state" style={{ padding: '14px 0 2px' }}>
              <p>Forecast matching is collecting observations. Error metrics will appear after forecasts have been matched with later live readings.</p>
            </div>
          )}
        </div>
      )}

      {prediction?.feature_importances && (
        <div className="card animate-in animate-in-delay-2" style={{ marginBottom: '18px' }}>
          <div className="chart-header">
            <span className="chart-title">🧠 AI Insight — Feature Importances</span>
          </div>
          <p style={{ fontSize: '0.82rem', color: '#94a3b8', marginBottom: '16px' }}>
            What the ML model considers most important when predicting future usage:
          </p>
          {Object.entries(prediction.feature_importances)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 8)
            .map(([name, importance]) => (
              <div key={name} style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                <span style={{ width: '120px', fontSize: '0.78rem', color: '#94a3b8' }}>{name}</span>
                <div style={{ flex: 1, height: '6px', background: 'rgba(255,255,255,0.05)', borderRadius: '3px', overflow: 'hidden' }}>
                  <div style={{ width: `${importance * 100}%`, height: '100%', background: 'var(--gradient-health)', borderRadius: '3px', transition: 'width 0.5s ease' }} />
                </div>
                <span style={{ fontSize: '0.75rem', color: '#64748b', width: '50px', textAlign: 'right' }}>{(importance * 100).toFixed(1)}%</span>
              </div>
            ))
          }
        </div>
      )}

      {/* ── Anomaly Detection Status ──────────────────────────── */}
      {prediction?.anomaly && (
        <div className="card animate-in animate-in-delay-2" style={{ marginBottom: '18px' }}>
          <div className="chart-header">
            <span className="chart-title">🛡️ Anomaly Detection</span>
          </div>
          <div className={`alert-card ${prediction.anomaly.available === false ? 'info' : prediction.anomaly.is_anomaly ? (prediction.anomaly.severity === 'critical' ? 'critical' : 'warning') : 'info'}`}>
            <span className="alert-icon">{prediction.anomaly.available === false ? 'ℹ️' : prediction.anomaly.is_anomaly ? '🔴' : '🟢'}</span>
            <div className="alert-content">
              <h4>{prediction.anomaly.available === false ? 'ML anomaly model unavailable' : prediction.anomaly.is_anomaly ? `Anomaly Detected (${prediction.anomaly.severity})` : 'No anomaly signal'}</h4>
              <p>{prediction.anomaly.available === false
                ? prediction.anomaly.reason || 'The live feature contract cannot be evaluated by this model.'
                : prediction.anomaly.is_anomaly
                  ? prediction.anomaly.details?.join('; ')
                  : 'The trained model did not flag this metric combination. This is not proof that the system is healthy.'}
              </p>
              {prediction.anomaly.available !== false && (
                <p style={{ fontSize: '0.72rem', color: '#64748b', marginTop: '4px' }}>
                  Anomaly Score: {prediction.anomaly.anomaly_score}
                </p>
              )}
            </div>
          </div>
          {prediction.process_anomalies?.length > 0 && (
            <div style={{ marginTop: '12px' }}>
              <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginBottom: '8px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Process Threshold Signals
              </div>
              {prediction.process_anomalies.slice(0, 5).map((pa, i) => (
                <div key={i} className={`alert-card ${pa.severity}`}>
                  <span className="alert-icon">{pa.severity === 'critical' ? '🔴' : '🟡'}</span>
                  <div className="alert-content">
                    <h4>{pa.message}</h4>
                    <p>PID: {pa.pid}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="card animate-in animate-in-delay-3">
        <LiveChart data={history} title="Recent Trends (used for prediction)" height={280} />
      </div>

      {isMLModel && naivePred && mlPred && (
        <div className="card animate-in animate-in-delay-4" style={{ marginTop: '18px' }}>
          <div className="chart-header">
            <span className="chart-title">Model Comparison — Naive vs ML</span>
          </div>
          <table className="process-table">
            <thead>
              <tr><th>Metric</th><th>Current</th><th>Naive Baseline</th><th>ML Prediction</th><th>Difference</th></tr>
            </thead>
            <tbody>
              <tr>
                <td className="process-name">CPU</td>
                <td>{current?.cpu?.toFixed(1)}%</td>
                <td>{naivePred.predicted_cpu_30s?.toFixed(1)}%</td>
                <td style={{ color: '#8b5cf6', fontWeight: 600 }}>{mlPred.predicted_cpu_30s?.toFixed(1)}%</td>
                <td style={{ color: '#06b6d4' }}>{Math.abs(mlPred.predicted_cpu_30s - naivePred.predicted_cpu_30s).toFixed(1)}%</td>
              </tr>
              <tr>
                <td className="process-name">RAM</td>
                <td>{current?.ram?.toFixed(1)}%</td>
                <td>{naivePred.predicted_ram_30s?.toFixed(1)}%</td>
                <td style={{ color: '#8b5cf6', fontWeight: 600 }}>{mlPred.predicted_ram_30s?.toFixed(1)}%</td>
                <td style={{ color: '#06b6d4' }}>{Math.abs(mlPred.predicted_ram_30s - naivePred.predicted_ram_30s).toFixed(1)}%</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
