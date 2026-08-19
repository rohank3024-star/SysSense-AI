import { useState, useEffect, useCallback } from 'react';
import LiveChart from '../components/LiveChart';
import { getPrediction, getMetricsHistory } from '../services/api';

export default function Prediction() {
  const [prediction, setPrediction] = useState(null);
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
  const naivePred = prediction?.naive_baseline;
  const current = prediction?.current;
  const cpuPred = isMLModel && mlPred ? mlPred.predicted_cpu_30s : naivePred?.predicted_cpu_30s;
  const ramPred = isMLModel && mlPred ? mlPred.predicted_ram_30s : naivePred?.predicted_ram_30s;

  return (
    <div>
      <div className="page-header animate-in">
        <h2>AI Prediction</h2>
        <p>Machine learning-powered resource usage forecasting — 30 seconds ahead</p>
      </div>

      <div className="card animate-in" style={{ marginBottom: '18px', display: 'flex', alignItems: 'center', gap: '12px' }}>
        <span className={`model-badge ${isMLModel ? 'ml' : 'naive'}`}>
          {isMLModel ? '🤖 ML Model Active' : '📊 Naive Baseline'}
        </span>
        <span style={{ fontSize: '0.82rem', color: '#94a3b8' }}>
          {isMLModel
            ? 'Random Forest Regressor is generating predictions based on historical patterns'
            : 'ML model not yet trained — using naive baseline (predicted = current). Train the model once you have enough data.'}
        </span>
      </div>

      <div className="prediction-grid animate-in animate-in-delay-1">
        <div className="card prediction-card">
          <div className="prediction-label">CPU Usage Prediction</div>
          <div className="prediction-values">
            <div>
              <div className="prediction-current">{current?.cpu?.toFixed(1) || '—'}%</div>
              <div style={{ fontSize: '0.7rem', color: '#64748b' }}>Current</div>
            </div>
            <div className="prediction-arrow">→</div>
            <div>
              <div className={`prediction-future ${getPredClass(current?.cpu, cpuPred)}`}>{cpuPred?.toFixed(1) || '—'}%</div>
              <div style={{ fontSize: '0.7rem', color: '#64748b' }}>In 30s</div>
            </div>
          </div>
          <div className="progress-bar" style={{ marginTop: '16px' }}>
            <div className="progress-bar-fill" style={{ width: `${cpuPred || 0}%`, background: 'var(--gradient-cpu)' }} />
          </div>
        </div>

        <div className="card prediction-card">
          <div className="prediction-label">RAM Usage Prediction</div>
          <div className="prediction-values">
            <div>
              <div className="prediction-current">{current?.ram?.toFixed(1) || '—'}%</div>
              <div style={{ fontSize: '0.7rem', color: '#64748b' }}>Current</div>
            </div>
            <div className="prediction-arrow">→</div>
            <div>
              <div className={`prediction-future ${getPredClass(current?.ram, ramPred)}`}>{ramPred?.toFixed(1) || '—'}%</div>
              <div style={{ fontSize: '0.7rem', color: '#64748b' }}>In 30s</div>
            </div>
          </div>
          <div className="progress-bar" style={{ marginTop: '16px' }}>
            <div className="progress-bar-fill" style={{ width: `${ramPred || 0}%`, background: 'var(--gradient-ram)' }} />
          </div>
        </div>

        <div className="card prediction-card">
          <div className="prediction-label">Disk Usage</div>
          <div className="prediction-values">
            <div>
              <div className="prediction-current">{current?.disk?.toFixed(1) || '—'}%</div>
              <div style={{ fontSize: '0.7rem', color: '#64748b' }}>Current</div>
            </div>
            <div className="prediction-arrow">→</div>
            <div>
              <div className="prediction-future stable">{current?.disk?.toFixed(1) || '—'}%</div>
              <div style={{ fontSize: '0.7rem', color: '#64748b' }}>Stable</div>
            </div>
          </div>
          <div className="progress-bar" style={{ marginTop: '16px' }}>
            <div className="progress-bar-fill" style={{ width: `${current?.disk || 0}%`, background: 'var(--gradient-disk)' }} />
          </div>
        </div>
      </div>

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
