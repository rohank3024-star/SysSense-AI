import { useState, useEffect, useCallback } from 'react';
import { getAlerts, getAlertHistory, getRecommendations, getPrediction } from '../services/api';

export default function Alerts() {
  const [alerts, setAlerts] = useState({ alerts: [], current: {} });
  const [history, setHistory] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [anomaly, setAnomaly] = useState(null);
  const [processAnomalies, setProcessAnomalies] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    try {
      const alertsRes = await getAlerts();
      setAlerts(alertsRes.data);
      setLoading(false);
    } catch (err) {
      console.warn('alerts failed:', err.message);
      if (loading) setLoading(false);
    }

    try {
      const histRes = await getAlertHistory(30);
      setHistory(histRes.data);
    } catch (err) {
      console.warn('alert history failed:', err.message);
    }

    try {
      const recsRes = await getRecommendations();
      setRecommendations(recsRes.data);
    } catch (err) {
      console.warn('recommendations failed:', err.message);
    }

    // Fetch anomaly data from prediction endpoint
    try {
      const predRes = await getPrediction();
      setAnomaly(predRes.data?.anomaly || null);
      setProcessAnomalies(predRes.data?.process_anomalies || []);
    } catch (err) {
      console.warn('anomaly data failed:', err.message);
    }
  }, [loading]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const getAlertIcon = (level) => {
    switch (level) {
      case 'critical': return '🔴';
      case 'warning':  return '🟡';
      default:         return '🟢';
    }
  };

  const formatTimestamp = (ts) => {
    if (!ts) return '';
    try { return new Date(ts).toLocaleString(); } catch { return ts; }
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <span>Loading alerts...</span>
      </div>
    );
  }

  // Separate threshold alerts from process alerts
  const thresholdAlerts = (alerts.alerts || []).filter(a => a.source !== 'anomaly_detection');
  const processAlerts = (alerts.alerts || []).filter(a => a.source === 'anomaly_detection');

  return (
    <div>
      <div className="page-header animate-in">
        <h2>Alerts & Recommendations</h2>
        <p>Threshold-based warnings, anomaly detection, and actionable optimization suggestions</p>
      </div>

      <div className="metric-cards animate-in animate-in-delay-1">
        <div className="card" style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.8px', fontWeight: 600 }}>Active Alerts</div>
          <div style={{ fontSize: '2.5rem', fontWeight: 800, color: alerts.alerts?.length > 0 ? '#ef4444' : '#00e676' }}>
            {alerts.alerts?.length || 0}
          </div>
        </div>
        <div className="card" style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.8px', fontWeight: 600 }}>CPU Now</div>
          <div style={{ fontSize: '2.5rem', fontWeight: 800, color: (alerts.current?.cpu || 0) > 90 ? '#ef4444' : (alerts.current?.cpu || 0) > 75 ? '#ffa726' : '#06b6d4' }}>
            {alerts.current?.cpu?.toFixed(1) || '—'}%
          </div>
        </div>
        <div className="card" style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.8px', fontWeight: 600 }}>RAM Now</div>
          <div style={{ fontSize: '2.5rem', fontWeight: 800, color: (alerts.current?.ram || 0) > 95 ? '#ef4444' : (alerts.current?.ram || 0) > 80 ? '#ffa726' : '#8b5cf6' }}>
            {alerts.current?.ram?.toFixed(1) || '—'}%
          </div>
        </div>
        <div className="card" style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.8px', fontWeight: 600 }}>Anomaly Status</div>
          <div style={{ fontSize: '2.5rem', fontWeight: 800, color: anomaly?.is_anomaly ? '#ef4444' : '#00e676' }}>
            {anomaly?.is_anomaly ? '⚠️' : '✅'}
          </div>
          <div style={{ fontSize: '0.72rem', color: '#64748b' }}>
            {anomaly?.is_anomaly ? anomaly.severity : 'Normal'}
          </div>
        </div>
      </div>

      <div className="grid-2 animate-in animate-in-delay-2">
        {/* ── Threshold Alerts ──────────────────────────────────── */}
        <div className="card">
          <div className="chart-header">
            <span className="chart-title">Threshold Alerts</span>
          </div>
          {thresholdAlerts.length > 0 ? (
            thresholdAlerts.map((alert, i) => (
              <div key={i} className={`alert-card ${alert.level}`}>
                <span className="alert-icon">{getAlertIcon(alert.level)}</span>
                <div className="alert-content">
                  <h4>{alert.message}</h4>
                  <p>Category: {alert.category}</p>
                </div>
              </div>
            ))
          ) : (
            <div className="empty-state" style={{ padding: '30px' }}>
              <div className="empty-state-icon">✅</div>
              <p style={{ color: '#00e676' }}>All systems normal — no active alerts</p>
            </div>
          )}
          {alerts.thresholds && (
            <div style={{ marginTop: '16px', padding: '12px', background: 'rgba(255,255,255,0.02)', borderRadius: '8px' }}>
              <div style={{ fontSize: '0.75rem', color: '#64748b', marginBottom: '8px', fontWeight: 600 }}>Alert Thresholds</div>
              <div style={{ fontSize: '0.78rem', color: '#94a3b8', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px' }}>
                <span>CPU Warning: {alerts.thresholds.cpu_warning}%</span>
                <span>CPU Critical: {alerts.thresholds.cpu_critical}%</span>
                <span>RAM Warning: {alerts.thresholds.ram_warning}%</span>
                <span>RAM Critical: {alerts.thresholds.ram_critical}%</span>
                <span>Disk Warning: {alerts.thresholds.disk_warning}%</span>
                <span>Disk Critical: {alerts.thresholds.disk_critical}%</span>
              </div>
            </div>
          )}
        </div>

        {/* ── Anomaly Detection ─────────────────────────────────── */}
        <div className="card">
          <div className="chart-header">
            <span className="chart-title">🧠 Anomaly Detection (ML)</span>
          </div>

          {/* ML Anomaly Status */}
          {anomaly && (
            <div className={`alert-card ${anomaly.is_anomaly ? (anomaly.severity === 'critical' ? 'critical' : 'warning') : 'info'}`}
                 style={{ marginBottom: '12px' }}>
              <span className="alert-icon">{anomaly.is_anomaly ? '🔴' : '🟢'}</span>
              <div className="alert-content">
                <h4>{anomaly.is_anomaly ? `Anomaly Detected (${anomaly.severity})` : 'System Normal'}</h4>
                <p>
                  {anomaly.is_anomaly
                    ? anomaly.details?.join('; ') || 'Unusual pattern detected'
                    : 'No anomalies detected by the ML model'}
                </p>
                <p style={{ fontSize: '0.72rem', color: '#64748b', marginTop: '4px' }}>
                  Score: {anomaly.anomaly_score} (lower = more anomalous)
                </p>
              </div>
            </div>
          )}

          {/* Process Anomalies */}
          {processAnomalies.length > 0 && (
            <>
              <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginBottom: '8px', marginTop: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Process Anomalies
              </div>
              {processAnomalies.slice(0, 5).map((pa, i) => (
                <div key={i} className={`alert-card ${pa.severity}`}>
                  <span className="alert-icon">{pa.severity === 'critical' ? '🔴' : '🟡'}</span>
                  <div className="alert-content">
                    <h4>{pa.message}</h4>
                    <p>PID: {pa.pid} · Type: {pa.type}</p>
                  </div>
                </div>
              ))}
            </>
          )}

          {/* Process alerts from threshold system */}
          {processAlerts.length > 0 && processAnomalies.length === 0 && (
            <>
              <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginBottom: '8px', marginTop: '12px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Process Warnings
              </div>
              {processAlerts.map((alert, i) => (
                <div key={i} className={`alert-card ${alert.level}`}>
                  <span className="alert-icon">{getAlertIcon(alert.level)}</span>
                  <div className="alert-content">
                    <h4>{alert.message}</h4>
                  </div>
                </div>
              ))}
            </>
          )}

          {!anomaly && processAnomalies.length === 0 && processAlerts.length === 0 && (
            <div className="empty-state" style={{ padding: '20px' }}>
              <p style={{ color: '#64748b' }}>Anomaly detection model not loaded. Train the model first.</p>
            </div>
          )}
        </div>
      </div>

      {/* ── Recommendations ───────────────────────────────────── */}
      <div className="card animate-in animate-in-delay-3" style={{ marginTop: '18px' }}>
        <div className="chart-header">
          <span className="chart-title">💡 AI Recommendations</span>
        </div>
        {recommendations.length > 0 ? recommendations.map((rec, i) => (
          <div key={i} className={`alert-card ${rec.priority}`}>
            <span className="alert-icon">{getAlertIcon(rec.priority)}</span>
            <div className="alert-content">
              <h4>{rec.message}</h4>
              <p>{rec.detail}</p>
            </div>
          </div>
        )) : (
          <div className="empty-state" style={{ padding: '30px' }}>
            <p>Loading recommendations...</p>
          </div>
        )}
      </div>

      {/* ── Alert History ─────────────────────────────────────── */}
      <div className="card animate-in animate-in-delay-4" style={{ marginTop: '18px' }}>
        <div className="chart-header">
          <span className="chart-title">Alert History</span>
        </div>
        {history.length > 0 ? (
          <table className="process-table">
            <thead>
              <tr><th>Time</th><th>Level</th><th>Message</th></tr>
            </thead>
            <tbody>
              {history.slice().reverse().map((alert, i) => (
                <tr key={i}>
                  <td style={{ fontSize: '0.78rem', color: '#64748b', whiteSpace: 'nowrap' }}>{formatTimestamp(alert.timestamp)}</td>
                  <td><span className={`status-badge ${alert.level === 'critical' ? 'stopped' : 'running'}`}>{alert.level}</span></td>
                  <td>{alert.message}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="empty-state" style={{ padding: '30px' }}>
            <div className="empty-state-icon">📋</div>
            <p>No alert history yet. Alerts are logged when thresholds are breached.</p>
          </div>
        )}
      </div>
    </div>
  );
}
