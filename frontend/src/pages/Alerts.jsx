import { useState, useEffect, useCallback } from 'react';
import { getAlerts, getAlertHistory, getRecommendations } from '../services/api';

export default function Alerts() {
  const [alerts, setAlerts] = useState({ alerts: [], current: {} });
  const [history, setHistory] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
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

  return (
    <div>
      <div className="page-header animate-in">
        <h2>Alerts & Recommendations</h2>
        <p>Threshold-based warnings and actionable optimization suggestions</p>
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
          <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.8px', fontWeight: 600 }}>Disk Now</div>
          <div style={{ fontSize: '2.5rem', fontWeight: 800, color: (alerts.current?.disk || 0) > 95 ? '#ef4444' : (alerts.current?.disk || 0) > 80 ? '#ffa726' : '#f97316' }}>
            {alerts.current?.disk?.toFixed(1) || '—'}%
          </div>
        </div>
      </div>

      <div className="grid-2 animate-in animate-in-delay-2">
        <div className="card">
          <div className="chart-header">
            <span className="chart-title">Active Alerts</span>
          </div>
          {alerts.alerts?.length > 0 ? (
            alerts.alerts.map((alert, i) => (
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

        <div className="card">
          <div className="chart-header">
            <span className="chart-title">💡 Recommendations</span>
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
      </div>

      <div className="card animate-in animate-in-delay-3" style={{ marginTop: '18px' }}>
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
