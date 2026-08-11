import { useState, useEffect, useCallback } from 'react';
import MetricCard from '../components/MetricCard';
import HealthScore from '../components/HealthScore';
import LiveChart from '../components/LiveChart';
import {
  getCurrentMetrics,
  getMetricsHistory,
  getHealthScore,
  getRecommendations,
} from '../services/api';

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null);
  const [history, setHistory] = useState([]);
  const [health, setHealth] = useState({ score: 0, label: 'Loading', color: '#64748b' });
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch each endpoint independently so one slow call doesn't block others
  const fetchData = useCallback(async () => {
    let anySuccess = false;

    try {
      const res = await getCurrentMetrics();
      setMetrics(res.data);
      anySuccess = true;
    } catch (err) {
      console.warn('metrics/current failed:', err.message);
    }

    try {
      const res = await getMetricsHistory(60);
      setHistory(res.data);
      anySuccess = true;
    } catch (err) {
      console.warn('metrics/history failed:', err.message);
    }

    try {
      const res = await getHealthScore();
      setHealth(res.data);
      anySuccess = true;
    } catch (err) {
      console.warn('health failed:', err.message);
    }

    try {
      const res = await getRecommendations();
      setRecommendations(res.data);
      anySuccess = true;
    } catch (err) {
      console.warn('recommendations failed:', err.message);
    }

    if (anySuccess) {
      setLoading(false);
      setError(null);
    } else if (loading) {
      setError('Cannot connect to SysSense API. Is the backend running on port 8000?');
      setLoading(false);
    }
  }, [loading]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 4000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const formatUptime = (seconds) => {
    if (!seconds) return '—';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${h}h ${m}m`;
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <span>Connecting to SysSense API...</span>
      </div>
    );
  }

  if (error && !metrics) {
    return (
      <div className="loading-container" style={{ flexDirection: 'column', gap: '16px' }}>
        <div style={{ fontSize: '2rem' }}>⚠️</div>
        <span style={{ color: '#ffa726' }}>{error}</span>
        <span style={{ fontSize: '0.8rem', color: '#64748b' }}>
          Run: <code>cd backend && uvicorn main:app --reload</code>
        </span>
      </div>
    );
  }

  return (
    <div>
      <div className="page-header animate-in">
        <h2>Dashboard</h2>
        <p>Real-time system monitoring &amp; health overview</p>
      </div>

      {/* ── Metric Cards ─────────────────────────────────────── */}
      <div className="metric-cards">
        <MetricCard
          type="cpu"
          label="CPU Usage"
          value={metrics?.cpu_percent}
          unit="%"
          detail={`${metrics?.cpu_count || '—'} cores · ${formatUptime(metrics?.uptime_seconds)} uptime`}
          icon="🔲"
        />
        <MetricCard
          type="ram"
          label="Memory"
          value={metrics?.ram_percent}
          unit="%"
          detail={`${metrics?.ram_used_gb || '—'} / ${metrics?.ram_total_gb || '—'} GB`}
          icon="🧠"
        />
        <MetricCard
          type="disk"
          label="Disk Usage"
          value={metrics?.disk_percent}
          unit="%"
          detail={`${metrics?.disk_used_gb || '—'} / ${metrics?.disk_total_gb || '—'} GB`}
          icon="💾"
        />
        <MetricCard
          type="net"
          label="Network"
          value={metrics?.net_sent_mb}
          unit=" MB↑"
          detail={`${metrics?.net_recv_mb?.toFixed(0) || '—'} MB received`}
          icon="🌐"
        />
      </div>

      {/* ── Chart + Health Score ──────────────────────────────── */}
      <div className="grid-dashboard animate-in animate-in-delay-2">
        <div className="card">
          <LiveChart
            data={history}
            title="Resource Usage — Last 60 Readings"
          />
        </div>
        <div className="card">
          <div className="chart-header">
            <span className="chart-title">System Health</span>
          </div>
          <HealthScore
            score={health.score}
            label={health.label}
            color={health.color}
          />
          {health.breakdown && (
            <div style={{ padding: '0 16px', fontSize: '0.78rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', marginBottom: '4px' }}>
                <span>CPU impact</span>
                <span>{health.breakdown.cpu_impact?.toFixed(1)}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', marginBottom: '4px' }}>
                <span>RAM impact</span>
                <span>{health.breakdown.ram_impact?.toFixed(1)}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#94a3b8', marginBottom: '4px' }}>
                <span>Disk impact</span>
                <span>{health.breakdown.disk_impact?.toFixed(1)}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── System Info + Recommendations ─────────────────────── */}
      <div className="grid-2 animate-in animate-in-delay-3">
        <div className="card">
          <div className="chart-header">
            <span className="chart-title">System Info</span>
          </div>
          <div style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
              <span>Processes</span>
              <span style={{ color: '#f1f5f9', fontWeight: 500 }}>{metrics?.process_count || '—'}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
              <span>Threads</span>
              <span style={{ color: '#f1f5f9', fontWeight: 500 }}>{metrics?.thread_count || '—'}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
              <span>CPU Cores</span>
              <span style={{ color: '#f1f5f9', fontWeight: 500 }}>{metrics?.cpu_count || '—'}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
              <span>RAM Total</span>
              <span style={{ color: '#f1f5f9', fontWeight: 500 }}>{metrics?.ram_total_gb || '—'} GB</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0' }}>
              <span>Uptime</span>
              <span style={{ color: '#f1f5f9', fontWeight: 500 }}>{formatUptime(metrics?.uptime_seconds)}</span>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="chart-header">
            <span className="chart-title">Recommendations</span>
          </div>
          {recommendations.length > 0 ? recommendations.map((rec, i) => (
            <div key={i} className={`alert-card ${rec.priority}`}>
              <span className="alert-icon">
                {rec.priority === 'critical' ? '🔴' :
                 rec.priority === 'warning' ? '🟡' : '🟢'}
              </span>
              <div className="alert-content">
                <h4>{rec.message}</h4>
                <p>{rec.detail}</p>
              </div>
            </div>
          )) : (
            <div className="empty-state" style={{ padding: '20px' }}>
              <p>Loading recommendations...</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
