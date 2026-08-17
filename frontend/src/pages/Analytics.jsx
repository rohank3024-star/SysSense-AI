import { useState, useEffect, useCallback } from 'react';
import LiveChart from '../components/LiveChart';
import { getMetricsSince, getMetricsAggregated, getMetricsCount } from '../services/api';

const TIME_RANGES = [
  { label: '1h',  hours: 1,   bucket: 5 },
  { label: '6h',  hours: 6,   bucket: 15 },
  { label: '24h', hours: 24,  bucket: 60 },
  { label: '7d',  hours: 168, bucket: 360 },
];

export default function Analytics() {
  const [range, setRange] = useState(TIME_RANGES[0]);
  const [rawData, setRawData] = useState([]);
  const [aggregated, setAggregated] = useState([]);
  const [totalCount, setTotalCount] = useState(0);
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    try {
      const rawRes = await getMetricsSince(range.hours);
      setRawData(rawRes.data);
      setLoading(false);
    } catch (err) {
      console.warn('metrics/since failed:', err.message);
      setLoading(false);
    }

    try {
      const aggRes = await getMetricsAggregated(range.hours, range.bucket);
      setAggregated(aggRes.data);
    } catch (err) {
      console.warn('metrics/aggregated failed:', err.message);
    }

    try {
      const countRes = await getMetricsCount();
      setTotalCount(countRes.data.count);
    } catch (err) {
      console.warn('metrics/count failed:', err.message);
    }
  }, [range]);

  useEffect(() => {
    setLoading(true);
    fetchData();
  }, [fetchData]);

  const stats = { cpu: { avg: 0, max: 0, min: 100 }, ram: { avg: 0, max: 0, min: 100 }, disk: { avg: 0, max: 0, min: 100 } };
  if (rawData.length > 0) {
    rawData.forEach((r) => {
      ['cpu', 'ram', 'disk'].forEach((key) => {
        const val = r[`${key}_percent`] || 0;
        stats[key].avg += val;
        stats[key].max = Math.max(stats[key].max, val);
        stats[key].min = Math.min(stats[key].min, val);
      });
    });
    ['cpu', 'ram', 'disk'].forEach((key) => { stats[key].avg = stats[key].avg / rawData.length; });
  }

  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <span>Loading analytics...</span>
      </div>
    );
  }

  return (
    <div>
      <div className="page-header animate-in">
        <h2>Analytics</h2>
        <p>Historical performance analysis — {totalCount.toLocaleString()} total data points collected</p>
      </div>

      <div className="chart-controls animate-in" style={{ marginBottom: '20px' }}>
        {TIME_RANGES.map((tr) => (
          <button key={tr.label} className={`chart-btn ${range.label === tr.label ? 'active' : ''}`} onClick={() => setRange(tr)}>
            {tr.label}
          </button>
        ))}
      </div>

      <div className="metric-cards animate-in animate-in-delay-1">
        {['cpu', 'ram', 'disk'].map((key) => (
          <div className="card" key={key}>
            <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.8px', fontWeight: 600 }}>
              {key === 'cpu' ? 'CPU' : key === 'ram' ? 'Memory' : 'Disk'} — {range.label}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px', textAlign: 'center' }}>
              <div>
                <div style={{ fontSize: '0.7rem', color: '#64748b', marginBottom: '2px' }}>AVG</div>
                <div style={{ fontSize: '1.3rem', fontWeight: 700, color: key === 'cpu' ? '#06b6d4' : key === 'ram' ? '#8b5cf6' : '#f97316' }}>{stats[key].avg.toFixed(1)}%</div>
              </div>
              <div>
                <div style={{ fontSize: '0.7rem', color: '#64748b', marginBottom: '2px' }}>MAX</div>
                <div style={{ fontSize: '1.3rem', fontWeight: 700, color: '#ef4444' }}>{stats[key].max.toFixed(1)}%</div>
              </div>
              <div>
                <div style={{ fontSize: '0.7rem', color: '#64748b', marginBottom: '2px' }}>MIN</div>
                <div style={{ fontSize: '1.3rem', fontWeight: 700, color: '#84cc16' }}>{stats[key].min.toFixed(1)}%</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="card animate-in animate-in-delay-2" style={{ marginTop: '18px' }}>
        <LiveChart
          data={rawData.length > 500 ? rawData.filter((_, i) => i % Math.ceil(rawData.length / 500) === 0) : rawData}
          title={`Resource Usage — Last ${range.label}`}
          height={320}
        />
      </div>

      {aggregated.length > 0 && (
        <div className="card animate-in animate-in-delay-3" style={{ marginTop: '18px' }}>
          <LiveChart
            data={aggregated.map((a) => ({ timestamp: a.bucket, cpu_percent: a.avg_cpu, ram_percent: a.avg_ram, disk_percent: a.avg_disk }))}
            title={`Aggregated Averages (${range.bucket}min buckets)`}
            height={280}
          />
          <div style={{ fontSize: '0.75rem', color: '#64748b', textAlign: 'center', marginTop: '8px' }}>
            {aggregated.length} buckets · {rawData.length} raw samples in this range
          </div>
        </div>
      )}
    </div>
  );
}
