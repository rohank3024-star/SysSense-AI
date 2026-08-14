import { useState, useEffect, useCallback } from 'react';
import ProcessTable from '../components/ProcessTable';
import { getProcesses, getProcessSummary } from '../services/api';

export default function ProcessManager() {
  const [processes, setProcesses] = useState([]);
  const [summary, setSummary] = useState(null);
  const [sortBy, setSortBy] = useState('cpu');
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    try {
      const procRes = await getProcesses(sortBy, 50);
      setProcesses(procRes.data);
      setLoading(false);
    } catch (err) {
      console.warn('processes failed:', err.message);
      if (!loading) return; // keep old data on refresh failures
    }

    try {
      const sumRes = await getProcessSummary();
      setSummary(sumRes.data);
    } catch (err) {
      console.warn('process summary failed:', err.message);
    }
  }, [sortBy, loading]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const handleSort = (field) => {
    setSortBy(field === 'memory' ? 'memory' : 'cpu');
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <span>Loading processes...</span>
      </div>
    );
  }

  return (
    <div>
      <div className="page-header animate-in">
        <h2>Process Manager</h2>
        <p>View and manage running processes — like Task Manager, but smarter</p>
      </div>

      {summary && (
        <div className="metric-cards animate-in animate-in-delay-1" style={{ marginBottom: '20px' }}>
          <div className="card" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.8px', fontWeight: 600 }}>Total Processes</div>
            <div style={{ fontSize: '2rem', fontWeight: 800, color: '#06b6d4' }}>{summary.total}</div>
          </div>
          {Object.entries(summary.by_status || {}).slice(0, 3).map(([status, count]) => (
            <div className="card" key={status} style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.8px', fontWeight: 600 }}>{status}</div>
              <div style={{ fontSize: '2rem', fontWeight: 800, color: '#8b5cf6' }}>{count}</div>
            </div>
          ))}
        </div>
      )}

      <div className="card animate-in animate-in-delay-2">
        <ProcessTable processes={processes} onSort={handleSort} />
      </div>
    </div>
  );
}
