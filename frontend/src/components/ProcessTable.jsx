import { useState } from 'react';

export default function ProcessTable({ processes, onSort }) {
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState('cpu');

  const filtered = processes.filter((p) =>
    p.name?.toLowerCase().includes(search.toLowerCase())
  );

  const handleSort = (field) => {
    setSortBy(field);
    if (onSort) onSort(field);
  };

  const getStatusClass = (status) => {
    if (!status) return '';
    const s = status.toLowerCase();
    if (s === 'running') return 'running';
    if (s === 'sleeping' || s === 'idle') return 'sleeping';
    if (s === 'stopped' || s === 'zombie') return 'stopped';
    return '';
  };

  const formatMemory = (mb) => {
    if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
    return `${mb.toFixed(1)} MB`;
  };

  return (
    <div className="process-table-wrapper">
      <div className="process-controls">
        <input
          type="text"
          className="search-input"
          placeholder="🔍 Search processes..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          id="process-search"
        />
        <select
          className="sort-select"
          value={sortBy}
          onChange={(e) => handleSort(e.target.value)}
          id="process-sort"
        >
          <option value="cpu">Sort by CPU %</option>
          <option value="memory">Sort by Memory %</option>
        </select>
      </div>

      <table className="process-table">
        <thead>
          <tr>
            <th onClick={() => handleSort('pid')}>PID</th>
            <th onClick={() => handleSort('name')}>Name</th>
            <th onClick={() => handleSort('cpu')}>CPU %</th>
            <th onClick={() => handleSort('memory')}>Memory %</th>
            <th>Memory</th>
            <th>Threads</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {filtered.map((proc, i) => (
            <tr key={`${proc.pid}-${i}`}>
              <td style={{ color: '#64748b', fontFamily: 'monospace' }}>{proc.pid}</td>
              <td className="process-name">{proc.name || 'Unknown'}</td>
              <td>
                <span style={{
                  color: proc.cpu_percent > 50 ? '#ef4444' :
                         proc.cpu_percent > 20 ? '#ffa726' : '#94a3b8'
                }}>
                  {(proc.cpu_percent || 0).toFixed(1)}%
                </span>
              </td>
              <td>
                <span style={{
                  color: proc.memory_percent > 10 ? '#ec4899' :
                         proc.memory_percent > 5 ? '#8b5cf6' : '#94a3b8'
                }}>
                  {(proc.memory_percent || 0).toFixed(2)}%
                </span>
              </td>
              <td style={{ color: '#64748b' }}>
                {formatMemory(proc.memory_rss_mb || 0)}
              </td>
              <td style={{ color: '#64748b' }}>{proc.threads || 0}</td>
              <td>
                <span className={`status-badge ${getStatusClass(proc.status)}`}>
                  {proc.status || 'unknown'}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {filtered.length === 0 && (
        <div className="empty-state">
          <div className="empty-state-icon">🔍</div>
          <p>No processes found matching "{search}"</p>
        </div>
      )}
    </div>
  );
}
