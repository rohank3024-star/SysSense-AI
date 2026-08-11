export default function MetricCard({ type, label, value, unit, detail, icon }) {
  const percent = typeof value === 'number' ? Math.min(value, 100) : 0;

  return (
    <div className={`card metric-card ${type} animate-in`}>
      <div className="metric-card-glow"></div>
      <div className="metric-card-header">
        <span className="metric-card-label">{label}</span>
        <span className="metric-card-icon">{icon}</span>
      </div>
      <div className="metric-card-value">
        {typeof value === 'number' ? value.toFixed(1) : value}
        <span className="metric-card-unit">{unit}</span>
      </div>
      {detail && <div className="metric-card-detail">{detail}</div>}
      {typeof percent === 'number' && unit === '%' && (
        <div className="progress-bar">
          <div
            className="progress-bar-fill"
            style={{ width: `${percent}%` }}
          />
        </div>
      )}
    </div>
  );
}
