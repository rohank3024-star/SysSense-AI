export default function HealthScore({ score, label, color }) {
  const radius = 65;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;

  return (
    <div className="health-score-container">
      <div className="health-ring">
        <svg width="160" height="160" viewBox="0 0 160 160">
          <circle
            className="health-ring-bg"
            cx="80" cy="80" r={radius}
          />
          <circle
            className="health-ring-fill"
            cx="80" cy="80" r={radius}
            stroke={color || '#06b6d4'}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
          />
        </svg>
        <div className="health-score-value">
          <div className="health-score-number" style={{ color: color }}>
            {Math.round(score)}
          </div>
          <div className="health-score-max">/100</div>
        </div>
      </div>
      <div className="health-score-label" style={{ color: color }}>
        {label || 'Good'}
      </div>
    </div>
  );
}
