import { NavLink } from 'react-router-dom';
import { useEffect, useState } from 'react';

const navItems = [
  { path: '/',           label: 'Dashboard',      icon: '📊' },
  { path: '/processes',  label: 'Processes',      icon: '⚙️' },
  { path: '/analytics',  label: 'Analytics',      icon: '📈' },
  { path: '/prediction', label: 'AI Prediction',  icon: '🤖' },
  { path: '/alerts',     label: 'Alerts',         icon: '🔔' },
];

export default function Sidebar() {
  const [theme, setTheme] = useState('dark');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((currentTheme) =>
      currentTheme === 'dark' ? 'light' : 'dark'
    );
  };
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <div className="sidebar-logo-icon">🧠</div>
          <div className="sidebar-logo-text">
            <h1>SysSense AI</h1>
            <span>System Monitor</span>
          </div>
        </div>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === '/'}
            className={({ isActive }) =>
              `sidebar-link ${isActive ? 'active' : ''}`
            }
          >
            <span className="sidebar-link-icon">{item.icon}</span>
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="theme-toggle-container">
        <button
          className="theme-toggle"
          onClick={toggleTheme}
          type="button"
        >
          <span className="theme-toggle-icon">
            {theme === 'dark' ? '☀️' : '🌙'}
          </span>
          <span>
            {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
          </span>
        </button>
      </div>

      <div className="sidebar-footer">
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span className="pulse-dot"></span>
          <span>System monitoring active</span>
        </div>
      </div>
    </aside>
  );
}
