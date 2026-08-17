import axios from 'axios';

const API_BASE = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 15000,
});

// ── Metrics ──────────────────────────────────────────────────────────

export const getCurrentMetrics = () => api.get('/metrics/current');

export const getMetricsHistory = (limit = 100) =>
  api.get('/metrics/history', { params: { limit } });

export const getMetricsSince = (hours = 1) =>
  api.get('/metrics/since', { params: { hours } });

export const getMetricsAggregated = (hours = 24, bucketMinutes = 60) =>
  api.get('/metrics/aggregated', { params: { hours, bucket_minutes: bucketMinutes } });

export const getMetricsCount = () => api.get('/metrics/count');

// ── Processes ────────────────────────────────────────────────────────

export const getProcesses = (sortBy = 'cpu', limit = 25, search = '') =>
  api.get('/processes', { params: { sort_by: sortBy, limit, search } });

export const getProcessSummary = () => api.get('/processes/summary');

// ── Alerts ───────────────────────────────────────────────────────────

export const getAlerts = () => api.get('/alerts');

export const getAlertHistory = (limit = 50) =>
  api.get('/alerts/history', { params: { limit } });

// ── Prediction ───────────────────────────────────────────────────────

export const getPrediction = () => api.get('/predict');

// ── Health & Recommendations ─────────────────────────────────────────

export const getHealthScore = () => api.get('/health');

export const getRecommendations = () => api.get('/recommendations');

// ── System Info ──────────────────────────────────────────────────────

export const getSystemInfo = () => api.get('/system-info');

export default api;
