# SysSense Backend — Starter Scaffold

Tested and working: collector, database layer, and all API endpoints.

## Setup

```bash
pip install -r requirements.txt
```

## Run (two terminals, both from this folder)

**Terminal 1 — start logging data (leave this running):**
```bash
python collector.py
```
Start this on Day 1 and leave it running in the background as much as
possible. Your ML model (Day 20+) is only as good as the history you've
collected by then.

**Terminal 2 — start the API:**
```bash
uvicorn main:app --reload
```
API docs (auto-generated, useful for testing without the frontend yet):
http://localhost:8000/docs

## Endpoints

| Endpoint | Returns |
|---|---|
| `GET /metrics/current` | live CPU/RAM/disk snapshot |
| `GET /metrics/history?limit=100` | logged history for charts |
| `GET /processes?sort_by=cpu&limit=15` | top processes |
| `GET /alerts` | rule-based threshold alerts |
| `GET /predict` | naive baseline (swap in your trained model later) |

## What to build next

1. **React dashboard** — point Axios at `http://localhost:8000`, poll
   `/metrics/current` and `/alerts` every few seconds, chart
   `/metrics/history` with Recharts.
2. **ML model** — once `sysense.db` has a few days of data, pull it with
   `database.get_all_metrics_for_training()`, engineer a "value 30s later"
   label, train a `RandomForestRegressor`, and compare its error against
   the naive baseline (`predicted = current`). That comparison number is
   the centerpiece of your report.
3. **Wire the real model into `/predict`** in `main.py`, replacing the
   placeholder — load it with `joblib` at startup so you're not
   retraining on every request.

## Notes

- `sysense.db` is created automatically on first run — no manual setup.
- CORS is pre-configured for React on ports 3000 and 5173 (CRA and Vite
  defaults). Update `allow_origins` in `main.py` if you use a different port.
- `disk_usage("/")` checks the root partition — fine on Linux/Mac; on
  Windows use `disk_usage("C:\\")` (Windows also uses `\\` in
  `psutil.disk_partitions()` if you want to check available drives).
