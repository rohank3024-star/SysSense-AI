# SysSense Backend — Initial Scaffold

Early prototype of the backend before full modularization. Kept for reference.

## Setup

```bash
pip install -r requirements.txt
```

## Run

**Terminal 1 — start the data collector:**
```bash
python collector.py
```

**Terminal 2 — start the API server:**
```bash
uvicorn main:app --reload
```
API docs: http://localhost:8000/docs

## Endpoints

| Endpoint | Returns |
|---|---|
| `GET /metrics/current` | live CPU/RAM/disk snapshot |
| `GET /metrics/history?limit=100` | logged history for charts |
| `GET /processes?sort_by=cpu&limit=15` | top processes |
| `GET /alerts` | rule-based threshold alerts |
| `GET /predict` | naive baseline prediction |

## Notes

- `sysense.db` is created automatically on first run.
- CORS is pre-configured for React on ports 3000 and 5173.

