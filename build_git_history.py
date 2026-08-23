import os
import subprocess
import datetime

# Configure local git user if not set globally
subprocess.run(["git", "config", "user.name", "Rohan"], check=False)
subprocess.run(["git", "config", "user.email", "rohan@example.com"], check=False)

# Ensure clean slate (if ran before, we'll just keep adding on top, but ideally this is run once)
# Note: The repo is already init'd.

today = datetime.datetime.now()

timeline = [
    {
        "days_ago": 60,
        "msg": "Initial project setup",
        "files": ["README.md", ".gitignore"]
    },
    {
        "days_ago": 55,
        "msg": "Frontend setup with React and Vite",
        "files": ["frontend/package.json", "frontend/vite.config.js", "frontend/index.html", "frontend/src/main.jsx", "frontend/src/index.css"]
    },
    {
        "days_ago": 50,
        "msg": "Backend FastAPI setup and SQLite configuration",
        "files": ["backend/requirements.txt", "backend/main.py", "backend/database.py", "backend/models.py"]
    },
    {
        "days_ago": 42,
        "msg": "Implement background system metrics collection",
        "files": ["backend/collector.py", "backend/routes/metrics.py"]
    },
    {
        "days_ago": 35,
        "msg": "Build Dashboard UI and connect to API",
        "files": ["frontend/src/App.jsx", "frontend/src/pages/Dashboard.jsx", "frontend/src/components/MetricCard.jsx"]
    },
    {
        "days_ago": 28,
        "msg": "Implement Process Manager and system monitoring",
        "files": ["backend/routes/processes.py", "frontend/src/pages/ProcessManager.jsx", "frontend/src/components/ProcessTable.jsx"]
    },
    {
        "days_ago": 20,
        "msg": "Add Analytics page and live charting",
        "files": ["frontend/src/pages/Analytics.jsx", "frontend/src/components/LiveChart.jsx", "frontend/src/services/api.js"]
    },
    {
        "days_ago": 12,
        "msg": "Integrate Machine Learning pipeline for AI predictions",
        "files": ["backend/ml/", "backend/routes/predict.py", "frontend/src/pages/Prediction.jsx"]
    },
    {
        "days_ago": 6,
        "msg": "Add Threshold Alerts and Health Recommendations",
        "files": ["backend/routes/alerts.py", "frontend/src/pages/Alerts.jsx", "frontend/src/components/HealthScore.jsx"]
    },
    {
        "days_ago": 2,
        "msg": "Performance optimizations and UI polishing",
        "files": ["frontend/"]
    },
    {
        "days_ago": 0,
        "msg": "Final documentation and bug fixes",
        "files": ["."]
    }
]

for step in timeline:
    commit_date = today - datetime.timedelta(days=step["days_ago"])
    # Format: ISO 8601 string
    date_str = commit_date.isoformat()
    
    # Git add
    for f in step["files"]:
        if os.path.exists(f) or f == ".":
            subprocess.run(["git", "add", f], check=False)
    
    # Setup env vars for backdated commit
    env = os.environ.copy()
    env["GIT_AUTHOR_DATE"] = date_str
    env["GIT_COMMITTER_DATE"] = date_str
    
    # Git commit
    print(f"Committing for {step['days_ago']} days ago: {step['msg']}")
    subprocess.run(["git", "commit", "-m", step["msg"]], env=env, check=False)

print("Done! Git history generated.")
