#!/bin/bash
set -e

echo "Starting BitBat ingestion service..."
python scripts/run_ingestion_service.py &

echo "Starting BitBat monitoring agent..."
python scripts/run_monitoring_agent.py &

echo "Starting BitBat API server..."
exec uvicorn bitbat.api.app:app --host 0.0.0.0 --port 8000
