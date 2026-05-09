#!/bin/bash
set -e

PRIMARY_API="${BITBAT_PRIMARY_API:-v2}"
V2_PORT="${BITBAT_V2_PORT:-8100}"
LEGACY_PORT="${BITBAT_LEGACY_PORT:-8000}"
LEGACY_SERVICES_ENABLED="${BITBAT_LEGACY_SERVICES_ENABLED:-false}"

if [ "${LEGACY_SERVICES_ENABLED}" = "true" ]; then
  echo "Starting BitBat ingestion service..."
  python scripts/run_ingestion_service.py &

  echo "Starting BitBat monitoring agent..."
  python scripts/run_monitoring_agent.py &
fi

if [ "${PRIMARY_API}" = "legacy" ]; then
  echo "Starting BitBat legacy API server on port ${LEGACY_PORT}..."
  exec uvicorn bitbat.api.app:app --host 0.0.0.0 --port "${LEGACY_PORT}"
fi

if [ -z "${BITBAT_V2_OPERATOR_TOKEN:-}" ] && [ "${BITBAT_V2_DEMO_MODE:-false}" != "true" ]; then
  echo "BITBAT_V2_OPERATOR_TOKEN must be set before starting the BitBat v2 API."
  echo "For local sandboxing only, set BITBAT_V2_DEMO_MODE=true to allow the demo fallback token."
  exit 1
fi

echo "Starting BitBat v2 API server on port ${V2_PORT}..."
exec uvicorn bitbat_v2.api.app:app --host 0.0.0.0 --port "${V2_PORT}"
