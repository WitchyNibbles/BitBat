#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f .env.devgod && -f .env.devgod.example ]]; then
  cp .env.devgod.example .env.devgod
  echo "created .env.devgod from .env.devgod.example"
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required for local devgod setup unless you provide a managed Postgres backend" >&2
  exit 1
fi

if ! docker version >/dev/null 2>&1; then
  echo "docker is installed but not usable from this environment; enable Docker Desktop integration or provide a managed Postgres backend" >&2
  exit 1
fi

set -a
source ./.env.devgod
set +a

if [[ -z "${DEVGOD_PROJECT_REPO_PATH:-}" || "${DEVGOD_PROJECT_REPO_PATH}" == "/absolute/path/to/repo" ]]; then
  export DEVGOD_PROJECT_REPO_PATH="$(pwd)"
fi

if [[ -z "${DEVGOD_PROJECT_SLUG:-}" ]]; then
  export DEVGOD_PROJECT_SLUG="$(basename "$(pwd)" | tr '[:upper:]' '[:lower:]')"
fi

if [[ -z "${DEVGOD_PROJECT_NAME:-}" ]]; then
  export DEVGOD_PROJECT_NAME="${DEVGOD_PROJECT_SLUG}"
fi

if [[ -z "${DEVGOD_DOCKER_CONTAINER_NAME:-}" ]]; then
  export DEVGOD_DOCKER_CONTAINER_NAME="devgod-postgres-${DEVGOD_PROJECT_SLUG}"
fi

docker compose -f docker-compose.devgod.yml up -d devgod-postgres

echo "waiting for devgod-postgres to become healthy"
for _ in {1..60}; do
  if [[ "$(docker inspect -f '{{.State.Health.Status}}' "${DEVGOD_DOCKER_CONTAINER_NAME}" 2>/dev/null || true)" == "healthy" ]]; then
    break
  fi
  sleep 2
done

if [[ "$(docker inspect -f '{{.State.Health.Status}}' "${DEVGOD_DOCKER_CONTAINER_NAME}" 2>/dev/null || true)" != "healthy" ]]; then
  echo "devgod-postgres did not become healthy" >&2
  docker logs "${DEVGOD_DOCKER_CONTAINER_NAME}" --tail 100 >&2 || true
  exit 1
fi

npm install
npm run devgod:migrate
npm run devgod:bootstrap
npm run devgod:verify:setup

echo ""
echo "devgod local setup complete"
echo "workspace: ${DEVGOD_WORKSPACE_SLUG}"
echo "project: ${DEVGOD_PROJECT_SLUG}"
echo "database: ${DEVGOD_CORE_DATABASE_URL}"
