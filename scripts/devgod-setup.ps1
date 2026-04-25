Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath ".env.devgod") -and (Test-Path -LiteralPath ".env.devgod.example")) {
    Copy-Item -LiteralPath ".env.devgod.example" -Destination ".env.devgod"
    Write-Host "created .env.devgod from .env.devgod.example"
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "docker is required for local devgod setup unless you provide a managed Postgres backend"
}

try {
    docker version | Out-Null
} catch {
    throw "docker is installed but not usable from this environment; enable Docker Desktop integration or provide a managed Postgres backend"
}

Get-Content ".env.devgod" | ForEach-Object {
    if ($_ -match '^\s*#' -or $_ -notmatch '=') {
        return
    }

    $parts = $_ -split '=', 2
    $value = $parts[1].Trim('"')
    [System.Environment]::SetEnvironmentVariable($parts[0], $value)
}

if (-not $env:DEVGOD_PROJECT_REPO_PATH -or $env:DEVGOD_PROJECT_REPO_PATH -eq "/absolute/path/to/repo") {
    $env:DEVGOD_PROJECT_REPO_PATH = (Get-Location).Path
}

if (-not $env:DEVGOD_PROJECT_SLUG) {
    $env:DEVGOD_PROJECT_SLUG = Split-Path -Leaf (Get-Location)
}

if (-not $env:DEVGOD_PROJECT_NAME) {
    $env:DEVGOD_PROJECT_NAME = $env:DEVGOD_PROJECT_SLUG
}

if (-not $env:DEVGOD_DOCKER_CONTAINER_NAME) {
    $env:DEVGOD_DOCKER_CONTAINER_NAME = "devgod-postgres-$($env:DEVGOD_PROJECT_SLUG)"
}

docker compose -f docker-compose.devgod.yml up -d devgod-postgres

Write-Host "waiting for devgod-postgres to become healthy"
$healthy = $false
for ($i = 0; $i -lt 60; $i++) {
    $status = ""
    try {
        $status = docker inspect -f "{{.State.Health.Status}}" $env:DEVGOD_DOCKER_CONTAINER_NAME 2>$null
    } catch {
        $status = ""
    }

    if ($status -eq "healthy") {
        $healthy = $true
        break
    }

    Start-Sleep -Seconds 2
}

if (-not $healthy) {
    docker logs $env:DEVGOD_DOCKER_CONTAINER_NAME --tail 100
    throw "devgod-postgres did not become healthy"
}

npm install
npm run devgod:migrate
npm run devgod:bootstrap
npm run devgod:verify:setup

Write-Host ""
Write-Host "devgod local setup complete"
Write-Host "workspace: $($env:DEVGOD_WORKSPACE_SLUG)"
Write-Host "project: $($env:DEVGOD_PROJECT_SLUG)"
Write-Host "database: $($env:DEVGOD_CORE_DATABASE_URL)"
