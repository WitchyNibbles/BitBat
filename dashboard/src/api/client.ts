/**
 * Typed API client for the BitBat REST API.
 *
 * Base URL is resolved from the VITE_API_URL environment variable,
 * falling back to http://localhost:8000 during local development.
 */

const BASE_URL =
  (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000';

// ---------------------------------------------------------------------------
// Response / Request interfaces
// ---------------------------------------------------------------------------

// Health

export interface HealthResponse {
  status: string;
  version: string;
  uptime_seconds: number;
}

export interface ServiceStatus {
  name: string;
  status: string;
  detail?: string | null;
}

export interface SchemaReadinessDetails {
  compatibility_state: string;
  is_compatible: boolean;
  can_auto_upgrade?: boolean;
  missing_columns?: Record<string, string[]>;
  missing_columns_text?: string | null;
  detail?: string | null;
}

export interface DetailedHealthResponse extends HealthResponse {
  services: ServiceStatus[];
  schema_readiness?: SchemaReadinessDetails | null;
}

// Predictions

export interface PredictionResponse {
  id: number;
  timestamp_utc: string;
  predicted_direction: string;
  predicted_return?: number | null;
  predicted_price?: number | null;
  actual_direction?: string | null;
  actual_return?: number | null;
  correct?: boolean | null;
  model_version: string;
  freq: string;
  horizon: string;
}

export interface PredictionListResponse {
  predictions: PredictionResponse[];
  total: number;
  freq: string;
  horizon: string;
}

// Performance

export interface PerformanceResponse {
  model_version?: string | null;
  freq: string;
  horizon: string;
  window_days: number;
  total_predictions: number;
  realized_predictions: number;
  hit_rate?: number | null;
  avg_return?: number | null;
  win_streak: number;
  lose_streak: number;
  mae?: number | null;
  rmse?: number | null;
  directional_accuracy?: number | null;
}

// System status

export interface SystemStatusResponse {
  database_ok: boolean;
  database_present: boolean;
  model_exists: boolean;
  dataset_exists: boolean;
  schema_readiness?: SchemaReadinessDetails | null;
  active_model_version?: string | null;
  total_predictions: number;
  last_prediction_time?: string | null;
}

// System logs

export interface SystemLogEntry {
  timestamp: string;
  level: string;
  message: string;
  service?: string | null;
}

export interface SystemLogsResponse {
  logs: SystemLogEntry[];
  total: number;
}

// Retraining events

export interface RetrainingEvent {
  id: number;
  started_at: string;
  trigger_reason: string;
  status: string;
  old_model_version?: string | null;
  new_model_version?: string | null;
  cv_improvement?: number | null;
  training_duration_seconds?: number | null;
}

export interface RetrainingEventsResponse {
  events: RetrainingEvent[];
  total: number;
}

// Performance snapshots

export interface PerformanceSnapshot {
  snapshot_time: string;
  model_version: string;
  hit_rate?: number | null;
  total_predictions: number;
  sharpe_ratio?: number | null;
  max_drawdown?: number | null;
}

export interface PerformanceSnapshotsResponse {
  snapshots: PerformanceSnapshot[];
}

// Ingestion status

export interface IngestionStatus {
  prices: string;
  news: string;
  prices_mtime?: string | null;
  news_mtime?: string | null;
}

// Training

export interface TrainingResponse {
  status: string;
  model_version?: string | null;
  duration_seconds?: number | null;
  error?: string | null;
}

// Settings

export interface SettingsResponse {
  preset: string;
  freq: string;
  horizon: string;
  tau: number;
  enter_threshold: number;
  valid_freqs: string[];
  valid_horizons: string[];
}

export interface SettingsUpdateRequest {
  preset?: string | null;
  freq?: string | null;
  horizon?: string | null;
  tau?: number | null;
  enter_threshold?: number | null;
}

// ---------------------------------------------------------------------------
// Fetch wrapper
// ---------------------------------------------------------------------------

class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${BASE_URL}${path}`;
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(body || `HTTP ${res.status}`, res.status);
  }
  return (await res.json()) as T;
}

function qs(params: Record<string, string | number | undefined | null>): string {
  const entries = Object.entries(params).filter(
    (pair): pair is [string, string | number] => pair[1] != null,
  );
  if (entries.length === 0) return '';
  return '?' + new URLSearchParams(entries.map(([k, v]) => [k, String(v)])).toString();
}

// ---------------------------------------------------------------------------
// Typed endpoint helpers
// ---------------------------------------------------------------------------

export const api = {
  // Health
  health: () => apiFetch<HealthResponse>('/health'),

  healthDetailed: () => apiFetch<DetailedHealthResponse>('/health/detailed'),

  // Predictions
  latestPrediction: (freq?: string, horizon?: string) =>
    apiFetch<PredictionResponse>(`/predictions/latest${qs({ freq, horizon })}`),

  predictionHistory: (freq?: string, horizon?: string, days?: number, limit?: number) =>
    apiFetch<PredictionListResponse>(
      `/predictions/history${qs({ freq, horizon, days, limit })}`,
    ),

  performance: (freq?: string, horizon?: string, days?: number) =>
    apiFetch<PerformanceResponse>(`/predictions/performance${qs({ freq, horizon, days })}`),

  // Analytics / System status
  systemStatus: (freq?: string, horizon?: string) =>
    apiFetch<SystemStatusResponse>(`/analytics/status${qs({ freq, horizon })}`),

  // System
  systemLogs: (limit?: number, level?: string) =>
    apiFetch<SystemLogsResponse>(`/system/logs${qs({ limit, level })}`),

  retrainingEvents: (limit?: number) =>
    apiFetch<RetrainingEventsResponse>(`/system/retraining-events${qs({ limit })}`),

  performanceSnapshots: (limit?: number) =>
    apiFetch<PerformanceSnapshotsResponse>(`/system/performance-snapshots${qs({ limit })}`),

  ingestionStatus: () => apiFetch<IngestionStatus>('/system/ingestion-status'),

  startTraining: (preset: string) =>
    apiFetch<TrainingResponse>('/system/training/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ preset }),
    }),

  // Settings
  getSettings: () => apiFetch<SettingsResponse>('/system/settings'),

  updateSettings: (settings: SettingsUpdateRequest) =>
    apiFetch<SettingsResponse>('/system/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    }),
} as const;
