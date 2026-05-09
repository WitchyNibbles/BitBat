const EXPLICIT_V2_BASE_URL = (import.meta.env.VITE_V2_API_URL as string | undefined) ?? '';
const V2_OPERATOR_TOKEN =
  (import.meta.env.VITE_V2_OPERATOR_TOKEN as string | undefined) ?? '';
const V2_BASE_URL_CACHE_KEY = 'bitbat.v2ApiBaseUrl';
const DEFAULT_V2_BASE_URL = 'http://localhost:8100';

function normalizeBaseUrl(value: string | undefined | null): string | null {
  const resolved = String(value ?? '').trim().replace(/\/+$/, '');
  return resolved ? resolved : null;
}

function candidateBaseUrls(): string[] {
  const candidates: string[] = [];
  const explicit = normalizeBaseUrl(EXPLICIT_V2_BASE_URL);
  const cached =
    typeof window !== 'undefined'
      ? normalizeBaseUrl(window.sessionStorage.getItem(V2_BASE_URL_CACHE_KEY))
      : null;

  if (explicit) {
    candidates.push(explicit);
  }
  if (cached && cached !== explicit) {
    candidates.push(cached);
  }

  if (typeof window !== 'undefined') {
    const { protocol, hostname, host } = window.location;
    candidates.push(`${protocol}//${host}`);
    candidates.push(`${protocol}//${hostname}:8100`);
    candidates.push(`${protocol}//${hostname}:8101`);
  }

  candidates.push(DEFAULT_V2_BASE_URL, 'http://localhost:8101');

  return Array.from(new Set(candidates.map((value) => value.replace(/\/+$/, '')))).filter(Boolean);
}

function requestHeaders(init?: RequestInit): HeadersInit {
  return {
    ...(init?.headers ?? {}),
    ...(V2_OPERATOR_TOKEN ? { 'X-BitBat-Operator-Token': V2_OPERATOR_TOKEN } : {}),
  };
}

async function probeBaseUrl(baseUrl: string): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 1500);
    const response = await fetch(`${baseUrl}/v1/health`, {
      method: 'GET',
      headers: requestHeaders(),
      signal: controller.signal,
    });
    window.clearTimeout(timeout);
    return response.status === 200 || response.status === 401 || response.status === 403;
  } catch {
    return false;
  }
}

let resolvedBaseUrlPromise: Promise<string> | null = null;

export async function resolveV2BaseUrl(): Promise<string> {
  if (resolvedBaseUrlPromise) {
    return resolvedBaseUrlPromise;
  }
  resolvedBaseUrlPromise = (async () => {
    const candidates = candidateBaseUrls();
    for (const candidate of candidates) {
      if (await probeBaseUrl(candidate)) {
        if (typeof window !== 'undefined') {
          window.sessionStorage.setItem(V2_BASE_URL_CACHE_KEY, candidate);
        }
        return candidate;
      }
    }
    return candidates[0] ?? DEFAULT_V2_BASE_URL;
  })();
  return resolvedBaseUrlPromise;
}

export interface V2HealthResponse {
  status: string;
  venue: string;
  product_id: string;
  trading_paused: boolean;
  event_count: number;
  last_event_at?: string | null;
}

export interface V2SignalResponse {
  signal_id: string;
  generated_at: string;
  product_id: string;
  venue: string;
  model_name: string;
  direction: 'buy' | 'sell' | 'hold';
  confidence: number;
  predicted_return: number;
  predicted_price: number;
  reasons: string[];
}

export interface V2PortfolioResponse {
  as_of: string;
  cash: number;
  position_qty: number;
  avg_entry_price: number;
  mark_price: number;
  realized_pnl: number;
  unrealized_pnl: number;
  equity: number;
  status: string;
}

export interface V2OrderResponse {
  order_id: string;
  decision_id: string;
  signal_id: string;
  created_at: string;
  side: string;
  quantity_btc: number;
  fill_price: number;
  status: string;
  filled_at?: string | null;
}

export interface V2OrdersResponse {
  orders: V2OrderResponse[];
}

export interface V2ControlResponse {
  status: string;
  control: {
    trading_paused: boolean;
    retrain_requested: boolean;
    last_acknowledged_alert?: string | null;
    updated_at: string;
  };
}

export interface V2SimulateResponse {
  signal: V2SignalResponse;
  decision: {
    decision_id: string;
    signal_id: string;
    decided_at: string;
    action: string;
    quantity_btc: number;
    reason: string;
    stale_data: boolean;
    trading_paused: boolean;
  };
  portfolio: V2PortfolioResponse;
  order?: V2OrderResponse | null;
}

export interface V2EventMessage {
  id?: string;
  type: string;
  data: Record<string, unknown>;
}

class V2ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'V2ApiError';
    this.status = status;
  }
}

function formatApiError(body: string, status: number): string {
  try {
    const parsed = JSON.parse(body) as { detail?: string };
    if (parsed.detail) return parsed.detail;
  } catch {
    // Ignore parse errors and fall back to a compact text string.
  }
  const compact = body.replace(/\s+/g, ' ').trim();
  return compact ? compact.slice(0, 220) : `HTTP ${status}`;
}

async function v2Fetch<T>(path: string, init?: RequestInit): Promise<T> {
  const baseUrl = await resolveV2BaseUrl();
  const response = await fetch(`${baseUrl}${path}`, {
    ...init,
    headers: requestHeaders(init),
  });
  if (!response.ok) {
    const body = await response.text().catch(() => '');
    throw new V2ApiError(formatApiError(body, response.status), response.status);
  }
  return (await response.json()) as T;
}

export const v2Api = {
  baseUrlHint: normalizeBaseUrl(EXPLICIT_V2_BASE_URL) ?? DEFAULT_V2_BASE_URL,
  resolveBaseUrl: resolveV2BaseUrl,
  health: () => v2Fetch<V2HealthResponse>('/v1/health'),
  portfolio: () => v2Fetch<V2PortfolioResponse>('/v1/portfolio'),
  latestSignal: () => v2Fetch<V2SignalResponse>('/v1/signals/latest'),
  orders: (limit = 12) => v2Fetch<V2OrdersResponse>(`/v1/orders?limit=${limit}`),
  pause: () => v2Fetch<V2ControlResponse>('/v1/control/pause', { method: 'POST' }),
  resume: () => v2Fetch<V2ControlResponse>('/v1/control/resume', { method: 'POST' }),
  retrain: () => v2Fetch<V2ControlResponse>('/v1/control/retrain', { method: 'POST' }),
  acknowledge: (message: string) =>
    v2Fetch<V2ControlResponse>('/v1/control/acknowledge', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    }),
  resetPaper: () =>
    v2Fetch<{ status: string; portfolio: V2PortfolioResponse }>('/v1/control/reset-paper', {
      method: 'POST',
    }),
  simulateCandle: (close: number) =>
    v2Fetch<V2SimulateResponse>('/v1/control/simulate-candle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        open: close - 180,
        high: close + 220,
        low: close - 260,
        close,
        volume: 12,
      }),
    }),
  syncMarket: () =>
    v2Fetch<V2SimulateResponse>('/v1/control/sync-market', {
      method: 'POST',
    }),
  streamUrl: async () => {
    const baseUrl = await resolveV2BaseUrl();
    return `${baseUrl}/v1/stream/events?token=${encodeURIComponent(V2_OPERATOR_TOKEN)}`;
  },
} as const;
