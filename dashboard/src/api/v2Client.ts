const EXPLICIT_V2_BASE_URL = (import.meta.env.VITE_V2_API_URL as string | undefined) ?? '';
const V2_OPERATOR_TOKEN =
  (import.meta.env.VITE_V2_OPERATOR_TOKEN as string | undefined) ?? '';
const V2_BASE_URL_CACHE_KEY = 'bitbat.v2ApiBaseUrl';
const V2_OPERATOR_TOKEN_CACHE_KEY = 'bitbat.v2OperatorToken';
const DEFAULT_V2_BASE_URL = 'http://localhost:8100';
const DEFAULT_LOCAL_DEV_OPERATOR_TOKEN = 'bitbat-local-dev-token';

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

function localTokenFallback(): string {
  if (typeof window === 'undefined') {
    return '';
  }

  const hostname = window.location.hostname.trim().toLowerCase();
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    return DEFAULT_LOCAL_DEV_OPERATOR_TOKEN;
  }

  return '';
}

export function resolveV2OperatorToken(): string {
  const explicit = String(V2_OPERATOR_TOKEN).trim();
  if (explicit) {
    return explicit;
  }

  if (typeof window !== 'undefined') {
    const cached = String(window.sessionStorage.getItem(V2_OPERATOR_TOKEN_CACHE_KEY) ?? '').trim();
    if (cached) {
      return cached;
    }
  }

  return localTokenFallback();
}

export function setV2OperatorToken(token: string): void {
  if (typeof window === 'undefined') {
    return;
  }

  const normalized = token.trim();
  if (normalized) {
    window.sessionStorage.setItem(V2_OPERATOR_TOKEN_CACHE_KEY, normalized);
  } else {
    window.sessionStorage.removeItem(V2_OPERATOR_TOKEN_CACHE_KEY);
  }
}

function requestHeaders(init?: RequestInit): HeadersInit {
  const operatorToken = resolveV2OperatorToken();
  return {
    ...(init?.headers ?? {}),
    ...(operatorToken ? { 'X-BitBat-Operator-Token': operatorToken } : {}),
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
    if (response.status === 401 || response.status === 403) {
      return true;
    }
    if (response.status !== 200) {
      return false;
    }

    const contentType = response.headers.get('content-type') ?? '';
    if (!contentType.includes('application/json')) {
      return false;
    }

    const body = (await response.json().catch(() => null)) as Record<string, unknown> | null;
    return body != null && typeof body.status === 'string';
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

export function resetV2BaseUrlResolution(): void {
  resolvedBaseUrlPromise = null;
  if (typeof window !== 'undefined') {
    window.sessionStorage.removeItem(V2_BASE_URL_CACHE_KEY);
  }
}

export interface V2HealthResponse {
  status: string;
  venue: string;
  product_id: string;
  trading_paused: boolean;
  event_count: number;
  last_event_at?: string | null;
  autorun: {
    enabled: boolean;
    interval_seconds: number;
    running: boolean;
    last_cycle_status?: string | null;
    last_cycle_started_at?: string | null;
    last_cycle_completed_at?: string | null;
    last_error?: string | null;
    last_processed_candle_start?: string | null;
    last_action?: string | null;
  };
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

export interface V2PaperAlertResponse {
  occurred_at: string;
  code: string;
  message: string;
}

export interface V2PaperPerformancePointResponse {
  occurred_at: string;
  equity: number;
  cash: number;
  position_qty: number;
  mark_price: number;
  realized_pnl: number;
  unrealized_pnl: number;
}

export interface V2PaperTradeResponse {
  closed_at: string;
  quantity_btc: number;
  entry_price: number;
  exit_price: number;
  gross_pnl: number;
  net_pnl: number;
  fees_paid: number;
  return_pct: number;
}

export interface V2PaperPerformanceResponse {
  as_of: string;
  starting_cash: number;
  equity: number;
  cash: number;
  position_qty: number;
  mark_price: number;
  realized_pnl: number;
  unrealized_pnl: number;
  net_pnl: number;
  net_pnl_pct: number;
  fees_paid: number;
  turnover_usd: number;
  trade_count: number;
  closed_trade_count: number;
  win_rate: number;
  expectancy_per_trade: number;
  max_drawdown_pct: number;
  exposure_pct: number;
  benchmark_equity: number;
  benchmark_return_pct: number;
  alpha_vs_buy_hold: number;
  last_signal_at?: string | null;
  last_signal_direction?: string | null;
  signal_confidence?: number | null;
  last_event_at?: string | null;
}

export interface V2PaperOrderResponse {
  order_id: string;
  created_at: string;
  filled_at?: string | null;
  side: string;
  quantity_btc: number;
  fill_price: number;
  status: string;
  notional_usd: number;
}

export interface V2PaperCockpitResponse {
  portfolio: V2PortfolioResponse;
  performance: V2PaperPerformanceResponse;
  latest_signal?: V2SignalResponse | null;
  recent_orders: V2PaperOrderResponse[];
  recent_alerts: V2PaperAlertResponse[];
  equity_curve: V2PaperPerformancePointResponse[];
  closed_trades: V2PaperTradeResponse[];
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

export class V2ApiError extends Error {
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
    if (parsed.detail) {
      return parsed.detail;
    }
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
  paper: () => v2Fetch<V2PaperCockpitResponse>('/v1/paper'),
  performance: () => v2Fetch<V2PaperPerformanceResponse>('/v1/performance'),
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
    return `${baseUrl}/v1/stream/events?token=${encodeURIComponent(resolveV2OperatorToken())}`;
  },
} as const;
