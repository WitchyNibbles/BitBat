const V2_BASE_URL =
  (import.meta.env.VITE_V2_API_URL as string | undefined) ?? 'http://localhost:8100';
const V2_OPERATOR_TOKEN =
  (import.meta.env.VITE_V2_OPERATOR_TOKEN as string | undefined) ?? '';

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
  const response = await fetch(`${V2_BASE_URL}${path}`, {
    ...init,
    headers: {
      ...(init?.headers ?? {}),
      ...(V2_OPERATOR_TOKEN ? { 'X-BitBat-Operator-Token': V2_OPERATOR_TOKEN } : {}),
    },
  });
  if (!response.ok) {
    const body = await response.text().catch(() => '');
    throw new V2ApiError(formatApiError(body, response.status), response.status);
  }
  return (await response.json()) as T;
}

export const v2Api = {
  baseUrl: V2_BASE_URL,
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
  streamUrl: `${V2_BASE_URL}/v1/stream/events?token=${encodeURIComponent(V2_OPERATOR_TOKEN)}`,
} as const;
