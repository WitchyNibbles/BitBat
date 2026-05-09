import type {
  V2PaperCockpitResponse,
  V2PaperPerformanceResponse,
  V2SignalResponse,
} from './v2Client';

export type Tone = 'positive' | 'warning' | 'critical' | 'neutral';

const USD_FORMATTER = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  maximumFractionDigits: 2,
});

const PERCENT_FORMATTER = new Intl.NumberFormat('en-US', {
  style: 'percent',
  maximumFractionDigits: 2,
  minimumFractionDigits: 2,
});

export interface FreshnessState {
  tone: Tone;
  label: string;
  detail: string;
  stale: boolean;
}

export interface SignalState {
  tone: Tone;
  label: string;
  detail: string;
}

export interface PaperCockpitViewModel {
  freshness: FreshnessState;
  signal: SignalState;
  benchmarkTone: Tone;
  benchmarkLabel: string;
  exposureLabel: string;
  tradeOutcomeLabel: string;
}

export function formatUsd(value: number): string {
  return USD_FORMATTER.format(value);
}

export function formatPct(value: number): string {
  return PERCENT_FORMATTER.format(value);
}

export function formatBtc(value: number): string {
  return `${value.toFixed(4)} BTC`;
}

export function formatNumber(value: number): string {
  return new Intl.NumberFormat('en-US', {
    maximumFractionDigits: 2,
  }).format(value);
}

export function formatTimestamp(value?: string | null): string {
  if (!value) {
    return 'No timestamp available';
  }

  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }

  return parsed.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

export function deriveFreshnessState(
  value?: string | null,
  now = new Date(),
  staleAfterMs = 5 * 60 * 1000,
): FreshnessState {
  if (!value) {
    return {
      tone: 'warning',
      label: 'No runtime event yet',
      detail: 'The UI cannot prove freshness until the first paper event is recorded.',
      stale: true,
    };
  }

  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return {
      tone: 'warning',
      label: 'Freshness timestamp unreadable',
      detail: value,
      stale: true,
    };
  }

  const ageMs = now.getTime() - parsed.getTime();
  if (ageMs <= staleAfterMs) {
    return {
      tone: 'positive',
      label: 'Fresh data',
      detail: `Last runtime event ${formatTimestamp(value)}`,
      stale: false,
    };
  }

  return {
    tone: ageMs <= staleAfterMs * 3 ? 'warning' : 'critical',
    label: 'Stale runtime',
    detail: `Last runtime event ${formatTimestamp(value)}`,
    stale: true,
  };
}

export function deriveSignalState(signal?: V2SignalResponse | null): SignalState {
  if (!signal) {
    return {
      tone: 'neutral',
      label: 'No signal yet',
      detail: 'The runtime has not emitted a paper-trading signal in this session.',
    };
  }

  if (signal.direction === 'buy') {
    return {
      tone: 'positive',
      label: 'Bias: buy',
      detail: `${formatPct(signal.predicted_return)} projected move at ${formatTimestamp(signal.generated_at)}`,
    };
  }

  if (signal.direction === 'sell') {
    return {
      tone: 'critical',
      label: 'Bias: sell',
      detail: `${formatPct(signal.predicted_return)} projected move at ${formatTimestamp(signal.generated_at)}`,
    };
  }

  return {
    tone: 'warning',
    label: 'Bias: hold',
    detail: `Hold decision emitted at ${formatTimestamp(signal.generated_at)}`,
  };
}

export function toPaperCockpitViewModel(
  cockpit: V2PaperCockpitResponse,
  performance: V2PaperPerformanceResponse = cockpit.performance,
  now = new Date(),
): PaperCockpitViewModel {
  const freshness = deriveFreshnessState(
    performance.last_event_at ?? cockpit.portfolio.as_of,
    now,
  );
  const signal = deriveSignalState(cockpit.latest_signal);
  const alpha = performance.alpha_vs_buy_hold;
  const benchmarkTone: Tone = alpha > 0 ? 'positive' : alpha < 0 ? 'critical' : 'neutral';
  const closedTrades = performance.closed_trade_count;
  const tradeOutcomeLabel =
    closedTrades > 0
      ? `${closedTrades} closed trades, ${formatPct(performance.win_rate)} win rate`
      : 'No closed trades yet';

  return {
    freshness,
    signal,
    benchmarkTone,
    benchmarkLabel:
      alpha === 0
        ? 'Tracking buy-and-hold'
        : `${alpha > 0 ? 'Ahead of' : 'Behind'} buy-and-hold by ${formatPct(Math.abs(alpha))}`,
    exposureLabel:
      performance.position_qty > 0
        ? `${formatPct(performance.exposure_pct)} of equity deployed`
        : 'Flat account, no BTC exposure',
    tradeOutcomeLabel,
  };
}
