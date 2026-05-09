import {
  deriveFreshnessState,
  deriveSignalState,
  toPaperCockpitViewModel,
} from './paperViewModel';
import type { V2PaperCockpitResponse } from './v2Client';

const samplePaper: V2PaperCockpitResponse = {
  portfolio: {
    as_of: '2026-05-09T10:00:00Z',
    cash: 9_500,
    position_qty: 0.01,
    avg_entry_price: 98_000,
    mark_price: 101_000,
    realized_pnl: 120,
    unrealized_pnl: 30,
    equity: 10_150,
    status: 'paper',
  },
  performance: {
    as_of: '2026-05-09T10:00:00Z',
    starting_cash: 10_000,
    equity: 10_150,
    cash: 9_500,
    position_qty: 0.01,
    mark_price: 101_000,
    realized_pnl: 120,
    unrealized_pnl: 30,
    net_pnl: 150,
    net_pnl_pct: 0.015,
    fees_paid: 4,
    turnover_usd: 2_100,
    trade_count: 2,
    closed_trade_count: 1,
    win_rate: 1,
    expectancy_per_trade: 150,
    max_drawdown_pct: 0.01,
    exposure_pct: 0.12,
    benchmark_equity: 10_040,
    benchmark_return_pct: 0.004,
    alpha_vs_buy_hold: 0.011,
    last_signal_at: '2026-05-09T09:59:00Z',
    last_signal_direction: 'buy',
    signal_confidence: 0.73,
    last_event_at: '2026-05-09T09:59:30Z',
  },
  latest_signal: {
    signal_id: 'sig-1',
    generated_at: '2026-05-09T09:59:00Z',
    product_id: 'BTC-USD',
    venue: 'coinbase',
    model_name: 'ritual-momentum-v1',
    direction: 'buy',
    confidence: 0.73,
    predicted_return: 0.014,
    predicted_price: 101_500,
    reasons: ['score=0.73', 'trend confirmed'],
  },
  recent_orders: [],
  recent_alerts: [],
  equity_curve: [],
  closed_trades: [],
};

describe('paperViewModel', () => {
  it('marks recent timestamps as fresh', () => {
    const state = deriveFreshnessState(
      '2026-05-09T09:59:30Z',
      new Date('2026-05-09T10:00:00Z'),
    );

    expect(state.stale).toBe(false);
    expect(state.tone).toBe('positive');
  });

  it('marks missing signals as neutral', () => {
    const signal = deriveSignalState(null);

    expect(signal.tone).toBe('neutral');
    expect(signal.label).toMatch(/no signal/i);
  });

  it('builds a benchmark-aware cockpit summary', () => {
    const viewModel = toPaperCockpitViewModel(
      samplePaper,
      samplePaper.performance,
      new Date('2026-05-09T10:00:00Z'),
    );

    expect(viewModel.benchmarkTone).toBe('positive');
    expect(viewModel.benchmarkLabel).toMatch(/ahead of buy-and-hold/i);
    expect(viewModel.signal.label).toMatch(/buy/i);
  });
});
