import { render, screen, waitFor } from '@testing-library/react';
import { PaperTrade } from './PaperTrade';
import type { V2PaperCockpitResponse } from '../api/v2Client';
import { v2Api } from '../api/v2Client';

vi.mock('../api/v2Client', () => ({
  v2Api: {
    paper: vi.fn(),
    performance: vi.fn(),
  },
  V2ApiError: class V2ApiError extends Error {
    status: number;

    constructor(message: string, status: number) {
      super(message);
      this.status = status;
    }
  },
}));

const mockedPaper: V2PaperCockpitResponse = {
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
  recent_orders: [
    {
      order_id: 'ord-1',
      created_at: '2026-05-09T09:59:20Z',
      filled_at: '2026-05-09T09:59:21Z',
      side: 'buy',
      quantity_btc: 0.01,
      fill_price: 100_950,
      status: 'filled',
      notional_usd: 1_009.5,
    },
  ],
  recent_alerts: [],
  equity_curve: [
    {
      occurred_at: '2026-05-09T09:30:00Z',
      equity: 10_000,
      cash: 10_000,
      position_qty: 0,
      mark_price: 100_000,
      realized_pnl: 0,
      unrealized_pnl: 0,
    },
  ],
  closed_trades: [],
};

describe('PaperTrade', () => {
  it('renders the paper-only cockpit with order history', async () => {
    vi.mocked(v2Api.paper).mockResolvedValue(mockedPaper);
    vi.mocked(v2Api.performance).mockResolvedValue(mockedPaper.performance);

    render(<PaperTrade />);

    await waitFor(() => {
      expect(screen.getByText(/paper trade cockpit/i)).toBeInTheDocument();
    });

    expect(screen.getByText(/recent orders/i)).toBeInTheDocument();
    expect(screen.getByText(/buy-and-hold delta/i)).toBeInTheDocument();
    expect(screen.getByText(/bias: buy/i)).toBeInTheDocument();
  });
});
