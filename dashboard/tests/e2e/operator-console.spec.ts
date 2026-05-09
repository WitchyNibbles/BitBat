import { expect, test } from '@playwright/test';

const samplePaper = {
  portfolio: {
    as_of: '2026-05-09T10:00:00Z',
    cash: 9500,
    position_qty: 0.01,
    avg_entry_price: 98000,
    mark_price: 101000,
    realized_pnl: 120,
    unrealized_pnl: 30,
    equity: 10150,
    status: 'paper',
  },
  performance: {
    as_of: '2026-05-09T10:00:00Z',
    starting_cash: 10000,
    equity: 10150,
    cash: 9500,
    position_qty: 0.01,
    mark_price: 101000,
    realized_pnl: 120,
    unrealized_pnl: 30,
    net_pnl: 150,
    net_pnl_pct: 0.015,
    fees_paid: 4,
    turnover_usd: 2100,
    trade_count: 2,
    closed_trade_count: 1,
    win_rate: 1,
    expectancy_per_trade: 150,
    max_drawdown_pct: 0.01,
    exposure_pct: 0.12,
    benchmark_equity: 10040,
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
    predicted_price: 101500,
    reasons: ['score=0.73', 'trend confirmed'],
  },
  recent_orders: [
    {
      order_id: 'ord-1',
      created_at: '2026-05-09T09:59:20Z',
      filled_at: '2026-05-09T09:59:21Z',
      side: 'buy',
      quantity_btc: 0.01,
      fill_price: 100950,
      status: 'filled',
      notional_usd: 1009.5,
    },
  ],
  recent_alerts: [
    {
      occurred_at: '2026-05-09T09:58:00Z',
      code: 'paper-demo',
      message: 'Simulated paper alert for verification.',
    },
  ],
  equity_curve: [
    {
      occurred_at: '2026-05-09T09:30:00Z',
      equity: 10000,
      cash: 10000,
      position_qty: 0,
      mark_price: 100000,
      realized_pnl: 0,
      unrealized_pnl: 0,
    },
    {
      occurred_at: '2026-05-09T10:00:00Z',
      equity: 10150,
      cash: 9500,
      position_qty: 0.01,
      mark_price: 101000,
      realized_pnl: 120,
      unrealized_pnl: 30,
    },
  ],
  closed_trades: [
    {
      closed_at: '2026-05-09T09:55:00Z',
      quantity_btc: 0.01,
      entry_price: 99500,
      exit_price: 100900,
      gross_pnl: 14,
      net_pnl: 12,
      fees_paid: 2,
      return_pct: 0.012,
    },
  ],
};

async function mockApis(page: import('@playwright/test').Page, mode: 'happy' | 'empty' | 'auth') {
  await page.route('**/v1/**', async (route) => {
    const url = route.request().url();
    if (mode === 'auth') {
      await route.fulfill({ status: 401, body: JSON.stringify({ detail: 'Invalid operator token.' }) });
      return;
    }
    if (url.includes('/v1/health')) {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'ok',
          venue: 'coinbase',
          product_id: 'BTC-USD',
          trading_paused: false,
          event_count: 14,
          last_event_at: samplePaper.performance.last_event_at,
          autorun: { enabled: false, interval_seconds: 15, running: false },
        }),
      });
      return;
    }
    if (url.includes('/v1/paper')) {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(
          mode === 'empty'
            ? {
                ...samplePaper,
                latest_signal: null,
                recent_orders: [],
                recent_alerts: [],
                closed_trades: [],
              }
            : samplePaper,
        ),
      });
      return;
    }
    if (url.includes('/v1/performance')) {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(samplePaper.performance),
      });
      return;
    }
    await route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
  });

  await page.route('**/health/detailed', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'ok',
        version: 'test',
        uptime_seconds: 100,
        services: [],
        schema_readiness: { compatibility_state: 'compatible', is_compatible: true },
      }),
    });
  });

  await page.route('**/analytics/status**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        database_ok: true,
        database_present: true,
        model_exists: true,
        dataset_exists: true,
        total_predictions: 12,
        last_prediction_time: '2026-05-09T09:50:00Z',
      }),
    });
  });

  await page.route('**/system/logs**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        logs: [
          {
            timestamp: '2026-05-09T09:58:00Z',
            level: 'info',
            message: 'System log line',
            service: 'runtime',
          },
        ],
        total: 1,
      }),
    });
  });

  await page.route('**/system/ingestion-status', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ prices: 'fresh', news: 'fresh' }),
    });
  });

  await page.route('**/system/retraining-events**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ events: [], total: 0 }),
    });
  });

  await page.route('**/settings', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        preset: 'balanced',
        freq: '1h',
        horizon: '4h',
        tau: 0.005,
        enter_threshold: 0.7,
        valid_freqs: ['1h'],
        valid_horizons: ['4h'],
      }),
    });
  });
}

test('renders the paper cockpit and navigates across key routes', async ({ page }) => {
  await mockApis(page, 'happy');
  await page.goto('/paper-trade');

  await expect(page.getByRole('heading', { name: /paper trade cockpit/i })).toBeVisible();
  await expect(page.getByText(/recent orders/i)).toBeVisible();
  await expect(page.getByText(/simulated paper alert/i)).toBeVisible();

  await page.getByRole('link', { name: /performance/i }).click();
  await expect(page.getByRole('heading', { name: /performance ledger/i })).toBeVisible();

  await page.getByRole('link', { name: /^System/ }).click();
  await expect(page.getByRole('heading', { name: /system health/i })).toBeVisible();
});

test('shows explicit empty states when no paper activity exists yet', async ({ page }) => {
  await mockApis(page, 'empty');
  await page.goto('/paper-trade');

  await expect(page.getByText(/no paper orders yet/i)).toBeVisible();
  await expect(page.getByText(/no closed trades yet/i)).toBeVisible();
  await expect(page.getByText(/no paper-trade alerts yet/i)).toBeVisible();
});

test('shows a clear auth failure for invalid operator tokens', async ({ page }) => {
  await mockApis(page, 'auth');
  await page.goto('/paper-trade');

  await expect(page.getByText(/invalid operator token/i)).toBeVisible();
});
