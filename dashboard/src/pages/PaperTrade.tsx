import { AlertTriangle, CandlestickChart, Clock3, Wallet } from 'lucide-react';
import { EquityCurveChart } from '../components/EquityCurveChart.tsx';
import {
  formatBtc,
  formatPct,
  formatTimestamp,
  formatUsd,
  toPaperCockpitViewModel,
} from '../api/paperViewModel.ts';
import type { V2PaperCockpitResponse, V2PaperPerformanceResponse } from '../api/v2Client.ts';
import { V2ApiError, v2Api } from '../api/v2Client.ts';
import { useApi } from '../hooks/useApi.ts';
import { usePolling } from '../hooks/usePolling.ts';
import styles from './PaperTrade.module.css';

const REFRESH_MS = 10_000;

interface PaperPageData {
  paper: V2PaperCockpitResponse;
  performance: V2PaperPerformanceResponse;
}

function toMessage(error: unknown): string {
  if (error instanceof V2ApiError && error.status === 401) {
    return 'Invalid operator token. Set VITE_V2_OPERATOR_TOKEN for the React dashboard.';
  }
  return error instanceof Error ? error.message : String(error);
}

async function loadPaperTrade(): Promise<PaperPageData> {
  const [paper, performance] = await Promise.all([v2Api.paper(), v2Api.performance()]);
  return { paper, performance };
}

export function PaperTrade() {
  const paperQuery = useApi(loadPaperTrade, []);

  usePolling(() => {
    paperQuery.refetch();
  }, REFRESH_MS);

  if (paperQuery.loading && !paperQuery.data) {
    return <div className="emptyState">Resolving the paper-trade cockpit from the v2 API...</div>;
  }

  if (paperQuery.error && !paperQuery.data) {
    return <div className="errorState">{toMessage(paperQuery.error)}</div>;
  }

  if (!paperQuery.data) {
    return <div className="emptyState">Paper-trade data is unavailable.</div>;
  }

  const { paper, performance } = paperQuery.data;
  const model = toPaperCockpitViewModel(paper, performance);
  const signal = paper.latest_signal;

  return (
    <div className={`pageStack ${styles.page}`}>
      <section className="pageHero">
        <span className="eyebrowLabel">simulated ledger</span>
        <div className="heroTitleRow">
          <div>
            <h2>Paper Trade cockpit</h2>
            <p className="lede">
              This route is the canonical paper-only account surface. It shows account mark,
              benchmark delta, fills, closed trades, and freshness without forcing the operator to
              reconstruct state from multiple endpoints.
            </p>
          </div>
        </div>
        <div className="statusStrip" role="status" aria-live="polite">
          <span className="statusPill" data-tone={model.freshness.tone}>
            <Clock3 size={14} />
            {model.freshness.label}
          </span>
          <span className="statusPill" data-tone={model.benchmarkTone}>
            <CandlestickChart size={14} />
            {model.benchmarkLabel}
          </span>
          <span className="statusPill" data-tone={signal ? model.signal.tone : 'neutral'}>
            <Wallet size={14} />
            {signal ? model.signal.label : 'No signal yet'}
          </span>
          <span className="statusPill" data-tone="neutral">
            <AlertTriangle size={14} />
            Paper labels stay explicit on every panel
          </span>
        </div>
      </section>

      <section className="metricGrid" aria-label="Paper-trade metrics">
        <article className="metricCard">
          <span className="metricLabel">Paper equity</span>
          <span className="metricValue">{formatUsd(performance.equity)}</span>
          <span className="metricDetail">As of {formatTimestamp(performance.as_of)}</span>
        </article>
        <article className="metricCard">
          <span className="metricLabel">Cash</span>
          <span className="metricValue">{formatUsd(performance.cash)}</span>
          <span className="metricDetail">{model.exposureLabel}</span>
        </article>
        <article className="metricCard">
          <span className="metricLabel">Net PnL</span>
          <span className="metricValue">{formatUsd(performance.net_pnl)}</span>
          <span className="metricDetail">{formatPct(performance.net_pnl_pct)}</span>
        </article>
        <article className="metricCard">
          <span className="metricLabel">Buy-and-hold delta</span>
          <span className="metricValue">{formatPct(performance.alpha_vs_buy_hold)}</span>
          <span className="metricDetail">
            Benchmark equity {formatUsd(performance.benchmark_equity)}
          </span>
        </article>
      </section>

      <section className="panelGrid">
        <article className="surfaceCard">
          <div className="surfaceHeader">
            <div>
              <h3>Portfolio state</h3>
              <p>Directly from `/v1/paper` and `/v1/performance`.</p>
            </div>
          </div>
          <div className="kvGrid">
            <div className="kvItem">
              <span className="kvLabel">Position</span>
              <span className="kvValue">{formatBtc(performance.position_qty)}</span>
            </div>
            <div className="kvItem">
              <span className="kvLabel">Avg entry</span>
              <span className="kvValue">{formatUsd(paper.portfolio.avg_entry_price)}</span>
            </div>
            <div className="kvItem">
              <span className="kvLabel">Mark price</span>
              <span className="kvValue">{formatUsd(performance.mark_price)}</span>
            </div>
            <div className="kvItem">
              <span className="kvLabel">Realized PnL</span>
              <span className="kvValue">{formatUsd(performance.realized_pnl)}</span>
            </div>
            <div className="kvItem">
              <span className="kvLabel">Unrealized PnL</span>
              <span className="kvValue">{formatUsd(performance.unrealized_pnl)}</span>
            </div>
            <div className="kvItem">
              <span className="kvLabel">Fees paid</span>
              <span className="kvValue">{formatUsd(performance.fees_paid)}</span>
            </div>
          </div>
        </article>

        <article className="surfaceCard">
          <div className="surfaceHeader">
            <div>
              <h3>Signal provenance</h3>
              <p>Prediction, not certainty. Order and fill remain separate concepts.</p>
            </div>
          </div>
          {signal ? (
            <div className="stackList">
              <div className="kvGrid">
                <div className="kvItem">
                  <span className="kvLabel">Direction</span>
                  <span className="kvValue">{signal.direction}</span>
                </div>
                <div className="kvItem">
                  <span className="kvLabel">Confidence</span>
                  <span className="kvValue">{formatPct(signal.confidence)}</span>
                </div>
                <div className="kvItem">
                  <span className="kvLabel">Predicted return</span>
                  <span className="kvValue">{formatPct(signal.predicted_return)}</span>
                </div>
                <div className="kvItem">
                  <span className="kvLabel">Predicted price</span>
                  <span className="kvValue">{formatUsd(signal.predicted_price)}</span>
                </div>
              </div>
              <ul className="plainList">
                {signal.reasons.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
            </div>
          ) : (
            <div className="emptyState">
              No signal has been generated yet. The account state can still be read, but signal
              provenance is empty.
            </div>
          )}
        </article>
      </section>

      <section className="surfaceCard">
        <EquityCurveChart
          points={paper.equity_curve}
          title="Equity curve"
          message={`Key message: paper equity is ${formatUsd(performance.equity)} and ${model.benchmarkLabel.toLowerCase()}.`}
        />
      </section>

      <section className="panelGrid">
        <article className="surfaceCard">
          <div className="surfaceHeader">
            <div>
              <h3>Recent orders</h3>
              <p>Intent and fill history without hidden state reconstruction.</p>
            </div>
          </div>
          {paper.recent_orders.length > 0 ? (
            <div className="tableScroll">
              <table className="ledgerTable">
                <thead>
                  <tr>
                    <th>Created</th>
                    <th>Side</th>
                    <th>Qty</th>
                    <th>Fill</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {paper.recent_orders.map((order) => (
                    <tr key={order.order_id}>
                      <td>{formatTimestamp(order.created_at)}</td>
                      <td>{order.side}</td>
                      <td>{formatBtc(order.quantity_btc)}</td>
                      <td>{formatUsd(order.fill_price)}</td>
                      <td>{order.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="emptyState">No paper orders yet.</div>
          )}
        </article>

        <article className="surfaceCard">
          <div className="surfaceHeader">
            <div>
              <h3>Closed trades</h3>
              <p>Outcome ledger with exact timestamps and fee-aware net PnL.</p>
            </div>
          </div>
          {paper.closed_trades.length > 0 ? (
            <div className="tableScroll">
              <table className="ledgerTable">
                <thead>
                  <tr>
                    <th>Closed</th>
                    <th>Entry</th>
                    <th>Exit</th>
                    <th>Net PnL</th>
                    <th>Return</th>
                  </tr>
                </thead>
                <tbody>
                  {paper.closed_trades.map((trade) => (
                    <tr key={`${trade.closed_at}-${trade.exit_price}`}>
                      <td>{formatTimestamp(trade.closed_at)}</td>
                      <td>{formatUsd(trade.entry_price)}</td>
                      <td>{formatUsd(trade.exit_price)}</td>
                      <td>{formatUsd(trade.net_pnl)}</td>
                      <td>{formatPct(trade.return_pct)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="emptyState">No closed trades yet.</div>
          )}
        </article>
      </section>

      <section className="surfaceCard">
        <div className="surfaceHeader">
          <div>
            <h3>Alerts and journal</h3>
            <p>Use `role="log"` semantics because this is a sequential event feed.</p>
          </div>
        </div>
        {paper.recent_alerts.length > 0 ? (
          <div className="logList" role="log" aria-live="polite">
            {paper.recent_alerts.map((alert) => (
              <div key={`${alert.occurred_at}-${alert.code}`} className="logEntry">
                <div className="logMeta">
                  <span>{alert.code}</span>
                  <span>{formatTimestamp(alert.occurred_at)}</span>
                </div>
                <div>{alert.message}</div>
              </div>
            ))}
          </div>
        ) : (
          <div className="emptyState">No paper-trade alerts yet.</div>
        )}
      </section>
    </div>
  );
}
