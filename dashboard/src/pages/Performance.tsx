import { BarChart3, GaugeCircle, Trophy } from 'lucide-react';
import { EquityCurveChart } from '../components/EquityCurveChart.tsx';
import {
  formatPct,
  formatTimestamp,
  formatUsd,
  toPaperCockpitViewModel,
} from '../api/paperViewModel.ts';
import type { V2PaperCockpitResponse, V2PaperPerformanceResponse } from '../api/v2Client.ts';
import { v2Api } from '../api/v2Client.ts';
import { useApi } from '../hooks/useApi.ts';
import { usePolling } from '../hooks/usePolling.ts';
import styles from './Performance.module.css';

const REFRESH_MS = 12_000;

interface PerformanceData {
  paper: V2PaperCockpitResponse;
  performance: V2PaperPerformanceResponse;
}

async function loadPerformance(): Promise<PerformanceData> {
  const [paper, performance] = await Promise.all([v2Api.paper(), v2Api.performance()]);
  return { paper, performance };
}

export function Performance() {
  const summary = useApi(loadPerformance, []);

  usePolling(() => {
    summary.refetch();
  }, REFRESH_MS);

  if (summary.loading && !summary.data) {
    return <div className="emptyState">Resolving paper performance from the v2 API...</div>;
  }

  if (summary.error && !summary.data) {
    return <div className="errorState">{summary.error}</div>;
  }

  if (!summary.data) {
    return <div className="emptyState">Performance data is unavailable.</div>;
  }

  const { paper, performance } = summary.data;
  const model = toPaperCockpitViewModel(paper, performance);

  return (
    <div className={`pageStack ${styles.page}`}>
      <section className="pageHero">
        <span className="eyebrowLabel">outcome analysis</span>
        <div className="heroTitleRow">
          <div>
            <h2>Performance ledger</h2>
            <p className="lede">
              This page answers one question: did paper trading actually create value versus just
              owning BTC? The chart and the scorecards reinforce that message in text.
            </p>
          </div>
        </div>
        <div className="statusStrip" role="status" aria-live="polite">
          <span className="statusPill" data-tone={model.benchmarkTone}>
            <Trophy size={14} />
            {model.benchmarkLabel}
          </span>
          <span className="statusPill" data-tone={model.freshness.tone}>
            <GaugeCircle size={14} />
            {model.freshness.detail}
          </span>
          <span className="statusPill" data-tone="neutral">
            <BarChart3 size={14} />
            {model.tradeOutcomeLabel}
          </span>
        </div>
      </section>

      <section className="metricGrid">
        <article className="metricCard">
          <span className="metricLabel">Net PnL</span>
          <span className="metricValue">{formatUsd(performance.net_pnl)}</span>
          <span className="metricDetail">{formatPct(performance.net_pnl_pct)}</span>
        </article>
        <article className="metricCard">
          <span className="metricLabel">Win rate</span>
          <span className="metricValue">{formatPct(performance.win_rate)}</span>
          <span className="metricDetail">{performance.closed_trade_count} closed trades</span>
        </article>
        <article className="metricCard">
          <span className="metricLabel">Expectancy per trade</span>
          <span className="metricValue">{formatUsd(performance.expectancy_per_trade)}</span>
          <span className="metricDetail">{formatUsd(performance.turnover_usd)} turnover</span>
        </article>
        <article className="metricCard">
          <span className="metricLabel">Max drawdown</span>
          <span className="metricValue">{formatPct(performance.max_drawdown_pct)}</span>
          <span className="metricDetail">{formatPct(performance.exposure_pct)} exposure</span>
        </article>
      </section>

      <section className="surfaceCard">
        <EquityCurveChart
          points={paper.equity_curve}
          title="Paper equity over time"
          message={`Key message: the paper account is ${model.benchmarkLabel.toLowerCase()} and closed ${formatTimestamp(performance.as_of)} at ${formatUsd(performance.equity)}.`}
        />
      </section>

      <section className="panelGrid">
        <article className="surfaceCard">
          <div className="surfaceHeader">
            <div>
              <h3>Benchmark and risk</h3>
              <p>Every performance claim remains grounded in a comparable baseline.</p>
            </div>
          </div>
          <div className="kvGrid">
            <div className="kvItem">
              <span className="kvLabel">Benchmark equity</span>
              <span className="kvValue">{formatUsd(performance.benchmark_equity)}</span>
            </div>
            <div className="kvItem">
              <span className="kvLabel">Benchmark return</span>
              <span className="kvValue">{formatPct(performance.benchmark_return_pct)}</span>
            </div>
            <div className="kvItem">
              <span className="kvLabel">Fees paid</span>
              <span className="kvValue">{formatUsd(performance.fees_paid)}</span>
            </div>
            <div className="kvItem">
              <span className="kvLabel">Trade count</span>
              <span className="kvValue">{performance.trade_count}</span>
            </div>
          </div>
        </article>

        <article className="surfaceCard">
          <div className="surfaceHeader">
            <div>
              <h3>Closed trade outcomes</h3>
              <p>Text table for accessible review; the chart is not the only source.</p>
            </div>
          </div>
          {paper.closed_trades.length > 0 ? (
            <div className="tableScroll">
              <table className="ledgerTable">
                <thead>
                  <tr>
                    <th>Closed</th>
                    <th>Net PnL</th>
                    <th>Return</th>
                  </tr>
                </thead>
                <tbody>
                  {paper.closed_trades.map((trade) => (
                    <tr key={`${trade.closed_at}-${trade.exit_price}`}>
                      <td>{formatTimestamp(trade.closed_at)}</td>
                      <td>{formatUsd(trade.net_pnl)}</td>
                      <td>{formatPct(trade.return_pct)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="emptyState">No closed paper trades are available yet.</div>
          )}
        </article>
      </section>
    </div>
  );
}
