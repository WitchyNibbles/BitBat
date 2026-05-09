import { Aperture, RefreshCcw, ShieldCheck, Waves } from 'lucide-react';
import type { IngestionStatus, SystemLogEntry, SystemStatusResponse } from '../api/client.ts';
import { api } from '../api/client.ts';
import {
  formatBtc,
  formatPct,
  formatTimestamp,
  formatUsd,
  toPaperCockpitViewModel,
} from '../api/paperViewModel.ts';
import type { V2HealthResponse, V2PaperCockpitResponse } from '../api/v2Client.ts';
import { v2Api } from '../api/v2Client.ts';
import { useApi } from '../hooks/useApi.ts';
import { usePolling } from '../hooks/usePolling.ts';
import styles from './Home.module.css';

const REFRESH_MS = 10_000;

interface CommandCenterData {
  health: V2HealthResponse;
  paper: V2PaperCockpitResponse;
  ingestion: IngestionStatus | null;
  system: SystemStatusResponse | null;
  logs: SystemLogEntry[];
}

function toMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function loadCommandCenter(): Promise<CommandCenterData> {
  const [healthResult, paperResult, ingestionResult, systemResult, logsResult] =
    await Promise.allSettled([
      v2Api.health(),
      v2Api.paper(),
      api.ingestionStatus(),
      api.systemStatus(),
      api.systemLogs(6),
    ]);

  if (healthResult.status === 'rejected') {
    throw new Error(toMessage(healthResult.reason));
  }
  if (paperResult.status === 'rejected') {
    throw new Error(toMessage(paperResult.reason));
  }

  return {
    health: healthResult.value,
    paper: paperResult.value,
    ingestion: ingestionResult.status === 'fulfilled' ? ingestionResult.value : null,
    system: systemResult.status === 'fulfilled' ? systemResult.value : null,
    logs: logsResult.status === 'fulfilled' ? logsResult.value.logs : [],
  };
}

export function Home() {
  const overview = useApi(loadCommandCenter, []);

  usePolling(() => {
    overview.refetch();
  }, REFRESH_MS);

  if (overview.loading && !overview.data) {
    return <div className="emptyState">Resolving the operator overview from the v2 runtime...</div>;
  }

  if (overview.error && !overview.data) {
    return <div className="errorState">{overview.error}</div>;
  }

  if (!overview.data) {
    return <div className="emptyState">Command center data is unavailable.</div>;
  }

  const { health, paper, ingestion, system, logs } = overview.data;
  const model = toPaperCockpitViewModel(paper);
  const runtimeLabel = health.trading_paused ? 'Paused' : 'Running';
  const signalDirection = paper.latest_signal?.direction ?? 'none';

  return (
    <div className={`pageStack ${styles.page}`}>
      <section className="pageHero">
        <span className="eyebrowLabel">ritual control room</span>
        <div className="heroTitleRow">
          <div>
            <h2>What is true right now</h2>
            <p className="lede">
              This surface compresses paper-account posture, runtime freshness, signal bias, and
              system drift into one operator read. Mystical framing stays in the copy. The numbers
              stay literal.
            </p>
          </div>
        </div>
        <div className="statusStrip" role="status" aria-live="polite">
          <span className="statusPill" data-tone={model.freshness.tone}>
            <RefreshCcw size={14} />
            {model.freshness.label}
          </span>
          <span
            className="statusPill"
            data-tone={health.trading_paused ? 'warning' : 'positive'}
          >
            <ShieldCheck size={14} />
            Runtime {runtimeLabel.toLowerCase()}
          </span>
          <span className="statusPill" data-tone={model.benchmarkTone}>
            <Waves size={14} />
            {model.benchmarkLabel}
          </span>
          <span className="statusPill" data-tone="neutral">
            <Aperture size={14} />
            Last event {formatTimestamp(health.last_event_at)}
          </span>
        </div>
      </section>

      <section className="metricGrid" aria-label="Command center metrics">
        <article className="metricCard">
          <span className="metricLabel">Paper equity</span>
          <span className="metricValue">{formatUsd(paper.performance.equity)}</span>
          <span className="metricDetail">{model.exposureLabel}</span>
        </article>
        <article className="metricCard">
          <span className="metricLabel">Signal bias</span>
          <span className="metricValue">{signalDirection.toUpperCase()}</span>
          <span className="metricDetail">{model.signal.detail}</span>
        </article>
        <article className="metricCard">
          <span className="metricLabel">Net PnL</span>
          <span className="metricValue">{formatUsd(paper.performance.net_pnl)}</span>
          <span className="metricDetail">{formatPct(paper.performance.net_pnl_pct)}</span>
        </article>
        <article className="metricCard">
          <span className="metricLabel">Alpha vs buy-and-hold</span>
          <span className="metricValue">{formatPct(paper.performance.alpha_vs_buy_hold)}</span>
          <span className="metricDetail">{model.tradeOutcomeLabel}</span>
        </article>
      </section>

      <section className="panelGrid">
        <article className="surfaceCard">
          <div className="surfaceHeader">
            <div>
              <h3>Runtime posture</h3>
              <p>Operational state with explicit source labels.</p>
            </div>
          </div>
          <div className="kvGrid">
            <div className="kvItem">
              <span className="kvLabel">Venue</span>
              <span className="kvValue">{health.venue}</span>
            </div>
            <div className="kvItem">
              <span className="kvLabel">Product</span>
              <span className="kvValue">{health.product_id}</span>
            </div>
            <div className="kvItem">
              <span className="kvLabel">Cash</span>
              <span className="kvValue">{formatUsd(paper.portfolio.cash)}</span>
            </div>
            <div className="kvItem">
              <span className="kvLabel">Position</span>
              <span className="kvValue">{formatBtc(paper.portfolio.position_qty)}</span>
            </div>
            <div className="kvItem">
              <span className="kvLabel">Mark price</span>
              <span className="kvValue">{formatUsd(paper.portfolio.mark_price)}</span>
            </div>
            <div className="kvItem">
              <span className="kvLabel">Autorun</span>
              <span className="kvValue">
                {health.autorun.enabled ? (health.autorun.running ? 'Running' : 'Idle') : 'Off'}
              </span>
            </div>
          </div>
        </article>

        <article className="surfaceCard">
          <div className="surfaceHeader">
            <div>
              <h3>Source-of-truth map</h3>
              <p>Which backend owns each visible claim.</p>
            </div>
          </div>
          <ul className="plainList">
            <li>`/v1/paper` owns account state, fills, alerts, and equity curve.</li>
            <li>`/v1/performance` owns benchmark delta, expectancy, and drawdown math.</li>
            <li>Legacy API endpoints only supply ingestion and system diagnostics here.</li>
          </ul>
        </article>
      </section>

      <section className="panelGrid">
        <article className="surfaceCard">
          <div className="surfaceHeader">
            <div>
              <h3>Signal interpretation</h3>
              <p>Plain-language cue adjacent to the ritual language.</p>
            </div>
          </div>
          <div className={styles.signalBlock}>
            <span className={`metricValue ${styles.signalValue}`}>{model.signal.label}</span>
            <p className="sectionNote">{model.signal.detail}</p>
            {paper.latest_signal?.reasons?.length ? (
              <ul className="plainList">
                {paper.latest_signal.reasons.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
            ) : (
              <div className="emptyState">No signal rationale has been recorded yet.</div>
            )}
          </div>
        </article>

        <article className="surfaceCard">
          <div className="surfaceHeader">
            <div>
              <h3>Environment checks</h3>
              <p>Legacy diagnostics stay explicit so they do not impersonate trading truth.</p>
            </div>
          </div>
          <div className="kvGrid">
            <div className="kvItem">
              <span className="kvLabel">Data freshness</span>
              <span className="kvValue">
                {ingestion ? `${ingestion.prices} prices / ${ingestion.news} news` : 'Unavailable'}
              </span>
            </div>
            <div className="kvItem">
              <span className="kvLabel">Database</span>
              <span className="kvValue">
                {system ? (system.database_ok ? 'Healthy' : 'Degraded') : 'Unavailable'}
              </span>
            </div>
            <div className="kvItem">
              <span className="kvLabel">Model artifact</span>
              <span className="kvValue">
                {system ? (system.model_exists ? 'Present' : 'Missing') : 'Unavailable'}
              </span>
            </div>
            <div className="kvItem">
              <span className="kvLabel">Total events</span>
              <span className="kvValue">{health.event_count}</span>
            </div>
          </div>
        </article>
      </section>

      <section className="surfaceCard">
        <div className="surfaceHeader">
          <div>
            <h3>Recent operational log</h3>
            <p>Text summary first, hover-free meaning second.</p>
          </div>
        </div>
        {logs.length > 0 ? (
          <div className="logList" role="log" aria-live="polite">
            {logs.map((log) => (
              <div
                key={`${log.timestamp}-${log.level}-${log.service ?? 'system'}-${log.message}`}
                className="logEntry"
              >
                <div className="logMeta">
                  <span>{log.level}</span>
                  <span>{log.timestamp}</span>
                  {log.service ? <span>{log.service}</span> : null}
                </div>
                <div>{log.message}</div>
              </div>
            ))}
          </div>
        ) : (
          <div className="emptyState">No recent log lines were returned from the legacy API.</div>
        )}
      </section>
    </div>
  );
}
