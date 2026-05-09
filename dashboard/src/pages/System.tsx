import type {
  DetailedHealthResponse,
  IngestionStatus,
  RetrainingEventsResponse,
  SystemLogsResponse,
  SystemStatusResponse,
} from '../api/client.ts';
import { api } from '../api/client.ts';
import { formatTimestamp } from '../api/paperViewModel.ts';
import type { V2HealthResponse } from '../api/v2Client.ts';
import { v2Api } from '../api/v2Client.ts';
import { useApi } from '../hooks/useApi.ts';
import { usePolling } from '../hooks/usePolling.ts';
import styles from './System.module.css';

const REFRESH_MS = 15_000;

interface SystemPageData {
  status: SystemStatusResponse;
  health: DetailedHealthResponse;
  ingestion: IngestionStatus;
  logs: SystemLogsResponse;
  retraining: RetrainingEventsResponse;
  v2Health: V2HealthResponse | null;
}

async function loadSystemPage(): Promise<SystemPageData> {
  const [status, health, ingestion, logs, retraining, v2Health] = await Promise.all([
    api.systemStatus(),
    api.healthDetailed(),
    api.ingestionStatus(),
    api.systemLogs(10),
    api.retrainingEvents(6),
    v2Api.health().catch(() => null),
  ]);

  return { status, health, ingestion, logs, retraining, v2Health };
}

export function System() {
  const system = useApi(loadSystemPage, []);

  usePolling(() => {
    system.refetch();
  }, REFRESH_MS);

  if (system.loading && !system.data) {
    return <div className="emptyState">Resolving system diagnostics...</div>;
  }

  if (system.error && !system.data) {
    return <div className="errorState">{system.error}</div>;
  }

  if (!system.data) {
    return <div className="emptyState">System diagnostics are unavailable.</div>;
  }

  const { status, health, ingestion, logs, retraining, v2Health } = system.data;

  return (
    <div className={`pageStack ${styles.page}`}>
      <section className="pageHero">
        <span className="eyebrowLabel">diagnostics and safeguards</span>
        <div className="heroTitleRow">
          <div>
            <h2>System health</h2>
            <p className="lede">
              This route is for environment truth: schema readiness, ingestion freshness, model
              presence, and log evidence. It does not make trading claims on behalf of the paper
              cockpit.
            </p>
          </div>
        </div>
      </section>

      <section className="metricGrid">
        <article className="metricCard">
          <span className="metricLabel">Database</span>
          <span className="metricValue">{status.database_ok ? 'Healthy' : 'Degraded'}</span>
          <span className="metricDetail">
            {status.database_present ? 'Database file present' : 'Database file missing'}
          </span>
        </article>
        <article className="metricCard">
          <span className="metricLabel">Schema readiness</span>
          <span className="metricValue">
            {health.schema_readiness?.is_compatible ? 'Compatible' : 'Needs review'}
          </span>
          <span className="metricDetail">
            {health.schema_readiness?.compatibility_state ?? 'Unknown'}
          </span>
        </article>
        <article className="metricCard">
          <span className="metricLabel">Ingestion</span>
          <span className="metricValue">{ingestion.prices}</span>
          <span className="metricDetail">News {ingestion.news}</span>
        </article>
        <article className="metricCard">
          <span className="metricLabel">v2 runtime</span>
          <span className="metricValue">{v2Health ? v2Health.status : 'Offline'}</span>
          <span className="metricDetail">
            {v2Health ? formatTimestamp(v2Health.last_event_at) : 'No operator auth or runtime'}
          </span>
        </article>
      </section>

      <section className="panelGrid">
        <article className="surfaceCard">
          <div className="surfaceHeader">
            <div>
              <h3>Compatibility and assets</h3>
              <p>Diagnostics stay literal and avoid leaking secret values.</p>
            </div>
          </div>
          <div className="kvGrid">
            <div className="kvItem">
              <span className="kvLabel">Model artifact</span>
              <span className="kvValue">{status.model_exists ? 'Present' : 'Missing'}</span>
            </div>
            <div className="kvItem">
              <span className="kvLabel">Dataset artifact</span>
              <span className="kvValue">{status.dataset_exists ? 'Present' : 'Missing'}</span>
            </div>
            <div className="kvItem">
              <span className="kvLabel">Predictions stored</span>
              <span className="kvValue">{status.total_predictions}</span>
            </div>
            <div className="kvItem">
              <span className="kvLabel">Latest prediction</span>
              <span className="kvValue">
                {status.last_prediction_time ? formatTimestamp(status.last_prediction_time) : 'None'}
              </span>
            </div>
          </div>
        </article>

        <article className="surfaceCard">
          <div className="surfaceHeader">
            <div>
              <h3>Retraining ledger</h3>
              <p>Last runs, triggers, and declared result.</p>
            </div>
          </div>
          {retraining.events.length > 0 ? (
            <div className="tableScroll">
              <table className="ledgerTable">
                <thead>
                  <tr>
                    <th>Started</th>
                    <th>Trigger</th>
                    <th>Status</th>
                    <th>New model</th>
                  </tr>
                </thead>
                <tbody>
                  {retraining.events.map((event) => (
                    <tr key={`${event.started_at}-${event.trigger_reason}`}>
                      <td>{formatTimestamp(event.started_at)}</td>
                      <td>{event.trigger_reason}</td>
                      <td>{event.status}</td>
                      <td>{event.new_model_version ?? '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="emptyState">No retraining events recorded yet.</div>
          )}
        </article>
      </section>

      <section className="surfaceCard">
        <div className="surfaceHeader">
          <div>
            <h3>System logs</h3>
            <p>Recent diagnostic output from the legacy runtime surface.</p>
          </div>
        </div>
        {logs.logs.length > 0 ? (
          <div className="logList" role="log" aria-live="polite">
            {logs.logs.map((log) => (
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
          <div className="emptyState">No system logs returned.</div>
        )}
      </section>
    </div>
  );
}
