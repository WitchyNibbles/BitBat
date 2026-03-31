import { StatusCard } from '../components/StatusCard.tsx';
import { PredictionPriceChart } from '../components/PredictionPriceChart.tsx';
import { LogFeed } from '../components/LogFeed.tsx';
import { useApi } from '../hooks/useApi.ts';
import { usePolling } from '../hooks/usePolling.ts';
import { api } from '../api/client.ts';
import styles from './Home.module.css';

const AUTO_REFRESH_MS = 5_000;

export function Home() {
  const health = useApi(() => api.healthDetailed(), []);
  const timeline = useApi(() => api.predictionTimeline(undefined, undefined, 14, 72), []);
  const logs = useApi(() => api.systemLogs(10), []);
  const ingestion = useApi(() => api.ingestionStatus(), []);

  // Keep the home page feeling live without aggressive polling.
  usePolling(() => {
    health.refetch();
    timeline.refetch();
    logs.refetch();
    ingestion.refetch();
  }, AUTO_REFRESH_MS);

  const agentStatus =
    health.data?.status === 'ok'
      ? 'Running'
      : health.data?.status === 'degraded'
        ? 'Degraded'
        : 'Offline';

  const agentColor: 'success' | 'amber' | 'danger' =
    agentStatus === 'Running'
      ? 'success'
      : agentStatus === 'Degraded'
        ? 'amber'
        : 'danger';

  return (
    <div className={styles.page}>
      <div className={styles.cards}>
        <div className={styles.chartSlot}>
          {timeline.data && timeline.data.points.length > 0 ? (
            <PredictionPriceChart
              points={timeline.data.points}
              pricePoints={timeline.data.price_points}
            />
          ) : (
            <StatusCard label="Prediction Chart" value="No data" status="muted" />
          )}
        </div>
        <StatusCard
          label="Agent Status"
          value={agentStatus}
          status={agentColor}
          sublabel={
            health.data
              ? `Uptime: ${Math.floor(health.data.uptime_seconds / 3600)}h`
              : undefined
          }
        />
        <StatusCard
          label="Data Freshness"
          value={
            ingestion.data
              ? `${ingestion.data.prices || '?'} / ${ingestion.data.news || '?'}`
              : 'Loading...'
          }
          sublabel="Prices / News"
          status="muted"
        />
      </div>

      <div className="divider">&#x2726;</div>

      <section>
        <div className={styles.sectionHeader}>
          <h2>Recent Activity</h2>
          <p className={styles.sectionNote}>
            Auto-refreshes every {AUTO_REFRESH_MS / 1000}s so new monitor events appear on their
            own.
          </p>
        </div>
        {logs.data ? (
          <LogFeed
            logs={logs.data.logs}
            lastUpdatedAt={logs.lastUpdatedAt}
            pollingLabel={`Polling every ${AUTO_REFRESH_MS / 1000}s`}
          />
        ) : (
          <p className={styles.empty}>Loading activity...</p>
        )}
      </section>
    </div>
  );
}
