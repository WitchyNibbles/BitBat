import { StatusCard } from '../components/StatusCard.tsx';
import { PredictionCard } from '../components/PredictionCard.tsx';
import { LogFeed } from '../components/LogFeed.tsx';
import { useApi } from '../hooks/useApi.ts';
import { usePolling } from '../hooks/usePolling.ts';
import { api } from '../api/client.ts';
import styles from './Home.module.css';

export function Home() {
  const health = useApi(() => api.healthDetailed(), []);
  const prediction = useApi(() => api.latestPrediction(), []);
  const logs = useApi(() => api.systemLogs(10), []);
  const ingestion = useApi(() => api.ingestionStatus(), []);

  // Auto-refresh every 60s
  usePolling(() => {
    health.refetch();
    prediction.refetch();
    logs.refetch();
    ingestion.refetch();
  }, 60_000);

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
        {prediction.data ? (
          <PredictionCard
            direction={prediction.data.predicted_direction}
            confidence={
              prediction.data.predicted_return != null
                ? Math.abs(prediction.data.predicted_return)
                : undefined
            }
            timestamp={prediction.data.timestamp_utc}
            modelVersion={prediction.data.model_version}
          />
        ) : (
          <StatusCard label="Latest Prediction" value="No data" status="muted" />
        )}
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
        <h2>Recent Activity</h2>
        {logs.data ? (
          <LogFeed logs={logs.data.logs} />
        ) : (
          <p className={styles.empty}>Loading activity...</p>
        )}
      </section>
    </div>
  );
}
