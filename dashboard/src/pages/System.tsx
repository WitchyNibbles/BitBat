import { useState } from 'react';
import type { ReactNode } from 'react';
import { StatusCard } from '../components/StatusCard.tsx';
import { DataTable } from '../components/DataTable.tsx';
import { LogFeed } from '../components/LogFeed.tsx';
import { useApi } from '../hooks/useApi.ts';
import { usePolling } from '../hooks/usePolling.ts';
import { api } from '../api/client.ts';
import styles from './System.module.css';

const SNAPSHOT_COLUMNS = [
  { key: 'time', label: 'Time' },
  { key: 'model', label: 'Model' },
  { key: 'accuracy', label: 'Accuracy', align: 'right' as const },
  { key: 'predictions', label: 'Predictions', align: 'right' as const },
  { key: 'sharpe', label: 'Sharpe', align: 'right' as const },
  { key: 'maxDD', label: 'Max DD', align: 'right' as const },
];

const RETRAIN_COLUMNS = [
  { key: 'time', label: 'Time' },
  { key: 'trigger', label: 'Trigger' },
  { key: 'status', label: 'Status' },
  { key: 'oldModel', label: 'Old Model' },
  { key: 'newModel', label: 'New Model' },
  { key: 'cvGain', label: 'CV Gain', align: 'right' as const },
  { key: 'duration', label: 'Duration', align: 'right' as const },
];

export function System() {
  const [logLimit, setLogLimit] = useState(20);

  const status = useApi(() => api.systemStatus(), []);
  const health = useApi(() => api.healthDetailed(), []);
  const logs = useApi(() => api.systemLogs(logLimit), [logLimit]);
  const snapshots = useApi(() => api.performanceSnapshots(20), []);
  const retraining = useApi(() => api.retrainingEvents(10), []);
  const ingestion = useApi(() => api.ingestionStatus(), []);

  usePolling(() => {
    status.refetch();
    health.refetch();
    logs.refetch();
  }, 30_000);

  // Derive status card values
  const dbStatus = status.data
    ? status.data.database_ok
      ? 'OK'
      : status.data.database_present
        ? 'Degraded'
        : 'Missing'
    : 'Loading...';
  const dbColor: 'success' | 'amber' | 'danger' | 'muted' =
    dbStatus === 'OK' ? 'success' : dbStatus === 'Degraded' ? 'amber' : dbStatus === 'Missing' ? 'danger' : 'muted';

  const schemaStatus = health.data?.schema_readiness
    ? health.data.schema_readiness.is_compatible
      ? 'Compatible'
      : 'Incompatible'
    : 'Unknown';
  const schemaColor: 'success' | 'danger' | 'muted' =
    schemaStatus === 'Compatible' ? 'success' : schemaStatus === 'Incompatible' ? 'danger' : 'muted';

  const modelStatus = status.data
    ? status.data.model_exists
      ? 'Available'
      : 'Missing'
    : 'Loading...';
  const modelColor: 'success' | 'danger' | 'muted' =
    modelStatus === 'Available' ? 'success' : modelStatus === 'Missing' ? 'danger' : 'muted';

  const ingestionLabel = ingestion.data ? 'Active' : 'Unknown';
  const ingestionSublabel = ingestion.data
    ? `Prices: ${ingestion.data.prices} / News: ${ingestion.data.news}`
    : undefined;

  // Build snapshot rows
  const snapshotRows: Record<string, ReactNode>[] = snapshots.data
    ? snapshots.data.snapshots.map((s) => ({
        time: s.snapshot_time,
        model: s.model_version,
        accuracy:
          s.hit_rate != null ? `${(s.hit_rate * 100).toFixed(1)}%` : '--',
        predictions: String(s.total_predictions),
        sharpe: s.sharpe_ratio != null ? s.sharpe_ratio.toFixed(2) : '--',
        maxDD:
          s.max_drawdown != null
            ? `${(s.max_drawdown * 100).toFixed(1)}%`
            : '--',
      }))
    : [];

  // Build retraining rows
  const retrainRows: Record<string, ReactNode>[] = retraining.data
    ? retraining.data.events.map((e) => ({
        time: e.started_at,
        trigger: e.trigger_reason,
        status: e.status,
        oldModel: e.old_model_version ?? '--',
        newModel: e.new_model_version ?? '--',
        cvGain:
          e.cv_improvement != null
            ? `${(e.cv_improvement * 100).toFixed(2)}%`
            : '--',
        duration:
          e.training_duration_seconds != null
            ? `${e.training_duration_seconds.toFixed(0)}s`
            : '--',
      }))
    : [];

  return (
    <div className={styles.page}>
      <h2>System Health</h2>

      <div className={styles.statusGrid}>
        <StatusCard label="Database" value={dbStatus} status={dbColor} />
        <StatusCard label="Schema" value={schemaStatus} status={schemaColor} />
        <StatusCard
          label="Model"
          value={modelStatus}
          status={modelColor}
          sublabel={status.data?.active_model_version ?? undefined}
        />
        <StatusCard
          label="Ingestion"
          value={ingestionLabel}
          status={ingestion.data ? 'success' : 'muted'}
          sublabel={ingestionSublabel}
        />
      </div>

      <div className="divider">&#x2726;</div>

      <section>
        <h3>Performance Snapshots</h3>
        {snapshotRows.length > 0 ? (
          <DataTable columns={SNAPSHOT_COLUMNS} rows={snapshotRows} />
        ) : (
          <p className={styles.empty}>No performance snapshots recorded.</p>
        )}
      </section>

      <div className="divider">&#x2726;</div>

      <section>
        <div className={styles.sectionHeader}>
          <h3>System Logs</h3>
          <select
            className={styles.logSelect}
            value={logLimit}
            onChange={(e) => setLogLimit(parseInt(e.target.value, 10))}
          >
            <option value={20}>20</option>
            <option value={50}>50</option>
            <option value={100}>100</option>
          </select>
        </div>
        {logs.data ? (
          <LogFeed logs={logs.data.logs} />
        ) : (
          <p className={styles.empty}>Loading logs...</p>
        )}
      </section>

      <div className="divider">&#x2726;</div>

      <section>
        <h3>Retraining History</h3>
        {retrainRows.length > 0 ? (
          <DataTable columns={RETRAIN_COLUMNS} rows={retrainRows} />
        ) : (
          <p className={styles.empty}>No retraining events recorded.</p>
        )}
      </section>
    </div>
  );
}
