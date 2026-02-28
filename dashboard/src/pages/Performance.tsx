import { useState } from 'react';
import type { ReactNode } from 'react';
import { MetricCard } from '../components/MetricCard.tsx';
import { AccuracyChart } from '../components/AccuracyChart.tsx';
import { StreakBar } from '../components/StreakBar.tsx';
import { DataTable } from '../components/DataTable.tsx';
import { useApi } from '../hooks/useApi.ts';
import { api } from '../api/client.ts';
import styles from './Performance.module.css';

const WINDOWS = [7, 30, 90] as const;
const FREQ = '1h';
const HORIZON = '4h';

const PRED_COLUMNS = [
  { key: 'time', label: 'Time' },
  { key: 'direction', label: 'Direction' },
  { key: 'confidence', label: 'Confidence', align: 'right' as const },
  { key: 'outcome', label: 'Outcome' },
  { key: 'correct', label: 'Correct', align: 'center' as const },
];

export function Performance() {
  const [days, setDays] = useState<number>(30);

  const perf = useApi(() => api.performance(FREQ, HORIZON, days), [days]);
  const history = useApi(
    () => api.predictionHistory(FREQ, HORIZON, days, 20),
    [days],
  );
  const snapshots = useApi(() => api.performanceSnapshots(20), []);

  const hitRateStr =
    perf.data?.hit_rate != null
      ? (perf.data.hit_rate * 100).toFixed(1)
      : '--';
  const avgReturnStr =
    perf.data?.avg_return != null
      ? (perf.data.avg_return * 100).toFixed(2)
      : '--';

  // Build streak from last 10 realized predictions
  const streakResults: boolean[] = history.data
    ? history.data.predictions
        .filter((p) => p.correct != null)
        .slice(0, 10)
        .map((p) => p.correct === true)
    : [];

  // Build table rows
  const rows: Record<string, ReactNode>[] = history.data
    ? history.data.predictions.map((p) => ({
        time: p.timestamp_utc,
        direction: p.predicted_direction.toUpperCase(),
        confidence:
          p.predicted_return != null
            ? `${(Math.abs(p.predicted_return) * 100).toFixed(1)}%`
            : '--',
        outcome: p.actual_direction ?? '--',
        correct:
          p.correct != null ? (p.correct ? 'Yes' : 'No') : '--',
      }))
    : [];

  return (
    <div className={styles.page}>
      <h2>Performance</h2>

      <div className={styles.windowRow}>
        {WINDOWS.map((w) => (
          <button
            key={w}
            className={`${styles.windowBtn} ${days === w ? styles.windowBtnActive : ''}`}
            onClick={() => setDays(w)}
          >
            {w}d
          </button>
        ))}
      </div>

      <div className={styles.metrics}>
        <MetricCard
          label="Total Predictions"
          value={perf.data?.total_predictions ?? '--'}
        />
        <MetricCard
          label="Realized"
          value={perf.data?.realized_predictions ?? '--'}
        />
        <MetricCard label="Hit Rate" value={hitRateStr} unit="%" />
        <MetricCard label="Avg Return" value={avgReturnStr} unit="%" />
      </div>

      <div className="divider">&#x2726;</div>

      {snapshots.data && snapshots.data.snapshots.length > 0 ? (
        <AccuracyChart snapshots={snapshots.data.snapshots} />
      ) : (
        <p className={styles.empty}>No accuracy snapshots available yet.</p>
      )}

      {streakResults.length > 0 && (
        <div className={styles.streakSection}>
          <span className={styles.streakLabel}>Last 10 results</span>
          <StreakBar results={streakResults} />
        </div>
      )}

      <div className="divider">&#x2726;</div>

      {rows.length > 0 ? (
        <DataTable columns={PRED_COLUMNS} rows={rows} />
      ) : (
        <p className={styles.empty}>No prediction history for this window.</p>
      )}
    </div>
  );
}
