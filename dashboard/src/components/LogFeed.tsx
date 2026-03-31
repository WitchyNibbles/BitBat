import type { SystemLogEntry } from '../api/client.ts';
import styles from './LogFeed.module.css';

interface LogFeedProps {
  logs: SystemLogEntry[];
  lastUpdatedAt?: Date | null;
  pollingLabel?: string;
}

function levelClass(level: string): string {
  const l = level.toLowerCase();
  if (l === 'info') return 'info';
  if (l === 'warning' || l === 'warn') return 'warning';
  if (l === 'error' || l === 'critical') return 'error';
  return 'debug';
}

function formatUpdatedAt(lastUpdatedAt?: Date | null): string {
  if (!lastUpdatedAt) return 'Waiting for first sync';
  return `Last synced ${lastUpdatedAt.toLocaleTimeString()}`;
}

export function LogFeed({ logs, lastUpdatedAt, pollingLabel }: LogFeedProps) {
  if (logs.length === 0) {
    return (
      <div className={styles.feed}>
        <div className={styles.statusRow}>
          <span className={styles.liveBadge}>Live</span>
          <span className={styles.statusText}>{formatUpdatedAt(lastUpdatedAt)}</span>
          {pollingLabel ? <span className={styles.statusText}>{pollingLabel}</span> : null}
        </div>
        <div className={styles.emptyState}>No events recorded yet.</div>
      </div>
    );
  }

  return (
    <div className={styles.feed}>
      <div className={styles.statusRow}>
        <span className={styles.liveBadge}>Live</span>
        <span className={styles.statusText}>{formatUpdatedAt(lastUpdatedAt)}</span>
        {pollingLabel ? <span className={styles.statusText}>{pollingLabel}</span> : null}
      </div>
      {logs.map((log) => {
        const cls = levelClass(log.level);
        const key = `${log.timestamp}-${log.level}-${log.service ?? 'system'}-${log.message}`;
        return (
          <div key={key} className={`${styles.entry} ${styles[cls]}`}>
            <div className={styles.header}>
              <span className={`${styles.badge} ${styles[cls]}`}>{log.level}</span>
              <span className={styles.timestamp}>{log.timestamp}</span>
              {log.service ? <span className={styles.service}>{log.service}</span> : null}
            </div>
            <div className={styles.message}>{log.message}</div>
          </div>
        );
      })}
    </div>
  );
}
