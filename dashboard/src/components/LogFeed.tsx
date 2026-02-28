import type { SystemLogEntry } from '../api/client.ts';
import styles from './LogFeed.module.css';

interface LogFeedProps {
  logs: SystemLogEntry[];
}

function levelClass(level: string): string {
  const l = level.toLowerCase();
  if (l === 'info') return 'info';
  if (l === 'warning' || l === 'warn') return 'warning';
  if (l === 'error' || l === 'critical') return 'error';
  return 'debug';
}

export function LogFeed({ logs }: LogFeedProps) {
  return (
    <div className={styles.feed}>
      {logs.map((log, i) => {
        const cls = levelClass(log.level);
        return (
          <div key={i} className={`${styles.entry} ${styles[cls]}`}>
            <div className={styles.header}>
              <span className={`${styles.badge} ${styles[cls]}`}>{log.level}</span>
              <span className={styles.timestamp}>{log.timestamp}</span>
            </div>
            <div className={styles.message}>{log.message}</div>
          </div>
        );
      })}
    </div>
  );
}
