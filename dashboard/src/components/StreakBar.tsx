import styles from './StreakBar.module.css';

interface StreakBarProps {
  results: boolean[];
}

export function StreakBar({ results }: StreakBarProps) {
  const wins = results.filter(Boolean).length;

  return (
    <div className={styles.wrapper}>
      <div className={styles.summary}>
        <span className={styles.summaryText}>
          {wins}/{results.length} correct
        </span>
        <span className={styles.legend}>Green = correct call. Red = missed call.</span>
      </div>
      <div className={styles.bar}>
        {results.map((win, i) => (
          <span
            key={`${win ? 'win' : 'loss'}-${i}`}
            className={`${styles.dot} ${win ? styles.win : styles.loss}`}
            title={win ? 'Correct prediction' : 'Incorrect prediction'}
          />
        ))}
      </div>
    </div>
  );
}
