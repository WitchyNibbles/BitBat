import styles from './StreakBar.module.css';

interface StreakBarProps {
  results: boolean[];
}

export function StreakBar({ results }: StreakBarProps) {
  return (
    <div className={styles.bar}>
      {results.map((win, i) => (
        <span
          key={i}
          className={`${styles.dot} ${win ? styles.win : styles.loss}`}
          title={win ? 'Win' : 'Loss'}
        />
      ))}
    </div>
  );
}
