import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import styles from './PredictionCard.module.css';

interface PredictionCardProps {
  direction: string;
  confidence?: number;
  timestamp?: string;
  modelVersion?: string;
}

function directionKey(direction: string): 'up' | 'down' | 'flat' {
  const d = direction.toLowerCase();
  if (d === 'up' || d === 'bull' || d === 'bullish') return 'up';
  if (d === 'down' || d === 'bear' || d === 'bearish') return 'down';
  return 'flat';
}

export function PredictionCard({
  direction,
  confidence,
  timestamp,
  modelVersion,
}: PredictionCardProps) {
  const key = directionKey(direction);

  const IconComponent = key === 'up' ? TrendingUp : key === 'down' ? TrendingDown : Minus;

  return (
    <div className={`${styles.card} ${styles[key]}`}>
      <div className={styles.header}>
        <span className={`${styles.icon} ${styles[key]}`}>
          <IconComponent size={24} />
        </span>
        <span className={`${styles.direction} ${styles[key]}`}>{direction}</span>
      </div>

      {confidence != null && (
        <span className={styles.confidence}>
          {(confidence * 100).toFixed(1)}% confidence
        </span>
      )}

      {(timestamp || modelVersion) && (
        <div className={styles.meta}>
          {timestamp && <span className={styles.metaItem}>{timestamp}</span>}
          {modelVersion && <span className={styles.metaItem}>model {modelVersion}</span>}
        </div>
      )}
    </div>
  );
}
