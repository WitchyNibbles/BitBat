import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import styles from './PredictionCard.module.css';

interface PredictionCardProps {
  direction: string;
  confidence?: number;
  pUp?: number | null;
  pDown?: number | null;
  pFlat?: number | null;
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
  pUp,
  pDown,
  pFlat,
  timestamp,
  modelVersion,
}: PredictionCardProps) {
  const key = directionKey(direction);
  const IconComponent = key === 'up' ? TrendingUp : key === 'down' ? TrendingDown : Minus;
  const conviction = confidence ?? (key === 'up' ? pUp : key === 'down' ? pDown : pFlat);
  const distribution = [
    { label: 'Up', value: pUp, toneClass: styles.fillUp },
    { label: 'Flat', value: pFlat, toneClass: styles.fillFlat },
    { label: 'Down', value: pDown, toneClass: styles.fillDown },
  ].filter((entry) => entry.value != null);

  return (
    <div className={`${styles.card} ${styles[key]}`}>
      <div className={styles.header}>
        <span className={`${styles.icon} ${styles[key]}`}>
          <IconComponent size={24} />
        </span>
        <span className={`${styles.direction} ${styles[key]}`}>{direction}</span>
      </div>

      {conviction != null && (
        <div className={styles.signalBlock}>
          <div className={styles.signalHeader}>
            <span className={styles.signalLabel}>Model conviction</span>
            <span className={styles.signalValue}>{(conviction * 100).toFixed(1)}%</span>
          </div>
          <p className={styles.signalHint}>Probability assigned to the current call.</p>
        </div>
      )}

      {distribution.length > 0 && (
        <div className={styles.distribution}>
          {distribution.map((entry) => (
            <div key={entry.label} className={styles.distributionRow}>
              <div className={styles.distributionMeta}>
                <span className={styles.distributionLabel}>{entry.label}</span>
                <span className={styles.distributionValue}>
                  {((entry.value ?? 0) * 100).toFixed(1)}%
                </span>
              </div>
              <div className={styles.track}>
                <span
                  className={`${styles.fill} ${entry.toneClass}`}
                  style={{ width: `${Math.max(0, Math.min(100, (entry.value ?? 0) * 100))}%` }}
                />
              </div>
            </div>
          ))}
        </div>
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
