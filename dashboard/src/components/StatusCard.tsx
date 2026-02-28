import styles from './StatusCard.module.css';

interface StatusCardProps {
  label: string;
  value: string;
  sublabel?: string;
  status?: 'success' | 'amber' | 'danger' | 'muted';
}

export function StatusCard({ label, value, sublabel, status }: StatusCardProps) {
  const valueClass = status
    ? `${styles.value} ${styles[status]}`
    : styles.value;

  return (
    <div className={styles.card}>
      <span className={styles.label}>{label}</span>
      <span className={valueClass}>{value}</span>
      {sublabel && <span className={styles.sublabel}>{sublabel}</span>}
    </div>
  );
}
