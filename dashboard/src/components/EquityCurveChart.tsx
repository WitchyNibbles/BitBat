import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { formatTimestamp, formatUsd } from '../api/paperViewModel';
import type { V2PaperPerformancePointResponse } from '../api/v2Client';
import styles from './EquityCurveChart.module.css';

interface EquityCurveChartProps {
  points: V2PaperPerformancePointResponse[];
  title: string;
  message: string;
}

function shortTime(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function TooltipContent({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{
    payload: V2PaperPerformancePointResponse;
  }>;
}) {
  if (!active || !payload || payload.length === 0) {
    return null;
  }

  const point = payload[0].payload;
  return (
    <div className={styles.tooltip}>
      <div className={styles.tooltipTime}>{formatTimestamp(point.occurred_at)}</div>
      <div className={styles.tooltipRow}>
        <span>Equity</span>
        <span>{formatUsd(point.equity)}</span>
      </div>
      <div className={styles.tooltipRow}>
        <span>Cash</span>
        <span>{formatUsd(point.cash)}</span>
      </div>
      <div className={styles.tooltipRow}>
        <span>Position</span>
        <span>{point.position_qty.toFixed(4)} BTC</span>
      </div>
      <div className={styles.tooltipRow}>
        <span>Mark</span>
        <span>{formatUsd(point.mark_price)}</span>
      </div>
    </div>
  );
}

export function EquityCurveChart({ points, title, message }: EquityCurveChartProps) {
  return (
    <section className={styles.card}>
      <div className={styles.header}>
        <div>
          <h3>{title}</h3>
          <p>{message}</p>
        </div>
      </div>
      <div className={styles.chartWrap}>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={points}>
            <CartesianGrid stroke="rgba(133, 153, 144, 0.12)" vertical={false} />
            <XAxis
              dataKey="occurred_at"
              tickFormatter={shortTime}
              minTickGap={32}
              stroke="rgba(168, 180, 173, 0.7)"
            />
            <YAxis
              domain={['dataMin - 100', 'dataMax + 100']}
              tickFormatter={(value) => formatUsd(Number(value))}
              stroke="rgba(168, 180, 173, 0.7)"
              width={96}
            />
            <Tooltip content={<TooltipContent />} />
            <Line
              type="monotone"
              dataKey="equity"
              stroke="var(--accent-strong)"
              strokeWidth={2.5}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
