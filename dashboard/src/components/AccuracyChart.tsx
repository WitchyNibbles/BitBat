import {
  ResponsiveContainer,
  LineChart,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Line,
} from 'recharts';
import type { PerformanceSnapshot } from '../api/client.ts';

interface AccuracyChartProps {
  snapshots: PerformanceSnapshot[];
}

interface DataPoint {
  time: string;
  accuracy: number | null;
}

function toDataPoints(snapshots: PerformanceSnapshot[]): DataPoint[] {
  return [...snapshots].reverse().map((s) => ({
    time: s.snapshot_time,
    accuracy: s.hit_rate != null ? s.hit_rate * 100 : null,
  }));
}

export function AccuracyChart({ snapshots }: AccuracyChartProps) {
  const data = toDataPoints(snapshots);

  return (
    <ResponsiveContainer width="100%" height={250}>
      <LineChart data={data}>
        <CartesianGrid
          strokeDasharray="3 3"
          stroke="var(--border)"
        />
        <XAxis
          dataKey="time"
          tick={{ fill: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 11 }}
          stroke="var(--border)"
        />
        <YAxis
          domain={[0, 100]}
          tick={{ fill: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 11 }}
          stroke="var(--border)"
        />
        <Tooltip
          contentStyle={{
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius)',
            fontFamily: 'var(--font-mono)',
            fontSize: 12,
            color: 'var(--text)',
          }}
        />
        <Line
          type="monotone"
          dataKey="accuracy"
          stroke="var(--accent)"
          strokeWidth={2}
          dot={{ r: 3, fill: 'var(--accent)' }}
          name="Accuracy %"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
