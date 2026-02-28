import {
  ResponsiveContainer,
  ComposedChart,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Area,
} from 'recharts';
import type { PredictionResponse } from '../api/client.ts';

interface TimelineChartProps {
  predictions: PredictionResponse[];
}

interface DataPoint {
  time: string;
  predicted_return: number | null;
  actual_return: number | null;
}

function toDataPoints(predictions: PredictionResponse[]): DataPoint[] {
  return predictions.map((p) => ({
    time: p.timestamp_utc,
    predicted_return: p.predicted_return ?? null,
    actual_return: p.actual_return ?? null,
  }));
}

export function TimelineChart({ predictions }: TimelineChartProps) {
  const data = toDataPoints(predictions);

  return (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={data}>
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
        <Area
          type="monotone"
          dataKey="predicted_return"
          stroke="var(--accent)"
          fill="var(--accent-dim)"
          strokeWidth={2}
          name="Predicted"
        />
        <Area
          type="monotone"
          dataKey="actual_return"
          stroke="var(--text-muted)"
          fill="none"
          strokeWidth={1.5}
          strokeDasharray="4 3"
          name="Actual"
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
