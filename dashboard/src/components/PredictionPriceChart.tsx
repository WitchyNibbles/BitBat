import {
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { PriceTimelinePoint, PredictionTimelinePoint } from '../api/client.ts';
import styles from './PredictionPriceChart.module.css';

interface PredictionPriceChartProps {
  points: PredictionTimelinePoint[];
  pricePoints: PriceTimelinePoint[];
}

interface ChartPoint {
  time: string;
  actualPrice: number | null;
  predictedPrice: number | null;
  predictedDirection: string;
  confidence: number | null;
  correct: boolean | null;
  isRealized: boolean;
}

const moneyFormatter = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  maximumFractionDigits: 0,
});

function formatMoney(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return 'n/a';
  return moneyFormatter.format(value);
}

function formatTime(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function directionLabel(direction: string): string {
  const normalized = direction.toLowerCase();
  if (normalized === 'up') return 'Up';
  if (normalized === 'down') return 'Down';
  return 'Flat';
}

function confidenceLabel(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return 'n/a';
  return `${(value * 100).toFixed(1)}%`;
}

function toPredictionPoints(points: PredictionTimelinePoint[]): ChartPoint[] {
  return [...points].map((point) => ({
    time: point.timestamp_utc,
    actualPrice: point.actual_price ?? null,
    predictedPrice: point.predicted_price ?? null,
    predictedDirection: point.predicted_direction,
    confidence: point.confidence ?? null,
    correct: point.correct ?? null,
    isRealized: point.is_realized,
  }));
}

function toSeriesData(
  pricePoints: PriceTimelinePoint[],
  points: PredictionTimelinePoint[],
): ChartPoint[] {
  const predictionMap = new Map(points.map((point) => [point.timestamp_utc, point]));
  const combined = new Map<string, ChartPoint>();

  pricePoints.forEach((point) => {
    const prediction = predictionMap.get(point.timestamp_utc);
    combined.set(point.timestamp_utc, {
      time: point.timestamp_utc,
      actualPrice: point.actual_price,
      predictedPrice: prediction?.predicted_price ?? null,
      predictedDirection: prediction?.predicted_direction ?? 'flat',
      confidence: prediction?.confidence ?? null,
      correct: prediction?.correct ?? null,
      isRealized: prediction?.is_realized ?? false,
    });
  });

  points.forEach((point) => {
    const existing = combined.get(point.timestamp_utc);
    combined.set(point.timestamp_utc, {
      time: point.timestamp_utc,
      actualPrice: existing?.actualPrice ?? point.actual_price ?? null,
      predictedPrice: point.predicted_price ?? null,
      predictedDirection: point.predicted_direction,
      confidence: point.confidence ?? null,
      correct: point.correct ?? null,
      isRealized: point.is_realized,
    });
  });

  return [...combined.values()].sort(
    (left, right) => new Date(left.time).getTime() - new Date(right.time).getTime(),
  );
}

function markerOpacity(point: ChartPoint): number {
  if (!point.isRealized) return 0.95;
  if (point.correct === false) return 0.45;
  return 0.9;
}

function renderTriangle(
  cx: number,
  cy: number,
  size: number,
  direction: 'up' | 'down',
): string {
  if (direction === 'up') {
    return `M ${cx} ${cy - size} L ${cx - size} ${cy + size} L ${cx + size} ${cy + size} Z`;
  }
  return `M ${cx} ${cy + size} L ${cx - size} ${cy - size} L ${cx + size} ${cy - size} Z`;
}

function markerColor(direction: string): string {
  if (direction === 'up') return 'var(--success)';
  if (direction === 'down') return 'var(--danger)';
  return 'var(--amber)';
}

function markerShape(direction: 'up' | 'down' | 'flat') {
  return (props: { cx?: number; cy?: number; payload?: ChartPoint }) => {
    const cx = props.cx ?? 0;
    const cy = props.cy ?? 0;
    const payload = props.payload;
    const opacity = payload ? markerOpacity(payload) : 0.9;
    const fill = markerColor(direction);

    if (direction === 'flat') {
      return (
        <circle
          cx={cx}
          cy={cy}
          r={5}
          fill={fill}
          fillOpacity={opacity}
          stroke="var(--surface)"
          strokeWidth={1.5}
        />
      );
    }

    return (
      <path
        d={renderTriangle(cx, cy, 6, direction)}
        fill={fill}
        fillOpacity={opacity}
        stroke="var(--surface)"
        strokeWidth={1.5}
      />
    );
  };
}

function TooltipContent({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ payload: ChartPoint }>;
  label?: string;
}) {
  if (!active || !payload || payload.length === 0) return null;
  const point = payload[0].payload;
  const status = !point.isRealized
    ? 'Pending'
    : point.correct === true
      ? 'Correct'
      : point.correct === false
        ? 'Missed'
        : 'Realized';

  return (
    <div className={styles.tooltip}>
      <div className={styles.tooltipTime}>{formatTime(label ?? point.time)}</div>
      <div className={styles.tooltipRow}>
        <span>Actual</span>
        <span>{formatMoney(point.actualPrice)}</span>
      </div>
      <div className={styles.tooltipRow}>
        <span>Predicted</span>
        <span>{formatMoney(point.predictedPrice)}</span>
      </div>
      <div className={styles.tooltipRow}>
        <span>Call</span>
        <span>{directionLabel(point.predictedDirection)}</span>
      </div>
      <div className={styles.tooltipRow}>
        <span>Conviction</span>
        <span>{confidenceLabel(point.confidence)}</span>
      </div>
      <div className={styles.tooltipRow}>
        <span>Status</span>
        <span>{status}</span>
      </div>
    </div>
  );
}

export function PredictionPriceChart({ points, pricePoints }: PredictionPriceChartProps) {
  const predictionPoints = toPredictionPoints(points);
  const data = toSeriesData(pricePoints, points);
  const latestPoint =
    predictionPoints.length > 0 ? predictionPoints[predictionPoints.length - 1] : null;
  const realizedPoints = predictionPoints.filter(
    (point) => point.isRealized && point.correct != null,
  );
  const realizedWins = realizedPoints.filter((point) => point.correct === true).length;
  const realizedAccuracy = realizedPoints.length > 0 ? realizedWins / realizedPoints.length : null;
  const predictedCoverageCount = predictionPoints.filter(
    (point) => point.predictedPrice != null,
  ).length;
  const predictedCoverage =
    predictionPoints.length > 0 ? predictedCoverageCount / predictionPoints.length : 0;

  const upPoints = predictionPoints.filter(
    (point) => point.predictedDirection === 'up' && point.actualPrice != null,
  );
  const downPoints = predictionPoints.filter(
    (point) => point.predictedDirection === 'down' && point.actualPrice != null,
  );
  const flatPoints = predictionPoints.filter(
    (point) => point.predictedDirection === 'flat' && point.actualPrice != null,
  );

  return (
    <section className={styles.card}>
      <div className={styles.header}>
        <div>
          <h2 className={styles.title}>Prediction vs BTC Price</h2>
          <p className={styles.subtitle}>
            Actual BTC close is the anchor line. Predicted price appears when the model emits a
            target, and markers show the direction call at each inference point.
          </p>
        </div>
        <div className={styles.legend}>
          <span className={styles.legendItem}>
            <span className={`${styles.legendSwatch} ${styles.actualSwatch}`} />
            Actual
          </span>
          <span className={styles.legendItem}>
            <span className={`${styles.legendSwatch} ${styles.predictedSwatch}`} />
            Predicted
          </span>
        </div>
      </div>

      <div className={styles.summaryRow}>
        <div className={styles.metric}>
          <span className={styles.metricLabel}>Latest call</span>
          <span className={styles.metricValue}>
            {latestPoint ? directionLabel(latestPoint.predictedDirection) : 'n/a'}
          </span>
        </div>
        <div className={styles.metric}>
          <span className={styles.metricLabel}>Conviction</span>
          <span className={styles.metricValue}>
            {latestPoint ? confidenceLabel(latestPoint.confidence) : 'n/a'}
          </span>
        </div>
        <div className={styles.metric}>
          <span className={styles.metricLabel}>Recent realized accuracy</span>
          <span className={styles.metricValue}>
            {realizedAccuracy != null ? `${(realizedAccuracy * 100).toFixed(1)}%` : 'n/a'}
          </span>
        </div>
      </div>

      <div className={styles.chartWrap}>
        <ResponsiveContainer width="100%" height={260}>
          <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
            <XAxis
              dataKey="time"
              minTickGap={32}
              tickFormatter={formatTime}
              tick={{ fill: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 11 }}
              stroke="var(--border)"
            />
            <YAxis
              tickFormatter={(value: number) => moneyFormatter.format(value)}
              tick={{ fill: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 11 }}
              stroke="var(--border)"
              width={78}
              domain={['auto', 'auto']}
            />
            <Tooltip content={<TooltipContent />} />
            <Line
              type="monotone"
              dataKey="actualPrice"
              stroke="var(--accent)"
              strokeWidth={2.5}
              dot={false}
              connectNulls
              name="Actual BTC"
            />
            <Line
              type="monotone"
              dataKey="predictedPrice"
              stroke="var(--success)"
              strokeWidth={2}
              strokeDasharray="6 4"
              dot={false}
              connectNulls
              name="Predicted price"
            />
            <Scatter data={upPoints} dataKey="actualPrice" shape={markerShape('up')} />
            <Scatter data={downPoints} dataKey="actualPrice" shape={markerShape('down')} />
            <Scatter data={flatPoints} dataKey="actualPrice" shape={markerShape('flat')} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className={styles.footer}>
        <span className={styles.footerNote}>
          Actual BTC is plotted from the full market series; markers only appear when the model ran.
        </span>
        <span className={styles.footerNote}>
          Predicted-price coverage: {predictedCoverageCount}/{predictionPoints.length} points
          {predictionPoints.length > 0 ? ` (${(predictedCoverage * 100).toFixed(0)}%)` : ''}
        </span>
      </div>
      {predictedCoverageCount === 0 ? (
        <p className={styles.coverageHint}>
          Predicted price is unavailable for this recent window. The model is still emitting
          direction calls, so the chart falls back to actual BTC price plus signal markers.
        </p>
      ) : null}
    </section>
  );
}
