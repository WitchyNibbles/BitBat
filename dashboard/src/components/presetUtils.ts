export const FREQ_HORIZON_LABELS: Record<string, string> = {
  '5m': '5 min',
  '15m': '15 min',
  '30m': '30 min',
  '1h': '1 hour',
  '4h': '4 hours',
  '1d': '1 day',
  '24h': '24 hours',
};

export function formatFreqHorizon(value: string): string {
  return FREQ_HORIZON_LABELS[value] ?? value;
}
