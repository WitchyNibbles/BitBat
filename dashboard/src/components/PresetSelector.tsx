import { Shield, Scale, Rocket, Zap, TrendingUp } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import styles from './PresetSelector.module.css';

interface PresetDef {
  id: string;
  name: string;
  description: string;
  icon: LucideIcon;
  color: string;
  freq: string;
  horizon: string;
}

interface PresetSelectorProps {
  selected: string;
  onSelect: (id: string) => void;
  onPresetData?: (freq: string, horizon: string) => void;
}

const FREQ_HORIZON_LABELS: Record<string, string> = {
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

const PRESETS: PresetDef[] = [
  {
    id: 'scalper',
    name: 'Scalper',
    description: '5m freq, rapid sub-hourly signals',
    icon: Zap,
    color: '#F59E0B',
    freq: '5m',
    horizon: '30m',
  },
  {
    id: 'conservative',
    name: 'Conservative',
    description: '24h horizon, high confidence',
    icon: Shield,
    color: '#3B82F6',
    freq: '1h',
    horizon: '24h',
  },
  {
    id: 'balanced',
    name: 'Balanced',
    description: '4h horizon, medium confidence',
    icon: Scale,
    color: '#10B981',
    freq: '1h',
    horizon: '4h',
  },
  {
    id: 'aggressive',
    name: 'Aggressive',
    description: '1h horizon, lower confidence',
    icon: Rocket,
    color: '#EF4444',
    freq: '1h',
    horizon: '1h',
  },
  {
    id: 'swing',
    name: 'Swing',
    description: '15m freq, sub-hourly swing signals',
    icon: TrendingUp,
    color: '#8B5CF6',
    freq: '15m',
    horizon: '1h',
  },
];

export function PresetSelector({ selected, onSelect, onPresetData }: PresetSelectorProps) {
  return (
    <div className={styles.grid}>
      {PRESETS.map((preset) => {
        const Icon = preset.icon;
        const isActive = selected === preset.id;
        return (
          <button
            key={preset.id}
            className={`${styles.card} ${isActive ? styles.active : ''}`}
            style={{ '--preset-color': preset.color } as React.CSSProperties}
            onClick={() => {
              onSelect(preset.id);
              onPresetData?.(preset.freq, preset.horizon);
            }}
          >
            <span className={styles.icon}>
              <Icon size={28} />
            </span>
            <span className={styles.name}>{preset.name}</span>
            <span className={styles.description}>{preset.description}</span>
            <span className={styles.params}>
              {formatFreqHorizon(preset.freq)} / {formatFreqHorizon(preset.horizon)}
            </span>
          </button>
        );
      })}
    </div>
  );
}
