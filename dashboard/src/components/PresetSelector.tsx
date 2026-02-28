import { Shield, Scale, Rocket } from 'lucide-react';
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
}

const PRESETS: PresetDef[] = [
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
];

export function PresetSelector({ selected, onSelect }: PresetSelectorProps) {
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
            onClick={() => onSelect(preset.id)}
          >
            <span className={styles.icon}>
              <Icon size={28} />
            </span>
            <span className={styles.name}>{preset.name}</span>
            <span className={styles.description}>{preset.description}</span>
            <span className={styles.params}>
              {preset.freq} / {preset.horizon}
            </span>
          </button>
        );
      })}
    </div>
  );
}
