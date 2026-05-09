import type { LucideIcon } from 'lucide-react';
import {
  Activity,
  BarChart3,
  Compass,
  FlaskConical,
  Home,
  MoonStar,
  Settings,
} from 'lucide-react';

export type AppPageKey =
  | 'command-center'
  | 'oracle'
  | 'paper-trade'
  | 'performance'
  | 'system'
  | 'settings'
  | 'quick-start';

export interface AppPageMeta {
  key: AppPageKey;
  path: string;
  label: string;
  cue: string;
  eyebrow: string;
  icon: LucideIcon;
  nav: boolean;
}

export const PAGE_META: Record<AppPageKey, AppPageMeta> = {
  'command-center': {
    key: 'command-center',
    path: '/command-center',
    label: 'Command Center',
    cue: 'At-a-glance runtime, freshness, and portfolio posture.',
    eyebrow: 'operator overview',
    icon: Home,
    nav: true,
  },
  oracle: {
    key: 'oracle',
    path: '/oracle',
    label: 'Oracle',
    cue: 'Signal context, live events, and operator controls.',
    eyebrow: 'signal and control',
    icon: MoonStar,
    nav: true,
  },
  'paper-trade': {
    key: 'paper-trade',
    path: '/paper-trade',
    label: 'Paper Trade',
    cue: 'Simulated account, fills, PnL, and ledger truth.',
    eyebrow: 'paper-only cockpit',
    icon: Compass,
    nav: true,
  },
  performance: {
    key: 'performance',
    path: '/performance',
    label: 'Performance',
    cue: 'Outcome analysis, benchmark delta, and trade quality.',
    eyebrow: 'performance ledger',
    icon: BarChart3,
    nav: true,
  },
  system: {
    key: 'system',
    path: '/system',
    label: 'System',
    cue: 'Diagnostics, schema readiness, and ingestion health.',
    eyebrow: 'diagnostics',
    icon: Activity,
    nav: true,
  },
  settings: {
    key: 'settings',
    path: '/settings',
    label: 'Settings',
    cue: 'Legacy model presets and threshold controls.',
    eyebrow: 'legacy tuning',
    icon: Settings,
    nav: true,
  },
  'quick-start': {
    key: 'quick-start',
    path: '/quick-start',
    label: 'Quick Start',
    cue: 'Transitional training flow for local experimentation.',
    eyebrow: 'guided setup',
    icon: FlaskConical,
    nav: true,
  },
};

export const NAV_ITEMS = (Object.values(PAGE_META) satisfies AppPageMeta[]).filter(
  (item) => item.nav,
);
