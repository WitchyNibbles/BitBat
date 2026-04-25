import { Home, Zap, BarChart3, Settings, Activity, MoonStar } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import styles from './Sidebar.module.css';

export type Page = 'oracle' | 'home' | 'quickstart' | 'performance' | 'settings' | 'system';

interface NavItem {
  page: Page;
  label: string;
  icon: LucideIcon;
}

const NAV_ITEMS: NavItem[] = [
  { page: 'oracle', label: 'Oracle', icon: MoonStar },
  { page: 'home', label: 'Dashboard', icon: Home },
  { page: 'quickstart', label: 'Quick Start', icon: Zap },
  { page: 'performance', label: 'Performance', icon: BarChart3 },
  { page: 'settings', label: 'Settings', icon: Settings },
  { page: 'system', label: 'System', icon: Activity },
];

interface SidebarProps {
  activePage: Page;
  onNavigate: (page: Page) => void;
}

export function Sidebar({ activePage, onNavigate }: SidebarProps) {
  return (
    <aside className={styles.sidebar}>
      <div className={styles.brand}>
        <span className={styles.sigil}>&#x2B21;</span>
        <span className={styles.title}>BitBat</span>
      </div>

      <div className="divider">&#x2726;</div>

      <nav className={styles.nav}>
        {NAV_ITEMS.map(({ page, label, icon: Icon }) => (
          <button
            key={page}
            className={`${styles.navItem} ${activePage === page ? styles.active : ''}`}
            onClick={() => onNavigate(page)}
          >
            <Icon size={16} />
            <span>{label}</span>
          </button>
        ))}
      </nav>

      <div className={styles.footer}>
        <span className={styles.version}>v0.2.0 / clean-room</span>
      </div>
    </aside>
  );
}
