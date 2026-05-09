import { NavLink } from 'react-router-dom';
import { NAV_ITEMS } from '../navigation';
import type { AppPageKey } from '../navigation';
import styles from './Sidebar.module.css';

interface SidebarProps {
  activePage: AppPageKey;
}

export function Sidebar({ activePage }: SidebarProps) {
  return (
    <aside className={styles.sidebar}>
      <div className={styles.brand}>
        <span className={styles.sigil}>&#x2B21;</span>
        <div>
          <span className={styles.title}>BitBat</span>
          <span className={styles.subtitle}>witchy-tech operator console</span>
        </div>
      </div>

      <div className="divider">&#x2726;</div>

      <nav className={styles.nav} aria-label="Primary">
        {NAV_ITEMS.map(({ key, path, label, cue, icon: Icon }) => (
          <NavLink
            key={key}
            to={path}
            className={({ isActive }) =>
              `${styles.navItem} ${isActive || activePage === key ? styles.active : ''}`
            }
          >
            <Icon size={16} />
            <span className={styles.navText}>
              <span>{label}</span>
              <span className={styles.navCue}>{cue}</span>
            </span>
          </NavLink>
        ))}
      </nav>

      <div className={styles.footer}>
        <span className={styles.version}>v0.2.0 / clean-room</span>
        <span className={styles.version}>authoritative surface: React dashboard</span>
      </div>
    </aside>
  );
}
