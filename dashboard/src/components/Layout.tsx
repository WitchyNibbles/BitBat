import type { ReactNode } from 'react';
import { Sidebar } from './Sidebar';
import type { Page } from './Sidebar';
import styles from './Layout.module.css';

const PAGE_TITLES: Record<Page, string> = {
  home: 'Dashboard',
  quickstart: 'Quick Start',
  performance: 'Performance',
  settings: 'Settings',
  system: 'System Health',
};

interface LayoutProps {
  activePage: Page;
  onNavigate: (page: Page) => void;
  statusColor?: 'success' | 'amber' | 'danger';
  children: ReactNode;
}

export function Layout({
  activePage,
  onNavigate,
  statusColor = 'amber',
  children,
}: LayoutProps) {
  return (
    <div className={styles.layout}>
      <Sidebar activePage={activePage} onNavigate={onNavigate} />
      <main className={styles.main}>
        <header className={styles.header}>
          <h1>{PAGE_TITLES[activePage]}</h1>
          <span className={`${styles.statusDot} ${styles[statusColor]}`} />
        </header>
        <div className={styles.content}>
          {children}
        </div>
      </main>
    </div>
  );
}
