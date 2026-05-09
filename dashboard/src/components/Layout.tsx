import type { ReactNode } from 'react';
import { PAGE_META } from '../navigation';
import type { AppPageKey } from '../navigation';
import { Sidebar } from './Sidebar';
import styles from './Layout.module.css';

interface LayoutProps {
  activePage: AppPageKey;
  children: ReactNode;
}

export function Layout({ activePage, children }: LayoutProps) {
  const meta = PAGE_META[activePage];

  return (
    <div className={styles.layout}>
      <Sidebar activePage={activePage} />
      <main className={styles.main}>
        <header className={styles.header}>
          <div>
            <div className={styles.eyebrow}>{meta.eyebrow}</div>
            <h1>{meta.label}</h1>
            <p className={styles.cue}>{meta.cue}</p>
          </div>
          <div className={styles.headerMeta}>
            <span className={styles.paperBadge}>paper only</span>
            <span className={styles.statusDot} aria-hidden="true" />
          </div>
        </header>
        <div className={styles.content}>{children}</div>
      </main>
    </div>
  );
}
