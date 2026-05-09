import { useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { Layout } from './components/Layout';
import type { Page } from './components/Sidebar';
import { Oracle } from './pages/Oracle';
import { Home } from './pages/Home';
import { QuickStart } from './pages/QuickStart';
import { Performance } from './pages/Performance';
import { Settings } from './pages/Settings';
import { System } from './pages/System';

const ENABLE_LEGACY_PAGES = import.meta.env.VITE_ENABLE_LEGACY_PAGES === 'true';

const ALL_PAGES: Record<Page, () => ReactNode> = {
  oracle: Oracle,
  home: Home,
  quickstart: QuickStart,
  performance: Performance,
  settings: Settings,
  system: System,
};

const PAGES: Record<Page, () => ReactNode> = ENABLE_LEGACY_PAGES
  ? ALL_PAGES
  : { oracle: Oracle } as Record<Page, () => ReactNode>;

function resolveInitialPage(): Page {
  const hash = window.location.hash.replace('#', '');
  if (hash in PAGES) {
    return hash as Page;
  }
  return 'oracle';
}

export default function App() {
  const [page, setPage] = useState<Page>(resolveInitialPage);

  useEffect(() => {
    const onHashChange = () => {
      const next = window.location.hash.replace('#', '');
      if (next in PAGES) {
        setPage(next as Page);
      }
    };
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const navigate = (nextPage: Page) => {
    setPage(nextPage);
    window.location.hash = nextPage;
  };

  const PageComponent = PAGES[page];

  return (
    <Layout
      activePage={page}
      onNavigate={navigate}
      showLegacyPages={ENABLE_LEGACY_PAGES}
    >
      <PageComponent />
    </Layout>
  );
}
