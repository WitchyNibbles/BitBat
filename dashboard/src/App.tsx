import { useState } from 'react';
import type { ReactNode } from 'react';
import { Layout } from './components/Layout';
import type { Page } from './components/Sidebar';
import { Home } from './pages/Home';
import { QuickStart } from './pages/QuickStart';
import { Performance } from './pages/Performance';
import { Settings } from './pages/Settings';
import { System } from './pages/System';

const PAGES: Record<Page, () => ReactNode> = {
  home: Home,
  quickstart: QuickStart,
  performance: Performance,
  settings: Settings,
  system: System,
};

export default function App() {
  const [page, setPage] = useState<Page>('home');
  const PageComponent = PAGES[page];

  return (
    <Layout activePage={page} onNavigate={setPage}>
      <PageComponent />
    </Layout>
  );
}
