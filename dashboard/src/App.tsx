import type { ReactNode } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { Layout } from './components/Layout';
import type { AppPageKey } from './navigation';
import { Home } from './pages/Home';
import { Oracle } from './pages/Oracle';
import { PaperTrade } from './pages/PaperTrade';
import { Performance } from './pages/Performance';
import { QuickStart } from './pages/QuickStart';
import { Settings } from './pages/Settings';
import { System } from './pages/System';

interface RoutedPageProps {
  page: AppPageKey;
  children: ReactNode;
}

function RoutedPage({ page, children }: RoutedPageProps) {
  return <Layout activePage={page}>{children}</Layout>;
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/command-center" replace />} />
      <Route
        path="/command-center"
        element={
          <RoutedPage page="command-center">
            <Home />
          </RoutedPage>
        }
      />
      <Route
        path="/oracle"
        element={
          <RoutedPage page="oracle">
            <Oracle />
          </RoutedPage>
        }
      />
      <Route
        path="/paper-trade"
        element={
          <RoutedPage page="paper-trade">
            <PaperTrade />
          </RoutedPage>
        }
      />
      <Route
        path="/performance"
        element={
          <RoutedPage page="performance">
            <Performance />
          </RoutedPage>
        }
      />
      <Route
        path="/system"
        element={
          <RoutedPage page="system">
            <System />
          </RoutedPage>
        }
      />
      <Route
        path="/settings"
        element={
          <RoutedPage page="settings">
            <Settings />
          </RoutedPage>
        }
      />
      <Route
        path="/quick-start"
        element={
          <RoutedPage page="quick-start">
            <QuickStart />
          </RoutedPage>
        }
      />
      <Route path="*" element={<Navigate to="/command-center" replace />} />
    </Routes>
  );
}
