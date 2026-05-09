import { fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import App from './App';

vi.mock('./pages/Home', () => ({
  Home: () => <div>Command Center Mock</div>,
}));

vi.mock('./pages/Oracle', () => ({
  Oracle: () => <div>Oracle Mock</div>,
}));

vi.mock('./pages/PaperTrade', () => ({
  PaperTrade: () => <div>Paper Trade Mock</div>,
}));

vi.mock('./pages/Performance', () => ({
  Performance: () => <div>Performance Mock</div>,
}));

vi.mock('./pages/System', () => ({
  System: () => <div>System Mock</div>,
}));

vi.mock('./pages/Settings', () => ({
  Settings: () => <div>Settings Mock</div>,
}));

vi.mock('./pages/QuickStart', () => ({
  QuickStart: () => <div>Quick Start Mock</div>,
}));

describe('App routing', () => {
  it('supports direct paper-trade deep links', () => {
    render(
      <MemoryRouter initialEntries={['/paper-trade']}>
        <App />
      </MemoryRouter>,
    );

    expect(screen.getByText('Paper Trade Mock')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /paper trade/i })).toHaveAttribute(
      'href',
      '/paper-trade',
    );
  });

  it('navigates with semantic sidebar links', () => {
    render(
      <MemoryRouter initialEntries={['/command-center']}>
        <App />
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByRole('link', { name: /performance/i }));

    expect(screen.getByText('Performance Mock')).toBeInTheDocument();
  });
});
