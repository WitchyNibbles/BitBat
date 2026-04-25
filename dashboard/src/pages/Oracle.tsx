import { useEffect, useState } from 'react';
import { MoonStar, Sparkles, ShieldAlert, Waves } from 'lucide-react';
import { v2Api, type V2EventMessage, type V2HealthResponse, type V2OrdersResponse, type V2PortfolioResponse, type V2SignalResponse } from '../api/v2Client.ts';
import styles from './Oracle.module.css';

function fmtUsd(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 2,
  }).format(value);
}

function fmtPct(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

function describeEvent(event: V2EventMessage): string {
  switch (event.type) {
    case 'candle.closed':
      return `Closed candle at ${String(event.data.close ?? '?')} with volume ${String(event.data.volume ?? '?')}.`;
    case 'signal.generated':
      return `Signal ${String(event.data.direction ?? 'hold')} with projected move ${String(event.data.predicted_return ?? '?')}.`;
    case 'decision.made':
      return `Decision ${String(event.data.action ?? 'hold')} because ${String(event.data.reason ?? 'n/a')}.`;
    case 'order.paper_filled':
      return `Paper order ${String(event.data.side ?? '?')} filled at ${String(event.data.fill_price ?? '?')}.`;
    case 'portfolio.updated':
      return `Equity now ${String(event.data.equity ?? '?')} with position ${String(event.data.position_qty ?? '?')} BTC.`;
    case 'alert.raised':
      return String(event.data.message ?? 'Alert raised.');
    default:
      return JSON.stringify(event.data);
  }
}

export function Oracle() {
  const [health, setHealth] = useState<V2HealthResponse | null>(null);
  const [signal, setSignal] = useState<V2SignalResponse | null>(null);
  const [portfolio, setPortfolio] = useState<V2PortfolioResponse | null>(null);
  const [orders, setOrders] = useState<V2OrdersResponse['orders']>([]);
  const [events, setEvents] = useState<V2EventMessage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  useEffect(() => {
    let closed = false;

    const load = async () => {
      const [healthResult, signalResult, portfolioResult, ordersResult] = await Promise.allSettled([
        v2Api.health(),
        v2Api.latestSignal(),
        v2Api.portfolio(),
        v2Api.orders(),
      ]);
      if (closed) return;

      if (healthResult.status === 'fulfilled') {
        setHealth(healthResult.value);
      } else {
        setError(
          healthResult.reason instanceof Error
            ? healthResult.reason.message
            : String(healthResult.reason),
        );
      }

      if (signalResult.status === 'fulfilled') {
        setSignal(signalResult.value);
      } else {
        const message =
          signalResult.reason instanceof Error
            ? signalResult.reason.message
            : String(signalResult.reason);
        if (message.includes('No v2 signal')) {
          setSignal(null);
        } else {
          setError(message);
        }
      }

      if (portfolioResult.status === 'fulfilled') {
        setPortfolio(portfolioResult.value);
      }

      if (ordersResult.status === 'fulfilled') {
        setOrders(ordersResult.value.orders);
      } else {
        setOrders([]);
      }
    };

    void load();

    const source = new EventSource(v2Api.streamUrl);
    source.onmessage = (message) => {
      const type = message.type || 'message';
      try {
        const data = JSON.parse(message.data) as Record<string, unknown>;
        setEvents((current) => [{ id: message.lastEventId, type, data }, ...current].slice(0, 18));
      } catch {
        setEvents((current) => [{ id: message.lastEventId, type, data: { raw: message.data } }, ...current].slice(0, 18));
      }
      void load();
    };
    const eventTypes = [
      'candle.closed',
      'features.computed',
      'signal.generated',
      'decision.made',
      'order.paper_filled',
      'portfolio.updated',
      'alert.raised',
    ] as const;
    eventTypes.forEach((eventType) => {
      source.addEventListener(eventType, (message) => {
        const payload = 'data' in message ? String(message.data) : '{}';
        try {
          const data = JSON.parse(payload) as Record<string, unknown>;
          setEvents((current) =>
            [{ id: message.lastEventId, type: eventType, data }, ...current].slice(0, 18),
          );
        } catch {
          setEvents((current) =>
            [{ id: message.lastEventId, type: eventType, data: { raw: payload } }, ...current].slice(0, 18),
          );
        }
        void load();
      });
    });
    source.onerror = () => {
      if (!closed) {
        setError('The ritual stream is unavailable. Start the v2 API on port 8100 or set VITE_V2_API_URL.');
      }
    };

    return () => {
      closed = true;
      source.close();
    };
  }, []);

  const runAction = async (label: string, action: () => Promise<unknown>) => {
    setBusy(label);
    setError(null);
    try {
      await action();
      const [nextHealth, nextSignal, nextPortfolio, nextOrders] = await Promise.all([
        v2Api.health(),
        v2Api.latestSignal(),
        v2Api.portfolio(),
        v2Api.orders(),
      ]);
      setHealth(nextHealth);
      setSignal(nextSignal);
      setPortfolio(nextPortfolio);
      setOrders(nextOrders.orders);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  const directionClass =
    signal?.direction === 'buy' ? styles.good : signal?.direction === 'sell' ? styles.bad : styles.neutral;

  return (
    <div className={styles.page}>
      <section className={styles.hero}>
        <div className={styles.heroTop}>
          <div>
            <div className={styles.eyebrow}>
              <MoonStar size={14} />
              witch market operator console
            </div>
            <h2 className={styles.title}>The Oracle Chamber</h2>
            <p className={styles.lede}>
              BitBat v2 watches BTC-USD, casts a deterministic signal, papers the trade, and records
              the outcome in a live ledger you can interrupt at any time.
            </p>
          </div>

          <div className={styles.signalBadge}>
            <div className={styles.signalLabel}>Current omen</div>
            <div className={`${styles.signalValue} ${directionClass}`}>
              {signal?.direction ?? 'loading'}
            </div>
            <div className={styles.signalMeta}>
              {signal ? `${fmtPct(signal.predicted_return)} projected move` : 'Waiting for a signal'}
            </div>
          </div>
        </div>

        <div className={styles.controls}>
          <button
            className={styles.button}
            disabled={busy !== null}
            onClick={() =>
              runAction('simulate', () =>
                v2Api.simulateCandle((portfolio?.mark_price ?? 100_000) + 250),
              )
            }
          >
            <Sparkles size={16} /> Cast Demo Candle
          </button>
          <button
            className={styles.button}
            disabled={busy !== null}
            onClick={() => runAction('sync-market', () => v2Api.syncMarket())}
          >
            Pull Live Coinbase Candle
          </button>
          {health?.trading_paused ? (
            <button
              className={styles.button}
              disabled={busy !== null}
              onClick={() => runAction('resume', () => v2Api.resume())}
            >
              <Waves size={16} /> Resume Trading
            </button>
          ) : (
            <button
              className={`${styles.button} ${styles.buttonDanger}`}
              disabled={busy !== null}
              onClick={() => runAction('pause', () => v2Api.pause())}
            >
              <ShieldAlert size={16} /> Pause Trading
            </button>
          )}
          <button
            className={`${styles.button} ${styles.buttonDanger}`}
            disabled={busy !== null}
            onClick={() => {
              if (!window.confirm('Reset the paper account and clear the current paper orders?')) {
                return;
              }
              void runAction('reset', () => v2Api.resetPaper());
            }}
          >
            Reset Paper Account
          </button>
          <button
            className={styles.button}
            disabled={busy !== null}
            onClick={() => runAction('retrain', () => v2Api.retrain())}
          >
            Request Retrain
          </button>
          <button
            className={styles.button}
            disabled={busy !== null}
            onClick={() =>
              runAction('acknowledge', () =>
                v2Api.acknowledge('operator acknowledged oracle alert'),
              )
            }
          >
            Acknowledge Alert
          </button>
        </div>

        <div className={styles.statusLine}>
          <span><span className={styles.dot} /> {health?.status ?? 'unknown'} runtime</span>
          <span>{health?.venue ?? 'coinbase'} / {health?.product_id ?? 'BTC-USD'}</span>
          <span>{health?.trading_paused ? 'trading paused' : 'trading armed for paper mode'}</span>
          <span>{health ? `${health.event_count} recorded events` : 'connecting...'}</span>
          {busy ? <span>running {busy}...</span> : null}
        </div>
        {error ? (
          <p className={styles.error} role="status" aria-live="polite">
            {error}
          </p>
        ) : null}
      </section>

      <section className={styles.grid}>
        <article className={`${styles.card} ${styles.span4}`}>
          <div className={styles.cardTitle}>Portfolio Equity</div>
          <div className={styles.metric}>{portfolio ? fmtUsd(portfolio.equity) : '...'}</div>
          <div className={styles.subline}>
            Cash {portfolio ? fmtUsd(portfolio.cash) : '...'} · Position {portfolio?.position_qty ?? 0} BTC
          </div>
        </article>

        <article className={`${styles.card} ${styles.span4}`}>
          <div className={styles.cardTitle}>Projected Price</div>
          <div className={styles.metric}>{signal ? fmtUsd(signal.predicted_price) : '...'}</div>
          <div className={styles.subline}>
            Confidence {signal ? fmtPct(signal.confidence) : '...'} · Model {signal?.model_name ?? '...'}
          </div>
        </article>

        <article className={`${styles.card} ${styles.span4}`}>
          <div className={styles.cardTitle}>PnL</div>
          <div className={styles.metric}>
            {portfolio ? fmtUsd(portfolio.realized_pnl + portfolio.unrealized_pnl) : '...'}
          </div>
          <div className={styles.subline}>
            Realized {portfolio ? fmtUsd(portfolio.realized_pnl) : '...'} · Unrealized{' '}
            {portfolio ? fmtUsd(portfolio.unrealized_pnl) : '...'}
          </div>
        </article>

        <article className={`${styles.card} ${styles.span5}`}>
          <div className={styles.cardTitle}>Signal Reasoning</div>
          <div className={styles.list}>
            {signal?.reasons?.length ? (
              signal.reasons.map((reason) => (
                <div key={reason} className={styles.row}>
                  <div>
                    <div className={styles.rowTitle}>{reason.split('=')[0]}</div>
                    <div className={styles.rowMeta}>Deterministic factor used in the current omen.</div>
                  </div>
                  <div className={styles.rowValue}>{reason.split('=')[1] ?? reason}</div>
                </div>
              ))
            ) : (
              <p className={styles.empty}>No reasoning captured yet.</p>
            )}
          </div>
        </article>

        <article className={`${styles.card} ${styles.span7}`}>
          <div className={styles.cardTitle}>Recent Paper Orders</div>
          <div className={styles.list}>
            {orders.length ? (
              orders.map((order) => (
                <div key={order.order_id} className={styles.row}>
                  <div>
                    <div className={styles.rowTitle}>
                      {order.side.toUpperCase()} · {order.quantity_btc} BTC
                    </div>
                    <div className={styles.rowMeta}>{new Date(order.created_at).toLocaleString()}</div>
                  </div>
                  <div className={styles.rowValue}>
                    {fmtUsd(order.fill_price)} · {order.status}
                  </div>
                </div>
              ))
            ) : (
              <p className={styles.empty}>No paper fills yet.</p>
            )}
          </div>
        </article>

        <article className={`${styles.card} ${styles.span12}`}>
          <div className={styles.cardTitle}>Live Ledger</div>
          <div className={styles.ledger} role="status" aria-live="polite">
            {events.length ? (
              events.map((event, index) => (
                <div key={`${event.id ?? 'evt'}-${index}`} className={styles.ledgerItem}>
                  <div className={styles.ledgerType}>{event.type}</div>
                  <div className={styles.ledgerText}>{describeEvent(event)}</div>
                </div>
              ))
            ) : (
              <p className={styles.empty}>
                The ledger is quiet. Cast a demo candle to wake the chamber.
              </p>
            )}
          </div>
        </article>
      </section>
    </div>
  );
}
