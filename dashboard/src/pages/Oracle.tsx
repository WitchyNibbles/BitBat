import { MoonStar, RefreshCcw, ShieldAlert, Sparkles, Waves } from 'lucide-react';
import { useEffect, useEffectEvent, useState } from 'react';
import {
  formatBtc,
  formatPct,
  formatTimestamp,
  formatUsd,
  toPaperCockpitViewModel,
} from '../api/paperViewModel.ts';
import type {
  V2EventMessage,
  V2HealthResponse,
  V2PaperCockpitResponse,
  V2SignalResponse,
} from '../api/v2Client.ts';
import { V2ApiError, v2Api } from '../api/v2Client.ts';
import { useApi } from '../hooks/useApi.ts';
import styles from './Oracle.module.css';

interface OracleSnapshot {
  health: V2HealthResponse;
  paper: V2PaperCockpitResponse;
  signal: V2SignalResponse | null;
}

function toMessage(error: unknown): string {
  if (error instanceof V2ApiError && error.status === 401) {
    return 'Invalid operator token. Set VITE_V2_OPERATOR_TOKEN for the React dashboard.';
  }
  return error instanceof Error ? error.message : String(error);
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

async function loadOracleSnapshot(): Promise<OracleSnapshot> {
  const [health, paper, signalResult] = await Promise.all([
    v2Api.health(),
    v2Api.paper(),
    v2Api.latestSignal().catch((error: unknown) => {
      const message = toMessage(error);
      if (message.includes('No v2 signal')) {
        return null;
      }
      throw error;
    }),
  ]);

  return {
    health,
    paper,
    signal: signalResult,
  };
}

export function Oracle() {
  const snapshot = useApi(loadOracleSnapshot, []);
  const [events, setEvents] = useState<V2EventMessage[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [apiBaseUrl, setApiBaseUrl] = useState(v2Api.baseUrlHint);
  const refetchSnapshot = useEffectEvent(() => {
    snapshot.refetch();
  });

  useEffect(() => {
    let closed = false;
    let source: EventSource | null = null;

    const eventTypes = [
      'candle.closed',
      'features.computed',
      'signal.generated',
      'decision.made',
      'order.paper_filled',
      'portfolio.updated',
      'alert.raised',
    ] as const;

    const connect = async () => {
      try {
        const resolvedBaseUrl = await v2Api.resolveBaseUrl();
        if (closed) {
          return;
        }
        setApiBaseUrl(resolvedBaseUrl);
        const streamUrl = await v2Api.streamUrl();
        if (closed) {
          return;
        }
        source = new EventSource(streamUrl);

        eventTypes.forEach((eventType) => {
          source?.addEventListener(eventType, (message) => {
            const raw = 'data' in message ? String(message.data) : '{}';
            let data: Record<string, unknown> = {};
            try {
              data = JSON.parse(raw) as Record<string, unknown>;
            } catch {
              data = { raw };
            }
            setEvents((current) =>
              [{ id: message.lastEventId, type: eventType, data }, ...current].slice(0, 16),
            );
            refetchSnapshot();
          });
        });
      } catch (error: unknown) {
        if (!closed) {
          setActionError(toMessage(error));
        }
      }
    };

    void connect();

    return () => {
      closed = true;
      source?.close();
    };
  }, []);

  const runAction = async (label: string, action: () => Promise<unknown>) => {
    setBusy(label);
    setActionError(null);
    try {
      await action();
      snapshot.refetch();
    } catch (error: unknown) {
      setActionError(toMessage(error));
    } finally {
      setBusy(null);
    }
  };

  if (snapshot.loading && !snapshot.data) {
    return <div className="emptyState">Resolving live signal and control state...</div>;
  }

  if (snapshot.error && !snapshot.data) {
    return <div className="errorState">{toMessage(snapshot.error)}</div>;
  }

  if (!snapshot.data) {
    return <div className="emptyState">Oracle state is unavailable.</div>;
  }

  const { health, paper, signal } = snapshot.data;
  const model = toPaperCockpitViewModel(paper);

  return (
    <div className={`pageStack ${styles.page}`}>
      <section className="pageHero">
        <span className="eyebrowLabel">live signal ritual</span>
        <div className="heroTitleRow">
          <div>
            <h2>Oracle chamber</h2>
            <p className="lede">
              This route handles live runtime context and manual operator actions. Signal, decision,
              order, and portfolio state remain visibly separate so the console never implies more
              certainty than the backend actually provides.
            </p>
          </div>
        </div>
        <div className="statusStrip" role="status" aria-live="polite">
          <span className="statusPill" data-tone={model.freshness.tone}>
            <RefreshCcw size={14} />
            {model.freshness.label}
          </span>
          <span
            className="statusPill"
            data-tone={health.trading_paused ? 'warning' : 'positive'}
          >
            <ShieldAlert size={14} />
            Trading {health.trading_paused ? 'paused' : 'running'}
          </span>
          <span className="statusPill" data-tone={signal ? model.signal.tone : 'neutral'}>
            <MoonStar size={14} />
            {signal ? model.signal.label : 'No signal yet'}
          </span>
        </div>
      </section>

      {actionError ? <div className="errorState">{actionError}</div> : null}

      <section className="metricGrid">
        <article className="metricCard">
          <span className="metricLabel">Resolved v2 API</span>
          <span className="metricValue">{apiBaseUrl}</span>
          <span className="metricDetail">Operator token stays in request headers only.</span>
        </article>
        <article className="metricCard">
          <span className="metricLabel">Current signal</span>
          <span className="metricValue">{signal?.direction?.toUpperCase() ?? 'NONE'}</span>
          <span className="metricDetail">{signal ? formatPct(signal.confidence) : 'Waiting for first signal'}</span>
        </article>
        <article className="metricCard">
          <span className="metricLabel">Paper equity</span>
          <span className="metricValue">{formatUsd(paper.performance.equity)}</span>
          <span className="metricDetail">{formatBtc(paper.portfolio.position_qty)}</span>
        </article>
        <article className="metricCard">
          <span className="metricLabel">Last event</span>
          <span className="metricValue">{formatTimestamp(health.last_event_at)}</span>
          <span className="metricDetail">{health.event_count} runtime events recorded</span>
        </article>
      </section>

      <section className="surfaceCard">
        <div className="surfaceHeader">
          <div>
            <h3>Operator controls</h3>
            <p>Passive telemetry is separate from actions that can change paper state.</p>
          </div>
        </div>
        <div className="actionRow">
          <button
            className="actionButton"
            disabled={busy !== null}
            onClick={() =>
              runAction('simulate-candle', () =>
                v2Api.simulateCandle((paper.portfolio.mark_price || 100_000) + 250),
              )
            }
          >
            <Sparkles size={16} />
            Cast demo candle
          </button>
          <button
            className="actionButton"
            disabled={busy !== null}
            onClick={() => runAction('sync-market', () => v2Api.syncMarket())}
          >
            <Waves size={16} />
            Pull live Coinbase candle
          </button>
          {health.trading_paused ? (
            <button
              className="actionButton"
              disabled={busy !== null}
              onClick={() => runAction('resume', () => v2Api.resume())}
            >
              Resume trading
            </button>
          ) : (
            <button
              className="actionButton dangerButton"
              disabled={busy !== null}
              onClick={() => {
                if (window.confirm('Pause paper trading? This stops new automated actions.')) {
                  void runAction('pause', () => v2Api.pause());
                }
              }}
            >
              Pause trading
            </button>
          )}
        </div>
      </section>

      <section className="panelGrid">
        <article className="surfaceCard">
          <div className="surfaceHeader">
            <div>
              <h3>Signal and portfolio context</h3>
              <p>Signal is a forecast. Portfolio is the account mark.</p>
            </div>
          </div>
          <div className="kvGrid">
            <div className="kvItem">
              <span className="kvLabel">Signal generated</span>
              <span className="kvValue">
                {signal ? formatTimestamp(signal.generated_at) : 'No signal yet'}
              </span>
            </div>
            <div className="kvItem">
              <span className="kvLabel">Predicted return</span>
              <span className="kvValue">
                {signal ? formatPct(signal.predicted_return) : 'Unavailable'}
              </span>
            </div>
            <div className="kvItem">
              <span className="kvLabel">Mark price</span>
              <span className="kvValue">{formatUsd(paper.portfolio.mark_price)}</span>
            </div>
            <div className="kvItem">
              <span className="kvLabel">Equity</span>
              <span className="kvValue">{formatUsd(paper.performance.equity)}</span>
            </div>
          </div>
          {signal?.reasons?.length ? (
            <ul className="plainList">
              {signal.reasons.map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
          ) : null}
        </article>

        <article className="surfaceCard">
          <div className="surfaceHeader">
            <div>
              <h3>Latest paper orders</h3>
              <p>Order intent and fill outcome stay visible without leaving the route.</p>
            </div>
          </div>
          {paper.recent_orders.length > 0 ? (
            <div className="tableScroll">
              <table className="ledgerTable">
                <thead>
                  <tr>
                    <th>Created</th>
                    <th>Side</th>
                    <th>Qty</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {paper.recent_orders.map((order) => (
                    <tr key={order.order_id}>
                      <td>{formatTimestamp(order.created_at)}</td>
                      <td>{order.side}</td>
                      <td>{formatBtc(order.quantity_btc)}</td>
                      <td>{order.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="emptyState">No paper orders recorded yet.</div>
          )}
        </article>
      </section>

      <section className="surfaceCard">
        <div className="surfaceHeader">
          <div>
            <h3>Runtime event feed</h3>
            <p>`role="log"` because the operator needs sequential updates, not decorative motion.</p>
          </div>
        </div>
        {events.length > 0 ? (
          <div className="logList" role="log" aria-live="polite">
            {events.map((event, index) => (
              <div key={`${event.id ?? event.type}-${index}`} className="logEntry">
                <div className="logMeta">
                  <span>{event.type}</span>
                  {event.id ? <span>ID {event.id}</span> : null}
                </div>
                <div>{describeEvent(event)}</div>
              </div>
            ))}
          </div>
        ) : (
          <div className="emptyState">
            Waiting for the event stream. New candles, signals, fills, and alerts will appear here.
          </div>
        )}
      </section>
    </div>
  );
}
