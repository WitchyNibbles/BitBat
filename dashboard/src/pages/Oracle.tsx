import { useEffect, useState } from 'react';
import { MoonStar, Sparkles, ShieldAlert, Waves } from 'lucide-react';
import {
  v2Api,
  type V2EventMessage,
  type V2HealthResponse,
  type V2OrdersResponse,
  type V2PaperResponse,
  type V2PortfolioResponse,
  type V2SignalResponse,
} from '../api/v2Client.ts';
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

function fmtMaybePct(value: number | null | undefined): string {
  return typeof value === 'number' ? fmtPct(value) : 'n/a';
}

function describeEvent(event: V2EventMessage): string {
  switch (event.type) {
    case 'candle.closed':
      return `Closed candle at ${String(event.data.close ?? '?')} with volume ${String(event.data.volume ?? '?')}.`;
    case 'signal.generated':
      return `Signal ${String(event.data.direction ?? 'hold')} with expected value ${String(event.data.expected_value_return ?? '?')} after costs.`;
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

function buildLedgerEvents(
  health: V2HealthResponse | null,
  signal: V2SignalResponse | null,
  paper: V2PaperResponse | null,
  orders: V2OrdersResponse['orders'],
): V2EventMessage[] {
  const ledgerEvents: V2EventMessage[] = [];
  if (health !== null) {
    ledgerEvents.push({
      type: 'runtime.snapshot',
      data: {
        status: health.status,
        signal_source: health.signal_source,
        event_count: health.event_count,
        promotion_verdict: health.promotion?.verdict ?? 'missing',
      },
    });
  }
  if (signal !== null) {
    ledgerEvents.push({
      id: signal.signal_id,
      type: 'signal.generated',
      data: {
        direction: signal.direction,
        predicted_return: signal.predicted_return,
        expected_value_return: signal.expected_value_return,
        expected_cost_return: signal.expected_cost_return,
        model_name: signal.model_name,
      },
    });
  }
  if (paper !== null) {
    ledgerEvents.push({
      type: 'paper.snapshot',
      data: {
        hold_rate: paper.performance.hold_rate,
        action_rate: paper.performance.action_rate,
        promotion_verdict: paper.promotion?.verdict ?? 'missing',
      },
    });
  }
  orders.slice(0, 5).forEach((order) => {
    ledgerEvents.push({
      id: order.order_id,
      type: order.status === 'filled' ? 'order.paper_filled' : 'order.paper_pending',
      data: {
        side: order.side,
        fill_price: order.fill_price,
        quantity_btc: order.quantity_btc,
        status: order.status,
      },
    });
  });
  return ledgerEvents;
}

export function Oracle() {
  const [tokenDraft, setTokenDraft] = useState(v2Api.getOperatorToken());
  const [tokenConfigured, setTokenConfigured] = useState(v2Api.hasOperatorToken());
  const [health, setHealth] = useState<V2HealthResponse | null>(null);
  const [signal, setSignal] = useState<V2SignalResponse | null>(null);
  const [portfolio, setPortfolio] = useState<V2PortfolioResponse | null>(null);
  const [paper, setPaper] = useState<V2PaperResponse | null>(null);
  const [orders, setOrders] = useState<V2OrdersResponse['orders']>([]);
  const [events, setEvents] = useState<V2EventMessage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  useEffect(() => {
    if (!tokenConfigured) {
      setHealth(null);
      setSignal(null);
      setPortfolio(null);
      setPaper(null);
      setOrders([]);
      setEvents([]);
      return;
    }

    let cancelled = false;

    const load = async () => {
      const [healthResult, signalResult, portfolioResult, paperResult, ordersResult] = await Promise.allSettled([
        v2Api.health(),
        v2Api.latestSignal(),
        v2Api.portfolio(),
        v2Api.paper(),
        v2Api.orders(),
      ]);
      if (cancelled) return;

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

      if (paperResult.status === 'fulfilled') {
        setPaper(paperResult.value);
      } else {
        setPaper(null);
      }

      if (ordersResult.status === 'fulfilled') {
        setOrders(ordersResult.value.orders);
      } else {
        setOrders([]);
      }

      const nextHealth = healthResult.status === 'fulfilled' ? healthResult.value : null;
      const nextSignal = signalResult.status === 'fulfilled' ? signalResult.value : null;
      const nextPaper = paperResult.status === 'fulfilled' ? paperResult.value : null;
      const nextOrders =
        ordersResult.status === 'fulfilled' ? ordersResult.value.orders : [];
      setEvents(buildLedgerEvents(nextHealth, nextSignal, nextPaper, nextOrders));
    };

    void load();
    const poller = window.setInterval(() => {
      void load();
    }, 10_000);

    return () => {
      cancelled = true;
      window.clearInterval(poller);
    };
  }, [tokenConfigured]);

  const runAction = async (label: string, action: () => Promise<unknown>) => {
    setBusy(label);
    setError(null);
    try {
      await action();
      const [nextHealth, nextSignal, nextPortfolio, nextPaper, nextOrders] = await Promise.all([
        v2Api.health(),
        v2Api.latestSignal(),
        v2Api.portfolio(),
        v2Api.paper(),
        v2Api.orders(),
      ]);
      setHealth(nextHealth);
      setSignal(nextSignal);
      setPortfolio(nextPortfolio);
      setPaper(nextPaper);
      setOrders(nextOrders.orders);
      setEvents(buildLedgerEvents(nextHealth, nextSignal, nextPaper, nextOrders.orders));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  const directionClass =
    signal?.direction === 'buy' ? styles.good : signal?.direction === 'sell' ? styles.bad : styles.neutral;
  const promotion = paper?.promotion ?? health?.promotion ?? null;
  const promotionClass =
    promotion?.verdict === 'promotable'
      ? styles.good
      : promotion?.verdict === 'blocked'
        ? styles.bad
        : styles.neutral;

  const saveOperatorToken = () => {
    const savedToken = v2Api.setOperatorToken(tokenDraft);
    setTokenDraft(savedToken);
    setTokenConfigured(Boolean(savedToken));
    setError(null);
  };

  const clearOperatorToken = () => {
    v2Api.clearOperatorToken();
    setTokenDraft('');
    setTokenConfigured(false);
    setError(null);
  };

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
              the outcome in a live ledger you can interrupt at any time. Enter an operator token
              to unlock the controls for this session.
            </p>
          </div>

          <div className={styles.signalBadge}>
            <div className={styles.signalLabel}>Current omen</div>
            <div className={`${styles.signalValue} ${directionClass}`}>
              {signal?.direction ?? 'loading'}
            </div>
            <div className={styles.signalMeta}>
              {signal ? `${fmtPct(signal.expected_value_return)} EV after costs` : 'Waiting for a signal'}
            </div>
          </div>
        </div>

        <div className={styles.authRow}>
          <label className={styles.authLabel} htmlFor="oracle-operator-token">
            Operator token
          </label>
          <input
            id="oracle-operator-token"
            className={styles.authInput}
            type="password"
            autoComplete="off"
            value={tokenDraft}
            onChange={(event) => setTokenDraft(event.target.value)}
            placeholder="Set the v2 operator token"
          />
          <button className={styles.button} type="button" onClick={saveOperatorToken}>
            Use Token
          </button>
          <button
            className={styles.button}
            type="button"
            onClick={clearOperatorToken}
          >
            Clear Token
          </button>
          <span className={styles.authHint}>
            Stored in this browser session only.
          </span>
        </div>

        <div className={styles.controls}>
          <button
            className={styles.button}
            disabled={busy !== null || !tokenConfigured}
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
            disabled={busy !== null || !tokenConfigured}
            onClick={() => runAction('sync-market', () => v2Api.syncMarket())}
          >
            Pull Live Coinbase Candle
          </button>
          {health?.trading_paused ? (
            <button
              className={styles.button}
              disabled={busy !== null || !tokenConfigured}
              onClick={() => runAction('resume', () => v2Api.resume())}
            >
              <Waves size={16} /> Resume Trading
            </button>
          ) : (
            <button
              className={`${styles.button} ${styles.buttonDanger}`}
              disabled={busy !== null || !tokenConfigured}
              onClick={() => runAction('pause', () => v2Api.pause())}
            >
              <ShieldAlert size={16} /> Pause Trading
            </button>
          )}
          <button
            className={`${styles.button} ${styles.buttonDanger}`}
            disabled={busy !== null || !tokenConfigured}
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
            disabled={busy !== null || !tokenConfigured}
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
          <span><span className={styles.dot} /> {health?.status ?? 'locked'} runtime</span>
          <span>{health?.venue ?? 'coinbase'} / {health?.product_id ?? 'BTC-USD'}</span>
          <span>signal source {health?.signal_source ?? 'unknown'}</span>
          <span>model {health?.signal_model_name ?? signal?.model_name ?? 'waiting'}</span>
          <span>promotion {promotion?.verdict ?? 'missing'}</span>
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

        <article className={`${styles.card} ${styles.span3}`}>
          <div className={styles.cardTitle}>Signal Edge</div>
          <div className={styles.metric}>{signal ? fmtPct(signal.expected_value_return) : '...'}</div>
          <div className={styles.subline}>
            Move {signal ? fmtPct(signal.expected_move_return) : '...'} · Costs{' '}
            {signal ? fmtPct(signal.expected_cost_return) : '...'}
          </div>
        </article>

        <article className={`${styles.card} ${styles.span3}`}>
          <div className={styles.cardTitle}>Projected Price</div>
          <div className={styles.metric}>{signal ? fmtUsd(signal.predicted_price) : '...'}</div>
          <div className={styles.subline}>
            Confidence {signal ? fmtPct(signal.confidence) : '...'} · Model {signal?.model_name ?? '...'}
          </div>
        </article>

        <article className={`${styles.card} ${styles.span3}`}>
          <div className={styles.cardTitle}>PnL</div>
          <div className={styles.metric}>
            {paper ? fmtUsd(paper.performance.net_pnl) : portfolio ? fmtUsd(portfolio.realized_pnl + portfolio.unrealized_pnl) : '...'}
          </div>
          <div className={styles.subline}>
            Net {paper ? fmtMaybePct(paper.performance.net_pnl_pct) : '...'} · Trades{' '}
            {paper?.performance.trade_count ?? 0}
          </div>
        </article>

        <article className={`${styles.card} ${styles.span3}`}>
          <div className={styles.cardTitle}>Promotion Gate</div>
          <div className={`${styles.metric} ${promotionClass}`}>{promotion?.verdict ?? 'missing'}</div>
          <div className={styles.subline}>
            {promotion?.label_mode ?? 'n/a'} · {promotion?.model_version ?? promotion?.winner ?? 'waiting'}
          </div>
        </article>

        <article className={`${styles.card} ${styles.span6}`}>
          <div className={styles.cardTitle}>Probability Split</div>
          <div className={styles.list}>
            {signal ? (
              [
                ['Up', signal.p_up],
                ['Down', signal.p_down],
                ['Flat', signal.p_flat],
                ['Hold Rate', paper?.performance.hold_rate ?? 0],
                ['Action Rate', paper?.performance.action_rate ?? 0],
              ].map(([label, value]) => (
                <div key={String(label)} className={styles.row}>
                  <div>
                    <div className={styles.rowTitle}>{label}</div>
                    <div className={styles.rowMeta}>Live probability and runtime act/pass split.</div>
                  </div>
                  <div className={styles.rowValue}>{fmtPct(Number(value))}</div>
                </div>
              ))
            ) : (
              <p className={styles.empty}>No probability contract yet.</p>
            )}
          </div>
        </article>

        <article className={`${styles.card} ${styles.span6}`}>
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

        <article className={`${styles.card} ${styles.span6}`}>
          <div className={styles.cardTitle}>Replay Gate Evidence</div>
          <div className={styles.list}>
            {promotion ? (
              [
                ['Replay Trades', String(promotion.replay_trade_count ?? 0)],
                ['Replay Hold Rate', fmtMaybePct(promotion.replay_hold_rate)],
                ['Replay EV', fmtMaybePct(promotion.replay_mean_expected_value_return)],
                ['Replay Net PnL', fmtMaybePct(promotion.replay_net_pnl_pct)],
                ['Calibration', promotion.replay_calibration_brier?.toFixed(3) ?? 'n/a'],
              ].map(([label, value]) => (
                <div key={String(label)} className={styles.row}>
                  <div>
                    <div className={styles.rowTitle}>{label}</div>
                    <div className={styles.rowMeta}>Most recent promotion replay evidence.</div>
                  </div>
                  <div className={styles.rowValue}>{String(value)}</div>
                </div>
              ))
            ) : (
              <p className={styles.empty}>No replay gate evidence persisted yet.</p>
            )}
          </div>
        </article>

        <article className={`${styles.card} ${styles.span6}`}>
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
          <div className={styles.cardTitle}>Abstain Breakdown</div>
          <div className={styles.list}>
            {paper && Object.keys(paper.performance.abstain_breakdown).length ? (
              Object.entries(paper.performance.abstain_breakdown).map(([reason, count]) => (
                <div key={reason} className={styles.row}>
                  <div>
                    <div className={styles.rowTitle}>{reason}</div>
                    <div className={styles.rowMeta}>Why the runtime chose not to act.</div>
                  </div>
                  <div className={styles.rowValue}>{count}</div>
                </div>
              ))
            ) : (
              <p className={styles.empty}>No abstain reasons recorded yet.</p>
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
