"use client";

import { useState } from "react";
import { EmptyState, Kpi, Panel } from "./components";
import { useDashboardData } from "./data";
import { MARKETS, MARKET_LABELS, SCORECARD_KEY, deltaTone, fmt, forwardClockPaused, intervalText, isStale, latestWeekEvents, parseScorecard, signedFmt, type ScorecardEvent } from "./scorecard-rules";

function betsLine(event: ScorecardEvent): string {
  const parts = (["live", "shadow", "placed"] as const)
    .filter((kind) => event.bets[kind]?.n)
    .map((kind) => `${kind} ${event.bets[kind].settled}/${event.bets[kind].n} settled, ${signedFmt(event.bets[kind].pnl_units, 2)} units`);
  return parts.length ? parts.join("; ") : "none recorded";
}

function EventTable({ event }: { event: ScorecardEvent }) {
  return (
    <div>
      <h3 className="inputs-h3">
        {event.name} <span className="inputs-muted">({event.tour.toUpperCase()}, {event.event_start}; {event.status})</span>
      </h3>
      <div className="table-scroll">
        <table aria-label={`Model scorecard for ${event.name}`}>
          <thead>
            <tr>
              <th><span className="th-text">Model version</span></th>
              <th><span className="th-text">Skill error R1+R2</span></th>
              <th><span className="th-text">vs production</span></th>
              {MARKETS.map((market) => <th key={market}><span className="th-text">{MARKET_LABELS[market]} vs prod</span></th>)}
              <th><span className="th-text">Finish vs close</span></th>
            </tr>
          </thead>
          <tbody>
            {event.arms.map((arm) => (
              <tr key={arm.arm} className={arm.arm === "challenger" ? "active-row" : undefined}>
                <td>{arm.label}</td>
                <td>{fmt(arm.rmse)}</td>
                <td className={deltaTone(arm.rmse_vs_champion)}>{arm.arm === "champion" ? "baseline" : signedFmt(arm.rmse_vs_champion)}</td>
                {MARKETS.map((market) => {
                  const value = arm.logloss_vs_champion?.[market];
                  return <td key={market} className={deltaTone(value)}>{arm.arm === "champion" ? "—" : signedFmt(value, 4)}</td>;
                })}
                <td className={deltaTone(arm.logloss_vs_close_mean)}>{signedFmt(arm.logloss_vs_close_mean, 4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="inputs-muted">Bets: {betsLine(event)}</p>
    </div>
  );
}

/** golfprice (production) scorecard inside the Scorecard and P&L view: latest week by model version, the forward record and plain-English headlines (golfprice/scorecard/latest.json). */
export function GolfpriceScorecardSection() {
  const { data, loading } = useDashboardData<unknown>(SCORECARD_KEY);
  const [now] = useState(() => Date.now());
  const card = parseScorecard(data);
  const events = card ? latestWeekEvents(card) : [];

  return (
    <Panel title="golfprice (production) scorecard" eyebrow="golfprice versus the previous production model" className="scorecard-panel">
      {loading ? (
        <p className="inputs-muted">Loading the weekly scorecard…</p>
      ) : !card ? (
        <EmptyState title="No scorecard published yet" detail="It appears after the first Monday settle publishes the weekly scorecard. Negative deltas mean golfprice did better." />
      ) : (
        <div className="scorecard-body">
          {isStale(card.generated_at, now) && <p className="inputs-note accent">This scorecard was generated {card.generated_at.slice(0, 10)}, more than 8 days ago; a newer settle has not published.</p>}
          {forwardClockPaused(card) && <p className="inputs-note accent" role="status" data-testid="forward-clock-paused">Forward clock paused: parity record pending owner. No event is counting toward the forward record until it is appended.</p>}
          <ul className="inputs-note scorecard-headlines">{card.headlines.map((line) => <li key={line}>{line}</li>)}</ul>
          <div className="kpi-grid">
            <Kpi label="Events counted" value={`${card.forward.events_counted} of ${card.forward.futility_check_at}`} detail={`Futility check; horizon ${card.forward.horizon_events} events`} tone="accent" />
            <Kpi label="Skill error vs previous model" value={intervalText(card.forward.by_arm.challenger?.rmse_vs_champion)} detail="Strokes, forward record, negative is better" />
            <Kpi label="Finish log-loss vs previous model" value={intervalText(card.forward.by_arm.challenger?.logloss_vs_champion_mean, 4)} detail="Average over markets, negative is better" />
            <Kpi label="Live bets" value={`${signedFmt(card.forward.bets.live?.pnl_units ?? 0, 2)}u`} detail={`Shadow ${signedFmt(card.forward.bets.shadow?.pnl_units ?? 0, 2)}u`} />
          </div>
          {events.length === 0 ? <EmptyState title="No settled event yet" detail="The latest week table fills in once a settle has run." /> : events.map((event) => <EventTable key={event.event_uid} event={event} />)}
          <p className="inputs-muted">
            Generated {card.generated_at}. The forward record counts PGA live events after the forward clock start{card.forward.clock_start_event_uid ? ` (${card.forward.clock_start_event_uid})` : ""}.
          </p>
        </div>
      )}
    </Panel>
  );
}
