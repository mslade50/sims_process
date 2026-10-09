"use client";

import { useState } from "react";
import { EmptyState, Kpi, Panel } from "./components";
import { useDashboardData } from "./data";
import { etTime } from "./lib";
import { FORWARD_RESTART_NOTE, MARKETS, MARKET_LABELS, SCORECARD_KEY, armName, dayLabel, deltaTone, fmt, forwardClockPaused, intervalText, isStale, latestWeekEvents, parseScorecard, shownArms, signedFmt, tourLabel, type ScorecardEvent } from "./scorecard-rules";

const SKILL_HINT = "How far off the model's skill estimates were in rounds 1 and 2, in strokes per round. Lower is better.";
const MARKET_HINT = "Prediction error against the market's closing prices for this market. Negative means the model was closer to what happened than the market was; positive means the market was closer.";

function betsLine(event: ScorecardEvent): string {
  const parts = ([["live", "live bets"], ["shadow", "paper bets (tracked, not placed)"], ["placed", "placed bets"]] as const)
    .filter(([kind]) => event.bets[kind]?.n)
    .map(([kind, name]) => {
      const bet = event.bets[kind];
      return `${bet.n} ${name}, ${bet.settled === bet.n ? "all settled" : `${bet.settled} settled`}, ${signedFmt(bet.pnl_units, 2)} units`;
    });
  return parts.length ? parts.join("; ") : "none recorded";
}

/** One sentence that says how the current model did, so the table is a reference rather than the message. */
function verdictLine(event: ScorecardEvent): string | null {
  const current = event.arms.find((arm) => arm.role === "production" || arm.arm === "challenger");
  if (!current) return null;
  const parts: string[] = [];
  if (current.rmse !== null && current.rmse !== undefined) {
    const d = current.rmse_vs_champion;
    parts.push(`Skill error was ${fmt(current.rmse, 2)} strokes a round${d !== null && d !== undefined && d !== 0 ? `, ${fmt(Math.abs(d), 3)} ${d < 0 ? "better" : "worse"} than the old model` : ""}.`);
  }
  if (current.markets_scored_vs_close) parts.push(`It beat the market's closing prices in ${current.markets_better_vs_close ?? 0} of ${current.markets_scored_vs_close} markets.`);
  return parts.length ? parts.join(" ") : null;
}

function EventTable({ event }: { event: ScorecardEvent }) {
  const arms = shownArms(event.arms);
  const markets = MARKETS.filter((market) => arms.some((arm) => arm.logloss_vs_close?.[market] !== undefined));
  const showAverage = arms.some((arm) => arm.logloss_vs_close_mean !== null && arm.logloss_vs_close_mean !== undefined);
  const verdict = verdictLine(event);
  return (
    <div>
      <h3 className="inputs-h3">
        {event.name} <span className="inputs-muted">({tourLabel(event.tour)}, started {dayLabel(event.event_start)}; {event.status})</span>
      </h3>
      {verdict && <p>{verdict}</p>}
      <div className="table-scroll">
        <table aria-label={`Model scorecard for ${event.name}`}>
          <thead>
            <tr>
              <th rowSpan={2}><span className="th-text">Model</span></th>
              <th rowSpan={2} title={SKILL_HINT}><span className="th-text">Skill error (strokes)</span></th>
              {(markets.length > 0 || showAverage) && <th colSpan={markets.length + (showAverage ? 1 : 0)} title={MARKET_HINT}><span className="th-text">Against the market close (negative is better)</span></th>}
            </tr>
            <tr>
              {markets.map((market) => <th key={market} title={MARKET_HINT}><span className="th-text">{MARKET_LABELS[market]}</span></th>)}
              {showAverage && <th title="The average of the markets to the left."><span className="th-text">Average</span></th>}
            </tr>
          </thead>
          <tbody>
            {arms.map((arm) => (
              <tr key={arm.arm} className={arm.role === "production" || arm.arm === "challenger" ? "active-row" : undefined}>
                <td>{armName(arm)}</td>
                <td>{fmt(arm.rmse)}</td>
                {markets.map((market) => {
                  const value = arm.logloss_vs_close?.[market];
                  return <td key={market} className={deltaTone(value)}>{signedFmt(value, 4)}</td>;
                })}
                {showAverage && <td className={deltaTone(arm.logloss_vs_close_mean)}>{signedFmt(arm.logloss_vs_close_mean, 4)}</td>}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="inputs-muted">Bets: {betsLine(event)}.</p>
    </div>
  );
}

/** The model scorecard inside the Scorecard and P&L view: latest settled week, the forward record, and a plain-English readout. */
export function GolfpriceScorecardSection() {
  const { data, loading } = useDashboardData<unknown>(SCORECARD_KEY);
  const [now] = useState(() => Date.now());
  const card = parseScorecard(data);
  const events = card ? latestWeekEvents(card) : [];
  const paused = forwardClockPaused(card);

  return (
    <Panel title="Model scorecard" eyebrow="How the current model did against the market" className="scorecard-panel">
      {loading ? (
        <p className="inputs-muted">Loading the weekly scorecard…</p>
      ) : !card ? (
        <EmptyState title="No scorecard yet" detail="It appears after the first Monday settle grades the previous week." />
      ) : (
        <div className="scorecard-body">
          {isStale(card.generated_at, now) && <p className="inputs-note accent">This scorecard was last updated {etTime(card.generated_at)}, more than 8 days ago. A newer settle has not published yet.</p>}
          {paused ? (
            <p className="inputs-note accent" role="status" data-testid="forward-clock-paused">{FORWARD_RESTART_NOTE}</p>
          ) : (
            <div className="kpi-grid">
              <Kpi label="Events counted" value={`${card.forward.events_counted} of ${card.forward.futility_check_at}`} detail={`First checkpoint; full test is ${card.forward.horizon_events} events`} tone="accent" hint="PGA events graded since the forward test started. At the checkpoint we decide whether the model is worth continuing with." />
              <Kpi label="Skill error vs previous model" value={intervalText(card.forward.by_arm.challenger?.rmse_vs_champion)} detail="Strokes a round; negative is better" hint={`${SKILL_HINT} The brackets show the likely range.`} />
              <Kpi label="Price error vs previous model" value={intervalText(card.forward.by_arm.challenger?.logloss_vs_champion_mean, 4)} detail="Average over markets; negative is better" hint="How far the model's finish-position probabilities were from what happened, compared with the previous model. Negative means closer." />
              {card.forward.bets.live?.n > 0 && <Kpi label="Live bets" value={`${signedFmt(card.forward.bets.live.pnl_units ?? 0, 2)}u`} detail={card.forward.bets.shadow?.n ? `Paper bets ${signedFmt(card.forward.bets.shadow.pnl_units ?? 0, 2)}u` : undefined} hint="Profit or loss in betting units on bets actually placed. Paper bets are tracked but not placed." />}
            </div>
          )}
          {events.length === 0 ? <EmptyState title="No graded event yet" detail="The latest week appears here once a settle has run." /> : events.map((event) => <EventTable key={event.event_uid} event={event} />)}
          <p className="inputs-muted">Updated {etTime(card.generated_at)}. Only PGA Tour events after the forward test starts count toward the forward record.</p>
        </div>
      )}
    </Panel>
  );
}
