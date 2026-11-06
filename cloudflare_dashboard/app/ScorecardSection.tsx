"use client";

import { useEffect, useState } from "react";
import { Chip, EmptyState, Section, Stat } from "./components";
import { useDashboardData } from "./data";
import { plainDate, scorecardLag, type EventLike } from "./freshness-rules";
import { etTime } from "./lib";
import { forwardHeadline, forwardRestartNote, MARKETS, MARKET_LABELS, SCORECARD_KEY, dayLabel, fmt, forwardClockPaused, gapBar, latestWeekEvents, marketReading, parseScorecard, shownArms, signedFmt, tourLabel, type ScorecardEvent } from "./scorecard-rules";
import "./performance.css";

const MARKET_HINT = "Prediction error against the market's closing prices for this market (log-loss). Negative means the model was closer to what happened than the market was; positive means the market was closer.";

function betsLine(event: ScorecardEvent): string {
  const parts = ([["live", "live bets"], ["shadow", "paper bets (tracked, not placed)"], ["placed", "placed bets"]] as const)
    .filter(([kind]) => event.bets[kind]?.n)
    .map(([kind, name]) => {
      const bet = event.bets[kind];
      return `${bet.n} ${bet.n === 1 ? name.replace("bets", "bet") : name}, ${bet.settled === bet.n ? "all settled" : `${bet.settled} settled`}, ${signedFmt(bet.pnl_units, 2)} units`;
    });
  return parts.length ? parts.join("; ") : "none recorded";
}

/** One sentence that says how the current model did, so the table is a reference rather than the message. */
function verdictLine(event: ScorecardEvent): string | null {
  const current = event.arms.find((arm) => arm.role === "production" || arm.arm === "challenger");
  if (!current) return null;
  const parts: string[] = [];
  if (current.rmse !== null && current.rmse !== undefined) parts.push(`Skill error was ${fmt(current.rmse, 2)} strokes a round.`);
  if (current.markets_scored_vs_close) parts.push(`It beat the market's closing prices in ${current.markets_better_vs_close ?? 0} of ${current.markets_scored_vs_close} markets.`);
  return parts.length ? parts.join(" ") : null;
}

/** Markets as rows, each with a signed gap against the market close and a zero-centred bar (left = model closer, right = market closer). */
function EventTable({ event }: { event: ScorecardEvent }) {
  const arms = shownArms(event.arms);
  const markets = MARKETS.filter((market) => arms.some((arm) => arm.logloss_vs_close?.[market] !== undefined));
  const max = Math.max(0, ...arms.flatMap((arm) => markets.map((m) => Math.abs(arm.logloss_vs_close?.[m] ?? 0))));
  const verdict = verdictLine(event);
  return (
    <div className="sc-event">
      <h3>{event.name} <small>{tourLabel(event.tour)}, started {dayLabel(event.event_start)}, {event.status}</small></h3>
      {verdict && <p className="sc-verdict">{verdict}</p>}
      {markets.length > 0 && <p className="sc-note">Negative gap and a bar to the left mean the model was closer to what happened; positive and right mean the market was.</p>}
      {markets.length > 0 && (
        <div className="table-scroll">
          <table className="sc-table" aria-label={`Model against the market close for ${event.name}`}>
            <thead>
              <tr>
                <th scope="col">Market</th>
                {arms.map((arm) => <th key={arm.arm} scope="col" className="num" title={MARKET_HINT}>Gap to market close <span className="sc-unit">log-loss, zero at centre</span></th>)}
              </tr>
            </thead>
            <tbody>
              {markets.map((market) => (
                arms.map((arm, i) => {
                  const value = arm.logloss_vs_close?.[market];
                  const reading = marketReading(value);
                  return (
                    <tr key={`${market}${arm.arm}`}>
                      {i === 0 && <th scope="row" rowSpan={arms.length}>{MARKET_LABELS[market]}</th>}
                      <td className={`num ${reading.tone}`}>{signedFmt(value, 4)}<span className="sc-word">{reading.word}</span>
                        <span className="sc-gap" role="img" aria-label={`${MARKET_LABELS[market]}: ${reading.word}`}>
                          <i className="l" style={{ width: `${value !== undefined && value < 0 ? gapBar(value, max) * 100 : 0}%` }} />
                          <i className="r" style={{ width: `${value !== undefined && value > 0 ? gapBar(value, max) * 100 : 0}%` }} />
                        </span>
                      </td>
                    </tr>
                  );
                })
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="sc-note">Bets: {betsLine(event)}.</p>
    </div>
  );
}

/** The model scorecard inside the Scorecard and P&L view: forward record first, then the latest settled week. */
export function GolfpriceScorecardSection() {
  const { data, loading: fetching } = useDashboardData<unknown>(SCORECARD_KEY);
  const [now] = useState(() => Date.now());
  // Never sit on "Loading" forever: after 10 seconds the panel says the scorecard is not available instead.
  const [gaveUp, setGaveUp] = useState(false);
  useEffect(() => {
    const timer = window.setTimeout(() => setGaveUp(true), 10_000);
    return () => window.clearTimeout(timer);
  }, []);
  const loading = fetching && !gaveUp;
  const card = parseScorecard(data);
  const index = useDashboardData<{ events?: EventLike[] }>("golfprice/index.json");
  const events = card ? latestWeekEvents(card) : [];
  const lag = card ? scorecardLag(card.events, index.data?.events ?? null, now) : null;
  const paused = forwardClockPaused(card);
  const interval = card?.forward.by_arm.challenger?.logloss_vs_close_mean;
  const reading = marketReading(interval?.mean);
  const live = card?.forward.bets.live;
  const paper = card?.forward.bets.shadow;

  return (
    <Section id="scorecard" title="Model scorecard" purpose="How the current model did against the market's closing prices.">
      {loading ? (
        <div className="sc-loading" aria-busy="true"><p className="sc-note">Loading the weekly scorecard...</p></div>
      ) : !card ? (
        <EmptyState title={fetching ? "The scorecard is not available right now" : "No scorecard yet"} detail={fetching ? "It did not load. Reload in a minute; it updates after each Monday settle." : "It appears after the first Monday settle grades the previous week."} nextCheck="Next check: the Monday settle" />
      ) : (
        <div>
          {lag?.state === "behind" && <p className="sc-note" role="status" data-testid="scorecard-behind"><Chip tone="warning">Behind</Chip> {lag.message}</p>}
          {paused ? (
            <p className="sc-note" role="status" data-testid="forward-clock-paused">{forwardRestartNote(card.forward?.clock_start_event_uid)}</p>
          ) : (
            <>
              <p className="sc-headline">{forwardHeadline(card.forward.events_counted, card.forward.futility_check_at, interval?.mean, interval?.lo, interval?.hi)}</p>
              <div className="kpi-grid">
                <Stat label="Events graded" value={String(card.forward.events_counted)} unit={`of ${card.forward.futility_check_at}`} note={`First checkpoint; the full test is ${card.forward.horizon_events} events`} hint="PGA events graded since the forward test started. At the checkpoint we decide whether the model is worth continuing with." />
                <Stat label="Gap to market close" value={signedFmt(interval?.mean, 4)} unit="log-loss" tone={reading.tone === "" ? "neutral" : reading.tone} note={`${reading.word}${interval?.lo === null || interval?.lo === undefined ? "; too few events for a 95% range" : `; 95% range ${signedFmt(interval.lo, 4)} to ${signedFmt(interval.hi, 4)}`}. Negative is better.`} hint="How far the model's finish-position probabilities were from what happened, compared with the market's closing prices. Negative means closer than the market." />
                {live && live.n > 0 && <Stat label="Live bets" value={signedFmt(live.pnl_units ?? 0, 2)} unit="units" tone={(live.pnl_units ?? 0) > 0 ? "positive" : (live.pnl_units ?? 0) < 0 ? "negative" : "neutral"} note={`${live.n} placed${paper?.n ? `; paper bets ${signedFmt(paper.pnl_units ?? 0, 2)} units (tracked, not placed)` : ""}`} hint="Profit or loss in betting units on bets actually placed. Paper bets are tracked but not placed." />}
              </div>
            </>
          )}
          {events.length === 0 ? <EmptyState title="No graded event yet" detail="The latest week appears here once a settle has run." nextCheck="Next check: the Monday settle" /> : events.map((event) => <EventTable key={event.event_uid} event={event} />)}
          <p className="sc-note">{lag?.newestSettled ? `Newest graded event starts ${plainDate(lag.newestSettled)}. ` : ""}Updated {etTime(card.generated_at)}. Only PGA Tour events after the forward test starts count toward the forward record.</p>
        </div>
      )}
    </Section>
  );
}
