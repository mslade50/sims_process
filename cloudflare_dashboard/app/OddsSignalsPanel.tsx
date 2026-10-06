"use client";

import { useState } from "react";
import { EmptyState, Panel } from "./components";
import { useDashboardData } from "./data";
import { ODDS_SIGNALS_KEY, oddsStale, parseOddsSignals, pct } from "./odds-signals-rules";

/** Latest odds-move signals (golfprice odds_reprice moment): EV against the fairs of the last full run, not re-simulated. */
export function OddsSignalsPanel() {
  const { data, loading } = useDashboardData<unknown>(ODDS_SIGNALS_KEY);
  const [now] = useState(() => Date.now());
  const doc = parseOddsSignals(data);

  return (
    <Panel eyebrow="Odds moves" title="Latest signals (not re-simulated)">
      {loading ? (
        <p className="inputs-muted">Loading the latest odds check…</p>
      ) : !doc ? (
        <EmptyState title="No odds check published yet" detail="It appears after the first odds check publishes golfprice/odds_signals/latest.json (runs by itself every 30 minutes from Monday afternoon to the Thursday tee)." />
      ) : (
        <div className="scorecard-body">
          {oddsStale(doc.generated_at, now) && <p className="inputs-note accent">The last odds check that found new quotes was {doc.generated_at}. Checks that find nothing new do not publish.</p>}
          <p className="inputs-muted">{doc.note} Generated {doc.generated_at} ({doc.moment}).</p>
          {doc.events.length === 0 && <EmptyState title="No event" detail="The check found no event before its first tee." />}
          {doc.events.map((event) => (
            <div key={event.event_uid}>
              <h3 className="inputs-h3">
                {event.name} <span className="inputs-muted">{event.n_signals} signals, {event.n_live} live; {event.new.length} new, {event.moved.length} moved, {event.dropped.length} dropped</span>
              </h3>
              <p className="inputs-muted">Fairs: {event.fair_source} arm of the run {event.base_run}. Alert: {event.alert.sent ? "sent" : event.alert.note}.</p>
              {event.signals.length === 0 ? (
                <p className="inputs-muted">No quote clears its threshold right now.</p>
              ) : (
                <div className="table-scroll">
                  <table aria-label={`Odds signals for ${event.name}`}>
                    <thead>
                      <tr>
                        <th><span className="th-text">Status</span></th>
                        <th><span className="th-text">Market</span></th>
                        <th><span className="th-text">Player</span></th>
                        <th><span className="th-text">Book</span></th>
                        <th><span className="th-text">Odds</span></th>
                        <th><span className="th-text">Fair</span></th>
                        <th><span className="th-text">EV</span></th>
                      </tr>
                    </thead>
                    <tbody>
                      {event.signals.slice(0, 40).map((signal) => (
                        <tr key={`${signal.family}-${signal.dg_id}-${signal.book}`} className={signal.status === "live" ? "active-row" : undefined}>
                          <td>{signal.status}</td>
                          <td>{signal.family}</td>
                          <td>{signal.player ?? signal.dg_id}</td>
                          <td>{signal.book}</td>
                          <td>{signal.dec.toFixed(2)}</td>
                          <td>{(signal.p_fair * 100).toFixed(1)}%</td>
                          <td>{pct(signal.ev)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}
