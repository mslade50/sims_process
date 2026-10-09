"use client";

import { useState } from "react";
import { EmptyState, Panel } from "./components";
import { useDashboardData } from "./data";
import { etTime } from "./lib";
import { ODDS_SIGNALS_KEY, alertText, bookName, firstLast, marketName, oddsStale, parseOddsSignals, pct, statusName } from "./odds-signals-rules";

/** Latest odds-move signals (golfprice odds_reprice moment): EV against the fairs of the last full run, not re-simulated. */
export function OddsSignalsPanel() {
  const { data, loading } = useDashboardData<unknown>(ODDS_SIGNALS_KEY);
  const [now] = useState(() => Date.now());
  const doc = parseOddsSignals(data);

  return (
    <Panel eyebrow="Odds moves" title="Latest betting edges (not re-simulated)">
      {loading ? (
        <p className="inputs-muted">Loading the latest odds check…</p>
      ) : !doc ? (
        <EmptyState title="No odds check published yet" detail="It appears after the first odds check, which runs by itself every 30 minutes from Monday afternoon to the Thursday tee." />
      ) : (
        <div className="scorecard-body">
          {oddsStale(doc.generated_at, now) && <p className="inputs-note accent">The last odds check that found new prices was {etTime(doc.generated_at)}. Checks that find nothing new do not publish.</p>}
          <p className="inputs-muted">Provisional: sportsbook prices compared with the model&apos;s numbers from the last full run, not re-simulated. The next full run decides. Checked {etTime(doc.generated_at)}.</p>
          {doc.events.length === 0 && <EmptyState title="No event" detail="The check found no event before its first tee." />}
          {doc.events.map((event) => (
            <div key={event.event_uid}>
              <h3 className="inputs-h3">
                {event.name} <span className="inputs-muted">{event.n_signals} edges ({event.n_live} live, {event.n_signals - event.n_live} paper only); {event.new.length} new, {event.moved.length} moved, {event.dropped.length} gone since the last check</span>
              </h3>
              <p className="inputs-muted">Compared with the model&apos;s prices from the run at {etTime(event.base_as_of)}. {alertText(event.alert)}</p>
              {event.signals.length === 0 ? (
                <p className="inputs-muted">No sportsbook price is far enough from the model right now.</p>
              ) : (
                <div className="table-scroll">
                  <table aria-label={`Odds signals for ${event.name}`}>
                    <thead>
                      <tr>
                        <th title="A live bet is one we would really place. Paper only means it is tracked but not placed."><span className="th-text">Type</span></th>
                        <th><span className="th-text">Market</span></th>
                        <th><span className="th-text">Player</span></th>
                        <th><span className="th-text">Sportsbook</span></th>
                        <th title="The sportsbook's payout for a 1 unit stake, including the stake."><span className="th-text">Odds (decimal)</span></th>
                        <th title="The model's own chance of this happening."><span className="th-text">Model chance</span></th>
                        <th title="Expected profit per unit staked if the model is right."><span className="th-text">Expected profit</span></th>
                      </tr>
                    </thead>
                    <tbody>
                      {event.signals.slice(0, 40).map((signal) => (
                        <tr key={`${signal.family}-${signal.dg_id}-${signal.book}`} className={signal.status === "live" ? "active-row" : undefined}>
                          <td>{statusName(signal.status)}</td>
                          <td>{marketName(signal.family)}</td>
                          <td>{signal.player ? firstLast(signal.player) : "Unnamed player"}</td>
                          <td>{bookName(signal.book)}</td>
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
