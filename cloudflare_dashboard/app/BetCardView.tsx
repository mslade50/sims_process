"use client";

import { useEffect, useMemo, useState } from "react";
import { EmptyState, LoadingState, PageIntro, Panel, SegmentedControl } from "./components";
import {
  BET_CARD_KEY, DEFAULT_FILTERS, GROUP_OPTIONS, MARKET_LABEL, OWNER_DATA_PREFIX, americanOdds, minPriceText, betText, bookLabel, booksIn, decodeCard, filterBets, formatET, historyText, pctText, timeAgo, totals,
  type Bet, type Card, type CardEvent, type Filters, type Group, type TierFilter,
} from "./bet-card-rules";
import "./bet-card.css";

type Load = { state: "loading" } | { state: "error"; message: string } | { state: "ready"; card: Card };

async function loadCard(): Promise<Card> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 30_000);
  try {
    const response = await fetch(`${OWNER_DATA_PREFIX}${BET_CARD_KEY}`, { signal: controller.signal, headers: { accept: "application/json" }, cache: "no-store" });
    if (response.status === 403) throw new Error("This page is for the owner only.");
    if (response.status === 404) throw new Error("The bet card has not been published yet.");
    if (!response.ok) throw new Error(`The bet card could not be loaded (HTTP ${response.status}).`);
    return decodeCard(await response.json());
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw new Error("Loading the bet card timed out. Try again.");
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

function BetRow({ bet }: { bet: Bet }) {
  const moved = bet.firstDec !== null && Math.abs(bet.firstDec - bet.dec) > 0.004;
  return (
    <li className={`bc-bet ${bet.tier === "LEAN IN" ? "lean" : "small"}`}>
      <div className="bc-head">
        <strong>{betText(bet)}</strong>
        {bet.isNew && <span className="bc-badge">NEW</span>}
      </div>
      <div className="bc-price">
        <span>{bookLabel(bet.book)} <b>{bet.dec.toFixed(2)}</b> ({americanOdds(bet.dec)})</span>
        {bet.minDec !== null && <span>take at or above {minPriceText(bet.minDec).dec} ({minPriceText(bet.minDec).american})</span>}
        <span><b>{bet.stakeU.toFixed(2)}u</b></span>
      </div>
      <div className="bc-facts">
        <span>our chance {pctText(bet.p)}</span>
        <span>edge {pctText(bet.edge)}</span>
        <span>{MARKET_LABEL[bet.market] ?? bet.market}</span>
        {bet.firstSeen && <span>first seen {formatET(bet.firstSeen)}{moved && bet.firstDec !== null ? ` at ${bet.firstDec.toFixed(2)}` : ""}</span>}
        {bet.alsoAt && <span>also at {bet.alsoAt}</span>}
        <span>{historyText(bet)}</span>
      </div>
      {bet.why && <details><summary>Why</summary>{bet.why}</details>}
    </li>
  );
}

function EventSection({ event, filters }: { event: CardEvent; filters: Filters }) {
  const shown = useMemo(() => filterBets(event.bets, filters), [event, filters]);
  const t = totals(shown);
  const tiers: Array<{ label: string; bets: Bet[] }> = [
    { label: "Lean in", bets: shown.filter((b) => b.tier === "LEAN IN") },
    { label: "Small stake", bets: shown.filter((b) => b.tier === "SMALL STAKE") },
  ];
  return (
    <Panel eyebrow={`${event.tour.toUpperCase()} · ${event.dateStart}`} title={event.name}>
      <p className="bc-moment">{event.moment}{event.pricedRound ? ` · pricing round ${event.pricedRound}` : ""} · {event.status}{event.note ? ` · ${event.note}` : ""}</p>
      {event.bets.length === 0 ? (
        <EmptyState title={event.status === "priced" ? "No bets right now" : event.status} detail={event.note ?? "No suggested bets for this event at the moment."} />
      ) : shown.length === 0 ? (
        <EmptyState title="No bets match these filters" detail="Change the tier, market or book filter." />
      ) : (
        <>
          {tiers.filter((g) => g.bets.length > 0).map((g) => (
            <div key={g.label}>
              <h3 className="bc-tier">{g.label} ({g.bets.length})</h3>
              <ul className="bc-list">{g.bets.map((b) => <BetRow key={b.key || `${b.market}${b.player}${b.book}`} bet={b} />)}</ul>
            </div>
          ))}
          <p className="bc-totals">{t.bets} {t.bets === 1 ? "bet" : "bets"}, {t.units.toFixed(2)} units</p>
        </>
      )}
      {event.coverage.length > 0 && (
        <details className="bc-cov">
          <summary>Coverage: fresh quotes from your books</summary>
          <table><thead><tr><th>Market</th><th>Book</th><th>Quotes</th></tr></thead>
            <tbody>{event.coverage.map((c) => <tr key={`${c.market}|${c.book}`}><td>{MARKET_LABEL[c.market] ?? c.market}</td><td>{bookLabel(c.book)}</td><td>{c.n}</td></tr>)}</tbody></table>
          <p className="bc-notes">Missing exchanges and BetCris or Pinnacle make-cut lines mean no data, not no edge.</p>
        </details>
      )}
    </Panel>
  );
}

function CardBody({ card }: { card: Card }) {
  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS);
  const books = useMemo(() => booksIn(card), [card]);
  return (
    <>
      <div className="bc-banner">
        <strong>Suggestions only. Nothing is placed automatically.</strong>
        <small>Generated {formatET(card.generatedAt)} ({timeAgo(card.generatedAt)})</small>
      </div>
      <Panel eyebrow="Filters" title="What to show">
        <div className="bc-filters">
          <SegmentedControl<TierFilter> label="Tier" value={filters.tier} onChange={(tier) => setFilters((f) => ({ ...f, tier }))} options={[{ value: "all", label: "All" }, { value: "lean", label: "Lean in" }, { value: "small", label: "Small" }]} />
          <SegmentedControl<Group> label="Market" value={filters.group} onChange={(group) => setFilters((f) => ({ ...f, group }))} options={GROUP_OPTIONS} />
          <label><span>Book</span>
            <select value={filters.book} onChange={(e) => setFilters((f) => ({ ...f, book: e.target.value }))}>
              <option value="all">All books</option>
              {books.map((b) => <option key={b} value={b}>{bookLabel(b)}</option>)}
            </select>
          </label>
        </div>
      </Panel>
      {card.events.length === 0 && <EmptyState title="No events" detail="The card lists no events right now." />}
      {card.events.map((e) => <EventSection key={e.eventUid} event={e} filters={filters} />)}
      {card.notes.length > 0 && <Panel eyebrow="Notes" title="How to read this"><ul className="bc-notes">{card.notes.map((n, i) => <li key={i}>{n}</li>)}</ul></Panel>}
    </>
  );
}

export function BetCardView() {
  const [load, setLoad] = useState<Load>({ state: "loading" });
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let live = true;
    loadCard()
      .then((card) => { if (live) setLoad({ state: "ready", card }); })
      .catch((error: unknown) => { if (live) setLoad({ state: "error", message: error instanceof Error ? error.message : "The bet card could not be loaded." }); });
    return () => { live = false; };
  }, [attempt]);

  const intro = <PageIntro eyebrow="Owner only" title="Bet card" description="Current suggested bets from the model, with the lowest price worth taking. Refreshed every 30 minutes before an event and after each round." />;
  if (load.state === "loading") return <div>{intro}<LoadingState label="Loading the bet card" /></div>;
  if (load.state === "error") return <div>{intro}<EmptyState title="The bet card is not available" detail={load.message} onRetry={() => { setLoad({ state: "loading" }); setAttempt((a) => a + 1); }} /></div>;
  return <div>{intro}<CardBody card={load.card} /></div>;
}
