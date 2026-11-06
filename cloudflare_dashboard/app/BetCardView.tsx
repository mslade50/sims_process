"use client";

import { useEffect, useMemo, useState } from "react";
import { Chip, EmptyState, InfoTip, LoadingState, PageHeader, Section, SegmentedControl } from "./components";
import {
  BET_CARD_KEY, DEFAULT_FILTERS, GROUP_OPTIONS, MARKET_LABEL, OWNER_DATA_PREFIX, activeFilterCount, americanOdds, betText, bookLabel, booksIn, decodeCard, filterBets, formatET, historyText, minPriceText, nextCheckForStage, quietReason, pctText, stageText, timeAgo, totals,
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

/** One bet: tap the row to open the evidence. Edge is the dominant number; the price to take and the lowest price worth taking sit under the name. */
function BetRow({ bet }: { bet: Bet }) {
  const moved = bet.firstDec !== null && Math.abs(bet.firstDec - bet.dec) > 0.004;
  const min = bet.minDec !== null ? minPriceText(bet.minDec) : null;
  return (
    <li className={`bc-bet ${bet.tier === "LEAN IN" ? "lean" : "small"}`}>
      <details>
        <summary aria-label={`${betText(bet)}, edge ${pctText(bet.edge)}, stake ${bet.stakeU.toFixed(2)} units. Show details`}>
          <div className="bc-main">
            <strong className="bc-name">{betText(bet)}</strong>
            <span className="bc-sub">
              <span>{MARKET_LABEL[bet.market] ?? bet.market}</span>
              <span className={`bc-tag ${bet.tier === "LEAN IN" ? "lean" : "small"}`}>{bet.tier === "LEAN IN" ? "Lean in" : "Small stake"}</span>
              {bet.isNew && <Chip tone="accent">New</Chip>}
            </span>
          </div>
          <div className="bc-edge" title="Our chance minus the break-even chance at this price">
            <b className="num">+{(bet.edge * 100).toFixed(1)}%</b>
            <small>edge</small>
          </div>
          <dl className="bc-price">
            <div><dt>Price now</dt><dd className="num"><b>{bet.dec.toFixed(2)}</b> <span>({americanOdds(bet.dec)}) {bookLabel(bet.book)}</span></dd></div>
            <div><dt>Take at or above</dt><dd className="num">{min ? <><b>{min.dec}</b> <span>({min.american})</span></> : "n/a"}</dd></div>
            <div className="bc-stake"><dt>Stake</dt><dd className="num"><b>{bet.stakeU.toFixed(2)}u</b></dd></div>
          </dl>
        </summary>
        <div className="bc-detail">
          <dl className="bc-evidence">
            <div><dt>Our chance</dt><dd className="num">{pctText(bet.p)}</dd></div>
            <div><dt>History</dt><dd>{historyText(bet)}</dd></div>
            {bet.firstSeen && <div><dt>First seen</dt><dd className="num">{formatET(bet.firstSeen)}{moved && bet.firstDec !== null ? ` at ${bet.firstDec.toFixed(2)}` : ""}</dd></div>}
            {bet.alsoAt && <div><dt>Also at</dt><dd className="num">{bet.alsoAt}</dd></div>}
          </dl>
          {bet.why && <p className="bc-why">{bet.why}</p>}
        </div>
      </details>
    </li>
  );
}

function BetList({ bets, label, hint }: { bets: Bet[]; label: string; hint: string }) {
  if (bets.length === 0) return null;
  return (
    <div className="bc-group">
      <h3 className="bc-tier">{label} <span className="num">{bets.length}</span><small>{hint}</small></h3>
      <ul className="bc-list">{bets.map((b) => <BetRow key={b.key || `${b.market}${b.player}${b.book}`} bet={b} />)}</ul>
    </div>
  );
}

function Coverage({ event }: { event: CardEvent }) {
  if (event.coverage.length === 0) return null;
  return (
    <details className="bc-cov">
      <summary>Which books were checked</summary>
      <table>
        <thead><tr><th>Market</th><th>Book</th><th className="num">Fresh quotes</th></tr></thead>
        <tbody>{event.coverage.map((c) => <tr key={`${c.market}|${c.book}`}><td>{MARKET_LABEL[c.market] ?? c.market}</td><td>{bookLabel(c.book)}</td><td className="num">{c.n}</td></tr>)}</tbody>
      </table>
      <p className="bc-note">Missing exchanges and BetCris or Pinnacle make-cut lines mean no data, not no edge.</p>
    </details>
  );
}

function EventSection({ event, filters }: { event: CardEvent; filters: Filters }) {
  const shown = useMemo(() => filterBets(event.bets, filters), [event, filters]);
  const t = totals(shown);
  const lean = shown.filter((b) => b.tier === "LEAN IN");
  const small = shown.filter((b) => b.tier === "SMALL STAKE");
  return (
    <Section
      id={`event-${event.eventUid.replace(/[^a-z0-9]+/gi, "-")}`}
      title={event.name}
      purpose={`${event.tour.toUpperCase()} · ${stageText(event)}`}
      actions={<span className="bc-total num">{t.bets} {t.bets === 1 ? "bet" : "bets"} · {t.units.toFixed(2)}u</span>}
    >
      {shown.length === 0 ? (
        <EmptyState title="No bets match these filters" detail={`${event.bets.length} suggested in total. Clear the filters to see them.`} />
      ) : (
        <>
          <BetList bets={lean} label="Lean in" hint="strongest rule bands, full stake" />
          <BetList bets={small} label="Small stake" hint="thinner edge or less history, smaller stake" />
        </>
      )}
      <Coverage event={event} />
    </Section>
  );
}

/** One compact line per event that has nothing to suggest: what state it is in, why, and when it is checked next. */
function QuietEvents({ events }: { events: CardEvent[] }) {
  if (events.length === 0) return null;
  return (
    <Section id="quiet" title="Nothing to suggest" purpose="Events with no bet right now, where each stands and when it is checked again.">
      <ul className="bc-status">
        {events.map((e) => {
          const stage = stageText(e);
          return (
          <li key={e.eventUid}>
            <div className="bc-status-head">
              <Chip>{e.tour.toUpperCase()}</Chip>
              <strong>{e.name}</strong>
              <Chip tone="warning">{stage}</Chip>
            </div>
            <p>{quietReason(e, stage)}</p>
            <p className="bc-next">{nextCheckForStage(e, stage)}</p>
            <Coverage event={e} />
          </li>
          );
        })}
      </ul>
    </Section>
  );
}

function FilterBar({ filters, setFilters, books, count }: { filters: Filters; setFilters: (f: Filters) => void; books: string[]; count: number }) {
  const active = activeFilterCount(filters);
  return (
    <div className="bc-bar" role="group" aria-label="Filter the bet card">
      <SegmentedControl<TierFilter> label="Tier" value={filters.tier} onChange={(tier) => setFilters({ ...filters, tier })} options={[{ value: "all", label: "All" }, { value: "lean", label: "Lean in" }, { value: "small", label: "Small" }]} />
      <label className="bc-sel"><span>Market</span>
        <select value={filters.group} onChange={(e) => setFilters({ ...filters, group: e.target.value as Group })}>
          {GROUP_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>
      </label>
      <label className="bc-sel"><span>Book</span>
        <select value={filters.book} onChange={(e) => setFilters({ ...filters, book: e.target.value })}>
          <option value="all">All books</option>
          {books.map((b) => <option key={b} value={b}>{bookLabel(b)}</option>)}
        </select>
      </label>
      <span className="bc-bar-end">
        <span className="num" role="status" aria-live="polite" aria-atomic="true">{count} {count === 1 ? "bet" : "bets"} shown</span>
        {active > 0 && <button type="button" className="bc-reset" onClick={() => setFilters(DEFAULT_FILTERS)}>Clear {active} {active === 1 ? "filter" : "filters"}</button>}
      </span>
    </div>
  );
}

function CardBody({ card }: { card: Card }) {
  const [filters, setFilters] = useState<Filters>(DEFAULT_FILTERS);
  const books = useMemo(() => booksIn(card), [card]);
  const withBets = card.events.filter((e) => e.bets.length > 0);
  const quiet = card.events.filter((e) => e.bets.length === 0);
  const all = totals(card.events.flatMap((e) => e.bets));
  const shownCount = withBets.reduce((s, e) => s + filterBets(e.bets, filters).length, 0);
  const top = Math.max(0, ...card.events.flatMap((e) => e.bets.map((b) => b.edge)));
  return (
    <>
      <div className="bc-answer" role="status">
        <div>
          <strong className="bc-headline">{all.bets === 0 ? "No bets to suggest right now" : `${all.bets} ${all.bets === 1 ? "bet" : "bets"} worth considering`}</strong>
          <span className="bc-meta num">{all.bets > 0 ? `${all.units.toFixed(2)} units in total, best edge +${(top * 100).toFixed(1)}%. ` : ""}Card built {formatET(card.generatedAt)} ({timeAgo(card.generatedAt)}).</span>
        </div>
        <span className="bc-quiet">Suggestions only. Nothing is placed automatically.</span>
      </div>
      {all.bets > 0 && <FilterBar filters={filters} setFilters={setFilters} books={books} count={shownCount} />}
      {card.events.length === 0 && <EmptyState title="No events" detail="The card lists no events right now." nextCheck="Next check: within 30 minutes" />}
      {withBets.map((e) => <EventSection key={e.eventUid} event={e} filters={filters} />)}
      <QuietEvents events={quiet} />
      {card.notes.length > 0 && (
        <details className="bc-howto">
          <summary>How to read this card</summary>
          <p className="bc-note"><InfoTip definition="Our modelled chance of winning the bet minus the chance the price implies. Positive means the price pays more than the bet is worth.">Edge</InfoTip> is our chance minus the price&apos;s break-even chance. The take-at price is the lowest price where the bet is still worth it. Rules:</p>
          <ul className="bc-notes">{card.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
        </details>
      )}
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

  const intro = <PageHeader title="Bet card" purpose="Suggested bets across every event being priced, with the lowest price worth taking." />;
  if (load.state === "loading") return <div>{intro}<LoadingState label="Loading the bet card" /></div>;
  if (load.state === "error") return <div>{intro}<EmptyState title="The bet card is not available" detail={load.message} onRetry={() => { setLoad({ state: "loading" }); setAttempt((a) => a + 1); }} /></div>;
  return <div>{intro}<CardBody card={load.card} /></div>;
}
