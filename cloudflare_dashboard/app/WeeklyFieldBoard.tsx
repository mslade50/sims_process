"use client";

import { useMemo, useState } from "react";
import { Search } from "lucide-react";
import { Chip, GapRail, SegmentedControl, Stat } from "./ui";
import { MARKETS, MARKET_LABEL, type ExplainDoc, type Market } from "./explain-rules";
import { boardRows, biggestGaps, railRange, searchBoard, sortBoard, type BoardRow, type BoardSort } from "./player-field-rules";
import { profileHref, signedZ } from "./player-profile-rules";
import "./player-field-board.css";

const STEP = 25;
const SORTS: Array<{ value: BoardSort; label: string }> = [
  { value: "model", label: "Our chance" }, { value: "gap", label: "Biggest gap" }, { value: "book", label: "Book" }, { value: "skill", label: "Skill" }, { value: "name", label: "A to Z" },
];
const MARKET_OPTIONS = MARKETS.map((m) => ({ value: m, label: MARKET_LABEL[m] }));
const pc = (v: number | null, d = 1) => (v === null ? "n/a" : `${v.toFixed(d)}%`);
const sg = (v: number | null) => (v === null ? "n/a" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(1)}`);

/** Whole-field comparison: one compact row per golfer (skill, our chance vs the book on a zero-centred rail, one-line driver); tap a row for the evidence. */
export function WeeklyFieldBoard({ doc, playerId }: { doc: ExplainDoc; playerId?: string | null }) {
  const [market, setMarket] = useState<Market>("win");
  const [sort, setSort] = useState<BoardSort>("model");
  const [query, setQuery] = useState("");
  const first = typeof window !== "undefined" && window.matchMedia("(max-width: 820px)").matches ? 10 : STEP;
  const [shown, setShown] = useState(first);
  const [open, setOpen] = useState<number | null>(null);
  const all = useMemo(() => boardRows(doc, market).filter((r) => !playerId || String(r.id) === playerId), [doc, market, playerId]);
  const range = useMemo(() => railRange(all), [all]);
  const ranges = useMemo(() => Object.fromEntries(MARKETS.map((m) => [m, railRange(boardRows(doc, m))])) as Record<Market, number>, [doc]);
  const gaps = useMemo(() => biggestGaps(all), [all]);
  const found = useMemo(() => sortBoard(searchBoard(all, query), sort), [all, query, sort]);
  const rows = found.slice(0, shown);
  const priced = all.filter((r) => r.book !== null).length;
  const fav = [...all].sort((a, b) => (b.model ?? -1) - (a.model ?? -1))[0];
  const top = gaps.above[0] && gaps.below[0] ? (Math.abs(gaps.above[0].gap as number) >= Math.abs(gaps.below[0].gap as number) ? gaps.above[0] : gaps.below[0]) : gaps.above[0] ?? gaps.below[0];
  const label = MARKET_LABEL[market];
  const pick = (r: BoardRow) => { setQuery(r.name); setOpen(r.id); setShown(first); };
  return (
    <div className="fb">
      <div className="kpi-grid fb-stats">
        <Stat label="Golfers in the field" value={String(all.length)} note={doc.kind === "live" ? `Live, after round ${doc.after_round}` : "Pre-tournament prices"} />
        <Stat label={`Our ${label.toLowerCase()} favourite`} value={fav?.name ?? "n/a"} note={fav ? `${pc(fav.model)} ours${fav.book !== null ? `, book ${pc(fav.book)}` : ""}` : undefined} />
        <Stat label="Largest gap to the book" value={top ? `${sg(top.gap)} pts` : "n/a"} note={top ? `${top.name}: ours ${pc(top.model)} vs book ${pc(top.book)}` : "No book prices for this market"} tone={top ? ((top.gap as number) > 0 ? "positive" : "negative") : "neutral"} />
        <Stat label="With a book price" value={`${priced} of ${all.length}`} note="Gap needs both numbers" />
      </div>

      <div className="fb-controls">
        <div className="fb-control"><span id="fb-market-l" className="fb-label">Market</span><SegmentedControl label="Market" value={market} options={MARKET_OPTIONS} onChange={(v) => { setMarket(v); setShown(first); }} /></div>
        <div className="fb-control"><span className="fb-label">Sort by</span><SegmentedControl label="Sort by" value={sort} options={SORTS} onChange={(v) => { setSort(v); setShown(first); }} /></div>
        <label className="fb-search"><Search size={16} aria-hidden="true" /><input type="search" aria-label="Search golfers in this field" placeholder="Search name or country" value={query} onChange={(e) => { setQuery(e.target.value); setShown(first); }} /></label>
      </div>

      {(gaps.above.length > 0 || gaps.below.length > 0) && !query && (
        <div className="fb-gaps" aria-label={`Largest ${label} gaps`}>
          <GapList title="We are higher than the book" tone="positive" rows={gaps.above} onPick={pick} />
          <GapList title="We are lower than the book" tone="negative" rows={gaps.below} onPick={pick} />
        </div>
      )}

      <div className="fb-list" role="list" aria-label={`${label} chances by golfer`}>
        <div className="fb-head" aria-hidden="true"><span>Golfer</span><span>Skill</span><span>{label}: our chance vs book, gap in points from a shared zero</span><span>Main driver</span></div>
        {rows.map((r) => {
          const isOpen = open === r.id;
          return (
            <div role="listitem" key={r.id} className={`fb-item${isOpen ? " open" : ""}`} id={`fb-${r.id}`}>
              <button type="button" className="fb-row" aria-expanded={isOpen} aria-controls={`fb-ev-${r.id}`} onClick={() => setOpen(isOpen ? null : r.id)}>
                <span className="fb-name"><b>{r.name}</b>{r.country && <small>{r.country}</small>}</span>
                <span className="fb-skill num" title="This week's skill against the field average, strokes per round (positive is stronger)"><small>Skill</small>{signedZ(r.skill, 2)}</span>
                <span className="fb-rail">{r.model !== null && r.book !== null ? <GapRail model={r.model} book={r.book} range={range} /> : <span className="fb-nobook num">Ours {pc(r.model)} · no book price</span>}</span>
                <span className="fb-why">{r.why || "No driver summary for this player."}</span>
                <span className="fb-chev" aria-hidden="true" />
              </button>
              {isOpen && <Evidence r={r} ranges={ranges} />}
            </div>
          );
        })}
        {!found.length && <p className="fb-empty" role="status">No golfer matches “{query}”. Clear the search to see the whole field.</p>}
      </div>
      <div className="fb-foot">
        <span className="num" role="status">Showing {Math.min(shown, found.length)} of {found.length}</span>
        {shown < found.length && <><button type="button" className="fb-btn" onClick={() => setShown(shown + STEP)}>Show {Math.min(STEP, found.length - shown)} more</button><button type="button" className="fb-btn" onClick={() => setShown(found.length)}>Show all</button></>}
        {query && <button type="button" className="fb-btn" onClick={() => setQuery("")}>Clear search</button>}
      </div>
    </div>
  );
}

function GapList({ title, tone, rows, onPick }: { title: string; tone: "positive" | "negative"; rows: BoardRow[]; onPick: (r: BoardRow) => void }) {
  if (!rows.length) return null;
  return (
    <div className="fb-gaplist">
      <h3><Chip tone={tone}>{tone === "positive" ? "▲ Above book" : "▼ Below book"}</Chip> <span>{title}</span></h3>
      <ul>{rows.map((r) => <li key={r.id}><button type="button" onClick={() => onPick(r)}><b>{r.name}</b><span className="num">{pc(r.model)} vs {pc(r.book)}</span><span className={`num fb-pts ${tone === "positive" ? "pos" : "neg"}`}>{sg(r.gap)} pts</span></button></li>)}</ul>
    </div>
  );
}

function Evidence({ r, ranges }: { r: BoardRow; ranges: Record<Market, number> }) {
  const p = r.player;
  return (
    <div className="fb-evidence" id={`fb-ev-${r.id}`}>
      <p className="fb-why-full"><b>Why the model has this price.</b> {r.why || "No driver summary for this player."}</p>
      {p.why_edge && <p className="fb-why-full"><b>Why we differ from the book.</b> {p.why_edge}</p>}
      <div className="fb-markets">
        {MARKETS.map((m) => {
          const mp = p.probs?.[m];
          const model = typeof mp?.model === "number" ? mp.model * 100 : null, book = typeof mp?.market === "number" ? mp.market * 100 : null;
          return <div key={m}><h4>{MARKET_LABEL[m]}</h4>{model !== null && book !== null ? <GapRail model={model} book={book} range={ranges[m]} /> : <span className="fb-nobook num">Ours {pc(model)} · no book price</span>}</div>;
        })}
      </div>
      <div className="fb-evfoot">
        {(p.tags ?? []).slice(0, 4).map((t) => <Chip key={t}>{t.replace(/_/g, " ")}</Chip>)}
        <a className="fb-btn" href={profileHref(r.id)}>Open {r.name}&apos;s profile</a>
      </div>
    </div>
  );
}
