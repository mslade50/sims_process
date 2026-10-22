"use client";

import { useEffect, useMemo, useState } from "react";
import { EmptyState, LoadingState, PageIntro, Panel, SegmentedControl } from "./components";
import {
  BET_BREAKDOWNS_KEY, EDGE_BUCKETS, MARKET_GROUPS, OWNER_BOOKS, OWNER_DATA_PREFIX, bookLabel, breakdown, decodeDataset, defaultFilters, examples, pct, verdict,
  type Basis, type Cell, type Dataset, type Filters, type Move, type RowDim, type Sample,
} from "./bet-breakdowns-rules";
import "./bet-breakdowns.css";

type Load = { state: "loading" } | { state: "error"; message: string } | { state: "ready"; data: Dataset };

async function loadDataset(): Promise<Dataset> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 90_000);
  try {
    const response = await fetch(`${OWNER_DATA_PREFIX}${BET_BREAKDOWNS_KEY}`, { signal: controller.signal, headers: { accept: "application/json" } });
    if (response.status === 403) throw new Error("This page is for the owner only.");
    if (response.status === 404) throw new Error("The bet breakdown data has not been published yet.");
    if (!response.ok) throw new Error(`The data could not be loaded (HTTP ${response.status}).`);
    return decodeDataset(await response.json());
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw new Error("Loading the data timed out. Try again.");
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

function toggle(list: number[], value: number): number[] {
  return list.includes(value) ? list.filter((v) => v !== value) : [...list, value].sort((a, b) => a - b);
}

function Chips({ label, options, selected, onChange, allLabel }: { label: string; options: Array<{ value: number; label: string }>; selected: number[]; onChange: (next: number[]) => void; allLabel?: string }) {
  return (
    <div className="bb-chips" role="group" aria-label={label}>
      <span className="bb-chips-label">{label}</span>
      {allLabel && (
        <button type="button" className={`bb-chip ${selected.length === 0 ? "on" : ""}`} aria-pressed={selected.length === 0} onClick={() => onChange([])}>{allLabel}</button>
      )}
      {options.map((o) => (
        <button key={o.value} type="button" className={`bb-chip ${selected.includes(o.value) ? "on" : ""}`} aria-pressed={selected.includes(o.value)} onClick={() => onChange(toggle(selected, o.value))}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

function CellView({ cell, onOpen, active }: { cell: Cell; onOpen: () => void; active: boolean }) {
  if (!cell.n) return <td className="bb-cell empty">–</td>;
  const v = verdict(cell);
  return (
    <td className={`bb-cell ${v ?? "thin"} ${active ? "active" : ""}`}>
      <button type="button" onClick={onOpen} title="Show seasons and example bets">
        <strong>{pct(cell.roi)}</strong>
        <small>{cell.n.toLocaleString()} bets</small>
        {cell.lo !== null && cell.hi !== null && <small className="bb-range">{pct(cell.lo, 0)} to {pct(cell.hi, 0)}</small>}
      </button>
    </td>
  );
}

const MOVE_OPTIONS: Array<{ value: Move; label: string }> = [
  { value: "all", label: "Any" },
  { value: "toward", label: "Toward us (+CLV)" },
  { value: "flat_or_against", label: "Flat or against" },
  { value: "flat", label: "Flat" },
  { value: "against", label: "Against us" },
];
const SAMPLE_OPTIONS: Array<{ value: Sample; label: string }> = [
  { value: "all", label: "All bets" }, { value: "oos", label: "Out of sample only" }, { value: "ins", label: "In-sample only" },
];
const ROW_OPTIONS: Array<{ value: RowDim; label: string }> = [
  { value: "book", label: "Book" }, { value: "market", label: "Market" }, { value: "season", label: "Season" },
  { value: "hist", label: "History" }, { value: "skill", label: "Skill" }, { value: "tour", label: "Tour" }, { value: "opp_hist", label: "Opponent history" },
];

function Detail({ data, filters, rowKey, rowLabel, bucket, cell, onClose }: { data: Dataset; filters: Filters; rowKey: number | null; rowLabel: string; bucket: number; cell: Cell; onClose: () => void }) {
  const ex = useMemo(() => examples(data, filters, rowKey, bucket), [data, filters, rowKey, bucket]);
  return (
    <Panel eyebrow={`${rowLabel} · edge ${EDGE_BUCKETS[bucket].label} · at the ${filters.basis}`} title="Inside this cell" actions={<button type="button" className="bb-close" onClick={onClose}>Close</button>}>
      <div className="bb-stats">
        <div><span>ROI</span><strong>{pct(cell.roi)}</strong><small>{cell.lo !== null ? `${pct(cell.lo)} to ${pct(cell.hi)}` : "too few events for a range"}</small></div>
        <div><span>Bets</span><strong>{cell.n.toLocaleString()}</strong><small>{cell.events} events</small></div>
        <div><span>Won</span><strong>{cell.winRate === null ? "–" : `${(cell.winRate * 100).toFixed(0)}%`}</strong><small>of bets returned a profit</small></div>
        <div><span>Claimed edge</span><strong>{pct(cell.meanEdge)}</strong><small>average at this price</small></div>
        <div><span>Line move</span><strong>{pct(cell.clv)}</strong><small>{cell.towardShare === null ? "no close price" : `${(cell.towardShare * 100).toFixed(0)}% moved toward us`}</small></div>
      </div>
      <h3 className="bb-sub">By season</h3>
      <div className="bb-seasons">
        {cell.bySeason.map((s) => (
          <div key={s.season} className={s.roi !== null && s.roi > 0 ? "pos" : "neg"}><span>{s.season}</span><strong>{pct(s.roi)}</strong><small>{s.n} bets</small></div>
        ))}
      </div>
      <h3 className="bb-sub">Biggest edges in this cell {ex.total > ex.list.length ? `(top ${ex.list.length} of ${ex.total.toLocaleString()})` : ""}</h3>
      <div className="bb-examples">
        <table>
          <thead><tr><th>Season</th><th>Event</th><th>Market</th><th>Book</th><th>Bet</th><th>Open</th><th>Close</th><th>Our %</th><th>Edge</th><th>Result</th></tr></thead>
          <tbody>
            {ex.list.map((e, k) => (
              <tr key={k}>
                <td>{e.season}{e.inSample ? <small className="bb-ins" title="In-sample: this bet's season or market was used to fit part of the model"> in-sample</small> : null}</td><td>{e.event}</td><td>{e.market}</td><td>{e.book}</td>
                <td>{e.player}{e.opponent ? <small> vs {e.opponent}</small> : null}</td>
                <td>{e.decOpen.toFixed(2)}</td><td>{e.decClose === null ? "–" : e.decClose.toFixed(2)}</td>
                <td>{(e.p * 100).toFixed(1)}%</td><td>{pct(e.edge)}</td>
                <td className={e.profit > 0 ? "pos" : e.profit < 0 ? "neg" : ""}>{e.profit > 0 ? `Won ${e.profit.toFixed(2)}u` : e.profit < 0 ? `Lost ${(-e.profit).toFixed(2)}u` : "Push"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function Explorer({ data }: { data: Dataset }) {
  const { meta } = data;
  const [filters, setFilters] = useState<Filters>(() => defaultFilters(meta));
  const [open, setOpen] = useState<{ rowKey: number | null; bucket: number } | null>(null);
  const set = (patch: Partial<Filters>) => { setFilters((f) => ({ ...f, ...patch })); setOpen(null); };
  const result = useMemo(() => breakdown(data, filters), [data, filters]);

  const marketIdx = (keys: string[]) => keys.map((k) => meta.markets.indexOf(k)).filter((i) => i >= 0);
  const ownerBooks = OWNER_BOOKS.map((b) => meta.books.indexOf(b)).filter((i) => i >= 0);
  const openRow = open ? (open.rowKey === null ? null : result.rows.find((r) => r.key === open.rowKey) ?? null) : null;
  const openCell = open ? (open.rowKey === null ? result.total[open.bucket] : openRow?.cells[open.bucket]) : null;

  return (
    <>
      <Panel eyebrow="Filters" title="What to look at">
        <div className="bb-filters">
          <div className="bb-line">
            <span className="bb-chips-label">Price</span>
            <SegmentedControl<Basis> label="Price" value={filters.basis} onChange={(basis) => set({ basis })} options={[{ value: "open", label: "At the open" }, { value: "close", label: "At the close" }]} />
            <span className="bb-chips-label">Line move</span>
            <SegmentedControl<Move> label="Line move" value={filters.move} onChange={(move) => set({ move })} options={MOVE_OPTIONS} />
          </div>
          {data.hasFit && (
            <div className="bb-line">
              <span className="bb-chips-label">Sample</span>
              <SegmentedControl<Sample> label="Sample" value={filters.sample} onChange={(sample) => set({ sample })} options={SAMPLE_OPTIONS} />
              <small className="bb-hint">Out of sample = 2025-26, excluding R3/R4 2-balls and 3-balls.</small>
            </div>
          )}
          <div className="bb-market-groups">
            {MARKET_GROUPS.map((g) => {
              const idx = marketIdx(g.keys);
              const all = idx.length > 0 && idx.every((i) => filters.markets.includes(i));
              return (
                <div key={g.label} className="bb-chips" role="group" aria-label={g.label}>
                  <button type="button" className={`bb-chips-label bb-group ${all ? "on" : ""}`} onClick={() => set({ markets: all ? filters.markets.filter((m) => !idx.includes(m)) : [...new Set([...filters.markets, ...idx])].sort((a, b) => a - b) })}>{g.label}</button>
                  {idx.map((i) => (
                    <button key={i} type="button" className={`bb-chip ${filters.markets.includes(i) ? "on" : ""}`} aria-pressed={filters.markets.includes(i)} onClick={() => set({ markets: toggle(filters.markets, i) })}>
                      {meta.market_labels[meta.markets[i]] ?? meta.markets[i]}
                    </button>
                  ))}
                </div>
              );
            })}
          </div>
          <div className="bb-chips" role="group" aria-label="Books">
            <span className="bb-chips-label">Books</span>
            <button type="button" className="bb-chip bb-quick" onClick={() => set({ books: ownerBooks })}>My books</button>
            <button type="button" className="bb-chip bb-quick" onClick={() => set({ books: meta.books.map((_, i) => i) })}>All</button>
            {meta.books.map((b, i) => (
              <button key={b} type="button" className={`bb-chip ${filters.books.includes(i) ? "on" : ""} ${OWNER_BOOKS.includes(b) ? "mine" : ""}`} aria-pressed={filters.books.includes(i)} onClick={() => set({ books: toggle(filters.books, i) })}>{bookLabel(b)}</button>
            ))}
          </div>
          <Chips label="Tour" options={meta.tours.map((t, i) => ({ value: i, label: t === "pga" ? "PGA Tour" : "DP World Tour" }))} selected={filters.tours} onChange={(tours) => set({ tours })} />
          <Chips label="Seasons" options={meta.seasons.map((s, i) => ({ value: i, label: data.fitSeasons.has(i) ? `${s} (fit year)` : String(s) }))} selected={filters.seasons} onChange={(seasons) => set({ seasons })} />
          <Chips label="History" allLabel="Any" options={meta.hist_buckets.map((h, i) => ({ value: i, label: h }))} selected={filters.hist} onChange={(hist) => set({ hist })} />
          <Chips label="Skill" allLabel="Any" options={meta.skill_buckets.map((h, i) => ({ value: i, label: h }))} selected={filters.skill} onChange={(skill) => set({ skill })} />
          {data.hasOppHist && (
            <Chips label="Opponent history (2-balls, 72-hole)" allLabel="Any" options={meta.hist_buckets.map((h, i) => ({ value: i, label: h }))} selected={filters.oppHist} onChange={(oppHist) => set({ oppHist })} />
          )}
          <div className="bb-line">
            <span className="bb-chips-label">Decimal odds</span>
            <label className="bb-num"><span>from</span><input type="number" min="1" step="0.05" inputMode="decimal" value={filters.minDec ?? ""} placeholder="any" onChange={(e) => set({ minDec: e.target.value === "" ? null : Number(e.target.value) })} /></label>
            <label className="bb-num"><span>to</span><input type="number" min="1" step="0.05" inputMode="decimal" value={filters.maxDec ?? ""} placeholder="any" onChange={(e) => set({ maxDec: e.target.value === "" ? null : Number(e.target.value) })} /></label>
            <button type="button" className="bb-chip bb-quick" onClick={() => { setFilters(defaultFilters(meta)); setOpen(null); }}>Reset filters</button>
          </div>
        </div>
      </Panel>

      <Panel
        eyebrow={`1-unit flat bets at the ${filters.basis} · range = 95%, events counted as clusters`}
        title="Return by edge bucket"
        actions={<label className="select-control bb-rows"><span>Rows</span><select value={filters.rowDim} onChange={(e) => set({ rowDim: e.target.value as RowDim })}>{ROW_OPTIONS.filter((o) => o.value !== "opp_hist" || data.hasOppHist).map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}</select></label>}
      >
        {result.considered === 0 ? (
          <EmptyState title="No bets match these filters" detail="Widen the filters: add a market, a book or a season." />
        ) : (
          <div className="bb-table">
            <table>
              <thead>
                <tr><th>{ROW_OPTIONS.find((o) => o.value === filters.rowDim)?.label}</th>{EDGE_BUCKETS.map((b) => <th key={b.key}>Edge {b.label}</th>)}<th>All edges ≥ 0</th></tr>
              </thead>
              <tbody>
                {result.rows.map((r) => (
                  <tr key={r.key}>
                    <th scope="row">{r.label}</th>
                    {r.cells.map((c, b) => <CellView key={b} cell={c} active={open?.rowKey === r.key && open?.bucket === b} onOpen={() => setOpen({ rowKey: r.key, bucket: b })} />)}
                    <td className={`bb-cell ${verdict(r.total) ?? "thin"}`}><span className="bb-static"><strong>{pct(r.total.roi)}</strong><small>{r.total.n.toLocaleString()} bets</small></span></td>
                  </tr>
                ))}
                {result.rows.length > 1 && (
                  <tr className="bb-total">
                    <th scope="row">All selected</th>
                    {result.total.map((c, b) => <CellView key={b} cell={c} active={open?.rowKey === null && open?.bucket === b} onOpen={() => setOpen({ rowKey: null, bucket: b })} />)}
                    <td className={`bb-cell ${verdict(result.grand) ?? "thin"}`}><span className="bb-static"><strong>{pct(result.grand.roi)}</strong><small>{result.grand.n.toLocaleString()} bets</small></span></td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
        <p className="bb-legend"><i className="pos" /> range above zero <i className="neg" /> range below zero <i className="flat" /> range includes zero <i className="thin" /> fewer than 30 bets or 10 events. Tap a cell for seasons and example bets.</p>
      </Panel>

      {open && openCell && (
        <Detail data={data} filters={filters} rowKey={open.rowKey} rowLabel={open.rowKey === null ? "All selected" : openRow?.label ?? ""} bucket={open.bucket} cell={openCell} onClose={() => setOpen(null)} />
      )}

      <Panel eyebrow="How to read this" title="About the data">
        <ul className="bb-notes">
          {(meta.notes ?? []).map((n, k) => <li key={k}>{n}</li>)}
          <li>Data built {data.builtAt.replace("T", " ").replace("Z", " UTC")}. {data.n.toLocaleString()} priced bets in the file (edges above -5% at the open or the close).</li>
        </ul>
      </Panel>
    </>
  );
}

export function BetBreakdownsView() {
  const [load, setLoad] = useState<Load>({ state: "loading" });
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let live = true;
    loadDataset()
      .then((data) => { if (live) setLoad({ state: "ready", data }); })
      .catch((error: unknown) => { if (live) setLoad({ state: "error", message: error instanceof Error ? error.message : "The data could not be loaded." }); });
    return () => { live = false; };
  }, [attempt]);

  const intro = (
    <PageIntro
      eyebrow="Owner only"
      title="Bet breakdowns"
      description="How current production's edges have paid historically, by market, book and edge size. Choose open or close prices, line movement, player history and skill."
    />
  );
  if (load.state === "loading") return <div>{intro}<LoadingState label="Loading about 400,000 historical bets (about 4 MB)" /></div>;
  if (load.state === "error") return <div>{intro}<EmptyState title="Bet breakdowns are not available" detail={load.message} onRetry={() => { setLoad({ state: "loading" }); setAttempt((a) => a + 1); }} /></div>;
  return <div>{intro}<Explorer data={load.data} /></div>;
}
