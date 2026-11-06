"use client";

import { useDeferredValue, useEffect, useMemo, useState } from "react";
import { DataThrough } from "./DataThrough";
import { dataThrough } from "./freshness-rules";
import { Chip, EmptyState, PageHeader, Panel, Section, SegmentedControl, Stat } from "./components";
import {
  BET_BREAKDOWNS_KEY, EDGE_BUCKETS, MARKET_GROUPS, OWNER_BOOKS, OWNER_DATA_PREFIX, bookLabel, breakdown, decodeDataset, defaultFilters, examples, pct, verdict,
  type Basis, type Cell, type Dataset, type Filters, type Move, type RowDim, type Sample,
} from "./bet-breakdowns-rules";
import "./bet-breakdowns.css";

const MOVE_WORD: Record<Move, string> = { all: "any line move", toward: "line moved toward us", flat_or_against: "line flat or against", flat: "line flat", against: "line moved against us" };

function roiTone(v: ReturnType<typeof verdict>): "positive" | "negative" | "neutral" { return v === "pos" ? "positive" : v === "neg" ? "negative" : "neutral"; }

/** ROI per edge bucket as zero-centred bars with the 95% range as a whisker and the bet count beside every bar. Tap a row for seasons and example bets. */
function EdgeChart({ cells, basis, activeBucket, onOpen }: { cells: Cell[]; basis: Basis; activeBucket: number | null; onOpen: (bucket: number) => void }) {
  const scale = Math.max(0.05, ...cells.flatMap((c) => (c.n ? [Math.abs(c.roi ?? 0), Math.abs(c.lo ?? 0), Math.abs(c.hi ?? 0)] : [0])));
  const at = (v: number) => 50 + (Math.max(-scale, Math.min(scale, v)) / scale) * 50;
  return (
    <ul className="bb-chart" aria-label={`Return per 1-unit bet by edge size, at the ${basis}`}>
      {cells.map((c, b) => {
        const v = verdict(c);
        const roi = c.roi ?? 0;
        return (
          <li key={EDGE_BUCKETS[b].key}>
            <button type="button" className={activeBucket === b ? "on" : ""} disabled={!c.n} onClick={() => onOpen(b)} aria-label={c.n ? `Edge ${EDGE_BUCKETS[b].label}: ${pct(c.roi)} on ${c.n} bets. Show seasons and examples` : `Edge ${EDGE_BUCKETS[b].label}: no bets`}>
              <span className="bb-ch-label">Edge {EDGE_BUCKETS[b].label}</span>
              <span className="bb-ch-track" aria-hidden="true">
                <i className="bb-ch-zero" />
                {c.n > 0 && <i className={`bb-ch-bar ${roi >= 0 ? "pos" : "neg"} ${v ?? "thin"}`} style={{ left: `${Math.min(at(0), at(roi))}%`, width: `${Math.abs(at(roi) - at(0))}%` }} />}
                {c.n > 0 && c.lo !== null && c.hi !== null && <i className="bb-ch-range" style={{ left: `${at(c.lo)}%`, width: `${at(c.hi) - at(c.lo)}%` }} />}
              </span>
              <span className="bb-ch-val num"><strong>{c.n ? pct(c.roi) : "–"}</strong><small>{c.n.toLocaleString()} bets{c.n && v === null ? ", thin" : ""}</small></span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

type Progress = { loaded: number; total: number | null };
type Load = { state: "loading"; progress: Progress } | { state: "error"; message: string } | { state: "ready"; data: Dataset };
const NO_PROGRESS: Progress = { loaded: 0, total: null };

/** Read the body in chunks so the skeleton can show real progress, then decode after the browser has painted. */
async function loadDataset(onProgress: (p: Progress) => void): Promise<Dataset> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 90_000);
  try {
    const response = await fetch(`${OWNER_DATA_PREFIX}${BET_BREAKDOWNS_KEY}`, { signal: controller.signal, headers: { accept: "application/json" } });
    if (response.status === 403) throw new Error("This page is for the owner only.");
    if (response.status === 404) throw new Error("The bet breakdown data has not been published yet.");
    if (!response.ok) throw new Error(`The data could not be loaded (HTTP ${response.status}).`);
    const length = Number(response.headers.get("content-length"));
    const total = Number.isFinite(length) && length > 0 ? length : null;
    let text: string;
    if (response.body && typeof response.body.getReader === "function") {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      const parts: string[] = [];
      let loaded = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        loaded += value.byteLength;
        parts.push(decoder.decode(value, { stream: true }));
        onProgress({ loaded, total });
      }
      parts.push(decoder.decode());
      text = parts.join("");
    } else {
      text = await response.text();
    }
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    return decodeDataset(JSON.parse(text));
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw new Error("Loading the data timed out. Try again.");
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

const mb = (n: number) => (n / 1_048_576).toFixed(1);

/** The page's shape while ~4 MB downloads: answer, filter bar, edge rows. Fixed heights, so nothing jumps when data lands. */
function BreakdownSkeleton({ progress }: { progress: Progress }) {
  const { loaded, total } = progress;
  const label = loaded === 0 ? "Starting the download of the historical bets" : `Downloading historical bets: ${mb(loaded)}${total ? ` of ${mb(total)}` : ""} MB`;
  return (
    <div className="bb-skel" aria-busy="true">
      <p className="bb-skel-status num" role="status">{label}</p>
      <div className="bb-skel-answer"><span className="skeleton" style={{ height: 36, width: "min(320px, 70%)" }} /><span className="skeleton" style={{ height: 16, width: "min(480px, 90%)" }} /></div>
      <div className="kpi-grid">{[0, 1, 2, 3].map((i) => <span key={i} className="skeleton" style={{ height: 92 }} />)}</div>
      <span className="skeleton" style={{ height: 46, marginTop: 16 }} />
      <div className="bb-skel-rows">{[0, 1, 2, 3, 4, 5].map((i) => <span key={i} className="skeleton" style={{ height: 44 }} />)}</div>
    </div>
  );
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

function filterSummary(data: Dataset, f: Filters): string {
  const { meta } = data;
  const names = (idx: number[], list: string[], all: number, label: (x: string) => string, noun: string) => idx.length === 0 ? `no ${noun}` : idx.length === all ? `all ${noun}` : idx.length <= 2 ? idx.map((i) => label(list[i])).join(" + ") : `${idx.length} ${noun}`;
  const parts = [
    f.basis === "open" ? "Open prices" : "Close prices",
    names(f.markets, meta.markets, meta.markets.length, (m) => meta.market_labels[m] ?? m, "markets"),
    names(f.tours, meta.tours, meta.tours.length, (t) => (t === "pga" ? "PGA Tour" : "DP World Tour"), "tours"),
    names(f.books, meta.books, meta.books.length, bookLabel, "books"),
    names(f.seasons, meta.seasons.map(String), meta.seasons.length, (x) => x, "seasons"),
    MOVE_WORD[f.move],
  ];
  if (f.sample !== "all") parts.push(f.sample === "oos" ? "out of sample only" : "in-sample only");
  if (f.hist.length) parts.push("history filtered");
  if (f.skill.length) parts.push("skill filtered");
  if (f.oppHist.length) parts.push("opponent history filtered");
  if (f.minDec !== null || f.maxDec !== null) parts.push(`odds ${f.minDec ?? "any"} to ${f.maxDec ?? "any"}`);
  return parts.join(" · ");
}

/** The answer first: overall return for the current selection, how many bets stand behind it, and whether the range clears zero. */
function Answer({ data, filters, grand, considered }: { data: Dataset; filters: Filters; grand: Cell; considered: number }) {
  void data;
  if (considered === 0) return <EmptyState title="No bets match these filters" detail="Open Filters and widen the selection: add a market, a book or a season." />;
  const v = verdict(grand);
  const reading = v === "pos" ? "Reliably positive: the 95% range is above zero." : v === "neg" ? "Reliably negative: the 95% range is below zero." : v === "flat" ? "Not distinguishable from zero: the 95% range includes it." : "Too few bets or events to judge.";
  return (
    <div className="bb-answer">
      <p className="bb-headline"><strong className={`num ${v === "pos" ? "pos" : v === "neg" ? "neg" : ""}`}>{pct(grand.roi)}</strong> return per unit staked <Chip tone={v === "pos" ? "positive" : v === "neg" ? "negative" : "warning"}>{v === "pos" ? "Reliable gain" : v === "neg" ? "Reliable loss" : v === "flat" ? "Inconclusive" : "Too few bets"}</Chip></p>
      <p className="bb-reading">{reading} Flat 1-unit bets at the {filters.basis}.</p>
      <div className="kpi-grid">
        <Stat label="Return on investment (ROI)" value={pct(grand.roi)} tone={roiTone(v)} note={grand.lo !== null && grand.hi !== null ? `95% range ${pct(grand.lo)} to ${pct(grand.hi)}` : "too few events for a range"} hint="Average profit per 1-unit flat bet, as a share of the stake." />
        <Stat label="Bets" value={grand.n.toLocaleString()} note={`${grand.events.toLocaleString()} events`} hint="The range counts whole events as one cluster, so many bets in one event do not look like many independent results." />
        <Stat label="Won" value={grand.winRate === null ? "–" : `${(grand.winRate * 100).toFixed(0)}%`} note="of bets returned a profit" />
        <Stat label="Claimed edge" value={pct(grand.meanEdge)} note="average at this price; ROI should land near it if the model is right" hint="Our probability times the decimal price, minus one." />
      </div>
    </div>
  );
}

function Explorer({ data }: { data: Dataset }) {
  const { meta } = data;
  const [filters, setFilters] = useState<Filters>(() => defaultFilters(meta));
  const [open, setOpen] = useState<{ rowKey: number | null; bucket: number } | null>(null);
  const set = (patch: Partial<Filters>) => { setFilters((f) => ({ ...f, ...patch })); setOpen(null); };
  const result = useMemo(() => breakdown(data, filters), [data, filters]);
  // The answer and filters paint first; the chart and the by-row breakdown follow in a lower-priority render.
  const later = useDeferredValue(result, null);

  const marketIdx = (keys: string[]) => keys.map((k) => meta.markets.indexOf(k)).filter((i) => i >= 0);
  const ownerBooks = OWNER_BOOKS.map((b) => meta.books.indexOf(b)).filter((i) => i >= 0);
  const openRow = open ? (open.rowKey === null ? null : result.rows.find((r) => r.key === open.rowKey) ?? null) : null;
  const openCell = open ? (open.rowKey === null ? result.total[open.bucket] : openRow?.cells[open.bucket]) : null;

  return (
    <>
      <Answer data={data} filters={filters} grand={result.grand} considered={result.considered} />

      <details className="bb-filterbox">
        <summary>
          <span className="bb-fb-title">Filters</span>
          <span className="bb-fb-sum">{filterSummary(data, filters)}</span>
          <span className="bb-fb-act">Change</span>
        </summary>
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
      </details>

      {later === null && result.considered > 0 && <div className="bb-skel-rows" aria-hidden="true">{[0, 1, 2, 3, 4, 5].map((i) => <span key={i} className="skeleton" style={{ height: 44 }} />)}</div>}
      {later !== null && later.considered > 0 && (
        <Section id="edge-chart" title="Return by edge size" purpose={`Profit per 1-unit flat bet at the ${filters.basis}. Bars start at zero; the thin line is the 95% range. Tap a bar for seasons and examples.`}>
          <EdgeChart cells={later.total} basis={filters.basis} activeBucket={open?.rowKey === null ? open.bucket : null} onOpen={(bucket) => setOpen({ rowKey: null, bucket })} />
        </Section>
      )}

      {later !== null && (
      <Panel
        eyebrow={`Same bets, split by ${ROW_OPTIONS.find((o) => o.value === filters.rowDim)?.label.toLowerCase()}. Range = 95%, events counted as clusters`}
        title={`Return by ${ROW_OPTIONS.find((o) => o.value === filters.rowDim)?.label.toLowerCase()}`}
        actions={<label className="select-control bb-rows"><span>Split by</span><select value={filters.rowDim} onChange={(e) => set({ rowDim: e.target.value as RowDim })}>{ROW_OPTIONS.filter((o) => o.value !== "opp_hist" || data.hasOppHist).map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}</select></label>}
      >
        {later.considered === 0 ? (
          <EmptyState title="No bets match these filters" detail="Widen the filters: add a market, a book or a season." />
        ) : (
          <>
            <div className="bb-table">
              <table>
                <thead>
                  <tr><th>{ROW_OPTIONS.find((o) => o.value === filters.rowDim)?.label}</th>{EDGE_BUCKETS.map((b) => <th key={b.key}>Edge {b.label}</th>)}<th>All edges ≥ 0</th></tr>
                </thead>
                <tbody>
                  {later.rows.map((r) => (
                    <tr key={r.key}>
                      <th scope="row">{r.label}</th>
                      {r.cells.map((c, b) => <CellView key={b} cell={c} active={open?.rowKey === r.key && open?.bucket === b} onOpen={() => setOpen({ rowKey: r.key, bucket: b })} />)}
                      <td className={`bb-cell ${verdict(r.total) ?? "thin"}`}><span className="bb-static"><strong>{pct(r.total.roi)}</strong><small>{r.total.n.toLocaleString()} bets</small></span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <ul className="bb-compact" aria-label={`Return by ${ROW_OPTIONS.find((o) => o.value === filters.rowDim)?.label.toLowerCase()}`}>
              {later.rows.map((r) => {
                const v = verdict(r.total);
                return (
                  <li key={r.key}>
                    <div className="bb-cr-head"><strong>{r.label}</strong><span className={`num bb-cr-total ${v ?? "thin"}`}>{pct(r.total.roi)}<small>{r.total.n.toLocaleString()} bets</small></span></div>
                    <div className="bb-cr-cells">
                      {r.cells.map((c, b) => c.n ? (
                        <button key={b} type="button" className={`bb-cr-cell ${verdict(c) ?? "thin"} ${open?.rowKey === r.key && open?.bucket === b ? "active" : ""}`} onClick={() => setOpen({ rowKey: r.key, bucket: b })} aria-label={`${r.label}, edge ${EDGE_BUCKETS[b].label}: ${pct(c.roi)} on ${c.n} bets. Show seasons and examples`}>
                          <small>Edge {EDGE_BUCKETS[b].label}</small><strong className="num">{pct(c.roi)}</strong><small className="num">{c.n.toLocaleString()}</small>
                        </button>
                      ) : null)}
                    </div>
                  </li>
                );
              })}
            </ul>
          </>
        )}
        <p className="bb-legend"><i className="pos" /> range above zero <i className="neg" /> range below zero <i className="flat" /> range includes zero <i className="thin" /> fewer than 30 bets or 10 events. Tap a cell for seasons and example bets.</p>
      </Panel>
      )}

      {open && openCell && (
        <Detail data={data} filters={filters} rowKey={open.rowKey} rowLabel={open.rowKey === null ? "All selected" : openRow?.label ?? ""} bucket={open.bucket} cell={openCell} onClose={() => setOpen(null)} />
      )}

      <details className="bb-about">
        <summary>How to read this: about the data</summary>
        <ul className="bb-notes">
          {(meta.notes ?? []).map((n, k) => <li key={k}>{n}</li>)}
          <li>Data built {data.builtAt.replace("T", " ").replace("Z", " UTC")}. {data.n.toLocaleString()} priced bets in the file (edges above -5% at the open or the close).</li>
        </ul>
      </details>
    </>
  );
}

export function BetBreakdownsView() {
  const [load, setLoad] = useState<Load>({ state: "loading", progress: NO_PROGRESS });
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let live = true;
    loadDataset((progress) => { if (live) setLoad((cur) => (cur.state === "loading" ? { state: "loading", progress } : cur)); })
      .then((data) => { if (live) setLoad({ state: "ready", data }); })
      .catch((error: unknown) => { if (live) setLoad({ state: "error", message: error instanceof Error ? error.message : "The data could not be loaded." }); });
    return () => { live = false; };
  }, [attempt]);

  const intro = (
    <PageHeader
      title="Bet breakdowns"
      purpose="How the model's claimed edges have actually paid in past seasons, by edge size, book and market."
    />
  );
  if (load.state === "loading") return <div>{intro}<BreakdownSkeleton progress={load.progress} /></div>;
  if (load.state === "error") return <div>{intro}<EmptyState title="Bet breakdowns are not available" detail={load.message} onRetry={() => { setLoad({ state: "loading", progress: NO_PROGRESS }); setAttempt((a) => a + 1); }} /></div>;
  const through = dataThrough(load.data.meta.max_event_date, "Graded bets through");
  const built = dataThrough(load.data.builtAt, "built");
  return <div>{intro}<DataThrough text={[through, built].filter(Boolean).join(" · ") || "The file does not say which event date it runs through."} /><Explorer data={load.data} /></div>;
}
