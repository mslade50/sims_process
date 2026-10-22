/**
 * Bet breakdowns (owner-only page, owner request 2026-10-21): historical returns of current production by market, book and edge bucket.
 *
 * Data: R2 owner/bet_breakdowns.json (sim_prep research/sportsbook_model/rebuild_v1/experiments/bet_breakdown_20261021/build.py),
 * one row per book quote x side with production's probability, the open and close decimal prices and the 1-unit result at each.
 * Columns arrive as base64 little-endian typed arrays. Everything here is pure (no DOM) so tests import it directly.
 */

export const BET_BREAKDOWNS_KEY = "bet_breakdowns.json";
export const OWNER_DATA_PREFIX = "/api/owner-data/";

export type Meta = {
  markets: string[];
  market_labels: Record<string, string>;
  tours: string[];
  books: string[];
  seasons: number[];
  events: string[];
  event_names: string[];
  players: string[];
  hist_buckets: string[];
  skill_buckets: string[];
  notes?: string[];
  fit_flags?: Record<string, string>;    // bit -> plain-words reason a bet is in-sample (owner 2026-10-22)
};
type RawDoc = { schema: string; built_at: string; meta: Meta; n: number; cols: Record<string, string> };

export type Dataset = {
  builtAt: string;
  meta: Meta;
  n: number;
  market: Uint8Array; tour: Uint8Array; book: Uint8Array; season: Uint8Array; event: Uint16Array; player: Uint16Array; opp: Uint16Array;
  p: Uint16Array; pTie: Uint16Array; voidTie: Uint8Array; decOpen: Uint16Array; decClose: Uint16Array; profitOpen: Int32Array; profitClose: Int32Array; hist: Uint8Array; skill: Uint8Array;
  /** In-sample flags per bet: bit 1 = season used to fit parts of the model, bit 2 = R3/R4 market (leaderboard layer fitted on every season); 0 = out of sample. */
  fit: Uint8Array;
  hasFit: boolean;
  /** Opponent's history bucket (2-ball and 72-hole matchups; 255 = none). All 255 for files built before the column existed. */
  oppHist: Uint8Array;
  hasOppHist: boolean;            // false for files built before the flag existed (every bet then reads as 0)
  fitSeasons: Set<number>;    // season indexes with any bit-1 bet
};

export const SCHEMA = "golfprice.bet_breakdowns.v2";
const NONE_PROFIT = -2147483648;
const NONE_BUCKET = 255;
const NONE_OPP = 65535;

function bytesFrom(b64: string): Uint8Array {
  if (typeof atob === "function") {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i += 1) out[i] = bin.charCodeAt(i);
    return out;
  }
  return new Uint8Array(Buffer.from(b64, "base64"));
}

function typed<T extends ArrayBufferView>(b64: string | undefined, Ctor: { new (buffer: ArrayBuffer): T; BYTES_PER_ELEMENT: number }, n: number, name: string): T {
  if (typeof b64 !== "string") throw new Error(`column ${name} is missing`);
  const bytes = bytesFrom(b64);
  if (bytes.byteLength !== n * Ctor.BYTES_PER_ELEMENT) throw new Error(`column ${name} has the wrong length`);
  const copy = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(copy).set(bytes);
  return new Ctor(copy);
}

/** Decode the published document. Throws a plain-words Error when it is not the expected shape. */
export function decodeDataset(json: unknown): Dataset {
  const doc = json as RawDoc;
  if (!doc || typeof doc !== "object" || doc.schema !== SCHEMA || !doc.meta || !doc.cols || !Number.isInteger(doc.n)) {
    throw new Error("The bet breakdown file is not in the expected format");
  }
  const { n, cols } = doc;
  const hasFit = typeof cols.fit === "string";
  const hasOppHist = typeof cols.opp_hist === "string";
  const oppHist = hasOppHist ? typed(cols.opp_hist, Uint8Array, n, "opp_hist") : new Uint8Array(n).fill(NONE_BUCKET);
  const fit = hasFit ? typed(cols.fit, Uint8Array, n, "fit") : new Uint8Array(n);
  const season = typed(cols.season, Uint8Array, n, "season");
  const fitSeasons = new Set<number>();
  for (let i = 0; i < n; i += 1) if (fit[i] & 1) fitSeasons.add(season[i]);
  return {
    fit, hasFit, fitSeasons, oppHist, hasOppHist,
    builtAt: String(doc.built_at ?? ""), meta: doc.meta, n,
    market: typed(cols.market, Uint8Array, n, "market"), tour: typed(cols.tour, Uint8Array, n, "tour"), book: typed(cols.book, Uint8Array, n, "book"),
    season, event: typed(cols.event, Uint16Array, n, "event"), player: typed(cols.player, Uint16Array, n, "player"),
    opp: typed(cols.opp, Uint16Array, n, "opp"), p: typed(cols.p, Uint16Array, n, "p"),
    pTie: typed(cols.p_tie, Uint16Array, n, "p_tie"), voidTie: typed(cols.void_tie, Uint8Array, n, "void_tie"), decOpen: typed(cols.dec_open, Uint16Array, n, "dec_open"),
    decClose: typed(cols.dec_close, Uint16Array, n, "dec_close"), profitOpen: typed(cols.profit_open, Int32Array, n, "profit_open"),
    profitClose: typed(cols.profit_close, Int32Array, n, "profit_close"), hist: typed(cols.hist, Uint8Array, n, "hist"), skill: typed(cols.skill, Uint8Array, n, "skill"),
  };
}

/* ------------------------------------------------------------------ filters */
export type Basis = "open" | "close";
export type Move = "all" | "toward" | "flat" | "against" | "flat_or_against";
export type RowDim = "book" | "market" | "season" | "hist" | "skill" | "tour" | "opp_hist";
export type Sample = "all" | "oos" | "ins";

export type Filters = {
  basis: Basis;
  markets: number[];          // indexes into meta.markets; empty = none selected
  tours: number[];
  books: number[];
  seasons: number[];
  hist: number[];             // empty = any (rows without a bucket included)
  skill: number[];
  oppHist: number[];          // opponent's history bucket; empty = any. Selecting any bucket keeps only 2-ball and 72-hole matchups
  move: Move;
  minDec: number | null;
  maxDec: number | null;
  rowDim: RowDim;
  sample: Sample;             // all bets, out of sample only (fit flag 0), in-sample only
};

/** Edge buckets (fractions): [lo, hi). The last is open-ended. */
export const EDGE_BUCKETS: Array<{ key: string; label: string; lo: number; hi: number }> = [
  { key: "0", label: "0-3%", lo: 0, hi: 0.03 },
  { key: "3", label: "3-5%", lo: 0.03, hi: 0.05 },
  { key: "5", label: "5-10%", lo: 0.05, hi: 0.1 },
  { key: "10", label: "10-15%", lo: 0.1, hi: 0.15 },
  { key: "15", label: "15-20%", lo: 0.15, hi: 0.2 },
  { key: "20", label: "20%+", lo: 0.2, hi: Infinity },
];
export function bucketOf(edge: number): number {
  for (let b = 0; b < EDGE_BUCKETS.length; b += 1) if (edge >= EDGE_BUCKETS[b].lo && edge < EDGE_BUCKETS[b].hi) return b;
  return -1;
}

/** The owner's books (owner, 2026-10-21): Bovada, Pinnacle, BetCris, BetOnline (exchanges are not in the DataGolf archive). */
export const OWNER_BOOKS = ["bovada", "pinnacle", "betcris", "betonline"];

export const MARKET_GROUPS: Array<{ label: string; keys: string[] }> = [
  { label: "Tournament matchups", keys: ["t72_match"] },
  { label: "Round 2-balls", keys: ["r1_match", "r2_match", "r3_match", "r4_match"] },
  { label: "Round 3-balls", keys: ["r1_3ball", "r2_3ball", "r3_3ball", "r4_3ball"] },
  { label: "Finish markets", keys: ["win", "top_5", "top_10", "top_20", "make_cut", "miss_cut"] },
];

export function defaultFilters(meta: Meta): Filters {
  const idx = (list: string[], keys: string[]) => keys.map((k) => list.indexOf(k)).filter((i) => i >= 0);
  const books = idx(meta.books, OWNER_BOOKS).sort((a, b) => a - b);
  return {
    basis: "open", markets: idx(meta.markets, ["t72_match"]), tours: [meta.tours.indexOf("pga")].filter((i) => i >= 0),
    books: books.length ? books : meta.books.map((_, i) => i), seasons: meta.seasons.map((_, i) => i),
    hist: [], skill: [], oppHist: [], move: "all", minDec: null, maxDec: null, rowDim: "book", sample: "all",
  };
}

/* ------------------------------------------------------------------ aggregation */
export type Cell = {
  n: number;
  roi: number | null;          // mean 1-unit profit
  lo: number | null;           // 95% range, event-clustered
  hi: number | null;
  events: number;
  winRate: number | null;      // share of bets with profit > 0
  meanEdge: number | null;     // claimed edge
  clv: number | null;          // mean (open price / close price - 1), rows with both prices
  towardShare: number | null;  // share whose price shortened by the close
  bySeason: Array<{ season: number; n: number; roi: number | null }>;
};
export type Breakdown = { rows: Array<{ key: number; label: string; cells: Cell[]; total: Cell }>; total: Cell[]; grand: Cell; considered: number };

type Acc = { n: number; sum: number; sumEdge: number; wins: number; clvN: number; clvSum: number; toward: number; ev: Map<number, [number, number]>; season: Map<number, [number, number]> };
const newAcc = (): Acc => ({ n: 0, sum: 0, sumEdge: 0, wins: 0, clvN: 0, clvSum: 0, toward: 0, ev: new Map(), season: new Map() });

function add(acc: Acc, profit: number, edge: number, event: number, season: number, clv: number | null) {
  acc.n += 1; acc.sum += profit; acc.sumEdge += edge; if (profit > 0) acc.wins += 1;
  if (clv !== null) { acc.clvN += 1; acc.clvSum += clv; if (clv > 1e-9) acc.toward += 1; }
  const e = acc.ev.get(event); if (e) { e[0] += profit; e[1] += 1; } else acc.ev.set(event, [profit, 1]);
  const s = acc.season.get(season); if (s) { s[0] += profit; s[1] += 1; } else acc.season.set(season, [profit, 1]);
}

/** Two-sided 95% Student t critical value (df >= 1); exact table to 30, then 1.96 + 2.5/df (within 0.002). */
const T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.16, 2.145, 2.131, 2.12, 2.11, 2.101, 2.093, 2.086, 2.08, 2.074, 2.069, 2.064, 2.06, 2.056, 2.052, 2.048, 2.045, 2.042];
export function t975(df: number): number {
  if (df < 1) return Infinity;
  return df <= 30 ? T975[Math.floor(df) - 1] : 1.96 + 2.5 / df;
}

/** Minimum independent events before a cell is coloured as clearly above or below zero. */
export const MIN_EVENTS = 10;

/**
 * Event-clustered 95% range of a ratio mean: bets in one event share weather, form and line moves, so events, not bets, are the
 * independent units. Uses a t critical value with (events - 1) degrees of freedom so a handful of events gives an honestly wide range.
 */
export function clusteredRange(sums: Array<[number, number]>, total: number, n: number): [number, number] | null {
  const g = sums.length;
  if (n < 2 || g < 2) return null;
  const mean = total / n;
  let ss = 0;
  for (const [s, k] of sums) { const r = s - mean * k; ss += r * r; }
  const se = Math.sqrt((g / (g - 1)) * ss) / n;
  const t = t975(g - 1);
  return [mean - t * se, mean + t * se];
}

function finish(acc: Acc, seasons: number[]): Cell {
  const range = clusteredRange([...acc.ev.values()], acc.sum, acc.n);
  return {
    n: acc.n, roi: acc.n ? acc.sum / acc.n : null, lo: range ? range[0] : null, hi: range ? range[1] : null, events: acc.ev.size,
    winRate: acc.n ? acc.wins / acc.n : null, meanEdge: acc.n ? acc.sumEdge / acc.n : null,
    clv: acc.clvN ? acc.clvSum / acc.clvN : null, towardShare: acc.clvN ? acc.toward / acc.clvN : null,
    bySeason: [...acc.season.entries()].sort((a, b) => a[0] - b[0]).map(([s, [sum, k]]) => ({ season: seasons[s] ?? s, n: k, roi: k ? sum / k : null })),
  };
}

function setOf(list: number[]): Set<number> { return new Set(list); }

/** Does row i pass the filters? Returns its edge and profit at the chosen basis, or null. */
export function rowView(d: Dataset, f: Filters, sets: ReturnType<typeof filterSets>, i: number): { edge: number; profit: number; clv: number | null } | null {
  if (!sets.markets.has(d.market[i]) || !sets.tours.has(d.tour[i]) || !sets.books.has(d.book[i]) || !sets.seasons.has(d.season[i])) return null;
  if (sets.hist.size && !sets.hist.has(d.hist[i])) return null;
  if (sets.skill.size && !sets.skill.has(d.skill[i])) return null;
  if (sets.oppHist.size && !sets.oppHist.has(d.oppHist[i])) return null;
  if (f.sample === "oos" && d.fit[i] !== 0) return null;
  if (f.sample === "ins" && d.fit[i] === 0) return null;
  const decOpen = d.decOpen[i] / 100;
  const decClose = d.decClose[i] ? d.decClose[i] / 100 : null;
  const clv = decClose ? decOpen / decClose - 1 : null;
  if (f.move !== "all") {
    if (clv === null) return null;
    const toward = clv > 1e-9, against = clv < -1e-9;
    if (f.move === "toward" && !toward) return null;
    if (f.move === "against" && !against) return null;
    if (f.move === "flat" && (toward || against)) return null;
    if (f.move === "flat_or_against" && toward) return null;
  }
  const dec = f.basis === "open" ? decOpen : decClose;
  if (dec === null) return null;
  if (f.minDec !== null && dec < f.minDec) return null;
  if (f.maxDec !== null && dec > f.maxDec) return null;
  const raw = f.basis === "open" ? d.profitOpen[i] : d.profitClose[i];
  if (raw === NONE_PROFIT) return null;
  return { edge: edgeAt(d, i, dec), profit: raw / 1000, clv };
}

/**
 * Expected return per unit staked at decimal price `dec`. Matchups that void on a tie carry P(win | no tie) and refund the stake
 * on a tie, so their edge is (1 - P(tie)) x (p x dec - 1); everything else (tie-loses books, 3-ball dead-heat shares, finish markets) is p x dec - 1.
 */
export function edgeAt(d: Dataset, i: number, dec: number): number {
  const plain = (d.p[i] / 65535) * dec - 1;
  return d.voidTie[i] ? (1 - d.pTie[i] / 65535) * plain : plain;
}

export function filterSets(f: Filters) {
  return { markets: setOf(f.markets), tours: setOf(f.tours), books: setOf(f.books), seasons: setOf(f.seasons), hist: setOf(f.hist), skill: setOf(f.skill), oppHist: setOf(f.oppHist ?? []) };
}

function rowKey(d: Dataset, dim: RowDim, i: number): number {
  switch (dim) {
    case "book": return d.book[i];
    case "market": return d.market[i];
    case "season": return d.season[i];
    case "hist": return d.hist[i];
    case "skill": return d.skill[i];
    case "tour": return d.tour[i];
    case "opp_hist": return d.oppHist[i];
  }
}

export function rowLabel(meta: Meta, dim: RowDim, key: number): string {
  if (dim === "opp_hist" && key === NONE_BUCKET) return "No single opponent";
  if ((dim === "hist" || dim === "skill") && key === NONE_BUCKET) return "Unknown";
  switch (dim) {
    case "book": return bookLabel(meta.books[key] ?? String(key));
    case "market": return meta.market_labels[meta.markets[key]] ?? meta.markets[key] ?? String(key);
    case "season": return String(meta.seasons[key] ?? key);
    case "hist": return meta.hist_buckets[key] ?? String(key);
    case "opp_hist": return `vs ${meta.hist_buckets[key] ?? String(key)}`;
    case "skill": return meta.skill_buckets[key] ?? String(key);
    case "tour": return ({ pga: "PGA Tour", euro: "DP World Tour" } as Record<string, string>)[meta.tours[key]] ?? String(key);
  }
}

const BOOK_LABELS: Record<string, string> = {
  betonline: "BetOnline", betcris: "BetCris", bovada: "Bovada", pinnacle: "Pinnacle", bet365: "bet365", draftkings: "DraftKings", fanduel: "FanDuel",
  betmgm: "BetMGM", caesars: "Caesars", unibet: "Unibet", circa: "Circa", williamhill: "William Hill", betway: "Betway", skybet: "Sky Bet",
};
export const bookLabel = (key: string) => BOOK_LABELS[key] ?? key;

export function breakdown(d: Dataset, f: Filters): Breakdown {
  const sets = filterSets(f);
  const rows = new Map<number, { cells: Acc[]; total: Acc }>();
  const total = EDGE_BUCKETS.map(newAcc);
  const grand = newAcc();
  let considered = 0;
  for (let i = 0; i < d.n; i += 1) {
    const v = rowView(d, f, sets, i);
    if (!v) continue;
    const b = bucketOf(v.edge);
    if (b < 0) continue;
    considered += 1;
    const key = rowKey(d, f.rowDim, i);
    let r = rows.get(key);
    if (!r) { r = { cells: EDGE_BUCKETS.map(newAcc), total: newAcc() }; rows.set(key, r); }
    add(r.cells[b], v.profit, v.edge, d.event[i], d.season[i], v.clv);
    add(total[b], v.profit, v.edge, d.event[i], d.season[i], v.clv);
    if (v.edge >= 0) { add(r.total, v.profit, v.edge, d.event[i], d.season[i], v.clv); add(grand, v.profit, v.edge, d.event[i], d.season[i], v.clv); }
  }
  const seasons = d.meta.seasons;
  return {
    rows: [...rows.entries()]
      .map(([key, r]) => ({ key, label: rowLabel(d.meta, f.rowDim, key), cells: r.cells.map((a) => finish(a, seasons)), total: finish(r.total, seasons) }))
      .sort((a, b) => (f.rowDim === "book" ? b.total.n - a.total.n : a.key - b.key)),
    total: total.map((a) => finish(a, seasons)), grand: finish(grand, seasons), considered,
  };
}

/* ------------------------------------------------------------------ examples */
export type Example = {
  event: string; season: number; market: string; book: string; player: string; opponent: string | null;
  decOpen: number; decClose: number | null; p: number; edge: number; profit: number; inSample: boolean;
};

/** Bets in one cell (row key + edge bucket; row key null = all rows), biggest edge first, at most `limit`. */
export function examples(d: Dataset, f: Filters, rowKeyValue: number | null, bucket: number, limit = 60): { list: Example[]; total: number } {
  const sets = filterSets(f);
  const hits: Array<{ i: number; edge: number; profit: number }> = [];
  for (let i = 0; i < d.n; i += 1) {
    const v = rowView(d, f, sets, i);
    if (!v || bucketOf(v.edge) !== bucket) continue;
    if (rowKeyValue !== null && rowKey(d, f.rowDim, i) !== rowKeyValue) continue;
    hits.push({ i, edge: v.edge, profit: v.profit });
  }
  hits.sort((a, b) => b.edge - a.edge);
  const m = d.meta;
  return {
    total: hits.length,
    list: hits.slice(0, limit).map(({ i, edge, profit }) => ({
      event: m.event_names[d.event[i]] ?? m.events[d.event[i]] ?? "", season: m.seasons[d.season[i]] ?? 0,
      market: m.market_labels[m.markets[d.market[i]]] ?? m.markets[d.market[i]], book: bookLabel(m.books[d.book[i]] ?? ""),
      player: displayName(m.players[d.player[i]] ?? ""), opponent: d.opp[i] === NONE_OPP ? null : displayName(m.players[d.opp[i]] ?? ""),
      decOpen: d.decOpen[i] / 100, decClose: d.decClose[i] ? d.decClose[i] / 100 : null, p: d.p[i] / 65535, edge, profit, inSample: d.fit[i] !== 0,
    })),
  };
}

/** "Scheffler, Scottie" -> "Scottie Scheffler". */
export function displayName(name: string): string {
  const parts = name.split(",").map((s) => s.trim()).filter(Boolean);
  return parts.length === 2 ? `${parts[1]} ${parts[0]}` : name;
}

/* ------------------------------------------------------------------ formatting */
export function pct(v: number | null, digits = 1): string {
  if (v === null || !Number.isFinite(v)) return "–";
  const text = (v * 100).toFixed(digits);
  if (Number(text) === 0) return `${(0).toFixed(digits)}%`;
  return `${v > 0 ? "+" : ""}${text}%`;
}

/** "pos" when the whole range is above zero, "neg" when below, else "flat"; null when too few bets or events to say. */
export function verdict(c: Pick<Cell, "n" | "events" | "lo" | "hi">, minN = 30): "pos" | "neg" | "flat" | null {
  if (c.n < minN || c.events < MIN_EVENTS || c.lo === null || c.hi === null) return null;
  if (c.lo > 0) return "pos";
  if (c.hi < 0) return "neg";
  return "flat";
}
