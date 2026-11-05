export type MarketPlayer = { id: number; name: string; wave: string | null; line: number; over_decimal: number; under_decimal: number; under_no_vig: number; model_mean: number; market_mean: number; shift: number; quote_at: string };
export type MarketOutcome = { status: "pending" | "partial" | "complete"; settled: number; actual_quoted_mean: number | null; model_error: number | null; market_error: number | null; model_brier: number | null; market_brier: number | null };
export type MarketBook = { book: string; quoted: number; field_players: number; model_field_mean: number; market_field_mean: number; delta: number; wave_weighted_field_mean: number | null; median_shift_field_mean: number | null; additive_devig_field_mean: number | null; quote_at: string; freshness_unknown: boolean; players: MarketPlayer[]; outcome: MarketOutcome };
export type MarketRecord = { id: string; round: number; origin: "recorded" | "reconstructed"; model_run: string; model_hash: string; model_as_of: string; capture_at: string; first_tee_utc: string; books: MarketBook[] };
export type MarketDocument = { event_uid: string; generated_at: string; records: MarketRecord[]; outcome_error: string | null };
const object = (v: unknown): Record<string, unknown> => v && typeof v === "object" && !Array.isArray(v) ? v as Record<string, unknown> : {};
const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const num = (v: unknown): number | null => finite(v) ? v : null;
const str = (v: unknown): string => typeof v === "string" ? v : "";
const time = (v: unknown): number => typeof v === "string" && /(Z|[+-]\d\d:\d\d)$/.test(v) ? Date.parse(v) : NaN;
export const marketKey = (uid: string) => `golfprice/market_scoring/${uid.replace(/:/g, "_")}/latest.json`;
export const bookName = (name: string) => ({ buckeye: "Buckeye", draftkings: "DraftKings", kalshi: "Kalshi" })[name] ?? name;
export const signedScore = (n: number | null, digits = 2) => n == null ? "—" : `${Number(n.toFixed(digits)) > 0 ? "+" : ""}${(Math.abs(n) < .5 * 10 ** -digits ? 0 : n).toFixed(digits)}`;

export function parseMarketScoring(raw: unknown, eventUid: string): MarketDocument | null {
  const d = object(raw);
  if (d.schema !== "golfprice.market_scoring.v1" || d.method !== "pmf-location-multiplicative-v1" || d.event_uid !== eventUid || !Array.isArray(d.records)) return null;
  const records: MarketRecord[] = [];
  for (const value of d.records) {
    const r = object(value);
    if (r.event_uid !== eventUid || !str(r.id) || !str(r.model_run) || !str(r.model_hash) || !Number.isInteger(r.round) || Number(r.round) < 1 || Number(r.round) > 4 || !["recorded", "reconstructed"].includes(str(r.origin)) || !Array.isArray(r.books)) continue;
    const issued = time(r.model_as_of), captured = time(r.capture_at), tee = time(r.first_tee_utc);
    if (![issued, captured, tee].every(Number.isFinite) || issued > captured || captured >= tee) continue;
    const books: MarketBook[] = [];
    for (const value of r.books) {
      const b = object(value), o = object(b.outcome);
      if (!str(b.book) || ![b.model_field_mean, b.market_field_mean, b.delta].every(finite) || !Number.isInteger(b.quoted) || Number(b.quoted) <= 0 || !Number.isInteger(b.field_players) || Number(b.field_players) < Number(b.quoted) || !Number.isFinite(time(b.quote_at)) || time(b.quote_at) > captured) continue;
      if (Math.abs(Number(b.market_field_mean) - Number(b.model_field_mean) - Number(b.delta)) > .00001) continue;
      const players = (Array.isArray(b.players) ? b.players : []).flatMap((value): MarketPlayer[] => {
        const p = object(value);
        if (!Number.isInteger(p.id) || !str(p.name) || ![p.line, p.over_decimal, p.under_decimal, p.under_no_vig, p.model_mean, p.market_mean, p.shift].every(finite) || Number(p.under_no_vig) < 0 || Number(p.under_no_vig) > 1 || Number(p.over_decimal) <= 1 || Number(p.under_decimal) <= 1) return [];
        return [{ id: Number(p.id), name: str(p.name), wave: str(p.wave) || null, line: Number(p.line), over_decimal: Number(p.over_decimal), under_decimal: Number(p.under_decimal), under_no_vig: Number(p.under_no_vig), model_mean: Number(p.model_mean), market_mean: Number(p.market_mean), shift: Number(p.shift), quote_at: str(p.quote_at) }];
      });
      const status = o.status === "complete" || o.status === "partial" ? o.status : "pending";
      books.push({ book: str(b.book), quoted: Number(b.quoted), field_players: Number(b.field_players), model_field_mean: Number(b.model_field_mean), market_field_mean: Number(b.market_field_mean), delta: Number(b.delta), wave_weighted_field_mean: num(b.wave_weighted_field_mean), median_shift_field_mean: num(b.median_shift_field_mean), additive_devig_field_mean: num(b.additive_devig_field_mean), quote_at: str(b.quote_at), freshness_unknown: b.freshness_unknown === true, players, outcome: { status, settled: num(o.settled) ?? 0, actual_quoted_mean: num(o.actual_quoted_mean), model_error: num(o.model_error), market_error: num(o.market_error), model_brier: num(o.model_brier), market_brier: num(o.market_brier) } });
    }
    if (books.length) records.push({ id: str(r.id), round: Number(r.round), origin: r.origin as MarketRecord["origin"], model_run: str(r.model_run), model_hash: str(r.model_hash), model_as_of: str(r.model_as_of), capture_at: str(r.capture_at), first_tee_utc: str(r.first_tee_utc), books });
  }
  records.sort((a, b) => time(a.capture_at) - time(b.capture_at) || a.id.localeCompare(b.id));
  return { event_uid: eventUid, generated_at: str(d.generated_at), records, outcome_error: str(d.outcome_error) || null };
}

// Never fall back to another forecast when a checkpoint has no matching quote.
export const matchingRecord = (records: MarketRecord[], run: string, round: number | null) => records.filter((r) => r.model_run === run && r.round === round).at(-1) ?? null;
export function historyRows(records: MarketRecord[]) {
  const latest = new Map<string, { record: MarketRecord; book: MarketBook }>();
  for (const record of [...records].sort((a, b) => time(a.capture_at) - time(b.capture_at))) for (const book of record.books) latest.set(`${record.round}:${book.book}`, { record, book });
  return [...latest.values()].sort((a, b) => a.record.round - b.record.round || a.book.book.localeCompare(b.book.book));
}
