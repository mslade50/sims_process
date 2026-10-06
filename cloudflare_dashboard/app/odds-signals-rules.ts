/**
 * Pure parsing for the published odds-move signals (golfprice/odds_signals/latest.json, schema golfprice.odds_signals.v1, written by
 * golfprice/ops/oddsmove.py). No runtime APIs: shared by the Run page panel and tests/odds-signals.test.mjs.
 */
export const ODDS_SIGNALS_KEY = "golfprice/odds_signals/latest.json";
export const ODDS_STALE_MS = 3 * 3_600_000;

export type OddsSignal = { family: string; dg_id: number; player: string | null; book: string; dec: number; p_fair: number; ev: number; status: string; source: string | null };
export type OddsEvent = {
  event_uid: string;
  name: string;
  base_run: string;
  base_as_of: string;
  fair_source: string;
  n_signals: number;
  n_live: number;
  signals: OddsSignal[];
  new: OddsSignal[];
  moved: Array<OddsSignal & { ev_ref: number }>;
  dropped: string[];
  alert: { sent: boolean; note: string };
};
export type OddsSignalsDoc = { schema: string; generated_at: string; moment: string; note: string; events: OddsEvent[] };

const isObject = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);
const num = (value: unknown, fallback = 0) => (typeof value === "number" && Number.isFinite(value) ? value : fallback);
const text = (value: unknown, fallback = "") => (typeof value === "string" ? value : fallback);

function parseSignal(value: unknown): OddsSignal | null {
  if (!isObject(value)) return null;
  return { family: text(value.family), dg_id: num(value.dg_id), player: typeof value.player === "string" ? value.player : null, book: text(value.book), dec: num(value.dec), p_fair: num(value.p_fair), ev: num(value.ev), status: text(value.status), source: typeof value.source === "string" ? value.source : null };
}

/** Tolerant parse: anything that is not the published document gives null (the panel then shows its empty state). */
export function parseOddsSignals(value: unknown): OddsSignalsDoc | null {
  if (!isObject(value) || value.schema !== "golfprice.odds_signals.v1" || !Array.isArray(value.events)) return null;
  const events: OddsEvent[] = [];
  for (const raw of value.events) {
    if (!isObject(raw)) continue;
    const list = (key: string) => (Array.isArray(raw[key]) ? (raw[key] as unknown[]).map(parseSignal).filter((s): s is OddsSignal => s !== null) : []);
    const alert = isObject(raw.alert) ? raw.alert : {};
    events.push({
      event_uid: text(raw.event_uid),
      name: text(raw.name, text(raw.event_uid)),
      base_run: text(raw.base_run),
      base_as_of: text(raw.base_as_of),
      fair_source: text(raw.fair_source),
      n_signals: num(raw.n_signals),
      n_live: num(raw.n_live),
      signals: list("signals"),
      new: list("new"),
      moved: list("moved").map((s, i) => ({ ...s, ev_ref: num(((raw.moved as Array<Record<string, unknown>>)[i] ?? {}).ev_ref) })),
      dropped: Array.isArray(raw.dropped) ? raw.dropped.filter((d): d is string => typeof d === "string") : [],
      alert: { sent: alert.sent === true, note: text(alert.note) },
    });
  }
  return { schema: "golfprice.odds_signals.v1", generated_at: text(value.generated_at), moment: text(value.moment), note: text(value.note), events };
}

export function oddsStale(generatedAt: string, now: number): boolean {
  const at = Date.parse(generatedAt);
  return !Number.isFinite(at) || now - at > ODDS_STALE_MS;
}

export const pct = (value: number, digits = 1) => `${value >= 0 ? "+" : ""}${(value * 100).toFixed(digits)}%`;
