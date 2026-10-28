/**
 * Data-freshness rules for the dashboard (owner intent 2026-10-28: every source is as current as possible and visibly so).
 * Pure TypeScript, no imports, so node's strip-types test loader and the Worker can share it.
 *
 * Everything here reads published fields. Nothing hard-codes a season, a generation, a hash or a date: a field that is absent gives
 * `null` (the label is simply not drawn), never a guess.
 */

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const DAY_MS = 86_400_000;

/** "2026-10-22" as-is; a full timestamp becomes its US Eastern calendar date; anything else null. A bare date never goes through Date (no zone shift). */
export function calendarDate(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const text = value.trim();
  const bare = /^(\d{4}-\d{2}-\d{2})(?: \(.*\))?$/.exec(text);
  if (bare) return bare[1];
  // "2026-10-07 21:51:30" without a zone is read as UTC, like the publisher's run stamps.
  const zoned = /^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}/.test(text) && !/(Z|UTC|[+-]\d{2}:?\d{2})$/.test(text) ? `${text.replace(" ", "T")}Z` : text.replace(" UTC", "Z").replace(" ", "T");
  const ms = Date.parse(zoned);
  if (!Number.isFinite(ms)) return null;
  const parts = new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit" }).format(ms);
  return /^\d{4}-\d{2}-\d{2}$/.test(parts) ? parts : null;
}

/** "2026-10-22" -> "Oct 22, 2026"; null when unreadable. */
export function plainDate(value: unknown): string | null {
  const d = calendarDate(value);
  if (!d) return null;
  const [y, m, day] = d.split("-").map(Number);
  return `${MONTHS[m - 1]} ${day}, ${y}`;
}

/** The standing "Data through Oct 22, 2026" line; null when the published date is absent or unreadable. */
export function dataThrough(value: unknown, prefix = "Data through"): string | null {
  const text = plainDate(value);
  return text ? `${prefix} ${text}` : null;
}

/** Whole calendar days from `date` to `now` (ET calendar), null when unreadable. Negative when the date is ahead of now. */
export function daysSince(value: unknown, now: number): number | null {
  const d = calendarDate(value);
  const today = calendarDate(new Date(now).toISOString());
  if (!d || !today) return null;
  return Math.round((Date.parse(`${today}T00:00:00Z`) - Date.parse(`${d}T00:00:00Z`)) / DAY_MS);
}

/** "Oct 22, 3:05 PM ET" for a UTC instant. */
export function etStamp(iso: unknown): string {
  const ms = typeof iso === "string" ? Date.parse(iso) : NaN;
  return Number.isFinite(ms) ? `${new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }).format(ms)} ET` : "an unknown time";
}

const obj = (x: unknown): Record<string, unknown> | null => (x && typeof x === "object" && !Array.isArray(x) ? (x as Record<string, unknown>) : null);
const str = (x: unknown): string | null => (typeof x === "string" && x ? x : null);

/* ------------------------------------------------------------------ player pages: shot data */
export type ShotFreshness = { through: string | null; generation: string | null };
/** catalog.deep_coverage.shot_data.{newest_shot_event_date, export_generation}. Both optional until the golfprice release publishes them. */
export function shotFreshness(catalog: unknown): ShotFreshness {
  const shot = obj(obj(obj(catalog)?.deep_coverage)?.shot_data);
  return { through: calendarDate(shot?.newest_shot_event_date), generation: str(shot?.export_generation) };
}
export function shotFreshnessLine(f: ShotFreshness): string | null {
  const through = dataThrough(f.through, "Shot data through");
  if (!through) return f.generation ? `Shot data export ${f.generation}` : null;
  return f.generation ? `${through} (export ${f.generation.slice(0, 12)})` : through;
}

/* ------------------------------------------------------------------ course history */
/** The document's published source_through_date; else the newest dated round it contains (derived from published rows, never from the clock). */
export function historyThrough(sourceThrough: unknown, roundDates: Array<string | null | undefined>): string | null {
  const published = calendarDate(sourceThrough);
  if (published) return published;
  const dates = roundDates.map(calendarDate).filter((d): d is string => !!d).sort();
  return dates.length ? dates[dates.length - 1] : null;
}

/* ------------------------------------------------------------------ weather */
const STAMP = /(\d{8}T\d{6}Z)/;
const runStamp = (run: unknown): string | null => STAMP.exec(typeof run === "string" ? run : "")?.[1] ?? null;

/** Immutable per-run copy published beside latest.json: golfprice/weather/<event_safe>/<kind>_<run>.json. Null when the pieces are unusable. */
export function weatherRunKey(weatherKey: string | null | undefined, kind: string | null | undefined, run: string | null | undefined): string | null {
  if (!weatherKey || !kind || !run || /[\\]|\.\./.test(`${kind}${run}`) || /\//.test(`${kind}${run}`)) return null;
  const slash = weatherKey.lastIndexOf("/");
  if (slash < 0) return null;
  return `${weatherKey.slice(0, slash + 1)}${kind}_${run}.json`;
}

export type BriefCurrency = { state: "current" | "behind" | "unknown"; message: string };
/** The brief is generated from a weather run; when the weather document is from a newer run than the brief, the brief text is out of date. */
export function briefCurrency(event: { weather_run?: string | null; weather_as_of?: string | null; weather_brief_run?: string | null; weather_brief_as_of?: string | null }): BriefCurrency {
  const wRun = event.weather_run ?? null, bRun = event.weather_brief_run ?? null;
  if (!wRun) return { state: "unknown", message: "" };
  if (!bRun) return { state: "behind", message: "The weather summary has not been published for the latest weather run, so the text below may describe an older forecast." };
  const ws = runStamp(wRun), bs = runStamp(bRun);
  let behind: boolean | null = null;
  if (ws && bs) behind = bs < ws;
  else if (wRun !== bRun) behind = true;
  else behind = false;
  if (!behind) return { state: "current", message: "" };
  const wMoment = event.weather_as_of ?? (ws ? `${ws.slice(0, 4)}-${ws.slice(4, 6)}-${ws.slice(6, 8)}T${ws.slice(9, 11)}:${ws.slice(11, 13)}:${ws.slice(13, 15)}Z` : null);
  const bMoment = event.weather_brief_as_of ?? (bs ? `${bs.slice(0, 4)}-${bs.slice(4, 6)}-${bs.slice(6, 8)}T${bs.slice(9, 11)}:${bs.slice(11, 13)}:${bs.slice(13, 15)}Z` : null);
  return { state: "behind", message: `This summary was written from the weather run at ${etStamp(bMoment)}, but the weather tables are from the newer run at ${etStamp(wMoment)}. The wording may not match the latest forecast.` };
}

/* ------------------------------------------------------------------ scorecard */
export type EventLike = { event_uid: string; name?: string | null; tour?: string | null; date_start?: string | null; date_end?: string | null };
export type ScorecardLag = { state: "current" | "behind" | "unknown"; newestSettled: string | null; newestCompleted: EventLike | null; message: string };
/** Days after an event ends that the Monday settle may still be pending before the scorecard is called behind. */
export const SETTLE_GRACE_DAYS = 2;
/**
 * Newest settled scorecard event against the newest completed event of the same tour in the published index (not generated_at: a scorecard
 * regenerated today can still lack last week's event). Completed = date_end strictly before today's Eastern date.
 */
export function scorecardLag(settled: Array<{ event_uid: string; event_start?: string | null }>, indexEvents: EventLike[] | null | undefined, now: number, tour = "pga"): ScorecardLag {
  const newestSettled = [...settled].map((e) => calendarDate(e.event_start)).filter((d): d is string => !!d).sort().pop() ?? null;
  if (!indexEvents) return { state: "unknown", newestSettled, newestCompleted: null, message: "" };
  const today = calendarDate(new Date(now).toISOString());
  const done = indexEvents.filter((e) => (e.tour ?? "").toLowerCase() === tour && e.date_end && calendarDate(e.date_end) && today && calendarDate(e.date_end)! < today)
    .sort((a, b) => calendarDate(b.date_end)!.localeCompare(calendarDate(a.date_end)!));
  const newest = done[0] ?? null;
  if (!newest) return { state: "unknown", newestSettled, newestCompleted: null, message: "" };
  const have = new Set(settled.map((e) => e.event_uid));
  if (have.has(newest.event_uid)) return { state: "current", newestSettled, newestCompleted: newest, message: "" };
  const ago = daysSince(newest.date_end, now);
  if (ago !== null && ago <= SETTLE_GRACE_DAYS) return { state: "current", newestSettled, newestCompleted: newest, message: "" };
  return {
    state: "behind", newestSettled, newestCompleted: newest,
    message: `The newest graded event here is ${newestSettled ? `the one starting ${plainDate(newestSettled)}` : "unknown"}, but ${newest.name ?? newest.event_uid} finished ${plainDate(newest.date_end)} and is not graded yet. The settle has not published.`,
  };
}

/* ------------------------------------------------------------------ glossary */
/** A published numeric/text field under its season-neutral key, else the same key suffixed with the season (`ridge_lambda`, then `ridge_lambda_2026`). Never a fixed season. */
export function seasonField<T = unknown>(source: Record<string, unknown> | null | undefined, base: string, season?: number | string | null): T | undefined {
  if (!source) return undefined;
  if (source[base] !== undefined && source[base] !== null) return source[base] as T;
  if (season !== undefined && season !== null && source[`${base}_${season}`] !== undefined) return source[`${base}_${season}`] as T;
  const suffixed = Object.keys(source).filter((k) => k.startsWith(`${base}_`) && /^\d{4}$/.test(k.slice(base.length + 1))).sort();
  return suffixed.length ? (source[suffixed[suffixed.length - 1]] as T) : undefined;
}
