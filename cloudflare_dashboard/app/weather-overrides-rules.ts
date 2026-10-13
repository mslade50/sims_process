/**
 * Owner weather edits (hour-by-hour overrides of the published forecast): one validation, shared by the Worker (worker/weather-overrides-api.ts),
 * the "Weather edits" page and tests/weather-overrides.test.mjs. Pure TypeScript, no runtime APIs.
 *
 * Contract with golfprice (do not change without changing the Python side): R2 object overrides/weather/<event_safe>.json,
 *   {schema: "golfprice.weather_override.v1", event_uid, updated_at, updated_by, note, hours: [{time_local, wind, gust, temp, rain, dir}]}
 * A null or absent field keeps the forecast value; an hour with every field null is dropped; an empty hours list means no edits.
 */
export const WEATHER_OVERRIDE_SCHEMA = "golfprice.weather_override.v1";
export const WEATHER_OVERRIDE_PREFIX = "overrides/weather/";
export const WEATHER_OVERRIDE_HISTORY_PREFIX = "overrides/weather/history/";
export const EVENT_SAFE_RE = /^[a-z]+_[0-9A-Za-z]+_\d{4}-\d{2}-\d{2}$/;
export const MAX_HOURS = 400;
export const MAX_NOTE = 500;

export const FIELDS = ["wind", "gust", "dir", "temp", "rain"] as const;
export type WeatherField = (typeof FIELDS)[number];
export type OverrideHour = { time_local: string; wind: number | null; gust: number | null; temp: number | null; rain: number | null; dir: number | null };
export type WeatherOverrideDoc = { schema: string; event_uid: string; updated_at: string; updated_by: string; note: string; hours: OverrideHour[] };

export const BOUNDS: Record<WeatherField, { lo: number; hi: number; unit: string; label: string }> = {
  wind: { lo: 0, hi: 60, unit: "mph", label: "Wind" },
  gust: { lo: 0, hi: 90, unit: "mph", label: "Gusts" },
  temp: { lo: -10, hi: 120, unit: "°F", label: "Temperature" },
  rain: { lo: 0, hi: 50, unit: "mm per hour", label: "Rain" },
  dir: { lo: 0, hi: 360, unit: "degrees the wind comes from", label: "Direction" },
};

export const eventSafe = (eventUid: string): string => eventUid.replace(/:/g, "_");

const HOUR_RE = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):00$/;
export function validTimeLocal(value: unknown): value is string {
  if (typeof value !== "string") return false;
  const m = HOUR_RE.exec(value);
  if (!m) return false;
  const [y, mo, d, h] = [Number(m[1]), Number(m[2]), Number(m[3]), Number(m[4])];
  if (h > 23 || mo < 1 || mo > 12 || d < 1) return false;
  const date = new Date(Date.UTC(y, mo - 1, d));
  return date.getUTCFullYear() === y && date.getUTCMonth() === mo - 1 && date.getUTCDate() === d;
}

/** Keep one decimal; 360 is stored as 0. */
export const roundTo1 = (value: number): number => Math.round(value * 10) / 10;

const isBlank = (value: unknown) => value === null || value === undefined || value === "";

/** One hour's fields (numbers, numeric strings from the inputs, or blank). Returns the plain-English problem for each bad field. */
export function checkHourFields(raw: Partial<Record<WeatherField, unknown>>): Partial<Record<WeatherField, string>> {
  const problems: Partial<Record<WeatherField, string>> = {};
  const nums: Partial<Record<WeatherField, number>> = {};
  for (const field of FIELDS) {
    const value = raw[field];
    if (isBlank(value)) continue;
    const n = typeof value === "number" ? value : typeof value === "string" && value.trim() !== "" ? Number(value) : NaN;
    const b = BOUNDS[field];
    if (!Number.isFinite(n)) problems[field] = `${b.label} must be a number.`;
    else if (n < b.lo || n > b.hi) problems[field] = `${b.label} must be between ${b.lo} and ${b.hi}${field === "dir" ? "" : ` ${b.unit}`}.`;
    else nums[field] = n;
  }
  if (nums.wind !== undefined && nums.gust !== undefined && nums.gust < nums.wind && !problems.gust) problems.gust = "Gusts cannot be lower than the wind.";
  return problems;
}

export type ValidatedOverride = { problems: string[]; hours: OverrideHour[]; note: string };

/** Validate and normalize a save request body ({hours, note}). Hours with nothing set are dropped; the result is sorted by time. */
export function validateWeatherOverride(body: unknown): ValidatedOverride {
  const problems: string[] = [];
  const empty: ValidatedOverride = { problems, hours: [], note: "" };
  if (!body || typeof body !== "object" || Array.isArray(body)) return { ...empty, problems: ["The request body must be an object with hours and a note."] };
  const raw = body as Record<string, unknown>;
  const rawHours = raw.hours === undefined || raw.hours === null ? [] : raw.hours;
  if (!Array.isArray(rawHours)) return { ...empty, problems: ["hours must be a list."] };
  if (rawHours.length > MAX_HOURS) problems.push(`Too many hours (${rawHours.length}); the limit is ${MAX_HOURS}.`);
  let note = "";
  if (raw.note !== undefined && raw.note !== null) {
    if (typeof raw.note !== "string") problems.push("The note must be text.");
    else if (raw.note.length > MAX_NOTE) problems.push(`The note is too long (${raw.note.length} characters; the limit is ${MAX_NOTE}).`);
    else note = raw.note.trim();
  }
  const seen = new Set<string>();
  const hours: OverrideHour[] = [];
  rawHours.slice(0, MAX_HOURS).forEach((item, i) => {
    const where = `Hour ${i + 1}`;
    if (!item || typeof item !== "object" || Array.isArray(item)) { problems.push(`${where} must be an object.`); return; }
    const h = item as Record<string, unknown>;
    if (!validTimeLocal(h.time_local)) { problems.push(`${where}: time_local must look like 2026-10-15T08:00 (a whole hour).`); return; }
    if (seen.has(h.time_local)) { problems.push(`${where}: ${h.time_local} appears more than once.`); return; }
    seen.add(h.time_local);
    const numeric: Partial<Record<WeatherField, unknown>> = {};
    for (const field of FIELDS) {
      if (!isBlank(h[field]) && typeof h[field] !== "number") problems.push(`${h.time_local}: ${BOUNDS[field].label} must be a number.`);
      else numeric[field] = h[field];
    }
    const fieldProblems = checkHourFields(numeric);
    for (const message of Object.values(fieldProblems)) problems.push(`${h.time_local}: ${message}`);
    const get = (field: WeatherField): number | null => {
      const value = h[field];
      if (typeof value !== "number" || !Number.isFinite(value)) return null;
      const rounded = roundTo1(value);
      return field === "dir" && rounded === 360 ? 0 : rounded;
    };
    const entry: OverrideHour = { time_local: h.time_local, wind: get("wind"), gust: get("gust"), temp: get("temp"), rain: get("rain"), dir: get("dir") };
    if (FIELDS.some((f) => entry[f] !== null)) hours.push(entry);
  });
  hours.sort((a, b) => a.time_local.localeCompare(b.time_local));
  return { problems, hours, note };
}

/** The history key for a save: overrides/weather/history/<YYYYMMDDTHHMMSSZ>-<event_safe>.json */
export function historyKey(now: number, safe: string): string {
  return `${WEATHER_OVERRIDE_HISTORY_PREFIX}${new Date(now).toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z")}-${safe}.json`;
}

/* ------------------------------------------------------------------ choosing the reprice job */
export type RepriceChoice = { type: "tuesday" | "wednesday" | "after_round"; params: { supersede: true }; reason: string };

/** Today's date in New York as YYYY-MM-DD. */
export function etDate(now: number): string {
  return new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit" }).format(new Date(now));
}

/**
 * Before the event starts (today in New York is before date_start): Wednesday reprice when rounds 1 and 2 both have tee-sheet coverage of at least 0.98,
 * else Tuesday refresh. Once the event has started: after_round. An unknown start date counts as not started.
 */
export function chooseRepriceJob(teeCoverage: Record<number, number | null | undefined>, dateStart: string | undefined | null, now: number): RepriceChoice {
  const started = typeof dateStart === "string" && /^\d{4}-\d{2}-\d{2}/.test(dateStart) && etDate(now) >= dateStart.slice(0, 10);
  if (started) return { type: "after_round", params: { supersede: true }, reason: "The event has started, so this reprices the live state." };
  const ready = [1, 2].every((r) => (teeCoverage[r] ?? 0) >= 0.98);
  return ready
    ? { type: "wednesday", params: { supersede: true }, reason: "Tee times are out for rounds 1 and 2, so this runs the Wednesday reprice." }
    : { type: "tuesday", params: { supersede: true }, reason: "Tee times are not complete yet, so this runs the Tuesday refresh." };
}

/* ------------------------------------------------------------------ display helpers */
const COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
export const compassLetters = (deg: number): string => COMPASS[Math.round((((deg % 360) + 360) % 360) / 45) % 8];

/* ------------------------------------------------------------------ v2: pre-tournament and in-event sections */
export const WEATHER_OVERRIDE_SCHEMA_V2 = "golfprice.weather_override.v2";
export type Phase = "pre" | "live";
export type Section = { hours: OverrideHour[]; note: string; updated_at: string; updated_by: string };
export type LiveSection = Section & { after_round: number };
export type WeatherOverrideDocV2 = { schema: string; event_uid: string; pre: Section | null; live: LiveSection | null };

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
function readSection(raw: unknown): Section | null {
  if (!isObj(raw)) return null;
  const checked = validateWeatherOverride({ hours: raw.hours, note: typeof raw.note === "string" ? raw.note.slice(0, MAX_NOTE) : "" });
  return { hours: checked.hours, note: checked.note, updated_at: typeof raw.updated_at === "string" ? raw.updated_at : "", updated_by: typeof raw.updated_by === "string" ? raw.updated_by : "" };
}

/** Read a stored object of either version as v2. A v1 object (top-level hours) becomes {pre: {...}}; anything unreadable gives empty sections. */
export function normalizeStored(raw: unknown, eventUid = ""): WeatherOverrideDocV2 {
  const out: WeatherOverrideDocV2 = { schema: WEATHER_OVERRIDE_SCHEMA_V2, event_uid: isObj(raw) && typeof raw.event_uid === "string" ? raw.event_uid : eventUid, pre: null, live: null };
  if (!isObj(raw)) return out;
  if (Array.isArray(raw.hours)) { out.pre = readSection(raw); return out; }   // v1
  out.pre = readSection(raw.pre);
  const live = readSection(raw.live);
  const after = isObj(raw.live) ? raw.live.after_round : null;
  if (live && typeof after === "number" && Number.isInteger(after) && after >= 1 && after <= 9) out.live = { ...live, after_round: after };
  return out;
}

/** Validate a v2 PUT body's phase and after_round (the hours and note go through validateWeatherOverride). */
export function validatePhase(body: unknown): { problems: string[]; phase?: Phase; after_round?: number } {
  const raw = isObj(body) ? body : {};
  if (raw.phase !== "pre" && raw.phase !== "live") return { problems: ['phase must be "pre" or "live".'] };
  if (raw.phase === "pre") return { problems: [], phase: "pre" };
  const a = raw.after_round;
  if (typeof a !== "number" || !Number.isInteger(a) || a < 1 || a > 9) return { problems: ["after_round must be a whole number from 1 to 9 for in-event edits."] };
  return { problems: [], phase: "live", after_round: a };
}

/** Replace one phase of the stored doc, keeping the other untouched. */
export function mergeSection(current: WeatherOverrideDocV2, phase: Phase, section: Section, afterRound: number | undefined, eventUid: string): WeatherOverrideDocV2 {
  return { schema: WEATHER_OVERRIDE_SCHEMA_V2, event_uid: eventUid, pre: phase === "pre" ? section : current.pre, live: phase === "live" ? { ...section, after_round: afterRound as number } : current.live };
}

export type RunLike = { kind?: string | null; after_round?: number | null; validated?: boolean | null };
/** Completed rounds so far = the highest after_round among the event's live runs (0 when none). */
export function completedRounds(runs: RunLike[] | null | undefined): number {
  return Math.max(0, ...(runs ?? []).filter((r) => r && r.kind === "live" && r.validated !== false && typeof r.after_round === "number").map((r) => r.after_round as number));
}
/** The event has started: today in New York is on or after date_start, or a live run exists. */
export function eventStarted(dateStart: string | undefined | null, runs: RunLike[] | null | undefined, now: number): boolean {
  return (runs ?? []).some((r) => r && r.kind === "live") || (typeof dateStart === "string" && /^\d{4}-\d{2}-\d{2}/.test(dateStart) && etDate(now) >= dateStart.slice(0, 10));
}

/** What the in-event tab shows for the saved live section: its edits apply only while its after_round equals the current completed-round count. */
export function liveStatus(saved: LiveSection | null, completed: number): { open: boolean; expiredFrom: number | null; prefill: LiveSection | null } {
  if (completed < 1) return { open: false, expiredFrom: null, prefill: null };
  if (saved && saved.after_round === completed) return { open: true, expiredFrom: null, prefill: saved };
  return { open: true, expiredFrom: saved && saved.hours.length ? saved.after_round : null, prefill: null };
}

/** Reprice for the selected mode: before the event = week-run choice (null once the event has started: pre edits no longer apply); during = after_round. */
export function repriceForMode(mode: Phase, teeCoverage: Record<number, number | null | undefined>, dateStart: string | undefined | null, runs: RunLike[] | null | undefined, now: number): RepriceChoice | null {
  if (mode === "live") return { type: "after_round", params: { supersede: true }, reason: "This reprices the live state with your in-event edits." };
  if (eventStarted(dateStart, runs, now)) return null;
  return chooseRepriceJob(teeCoverage, dateStart, now);
}

/** Weather page: which edits the published forecast carries, as a short phrase. */
export function ownerEditKind(summary: { phase?: string; after_round?: number | null; status?: string; count: number } | null): string {
  if (!summary) return "";
  return summary.phase === "live" ? `in-event edits made after round ${summary.after_round ?? "?"}` : "pre-tournament edits";
}
