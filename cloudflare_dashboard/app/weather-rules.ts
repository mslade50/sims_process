/**
 * Pure parsing and table logic for the golfprice weather document (golfprice/weather/<event>/latest.json, schema golfprice.weather.v1, written per run
 * as inputs/weather.json by the golfprice weather layer). No runtime APIs: shared by WeatherEffectsView and tests/weather.test.mjs.
 * Everything optional is tolerated missing; anything that is not the document gives null (the view then shows its empty state).
 */
export const WEATHER_SCHEMA = "golfprice.weather.v1";

export type Venue = { name: string; lat: number | null; lon: number | null; class: string; wind_slope: number | null; wind_slope_source: string };
export type HourlyPoint = {
  round: number; hour: number; time: string;
  wind: number | null; wind_p10: number | null; wind_p90: number | null; gust: number | null;
  temp: number | null; temp_p10: number | null; temp_p90: number | null; rain: number | null; dir: number | null;
};
export type Wave = { round: number; wave: string; window: [number, number] | null; windowText: string; mean_wind: number | null; mean_temp: number | null; rel: number | null; rel_p10: number | null; rel_p90: number | null };
export type PlayerRound = { tee_time: string; start_hole: number | null; mean_off_strokes: number | null; mean_off_rel: number | null; tod_strokes: number | null; sd_mult: number | null; components: Record<string, number> };
export type PlayerRow = { dg_id: number; name: string; rounds: Record<number, PlayerRound>; total12: number | null; sdMax: number | null };
export type FieldRound = { round: number; common_strokes: number | null; cut_line_shift: number | null };
export type WeatherDoc = {
  schema: string; event_uid: string; as_of: string; synthetic: boolean; label: string;
  venue: Venue;
  forecast: { issued_at: string; models: string[]; hourly: HourlyPoint[] };
  waves: Wave[]; players: PlayerRow[]; field: FieldRound[]; notes: string[]; rounds: number[];
};

const isObject = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);
const numOrNull = (value: unknown): number | null => (typeof value === "number" && Number.isFinite(value) ? value : null);
const text = (value: unknown, fallback = ""): string => (typeof value === "string" ? value : fallback);
const list = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);

/** "2026-10-08T07:30", "07:30", "7:30:00" or a plain number of hours to decimal local hours; null when unreadable. */
export function toHour(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value !== "string") return null;
  const all = value.match(/(\d{1,2}):(\d{2})/g);
  if (!all) return null;
  const [h, m] = all[all.length - 1].split(":");
  return Number(h) + Number(m) / 60;
}

/** "07:05-09:40", "07:05 - 09:40", "07:05 to 09:40", [a, b] or {start, end}. */
export function parseWindow(value: unknown): [number, number] | null {
  let a: number | null = null;
  let b: number | null = null;
  if (typeof value === "string") {
    const parts = value.match(/\d{1,2}:\d{2}/g);
    if (parts && parts.length >= 2) { a = toHour(parts[0]); b = toHour(parts[1]); }
  } else if (Array.isArray(value) && value.length >= 2) {
    a = toHour(value[0]); b = toHour(value[1]);
  } else if (isObject(value)) {
    a = toHour(value.start ?? value.from); b = toHour(value.end ?? value.to);
  }
  return a !== null && b !== null ? (a <= b ? [a, b] : [b, a]) : null;
}

export function hhmm(hour: number | null): string {
  if (hour === null) return "-";
  const total = Math.round(hour * 60);
  return `${String(Math.floor(total / 60) % 24).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

function parseHourly(value: unknown): HourlyPoint | null {
  if (!isObject(value)) return null;
  const hour = toHour(value.time_local);
  const round = numOrNull(value.round);
  if (hour === null || round === null) return null;
  return {
    round, hour, time: text(value.time_local),
    wind: numOrNull(value.wind), wind_p10: numOrNull(value.wind_p10), wind_p90: numOrNull(value.wind_p90), gust: numOrNull(value.gust),
    temp: numOrNull(value.temp), temp_p10: numOrNull(value.temp_p10), temp_p90: numOrNull(value.temp_p90), rain: numOrNull(value.rain), dir: numOrNull(value.dir),
  };
}

function parseWave(value: unknown): Wave | null {
  if (!isObject(value)) return null;
  const round = numOrNull(value.round);
  if (round === null) return null;
  return {
    round, wave: typeof value.wave === "string" || typeof value.wave === "number" ? String(value.wave) : "", window: parseWindow(value.tee_window_local),
    windowText: typeof value.tee_window_local === "string" ? value.tee_window_local : "",
    mean_wind: numOrNull(value.mean_wind), mean_temp: numOrNull(value.mean_temp),
    rel: numOrNull(value.expected_strokes_rel), rel_p10: numOrNull(value.expected_strokes_rel_p10), rel_p90: numOrNull(value.expected_strokes_rel_p90),
  };
}

/** Collapse the long player records (one per player and round) into one row per player. */
export function groupPlayers(value: unknown): PlayerRow[] {
  const byId = new Map<number, PlayerRow>();
  for (const raw of list(value)) {
    if (!isObject(raw)) continue;
    const id = numOrNull(raw.dg_id);
    const round = numOrNull(raw.round);
    if (id === null || round === null) continue;
    const row = byId.get(id) ?? { dg_id: id, name: text(raw.name, String(id)), rounds: {}, total12: null, sdMax: null };
    const components: Record<string, number> = {};
    if (isObject(raw.components)) for (const [k, v] of Object.entries(raw.components)) { const n = numOrNull(v); if (n !== null) components[k] = n; }
    row.rounds[round] = {
      tee_time: text(raw.tee_time_local), start_hole: numOrNull(raw.start_hole),
      mean_off_strokes: numOrNull(raw.mean_off_strokes), mean_off_rel: numOrNull(raw.mean_off_rel), tod_strokes: numOrNull(raw.tod_strokes), sd_mult: numOrNull(raw.sd_mult), components,
    };
    byId.set(id, row);
  }
  for (const row of byId.values()) {
    const early = [1, 2].map((r) => row.rounds[r]).filter((r): r is PlayerRound => !!r);
    const effects = early.map((r) => r.mean_off_rel ?? r.mean_off_strokes).filter((v): v is number => v !== null);
    row.total12 = effects.length ? effects.reduce((a, b) => a + b, 0) : null;
    const sds = Object.values(row.rounds).map((r) => r.sd_mult).filter((v): v is number => v !== null);
    row.sdMax = sds.length ? sds.reduce((best, v) => (Math.abs(v - 1) > Math.abs(best - 1) ? v : best), sds[0]) : null;
  }
  return [...byId.values()];
}

export function parseWeather(value: unknown): WeatherDoc | null {
  if (!isObject(value) || value.schema !== WEATHER_SCHEMA) return null;
  const venue = isObject(value.venue) ? value.venue : {};
  const forecast = isObject(value.forecast) ? value.forecast : {};
  const hourly = list(forecast.hourly).map(parseHourly).filter((h): h is HourlyPoint => h !== null).sort((a, b) => a.round - b.round || a.hour - b.hour);
  const waves = list(value.waves).map(parseWave).filter((w): w is Wave => w !== null);
  const players = groupPlayers(value.players);
  const rawField = Array.isArray(value.field) ? value.field : isObject(value.field) ? [value.field] : [];
  const field: FieldRound[] = rawField.filter(isObject).map((f) => ({ round: numOrNull(f.round) ?? 0, common_strokes: numOrNull(f.common_strokes), cut_line_shift: numOrNull(f.cut_line_shift) }));
  const rounds = [...new Set([...hourly.map((h) => h.round), ...waves.map((w) => w.round), ...field.map((f) => f.round).filter((r) => r > 0)])].sort((a, b) => a - b);
  const label = text(value.label) || text(value.synthetic_label);
  return {
    schema: WEATHER_SCHEMA, event_uid: text(value.event_uid), as_of: text(value.as_of), synthetic: value.synthetic === true || label.toLowerCase().includes("synthetic"), label,
    venue: { name: text(venue.name), lat: numOrNull(venue.lat), lon: numOrNull(venue.lon), class: text(venue.class), wind_slope: numOrNull(venue.wind_slope), wind_slope_source: text(venue.wind_slope_source) },
    forecast: { issued_at: text(forecast.issued_at), models: list(forecast.models).filter((m): m is string => typeof m === "string"), hourly },
    waves, players, field, notes: list(value.notes).filter((n): n is string => typeof n === "string"), rounds,
  };
}

export type SortKey = "name" | "total12" | "r1" | "r2" | "r3" | "r4" | "sd";
export function sortKeyValue(row: PlayerRow, key: SortKey): number | string | null {
  switch (key) {
    case "name": return row.name;
    case "total12": return row.total12;
    case "sd": return row.sdMax;
    default: {
      const round = row.rounds[Number(key.slice(1))];
      return round ? (round.mean_off_rel ?? round.mean_off_strokes) : null;
    }
  }
}

/** Sort with missing values always last, whichever the direction. Ties by name. */
export function sortPlayers(rows: PlayerRow[], key: SortKey, dir: "asc" | "desc"): PlayerRow[] {
  const sign = dir === "asc" ? 1 : -1;
  return [...rows].sort((a, b) => {
    const va = sortKeyValue(a, key);
    const vb = sortKeyValue(b, key);
    if (va === null && vb === null) return a.name.localeCompare(b.name);
    if (va === null) return 1;
    if (vb === null) return -1;
    const c = typeof va === "string" || typeof vb === "string" ? String(va).localeCompare(String(vb)) : va - vb;
    return c !== 0 ? sign * c : a.name.localeCompare(b.name);
  });
}

const norm = (s: string) => s.toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "").replace(/[^a-z0-9 ]/g, " ").replace(/\s+/g, " ").trim();
/** Name search: "Last, First" and "First Last" both match; a dg_id matches too. */
export function filterPlayers(rows: PlayerRow[], query: string): PlayerRow[] {
  const terms = norm(query).split(" ").filter(Boolean);
  if (!terms.length) return rows;
  return rows.filter((row) => {
    const hay = `${norm(row.name)} ${row.dg_id}`;
    return terms.every((t) => hay.includes(t));
  });
}

export const signedFixed = (value: number | null, digits = 2): string => (value === null ? "-" : `${value > 0 ? "+" : ""}${value.toFixed(digits)}`);

export function forecastAgeHours(issuedAt: string, now: number): number | null {
  const t = Date.parse(issuedAt);
  return Number.isFinite(t) ? Math.max(0, (now - t) / 3_600_000) : null;
}

export const COMPONENT_LABELS: Record<string, string> = { wind: "Wind", gust: "Gusts", temp: "Temperature", rain: "Rain", tod: "Time of day" };

/** One sentence per effect column, for hover text. */
export const COMPONENT_HINTS: Record<string, string> = {
  wind: "Strokes added (or saved) by the average wind during this player's round.",
  gust: "Strokes added (or saved) by gusts.",
  temp: "Strokes added (or saved) by the temperature.",
  rain: "Strokes added (or saved) by rain.",
};

/** Weather models by what people call them. Unknown names are shown as given, upper-cased. */
const MODEL_NAMES: Record<string, string> = { gfs: "GFS (US)", ifs: "ECMWF (European)", icon: "ICON (German)", gem: "GEM (Canadian)", aifs: "ECMWF AI", ukmo: "UK Met Office", arpege: "Arpege (French)" };
export const modelName = (model: string): string => MODEL_NAMES[model.toLowerCase()] ?? model.toUpperCase();

/** How the course's wind sensitivity was estimated, in plain words. */
export function windSourceText(source: string): string {
  const s = source.toLowerCase();
  if (!s) return "";
  if (s.includes("own history")) return "Estimated from this course's own history";
  if (s.includes("class")) return "Estimated from similar courses";
  if (s.includes("default")) return "A default value (no course history yet)";
  return source;
}

/** The moment inside a run name such as "live_R1_20261008T122100Z", as an ISO string; null when the name carries none. */
export function runMoment(run: string | null | undefined): string | null {
  const m = /(\d{4})(\d\d)(\d\d)T(\d\d)(\d\d)(\d\d)Z/.exec(run ?? "");
  return m ? `${m[1]}-${m[2]}-${m[3]}T${m[4]}:${m[5]}:${m[6]}Z` : null;
}

const etWords = (iso: string | null | undefined): string => {
  const ms = Date.parse(iso ?? "");
  return Number.isFinite(ms) ? `${new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }).format(ms)} ET` : "an unknown time";
};

/**
 * D6: the weather document's run against the price checkpoint (same guard shape as explain_run in ExplainViews).
 * Mismatch when the published weather run differs from the priced run, or when the loaded document is not the one the index points to.
 */
export type WeatherCheckpointIndex = { weather_run?: string | null; weather_as_of?: string | null; explain_run?: string | null };
export type WeatherCheckpoint = { weatherRun: string | null; priceRun: string | null; weatherAsOf: string | null; state: "match" | "mismatch" | "unknown"; message: string };
const sameInstant = (a: string | null | undefined, b: string | null | undefined): boolean => !!a && !!b && Number.isFinite(Date.parse(a)) && Date.parse(a) === Date.parse(b);
export function weatherCheckpoint(event: WeatherCheckpointIndex, docAsOf: string | null | undefined): WeatherCheckpoint {
  const weatherRun = event.weather_run ?? null;
  const priceRun = event.explain_run ?? null;
  const weatherAsOf = event.weather_as_of ?? docAsOf ?? null;
  if (!weatherRun || !priceRun) return { weatherRun, priceRun, weatherAsOf, state: "unknown", message: "The site cannot tell which run this weather belongs to, so it cannot check it against the current prices." };
  if (weatherRun !== priceRun) return { weatherRun, priceRun, weatherAsOf, state: "mismatch", message: `This weather is from the run at ${etWords(runMoment(weatherRun) ?? weatherAsOf)}, but the latest prices are from the run at ${etWords(runMoment(priceRun))}. The weather effects shown may not be the ones in the current prices.` };
  if (event.weather_as_of && docAsOf && !sameInstant(event.weather_as_of, docAsOf)) return { weatherRun, priceRun, weatherAsOf, state: "mismatch", message: `The loaded weather file is from ${etWords(docAsOf)}, but this run should have ${etWords(event.weather_as_of)}. Reload before trusting it.` };
  return { weatherRun, priceRun, weatherAsOf, state: "match", message: "Weather and prices come from the same run." };
}
