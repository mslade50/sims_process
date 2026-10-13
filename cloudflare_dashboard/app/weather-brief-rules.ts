/**
 * Pure logic for the weather brief (golfprice.weather_brief.v1, published as weather_brief_key in the golfprice index).
 * The brief is written for a reader, not an analyst: every number the page shows goes through the wording helpers here, so a
 * negative number is never shown as a minus sign. Sign convention of the file: NEGATIVE = easier scoring / helps the player.
 * No DOM or React: WeatherBriefView and tests/weather-brief.test.mjs import this directly.
 */
export const BRIEF_SCHEMA_PREFIX = "golfprice.weather_brief";

export type Slot = { window: string; wind: number | null; gust: number | null; temp: number | null; rain_prob: number | null };
export type RoundForecast = { AM: Slot | null; PM: Slot | null; day: Slot | null };
export type BriefRound = {
  round: number; date_local: string; status: string; tee_coverage: number | null; forecast: RoundForecast;
  field: number | null; wave: number | null; spread: number | null; plain: string;
};
export type PlayerRound = { round: number; wave: string | null; tee: string; strokes: number | null; components: Record<string, number> };
export type Probs = { win: number | null; top_5: number | null; top_10: number | null; top_20: number | null };
export type BriefPlayer = { dg_id: number; name: string; skill_rank: number | null; total: number | null; per_round: PlayerRound[]; prob: Probs; plain: string };
export type Scenario = {
  id: string; label: string; wind: number; temp: number; rain: number; pm: number;
  rounds: Array<{ round: number; field: number | null; wave: number | null; spread: number | null }>;
  movers: Map<number, number>;
};
export type HistoryPoint = { issued_at: string; rounds: Array<{ round: number; field: number | null; wave: number | null }> };
export type Brief = {
  schema: string; event_uid: string; event_name: string; as_of: string; model_version: string;
  forecast: { issued_at: string; age_hours: number | null; source: string; n_scenarios: number | null };
  status: { level: string; plain: string };
  headline: string[];
  course: { name: string; cls: string; slope: number | null; tourSlope: number | null; multiplier: number | null; basis: string; plain: string };
  rounds: BriefRound[]; players: BriefPlayer[];
  whatIf: { basis: string; scenarios: Scenario[] } | null;
  history: HistoryPoint[];
  glossary: Record<string, string>;
};

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const str = (v: unknown, fallback = ""): string => (typeof v === "string" ? v : fallback);
const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);

function parseSlot(v: unknown): Slot | null {
  if (!isObject(v)) return null;
  const slot = { window: str(v.window), wind: num(v.wind), gust: num(v.gust), temp: num(v.temp), rain_prob: num(v.rain_prob) };
  return slot.wind === null && slot.gust === null && slot.temp === null && slot.rain_prob === null ? null : slot;
}

function parseRound(v: unknown): BriefRound | null {
  if (!isObject(v)) return null;
  const round = num(v.round);
  if (round === null) return null;
  const f = isObject(v.forecast) ? v.forecast : {};
  return {
    round, date_local: str(v.date_local), status: str(v.status), tee_coverage: num(v.tee_coverage),
    forecast: { AM: parseSlot(f.AM), PM: parseSlot(f.PM), day: parseSlot(f.day) },
    field: num(v.field_effect_strokes), wave: num(v.wave_split_strokes), spread: num(v.spread_multiplier), plain: str(v.plain),
  };
}

function parsePlayer(v: unknown): BriefPlayer | null {
  if (!isObject(v)) return null;
  const id = num(v.dg_id);
  if (id === null) return null;
  const pc = isObject(v.prob_change) ? v.prob_change : {};
  const per_round: PlayerRound[] = arr(v.per_round).filter(isObject).map((r) => {
    const components: Record<string, number> = {};
    if (isObject(r.components)) for (const [k, x] of Object.entries(r.components)) { const n = num(x); if (n !== null) components[k] = n; }
    return { round: num(r.round) ?? 0, wave: typeof r.wave === "string" ? r.wave : null, tee: str(r.tee_time_local), strokes: num(r.strokes), components };
  }).filter((r) => r.round > 0);
  return {
    dg_id: id, name: str(v.name, String(id)), skill_rank: num(v.skill_rank), total: num(v.total_strokes), per_round,
    prob: { win: num(pc.win), top_5: num(pc.top_5), top_10: num(pc.top_10), top_20: num(pc.top_20) }, plain: str(v.plain),
  };
}

function parseScenario(v: unknown): Scenario | null {
  if (!isObject(v)) return null;
  const rounds = arr(v.rounds).filter(isObject).map((r) => ({ round: num(r.round) ?? 0, field: num(r.field_effect_strokes), wave: num(r.wave_split_strokes), spread: num(r.spread_multiplier) })).filter((r) => r.round > 0);
  const movers = new Map<number, number>();
  // Every player's total when the brief carries it (player_totals, keyed by dg_id); otherwise only the biggest movers, the rest scaled.
  if (isObject(v.player_totals)) for (const [k, x] of Object.entries(v.player_totals)) { const id = Number(k); const t = num(x); if (Number.isFinite(id) && t !== null) movers.set(id, t); }
  for (const m of arr(v.top_movers)) if (isObject(m)) { const id = num(m.dg_id); const t = num(m.total_strokes); if (id !== null && t !== null && !movers.has(id)) movers.set(id, t); }
  return { id: str(v.id), label: str(v.label), wind: num(v.wind_delta_mph) ?? 0, temp: num(v.temp_delta_f) ?? 0, rain: num(v.rain_delta_prob) ?? 0, pm: num(v.pm_wind_delta_mph) ?? 0, rounds, movers };
}

/** Parse the document; anything that is not a weather brief gives null so the page falls back to the older weather file. */
export function parseBrief(value: unknown): Brief | null {
  if (!isObject(value) || !str(value.schema).startsWith(BRIEF_SCHEMA_PREFIX)) return null;
  const fc = isObject(value.forecast) ? value.forecast : {};
  const st = isObject(value.status) ? value.status : {};
  const co = isObject(value.course) ? value.course : {};
  const wi = isObject(value.what_if) ? value.what_if : null;
  const scenarios = wi ? arr(wi.scenarios).map(parseScenario).filter((s): s is Scenario => s !== null) : [];
  return {
    schema: str(value.schema), event_uid: str(value.event_uid), event_name: str(value.event_name), as_of: str(value.as_of), model_version: str(value.model_version),
    forecast: { issued_at: str(fc.issued_at), age_hours: num(fc.age_hours), source: str(fc.source), n_scenarios: num(fc.n_scenarios) },
    status: { level: str(st.level, "full"), plain: str(st.plain) },
    headline: arr(value.headline).filter((h): h is string => typeof h === "string" && h.trim() !== ""),
    course: { name: str(co.name), cls: str(co.class), slope: num(co.wind_strokes_per_mph), tourSlope: num(co.tour_wind_strokes_per_mph), multiplier: num(co.wind_multiplier), basis: str(co.basis), plain: str(co.plain) },
    rounds: arr(value.rounds).map(parseRound).filter((r): r is BriefRound => r !== null).sort((a, b) => a.round - b.round),
    players: arr(value.players).map(parsePlayer).filter((p): p is BriefPlayer => p !== null),
    whatIf: scenarios.length ? { basis: wi ? str(wi.basis) : "", scenarios } : null,
    history: arr(value.history).filter(isObject).map((h) => ({
      issued_at: str(h.issued_at),
      rounds: arr(h.rounds).filter(isObject).map((r) => ({ round: num(r.round) ?? 0, field: num(r.field_effect_strokes), wave: num(r.wave_split_strokes) })).filter((r) => r.round > 0),
    })),
    glossary: isObject(value.glossary) ? Object.fromEntries(Object.entries(value.glossary).filter(([, x]) => typeof x === "string")) as Record<string, string> : {},
  };
}

/** The key of the brief for an index event, or null when the index does not point to one (the page then uses the older weather file). */
export const briefKeyFor = (event: { weather_brief_key?: string | null } | null | undefined): string | null => (event?.weather_brief_key ? event.weather_brief_key : null);

/* ------------------------------------------------------------------ wording: never a raw sign */
const TINY = 0.005;
const fix = (v: number, d = 2) => Math.abs(v).toFixed(d);
const unit = (v: number, d = 2) => (fix(v, d) === "1.00" ? "stroke" : "strokes");

/** The whole field: scoring "easier" or "harder" than in neutral conditions. */
export function fieldWords(v: number | null, d = 2): string {
  if (v === null) return "not available";
  if (Math.abs(v) < TINY) return "no real change";
  return `${v < 0 ? "easier" : "harder"} by ${fix(v, d)} ${unit(v, d)}`;
}
/** A player or group: the weather helps or hurts. */
export function helpWords(v: number | null, d = 2): string {
  if (v === null) return "not available";
  if (Math.abs(v) < TINY) return "no real effect";
  return `${v < 0 ? "helps" : "hurts"} by ${fix(v, d)} ${unit(v, d)}`;
}
/** Wave split is PM minus AM expected strokes for the same player: positive means the afternoon wave plays harder, so the morning wave is favored. */
export function waveWords(v: number | null, d = 2): string {
  if (v === null) return "no tee sheet yet";
  if (Math.abs(v) < 0.02) return "the waves are about even";
  return `${v > 0 ? "Morning" : "Afternoon"} wave favored by ${fix(v, d)} ${unit(v, d)}`;
}
/** Which wave is favored, as a tone for coloring: "am", "pm" or "even". */
export const waveSide = (v: number | null): "am" | "pm" | "even" | "none" => (v === null ? "none" : Math.abs(v) < 0.02 ? "even" : v > 0 ? "am" : "pm");
export function spreadWords(m: number | null): string {
  if (m === null) return "not available";
  const pct = (m - 1) * 100;
  if (Math.abs(pct) < 0.3) return "scores spread out as usual";
  return pct > 0 ? `scores spread wider (${pct.toFixed(1)}% more), which helps long shots` : `scores bunch up (${Math.abs(pct).toFixed(1)}% less spread), which helps the favorites`;
}
/** Probability change in points, with a direction word. */
export function pointsWords(p: number | null): string {
  if (p === null) return "n/a";
  const pts = p * 100;
  if (Math.abs(pts) < 0.05) return "about the same";
  return `${pts > 0 ? "up" : "down"} ${Math.abs(pts).toFixed(Math.abs(pts) < 1 ? 2 : 1)} pts`;
}
export function tone(v: number | null): "good" | "bad" | "flat" {
  return v === null || Math.abs(v) < TINY ? "flat" : v < 0 ? "good" : "bad";
}

export const COMPONENT_NAMES: Record<string, string> = {
  wind: "Wind", gust: "Gusts", temp: "Temperature", rain: "Rain", time_of_day: "Time of day", tod: "Time of day", wind_trajectory: "Wind changing through the round",
};
export const componentName = (key: string): string => COMPONENT_NAMES[key] ?? key.replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase());

/** "Wind: helps by 0.03 strokes, Gusts: hurts by 0.01 strokes", dropping parts that round to nothing. */
export function componentWords(components: Record<string, number>): string {
  const parts = Object.entries(components).filter(([, v]) => Math.abs(v) >= TINY).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).map(([k, v]) => `${componentName(k)} ${helpWords(v).replace(" by", "")}`);
  return parts.length ? parts.join("; ") : "Nothing material";
}

/** The sensitivity bar: this course against the tour average. */
export function windSensitivity(course: Brief["course"]): { ratio: number | null; label: string; perFive: number | null } {
  const ratio = course.multiplier ?? (course.slope !== null && course.tourSlope ? course.slope / course.tourSlope : null);
  const label = ratio === null ? "Wind sensitivity unknown" : ratio >= 1.5 ? "Very wind-sensitive" : ratio >= 1.15 ? "More wind-sensitive than average" : ratio > 0.87 ? "About average for wind" : ratio > 0.6 ? "Less wind-sensitive than average" : "Barely affected by wind";
  return { ratio, label, perFive: course.slope === null ? null : course.slope * 5 };
}

export const STATUS_TITLE: Record<string, string> = { full: "Full weather effects in the prices", field_only: "Only the field-wide effect is in the prices", off: "Weather is not in the prices" };
export const statusTitle = (level: string): string => STATUS_TITLE[level] ?? "Weather status";

/** Forecast age in plain words. */
export function ageWords(hours: number | null): string {
  if (hours === null) return "";
  if (hours < 1) return "under an hour old";
  if (hours < 48) return `${Math.round(hours)} hour${Math.round(hours) === 1 ? "" : "s"} old`;
  return `${Math.round(hours / 24)} days old`;
}

/* ------------------------------------------------------------------ players list */
export type SortMode = "helped" | "hurt" | "name" | "skill";
const fold = (s: string) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/[^a-z0-9 ]/g, " ").replace(/\s+/g, " ").trim();
export function searchBrief<T extends { name: string; dg_id: number }>(rows: T[], query: string): T[] {
  const terms = fold(query).split(" ").filter(Boolean);
  if (!terms.length) return rows;
  return rows.filter((r) => { const hay = `${fold(r.name)} ${r.dg_id}`; return terms.every((t) => hay.includes(t)); });
}
/** Most helped first by default (the most negative total). Players with no total sit last whatever the order. */
export function orderPlayers<T extends { name: string; skill_rank: number | null; total: number | null }>(rows: T[], mode: SortMode): T[] {
  const key = (r: T): number | string | null => (mode === "name" ? r.name : mode === "skill" ? r.skill_rank : mode === "hurt" ? (r.total === null ? null : -r.total) : r.total);
  return [...rows].sort((a, b) => {
    const x = key(a), y = key(b);
    if (x === null && y === null) return a.name.localeCompare(b.name);
    if (x === null) return 1;
    if (y === null) return -1;
    const c = typeof x === "string" || typeof y === "string" ? String(x).localeCompare(String(y)) : x - y;
    return c !== 0 ? c : a.name.localeCompare(b.name);
  });
}
export const TOP_SKILL = 30;
export const isTopSkill = (p: { skill_rank: number | null }): boolean => p.skill_rank !== null && p.skill_rank <= TOP_SKILL;

/* ------------------------------------------------------------------ what if: sliders over the published scenario grid */
export type Sliders = { wind: number; pm: number; temp: number; wet: boolean };
export const NEUTRAL_SLIDERS: Sliders = { wind: 0, pm: 0, temp: 0, wet: false };
export const isNeutral = (s: Sliders): boolean => s.wind === 0 && s.pm === 0 && s.temp === 0 && !s.wet;
export type RoundOut = { round: number; field: number | null; wave: number | null; spread: number | null };
export type WhatIfResult = { rounds: RoundOut[]; totals: Map<number, number>; approx: Set<number> };
export type WhatIfModel = { axes: { wind: boolean; pm: boolean; temp: boolean; wet: boolean }; ranges: { wind: [number, number]; pm: [number, number]; temp: [number, number] }; evaluate: (s: Sliders) => WhatIfResult };

type Point = { x: number; rounds: RoundOut[]; totals: Map<number, number>; approx: Set<number> };

/**
 * The published scenarios change one thing at a time (wind, PM-only wind, temperature, wet) around the base forecast. The sliders add
 * each change's effect on the base numbers, interpolating linearly between published points and holding the last published value
 * beyond them. Players outside a scenario's list of biggest movers are scaled from their base effect (least squares through the
 * movers), so their numbers are marked as approximate.
 */
export function buildWhatIf(brief: Brief): WhatIfModel | null {
  const wi = brief.whatIf;
  if (!wi) return null;
  const base = wi.scenarios.find((s) => s.id === "base") ?? wi.scenarios.find((s) => s.wind === 0 && s.temp === 0 && s.rain === 0 && s.pm === 0);
  if (!base) return null;
  const baseTotals = new Map<number, number>(brief.players.filter((p) => p.total !== null).map((p) => [p.dg_id, p.total as number]));
  const baseRounds: RoundOut[] = base.rounds.map((r) => ({ ...r }));

  const pointFor = (s: Scenario, x: number): Point => {
    const rounds = baseRounds.map((b) => {
      const r = s.rounds.find((q) => q.round === b.round);
      return { round: b.round, field: r && b.field !== null && r.field !== null ? r.field - b.field : null, wave: r && b.wave !== null && r.wave !== null ? r.wave - b.wave : null, spread: r && b.spread !== null && r.spread !== null ? r.spread - b.spread : null };
    });
    // Slope of the change on the base effect, from the players the scenario lists; everyone else is scaled by it.
    let sxy = 0, den = 0;
    const totals = new Map<number, number>(), approx = new Set<number>();
    for (const [id, t] of s.movers) { const b = baseTotals.get(id); if (b === undefined) continue; totals.set(id, t - b); sxy += b * (t - b); den += b * b; }
    const k = den > 1e-9 ? sxy / den : 0;
    for (const [id, b] of baseTotals) if (!totals.has(id)) { totals.set(id, k * b); approx.add(id); }
    return { x, rounds, totals, approx };
  };
  const zero = (x: number): Point => ({ x, rounds: baseRounds.map((b) => ({ round: b.round, field: 0, wave: 0, spread: 0 })), totals: new Map([...baseTotals.keys()].map((id) => [id, 0])), approx: new Set() });
  const one = (pred: (s: Scenario) => boolean, xOf: (s: Scenario) => number): Point[] => {
    const pts = wi.scenarios.filter((s) => s !== base && pred(s)).map((s) => pointFor(s, xOf(s)));
    return [zero(0), ...pts.filter((p) => p.x !== 0)].sort((a, b) => a.x - b.x);
  };
  const wind = one((s) => s.wind !== 0 && s.temp === 0 && s.rain === 0 && s.pm === 0, (s) => s.wind);
  const pm = one((s) => s.pm !== 0 && s.wind === 0 && s.temp === 0 && s.rain === 0, (s) => s.pm);
  const temp = one((s) => s.temp !== 0 && s.wind === 0 && s.rain === 0 && s.pm === 0, (s) => s.temp);
  const wet = one((s) => s.rain > 0 && s.wind === 0 && s.temp === 0 && s.pm === 0, () => 1);
  const range = (pts: Point[]): [number, number] => [pts[0].x, pts[pts.length - 1].x];

  const interp = (pts: Point[], x: number): Point => {
    if (pts.length === 1) return zero(x);
    if (x <= pts[0].x) return pts[0];
    if (x >= pts[pts.length - 1].x) return pts[pts.length - 1];
    let i = 0;
    while (pts[i + 1].x < x) i++;
    const a = pts[i], b = pts[i + 1], f = (x - a.x) / (b.x - a.x);
    const mix = (u: number | null, v: number | null) => (u === null || v === null ? null : u + (v - u) * f);
    return {
      x,
      rounds: a.rounds.map((r, j) => ({ round: r.round, field: mix(r.field, b.rounds[j].field), wave: mix(r.wave, b.rounds[j].wave), spread: mix(r.spread, b.rounds[j].spread) })),
      totals: new Map([...a.totals].map(([id, v]) => [id, v + ((b.totals.get(id) ?? v) - v) * f])),
      approx: new Set([...a.approx, ...b.approx]),
    };
  };

  const axes = { wind: wind.length > 1, pm: pm.length > 1, temp: temp.length > 1, wet: wet.length > 1 };
  return {
    axes,
    ranges: { wind: range(wind), pm: range(pm), temp: range(temp) },
    evaluate: (s) => {
      const parts = [axes.wind ? interp(wind, s.wind) : null, axes.pm ? interp(pm, s.pm) : null, axes.temp ? interp(temp, s.temp) : null, axes.wet && s.wet ? wet[1] : null].filter((p): p is Point => p !== null);
      const rounds = baseRounds.map((b, j) => {
        const add = (key: "field" | "wave" | "spread") => (b[key] === null ? null : b[key]! + parts.reduce((acc, p) => acc + (p.rounds[j][key] ?? 0), 0));
        return { round: b.round, field: add("field"), wave: add("wave"), spread: add("spread") };
      });
      const totals = new Map<number, number>(), approx = new Set<number>();
      for (const [id, b] of baseTotals) {
        totals.set(id, b + parts.reduce((acc, p) => acc + (p.totals.get(id) ?? 0), 0));
        if (parts.some((p) => p.approx.has(id))) approx.add(id);
      }
      return { rounds, totals, approx };
    },
  };
}

/** Probability change when the player's total moves: scaled from the published change per stroke (the player's own, or the field's median where the player's is too small to read). */
export function scaledProb(player: BriefPlayer, key: keyof Probs, newTotal: number | null, medianSlope: Partial<Record<keyof Probs, number>>): number | null {
  const p = player.prob[key];
  if (newTotal === null) return null;
  if (player.total !== null && Math.abs(player.total) >= 0.03 && p !== null) return p * (newTotal / player.total);
  const slope = medianSlope[key];
  return slope === undefined ? p : slope * newTotal;
}
/** Probability change per stroke of total weather effect, as the field's median. */
export function medianProbSlopes(players: BriefPlayer[]): Partial<Record<keyof Probs, number>> {
  const out: Partial<Record<keyof Probs, number>> = {};
  for (const key of ["win", "top_5", "top_10", "top_20"] as const) {
    const xs = players.filter((p) => p.total !== null && Math.abs(p.total) >= 0.05 && p.prob[key] !== null).map((p) => (p.prob[key] as number) / (p.total as number)).sort((a, b) => a - b);
    if (xs.length) out[key] = xs[Math.floor(xs.length / 2)];
  }
  return out;
}

/* ------------------------------------------------------------------ history */
export type HistorySeries = { round: number; field: Array<number | null>; wave: Array<number | null>; labels: string[] };
/** One series per round that appears in at least two captures this week (a single capture has no movement to show). */
export function historySeries(brief: Brief): HistorySeries[] {
  const rounds = [...new Set(brief.history.flatMap((h) => h.rounds.map((r) => r.round)))].sort((a, b) => a - b);
  const out: HistorySeries[] = [];
  for (const round of rounds) {
    const caps = brief.history.map((h) => ({ at: h.issued_at, r: h.rounds.find((r) => r.round === round) })).filter((c) => c.r);
    if (caps.length < 2) continue;
    out.push({ round, field: caps.map((c) => c.r!.field), wave: caps.map((c) => c.r!.wave), labels: caps.map((c) => c.at) });
  }
  return out;
}
/** "Got easier by 0.30 strokes since Mon" style sentence for a series; empty when there is no movement to report. */
export function movementWords(values: Array<number | null>, kind: "field" | "wave"): string {
  const v = values.filter((x): x is number => x !== null);
  if (v.length < 2) return "";
  const d = v[v.length - 1] - v[0];
  if (Math.abs(d) < 0.02) return "Barely moved this week";
  if (kind === "field") return `${d < 0 ? "Got easier" : "Got harder"} by ${Math.abs(d).toFixed(2)} strokes since the first look`;
  return `The ${d > 0 ? "morning" : "afternoon"} wave's edge grew by ${Math.abs(d).toFixed(2)} strokes since the first look`;
}

/** Plain slot description: "14 mph wind, gusts 22, 78°F, 30% chance of rain". */
export function slotWords(s: Slot | null): string {
  if (!s) return "no forecast";
  const parts: string[] = [];
  if (s.wind !== null) parts.push(`${Math.round(s.wind)} mph wind`);
  if (s.gust !== null) parts.push(`gusts ${Math.round(s.gust)}`);
  if (s.temp !== null) parts.push(`${Math.round(s.temp)}°F`);
  if (s.rain_prob !== null) parts.push(`${Math.round(s.rain_prob * 100)}% chance of rain`);
  return parts.join(", ");
}
