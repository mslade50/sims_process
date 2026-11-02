/**
 * Pure parsing for the golfprice course history document (golfprice/history/<event>.json, schema golfprice.course_history.v1, written at publish by
 * golfprice/course_history.py): earlier editions of this week's event or course, round by round. Shared by CourseHistoryPanel and tests.
 * Anything that is not the document gives null; missing numbers stay null (shown as a dash, never guessed).
 */
import { historyThrough } from "./freshness-rules.ts";
export const COURSE_HISTORY_SCHEMA = "golfprice.course_history.v1";
export type HistoryRound = {
  round: number; date: string | null; course: string | null; at_venue: boolean; par: number | null; players: number | null;
  field_avg: number | null; vs_par: number | null; wind: number | null; temp: number | null; rain: number | null;
  weather_strokes: number | null; field_skill: number | null; adjusted_vs_par: number | null;
};
export type HistoryEdition = { event_uid: string; year: number; start: string | null; name: string | null; same_event: boolean; courses: string[]; rounds: HistoryRound[] };
export type HistoryByRound = { round: number; editions: number; vs_par: number | null; wind: number | null; temp: number | null; weather_strokes: number | null; field_skill: number | null; adjusted_vs_par: number | null };
export type CourseHistoryDoc = { status: "available" | "unavailable"; reason: string | null; course: string | null; editions: HistoryEdition[]; by_round: HistoryByRound[]; weather_model: string | null; /** Newest data date the document was built from (published source_through_date, else the newest round it contains). */ through: string | null };

const num = (x: unknown): number | null => typeof x === "number" && Number.isFinite(x) ? x : null;
const str = (x: unknown): string | null => typeof x === "string" && x ? x : null;
const obj = (x: unknown): Record<string, unknown> | null => x && typeof x === "object" && !Array.isArray(x) ? x as Record<string, unknown> : null;
const arr = (x: unknown): unknown[] => Array.isArray(x) ? x : [];

export function parseCourseHistory(raw: unknown): CourseHistoryDoc | null {
  const d = obj(raw);
  if (!d || d.schema !== COURSE_HISTORY_SCHEMA) return null;
  const editions = arr(d.editions).flatMap((e): HistoryEdition[] => {
    const o = obj(e), year = num(o?.year);
    if (!o || year == null || typeof o.event_uid !== "string") return [];
    const rounds = arr(o.rounds).flatMap((r): HistoryRound[] => {
      const x = obj(r), round = num(x?.round);
      if (!x || round == null) return [];
      return [{ round, date: str(x.date), course: str(x.course), at_venue: x.at_venue === true, par: num(x.par), players: num(x.players), field_avg: num(x.field_avg), vs_par: num(x.vs_par),
        wind: num(x.wind), temp: num(x.temp), rain: num(x.rain), weather_strokes: num(x.weather_strokes), field_skill: num(x.field_skill), adjusted_vs_par: num(x.adjusted_vs_par) }];
    });
    return [{ event_uid: o.event_uid, year, start: str(o.start), name: str(o.name), same_event: o.same_event === true, courses: arr(o.courses).filter((c): c is string => typeof c === "string"), rounds }];
  });
  const by_round = arr(d.by_round).flatMap((r): HistoryByRound[] => {
    const x = obj(r), round = num(x?.round);
    return x && round != null ? [{ round, editions: num(x.editions) ?? 0, vs_par: num(x.vs_par), wind: num(x.wind), temp: num(x.temp), weather_strokes: num(x.weather_strokes), field_skill: num(x.field_skill), adjusted_vs_par: num(x.adjusted_vs_par) }] : [];
  });
  const through = historyThrough(d.source_through_date, editions.flatMap((e) => e.rounds.map((r) => r.date)));
  return { status: d.status === "available" && editions.length ? "available" : "unavailable", reason: str(d.reason), course: str(d.course), editions, by_round, weather_model: str(d.weather_model), through };
}

export type HistoryMetric = "vs_par" | "adjusted_vs_par" | "wind" | "weather_strokes" | "field_skill";
/** One edition's round, with every course it was played on merged (player-weighted) for the bar, and the per-course rows kept for the tooltip. */
export type RoundCell = { year: number; name: string | null; same_event: boolean; players: number | null; field_avg: number | null; vs_par: number | null; wind: number | null; temp: number | null;
  rain: number | null; weather_strokes: number | null; field_skill: number | null; adjusted_vs_par: number | null; courses: HistoryRound[] };
const KEYS = ["field_avg", "vs_par", "wind", "temp", "rain", "weather_strokes", "field_skill", "adjusted_vs_par"] as const;

function mergeCourses(rs: HistoryRound[]): Omit<RoundCell, "year" | "name" | "same_event" | "courses"> {
  const out: Record<string, number | null> = { players: rs.reduce((s, r) => s + (r.players ?? 0), 0) || null };
  for (const k of KEYS) {
    const have = rs.filter((r) => r[k] != null);
    const w = have.reduce((s, r) => s + (r.players ?? 1), 0);
    out[k] = have.length && w > 0 ? have.reduce((s, r) => s + (r[k] as number) * (r.players ?? 1), 0) / w : null;
  }
  return out as Omit<RoundCell, "year" | "name" | "same_event" | "courses">;
}

/** One chart row per round number; per edition year a bar value `y<year>` (the chosen measure) and the full cell `c<year>` for the hover card.
 * years run oldest first so colours step light (old) to dark (recent). */
export function historyRoundRows(doc: CourseHistoryDoc, metric: HistoryMetric): { rows: Array<Record<string, unknown>>; years: number[]; rounds: number[] } {
  const eds = [...doc.editions].sort((a, b) => a.year - b.year || (a.start ?? "").localeCompare(b.start ?? ""));
  const years = [...new Set(eds.map((e) => e.year))];
  const rounds = [...new Set(eds.flatMap((e) => e.rounds.map((r) => r.round)))].sort((a, b) => a - b);
  const rows = rounds.map((rn) => {
    const row: Record<string, unknown> = { round: rn, label: `Round ${rn}` };
    for (const y of years) {
      const rs = eds.filter((e) => e.year === y).flatMap((e) => e.rounds.filter((r) => r.round === rn));
      if (!rs.length) { row[`y${y}`] = null; continue; }
      const e = eds.find((x) => x.year === y)!;
      const cell: RoundCell = { year: y, name: e.name, same_event: eds.filter((x) => x.year === y).every((x) => x.same_event), ...mergeCourses(rs), courses: rs };
      row[`c${y}`] = cell;
      row[`y${y}`] = cell[metric];
    }
    return row;
  });
  return { rows, years, rounds };
}

export const signed = (n: number | null, d = 2): string => n == null ? "—" : `${n > 0 && Number(n.toFixed(d)) !== 0 ? "+" : ""}${Number(n.toFixed(d)) === 0 ? (0).toFixed(d) : n.toFixed(d)}`;
