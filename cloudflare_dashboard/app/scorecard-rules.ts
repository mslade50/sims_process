/**
 * golfprice weekly scorecard (golfprice.scorecard.v1, published to golfprice/scorecard/latest.json by golfprice/scorecard.py).
 * Pure TypeScript: parsing that never throws, freshness and number formatting shared by the Performance view and tests/scorecard.test.mjs.
 */

export const SCORECARD_KEY = "golfprice/scorecard/latest.json";
export const SCORECARD_STALE_DAYS = 8;
export const MARKETS = ["win", "top_5", "top_10", "top_20", "make_cut"] as const;
export const MARKET_LABELS: Record<string, string> = { win: "Win", top_5: "Top 5", top_10: "Top 10", top_20: "Top 20", make_cut: "Make cut" };

export type Interval = { n: number; mean: number | null; lo: number | null; hi: number | null };
export type ScorecardArm = {
  arm: string;
  label: string;
  role?: string;
  rmse: number | null;
  rmse_vs_champion: number | null;
  logloss_vs_champion?: Record<string, number>;
  logloss_vs_close?: Record<string, number>;
  logloss_vs_champion_mean: number | null;
  logloss_vs_close_mean: number | null;
};
export type BetSummary = { n: number; settled: number; pnl_units: number | null; roi: number | null };
export type ScorecardEvent = {
  event_uid: string;
  name: string;
  tour: string;
  event_start: string;
  status: string;
  finish_settled: boolean;
  arms: ScorecardArm[];
  bets: Record<string, BetSummary>;
};
export type ScorecardForward = {
  clock_start_event_uid: string | null;
  events_counted: number;
  futility_check_at: number;
  horizon_events: number;
  progress_to_futility_check: number;
  progress_to_horizon: number;
  by_arm: Record<string, { label: string; events: number; rmse_vs_champion: Interval; logloss_vs_champion_mean: Interval; logloss_vs_close_mean: Interval }>;
  bets: Record<string, BetSummary>;
  rule?: string;
};
export type Scorecard = {
  schema: string;
  generated_at: string;
  headlines: string[];
  latest_week: { event_start: string | null; event_uids: string[] };
  events: ScorecardEvent[];
  n_events_settled: number;
  forward: ScorecardForward;
};

const isObject = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);

/** A usable scorecard or null (wrong schema, missing parts): the view then shows its empty state instead of crashing. */
export function parseScorecard(value: unknown): Scorecard | null {
  if (!isObject(value) || value.schema !== "golfprice.scorecard.v1") return null;
  if (!Array.isArray(value.headlines) || !Array.isArray(value.events) || !isObject(value.forward) || !isObject(value.latest_week)) return null;
  if (typeof value.generated_at !== "string") return null;
  if (!isObject(value.forward.by_arm) || !isObject(value.forward.bets)) return null;
  return value as unknown as Scorecard;
}

export function ageDays(generatedAt: string, now: number): number | null {
  const ms = Date.parse(generatedAt);
  return Number.isFinite(ms) ? (now - ms) / 86_400_000 : null;
}

export const isStale = (generatedAt: string, now: number, limitDays = SCORECARD_STALE_DAYS): boolean => {
  const age = ageDays(generatedAt, now);
  return age === null || age > limitDays;
};

/** Events of the latest settled week, newest first. */
export function latestWeekEvents(card: Scorecard): ScorecardEvent[] {
  const wanted = new Set(card.latest_week.event_uids);
  return card.events.filter((event) => wanted.has(event.event_uid));
}

export const fmt = (value: number | null | undefined, digits = 3): string => (value === null || value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits));
export const signedFmt = (value: number | null | undefined, digits = 3): string =>
  value === null || value === undefined || !Number.isFinite(value) ? "—" : `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;

/** Negative is better for every delta in the scorecard (strokes, log-loss). */
export const deltaTone = (value: number | null | undefined): "positive" | "negative" | "" => (value === null || value === undefined || value === 0 ? "" : value < 0 ? "positive" : "negative");

export const intervalText = (interval: Interval | undefined, digits = 3): string => {
  if (!interval || interval.mean === null) return "—";
  const ci = interval.lo === null || interval.hi === null ? "too few events for an interval" : `95% ${signedFmt(interval.lo, digits)} to ${signedFmt(interval.hi, digits)}`;
  return `${signedFmt(interval.mean, digits)} (${ci})`;
};
