export type DataRow = Record<string, unknown>;

/** Chart series colors as CSS variables (tokens.css), so charts follow the light/dark theme. */
export const palette = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)", "var(--chart-5)", "var(--chart-6)"];

export function numberValue(value: unknown, fallback = 0): number {
  if (typeof value === "number") return Number.isFinite(value) ? value : fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function titleCase(value: unknown): string {
  return String(value ?? "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function uniqueStrings(rows: DataRow[], key: string): string[] {
  return [...new Set(rows.map((row) => String(row[key] ?? "").trim()).filter(Boolean))].sort((a, b) =>
    a.localeCompare(b),
  );
}

export function americanOdds(probability: number): string {
  if (!Number.isFinite(probability) || probability <= 0 || probability >= 1) return "—";
  if (probability >= 0.5) return String(Math.round((-probability / (1 - probability)) * 100));
  return `+${Math.round(((1 - probability) / probability) * 100)}`;
}

export function formatCell(value: unknown, key = ""): string {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "object") return JSON.stringify(value);
  if (typeof value === "number") {
    if (/prob|pct|percent/i.test(key) && Math.abs(value) <= 1) return `${(value * 100).toFixed(1)}%`;
    if (/edge|roi|rate/i.test(key)) return `${value.toFixed(1)}%`;
    if (Number.isInteger(value)) return value.toLocaleString();
    return value.toLocaleString(undefined, { maximumFractionDigits: 3 });
  }
  return titleCase(value);
}

export function displayDate(value: unknown): string {
  return etTime(value, "Not available");
}

/** Owner rule: every timestamp on the site reads in US Eastern time ("Oct 8, 12:10 PM ET"), never UTC or raw ISO. */
export function etTime(value: unknown, fallback = "—"): string {
  const date = new Date(String(value ?? ""));
  if (value == null || value === "" || Number.isNaN(date.getTime())) return fallback;
  return `${new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date)} ET`;
}

export function safeMean(values: number[]): number {
  const finite = values.filter(Number.isFinite);
  return finite.length ? finite.reduce((sum, value) => sum + value, 0) / finite.length : 0;
}

export function weightedMean(rows: DataRow[], valueKey: string, weightKey = "rounds"): number {
  let weightedTotal = 0;
  let totalWeight = 0;

  for (const row of rows) {
    const value = numberValue(row[valueKey], Number.NaN);
    const weight = numberValue(row[weightKey], 0);
    if (!Number.isFinite(value) || weight <= 0) continue;
    weightedTotal += value * weight;
    totalWeight += weight;
  }

  if (totalWeight > 0) return weightedTotal / totalWeight;
  return safeMean(rows.map((row) => numberValue(row[valueKey], Number.NaN)));
}

export function diagnosticRoundCount(rows: DataRow[]): number {
  const totalRows = rows.filter((row) => String(row.category) === "total");
  if (totalRows.length) return sum(totalRows.map((row) => numberValue(row.rounds)));

  const roundsByEvent = new Map<string, number>();
  for (const row of rows) {
    const eventKey = `${String(row.year ?? "")}:${String(row.event_id ?? "")}`;
    roundsByEvent.set(eventKey, Math.max(roundsByEvent.get(eventKey) ?? 0, numberValue(row.rounds)));
  }
  return sum([...roundsByEvent.values()]);
}

export function diagnosticEventCount(rows: DataRow[]): number {
  return new Set(rows.map((row) => `${String(row.year ?? "")}:${String(row.event_id ?? "")}`)).size;
}

export function sum(values: number[]): number {
  return values.filter(Number.isFinite).reduce((total, value) => total + value, 0);
}

/* ------------------------------------------------------------------ shared helpers for later consolidation (site audit F1)
 * These are the target definitions. The ~10 local copies of `finite` and the divergent `pct` functions elsewhere are NOT replaced yet:
 * moving imports can break the ScoringView test harness, and the two existing `pct`s disagree on sign. */

/** True for a finite number (not NaN, not Infinity, not a numeric string). */
export const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);

/**
 * Probability or share as a percentage, UNSIGNED (0.1234 -> "12.3%"; below 0.1% two decimals; null/NaN -> "—"). Same convention as
 * explain-rules.pct and ScoringView.pct. odds-signals-rules.pct is the odd one out: it always prefixes "+" for non-negative values, which is
 * right for an edge or change but wrong for a probability; use `signedPct` where a sign is wanted.
 */
export function pct(value: number | null | undefined, digits = 1): string {
  if (!finite(value)) return "—";
  return Math.abs(value) < 0.001 && value !== 0 ? `${(value * 100).toFixed(2)}%` : `${(value * 100).toFixed(digits)}%`;
}

/** Percentage with an explicit sign for gains/changes (+1.2%, -0.4%, 0.0% unsigned at exactly zero). */
export function signedPct(value: number | null | undefined, digits = 1): string {
  if (!finite(value)) return "—";
  return `${value > 0 ? "+" : ""}${(value * 100).toFixed(digits)}%`;
}

/** Date only ("Oct 8"), in the viewer's locale and zone; "—" for anything unparseable. Use displayDate for date plus time. */
export function shortDate(value: unknown): string {
  const date = new Date(String(value ?? ""));
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric" }).format(date);
}

/* ------------------------------------------------------------------ plain event and player-style names (site cleanup, October 2026) */
const EVENT_NAMES: Record<string, string> = {
  the_open: "The Open", rocket: "Rocket Classic", us_open: "U.S. Open", travelers: "Travelers Championship", pga_c: "PGA Championship",
  masters: "The Masters", rbc_canada: "RBC Canadian Open", rbc: "RBC Heritage", wyndham: "Wyndham Championship", players: "The Players Championship",
  houston: "Houston Open", scottish: "Scottish Open", cadillac: "Cadillac Championship", st_jude: "FedEx St. Jude Championship", memorial: "The Memorial",
  valspar: "Valspar Championship", valero: "Valero Texas Open", schwab: "Charles Schwab Challenge", deere: "John Deere Classic", truist: "Truist Championship",
  cjcup: "CJ Cup", "3m_open": "3M Open", bayhill: "Arnold Palmer Invitational (Bay Hill)", wm_phoenix: "WM Phoenix Open", cognizant: "Cognizant Classic",
  genesis: "Genesis Invitational", att: "AT&T Pebble Beach", bmw: "BMW Championship",
};
/** A readable event name for a publisher slug ("pga_c" becomes "PGA Championship"); unknown slugs fall back to title case with a trailing number removed. */
export function eventDisplayName(value: unknown): string {
  const raw = String(value ?? "").trim();
  if (!raw) return "";
  const slug = raw.toLowerCase().replace(/[\s-]+/g, "_");
  return EVENT_NAMES[slug] ?? EVENT_NAMES[slug.replace(/_?\d+$/, "")] ?? titleCase(raw.replace(/\s+\d+$/, ""));
}

const STYLE_NAMES: Record<string, string> = {
  APP_ARG: "Approach and short game", APP_PUTT: "Approach and putting", ARG_PUTT: "Short game and putting", OTT_APP: "Driving and approach", OTT_PUTT: "Driving and putting",
  Stud: "Top all-round player", "Balanced APP": "Balanced, strong approach", "Balanced ARG": "Balanced, strong short game", "Balanced OTT": "Balanced, strong driver", "Balanced Putter": "Balanced, strong putter",
  "Long Wild": "Long and wild", "Long Accurate": "Long and accurate", "Short Accurate": "Short and accurate", "Low Skill": "Below tour average", Unknown: "Not classified",
};
/** One-line definitions shown as hover text and in the key under player-style charts. */
export const STYLE_DEFINITIONS: Record<string, string> = {
  "Top all-round player": "Well above the field in every part of the game.",
  "Below tour average": "Below the tour average in most parts of the game.",
  "Long and wild": "Drives it far but not accurately.", "Long and accurate": "Drives it far and keeps it in play.", "Short and accurate": "Keeps it in play but gives up distance.",
  "Ball Striker": "Strong off the tee and with approach shots.", "Short Game Specialist": "Gains most around the green and putting.", "Elite Putter": "Putting is clearly his best skill.",
  "Not classified": "Not enough data to assign a style.",
};
/** A plain player-style name ("OTT_PUTT" becomes "Driving and putting"). */
export const styleName = (value: unknown): string => STYLE_NAMES[String(value ?? "").trim()] ?? String(value ?? "");
