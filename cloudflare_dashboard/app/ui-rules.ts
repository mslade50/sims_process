/** Pure helpers for the design-system components (count-up text, freshness). No DOM, so tests can import them directly. */

export type NumericText = { prefix: string; value: number; decimals: number; suffix: string; grouped: boolean };

/** Splits "$1,234.5%" style text into prefix, number, suffix. Returns null when the text has no leading number to animate. */
export function parseNumericText(text: string): NumericText | null {
  const match = /^([^0-9+\-−]*)([+\-−]?)(\d[\d,]*)(\.\d+)?([\s\S]*)$/.exec(text);
  if (!match) return null;
  const [, prefix, sign, int, frac = "", suffix] = match;
  const value = Number(`${int.replaceAll(",", "")}${frac}`);
  if (!Number.isFinite(value)) return null;
  return {
    prefix: prefix + (sign === "+" ? "+" : ""),
    value: sign === "-" || sign === "−" ? -value : value,
    decimals: frac ? frac.length - 1 : 0,
    suffix,
    grouped: int.includes(","),
  };
}

const easeOutCubic = (t: number) => 1 - (1 - t) ** 3;

/** The text at `progress` (0..1) of a count-up from zero to `text`. progress >= 1 returns `text` unchanged. */
export function countUpText(text: string, progress: number): string {
  if (progress >= 1) return text;
  const parsed = parseNumericText(text);
  if (!parsed) return text;
  const current = parsed.value * easeOutCubic(Math.max(0, progress));
  const body = Math.abs(current).toLocaleString("en-US", { minimumFractionDigits: parsed.decimals, maximumFractionDigits: parsed.decimals, useGrouping: parsed.grouped });
  const negative = current < 0 && Number(body.replaceAll(",", "")) !== 0;
  const plus = parsed.prefix.endsWith("+");
  const prefix = plus ? parsed.prefix.slice(0, -1) : parsed.prefix;
  return `${prefix}${negative ? "−" : plus ? "+" : ""}${body}${parsed.suffix}`;
}

export type FreshnessTone = "live" | "fresh" | "aging" | "stale" | "unknown";

/** Parses ISO timestamps, also the "2026-10-05T14-50-00-055214Z" folder style used by golfprice run ids. */
export function parseTimestamp(value: unknown): number | null {
  if (typeof value !== "string" || !value) return null;
  const folder = /^(\d{4}-\d\d-\d\d)T(\d\d)-(\d\d)-(\d\d)(?:-\d+)?Z$/.exec(value);
  const direct = Date.parse(folder ? `${folder[1]}T${folder[2]}:${folder[3]}:${folder[4]}Z` : value);
  return Number.isFinite(direct) ? direct : null;
}

export function relativeAge(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 45) return "just now";
  if (s < 5400) return `${Math.max(1, Math.round(s / 60))} min ago`;
  if (s < 172800) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}

/** live < 90 min, fresh < 24 h, aging < 72 h, stale beyond. */
export function freshnessTone(ageMs: number | null): FreshnessTone {
  if (ageMs === null) return "unknown";
  if (ageMs < 90 * 60_000) return "live";
  if (ageMs < 24 * 3_600_000) return "fresh";
  if (ageMs < 72 * 3_600_000) return "aging";
  return "stale";
}

/** User-selectable accents. d = fill/text on dark surfaces, l = fill/text on light surfaces (both validated for contrast in tests/theme.test.mjs). */
export const ACCENTS = [
  { key: "gold", name: "Gold", d: "#e6bf6a", l: "#7d5600" },
  { key: "fairway", name: "Fairway", d: "#5fd6a4", l: "#09683f" },
  { key: "sky", name: "Sky", d: "#8db4ff", l: "#1d5fb8" },
  { key: "rose", name: "Rose", d: "#f58fb0", l: "#b0306a" },
] as const;
export type AccentKey = (typeof ACCENTS)[number]["key"];
export type ThemeMode = "auto" | "dark" | "light";
export const THEME_STORAGE_KEY = "golf-dashboard-theme";
export const ACCENT_STORAGE_KEY = "golf-dashboard-accent-v2";

/** Inline script run before first paint so a saved theme/accent never flashes the default. */
export function themeBootScript(): string {
  return `try{var d=document.documentElement,A=${JSON.stringify(Object.fromEntries(ACCENTS.map((a) => [a.key, [a.d, a.l]])))},t=localStorage.getItem("${THEME_STORAGE_KEY}"),k=localStorage.getItem("${ACCENT_STORAGE_KEY}");if(t==="light"||t==="dark")d.dataset.theme=t;if(A[k]){d.style.setProperty("--accent-d",A[k][0]);d.style.setProperty("--accent-l",A[k][1])}if(localStorage.getItem("golf-dashboard-density")==="compact")d.dataset.density="compact"}catch(e){}`;
}
