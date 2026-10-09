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

/** Daily-refreshed reference catalogs (player profiles): fresh through 7 days, amber ("aging") beyond. */
export const CATALOG_STALE_MS = 7 * 86_400_000;
export function catalogFreshnessTone(ageMs: number | null): FreshnessTone {
  if (ageMs === null) return "unknown";
  return ageMs > CATALOG_STALE_MS ? "aging" : "fresh";
}

/* ------------------------------------------------------------------ header freshness (site audit C3) */
export type IndexEventLite = {
  name?: string; course?: string; date_start?: string; event_uid?: string;
  runs?: Array<{ as_of?: string }>;
  /** Added by runner release 1; ignored until present. */
  published_at?: string; held_back?: unknown;
};
export type EventAge = { name: string; asOf: string | null; asOfMs: number | null; ageMs: number | null; publishedAt: string | null; heldBack: string | null };
export type WeekSummary = {
  title: string; sub: string;
  /** OLDEST per-event as-of (each event's newest run), so one stale event cannot hide behind a fresh one. */
  oldestAsOf: string | null;
  events: EventAge[];
  tooltip: string;
  /** Display path for runner-release-1 fields: null (nothing shown) until index.json carries them. */
  publishedLine: string | null;
  heldBackNote: string | null;
};

/** Text for a `held_back` value: a string, {reason|message}, or a list of those. Empty/false/absent -> null. */
export function heldBackText(value: unknown): string | null {
  if (!value) return null;
  if (typeof value === "string") return value.trim() || null;
  if (Array.isArray(value)) return value.map(heldBackText).filter(Boolean).join("; ") || null;
  if (typeof value === "object") {
    const o = value as Record<string, unknown>;
    return heldBackText(o.reason ?? o.message ?? o.detail) ?? "a newer run was held back";
  }
  return "a newer run was held back";
}

export function summarizeWeek(index: { events?: IndexEventLite[] } | null | undefined, now: number): WeekSummary | null {
  const dated = (index?.events ?? []).filter((e) => e.date_start);
  if (!dated.length) return null;
  const newest = dated.map((e) => e.date_start as string).sort().at(-1);
  const week = dated.filter((e) => e.date_start === newest);
  const events: EventAge[] = week.map((e) => {
    const stamps = (e.runs ?? []).map((r) => ({ raw: r.as_of ?? "", ms: parseTimestamp(r.as_of) })).filter((s): s is { raw: string; ms: number } => s.ms !== null);
    const last = stamps.sort((a, b) => a.ms - b.ms).at(-1) ?? null;
    return {
      name: e.name ?? e.event_uid ?? "event",
      asOf: last?.raw ?? null, asOfMs: last?.ms ?? null, ageMs: last ? now - last.ms : null,
      publishedAt: typeof e.published_at === "string" && parseTimestamp(e.published_at) !== null ? e.published_at : null,
      heldBack: heldBackText(e.held_back),
    };
  });
  const withRun = events.filter((e) => e.asOfMs !== null).sort((a, b) => (a.asOfMs as number) - (b.asOfMs as number));
  const tooltip = events.map((e) => `${e.name}: ${e.ageMs === null ? "no run yet" : `run ${relativeAge(e.ageMs)}`}${e.publishedAt ? `; published ${relativeAge(now - (parseTimestamp(e.publishedAt) as number))}` : ""}`).join("\n");
  const published = events.filter((e) => e.publishedAt).sort((a, b) => (parseTimestamp(a.publishedAt) as number) - (parseTimestamp(b.publishedAt) as number));
  const held = events.filter((e) => e.heldBack);
  return {
    title: week.map((e) => e.name ?? e.event_uid ?? "").join(" · "),
    sub: week.map((e) => e.course ?? "").filter(Boolean).join(" · "),
    oldestAsOf: withRun[0]?.asOf ?? null,
    events, tooltip,
    publishedLine: published.length ? `Published ${relativeAge(now - (parseTimestamp(published[0].publishedAt) as number))}` : null,
    heldBackNote: held.length ? `Newer run not published: ${held.map((e) => (events.length > 1 ? `${e.name}: ${e.heldBack}` : e.heldBack)).join("; ")}` : null,
  };
}

/** User-selectable accents. d = fill/text on dark surfaces, l = fill/text on light surfaces (both validated for contrast in tests/theme.test.mjs). */
export const ACCENTS = [
  { key: "sky", name: "Sky", d: "#8db4ff", l: "#1d5fb8" },
  { key: "gold", name: "Gold", d: "#e6bf6a", l: "#7d5600" },
  { key: "fairway", name: "Fairway", d: "#5fd6a4", l: "#09683f" },
  { key: "rose", name: "Rose", d: "#f58fb0", l: "#b0306a" },
] as const;
export type AccentKey = (typeof ACCENTS)[number]["key"];
export type ThemeMode = "auto" | "dark" | "light";
export const THEME_STORAGE_KEY = "golf-dashboard-theme";
export const ACCENT_STORAGE_KEY = "golf-dashboard-accent-v3";

/** Inline script run before first paint so a saved theme/accent never flashes the default. */
export function themeBootScript(): string {
  return `try{var d=document.documentElement,A=${JSON.stringify(Object.fromEntries(ACCENTS.map((a) => [a.key, [a.d, a.l]])))},t=localStorage.getItem("${THEME_STORAGE_KEY}"),k=localStorage.getItem("${ACCENT_STORAGE_KEY}");if(t==="light"||t==="dark")d.dataset.theme=t;if(A[k]){d.style.setProperty("--accent-d",A[k][0]);d.style.setProperty("--accent-l",A[k][1])}if(localStorage.getItem("golf-dashboard-density")==="compact")d.dataset.density="compact"}catch(e){}`;
}
