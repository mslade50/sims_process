/** Playing-style descriptors published by the golfprice profile builder (archetype.style). Dependency-free. */
export type StyleDescriptor = {
  key: string; group: string; title: string; label: string; value: number; display?: string; unit?: string;
  percentile: number; interval_90?: number[] | null; level: "headline" | "notable" | "typical"; direction: "high" | "low" | "mid";
  sentence: string; clause?: string | null; n_events?: number; n_units?: number; reference_players?: number; reliability?: number | null; low_signal?: boolean; typical_value?: number;
};
export type PlayerStyle = { descriptors: StyleDescriptor[]; rounds?: number; regular?: boolean };
export type StyleMeta = { cohort_definition?: string; window_days?: number; residual?: string; uncertainty_method?: string; reliability_rule?: string; round_selection?: string; unavailable?: string[]; levels?: Record<string, string> };

const fin = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const LEVEL_ORDER = { headline: 0, notable: 1, typical: 2 } as const;
export const GROUP_ORDER = ["Consistency", "Pace across a week", "Momentum", "Pressure", "Conditions"];

/** Valid published style block, or null so the caller can fall back to the three-trait view. */
export function styleOf(archetype: unknown): PlayerStyle | null {
  const s = (archetype as { style?: unknown } | null | undefined)?.style as PlayerStyle | undefined;
  if (!s || !Array.isArray(s.descriptors)) return null;
  const descriptors = s.descriptors.filter(d => d && typeof d.key === "string" && fin(d.percentile) && fin(d.value) && typeof d.sentence === "string" && d.level in LEVEL_ORDER);
  return descriptors.length ? { ...s, descriptors } : null;
}

/** How far a descriptor sits from the middle of the cohort (0-50). */
export const extremity = (d: StyleDescriptor): number => Math.abs(d.percentile - 50);

/** Strongest first: headlines, then notable, then by distance from the middle of the cohort. */
export const ranked = (ds: StyleDescriptor[]): StyleDescriptor[] => [...ds].sort((a, b) => LEVEL_ORDER[a.level] - LEVEL_ORDER[b.level] || extremity(b) - extremity(a));

/** Plain wording of a percentile: "bigger than 83% of PGA regulars" is built from `high`/`low` words by the caller. */
export function shareBeyond(d: StyleDescriptor): number { return Math.round(d.direction === "low" ? 100 - d.percentile : d.percentile); }

/** Two to four sentences from the strongest surfaced descriptors; honest when nothing stands out. */
export function narrative(style: PlayerStyle, name?: string): string {
  const who = name?.trim() || "This player", ds = ranked(style.descriptors).filter(d => d.level !== "typical" && d.direction !== "mid" && d.clause);
  const lead = ds.filter(d => d.level === "headline"), pool = (lead.length ? lead : ds).slice(0, 4);
  const rounds = style.rounds ? ` Based on ${style.rounds.toLocaleString()} PGA Tour rounds, all four rounds of each week.` : "";
  if (!pool.length) return `${who} has no round-pattern that clearly separates him from a typical PGA regular: swings, starts, finishes and momentum all look ordinary.${rounds}`;
  const clauses = pool.map(d => d.clause as string);
  const list = clauses.length === 1 ? clauses[0] : `${clauses.slice(0, -1).join("; ")}; and ${clauses[clauses.length - 1]}`;
  const hedge = lead.length ? "" : " These are leanings rather than firm traits: the evidence is thin.";
  return `${who} ${list}.${hedge}${rounds}`;
}

export type StyleGroup = { group: string; items: StyleDescriptor[] };
/** Cards grouped by theme, strongest first within each group. */
export function grouped(style: PlayerStyle): StyleGroup[] {
  const names = [...GROUP_ORDER, ...style.descriptors.map(d => d.group).filter(g => !GROUP_ORDER.includes(g))];
  return [...new Set(names)].map(group => ({ group, items: ranked(style.descriptors.filter(d => d.group === group)) })).filter(g => g.items.length);
}

/** "likely range" text for the hover/disclosure; null when no interval was published. */
export function intervalText(d: StyleDescriptor): string | null {
  const iv = d.interval_90;
  return Array.isArray(iv) && iv.length === 2 && fin(iv[0]) && fin(iv[1]) ? `${iv[0].toFixed(2)} to ${iv[1].toFixed(2)}` : null;
}

/** Position of the player within the cohort as a 0-100 marker for the card strip. */
export const markerPosition = (d: StyleDescriptor): number => Math.max(2, Math.min(98, d.percentile));

/** A typical PGA regular's own round-to-round swing (strokes), when the style block is published. */
export const typicalSwing = (archetype: unknown): number | null => { const v = styleOf(archetype)?.descriptors.find(d => d.key === "swing")?.typical_value; return typeof v === "number" && v > 0 ? v : null; };
