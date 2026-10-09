/**
 * One vocabulary for every player-strength number on the site (site audit, October 2026).
 * Four short names; the long form lives in a tooltip or the Method disclosure. Never invent
 * a fifth label for one of these quantities in a view; import from here.
 */
export const LABELS = {
  /** Descriptive headline: recency-weighted R1-R2 adjusted SG minus the fixed 2025 PGA round anchor. */
  vsPgaAvg: { short: "vs PGA avg", long: "SG vs PGA Tour average, per round: recent R1-R2 form on the 2025 PGA average-round anchor, unshrunk", unit: "SG/round" },
  /** Saved event model skill re-expressed on the tour scale: mu + field_offset (F02 field mean). */
  thisWeekPga: { short: "This week (PGA scale)", long: "Saved event model skill plus this field's offset to the PGA scale (F02 theta mean of the field); shrunk, with location and weather", unit: "SG/round" },
  /** Saved event model skill centred on the active field (mean exactly 0). */
  vsField: { short: "vs field", long: "Saved event model skill minus the active-field mean; positive is stronger than this field's average", unit: "SG/round" },
  /** Method-only name for the unshrunk value once a shrunk headline exists. */
  unshrunkForm: { short: "unshrunk form", long: "The same recency-weighted R1-R2 value before shrinkage toward the player's primary-tour mean", unit: "SG/round" },
} as const;
export type LabelKey = keyof typeof LABELS;

/** The one standing sentence that accompanies the two numbers when they share a screen (step0_gap REPORT). */
export const GAP_METHOD_SENTENCE =
  "'vs PGA avg' is R1-2 form against the 2025 PGA average, unshrunk. 'This week' is the model's all-round skill on the current tour scale, shrunk and adjusted for location and weather. Expect the first to read 0.15-0.2 higher for PGA players (up to 0.35 in strong small fields), and about 0.3 apart per player.";

/** Zero-point sentence for every PGA-scale number (BRIEF_v2 A3 / B2 row 8). */
export const ZERO_SENTENCE = "Zero is the average 2025 PGA Tour R1-R2 round (round-weighted), not an equal-weight average player; a typical full-time PGA regular reads about +0.4.";

/** Plain names for saved-component keys that appear as raw headers today (BRIEF_v2 B4). Prefer the explain document's components_legend label when present; fall back here. */
const PLAIN: Record<string, string> = {
  act: "Activity", disp: "Dispersion", sklv: "Skill level", sit: "Situation", xtour: "Cross-tour", chl_total: "CHL base", location: "Location", override: "Override",
  b8_delta: "B8 live shift", contention: "Contention shift", mu: "Model skill", mu_live: "Live model skill", mu_tour: LABELS.thisWeekPga.short,
};
export function plainName(key: string, legend?: Record<string, { label?: string } | undefined>): string {
  return legend?.[key]?.label ?? PLAIN[key] ?? key.replaceAll("_", " ");
}
