/**
 * One vocabulary for every player-strength number on the site (site audit, October 2026).
 * Four short names; the long form lives in a tooltip or the Method disclosure. Never invent
 * a fifth label for one of these quantities in a view; import from here.
 */
export const LABELS = {
  /** Descriptive headline: recency-weighted R1-R2 adjusted SG, pulled toward the player's main-tour level, minus the fixed 2025 PGA round anchor. */
  vsPgaAvg: { short: "vs PGA avg", long: "Strokes per round better (+) or worse (-) than the average 2025 PGA Tour round, judged from recent form in rounds 1 and 2. Players with few rounds are pulled toward the typical level of their main tour, so a small hot streak is not overstated", unit: "SG/round" },
  /** Saved event model skill re-expressed on the tour scale: mu + field_offset (F02 field mean). */
  thisWeekPga: { short: "This week (PGA scale)", long: "The model's skill estimate for this week on the PGA Tour scale: its saved skill number plus how this field compares with a normal PGA field, including course location and weather", unit: "SG/round" },
  /** Saved event model skill centred on the active field (mean exactly 0). */
  vsField: { short: "vs field", long: "The model's skill estimate for this week minus the average of this week's field; positive means stronger than the field's average", unit: "SG/round" },
  /** Method-only name for the unshrunk value now that the headline is shrunk. */
  unshrunkForm: { short: "unshrunk form", long: "The same recent-form figure before it is pulled toward the player's main-tour level", unit: "SG/round" },
} as const;
export type LabelKey = keyof typeof LABELS;

/** The one standing sentence that accompanies the two numbers when they share a screen (step0_gap REPORT). */
export const GAP_METHOD_SENTENCE =
  "'vs PGA avg' is recent form (rounds 1 and 2) on the average-2025-PGA-round scale. 'This week' is the model's all-round skill on the same scale, adjusted for course location and weather. A few tenths of a stroke apart per player is normal; the gap is form versus model.";

/** Zero-point sentence for every PGA-scale number (BRIEF_v2 A3 / B2 row 8). */
export const ZERO_SENTENCE = "Zero is the average round on the 2025 PGA Tour (opening two rounds, every round counted equally), not an average player; a typical full-time PGA regular reads about +0.4.";

/** Plain names for saved-component keys that appear as raw headers today (BRIEF_v2 B4). Prefer the explain document's components_legend label when present; fall back here. */
const PLAIN: Record<string, string> = {
  act: "Activity", disp: "Blow-up risk", sklv: "Skill level", sit: "Layoff and age", xtour: "Other tours", chl_total: "Base skill", location: "Course location", override: "Owner override",
  b8_delta: "Live shift", contention: "Leaderboard-position shift", mu: "Model skill", mu_live: "Live model skill", mu_tour: LABELS.thisWeekPga.short,
};
export function plainName(key: string, legend?: Record<string, { label?: string } | undefined>): string {
  return legend?.[key]?.label ?? PLAIN[key] ?? key.replaceAll("_", " ");
}
