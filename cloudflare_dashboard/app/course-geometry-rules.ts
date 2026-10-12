/**
 * Course geometry and player fit ("course_geometry.v1" / "course_fit.v1", published by golfprice/course_geometry.py + course_fit_traits.py).
 * Pure functions only (no imports) so node's strip-types test loader can run them; CourseGeometry.tsx draws what these return.
 *
 * The publisher writes every label, sentence and badge text; this file never rewrites them, never recomputes a percentile or a threshold,
 * and never prices anything. Everything is descriptive. A missing pointer or a payload that fails validation yields null: the card stays hidden.
 */
export const GEOMETRY_INDEX_KEY = "golfprice/course_geometry/index.json";
export const PREFIX_ROOT = "golfprice/course_geometry/geometry-";
export const DESCRIPTIVE_FALLBACK = "Descriptive, not priced: nothing here moves a price or a bet.";
export const BADGE_NOTE = "descriptive, not priced";

export type Flag = "weak" | "proxy" | null;
export type Grade = "A" | "B" | "C";
export type VenueEntry = { key: string; name: string | null; course_name: string | null; event_ids: string[]; event_uids: string[]; coverage: "full" | "yardage_only"; editions: number; coord_editions: number };
export type Current = { status: string | null; event_uid: string | null; event_id: string | null; name: string | null; venue_uid: string | null; venue_uids: string[]; fit_keys: Record<string, string>; course_name: string | null; ambiguous: boolean };
export type GeoIndex = { current: Current | null; by_event: Record<string, string[]>; generation: string; prefix: string; reference_key: string; traits_key: string | null; as_of: string; descriptive: string; venues: Record<string, VenueEntry>; fits: Record<string, string> };
export type Feature = { key: string; label: string; unit: string; display: number | null; pct: number; z: number | null; n_editions: number; evidence: Grade; flag: Flag; text: string };
export type Card = { venue_uid: string; course_name: string | null; event_name: string; as_of: string; editions: number; coord_editions: number; seasons: number[]; coverage: "full" | "yardage_only"; par: number | null; yardage: number | null; note: string | null; features: Feature[] };
export type Badge = { pair: string; grade: Grade; kind: "fit" | "misfit"; label: string; score: number; trait_z: number | null; trait_n: number | null; venue_z: number | null; hidden: boolean; est_strokes_round: number | null };
export type TraitValue = { z: number | null; n: number; v: number | null };
export type FitPlayer = { dg_id: number; name: string | null; traits: Record<string, TraitValue>; badges: Badge[]; hidden_badges: Badge[]; note: string | null };
export type PairInfo = { label: string; grade: Grade; why: string; hidden_default: boolean };
export type Fit = { venue_uid: string; course_name: string | null; event_name: string; as_of: string; coverage: "full" | "yardage_only"; descriptive: string; player_source: string; pairs: Record<string, PairInfo>; players: FitPlayer[] };
export type TraitDef = { key: string; label: string; unit: string; min_n: number };
export type Reference = { grades: Record<string, string>; traits: TraitDef[]; pairs: Array<{ key: string; label: string; grade: Grade; why: string }> };
export type TraitsDoc = { as_of: string; descriptive: string; players: Record<string, { name: string | null; t: Record<string, TraitValue> }> };

const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const str = (v: unknown): string | null => (typeof v === "string" && v.length ? v : null);
const grade = (v: unknown): Grade | null => (v === "A" || v === "B" || v === "C" ? v : null);
const coverage = (v: unknown): "full" | "yardage_only" | null => (v === "full" || v === "yardage_only" ? v : null);
const strList = (v: unknown): string[] => (Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : []);
const safeKey = (k: unknown, prefix: string): string | null => (typeof k === "string" && k.startsWith(prefix) && !k.includes("..") && k.endsWith(".json") ? k : null);

/* ------------------------------------------------------------------ parsing */
export function parseGeoIndex(raw: unknown): GeoIndex | null {
  if (!isObj(raw) || raw.schema_version !== "course_geometry.index.v1") return null;
  const prefix = raw.prefix;
  if (typeof prefix !== "string" || !prefix.startsWith(PREFIX_ROOT) || prefix.includes("..") || !prefix.endsWith("/")) return null;
  const reference_key = safeKey(raw.reference_key, prefix);
  if (!reference_key || !isObj(raw.venues)) return null;
  const venues: Record<string, VenueEntry> = {};
  for (const [uid, v] of Object.entries(raw.venues)) {
    if (!isObj(v)) continue;
    const key = safeKey(v.key, prefix), cov = coverage(v.coverage);
    if (!key || !cov) continue;
    venues[uid] = { key, name: str(v.name), course_name: str(v.course_name), event_ids: strList(v.event_ids), event_uids: strList(v.event_uids), coverage: cov, editions: num(v.editions) ?? 0, coord_editions: num(v.coord_editions) ?? 0 };
  }
  const fits: Record<string, string> = {};
  if (isObj(raw.fits)) for (const [uid, k] of Object.entries(raw.fits)) { const key = safeKey(k, prefix); if (key) fits[uid] = key; }
  const by_event: Record<string, string[]> = {};
  if (isObj(raw.by_event)) for (const [k, v] of Object.entries(raw.by_event)) by_event[k] = strList(v);
  let current: Current | null = null;
  if (isObj(raw.current)) {
    const c = raw.current;
    const fit_keys: Record<string, string> = {};
    if (isObj(c.fit_keys)) for (const [uid, k] of Object.entries(c.fit_keys)) { const key = safeKey(k, prefix); if (key) fit_keys[uid] = key; }
    current = { status: str(c.status), event_uid: str(c.event_uid), event_id: str(c.event_id), name: str(c.name), venue_uid: str(c.venue_uid), venue_uids: strList(c.venue_uids), fit_keys, course_name: str(c.course_name), ambiguous: c.ambiguous === true };
  }
  return { current, by_event, generation: str(raw.generation) ?? "", prefix, reference_key, traits_key: safeKey(raw.traits_key, prefix), as_of: str(raw.as_of) ?? "", descriptive: str(raw.descriptive) ?? DESCRIPTIVE_FALLBACK, venues, fits };
}

/** Series id of an event_uid: "pga:528:2026-10-22" -> "pga:528". */
export const seriesOf = (eventUid: string): string => eventUid.split(":").slice(0, 2).join(":");

/**
 * Courses for the page's event_uid, by stable ids only (names are display-only). Order: the publisher's current event (one named course, or
 * all courses of a multi-course event when it did not name one); any venue listing this exact event_uid; the series id only when it maps
 * to exactly one venue. A rotating course with an unknown edition, or an unknown event: empty (the card hides).
 */
export function venuesForEvent(index: GeoIndex | null, eventUid: string | null | undefined): string[] {
  if (!index || !eventUid) return [];
  const known = (uids: string[]) => [...new Set(uids)].filter((u) => !!index.venues[u]);
  const cur = index.current;
  if (cur && cur.event_uid === eventUid) {
    if (cur.status === "no_venue_card") return [];
    if (cur.venue_uid) return known([cur.venue_uid]);
    return known(cur.venue_uids);
  }
  const exact = Object.entries(index.venues).filter(([, v]) => v.event_uids.includes(eventUid)).map(([uid]) => uid);
  if (exact.length === 1) return exact;
  if (exact.length > 1) return [];
  const mapped = index.by_event[eventUid];
  if (mapped && mapped.length === 1 && index.venues[mapped[0]]) return [mapped[0]];
  const series = index.by_event[seriesOf(eventUid)];
  return series && series.length === 1 && index.venues[series[0]] ? [series[0]] : [];
}

/** Single-course lookup: the one course, or null when the event has none or several (use venuesForEvent for the switcher). */
export function venueForEvent(index: GeoIndex | null, eventUid: string | null | undefined): string | null {
  const list = venuesForEvent(index, eventUid);
  return list.length === 1 ? list[0] : null;
}

/** Fit-list key for a course: the pointer's `fits`, else the current event's per-course `fit_keys`. */
export function fitKeyFor(index: GeoIndex | null, venueUid: string | null | undefined): string | null {
  if (!index || !venueUid) return null;
  return index.fits[venueUid] ?? index.current?.fit_keys[venueUid] ?? null;
}

/** Selected course from a list: the chosen one if still present, else the first. */
export function pickVenue(uids: string[], chosen: string | null | undefined): string | null {
  return chosen && uids.includes(chosen) ? chosen : uids[0] ?? null;
}

/** Switcher label for a course tab, and the heading over several courses. */
export function courseLabel(index: GeoIndex, uid: string): string {
  const v = index.venues[uid];
  return v?.course_name ?? v?.name ?? uid;
}
export const playedOverText = (n: number): string => `Played over ${n} courses`;

function parseFeature(raw: unknown): Feature | null {
  if (!isObj(raw)) return null;
  const key = str(raw.key), label = str(raw.label), text = str(raw.text), ev = grade(raw.evidence), pct = num(raw.pct);
  if (!key || !label || !text || !ev || pct === null || pct < 0 || pct > 100) return null;
  const flag: Flag = raw.flag === "weak" || raw.flag === "proxy" ? raw.flag : null;
  return { key, label, unit: str(raw.unit) ?? "", display: num(raw.display), pct, z: num(raw.z), n_editions: num(raw.n_editions) ?? 0, evidence: ev, flag, text };
}

export function parseCard(raw: unknown, venueUid: string): Card | null {
  if (!isObj(raw) || raw.schema_version !== "course_geometry.v1" || raw.venue_uid !== venueUid) return null;
  const cov = coverage(raw.coverage);
  if (!cov || !Array.isArray(raw.features)) return null;
  const features = raw.features.map(parseFeature).filter((f): f is Feature => f !== null);
  if (!features.length) return null;
  return {
    venue_uid: venueUid, course_name: str(raw.course_name), event_name: str(raw.event_name) ?? "", as_of: str(raw.as_of) ?? "", editions: num(raw.editions) ?? 0, coord_editions: num(raw.coord_editions) ?? 0,
    seasons: Array.isArray(raw.seasons) ? raw.seasons.filter((s): s is number => Number.isInteger(s)) : [], coverage: cov, par: num(raw.par), yardage: num(raw.yardage), note: str(raw.note), features,
  };
}

function parseBadge(raw: unknown): Badge | null {
  if (!isObj(raw)) return null;
  const g = grade(raw.grade), label = str(raw.label), pair = str(raw.pair), score = num(raw.score);
  if (!g || !label || !pair || score === null || (raw.kind !== "fit" && raw.kind !== "misfit")) return null;
  return { pair, grade: g, kind: raw.kind, label, score, trait_z: num(raw.trait_z), trait_n: num(raw.trait_n), venue_z: num(raw.venue_z), hidden: raw.hidden === true, est_strokes_round: num(raw.est_strokes_round) };
}

function parseTraitMap(raw: unknown): Record<string, TraitValue> {
  const out: Record<string, TraitValue> = {};
  if (!isObj(raw)) return out;
  for (const [k, t] of Object.entries(raw)) if (isObj(t)) out[k] = { z: num(t.z), n: num(t.n) ?? 0, v: num(t.v) };
  return out;
}

export function parseFit(raw: unknown, venueUid: string): Fit | null {
  if (!isObj(raw) || raw.schema_version !== "course_fit.v1" || raw.venue_uid !== venueUid || !Array.isArray(raw.players) || !isObj(raw.pairs)) return null;
  const cov = coverage(raw.coverage);
  if (!cov) return null;
  const pairs: Record<string, PairInfo> = {};
  for (const [k, p] of Object.entries(raw.pairs)) if (isObj(p) && grade(p.grade) && str(p.label)) pairs[k] = { label: p.label as string, grade: p.grade as Grade, why: str(p.why) ?? "", hidden_default: p.hidden_default === true };
  const players: FitPlayer[] = [];
  for (const p of raw.players) {
    if (!isObj(p) || !Number.isSafeInteger(p.dg_id)) continue;
    const list = (v: unknown) => (Array.isArray(v) ? v.map(parseBadge).filter((b): b is Badge => b !== null) : []);
    players.push({ dg_id: p.dg_id as number, name: str(p.name), traits: parseTraitMap(p.traits), badges: list(p.badges), hidden_badges: list(p.hidden_badges), note: str(p.note) });
  }
  return { venue_uid: venueUid, course_name: str(raw.course_name), event_name: str(raw.event_name) ?? "", as_of: str(raw.as_of) ?? "", coverage: cov, descriptive: str(raw.descriptive) ?? DESCRIPTIVE_FALLBACK, player_source: str(raw.player_source) ?? "", pairs, players };
}

export function parseReference(raw: unknown): Reference | null {
  if (!isObj(raw) || raw.schema_version !== "course_fit.v1") return null;
  const grades: Record<string, string> = {};
  if (isObj(raw.grades)) for (const [k, v] of Object.entries(raw.grades)) if (typeof v === "string") grades[k] = v;
  const traits: TraitDef[] = Array.isArray(raw.traits) ? raw.traits.flatMap((t) => (isObj(t) && str(t.key) && str(t.label) ? [{ key: t.key as string, label: t.label as string, unit: str(t.unit) ?? "", min_n: num(t.min_n) ?? 0 }] : [])) : [];
  const pairs = Array.isArray(raw.pairs) ? raw.pairs.flatMap((p) => (isObj(p) && str(p.key) && grade(p.grade) ? [{ key: p.key as string, label: str(p.label) ?? "", grade: p.grade as Grade, why: str(p.why) ?? "" }] : [])) : [];
  return { grades, traits, pairs };
}

export function parseTraits(raw: unknown): TraitsDoc | null {
  if (!isObj(raw) || raw.schema_version !== "course_fit.v1" || !isObj(raw.players)) return null;
  const players: TraitsDoc["players"] = {};
  for (const [id, p] of Object.entries(raw.players)) if (isObj(p)) players[id] = { name: str(p.name), t: parseTraitMap(p.t) };
  return { as_of: str(raw.as_of) ?? "", descriptive: str(raw.descriptive) ?? DESCRIPTIVE_FALLBACK, players };
}

/* ------------------------------------------------------------------ card */
/** Main bars in display order; the rest sit behind "more measures". tee_bk is selection-contaminated and is shown only on request. */
export const MAIN_KEYS = ["par4_len", "dog_L", "dog_R", "tee_wo", "app_wo", "long_app", "rough_sev", "green_area"];
export const MORE_KEYS = ["dog_net", "dog_abs", "fw_hit", "tee_bk"];

/** Meaning of the two flags (the publisher only sends the flag name). */
export const FLAG_HELP: Record<string, { tag: string; help: string }> = {
  weak: { tag: "weak", help: "Weak: edition-to-edition noise is as large as the difference between courses, so treat this bar as a rough hint." },
  proxy: { tag: "proxy", help: "Proxy: estimated from where shots finished, not measured on the course." },
};

export type BarRow = { key: string; label: string; pct: number; text: string; evidence: Grade; flag: Flag; muted: boolean; editions: number; ordinal: string };

export function ordinalPercentile(p: number): string {
  const n = Math.round(p), mod100 = n % 100, mod10 = n % 10;
  const suffix = mod100 >= 11 && mod100 <= 13 ? "th" : mod10 === 1 ? "st" : mod10 === 2 ? "nd" : mod10 === 3 ? "rd" : "th";
  return `${n}${suffix} percentile of venues`;
}

export function cardRows(card: Card, keys: string[] = MAIN_KEYS): BarRow[] {
  const by = new Map(card.features.map((f) => [f.key, f]));
  const rows: BarRow[] = [];
  for (const k of keys) {
    const f = by.get(k);
    if (!f) continue;
    rows.push({ key: f.key, label: f.label, pct: f.pct, text: f.text, evidence: f.evidence, flag: f.flag, muted: f.flag !== null, editions: f.n_editions, ordinal: ordinalPercentile(f.pct) });
  }
  return rows;
}

/** A card whose coverage is yardage-only shows the note and the par-4 length bar only, whatever else the payload carries. */
export function cardView(card: Card): { fallback: boolean; main: BarRow[]; more: BarRow[]; summary: string } {
  const fallback = card.coverage === "yardage_only";
  const main = cardRows(card, fallback ? ["par4_len"] : MAIN_KEYS);
  const more = fallback ? [] : cardRows(card, MORE_KEYS);
  const eds = card.editions === 1 ? "1 edition" : `${card.editions} editions`;
  const summary = fallback ? `${eds}, no shot coordinates published` : `${eds}, ${card.coord_editions} with shot coordinates`;
  return { fallback, main, more, summary };
}

/* ------------------------------------------------------------------ fit badges */
export function displayName(name: string | null | undefined): string {
  if (!name) return "Unknown golfer";
  const i = name.indexOf(",");
  return i < 0 ? name.trim() : `${name.slice(i + 1).trim()} ${name.slice(0, i).trim()}`.trim();
}

/** Grade C is noisy: it is never shown without the toggle, even if the publisher listed it among the visible badges. */
export function visibleBadges(p: FitPlayer, showNoisy: boolean): Badge[] {
  const all = [...p.badges, ...p.hidden_badges];
  return all.filter((b) => (showNoisy ? true : b.grade !== "C" && !b.hidden)).sort((a, b) => Math.abs(b.score) - Math.abs(a.score));
}

export function hasNoisy(fit: Fit | null): boolean {
  return !!fit && fit.players.some((p) => [...p.badges, ...p.hidden_badges].some((b) => b.grade === "C" || b.hidden));
}

/** Strokes per round appear only for grade A (the one pair that survived a walk-forward test). */
export function strokesText(b: Badge): string | null {
  if (b.grade !== "A" || b.est_strokes_round === null) return null;
  const v = b.est_strokes_round;
  return `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(2)} strokes/round`;
}

export function badgeMeta(b: Badge): string {
  const n = b.trait_n === null ? "sample size unknown" : `n ${b.trait_n.toLocaleString("en-US")}`;
  return `Grade ${b.grade} · ${n}`;
}

export type FitEntry = { dg_id: number; name: string; badge: Badge; others: number };
/** Compact lists: each golfer once, under the side of his strongest visible badge. */
export function fitLists(fit: Fit, showNoisy: boolean, limit = 6): { fits: FitEntry[]; misfits: FitEntry[]; nFits: number; nMisfits: number } {
  const fits: FitEntry[] = [], misfits: FitEntry[] = [];
  for (const p of fit.players) {
    const vis = visibleBadges(p, showNoisy);
    if (!vis.length) continue;
    const entry = { dg_id: p.dg_id, name: displayName(p.name), badge: vis[0], others: vis.length - 1 };
    (vis[0].kind === "fit" ? fits : misfits).push(entry);
  }
  const byStrength = (a: FitEntry, b: FitEntry) => Math.abs(b.badge.score) - Math.abs(a.badge.score);
  fits.sort(byStrength); misfits.sort(byStrength);
  return { fits: fits.slice(0, limit), misfits: misfits.slice(0, limit), nFits: fits.length, nMisfits: misfits.length };
}

export function playerCoverage(fit: Fit): string {
  return fit.player_source === "field" ? "Entered field: golfers entered in this event." : fit.player_source === "top_by_skill" ? "Top 40 by skill: the 40 strongest golfers by current model strength, not the entered field." : "";
}

export type TraitRow = { key: string; label: string; unit: string; value: number; z: number; n: number };
/** Reliable traits only: z and v exist only when the sample passed the publisher's gate. The rest are counted, not shown. */
export function reliableTraits(traits: Record<string, TraitValue>, reference: Reference | null): { rows: TraitRow[]; withheld: number } {
  const defs = reference?.traits ?? [];
  const rows: TraitRow[] = [];
  let withheld = 0;
  for (const d of defs) {
    const t = traits[d.key];
    if (!t) continue;
    if (t.z === null || t.v === null) { withheld++; continue; }
    rows.push({ key: d.key, label: d.label, unit: d.unit, value: t.v, z: t.z, n: t.n });
  }
  return { rows, withheld };
}

export function traitValueText(r: TraitRow): string {
  const digits = Math.abs(r.value) < 1 ? 3 : 1;
  return `${r.value > 0 ? "+" : r.value < 0 ? "−" : ""}${Math.abs(r.value).toFixed(digits)} ${r.unit}`.trim();
}
