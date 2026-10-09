import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { renderView } from "./helpers/render-harness.mjs";
import {
  barGeometry, describeFilters, liveChips, regularTick, roundSetBadge, sgBarRows, sparklinePaths, supportBadge, tourMixBadge,
  CAVEAT_REFERENCE_ONLY, NO_CATEGORY_DATA, WEEKEND_ROUNDS_NOTE,
} from "../app/player-hero-rules.ts";

const fx = (path) => JSON.parse(readFileSync(new URL(`./fixtures/${path}`, import.meta.url), "utf8"));
const catalog = fx("player_profiles/catalog.json");
const batch = fx("player_profiles/batch_0000.json");
const live = fx("explain_live_v6.json");
const blair = batch.profiles["17639"];
const blairEntry = catalog.players.find((p) => p.dg_id === 17639);
const weekly = { eventUid: null, options: [], savedDoc: null, onEvent() {} };
const hero = (over = {}) => renderView("PlayerHero.tsx", "PlayerHero", {
  props: { entry: blairEntry, profile: blair, catalog, photo: null, weekly, deep: null, deepLoading: false, deepError: null, ...over },
});

// ---- rules -----------------------------------------------------------------------------------------------------------
test("support badge: amber under 30 rounds, n<12 when unsupported, plain otherwise", () => {
  assert.deepEqual(supportBadge({ n_rounds: 158, status: "available" }), { tone: "neutral", text: "158 rounds" });
  assert.equal(supportBadge({ n_rounds: 20, status: "available" }).tone, "warning");
  assert.equal(supportBadge({ n_rounds: 29, status: "available" }).tone, "warning");
  assert.deepEqual(supportBadge({ n_rounds: 8, status: "insufficient_data" }), { tone: "negative", text: "under 12 rounds" });
  assert.equal(supportBadge(undefined).text, "under 12 rounds");
});

test("tour-mix badge only when the primary tour differs from the event tour or PGA share is under half", () => {
  const pga = [{ tour: "pga", weight_fraction: 0.97 }, { tour: "euro", weight_fraction: 0.03 }];
  assert.equal(tourMixBadge(pga, "pga"), null);
  assert.equal(tourMixBadge(pga, null), null);
  assert.match(tourMixBadge(pga, "euro"), /mostly PGA rounds \(97%\).*about 0\.2 strokes/);
  const kft = [{ tour: "kft", weight_fraction: 0.6 }, { tour: "pga", weight_fraction: 0.4 }];
  assert.match(tourMixBadge(kft, "pga"), /mostly KFT rounds/);
  assert.match(tourMixBadge(kft, null), /mostly KFT/);
  assert.equal(tourMixBadge([], "pga"), null);
  assert.equal(tourMixBadge(undefined, "pga"), null);
});

test("regular tick is read from the catalog only; absent or malformed stays absent (never computed)", () => {
  assert.equal(regularTick({}), null);
  assert.equal(regularTick({ regular_tick: {} }), null);
  assert.equal(regularTick({ regular_tick: { value: NaN } }), null);
  assert.deepEqual(regularTick({ regular_tick: { value: 0.39, as_of: "2026-10-08" } }), { value: 0.39, asOf: "2026-10-08" });
  assert.deepEqual(regularTick({ regular_tick: { unshrunk: 0.4 } }), { value: 0.4, asOf: null });
  assert.equal(regularTick(null), null);
});

test("bar geometry: zero sits mid-scale, values clamp, positives fill right of zero", () => {
  const g = barGeometry(1, 0.4);
  assert.equal(g.zero, 50);
  assert.ok(g.left === 50 && g.width > 0 && g.width < 50);
  assert.ok(g.tick > 50);
  assert.equal(barGeometry(-1, null).width, barGeometry(1, null).width);
  assert.ok(barGeometry(-1, null).left < 50);
  assert.equal(barGeometry(null, null).width, 0);
  assert.equal(barGeometry(99, null).hi, 99);
});

test("round-set badge only off the R1-R2 default; summary text describes the active filters", () => {
  assert.equal(roundSetBadge([1, 2]), null);
  assert.equal(roundSetBadge([1, 2, 3, 4]), WEEKEND_ROUNDS_NOTE);
  assert.equal(roundSetBadge([3]), WEEKEND_ROUNDS_NOTE);
  const base = { window: "recent2y", season: "", tour: "", major: false, difficulty: "", strength: "", search: "", rounds: [1, 2], contention: "all", event: "", course: "" };
  assert.equal(describeFilters(base).text, "R1-R2 · last 2 years · no situational filter");
  assert.equal(describeFilters(base).roundBadge, null);
  const all = describeFilters({ ...base, rounds: [1, 2, 3, 4] });
  assert.equal(all.text, "R1-R4 · last 2 years · no situational filter");
  assert.equal(all.roundBadge, WEEKEND_ROUNDS_NOTE);
  const sit = describeFilters({ ...base, contention: "near", rounds: [3, 4], major: true, window: "all" });
  assert.match(sit.text, /^R3-R4 · all published history · in contention \(estimated from earlier rounds\), majors only$/);
  assert.equal(sit.situational.length, 2);
});

test("live chips: transient latent, and priced strength = mu_live + week latent + contention", () => {
  assert.equal(liveChips(null), null);
  assert.equal(liveChips({ mu_live: 0.1 }), null);
  const c = liveChips({ mu_live: 0.112, week_latent_mean: 0.018, contention: null });
  assert.equal(c.transient, 0.018);
  assert.ok(Math.abs(c.priced - 0.13) < 1e-12);
  assert.equal(c.contentionIncluded, false);
  const d = liveChips({ mu_live: 0.1, week_latent_mean: 0.02, contention: -0.05 });
  assert.ok(Math.abs(d.priced - 0.07) < 1e-12);
  assert.equal(d.contentionIncluded, true);
  assert.equal(liveChips({ week_latent_mean: 0.02 }).priced, null);
  assert.ok(liveChips({ tier: 0, mu_live: 0.1, week_latent_mean: 0.02 }), "active player keeps the chips");
  for (const tier of [1, 2, 3, 4]) assert.equal(liveChips({ tier, mu_live: 0.1, week_latent_mean: 0.02 }), null, `tier ${tier} plays no next round`);
});

test("the v6 live fixture publishes week_latent_mean for every player (A7 needs no runner change)", () => {
  assert.equal(live.kind, "live");
  assert.ok(live.players.length > 0 && live.players.every((p) => Number.isFinite(p.live.week_latent_mean) && "mu_live" in p.live && "contention" in p.live));
});

test("SG bar rows: needs 3 rounds with a Z; availability false when no axis has data", () => {
  const none = sgBarRows(["a", "b", "c", "d"].map((key) => ({ key, label: key, value: null, z: null, n: 0 })));
  assert.equal(none.available, false);
  assert.ok(none.rows.every((r) => r.status === "none"));
  const some = sgBarRows([{ key: "a", label: "A", value: 0.5, z: 1, n: 10 }, { key: "b", label: "B", value: 0.1, z: null, n: 2 }, { key: "c", label: "C", value: null, z: null, n: 0 }]);
  assert.equal(some.available, true);
  assert.deepEqual(some.rows.map((r) => r.status), ["ok", "thin", "none"]);
  assert.ok(some.rows[0].width > 0 && some.rows[1].width === 0);
});

test("sparkline needs two points and at least one finite SMA", () => {
  assert.equal(sparklinePaths([]), null);
  assert.equal(sparklinePaths([{ time: 1, sma20: null, sma50: null }, { time: 2, sma20: null, sma50: null }]), null);
  const s = sparklinePaths([{ time: 1, sma20: -0.5, sma50: null }, { time: 2, sma20: 0.5, sma50: 0.1 }]);
  assert.match(s.sma20, /^M0\.0,.*L180\.0,/);
  assert.ok(s.zero !== null);
});

// ---- rendered hero ---------------------------------------------------------------------------------------------------------
test("hero renders the headline value, caveat rows 1 and 2, the zero sentence, support badge and no tick without a stored tick", () => {
  const html = hero();
  assert.match(html, /data-testid="player-hero"/);
  assert.match(html, /\+0\.73 <small/);
  assert.match(html, /title="\+0\.728 strokes per round"/);
  assert.match(html, /vs PGA avg/);
  assert.match(html, /vs 2025 PGA avg round/);
  assert.ok(html.includes(CAVEAT_REFERENCE_ONLY));
  assert.match(html, /Raw average; not adjusted for small samples/);
  assert.match(html, /Zero is the average round on the 2025 PGA Tour \(opening two rounds/);
  assert.match(html, /158 rounds/);
  assert.equal(html.includes("regular-tick"), false);
  assert.equal(html.includes("typical regular"), false);
  assert.match(html, /Zac|Blair/);
});

test("hero shows the stored typical-regular tick only when the catalog carries one", () => {
  const html = hero({ catalog: { ...catalog, regular_tick: { value: 0.39, as_of: "2026-10-08" } } });
  assert.match(html, /data-testid="regular-tick"/);
  assert.match(html, /typical regular \+0\.39/);
});

test("hero badges: amber thin-sample, n<12 unsupported, and tour mix only when it applies", () => {
  const thin = hero({ profile: { ...blair, pga_benchmark: { ...blair.pga_benchmark, n_rounds: 20 } } });
  assert.match(thin, /tone-warning[^>]*>20 rounds · thin sample/);
  const unsupported = hero({ profile: { ...blair, pga_benchmark: { ...blair.pga_benchmark, n_rounds: 5, status: "insufficient_data", value: null } } });
  assert.match(unsupported, /under 12 rounds/);
  assert.match(unsupported, /Unavailable/);
  assert.equal(unsupported.includes('data-testid="regular-tick"'), false);
  assert.equal(hero().includes("mostly"), false); // Blair: 67% PGA, event tour unknown
  assert.match(hero({ weekly: { ...weekly, eventTour: "euro" } }), /mostly PGA rounds \(67%\)/);
  assert.match(hero({ profile: { ...blair, pga_benchmark: { ...blair.pga_benchmark, decomposition: { ...blair.pga_benchmark.decomposition, tour_weights: [{ tour: "kft", n_rounds: 90, weight_fraction: 0.8 }, { tour: "pga", n_rounds: 20, weight_fraction: 0.2 }] } } } }), /mostly KFT rounds \(80%\)/);
});

test("as-was gate: an archived checkpoint before the last observation withholds the number; a later one shows it", () => {
  const early = hero({ checkpoint: "2026-08-01T00:00:00Z" });
  assert.match(early, /Withheld/);
  assert.match(early, /Withheld for this old checkpoint: the profile is rebuilt from the latest data/);
  assert.equal(early.includes("+0.73 <small"), false);
  const late = hero({ checkpoint: "2026-10-08T12:00:00Z" });
  assert.match(late, /\+0\.73 <small/);
  assert.equal(late.includes("not as-was"), false);
});

test("trend teaser: skeleton while the deep batch loads, honest note when it is missing", () => {
  assert.match(hero({ deepLoading: true }), /data-testid="trend-skeleton"/);
  assert.match(hero({ deepError: "Data request failed (404)" }), /data-testid="trend-unavailable"[\s\S]*Round-by-round history did not load/);
  assert.equal(hero().includes('data-testid="trend-teaser"'), false);
});

function deepWithRounds(n) {
  const rounds = Array.from({ length: n }, (_, i) => ({ round: (i % 2) + 1, date: `2026-0${1 + (i % 9)}-${10 + (i % 15)}`, sg_basis: "source_adjusted", sg_total: 0.5 + (i % 5) * 0.2, sg_ott: 0.3, sg_app: 0.2, sg_arg: 0.1, sg_putt: -0.2 }));
  return { as_of: "2026-10-08T00:00:00Z", history: { events: Array.from({ length: n }, (_, i) => ({ id: `e${i}`, name: `Event ${i}`, date: `2026-0${1 + (i % 9)}-${10 + (i % 15)}`, year: 2026, tour: "pga", round_data: [rounds[i]] })) } };
}
test("trend teaser renders a sparkline linking to the full chart once 20+ rounds exist", () => {
  const html = hero({ deep: deepWithRounds(60) });
  assert.match(html, /data-testid="trend-teaser"/);
  assert.match(html, /href="#player-history"/);
  assert.match(html, /20-round [+-]/);
});

test("week chips: vs field, win and top 10 from the saved doc; live checkpoints add the A7 chips", () => {
  const p = live.players.find((q) => q.live.week_latent_mean !== null);
  const entry = { ...blairEntry, dg_id: p.id };
  const profile = { ...blair, identity: { ...blair.identity, dg_id: p.id } };
  const html = hero({ entry, profile, weekly: { ...weekly, eventUid: live.event_uid, eventName: live.event.name, eventTour: "pga", savedDoc: live } });
  assert.match(html, /vs field/);
  assert.match(html, /win/);
  assert.match(html, /top 10/);
  assert.match(html, /data-testid="live-chips"/);
  assert.match(html, /skill for next round/);
  assert.match(html, /Measured against everyone who started, not only players still in the field/);
  const week = hero({ entry, profile, weekly: { ...weekly, savedDoc: { ...live, kind: "week" } } });
  assert.equal(week.includes("live-chips"), false);
});

test("hero without a saved field gives a next action instead of a dead end", () => {
  const html = hero();
  assert.match(html, /data-testid="week-absent"/);
  assert.match(html, /Show the full field/);
  const absent = hero({ weekly: { ...weekly, savedDoc: { ...live, players: [] } } });
  assert.match(absent, /Not in this week&#x27;s field/);
});

// ---- SG bars -----------------------------------------------------------------------------------------------------------------
const axisRef = { schema_version: "observed_category_reference.v1", axes: Object.fromEntries(["sg_ott", "sg_app", "sg_arg", "sg_putt"].map((k) => [k, { mean: 0, sd: 0.5 }])) };
const bars = (over = {}) => renderView("PlayerSgBars.tsx", "PlayerSgBars", { props: { profile: blair, deep: null, loading: false, categoryReference: axisRef, tours: ["kft", "euro"], ...over } });
test("SG bars: n/a state is explicit and names the tours (no category data)", () => {
  const html = bars();
  assert.match(html, /data-testid="sg-na"/);
  assert.ok(html.includes(NO_CATEGORY_DATA));
  assert.match(html, /KFT, EURO/);
  assert.equal(html.includes("sg-fill"), false);
});
test("SG bars: n/a names the real cause (PGA-covered player: reference, deep history, or too few rounds)", () => {
  const pga = bars({ tours: ["pga"], deep: deepWithRounds(1), categoryReference: undefined });
  assert.match(pga, /PGA comparison averages were not published/);
  assert.equal(pga.includes("this player&#x27;s rounds are") || pga.includes("published for PGA and LIV rounds only"), false);
  assert.match(bars({ tours: ["pga"] }), /round-by-round history has not loaded/);
  const thin = bars({ tours: ["pga"], deep: { ...deepWithRounds(24), history: { events: [] } } });
  assert.match(thin, /Fewer than 3 rounds with a strokes-gained breakdown/);
});
test("SG bars: skeleton while loading; bars with Z and rounds when category data exists; no window toggle", () => {
  assert.match(bars({ loading: true }), /data-testid="sg-skeleton"/);
  const html = bars({ deep: deepWithRounds(24) });
  assert.match(html, /data-testid="sg-bars"/);
  assert.match(html, /Off the tee/);
  assert.match(html, /Putting/);
  assert.match(html, /ph-sg-fill/);
  assert.equal(/<select|toggle/i.test(html), false);
  assert.match(html, /Rounds 1 and 2 over the last 2 years/);
});

test("W1: layout guards - hero identity keeps a minimum column; legacy header row wraps on phones", async () => {
  const { readFileSync: rf } = await import("node:fs");
  const hero = rf(new URL("../app/player-hero.css", import.meta.url), "utf8");
  assert.match(hero, /\.ph-identity\{grid-template-columns:auto minmax\(180px,1fr\) auto/);
  assert.match(hero, /@media\(max-width:1100px\)\{\.ph-stats\{grid-template-columns:minmax\(0,1fr\)\}/);
  const shell = rf(new URL("../app/shell.css", import.meta.url), "utf8");
  assert.match(shell, /\.topbar \.event-context \{[^}]*flex-wrap: wrap/);
});

test("week chip shows saved model skill on the PGA scale (mu_tour) with its label; live variant; fallback keeps vs field", () => {
  const p = live.players.find((q) => q.live.week_latent_mean !== null);
  const entry = { ...blairEntry, dg_id: p.id };
  const profile = { ...blair, identity: { ...blair.identity, dg_id: p.id } };
  const base = { ...weekly, eventUid: live.event_uid, eventName: live.event.name, eventTour: "pga" };
  const week = hero({ entry, profile, weekly: { ...base, savedDoc: { ...live, kind: "week" }, pgaSkill: { value: 0.851, offset: 0.1002, live: false, preEvent: 0.851 } } });
  assert.match(week, /data-testid="pga-chip"/);
  assert.match(week, /<b>\+0\.85<\/b><small>This week \(PGA scale\)<\/small>/);
  assert.match(week, /saved skill plus this field&#x27;s offset to the PGA scale \(\+0\.100, how this field compares with a normal PGA field\)/);
  assert.match(week, /win/); assert.match(week, /top 10/);
  const lv = hero({ entry, profile, weekly: { ...base, savedDoc: live, pgaSkill: { value: 0.7, offset: 0.1002, live: true, preEvent: 0.851 } } });
  assert.match(lv, /This week \(PGA scale\), live[\s\S]*Live skill plus the pre-event field offset/);
  assert.match(lv, /The pre-event value was \+0\.851/);
  const fb = hero({ entry, profile, weekly: { ...base, savedDoc: { ...live, kind: "week" } } });
  assert.equal(fb.includes("pga-chip"), false);
  assert.match(fb, /vs field/);
});
