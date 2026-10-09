import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";
import { renderView } from "./helpers/render-harness.mjs";
import { GAP_METHOD_SENTENCE, LABELS, ZERO_SENTENCE, plainName } from "../app/labels.ts";
import { pct } from "../app/explain-rules.ts";
import { checkpointBenchmarkValue, savedFieldSkill, savedSkill, signedZ } from "../app/player-profile-rules.ts";
import {
  ACCURACY_SENTENCE, COL, THIN_BADGE, benchmarkCell, buildTable, columnPlan, expandEntries, fieldMethod, reconciliationHeaders, parseInputs, residualFlag, residualText, tourMixBadge,
} from "../app/weekly-rules.ts";

const deps = { LABELS, plainName, signedZ, pct, savedFieldSkill, checkpointBenchmarkValue, savedSkill };
const read = (rel) => JSON.parse(readFileSync(new URL(rel, import.meta.url), "utf8"));

const reference = { id: "ref", status: "available", available_after: "2026-01-01T00:00:00Z", label: "r", population: "p", year: 2025, rounds: [1, 2], adjusted_sg_mean: -0.13, n_rounds: 1, n_players: 1, n_events: 1 };
const rating = (value, extra = {}) => ({ schema_version: "player_benchmark.v1", status: "available", value, n_rounds: 100, n_events: 20, n_eff: 40, weight_on_data: 1, last_observation: "2026-08-01", as_of: "2026-10-08T00:00:00Z", reference_id: "ref", unit: "strokes/round", method: {}, ...extra });
const prob = (m) => ({ model: m, market: null, fair: null, n_books: null, edge: null, rel: null, logit: null });
const mkPlayer = (id, name, mu, extra = {}) => ({ id, name, withdrawn: false, n_prior_rounds: 50, mu, components: { act: 0.1, sklv: 0.2 }, probs: { win: prob(0.05), top_5: prob(0.2), top_10: prob(0.3), top_20: prob(0.5), make_cut: prob(0.8) }, live: null, ...extra });
const doc = (kind, players) => ({ kind, as_of: "2026-10-08T12:00:00Z", event: { name: "E", tour: "pga", cut_round: 2, cut_rule: "top 65" }, components_legend: { act: { label: "Activity", group: "g", meaning: "m" } }, players });
const live = (mu_live) => ({ mu_live, b8_delta: 0.01, contention: 0.02 });
const inputs = parseInputs({ event: { field_strength: { field_offset: 0.1002, vintage: "2026-10-05 06:00:00", n_imputed: 3, n_field: 2 } }, kind: "week", players: [
  { dg_id: 1, challenger: { mu: 0.4, mu_tour: 0.5, breakdown: { chl_total: 0.3, sum_of_components: 0.3, chl_families: { act: 0.1 }, location: { total: 0.1 }, override: { total: 0 } } }, history: { chl_mu_untouched_by_location: 0.3 } },
  { dg_id: 2, challenger: { mu: -0.4, mu_tour: -0.3, breakdown: { chl_total: -0.4, sum_of_components: -0.4000001, chl_families: { act: -0.1 } } } },
] });
const catalog = { players: [
  { dg_id: 1, pga_benchmark: rating(0.9, { decomposition: { tour_weights: [{ tour: "pga", weight_fraction: 0.97 }] } }) },
  { dg_id: 2, pga_benchmark: rating(null, { status: "insufficient", n_rounds: 4, decomposition: { tour_weights: [{ tour: "euro", weight_fraction: 0.8 }, { tour: "pga", weight_fraction: 0.2 }] } }) },
], pga_benchmark_reference: reference };

test("pre-event default columns: This week, vs field, n + badge, Win, Make cut; no live or PGA-avg columns", () => {
  const plan = columnPlan("week", deps);
  assert.deepEqual(plan.defaults, [COL.player, LABELS.thisWeekPga.short, LABELS.vsField.short, COL.nTour, COL.win, COL.makeCut]);
  assert.ok(!plan.defaults.includes(LABELS.vsPgaAvg.short), "vs PGA avg stays off the default set");
  assert.ok(!plan.defaults.includes(COL.gap), "the model-minus-form gap is picker-only");
  assert.ok(plan.picker.includes(LABELS.vsPgaAvg.short) && plan.picker.includes(COL.gap));
});

test("live default columns add Live skill, B8 and Contention", () => {
  const plan = columnPlan("live", deps);
  assert.deepEqual(plan.defaults.slice(-3), [COL.liveSkill, COL.b8, COL.contention]);
  assert.equal(plan.defaults.length, 9);
});

test("the picker can reach every hidden column: vs PGA avg, gap, components, CHL families, location, override, weather, reconciliation", () => {
  const players = [mkPlayer(1, "A, A", 0.4), mkPlayer(2, "B, B", -0.4)];
  const t = buildTable({ doc: doc("week", players), inputs, catalog, players, deps });
  for (const col of [LABELS.vsPgaAvg.short, COL.gap, COL.top10, COL.base, COL.location, COL.override, COL.sum, COL.baseline, COL.residual, "Activity", "Skill level", "CHL Activity"]) {
    assert.ok(t.ordered.includes(col), `picker has ${col}`);
    assert.ok(col in t.rows[0], `row carries ${col}`);
  }
  assert.ok(!t.ordered.slice(0, 9).includes(LABELS.vsPgaAvg.short), "vs PGA avg is not among the columns DataTable opens on today");
  assert.deepEqual(t.defaults, columnPlan("week", deps).defaults);
  assert.ok(!Object.keys(t.rows[0]).some((k) => /^(act|sklv)$/.test(k)), "no raw keys as headers");
});

test("This week comes from challenger.mu_tour and reads n/a when the inputs document is missing", () => {
  const players = [mkPlayer(1, "A, A", 0.4)];
  const withInputs = buildTable({ doc: doc("week", players), inputs, catalog, players, deps });
  assert.equal(withInputs.rows[0][LABELS.thisWeekPga.short], "+0.500");
  const without = buildTable({ doc: doc("week", players), inputs: null, catalog, players, deps });
  assert.equal(without.rows[0][LABELS.thisWeekPga.short], "n/a");
});

test("live rows carry live skill, B8 and contention", () => {
  const players = [mkPlayer(1, "A, A", 0.4, { live: live(0.45) })];
  const t = buildTable({ doc: doc("live", players), inputs, catalog, players, deps });
  assert.equal(t.rows[0][COL.liveSkill], "+0.450");
  assert.equal(t.rows[0][COL.b8], "+0.010");
  assert.equal(t.rows[0][COL.contention], "+0.020");
  assert.equal(t.rows[0][COL.preEventSkill], "+0.400");
});

test("residual shows 4 dp, or a rounding note with no flag when |res| <= 1e-5", () => {
  assert.equal(residualText(0.0000001, signedZ), "<1e-5 (rounding)");
  assert.equal(residualText(-0.000009, signedZ), "<1e-5 (rounding)");
  assert.equal(residualText(0.01234, signedZ), "FLAG +0.0123");
  assert.equal(residualText(null, signedZ), "—");
  assert.equal(residualFlag(0.00001), false);
  assert.equal(residualFlag(-0.00002), true);
  const players = [mkPlayer(1, "A, A", 0.4), mkPlayer(2, "B, B", -0.4)];
  const t = buildTable({ doc: doc("week", players), inputs, catalog, players, deps });
  assert.equal(t.rows[0][COL.residual], "FLAG +0.1000"); // mu 0.4 vs component sum 0.3
  assert.equal(t.rows[1][COL.residual], "<1e-5 (rounding)"); // 1e-7 apart
});

test("a thin benchmark shows the n<12 rounds badge, never Unavailable", () => {
  assert.equal(benchmarkCell(rating(null, { status: "insufficient", n_rounds: 4 }), reference, "2026-10-09T00:00:00Z", deps), THIN_BADGE);
  assert.equal(benchmarkCell(rating(0.5, { n_rounds: 8, status: "insufficient" }), reference, "2026-10-09T00:00:00Z", deps), THIN_BADGE);
  assert.equal(benchmarkCell(undefined, reference, "2026-10-09T00:00:00Z", deps), "n/a");
  const players = [mkPlayer(2, "B, B", -0.4)];
  const t = buildTable({ doc: doc("week", players), inputs, catalog, players, deps });
  assert.equal(t.rows[0][LABELS.vsPgaAvg.short], THIN_BADGE);
  assert.ok(!JSON.stringify(t.rows).includes("Unavailable"));
});

test("vs PGA avg is gated on archived checkpoints (as-was): history newer than the checkpoint is withheld", () => {
  const current = benchmarkCell(rating(0.9), reference, "2026-10-08T12:00:00Z", deps);
  assert.equal(current, "+0.900");
  const archived = benchmarkCell(rating(0.9), reference, "2026-06-01T12:00:00Z", deps); // last observation 2026-08-01 is after this checkpoint
  assert.notEqual(archived, "+0.900");
  assert.equal(archived, "after this checkpoint");
  const early = benchmarkCell(rating(0.9), reference, "2025-12-01T00:00:00Z", deps); // reference not yet available
  assert.equal(early, "after this checkpoint");
  const players = [mkPlayer(1, "A, A", 0.4)];
  const d = { ...doc("week", players), as_of: "2026-06-01T12:00:00Z" };
  const t = buildTable({ doc: d, inputs, catalog, players, deps });
  assert.equal(t.rows[0][LABELS.vsPgaAvg.short], "after this checkpoint");
  assert.equal(t.rows[0][COL.gap], "n/a");
});

test("model-minus-form gap is This week minus vs PGA avg when both exist", () => {
  const players = [mkPlayer(1, "A, A", 0.4)];
  const t = buildTable({ doc: doc("week", players), inputs, catalog, players, deps });
  assert.equal(t.rows[0][COL.gap], "-0.400"); // 0.5 - 0.9
});

test("tour-mix badge only when the primary tour differs from the event tour or PGA share is under 50%", () => {
  assert.equal(tourMixBadge(catalog.players[0].pga_benchmark, "pga"), null);
  assert.match(tourMixBadge(catalog.players[1].pga_benchmark, "pga"), /EURO-led, PGA 20%/);
  assert.equal(tourMixBadge(undefined, "pga"), null);
  assert.equal(tourMixBadge(rating(1, { decomposition: { tour_weights: [{ tour: "pga", weight_fraction: 0.4 }, { tour: "euro", weight_fraction: 0.35 }, { tour: "kft", weight_fraction: 0.25 }] } }), "pga"), "PGA-led, PGA 40%");
});

test("field method: offset, label, estimator, vintage, n_imputed (n/a when absent) and the accuracy sentence", () => {
  const m = fieldMethod(inputs, "week");
  assert.equal(m.offsetText, "+0.100");
  assert.equal(m.offsetLabel, "field offset");
  assert.equal(m.vintage, "2026-10-05");
  assert.equal(m.imputed, "3");
  assert.match(m.estimator, /F02 theta/);
  assert.equal(fieldMethod(inputs, "live").offsetLabel, "pre-event field offset");
  const bare = fieldMethod(parseInputs({ event: {}, players: [] }), "week");
  assert.equal(bare.imputed, "n/a");
  assert.equal(bare.offsetText, "n/a");
  assert.match(ACCURACY_SENTENCE, /SD 0\.12.*up to 0\.4/);
});

test("expand card lists every column except Player, This week and Win", () => {
  const players = [mkPlayer(1, "A, A", 0.4)];
  const t = buildTable({ doc: doc("week", players), inputs, catalog, players, deps });
  const keys = expandEntries(t.rows[0], deps).map(([k]) => k);
  assert.ok(!keys.includes(COL.player) && !keys.includes(COL.win) && !keys.includes(LABELS.thisWeekPga.short));
  assert.ok(keys.includes(COL.makeCut) && keys.includes(LABELS.vsPgaAvg.short));
  assert.ok(!keys.includes("__row"));
});

// ---- rendered view over the real published live checkpoint (pga:527) ------------------------------------------------------
const publicPath = (key) => new URL(`../public/${key}`, import.meta.url);
const haveFixtures = existsSync(publicPath("golfprice/index.json"));
const profileCatalog = read("./fixtures/player_profiles/catalog.json");

test("/weekly-players renders the table through DataTable with the Method sentence, closed audit expander and live defaults", { skip: !haveFixtures }, () => {
  const index = JSON.parse(readFileSync(publicPath("golfprice/index.json"), "utf8"));
  const data = (key) => {
    if (key === "golfprice/index.json") return index;
    if (key === "golfprice/player_profiles/dossier-review.json") return profileCatalog;
    if (key === "golfprice/player_profiles/fields.json") return undefined;
    const file = publicPath(key);
    return existsSync(file) ? JSON.parse(readFileSync(file, "utf8")) : undefined;
  };
  const html = renderView("WeeklyPlayersView.tsx", "WeeklyPlayersView", { data, search: "?e=pga%3A527%3A2026-10-08", pathname: "/weekly-players" });
  assert.match(html, /data-table-wrap/, "rendered through DataTable");
  assert.match(html, /Download Weekly players CSV/);
  assert.ok(html.includes(GAP_METHOD_SENTENCE.replaceAll("'", "&#x27;")), "gap method sentence present");
  assert.match(html, /<summary>Method<\/summary>/);
  assert.ok(html.includes(ZERO_SENTENCE.slice(0, 40)));
  assert.match(html, /Pre-event field offset/);
  assert.match(html, /Saved baseline reconciliation \(audit\)/);
  assert.doesNotMatch(html, /<details[^>]*open[^>]*><summary>Saved baseline reconciliation/);
  const thead = html.match(/<thead>(.*?)<\/thead>/)[1];
  const headers = [...thead.matchAll(/<th[^>]*><button[^>]*>(.*?)<\/button>/g)].map((m) => m[1].replace(/<[^>]+>/g, ""));
  assert.deepEqual(headers, ["Player", "This week (PGA scale)", "vs field", "n (tour mix)", "Win", "Make cut", "Live skill", "B8 live shift", "Contention shift"], "live default columns, labels verbatim");
  assert.match(thead, /class="sticky-first"/);
  assert.match(thead, /class="mobile-hide"/);
  assert.equal([...thead.matchAll(/<th class="(?!mobile-hide)[^"]*"/g)].length, 3, "phone keeps Player, This week, Win");
  assert.doesNotMatch(html, /Unavailable<\/td>|>Unavailable</);
});

test("W1: reconciliation headers go through plainName, never raw keys", () => {
  const h = reconciliationHeaders(["act", "sit", "sklv", "xtour", "level_form"], doc("week", []).components_legend, plainName).map((x) => x.label);
  assert.deepEqual(h.slice(0, 4), ["Activity", "Situation", "Skill level", "Cross-tour"]);
  assert.equal(h[4], "level form");
  assert.equal(h.includes("act") || h.includes("sit") || h.includes("sklv"), false);
});
