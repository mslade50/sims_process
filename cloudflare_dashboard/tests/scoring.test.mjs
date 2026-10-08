import test from "node:test";
import assert from "node:assert/strict";
import { betEv, breakEvenDecimal, curveRows, evBreakEvenShift, expectationBridge, marketNoVig, overUnder, parseScoring, resolvedOverShare, shiftedOverUnder, scoringEvidence } from "../app/scoring-rules.ts";

test("integer lines preserve pushes and exclude them from profit and loss", () => {
  const p = overUnder([[69, 0.2], [70, 0.3], [71, 0.25], [72, 0.25]], 70);
  assert.deepEqual(p, { over: 0.5, under: 0.2, push: 0.3 });
  assert.equal(betEv(p, "over", 2), 0.3);
  assert.equal(betEv(p, "under", 2), -0.3);
  assert.equal(breakEvenDecimal(p, "over"), 1.4);
});

test("half-point line has no push and break-even price uses opposing win chance", () => {
  const p = overUnder([[69, 0.2], [70, 0.3], [71, 0.25], [72, 0.25]], 70.5);
  assert.deepEqual(p, { over: 0.5, under: 0.5, push: 0 });
  assert.equal(breakEvenDecimal(p, "under"), 2);
  assert.ok(Math.abs(betEv(p, "over", 1.8) + 0.1) < 1e-12);
});

test("fractional sensitivity translates the PMF with explicit linear interpolation", () => {
  const p = shiftedOverUnder([[70, 0.5], [71, 0.5]], 70, 0.5);
  assert.deepEqual(p, { over: 0.75, under: 0, push: 0.25 });
  assert.deepEqual(shiftedOverUnder([[70, 0.5], [71, 0.5]], 70, -0.5), { over: 0.25, under: 0.25, push: 0.5 });
});

test("two-way market shares remove proportional overround and reject invalid odds", () => {
  const fair = marketNoVig(1.8, 2.2);
  assert.ok(Math.abs(fair.over - 0.55) < 1e-12 && Math.abs(fair.under - 0.45) < 1e-12);
  assert.equal(marketNoVig(1, 2), null);
  assert.equal(betEv({ over: 0.5, under: 0.5, push: 0 }, "over", 1), null);
});

test("PMF and CDF overlays retain player identities and exact probabilities", () => {
  const pmf = curveRows([{ id: 1, pmf: [[70, 0.25], [71, 0.75]] }, { id: 2, pmf: [[69, 0.5], [70, 0.5]] }], false);
  assert.deepEqual(pmf, [{ score: 69, p_1: 0, p_2: 50 }, { score: 70, p_1: 25, p_2: 50 }, { score: 71, p_1: 75, p_2: 0 }]);
  const cdf = curveRows([{ id: 1, pmf: [[70, 0.25], [71, 0.75]] }], true);
  assert.deepEqual(cdf, [{ score: 70, p_1: 25 }, { score: 71, p_1: 100 }]);
});

test("EV break-even shift reports a signed baseline scenario and null when outside range", () => {
  assert.ok(Math.abs(evBreakEvenShift([[70, 0.5], [71, 0.5]], 70.5, "over", 2) - 0) < 1e-8);
  assert.equal(evBreakEvenShift([[70, 1]], 70.5, "over", 1.1, 0.5), null);
});

test("parser fails closed on wrong schema, malformed PMFs, and materially unnormalized PMFs", () => {
  assert.equal(parseScoring({ schema: "golfprice.explain.v1" }), null);
  const good = {
    schema: "golfprice.scoring.v1", status: "available", round: 2, run_id: "r2", probability_basis: "model", method: "joint", n_draws: 100,
    field: { mean: 71, sd: 1, p10: 70, p50: 71, p90: 72, players: 1, active_players_min: 1, active_players_max: 1, outcome_label: "active field", quantile_curve: [[0, 69], [0.5, 71], [1, 73]] },
    baseline: { course_score: 71, common_weather: 0, source: "x", detail: null, median_hole_observations: 10, round_par: 72, arithmetic_residual: 0 },
    players: [{ id: 1, name: "Slade, M", mean: 71, sd: 1, p10: 70, p50: 71, p90: 72, pmf: [[70, 0.5], [72, 0.5]], n_draws: 100, components: [], data_depth: 0, tee_time: null }],
    confidence: { mean_interval: null, status: "not_calibrated", reasons: [], monte_carlo_se: { field: 0.1, basis: "simulation precision only" } }, drivers: [], source_notes: [], weather: { source: null, freshness: null, status: "available", applied_in_prices: true, label: "weather" },
  };
  assert.equal(parseScoring(good)?.players.length, 1);
  assert.deepEqual(parseScoring(good)?.field.quantile_curve, [[0, 69], [0.5, 71], [1, 73]]);
  assert.equal(parseScoring({ ...good, field: { ...good.field, quantile_curve: [[0.8, 70], [0.2, 72]] } }), null);
  const unavailable = { ...good, status: "unavailable", round: null, method: null, n_draws: 0, field: { mean: null, sd: null, p10: null, p50: null, p90: null, players: 0, outcome_label: "active field" }, baseline: { course_score: null, common_weather: null, source: null, detail: "draws absent", median_hole_observations: null, round_par: null, arithmetic_residual: null }, players: [], confidence: { mean_interval: null, status: "not_calibrated", reasons: ["unavailable"], monte_carlo_se: null }, source_notes: ["draws absent"] };
  assert.equal(parseScoring(unavailable)?.status, "unavailable");
  assert.equal(parseScoring({ ...unavailable, field: { ...unavailable.field, mean: 72 } }), null);
  assert.equal(parseScoring({ ...good, players: [{ ...good.players[0], pmf: [[70, 1.2]] }] }), null);
  assert.equal(parseScoring({ ...good, players: [{ ...good.players[0], pmf: [[70, "likely"]] }] }), null);
});

test("score bridge reconciles field and player means without double-counting residuals or treating missing weather as zero", () => {
  const doc = { baseline: { course_score: 71, common_weather: -0.5, arithmetic_residual: 0.1 }, drivers: [{ key: "skill", label: "Skill", mean_score_shift: 0 }, { key: "unexplained_residual", label: "Residual", mean_score_shift: 0.1 }] };
  assert.equal(expectationBridge(doc).total, 70.6);
  const player = { components: [{ key: "skill", label: "Skill", strokes: -1, note: null }, { key: "unexplained_residual", label: "Residual", strokes: 0.2, note: null }] };
  assert.equal(expectationBridge(doc, player).total, 69.7);
  assert.equal(expectationBridge({ ...doc, baseline: { ...doc.baseline, common_weather: null } }).total, null);
});

test("model-versus-market shares compare resolved outcomes at integer lines", () => {
  const p = overUnder([[69, 0.2], [70, 0.3], [71, 0.5]], 70);
  assert.ok(Math.abs(resolvedOverShare(p) - 5 / 7) < 1e-12);
  assert.equal(resolvedOverShare({ over: 0, under: 0, push: 1 }), null);
  assert.equal(resolvedOverShare(overUnder([[69, 0.2], [70, 0.3], [71, 0.5]], 70.5)), 0.5);
});


// Render the actual TSX with only the asynchronous data hook and shared shell isolated.
// No browser-size dependence: the view must name field/player chart and empty states in HTML.
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";
import { readFileSync } from "node:fs";
import { createRequire, Module } from "node:module";
import { fileURLToPath } from "node:url";
import * as scoringRules from "../app/scoring-rules.ts";
const require = createRequire(import.meta.url);
function renderedScoring({ payload, error = null, loading = false, player = false, expectedRun = "fixture" } = {}) {
  const filename = fileURLToPath(new URL("../app/ScoringView.tsx", import.meta.url));
  const source = readFileSync(filename, "utf8");
  const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022 } }).outputText;
  const module = new Module(filename);
  module.filename = filename;
  let state = 0;
  module.require = (id) => {
    if (id === "./data") return { useDashboardData: () => ({ data: payload, loading, error }) };
    if (id === "./scoring-rules") return scoringRules;
    if (id === "./explain-rules" || id === "./distributions-rules" || id.endsWith(".css")) return {};
    if (id === "react") return { ...React, useState: (initial) => { const slot = state++; return [slot === 2 && player ? false : typeof initial === "function" ? initial() : initial, () => {}]; } };
    if (id === "./components") return {
      Panel: ({ title, eyebrow, children }) => React.createElement("section", null, React.createElement("h3", null, title), React.createElement("small", null, eyebrow), children),
      EmptyState: ({ title, detail }) => React.createElement("div", { role: "status" }, title, " ", detail),
      LoadingState: ({ label }) => React.createElement("div", { role: "status" }, label),
    };
    return require(id);
  };
  module._compile(compiled, filename);
  return renderToStaticMarkup(React.createElement(module.exports.Snapshot, { runKey: "fixture.json", expectedRun, latest: false }));
}
function scoringFixture() {
  return {
    schema: "golfprice.scoring.v1", status: "available", round: 2, run_id: "fixture", probability_basis: "model", method: "paired field draws", n_draws: 100,
    field: { mean: 71, sd: 1, p10: 70, p50: 71, p90: 72, players: 1, active_players_min: 1, active_players_max: 1, outcome_label: "Active field average", quantile_curve: [[0, 69], [0.5, 71], [1, 73]] },
    baseline: { course_score: 71, common_weather: 0, source: "hole posterior", detail: null, median_hole_observations: 100, round_par: 72, arithmetic_residual: 0,
      course_provenance: { model_version: "release-fixture", layout: { source: "published prior-year card", year: 2025, current_event_confirmed: false }, general_fit: { start: 2020, end: 2025, cutoff: "2025-12-31", observations: 10000, method: "general hole prior", skill_response_cutoff: "2022-12-31" }, venue_history: { observations: 100, editions: [2024, 2025], cutoff: "2025-12-31" }, historical_weather: { status: "not_normalized", reference: "Weather present in historical outcomes", field_adjustment: "Historical field strength is not separately normalized in these hole counts", method: "Current weather uses a fitted deviation from venue-typical weather when available", note: "Historical hole outcomes are not individually weather-neutralized; weather reference and scoring anchor can differ." }, field_strength: { status: "not_calibrated", detail: "Absolute field strength is not calibrated." } } },
    players: [{ id: 1, name: "Test, Player", mean: 71, sd: 1, p10: 70, p50: 71, p90: 72, pmf: [[70, 0.5], [72, 0.5]], n_draws: 100, components: [{ key: "skill", label: "Centered skill", strokes: 0, note: null }], data_depth: 20, tee_time: null }],
    confidence: { mean_interval: null, status: "not_calibrated", reasons: ["No validated true-mean interval."], monte_carlo_se: { field: 0.1, basis: "simulation precision only" } }, drivers: [], source_notes: [], weather: { source: "forecast fixture", freshness: "2026-10-08", status: "available", applied_in_prices: true },
    round_update: { status: "not_applied", applied: false, target_round: 2, source_rounds: [1], adjustment_strokes: null, observations: 100, reason: "Residual method awaits validation", observed_rounds: [{ round: 1, field_actual: 70.2, field_expected: 71, weather_strokes: 0.1, residual: -0.9, players: 100 }] },
  };
}
test("rendered scoring separates layout, general fit, venue history, weather and unapplied actuals", () => {
  const fixture = scoringFixture();
  const html = renderedScoring({ payload: { scoring: fixture } });
  for (const label of ["General hole fit", "Own-venue hole history", "Historical weather reference", "Historical field strength is not separately normalized", "Historical hole outcomes are not individually weather-neutralized", "Observed-round baseline update", "Not applied", "Residual method awaits validation", "Skill response fit cutoff 2022-12-31", "Editions: 2024, 2025", "70.20", "Paired field quantile curve", "Main data limitation", "No validated interval for the true mean"]) assert.ok(html.includes(label), label);
  assert.match(html, /current setup unconfirmed/);
  assert.match(html, /does not move the field anchor/);
  assert.match(html, /<details class="score-details score-investigate"><summary>Compare a manual player line/);
  assert.doesNotMatch(html, /HOW CONFIDENT|Read with care|Vs your manual no-vig input/);
  assert.equal(parseScoring(fixture).confidence.mean_interval, null);
});
test("rendered legacy checkpoint keeps its stored score and does not imply later fit or update", () => {
  const fixture = scoringFixture();
  delete fixture.baseline.course_provenance;
  delete fixture.round_update;
  const html = renderedScoring({ payload: { scoring: fixture } });
  assert.match(html, /71.00/);
  assert.match(html, /original model fit/);
  assert.match(html, /has not been recalculated with a later release/);
  assert.match(html, /Application not verified/);
  assert.doesNotMatch(html, /release-fixture|2025 geometry|strokes applied to round/);
});
test("staged round actuals show unavailable expected, weather and residual without implying zero", () => {
  const fixture = scoringFixture();
  fixture.round_update = { status: "staged_not_applied", applied: false, target_round: 2, source_rounds: [1], adjustment_strokes: null, observations: 100, reason: "Shared course correction remains staged", method: "Relative player-form update; shared course correction remains staged", observed_rounds: [{ round: 1, field_actual: 70.2, field_expected: null, weather_strokes: null, residual: null, players: 100 }] };
  const html = renderedScoring({ payload: { scoring: fixture } });
  assert.match(html, /Inspect completed-round actuals/);
  assert.match(html, /Staged, not applied/);
  assert.match(html, /70\.20/);
  assert.equal((html.match(/Not available/g) ?? []).length, 3);
  const actualsTable = html.match(/<table class="score-observed-table">([\s\S]*?)<\/table>/)?.[1] ?? "";
  assert.doesNotMatch(actualsTable, /0\.000 strokes|0\.00<\/td>/);
  assert.match(html, /Relative player-form update; shared course correction remains staged/);
  assert.doesNotMatch(html, /strokes applied to round/);
});
test("applied update requires matching round, explicit status, flag and finite amount", () => {
  const fixture = scoringFixture();
  fixture.round_update = { ...fixture.round_update, status: "applied", applied: true, adjustment_strokes: -0.15 };
  assert.equal(scoringEvidence(parseScoring(fixture)).updateApplied, true);
  assert.match(renderedScoring({ payload: { scoring: fixture } }), /-0.150 strokes applied to round 2/);
  for (const patch of [{ target_round: 3 }, { applied: false }, { adjustment_strokes: "-0.15" }, { status: "unavailable" }]) {
    const doc = parseScoring({ ...fixture, round_update: { ...fixture.round_update, ...patch } });
    assert.equal(scoringEvidence(doc).updateApplied, false);
    assert.doesNotMatch(renderedScoring({ payload: { scoring: { ...fixture, round_update: { ...fixture.round_update, ...patch } } } }), /strokes applied to round/);
  }
});
test("zero history outranks layout warning and malformed optional metadata is empty-safe", () => {
  const fixture = scoringFixture();
  fixture.baseline.median_hole_observations = 0;
  assert.match(scoringEvidence(parseScoring(fixture)).warning, /Median own-venue/);
  fixture.baseline.course_provenance = "broken";
  fixture.round_update = [];
  const parsed = parseScoring(fixture);
  assert.ok(parsed);
  assert.equal(parsed.baseline.course_provenance, null);
  assert.equal(parsed.round_update, null);
  assert.match(renderedScoring({ payload: { scoring: fixture } }), /Median own-venue/);
});
test("rendered field/player distributions and data failures retain explicit meanings", () => {
  const fixture = scoringFixture();
  const playerHtml = renderedScoring({ payload: { scoring: fixture }, player: true });
  assert.match(playerHtml, /Cumulative distributions for Player Test/);
  assert.match(playerHtml, /UNIT-STAKED EXPECTED VALUE/);
  const fieldHtml = renderedScoring({ payload: { scoring: fixture } });
  assert.match(fieldHtml, /PLAYER SELECTION REQUIRED/);
  assert.match(renderedScoring({ error: "offline" }), /Scoring expectation unavailable.*offline/);
  assert.match(renderedScoring({ loading: true }), /Loading scoring expectation/);
  assert.match(renderedScoring({ payload: { scoring: fixture }, expectedRun: "different" }), /Scoring checkpoint does not match/);
  assert.match(renderedScoring({ payload: { scoring: { schema: "wrong" } } }), /Scoring expectation unavailable/);
  delete fixture.field.quantile_curve;
  assert.match(renderedScoring({ payload: { scoring: fixture } }), /Field curve unavailable/);
});
