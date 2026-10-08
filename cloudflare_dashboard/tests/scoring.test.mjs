import test from "node:test";
import assert from "node:assert/strict";
import { betEv, breakEvenDecimal, curveRows, evBreakEvenShift, marketNoVig, overUnder, parseScoring, shiftedOverUnder } from "../app/scoring-rules.ts";

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
