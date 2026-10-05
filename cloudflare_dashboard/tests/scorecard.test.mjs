import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { deltaTone, intervalText, isStale, latestWeekEvents, parseScorecard, signedFmt } from "../app/scorecard-rules.ts";

const card = {
  schema: "golfprice.scorecard.v1",
  generated_at: "2026-10-05T15:00:00Z",
  headlines: ["x"],
  latest_week: { event_start: "2026-10-01", event_uids: ["pga:2:2026-10-01"] },
  events: [
    { event_uid: "pga:2:2026-10-01", name: "A", tour: "pga", event_start: "2026-10-01", status: "settled", finish_settled: true, arms: [], bets: {} },
    { event_uid: "pga:1:2026-09-24", name: "B", tour: "pga", event_start: "2026-09-24", status: "settled", finish_settled: true, arms: [], bets: {} },
  ],
  n_events_settled: 2,
  forward: { clock_start_event_uid: null, events_counted: 0, futility_check_at: 40, horizon_events: 80, progress_to_futility_check: 0, progress_to_horizon: 0, by_arm: {}, bets: {} },
};

test("parseScorecard accepts the published shape and rejects everything else without throwing", () => {
  assert.ok(parseScorecard(card));
  for (const bad of [null, undefined, 1, "x", [], {}, { ...card, schema: "other" }, { ...card, events: null }, { ...card, forward: null }, { ...card, generated_at: 5 }, { ...card, forward: { ...card.forward, by_arm: null } }]) {
    assert.equal(parseScorecard(bad), null);
  }
});

test("staleness is 8 days and an unparseable stamp counts as stale", () => {
  const now = Date.parse("2026-10-12T15:00:00Z");
  assert.equal(isStale("2026-10-05T15:00:00Z", now), false);
  assert.equal(isStale("2026-10-04T14:00:00Z", now), true);
  assert.equal(isStale("garbage", now), true);
});

test("latest week keeps only that week's events; negative deltas are the good ones", () => {
  assert.deepEqual(latestWeekEvents(card).map((e) => e.name), ["A"]);
  assert.equal(deltaTone(-0.01), "positive");
  assert.equal(deltaTone(0.01), "negative");
  assert.equal(deltaTone(null), "");
  assert.equal(signedFmt(0.0123, 3), "+0.012");
  assert.equal(signedFmt(null), "—");
  assert.equal(intervalText({ n: 1, mean: -0.02, lo: null, hi: null }), "-0.020 (too few events for an interval)");
  assert.equal(intervalText({ n: 5, mean: -0.02, lo: -0.04, hi: 0.0 }), "-0.020 (95% -0.040 to 0.000)");
});

test("the scorecard is a section inside the Performance view, not a new tab", async () => {
  const views = await readFile(new URL("../app/views.tsx", import.meta.url), "utf8");
  const app = await readFile(new URL("../app/DashboardApp.tsx", import.meta.url), "utf8");
  assert.match(views, /<GolfpriceScorecardSection \/>/);
  assert.doesNotMatch(app, /scorecard/i);
  const section = await readFile(new URL("../app/ScorecardSection.tsx", import.meta.url), "utf8");
  assert.match(section, /No scorecard published yet/);
  assert.match(section, /golfprice model scorecard/);
});
