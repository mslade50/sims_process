import test from "node:test";
import assert from "node:assert/strict";
import { parseMarketScoring, matchingRecord, historyRows, marketKey, signedScore } from "../app/market-scoring-rules.ts";
import { renderView } from "./helpers/render-harness.mjs";
const uid = "pga:457:2026-11-05";
const player = { id: 1, name: "Test Player", wave: "early", line: 69.5, over_decimal: 2, under_decimal: 2, under_no_vig: .5, model_mean: 70, market_mean: 69, shift: -1, quote_at: "2026-11-05T10:00:00Z" };
const book = { book: "buckeye", quoted: 1, field_players: 120, model_field_mean: 70, market_field_mean: 69, delta: -1, quote_at: player.quote_at, freshness_unknown: true, players: [player], outcome: { status: "pending", settled: 0 } };
const record = { id: "a", event_uid: uid, origin: "reconstructed", round: 1, model_run: "original", model_hash: "hash", model_as_of: "2026-11-05T09:00:00Z", capture_at: "2026-11-05T10:00:00Z", first_tee_utc: "2026-11-05T13:30:00Z", books: [book] };
const raw = { event_uid: uid, schema: "golfprice.market_scoring.v1", method: "pmf-location-multiplicative-v1", generated_at: "2026-11-05T11:00:00Z", records: [record] };

test("validates identity, method, temporal cutoff and arithmetic rather than guessing", () => {
  assert.equal(parseMarketScoring(raw, "another-event"), null);
  assert.equal(parseMarketScoring({ ...raw, method: "unknown" }, uid), null);
  for (const change of [{ capture_at: record.first_tee_utc }, { model_as_of: "2026-11-06T00:00:00Z" }, { event_uid: "other" }, { capture_at: "2026-11-05T10:00:00" }, { books: [{ ...book, delta: 5 }] }]) assert.equal(parseMarketScoring({ ...raw, records: [{ ...record, ...change }] }, uid).records.length, 0);
});
test("checkpoint binding never borrows a different forecast or round", () => {
  const d = parseMarketScoring(raw, uid);
  assert.equal(matchingRecord(d.records, "original", 1).id, "a");
  assert.equal(matchingRecord(d.records, "newer", 1), null);
  assert.equal(matchingRecord(d.records, "original", 2), null);
  assert.equal(marketKey(uid), "golfprice/market_scoring/pga_457_2026-11-05/latest.json");
});
test("history picks latest pre-tee per round/book without counting duplicate outcomes", () => {
  const d = parseMarketScoring({ ...raw, records: [record, { ...record, id: "b", capture_at: "2026-11-05T11:00:00Z" }] }, uid);
  assert.equal(matchingRecord(d.records, "original", 1).id, "b");
  assert.equal(historyRows(d.records).length, 1);
  assert.equal(historyRows(d.records)[0].record.id, "b");
  assert.equal(signedScore(-.001), "0.00");
});
test("real component renders paired forecast, unknown freshness, pending outcomes and original prices", () => {
  const html = renderView("MarketScoringPanel.tsx", "MarketScoringPanel", { data: { [marketKey(uid)]: raw }, props: { eventUid: uid, modelRun: "original", round: 1 } });
  for (const text of [/69.00/, /70.00/, /-1.00/, /Reconstructed/, /Availability unknown/, /pending/, /Test Player/, /2.000 \/ 2.000/, /50.0%/, /not independent course observations/]) assert.match(html, text);
  assert.doesNotMatch(html, /NaN|undefined|\[object Object\]/);
});
test("missing matching checkpoint stays empty while archived history remains inspectable", () => {
  const html = renderView("MarketScoringPanel.tsx", "MarketScoringPanel", { data: { [marketKey(uid)]: raw }, props: { eventUid: uid, modelRun: "newer", round: 1 } });
  assert.match(html, /No saved two-sided book comparison matches/);
  assert.doesNotMatch(html, /paired with the selected simulation/);
  assert.match(html, /Round history/);
});
test("built Worker serves the exact new R2 key through its existing read-only route", async () => {
  const worker = (await import("../dist/server/index.js")).default;
  const reads = [];
  const env = { DASHBOARD_DATA: { get: async (key) => {
    reads.push(key);
    return { httpEtag: '"fixture"', body: JSON.stringify(raw), writeHttpMetadata() {} };
  } } };
  const url = "https://golf.example/api/" + marketKey(uid);
  const ctx = { waitUntil() {}, passThroughOnException() {} };
  const response = await worker.fetch(new Request(url), env, ctx);
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("cache-control"), "no-cache");
  assert.deepEqual(await response.json(), raw);
  assert.deepEqual(reads, [marketKey(uid)]);
  const denied = await worker.fetch(new Request(url, { method: "POST" }), env, ctx);
  assert.equal(denied.status, 405);
  assert.equal(reads.length, 1);
});
