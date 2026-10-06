import assert from "node:assert/strict";
import test from "node:test";
import { ODDS_SIGNALS_KEY, oddsStale, parseOddsSignals, pct } from "../app/odds-signals-rules.ts";

const sig = (over = {}) => ({ family: "make_cut", dg_id: 7, player: "Aaa, A", book: "bet365", dec: 1.8, p_fair: 0.6, ev: 0.08, status: "live", age_h: 0.5, source: "dg_api", ...over });
const doc = {
  schema: "golfprice.odds_signals.v1",
  generated_at: "2026-10-06T15:00:00Z",
  moment: "Tue 11:00 ET",
  note: "Provisional.",
  events: [{ event_uid: "pga:554:2026-10-08", name: "Test Open", base_run: "20261006T100000Z", base_as_of: "2026-10-06T10:00:00Z", fair_source: "challenger", n_signals: 1, n_live: 1, signals: [sig()], new: [sig()], moved: [{ ...sig({ dg_id: 8 }), ev_ref: 0.04 }], dropped: ["win|9|x"], alert: { sent: true, note: "sent" } }],
};

test("the published odds-signals document parses; the key is under golfprice/", () => {
  assert.equal(ODDS_SIGNALS_KEY, "golfprice/odds_signals/latest.json");
  const parsed = parseOddsSignals(doc);
  assert.equal(parsed.events.length, 1);
  const event = parsed.events[0];
  assert.equal(event.signals[0].ev, 0.08);
  assert.equal(event.moved[0].ev_ref, 0.04);
  assert.deepEqual(event.dropped, ["win|9|x"]);
  assert.equal(event.alert.sent, true);
});

test("anything else is null or tolerated: wrong schema, junk events, missing fields", () => {
  assert.equal(parseOddsSignals(null), null);
  assert.equal(parseOddsSignals({ schema: "other", events: [] }), null);
  assert.equal(parseOddsSignals({ schema: "golfprice.odds_signals.v1" }), null);
  const parsed = parseOddsSignals({ schema: "golfprice.odds_signals.v1", events: [null, { event_uid: "e", signals: [1, { family: "win" }] }] });
  assert.equal(parsed.events.length, 1);
  assert.equal(parsed.events[0].signals.length, 1);
  assert.equal(parsed.events[0].alert.sent, false);
});

test("staleness and percent formatting", () => {
  assert.equal(oddsStale("2026-10-06T15:00:00Z", Date.parse("2026-10-06T16:00:00Z")), false);
  assert.equal(oddsStale("2026-10-06T15:00:00Z", Date.parse("2026-10-06T19:00:00Z")), true);
  assert.equal(oddsStale("garbage", 0), true);
  assert.equal(pct(0.08), "+8.0%");
  assert.equal(pct(-0.025), "-2.5%");
});
