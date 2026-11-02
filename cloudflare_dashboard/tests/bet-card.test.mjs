import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULT_FILTERS, SCHEMA, americanOdds, minPriceText, betText, booksIn, decodeCard, filterBets, formatET, groupOf, historyText, timeAgo, totals,
} from "../app/bet-card-rules.ts";

// Bet card (owner-only page): decoding, grouping, plain-words labels, American odds, NEW badge, filters.

const row = (o) => ({
  event_uid: "pga:1:2026-10-22", event_name: "Event A", tour: "pga", tier: "SMALL STAKE", market: "matchup_72", round: 0, book: "betonline",
  player: "Hoey, Rico", opponent: "Yu, Kevin", dec: 2.13, min_dec: 1.99, p: 0.53, edge: 0.05, hist: 597, opp_hist: 440, stake_u: 1, also_at: "", key: "k", new: false, why: "w",
  first_seen: "2026-10-22T10:05:00Z", first_dec: 2.13, ...o,
});
const doc = (rows, events) => ({
  schema: SCHEMA, generated_at: "2026-10-22T10:05:00Z", notes: ["n"],
  events: events ?? [{ event_uid: "pga:1:2026-10-22", event_name: "Event A", tour: "pga", date_start: "2026-10-22", status: "priced", moment: "pre-event", coverage_fresh_owner_quotes: { "make_cut|betonline": 10, "matchup_72|bovada": 4 } }],
  rows,
});

test("decodes events, sorts LEAN IN first then edge descending, stable on ties", () => {
  const card = decodeCard(doc([
    row({ key: "a", edge: 0.2 }), row({ key: "b", tier: "LEAN IN", edge: 0.08 }), row({ key: "c", tier: "LEAN IN", edge: 0.12 }), row({ key: "d", edge: 0.2 }),
  ]));
  assert.deepEqual(card.events[0].bets.map((b) => b.key), ["c", "b", "a", "d"]);
  assert.equal(card.events[0].units, 4);
  assert.deepEqual(card.events[0].coverage, [{ market: "make_cut", book: "betonline", n: 10 }, { market: "matchup_72", book: "bovada", n: 4 }]);
  assert.deepEqual(card.notes, ["n"]);
});

test("rows attach to their own event; events with no rows and a note are kept", () => {
  const card = decodeCard(doc([row({ event_uid: "x" })], [
    { event_uid: "x", event_name: "X", status: "priced", moment: "after round 1" },
    { event_uid: "y", event_name: "Y", status: "nothing to price now", note: "off week", moment: "pre-event" },
  ]));
  assert.equal(card.events[0].bets.length, 1);
  assert.equal(card.events[1].bets.length, 0);
  assert.equal(card.events[1].note, "off week");
  assert.equal(card.events[1].status, "nothing to price now");
});

test("refuses other schemas and malformed documents in plain words", () => {
  assert.throws(() => decodeCard({ ...doc([]), schema: "golfprice.bet_card.v2" }), /not in the expected format/);
  assert.throws(() => decodeCard(null), /not in the expected format/);
  assert.throws(() => decodeCard({ schema: SCHEMA }), /not in the expected format/);
});

test("drops unusable rows but keeps good ones", () => {
  const card = decodeCard(doc([row({ key: "ok" }), row({ dec: "x" }), row({ tier: "MAYBE" }), null]));
  assert.equal(card.events[0].bets.length, 1);
});

test("plain-words bet labels", () => {
  assert.equal(betText({ market: "matchup_72", round: 0, player: "Hoey, Rico", opponent: "Yu, Kevin" }), "Rico Hoey over Kevin Yu, 72-hole");
  assert.equal(betText({ market: "matchup_round", round: 1, player: "Chatfield, Davis", opponent: "Svensson, Jesper" }), "Davis Chatfield over Jesper Svensson, R1");
  assert.equal(betText({ market: "make_cut", round: 0, player: "Kohles, Ben", opponent: null }), "Ben Kohles to make the cut");
  assert.equal(betText({ market: "mc", round: 0, player: "Kohles, Ben", opponent: null }), "Ben Kohles to miss the cut");
  assert.equal(betText({ market: "top_5", round: 0, player: "Whaley, Vince", opponent: null }), "Vince Whaley top 5");
  assert.equal(betText({ market: "top_20", round: 0, player: "Whaley, Vince", opponent: null }), "Vince Whaley top 20");
});

test("American odds from decimal", () => {
  assert.equal(americanOdds(2.13), "+113");
  assert.equal(americanOdds(2), "+100");
  assert.equal(americanOdds(1.5), "-200");
  assert.equal(americanOdds(1.91), "-110");
  assert.equal(americanOdds(1), "n/a");
});

test("NEW badge flag and first-seen fields decode", () => {
  const card = decodeCard(doc([row({ new: true, first_dec: 2.2 }), row({ key: "z" })]));
  const [a, b] = card.events[0].bets;
  assert.equal(a.isNew, true);
  assert.equal(a.firstDec, 2.2);
  assert.equal(b.isNew, false);
});

test("filters by tier, market group and book", () => {
  const card = decodeCard(doc([
    row({ key: "1", tier: "LEAN IN" }), row({ key: "2", market: "make_cut", opponent: null, book: "bovada" }), row({ key: "3", market: "top_5", opponent: null, book: "bovada", tier: "LEAN IN" }),
  ]));
  const bets = card.events[0].bets;
  assert.equal(filterBets(bets, DEFAULT_FILTERS).length, 3);
  assert.deepEqual(filterBets(bets, { ...DEFAULT_FILTERS, tier: "lean" }).map((b) => b.key), ["1", "3"]);
  assert.deepEqual(filterBets(bets, { ...DEFAULT_FILTERS, tier: "small" }).map((b) => b.key), ["2"]);
  assert.deepEqual(filterBets(bets, { ...DEFAULT_FILTERS, group: "cut" }).map((b) => b.key), ["2"]);
  assert.deepEqual(filterBets(bets, { ...DEFAULT_FILTERS, group: "top" }).map((b) => b.key), ["3"]);
  assert.deepEqual(filterBets(bets, { ...DEFAULT_FILTERS, book: "bovada" }).map((b) => b.key), ["3", "2"]);
  assert.deepEqual(booksIn(card), ["betonline", "bovada"]);
  assert.deepEqual(totals(filterBets(bets, { ...DEFAULT_FILTERS, tier: "lean" })), { bets: 2, units: 2 });
  assert.equal(groupOf("matchup_round"), "matchups");
});

test("history text and times", () => {
  assert.equal(historyText({ hist: 597, oppHist: 440, opponent: "Yu, Kevin" }), "597 rounds vs 440 for the opponent");
  assert.equal(historyText({ hist: null, oppHist: null, opponent: null }), "no history");
  assert.equal(formatET("2026-10-22T10:05:00Z"), "2026-10-22 06:05 ET");
  assert.equal(formatET("garbage"), "unknown time");
  const now = new Date("2026-10-22T12:00:00Z");
  assert.equal(timeAgo("2026-10-22T11:55:00Z", now), "5 min ago");
  assert.equal(timeAgo("2026-10-22T10:00:00Z", now), "2 h ago");
  assert.equal(timeAgo("2026-10-22T10:30:00Z", now), "1 h 30 min ago");
});


test("minimum price is never shown below the threshold (decimal up, American at or above)", () => {
  for (const t of [1.921, 2.084, 1.5, 2.0, 2.001, 3.333]) {
    const m = minPriceText(t);
    assert.ok(Number(m.dec) >= t - 1e-12, `${t} -> ${m.dec}`);
    const a = Number(m.american);
    const back = a > 0 ? 1 + a / 100 : 1 + 100 / -a;
    assert.ok(back >= t - 1e-9, `${t} -> ${m.american} = ${back}`);
  }
  assert.deepEqual(minPriceText(1.921), { dec: "1.93", american: "-108" });
  assert.deepEqual(minPriceText(2.084), { dec: "2.09", american: "+109" });
});
