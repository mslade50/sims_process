import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  ageText, biggestGaps, eventLede, heroFacts, lastSeenRunRef, parseExplain, plainName, previousRunRef, ptsText, relText, runsBefore, spanText, whatChanged,
} from "../app/explain-rules.ts";

const read = async (n) => JSON.parse(await readFile(new URL(`./fixtures/${n}`, import.meta.url), "utf8"));
const weekRaw = await read("explain_week.json");
const liveRaw = await read("explain_live.json");
const clone = (o) => JSON.parse(JSON.stringify(o));

/** A "current" week document derived from the fixture, one hour later than the fixture, with chosen edits applied to its players/course card. */
function pair(edit = () => {}) {
  const prev = parseExplain(clone(weekRaw));
  const raw = clone(weekRaw);
  raw.run = "20261001T110500Z";
  raw.as_of = "2026-10-01T11:05:00Z";
  edit(raw);
  return { prev, cur: parseExplain(raw) };
}
const byId = (raw, i) => raw.players[i];

test("age and span text", () => {
  assert.equal(ageText(null), null);
  assert.equal(ageText(30_000), "just now");
  assert.equal(ageText(-5000), "just now");
  assert.equal(ageText(12 * 60_000), "12 min ago");
  assert.equal(ageText(3 * 3600_000), "3 h ago");
  assert.equal(ageText(3 * 86400_000), "3 days ago");
  assert.equal(spanText(5 * 3600_000), "5 h");
  assert.equal(spanText(10_000), "moments");
});

test("names read First Last", () => {
  assert.equal(plainName("Scheffler, Scottie"), "Scottie Scheffler");
  assert.equal(plainName("Tiger Woods"), "Tiger Woods");
});

test("previous run and last-seen run come from the index, strictly older and validated", () => {
  const event = { distribution_runs: [
    { key: "k1", kind: "week", run: "A", as_of: "2026-10-06T10:00:00Z", validated: true },
    { key: "k2", kind: "week", run: "B", as_of: "2026-10-07T10:00:00Z", validated: true },
    { key: "k3", kind: "week", run: "C", as_of: "2026-10-07T12:00:00Z", validated: false },
    { key: "k4", kind: "live", run: "R1_X", as_of: "2026-10-08T10:00:00Z", validated: true },
    { key: "k5", kind: "week", run: "D" , as_of: null },
    { key: "", kind: "week", run: "E", as_of: "2026-10-06T00:00:00Z" },
  ] };
  const doc = { kind: "live", run: "R1_X", as_of: "2026-10-08T10:00:00Z" };
  assert.deepEqual(runsBefore(event, doc).map((r) => r.run), ["B", "A", "D"].filter((x) => x !== "D"));
  assert.equal(previousRunRef(event, doc).key, "k2");
  assert.equal(lastSeenRunRef(event, doc, "week_A").key, "k1");
  assert.equal(lastSeenRunRef(event, doc, "week_B"), null, "the previous run is the default, not a separate choice");
  assert.equal(lastSeenRunRef(event, doc, "live_R1_X"), null);
  assert.equal(lastSeenRunRef(event, doc, null), null);
  assert.equal(previousRunRef({ distribution_runs: [] }, doc), null);
  assert.equal(previousRunRef({}, doc), null);
  assert.equal(previousRunRef(event, { kind: "week", run: "Z", as_of: null }), null, "no time, no order");
});

test("identical documents give a quiet result; other events and the same run give null", () => {
  const { cur, prev } = pair();
  const c = whatChanged(cur, prev);
  assert.ok(c);
  assert.equal(c.quiet, true);
  assert.equal(c.count, 0);
  assert.equal(c.partial, false);
  assert.equal(c.gapMs, 3600_000);
  assert.deepEqual(c.headline, []);
  assert.equal(whatChanged(prev, prev), null);
  const other = clone(weekRaw);
  other.event_uid = "pga:other";
  other.run = "X";
  assert.equal(whatChanged(cur, parseExplain(other)), null);
});

test("no earlier document and no in-play history hides the strip", () => {
  const { cur } = pair();
  assert.equal(whatChanged(cur, null), null);
});

test("price movers: percentage points, thresholds, order, and whether the sportsbook moved", () => {
  const { cur, prev } = pair((raw) => {
    byId(raw, 0).probs.win.model += 0.012;     // +1.2 pts
    byId(raw, 0).probs.win.market += 0.004;    // sportsbook +0.4 pts
    byId(raw, 1).probs.win.model -= 0.03;      // -3 pts
    byId(raw, 2).probs.win.model += 0.0004;    // +0.04 pts: below the 0.1 pt threshold
  });
  const c = whatChanged(cur, prev);
  const win = c.movers.win;
  assert.deepEqual(win.map((m) => m.id), [prev.players[1].id, prev.players[0].id].filter((id) => win.some((m) => m.id === id)).length === 2 ? [prev.players[1].id, prev.players[0].id] : []);
  assert.ok(Math.abs(win[0].delta - -3) < 1e-9);
  assert.ok(Math.abs(win[1].delta - 1.2) < 1e-9);
  assert.ok(Math.abs(win[1].marketDelta - 0.4) < 1e-9);
  assert.equal(win.length, 2);
  assert.equal(c.quiet, false);
  assert.ok(c.headline.includes("2 win prices moved"));
  assert.equal(c.partial, false);
});

test("field changes: added, removed and newly withdrawn players are named", () => {
  const { cur, prev } = pair((raw) => {
    byId(raw, 3).withdrawn = true;
    const dropped = raw.players.pop();
    raw.players.push({ ...clone(byId(raw, 0)), id: 999999, name: "New, Guy" });
    void dropped;
  });
  const c = whatChanged(cur, prev);
  assert.deepEqual(c.field.added, ["Guy New"]);
  assert.equal(c.field.removed.length, 1);
  assert.equal(c.field.withdrawn.length, 1);
  assert.ok(c.headline.includes("2 players out"));
  assert.ok(c.headline.includes("1 added"));
});

test("weather shift counts only rounds still to play and ignores small changes", () => {
  const withWx = (raw, common) => {
    raw.course_card.weather = { status: "ok", applied_in_prices: true, label: null, layer: "v2.6", issued_at: "2026-10-01T09:00:00Z", field: common.map((c, i) => ({ round: i + 1, common: c, cut_line_shift: c })) };
  };
  const prevRaw = clone(weekRaw);
  withWx(prevRaw, [0, -0.2, -0.4, -0.4]);
  const raw = clone(weekRaw);
  raw.run = "20261001T110500Z";
  raw.as_of = "2026-10-01T11:05:00Z";
  withWx(raw, [0, -0.5, -0.42, -0.4]);
  raw.course_card.weather.issued_at = "2026-10-01T11:00:00Z";
  const c = whatChanged(parseExplain(raw), parseExplain(prevRaw));
  assert.deepEqual(c.weather.rounds.map((r) => r.round), [2]);
  assert.ok(Math.abs(c.weather.rounds[0].delta - -0.3) < 1e-9);
  assert.equal(c.weather.reissued, true);
  assert.ok(c.headline.includes("forecast shifted"));
  // a live document ignores rounds already played
  const liveCur = clone(raw);
  liveCur.kind = "live";
  liveCur.after_round = 2;
  const cc = whatChanged(parseExplain(liveCur), parseExplain(prevRaw));
  assert.deepEqual(cc.weather.rounds, []);
});

test("gap to the sportsbook moves are tracked before the event only", () => {
  const { cur, prev } = pair((raw) => {
    const p = byId(raw, 0);
    p.probs.win.market = Math.max(p.probs.win.market, 0.05);
    p.probs.win.rel = (p.probs.win.rel ?? 0) + 0.2;
  });
  prev.players[0].probs.win.market = Math.max(prev.players[0].probs.win.market, 0.05);
  const c = whatChanged(cur, prev);
  assert.equal(c.gaps[0].id, prev.players[0].id);
  assert.ok(Math.abs(c.gaps[0].delta - 0.2) < 1e-9);
});

test("simulated scoring and winning score shifts need both documents' draws", () => {
  const fs = (mean, med) => ({ rounds: [{ round: 1, mean_strokes: 70 + mean, mean_vs_par: mean, players: 40, basis: "simulated" }], winning: { mode: med, mode_share: 0.2, median: med, mean: med, p10: med - 3, p90: med + 3, n_draws: 1000 } });
  const prevRaw = clone(weekRaw);
  prevRaw.field_scores = fs(-0.5, -17);
  const raw = clone(weekRaw);
  raw.run = "20261001T110500Z";
  raw.as_of = "2026-10-01T11:05:00Z";
  raw.field_scores = fs(-0.8, -19);
  const c = whatChanged(parseExplain(raw), parseExplain(prevRaw));
  assert.equal(c.scoring.rounds[0].round, 1);
  assert.ok(Math.abs(c.scoring.rounds[0].delta - -0.3) < 1e-9);
  assert.equal(c.scoring.winning.delta, -2);
  assert.ok(c.headline.includes("winning score lower"));
  // draws missing from the earlier document: nothing invented
  const none = whatChanged(parseExplain(raw), parseExplain(clone(weekRaw)));
  assert.equal(none.scoring.winning, null);
  assert.deepEqual(none.scoring.rounds, []);
});

test("manual adjustments added or removed are listed", () => {
  const { cur, prev } = pair((raw) => { raw.model.overrides_applied = ["t-skill"]; });
  const c = whatChanged(cur, prev);
  assert.deepEqual(c.overrides.added, ["t-skill"]);
  assert.deepEqual(c.overrides.removed, []);
  assert.ok(c.headline.includes("manual adjustments changed"));
});

test("an in-play document falls back to its own since-last numbers when the earlier document is missing", () => {
  const live = parseExplain(clone(liveRaw));
  const c = whatChanged(live, null);
  assert.ok(c, "live fixture carries since_last");
  assert.equal(c.partial, true);
  assert.equal(c.baseline.run, "R0_20241007T000000Z");
  assert.equal(c.baseline.after_round, 0);
  assert.deepEqual(c.field, { added: [], removed: [], withdrawn: [] });
  assert.deepEqual(c.gaps, []);
  for (const rows of Object.values(c.movers)) for (const m of rows) assert.ok(Math.abs(m.delta) >= 0.1 - 1e-12);
  // a week document has no such numbers
  assert.equal(whatChanged(parseExplain(clone(weekRaw)), null), null);
});

test("hero facts: favourite, biggest disagreement, scoring", () => {
  const doc = parseExplain(clone(weekRaw));
  const h = heroFacts(doc);
  assert.ok(h.favourite.model > 0);
  assert.equal(h.favourite.live, false);
  if (h.second) {
    assert.equal(h.second.kind, "gap");
    const rel = Math.abs(h.second.rel);
    for (const p of doc.players) if (!p.withdrawn && (p.probs.win.market ?? 0) >= 0.02 && p.probs.win.rel !== null) assert.ok(Math.abs(p.probs.win.rel) <= rel + 1e-12);
  }
  assert.equal(h.scoring, null, "fixture has no simulation draws: nothing invented");
  const withDraws = parseExplain({ ...clone(weekRaw), field_scores: { rounds: [{ round: 1, mean_strokes: 70.2, mean_vs_par: -0.8, players: 40, basis: "simulated" }], winning: null } });
  assert.deepEqual(heroFacts(withDraws).scoring, { kind: "round", round: 1, strokes: 70.2, vsPar: -0.8, basis: "simulated" });
  const withWin = parseExplain({ ...clone(weekRaw), field_scores: { rounds: [], winning: { mode: -17, mode_share: 0.2, median: -17, mean: -17, p10: -20, p90: -14, n_draws: 10 } } });
  assert.equal(heroFacts(withWin).scoring.kind, "winning");
  const liveHero = heroFacts(parseExplain(clone(liveRaw)));
  assert.equal(liveHero.favourite.live, true);
  assert.equal(liveHero.favourite.market, null);
  assert.ok(liveHero.second === null || liveHero.second.kind === "move");
});

test("biggest gaps: both directions, shared scale, in-line players left out", () => {
  const doc = parseExplain(clone(weekRaw));
  const g = biggestGaps(doc, 3);
  assert.ok(g.higher.length <= 3 && g.lower.length <= 3);
  for (const r of g.higher) assert.ok(r.rel >= 0.05);
  for (const r of g.lower) assert.ok(r.rel <= -0.05);
  for (let i = 1; i < g.higher.length; i++) assert.ok(g.higher[i - 1].rel >= g.higher[i].rel);
  assert.ok(g.scale >= 0.25);
  const only = biggestGaps(doc, 3, [doc.players[0].id]);
  assert.ok([...only.higher, ...only.lower].every((r) => r.id === doc.players[0].id));
});

test("text helpers never print NaN or -0", () => {
  assert.equal(ptsText(0.0001), "no change");
  assert.equal(ptsText(1.234), "+1.2 pts");
  assert.equal(ptsText(-0.84), "−0.8 pts");
  assert.equal(ptsText(null), "-");
  assert.equal(relText(0.394), "+39%");
  assert.equal(relText(-0.004), "0%");
  assert.equal(relText(NaN), "-");
});

test("event lede reads as one sentence", () => {
  const s = eventLede(parseExplain(clone(weekRaw)));
  assert.match(s, /players, /);
  assert.ok(!/undefined|NaN|null/.test(s));
});
