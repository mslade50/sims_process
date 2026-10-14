import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  ageText, biggestGaps, eventLede, heroFacts, lastSeenRunRef, parseExplain, plainName, previousRunRef, ptsText, relText, runsBefore, spanText,
} from "../app/explain-rules.ts";
import {
  BOOK_GAP_CHANGE_MIN, BOOK_MOVE_REL_MIN, CAUSE_CHIPS, CAUSE_SHARE_MIN, CAUSE_STROKES_MIN, MAX_MOVERS_SERVED, NOISE_Z, attributionContext, bookMoves, changesPayload, logitPerStroke, parseChangesPayload, sportsbookAlertLines, whatChanged,
} from "../app/changes-rules.ts";
import { ROLE, buildPair, clone as cl, idOf } from "./fixtures/changes-pair.mjs";

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

/* ------------------------------------------------------------------ causes behind each move */
const win = (c, id) => c.movers.win.find((m) => m.id === id);
const keys = (m) => m.causes.map((x) => x.key);

test("each price move is split into labelled causes with strokes and a share of the probability move", () => {
  const { prev, cur } = buildPair();
  const c = whatChanged(parseExplain(cur), parseExplain(prev));
  // weather (mu excludes weather, so the strokes come from the wx components) and the later wave
  const w = win(c, idOf(cur, "weather"));
  const wx = w.causes.find((x) => x.key === "weather");
  assert.ok(wx, "weather named");
  assert.equal(w.causes[0].key, "weather");
  assert.equal(wx.chip, "weather helps 0.12");
  assert.ok(Math.abs(wx.strokes - 0.12) < 1e-9);
  assert.match(wx.detail, /wave early → late/);
  assert.ok(wx.pts > 0 && w.delta > wx.pts, "the field effect adds to the weather share");
  assert.ok(w.separable);
  // skill update
  const s = win(c, idOf(cur, "skill"));
  assert.equal(s.causes[0].key, "skill");
  assert.equal(s.causes[0].chip, "skill update helps 0.15");
  // owner override
  const o = win(c, idOf(cur, "override"));
  assert.equal(o.causes[0].key, "override");
  assert.equal(o.causes[0].chip, "override helps 0.30");
  // course and travel: both named, course first
  const ct = win(c, idOf(cur, "courseTravel"));
  assert.deepEqual(keys(ct), ["course", "travel"]);
  assert.equal(ct.causes[1].chip, "travel hurts 0.05");
  assert.ok(ct.causes[1].pts < 0 && ct.causes[0].pts > 0);
  assert.ok(ct.causes.length <= CAUSE_CHIPS);
});

test("a withdrawal is named as the cause of the field effect, never as the player's own doing", () => {
  const { prev, cur, wdWin } = buildPair();
  const c = whatChanged(parseExplain(cur), parseExplain(prev));
  const e = win(c, idOf(cur, "fieldOnly"));
  assert.equal(e.causes[0].key, "field");
  assert.equal(e.causes[0].chip, "Koivun WD");
  assert.equal(e.causes[0].strokes, null);
  assert.ok(Math.abs(e.causes[0].pts - e.delta) < 1e-9, "all of it is the field effect");
  assert.match(e.causes[0].detail, /Jackson Koivun left the field/);
  const w = win(c, idOf(cur, "weather"));
  assert.ok(w.causes.some((x) => x.key === "field" && x.chip === "Koivun WD"));
  assert.ok(wdWin > 0.05);
  assert.deepEqual(c.field.withdrawn, ["Jackson Koivun"]);
});

test("own-skill shares add up: strokes become probability through the market slope, the rest is the field", () => {
  const { prev, cur } = buildPair();
  // no withdrawal: the weather share is the whole move
  const noWd = cl(cur);
  noWd.players[ROLE.wd].withdrawn = false;
  for (const [i, p] of noWd.players.entries()) if (i !== ROLE.wd) { p.probs.win.model *= 1 - prev.players[ROLE.wd].probs.win.model; p.probs.win.rel = (p.probs.win.model - p.probs.win.market) / p.probs.win.market; }
  const c = whatChanged(parseExplain(noWd), parseExplain(prev));
  const w = win(c, idOf(cur, "weather"));
  assert.equal(w.causes.length, 1, "only weather carries a share");
  assert.ok(Math.abs(w.causes[0].pts - w.delta) < 1e-6, `${w.causes[0].pts} vs ${w.delta}`);
  assert.ok(!c.movers.win.some((m) => m.id === idOf(cur, "fieldOnly")), "no field effect without a field change");
});

test("the published strokes_per_logit is the logit change per stroke, and top 5 is fitted from the field", () => {
  const doc = parseExplain(clone(weekRaw));
  const fit = (m) => {
    const rows = doc.players.filter((p) => !p.withdrawn && p.probs[m].model > 0.0005 && p.probs[m].model < 0.9995);
    const mx = rows.reduce((a, p) => a + p.mu, 0) / rows.length;
    const lg = (p) => Math.log(p / (1 - p));
    const my = rows.reduce((a, p) => a + lg(p.probs[m].model), 0) / rows.length;
    return rows.reduce((a, p) => a + (p.mu - mx) * (lg(p.probs[m].model) - my), 0) / rows.reduce((a, p) => a + (p.mu - mx) ** 2, 0);
  };
  for (const m of ["top_10", "top_20", "make_cut"]) assert.ok(Math.abs(logitPerStroke(doc, m) / fit(m) - 1) < 0.15, m);
  assert.ok(Math.abs(logitPerStroke(doc, "top_5") / fit("top_5") - 1) < 1e-9, "fallback is the fit across the field");
});

test("causes that cannot be separated say so, and noise is called noise", () => {
  const { prev, cur } = buildPair();
  const raw = cl(cur);
  raw.players[ROLE.skill].components = {};
  let c = whatChanged(parseExplain(raw), parseExplain(prev));
  const m = win(c, idOf(cur, "skill"));
  assert.equal(m.separable, false);
  assert.deepEqual(m.causes.map((x) => x.chip), ["not separable"]);
  // a market without a usable slope (small field, nothing to fit): strokes are shown, no probability share is invented
  const slopeless = cl(cur);
  slopeless.edge_model.strokes_per_logit = {};
  slopeless.players = slopeless.players.slice(0, 5);
  const prevSmall = cl(prev);
  prevSmall.players = prevSmall.players.slice(0, 5);
  c = whatChanged(parseExplain(slopeless), parseExplain(prevSmall));
  const w = win(c, idOf(cur, "weather"));
  assert.equal(w.causes[0].chip, "weather helps 0.12");
  assert.equal(w.causes[0].pts, null);
  // a small move with no cause: within simulation noise
  const quiet = cl(weekRaw);
  quiet.run = "later";
  quiet.as_of = "2026-10-01T11:05:00Z";
  quiet.players[0].probs.win.model += 0.0015;
  c = whatChanged(parseExplain(quiet), parseExplain(clone(weekRaw)));
  const n = c.movers.win[0];
  assert.equal(n.causes[0].key, "noise");
  assert.equal(n.causes[0].chip, "within simulation noise");
  // a big move with no cause and nothing in the field is attributed to "field and other players", not to the player
  quiet.players[0].probs.win.model += 0.02;
  c = whatChanged(parseExplain(quiet), parseExplain(clone(weekRaw)));
  assert.equal(c.movers.win[0].causes[0].chip, "field and other players");
  assert.ok(NOISE_Z >= 2 && CAUSE_SHARE_MIN === 0.15 && CAUSE_STROKES_MIN === 0.01);
});

test("a shift common to the whole field is reported as field level, not as every player's own skill", () => {
  const { prev, cur } = buildPair();
  const raw = cl(cur);
  for (const p of raw.players) { p.mu += 0.04; p.mu_rel += 0.04; }
  const ctx = attributionContext(parseExplain(raw), parseExplain(prev));
  assert.ok(Math.abs(ctx.fieldLevel - 0.04) < 1e-9, String(ctx.fieldLevel));
  const c = whatChanged(parseExplain(raw), parseExplain(prev));
  assert.ok(Math.abs(c.attribution.fieldLevelStrokes - 0.04) < 1e-9);
  const m = win(c, idOf(cur, "weather"));
  assert.ok(!m.causes.some((x) => x.key === "unseparated"), "not blamed on the player");
});

test("in play: the since-last numbers alone still give an in-play cause, with the scores as detail", () => {
  const live = clone(liveRaw);
  const withMove = live.players.find((p) => p.live && p.live.since_last.win);
  withMove.live.d_mu_since_last = 0.31;
  withMove.live.rounds = [{ round: 1, score: 68, sg_total: 2.1 }];
  const c = whatChanged(parseExplain(live), null);
  assert.ok(c.partial);
  const row = Object.values(c.movers).flat().find((m) => m.id === withMove.id);
  assert.ok(row, "mover found");
  const ip = row.causes.find((x) => x.key === "in_play");
  assert.ok(ip, JSON.stringify(row.causes));
  assert.equal(ip.chip, "in-play helps 0.31");
  assert.match(ip.detail ?? "", /scored 68 in round 1/);
});

/* ------------------------------------------------------------------ sportsbook moves */
const bookFor = (c, cur, role, market = "win") => c.bookMoves.find((b) => b.id === idOf(cur, role) && b.market === market);

test("sportsbook moves: who moved, which way the gap went, and what is actionable", () => {
  const { prev, cur } = buildPair();
  const c = whatChanged(parseExplain(cur), parseExplain(prev));
  const away = bookFor(c, cur, "bookAway");
  assert.equal(away.kind, "book_away");
  assert.equal(away.actionable, "growing");
  assert.match(away.text, /^Book drifted Jackson Suber away from our price: gap \+50% → \+100%$/);
  const toward = bookFor(c, cur, "bookToward");
  assert.equal(toward.kind, "book_toward");
  assert.equal(toward.actionable, "closing");
  assert.match(toward.text, /^Book shortened Rico Hoey toward our price: gap \+100% → \+33%$/);
  const mt = bookFor(c, cur, "modelToward");
  assert.equal(mt.kind, "model_toward");
  assert.equal(mt.actionable, null, "only moves by the sportsbook are actionable");
  assert.match(mt.text, /^We moved Kristoffer Ventura toward the book/);
  assert.equal(bookFor(c, cur, "modelAway").kind, "model_away");
  const both = bookFor(c, cur, "both");
  assert.equal(both.kind, "both");
  assert.equal(both.actionable, "growing");
  assert.match(both.text, /^Both moved on Kevin Yu \(book shortened, our price up\)/);
  // actionable cases come first
  const firstOthers = c.bookMoves.findIndex((b) => !b.actionable);
  assert.ok(c.bookMoves.slice(firstOthers).every((b) => !b.actionable));
  assert.ok(c.headline.some((h) => /sportsbook moves? to act on/.test(h)));
  assert.equal(BOOK_MOVE_REL_MIN, 0.03);
  assert.equal(BOOK_GAP_CHANGE_MIN, 0.05);
});

test("sportsbook moves ignore small moves, tandem moves, in-play runs and long shots", () => {
  const { prev, cur } = buildPair();
  const set = (doc, role, model, market) => { const p = doc.players[ROLE[role]]; p.probs.win.model = model; p.probs.win.market = market; p.probs.win.rel = (model - market) / market; };
  const raw = cl(cur);
  set(raw, "bookAway", 0.030, 0.0201);              // the book moved 0.01 pts: below the bar
  set(raw, "bookToward", 0.033, 0.0165);            // both move together by the same proportion: the gap is unchanged
  const c = whatChanged(parseExplain(raw), parseExplain(prev));
  assert.ok(!bookFor(c, cur, "bookAway"));
  assert.ok(!bookFor(c, cur, "bookToward"));
  const live = cl(cur);
  live.kind = "live"; live.run = "R1_x"; live.after_round = 1;
  assert.deepEqual(bookMoves(parseExplain(live), parseExplain(prev)), []);
  const longShot = cl(prev);
  const longShotCur = cl(cur);
  set(longShot, "bookAway", 0.004, 0.003); set(longShotCur, "bookAway", 0.004, 0.002);
  assert.ok(!bookMoves(parseExplain(longShotCur), parseExplain(longShot)).some((b) => b.id === idOf(cur, "bookAway") && b.market === "win"));
});

test("alert lines: only the actionable sportsbook moves, worded for a message", () => {
  const { prev, cur } = buildPair();
  const c = whatChanged(parseExplain(cur), parseExplain(prev));
  const lines = sportsbookAlertLines(c.bookMoves, 10);
  assert.ok(lines.length >= 3);
  assert.ok(lines.every((l) => /^Edge (growing|closing): /.test(l)));
  assert.ok(lines.some((l) => l.startsWith("Edge growing: Book drifted Jackson Suber")));
  assert.equal(sportsbookAlertLines(c.bookMoves, 1).length, 1);
  assert.deepEqual(sportsbookAlertLines([]), []);
});

test("the served diff keeps the biggest movers per market and round-trips through the parser", () => {
  const { prev, cur } = buildPair();
  const c = whatChanged(parseExplain(cur), parseExplain(prev));
  const payload = changesPayload("pga:554:2026-10-01", "from.json", "to.json", c);
  assert.ok(Object.values(payload.change.movers).every((r) => r.length <= MAX_MOVERS_SERVED));
  assert.equal(payload.moverTotals.win, c.movers.win.length);
  const back = parseChangesPayload(JSON.parse(JSON.stringify(payload)));
  assert.deepEqual(back.movers.win[0], payload.change.movers.win[0]);
  assert.equal(back.bookMoves.length, c.bookMoves.length);
  assert.equal(parseChangesPayload({ schema: "other" }), null);
  assert.equal(parseChangesPayload({ ...payload, change: { ...payload.change, movers: { win: [{ id: 1 }] } } }), null);
  assert.equal(parseChangesPayload(null), null);
});
