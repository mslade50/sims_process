// Two realistic runs of one event for the What changed tests, built by perturbing the real-shaped explain_week.json fixture one cause at a time.
// Probabilities are moved through the same logit-per-stroke slope the export publishes, so every cause has a known size.
import { readFile } from "node:fs/promises";

const base = JSON.parse(await readFile(new URL("./explain_week.json", import.meta.url), "utf8"));
export const clone = (o) => JSON.parse(JSON.stringify(o));
const MARKETS = ["win", "top_5", "top_10", "top_20", "make_cut"];
const SLOPE = { win: 1.961, top_5: 1.93, top_10: 1.66, top_20: 1.53, make_cut: 0.79 };
const logit = (p) => Math.log(p / (1 - p));
const expit = (x) => 1 / (1 + Math.exp(-x));
const WX = ["wx_wind", "wx_gust", "wx_temp", "wx_rain", "wx_tod", "wx_traj"];

/** Move one player's model chance in every market by `strokes` a round, as the simulation would. */
function shift(p, strokes) {
  for (const m of MARKETS) {
    const e = p.probs[m];
    e.model = expit(logit(e.model) + SLOPE[m] * strokes);
    e.rel = (e.model - e.market) / e.market;
  }
}
function setWin(p, model, market) {
  p.probs.win.model = model;
  p.probs.win.market = market;
  p.probs.win.rel = (model - market) / market;
}

/** Indexes into the fixture's players, by role. */
export const ROLE = { wd: 0, weather: 1, skill: 2, override: 3, courseTravel: 4, fieldOnly: 6, bookAway: 7, bookToward: 8, modelToward: 9, both: 10, modelAway: 11 };

export function buildPair() {
  const prev = clone(base);
  prev.event_uid = base.event_uid;
  for (const p of prev.players) for (const k of WX) p.components[k] = 0;
  // pin the book cases (win market) before the event
  setWin(prev.players[ROLE.bookAway], 0.030, 0.020);
  setWin(prev.players[ROLE.bookToward], 0.030, 0.015);
  setWin(prev.players[ROLE.modelToward], 0.040, 0.030);
  setWin(prev.players[ROLE.both], 0.040, 0.030);
  setWin(prev.players[ROLE.modelAway], 0.025, 0.020);
  prev.course_card.weather = { status: "ok", applied_in_prices: true, label: null, layer: "v2.6", issued_at: "2026-10-01T09:00:00Z", field: [{ round: 1, common: 0, cut_line_shift: 0 }, { round: 2, common: -0.2, cut_line_shift: -0.2 }] };

  const cur = clone(prev);
  cur.run = "20261001T110500Z";
  cur.as_of = "2026-10-01T11:05:00Z";
  const P = cur.players;
  const wdWin = prev.players[ROLE.wd].probs.win.model;
  // a: skill update
  P[ROLE.skill].components.kalman += 0.15; P[ROLE.skill].mu += 0.15; P[ROLE.skill].mu_rel += 0.15; shift(P[ROLE.skill], 0.15);
  // b: owner override (mu_untouched stays)
  P[ROLE.override].components.override += 0.30; P[ROLE.override].mu += 0.30; P[ROLE.override].mu_rel += 0.30; shift(P[ROLE.override], 0.30);
  // c: course fit up, travel down
  P[ROLE.courseTravel].components.course += 0.10; P[ROLE.courseTravel].components.loc_travel -= 0.05; P[ROLE.courseTravel].mu += 0.05; P[ROLE.courseTravel].mu_rel += 0.05; shift(P[ROLE.courseTravel], 0.05);
  // d: weather (mu excludes weather; it is applied on top) and a later wave
  P[ROLE.weather].components.wx_wind = 0.08; P[ROLE.weather].components.wx_tod = 0.04; P[ROLE.weather].wave = "late"; shift(P[ROLE.weather], 0.12);
  cur.course_card.weather.field[1].common = -0.5;
  cur.course_card.weather.issued_at = "2026-10-01T11:00:00Z";
  // e: the favourite withdraws; everybody else's win chance rises by 1 / (1 - his chance)
  P[ROLE.wd].withdrawn = true;
  for (const [i, p] of P.entries()) if (i !== ROLE.wd) { p.probs.win.model /= 1 - wdWin; p.probs.win.rel = (p.probs.win.model - p.probs.win.market) / p.probs.win.market; }
  // book cases, after the field effect so they are exact
  setWin(P[ROLE.bookAway], 0.030, 0.015);
  setWin(P[ROLE.bookToward], 0.030, 0.0225);
  setWin(P[ROLE.modelToward], 0.0325, 0.030);
  setWin(P[ROLE.both], 0.050, 0.035);
  setWin(P[ROLE.modelAway], 0.030, 0.020);
  return { prev, cur, wdWin };
}
export const idOf = (doc, role) => doc.players[ROLE[role]].id;
