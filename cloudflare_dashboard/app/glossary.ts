/**
 * Plain-English definitions for the Model inputs view (and anything else that shows the same numbers).
 * One sentence each. Keys are the data keys the views use; `define(key)` returns "" for an unknown key so a
 * missing entry never breaks a page. HEADERS holds the human column header for the same keys.
 * Definitions were written against the golfprice publisher (inputs_export.py, feature_glossary.py, INPUTS_SCHEMA.md).
 */

/** Skill-model families that make up the player's mean, in the order the breakdown adds them. */
export const FAMILY_DEFS: Record<string, string> = {
  level_form: "Averages of the player's recent strokes gained over different memory lengths: who is good, and how hot or cold he is right now.",
  kalman: "A running skill estimate that updates after every round and grows less certain while a player is idle.",
  xtour: "Evidence from outside the four core tours (feeder, Asian, Southern-hemisphere and senior tours); matters most for players with thin PGA or DP World Tour records.",
  category: "Skill split into off the tee, approach, around the green and putting, plus traditional stats; putting is noisy and mean-reverting, ball-striking is stickier.",
  sit: "Circumstances of this start: how long since the player last played and an age-curve drift.",
  act: "How much the player has been playing lately and how much history the model can see for him.",
  sklv: "A statistically shrunk skill estimate, the hot or cold hand over his last few events, and the recent trend.",
  course: "Course fit (does his driving profile suit this venue) plus course history (has he beaten expectations here), combined into one strokes-per-round effect.",
  thin: "Pulls players with very little history toward a sensible prior built from their tours and how they got into the field.",
  disp: "How volatile the player's rounds are, measured by his share of blow-up rounds.",
  other: "Small leftover model terms that are not about skill history, such as the no-history flag.",
};

const FAMILY_KEYS = Object.keys(FAMILY_DEFS);

export const GLOSSARY: Record<string, string> = {
  // headline numbers
  mu: "Rating against this week's field: expected strokes per round better (+) or worse (-) than the average player in this field (0 is average). Higher is better. This is the number the prices use.",
  sd: "Std dev: the standard deviation of this player's single-round score around their expected score, in strokes. The simulation uses it for every round.",
  mu_untouched: "Rating before your manual override was applied.",
  sd_untouched: "Std dev before your manual override was applied.",
  mu_tour: "The same expected strokes gained, shifted to the average PGA Tour player scale so it can be compared across weeks. Reference only; prices use the field-relative number.",
  se_kernel: "Extra rating uncertainty on top of the player's std dev. It is zero in the current model, so it is hidden unless a run uses it.",
  extra_round_volatility_sd: "Extra round-to-round std dev the simulation adds on top of hole and day luck so each player matches their target std dev.",
  thin_data: "Fewer than 50 rounds of history: this rating leans on a typical player's level, so treat it as an informed guess.",
  prior_rounds: "Rounds of history the model can see for this player; fewer rounds means a less certain rating.",
  overridden: "Whether you have a manual override active on this player for this event.",
  override_total: "Total strokes per round added or removed by your manual overrides on this player.",
  amateur: "Whether the player is an amateur.",
  country: "The player's nationality.",
  dg_id: "The player's Data Golf identifier.",
  name: "Player name.",
  // probabilities
  prob_win: "The model's chance this player wins outright (ties split), before blending with the market.",
  prob_top_5: "The model's chance of finishing in the top 5, ties counted fractionally.",
  prob_top_10: "The model's chance of finishing in the top 10, ties counted fractionally.",
  prob_top_20: "The model's chance of finishing in the top 20, ties counted fractionally.",
  prob_make_cut: "The model's chance of making the cut.",
  // pieces of the mean
  location: "Adjustment for where the event is relative to the player's home and recent travel: home base, time-zone and distance effects, nationality, and a small refit correction.",
  location_total: "Adjustment for where the event is relative to the player's home and recent travel: home base, time-zone and distance effects, nationality, and a small refit correction.",
  location_home_travel: "Effect of how much of his recent golf was in this region, plus time-zone, continent moves and climate or links suitability.",
  location_nationality: "Effect of playing in or near the player's home country, how far away the venue is from it, and a recent flight in from another continent.",
  location_refit: "Small correction because the skill weights were refit together with the location terms.",
  course_fit: "How much this course moves the player's expected strokes per round, course fit and course history combined.",
  course_fit_part: "Strokes per round from course fit: how well the player's driving distance and accuracy suit this venue (what has mattered here in past editions, shrunk toward the tour average). Blank on runs before the split.",
  course_history_part: "Strokes per round from course history: how much better or worse than expected the player has done at this venue, heavily shrunk so a few visits barely move it. Blank on runs before the split.",
  fit_rs_ddacc: "How well the player's driving distance and accuracy suit this venue, based on what has mattered here in past editions (shrunk toward the tour average).",
  course_history: "How much better or worse than expected the player has done at this venue; heavily shrunk so a few visits barely move it.",
  course_sd_mult: "Course spread factor: how much this course widens (above 1) or narrows (below 1) scores compared with a typical course; 1.00 is typical.",
  sd_shrunk: "The player's own std dev, pulled toward the tour norm when they have few rounds; the starting point before the course adjustment.",
  skill_total: "Sum of the skill-model families below, before the location adjustment and any override.",
  rounding: "Tiny leftover from rounding the components; the components add up to the final mu.",
  override: "Strokes per round added or removed by your manual override, including the common shift that keeps the field average at zero.",
  weather_round: "Strokes per round added for this round from the tee wave (early or late) and forecast wind. It is added to the rating for that round only and is not part of the rating above.",
  // course tab
  expected_vs_par: "The average score on this hole for a zero-skill reference player, relative to par.",
  birdie_pct: "Share of the time the reference player makes birdie or better on this hole.",
  bogey_pct: "Share of the time the reference player makes bogey or worse on this hole.",
  eagle_pct: "Chance of eagle or better.",
  par_pct: "Chance of making par.",
  double_pct: "Chance of a double bogey.",
  triple_pct: "Chance of triple bogey or worse.",
  sensitivity: "How much skill matters on this hole: 1.00 is an average hole, above 1 rewards the better player more, below 1 is closer to a coin flip.",
  yardage: "Length of the hole in yards.",
  par: "Par for the hole or round.",
  scoring_avg: "Average score relative to par for a field-average player on this course and round.",
  scoring_avg_untouched: "The same scoring average before your course override.",
  birdies_per_round: "Birdies or better per round for the field-average player.",
  bogeys_per_round: "Bogeys or worse per round for the field-average player.",
  // This week / Why priced vocabulary (inline definitions)
  ex_our_chance: "Our chance: how often the player wins (or finishes in the top N) across the simulated tournaments.",
  ex_market_chance: "Sportsbook chance: the chance implied by the odds after removing the bookmaker's margin, averaged over the books that price it.",
  ex_gap: "Gap to the sportsbook: how far our chance is from the sportsbook's, as a share of the sportsbook's chance. +39% means ours is 39% higher. Under 5% counts as in line.",
  ex_pts: "Percentage points: the plain difference between two chances. Going from 10.0% to 10.8% is +0.8 points.",
  ex_field_strength: "Field strength: the field's average skill in strokes a round against an average PGA Tour player, on the Data Golf scale. Plus means a stronger field.",
  ex_dg_skill_gap: "Skill gap vs Data Golf: our pre-tournament skill estimate minus Data Golf's final estimate, in strokes a round against this field's average (before weather). + means we rate the player higher. Data Golf is a reference only.",
  ex_odd_one_out: "Odd one out: the source whose chances sit furthest from the other two across win, top 10 and top 20 (average gap in log-odds). If two sources agree within 0.15 and the third is about 1.8 times further from both, the third is marked.",
  ex_peers: "Players priced like them: the players with the nearest sportsbook win chance (up to three below and three above). What they share with this player does not explain why we differ from the market on this player; what is particular to this player does.",
  ex_dg_check: "Data Golf odds check: Data Golf's stated skill number is compared with the skill its own win and top-20 odds imply across this field. When they differ by more than about a quarter of a stroke the stated number is unreliable for this player, so compare chances instead.",
  ex_skill_gap: "Skill gap: strokes a round better (+) or worse (-) than the sportsbook's prices imply, allowing for how boom-or-bust each side sees the player.",
  ex_winning_score: "Winning score: the lowest 72-hole total in each simulated tournament, to par. Most likely is the single score that wins most often; the range covers 8 in 10 simulations.",
  ex_scoring_avg: "Expected scoring average: the field's average score in a round across the simulations, including the weather forecast. A round already played shows the real average.",
  ex_contenders: "Contenders: the players with the best average win chance (ours and the sportsbook's), so long shots with thin records do not crowd out the players a bet depends on.",
  ex_weather_effect: "Weather effect: strokes a round the forecast adds (plus, harder) or takes off (minus, easier) for the whole field or for a tee-time wave.",
  ex_in_line: "In line: our chance is within 5% of the sportsbook's, too close to call a difference.",
  ex_priced: "When the simulation behind these prices was run (US Eastern time).",
  // engine
  tau: "How far a player's form can drift up or down during the week in the simulation, in strokes per round. Everyone gets their own random drift.",
  v_hole: "How much pure hole-to-hole luck adds to a round (a variance, in strokes squared).",
  hole_a: "How much one hole's result carries over into the next (0 means holes are independent).",
  seed_spread: "How far any win probability moves between independent simulation batches; smaller means steadier prices. Win odds are accurate to about this many percentage points.",
  sd_scale: "A small overall scaling applied to every player's std dev (1.00 means none).",
  // run / odds
  n_sims: "How many full tournaments the simulation played out to produce the probabilities.",
  odds_age: "How old each book's prices were when the model run was built.",
  fetched: "When we pulled the prices from the book.",
  book_update: "When the book itself last changed its prices.",
};

for (const key of FAMILY_KEYS) GLOSSARY[`chl_${key}`] = FAMILY_DEFS[key];
GLOSSARY.chl_total = GLOSSARY.skill_total;

/** Human column / label text for the same keys. */
export const HEADERS: Record<string, string> = {
  name: "Player",
  mu: "Rating vs field",
  sd: "Std dev",
  mu_untouched: "Rating before your override",
  sd_untouched: "Std dev before your override",
  se_kernel: "Rating uncertainty ±",
  location: "Location",
  course_fit: "Course total",
  course_fit_part: "Course fit",
  course_history_part: "Course history",
  fit_rs_ddacc: "Venue fit, raw value",
  course_history: "Course history, raw value",
  course_sd_mult: "Course spread factor",
  override_total: "Your override",
  prior_rounds: "Rounds of history",
  prob_win: "Win",
  prob_top_5: "Top 5",
  prob_top_10: "Top 10",
  prob_top_20: "Top 20",
  prob_make_cut: "Make cut",
  country: "Country",
  amateur: "Amateur",
  dg_id: "Data Golf ID",
  expected_vs_par: "Expected score vs par",
  birdie_pct: "Birdie or better",
  bogey_pct: "Bogey or worse",
  eagle_pct: "Eagle or better",
  par_pct: "Makes par",
  double_pct: "Double bogey",
  triple_pct: "Triple or worse",
  sensitivity: "Skill matters (1 = avg hole)",
  hole: "Hole",
  par: "Par",
  yardage: "Yards",
  round: "Round",
  scoring_avg: "Scoring average vs par",
  scoring_avg_untouched: "Before your course override",
  birdies_per_round: "Birdies per round",
  bogeys_per_round: "Bogeys per round",
};

/** Titles for the per-family skill columns come from the family list the view owns; this lets a view register them once. */
export function registerFamilyLabels(families: Array<[string, string]>): void {
  for (const [key, label] of families) HEADERS[`chl_${key}`] = label;
}

/** One-sentence definition for a key, or "" when none is written. */
export function define(key: string): string {
  return GLOSSARY[key] ?? "";
}

/** Definitions for a list of column keys, shaped for DataTable's headerTitles. */
export function titlesFor(keys: string[], extra: Record<string, string> = {}): Record<string, string> {
  const out: Record<string, string> = {};
  for (const key of keys) {
    const text = extra[key] ?? GLOSSARY[key];
    if (text) out[key] = text;
  }
  return out;
}

/* Distribution shapes (Same chance to win, different weeks). */
Object.assign(GLOSSARY, {
  spread: "How much a player's single-round score swings around his expected score, in strokes (the model's sd). Higher spread means more blow-up and breakout rounds, so more of his win chance comes from rare big weeks.",
  steady: "Steady: more top-10 and top-20 finishes than a player's win chance usually comes with. A high floor, with less of his chance resting on one huge week.",
  boom_bust: "Boom-or-bust: fewer top finishes than his win chance usually comes with, so more of his chance rides on a rare big week (and the rest of the time he fades).",
  typical_shape: "The typical shape: the curve fitted across this week's top 30 relating win chance to top-5, top-10, top-20 and make-cut chance. A player above the curve is steadier than typical; below it, more boom-or-bust.",
});
