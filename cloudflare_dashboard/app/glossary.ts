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
  sd: "Round swing: how much this player's single-round score typically moves around his expected score, in strokes (a standard deviation). The simulation uses it for every round.",
  mu_untouched: "Rating before your manual override was applied.",
  sd_untouched: "Round swing before your manual override was applied.",
  mu_tour: "The same expected strokes gained, shifted to the average PGA Tour player scale so it can be compared across weeks. Reference only; prices use the field-relative number.",
  se_kernel: "Extra rating uncertainty on top of the player's round swing. It is zero in the current model, so it is hidden unless a run uses it.",
  extra_round_volatility_sd: "Extra round-to-round swing the simulation adds on top of hole and day luck so each player matches his target round swing.",
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
  course_fit: "How much this course moves the player's expected strokes per round, combining how well his game suits the venue and how he has done here before.",
  fit_rs_ddacc: "How well the player's driving distance and accuracy suit this venue, based on what has mattered here in past editions (shrunk toward the tour average).",
  course_history: "How much better or worse than expected the player has done at this venue; heavily shrunk so a few visits barely move it.",
  course_sd_mult: "Course spread factor: how much this course widens (above 1) or narrows (below 1) scores compared with a typical course; 1.00 is typical.",
  sd_shrunk: "The player's own round swing, pulled toward the tour norm when he has few rounds; the starting point before the course adjustment.",
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
  // engine
  tau: "How far a player's form can drift up or down during the week in the simulation, in strokes per round. Everyone gets their own random drift.",
  v_hole: "How much pure hole-to-hole luck adds to a round (a variance, in strokes squared).",
  hole_a: "How much one hole's result carries over into the next (0 means holes are independent).",
  seed_spread: "How far any win probability moves between independent simulation batches; smaller means steadier prices. Win odds are accurate to about this many percentage points.",
  sd_scale: "A small overall scaling applied to every player's round swing (1.00 means none).",
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
  sd: "Round swing",
  mu_untouched: "Rating before your override",
  sd_untouched: "Swing before your override",
  se_kernel: "Rating uncertainty ±",
  location: "Location",
  course_fit: "Course fit and history",
  fit_rs_ddacc: "Venue fit (distance and accuracy)",
  course_history: "Course history",
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
