"use client";

import { GAP_METHOD_SENTENCE, LABELS, ZERO_SENTENCE } from "./labels";
import { etTime } from "./lib";
import { NOT_YET_SHRUNK, finite, isShrunk, shortProfileDate, signedZ, unshrunkValue, benchmarkValue, type PgaBenchmark, type PgaBenchmarkReference } from "./player-profile-rules";
import { CROSS_TOUR_BAND, INTERVAL_CLAUSE, RADAR_LABEL, regularTick } from "./player-hero-rules";

const pct = (v: unknown) => (finite(v) ? `${(v * 100).toFixed(0)}%` : "unknown");
const tourName = (t: unknown) => (typeof t === "string" && t ? t.toUpperCase() : null);

/** The ONE "Method and caveats" disclosure for the player page (B2). Plain English first; identifiers and raw fields sit in the last, collapsed block. */
export function PlayerMethod({ rating, reference, catalog }: { rating: PgaBenchmark | undefined; reference: PgaBenchmarkReference | undefined; catalog?: unknown }) {
  const method = rating?.method;
  const interval = rating?.decomposition?.historical_mean_interval;
  const headline = benchmarkValue(rating, reference);
  const unshrunk = unshrunkValue(rating, reference);
  const shrunk = isShrunk(rating, reference);
  const tick = regularTick(catalog);
  const priorTour = tourName(rating?.shrink_prior_tour);
  const weights = rating?.decomposition?.tour_weights;
  return (
    <details className="pp-details ph-method" data-testid="player-method">
      <summary>Method and caveats</summary>
      <h4>The headline: {LABELS.vsPgaAvg.short}</h4>
      <p>{LABELS.vsPgaAvg.long}.</p>
      <p>{ZERO_SENTENCE}</p>
      <p>
        The zero point is the average round on the {reference?.year ?? "2025"} PGA Tour (rounds {reference?.rounds?.join(" and ") ?? "1 and 2"}), which is {signedZ(reference?.adjusted_sg_mean, 3)} strokes on the data provider&apos;s scale, from {reference?.n_rounds?.toLocaleString("en-US") ?? "an unknown number of"} rounds by {reference?.n_players ?? "an unknown number of"} players. It stays fixed so numbers are comparable week to week.
      </p>
      <p>
        Recent form: {method ? `rounds from the last ${Math.round(method.window_days / 365 * 10) / 10} years count, and a round's weight halves every ${Math.round(method.half_life_days / 30)} months` : "the weighting is unavailable"}. Only rounds with published adjusted strokes gained are used, and a player needs at least {method?.min_rounds ?? 12} rounds to get a number.
      </p>
      {shrunk ? (
        <p data-testid="shrunk-estimate">
          Shrinking: the recent-form figure is pulled toward the typical level of {priorTour ? `${priorTour} players` : "the player's main tour"}, giving the player&apos;s own rounds {finite(rating?.shrink_weight_on_data) ? `${pct(rating.shrink_weight_on_data)} of the weight` : "most of the weight"}. Players with many rounds barely move; players with few rounds move a lot.
          {finite(unshrunk) && <> <b title={LABELS.unshrunkForm.long}>Unshrunk form: {signedZ(unshrunk, 3)}</b>; headline after shrinking: <b>{signedZ(headline, 3)}</b> strokes per round.</>}
          {rating?.shrink_stale && <> This player has been away for several months, so recent form leans on older rounds.</>}
        </p>
      ) : (
        <p data-testid="shrunk-estimate">{NOT_YET_SHRUNK}</p>
      )}
      <p>Rounds behind it: {finite(rating?.n_rounds) ? rating.n_rounds : "unknown"} rounds in {finite(rating?.n_events) ? rating.n_events : "unknown"} events, most recent {shortProfileDate(rating?.last_observation ?? undefined)}. {finite(rating?.weight_on_data) && !shrunk ? <>Weight on the player&apos;s own rounds {pct(rating.weight_on_data)}.</> : null}</p>
      <p>
        Range: for the unshrunk form, a rough 95% range is {finite(interval?.lower) && finite(interval?.upper) ? `${signedZ(interval.lower, 2)} to ${signedZ(interval.upper, 2)}` : "unavailable"}. It describes how settled this player&apos;s past results are, and {INTERVAL_CLAUSE}. It is not a forecast range.
      </p>
      {weights && weights.length > 0 && (
        <p>Where the rounds were played: {weights.map(t => `${t.tour.toUpperCase()} ${pct(t.weight_fraction)} (${t.n_rounds} rounds)`).join(", ")}. Rounds on other tours are harder to compare, so {CROSS_TOUR_BAND}.</p>
      )}
      {tick && <p>The dashed tick on the bar marks a typical full-time PGA regular ({signedZ(tick.value, 2)}{tick.asOf ? `, as of ${shortProfileDate(tick.asOf)}` : ""}).</p>}
      <p>This is rebuilt from the latest data each time, so an old checkpoint cannot show it as it looked back then.</p>
      <h4>How this relates to this week&apos;s model number</h4>
      <p>{GAP_METHOD_SENTENCE}</p>
      <h4>Other caveats</h4>
      <ul>
        <li>The strokes-gained bars compare plain averages with a fixed 3-year PGA average; there is no category breakdown for tours other than PGA and LIV. The radar shows {RADAR_LABEL}; it is not an overall score.</li>
        <li>The four categories use different numbers of shots and do not add up to total strokes gained. Rankings need at least 100 shots and 10 rounds; below that they are provisional.</li>
        <li>Adjusted strokes gained is the data provider&apos;s field-adjusted scale, which differs from raw published scoring.</li>
        <li>Small samples cannot establish a trait. Who was in contention is estimated from earlier rounds, and weekend rounds only exist for players who made the cut.</li>
      </ul>
      <details className="pp-details ph-technical" data-testid="player-technical">
        <summary>Technical details</summary>
        <p>PGA average reference: {reference?.id ?? "unavailable"}; usable from {etTime(reference?.available_after, "an unknown date")}. {reference?.n_events ?? "Unknown number of"} events.</p>
        <p>Rating built {etTime(rating?.as_of, "at an unknown time")}; status {rating?.status ?? "unavailable"}; recency-weighted adjusted strokes gained {signedZ(rating?.raw_adjusted_sg_mean, 3)} minus the zero point {signedZ(reference?.adjusted_sg_mean, 3)} = unshrunk form {signedZ(unshrunk, 3)}.</p>
        <p>Weight of the player&apos;s rounds: {finite(rating?.n_eff) ? rating.n_eff.toFixed(2) : "unavailable"} (recency-weighted count) and {finite(rating?.effective_sample_size) ? rating.effective_sample_size.toFixed(2) : "unavailable"} (effective sample size). They measure different things.</p>
        {method && <p>Window {method.window_days} days, half-life {method.half_life_days} days, at least {method.min_rounds} rounds, {method.min_events} events and {method.min_n_eff} recency-weighted rounds.</p>}
      </details>
    </details>
  );
}
