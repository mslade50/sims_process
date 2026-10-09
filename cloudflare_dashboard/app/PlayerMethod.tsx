"use client";

import { GAP_METHOD_SENTENCE, LABELS, ZERO_SENTENCE } from "./labels";
import { benchmarkValue, finite, signedZ, type PgaBenchmark, type PgaBenchmarkReference } from "./player-profile-rules";
import { CROSS_TOUR_BAND, INTERVAL_CLAUSE, RADAR_LABEL, regularTick } from "./player-hero-rules";

const pct = (v: unknown) => (finite(v) ? `${(v * 100).toFixed(0)}%` : "unavailable");

/** The ONE "Method and caveats" disclosure for the player page (B2). Everything that is not one of the ten load-bearing inline caveats lives here. */
export function PlayerMethod({ rating, reference, catalog }: { rating: PgaBenchmark | undefined; reference: PgaBenchmarkReference | undefined; catalog?: unknown }) {
  const method = rating?.method;
  const interval = rating?.decomposition?.historical_mean_interval;
  const value = benchmarkValue(rating, reference);
  const shrunk = (rating as (PgaBenchmark & { shrunk_value?: number | null; shrink_prior_id?: string }) | undefined);
  const tick = regularTick(catalog);
  return (
    <details className="pp-details ph-method" data-testid="player-method">
      <summary>Method and caveats</summary>
      <h4>The headline: {LABELS.vsPgaAvg.long}</h4>
      <p>{ZERO_SENTENCE}</p>
      <p>Anchor: {reference?.population ?? "PGA benchmark reference unavailable."} Round-weighted mean of published adjusted SG from {reference?.year ?? "the reference year"} PGA rounds {reference?.rounds?.join(" and ") ?? "1 and 2"}, no qualification filter: {signedZ(reference?.adjusted_sg_mean, 3)} adjusted SG/round over {reference?.n_rounds ?? "unavailable"} rounds, {reference?.n_players ?? "unavailable"} players, {reference?.n_events ?? "unavailable"} events. Reference ID {reference?.id ?? "unavailable"}; available after {reference?.available_after ?? "unavailable"}.</p>
      <p>Window and weighting: {method ? `${method.window_days}-day window, ${method.half_life_days}-day half-life, minimum ${method.min_rounds} rounds, ${method.min_events} events and ${method.min_n_eff} decay weight` : "method unavailable"}. Only published provider-adjusted SG rounds count; missing values stay unavailable.</p>
      <p>Shrinkage status: {method ? (method.prior_rounds === 0 ? `none. The headline is the unshrunk (${LABELS.unshrunkForm.short}) value.` : `${method.prior_rounds}-round prior`) : "unavailable"}{rating?.weight_on_data !== undefined && <> Data weight {pct(rating.weight_on_data)}.</>}</p>
      {finite(shrunk?.shrunk_value) && (
        <p data-testid="shrunk-estimate"><b>DataGolf-style shrunk estimate: {signedZ(shrunk.shrunk_value, 3)}</b> SG/round{shrunk.shrink_prior_id ? ` (prior ${shrunk.shrink_prior_id})` : ""}. Shown for reference only; the headline stays unshrunk until the owner decides.</p>
      )}
      <p>Sample naming: <code>n_eff</code> is the decay-weight sum ({finite(rating?.n_eff) ? rating.n_eff.toFixed(2) : "unavailable"}); <code>effective_sample_size</code> is the Kish effective sample ({finite(rating?.effective_sample_size) ? rating.effective_sample_size.toFixed(2) : "unavailable"}). They are different quantities.</p>
      <p>Calculation: recency-weighted provider adjusted SG {signedZ(rating?.raw_adjusted_sg_mean, 3)} minus the fixed anchor {signedZ(reference?.adjusted_sg_mean, 3)} = {signedZ(value, 3)} strokes/round. Rating as of {rating?.as_of ?? "unavailable"}; last observation {rating?.last_observation ?? "unavailable"}; status {rating?.status ?? "unavailable"}.</p>
      <p>Interval: approximate 95% event-cluster sampling interval {finite(interval?.lower) && finite(interval?.upper) ? `${signedZ(interval.lower, 3)} to ${signedZ(interval.upper, 3)}` : "unavailable"}; it {INTERVAL_CLAUSE} and provider revisions. It is not a forecast interval.</p>
      <p>Tour contribution: {rating?.decomposition?.tour_weights?.map(t => `${t.tour.toUpperCase()} ${pct(t.weight_fraction)} (${t.n_rounds} rounds)`).join(" · ") ?? "unavailable"}. Treat cross-tour comparisons as accurate to roughly ±0.2 ({CROSS_TOUR_BAND}); the sign of any gap is a method artifact.</p>
      {tick && <p>The tick on the bar marks a typical full-time PGA regular ({signedZ(tick.value, 2)}{tick.asOf ? `, as of ${tick.asOf.slice(0, 10)}` : ""}), stored with the catalog, not computed in the browser.</p>}
      <p>Revised, not as-was: profiles are recomputed from the latest revised history, so an archived checkpoint cannot show them as known at the time.</p>
      <h4>How this relates to this week&apos;s model number</h4>
      <p>The headline and this week&apos;s saved model skill (&quot;{LABELS.thisWeekPga.short}&quot;) now share the PGA-scale zero; the remaining gap is form versus model.</p>
      <p>{GAP_METHOD_SENTENCE}</p>
      <h4>Other caveats</h4>
      <ul>
        <li>Category bars use observed means against a fixed 3-year PGA reference; there is no category strokes gained on non-PGA/LIV tours. The radar is {RADAR_LABEL}; it is not a composite score and not absolute strength.</li>
        <li>Shot metrics have different opportunity mixes and do not add to total SG. Ranks need at least 100 shots and 10 rounds; below that they are provisional.</li>
        <li>Adjusted SG is the provider&apos;s field-adjusted scale, which differs from the raw published scale.</li>
        <li>Small slices cannot establish a trait. Contention is reconstructed from completed prior rounds, not a certified roster. Weekend rounds are selected by the cut.</li>
      </ul>
    </details>
  );
}
