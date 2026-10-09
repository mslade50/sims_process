"use client";

import type { ReactNode } from "react";
import { Badge, FreshnessBadge, Skeleton } from "./ui";
import { LABELS, ZERO_SENTENCE } from "./labels";
import { checkpointBenchmarkValue, benchmarkValue, savedFieldSkill, shortProfileDate, type PlayerProfile, type PgaBenchmarkReference, type ProfileEntry } from "./player-profile-rules";
import type { ExplainDoc } from "./explain-rules";
import type { DeepProfile } from "./player-deep-rules";
import { performanceSeries, visiblePerformance } from "./player-performance-rules";
import {
  CAVEAT_DESCRIPTIVE, CAVEAT_REFERENCE_ONLY, INTERVAL_CLAUSE, LIVE_CENTRING_NOTE, ZERO_LABEL, barGeometry, liveChips, regularTick, signed, sparklinePaths, supportBadge, tourMixBadge,
} from "./player-hero-rules";
import "./player-hero.css";

export type HeroWeekly = {
  eventUid: string | null;
  eventName?: string;
  eventTour?: string | null;
  options: Array<{ uid: string; label: string }>;
  savedDoc: ExplainDoc | null;
  onEvent: (uid: string) => void;
};
export type HeroProps = {
  entry: ProfileEntry;
  profile: PlayerProfile;
  catalog: { regular_tick?: unknown; pga_benchmark_reference?: PgaBenchmarkReference };
  photo: ReactNode;
  weekly: HeroWeekly;
  deep: DeepProfile | null;
  deepLoading: boolean;
  deepError: string | null;
  /** Archived checkpoint as-of, when this page is ever shown for an old checkpoint: PGA-scale numbers keep the as-was gate. */
  checkpoint?: string | null;
};

const pct1 = (v: number | null | undefined) => (typeof v === "number" && Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : "n/a");

function HeadlineBar({ value, tick }: { value: number; tick: { value: number; asOf: string | null } | null }) {
  const g = barGeometry(value, tick?.value ?? null);
  return (
    <div className="ph-bar" role="img" aria-label={`${LABELS.vsPgaAvg.short} ${signed(value)} strokes per round on a scale from ${g.lo} to +${g.hi}${tick ? `; typical PGA regular ${signed(tick.value)}` : ""}`}>
      <span className="ph-bar-fill" data-sign={value >= 0 ? "pos" : "neg"} style={{ left: `${g.left}%`, width: `${g.width}%` }} />
      <span className="ph-bar-zero" style={{ left: `${g.zero}%` }} />
      {g.tick !== null && tick && <span className="ph-bar-tick" data-testid="regular-tick" style={{ left: `${g.tick}%` }} title={`Typical full-time PGA regular ${signed(tick.value)}${tick.asOf ? `, as of ${tick.asOf.slice(0, 10)}` : ""}`} />}
      <span className="ph-bar-scale"><b>{g.lo}</b><b>0</b><b>+{g.hi}</b></span>
      {tick && g.tick !== null && <span className="ph-bar-tick-label" style={{ left: `${g.tick}%` }}>typical regular {signed(tick.value)}</span>}
    </div>
  );
}

function TrendTeaser({ deep, loading, error, anchor }: { deep: DeepProfile | null; loading: boolean; error: string | null; anchor: number | null | undefined }) {
  if (loading) return <div className="ph-trend" data-testid="trend-skeleton" aria-label="Loading trend"><span className="eyebrow">Trend</span><Skeleton lines={2} height={12} /></div>;
  const series = deep ? performanceSeries(deep.history?.events ?? [], anchor, false) : [];
  const visible = deep ? visiblePerformance(series, deep.as_of, "3y") : [];
  const paths = sparklinePaths(visible, 180, 44);
  const latest = series.at(-1);
  if (!paths || !latest) {
    return <div className="ph-trend" data-testid="trend-unavailable"><span className="eyebrow">Trend</span><p className="ph-muted">{error ? "Deep history unavailable." : "Needs 20 eligible adjusted rounds for the first trend line."}</p></div>;
  }
  const diff = typeof latest.sma20 === "number" && typeof latest.sma50 === "number" ? latest.sma20 - latest.sma50 : null;
  return (
    <div className="ph-trend" data-testid="trend-teaser">
      <span className="eyebrow">Trend · R1-R2</span>
      <svg viewBox="0 0 180 44" role="img" aria-label={`Teaser: 20 and 50 round averages, past 3 years. 20-round ${signed(latest.sma20)}, 50-round ${signed(latest.sma50)}.`} preserveAspectRatio="none">
        {paths.zero !== null && <line x1="0" x2="180" y1={paths.zero} y2={paths.zero} className="ph-spark-zero" />}
        {paths.sma50 && <path d={paths.sma50} className="ph-spark-50" fill="none" vectorEffect="non-scaling-stroke" />}
        {paths.sma20 && <path d={paths.sma20} className="ph-spark-20" fill="none" vectorEffect="non-scaling-stroke" />}
      </svg>
      <p className="ph-muted">20-round {signed(latest.sma20)} · 50-round {signed(latest.sma50)}{diff !== null && <> · recent minus longer {signed(diff)}</>}</p>
      <a className="ph-link" href="#player-history">Open the full trend chart</a>
    </div>
  );
}

export function PlayerHero({ entry, profile, catalog, photo, weekly, deep, deepLoading, deepError, checkpoint = null }: HeroProps) {
  const rating = profile.pga_benchmark;
  const reference = profile.pga_benchmark_reference ?? catalog.pga_benchmark_reference;
  const value = checkpointBenchmarkValue(rating, reference, checkpoint);
  const withheld = value === null && !!checkpoint && benchmarkValue(rating, reference) !== null;
  const tick = regularTick(catalog);
  const support = supportBadge(rating);
  const mix = tourMixBadge(rating?.decomposition?.tour_weights, weekly.eventTour ?? weekly.savedDoc?.event.tour ?? null);
  const interval = rating?.decomposition?.historical_mean_interval;
  const dgId = profile.identity.dg_id;
  const model = savedFieldSkill(weekly.savedDoc, dgId);
  const row = weekly.savedDoc?.players.find(p => p.id === dgId && !p.withdrawn);
  const live = weekly.savedDoc?.kind === "live" ? liveChips((row as unknown as { live?: Parameters<typeof liveChips>[0] } | undefined)?.live) : null;
  const unsupported = value === null && !withheld;
  return (
    <section className="ph-hero" data-testid="player-hero" aria-label="Player summary">
      <div className="ph-row ph-identity">
        {photo}
        <div className="ph-id-text">
          <span className="eyebrow">Player</span>
          <h2>{entry.name}</h2>
          <p>{entry.country ?? entry.country_code ?? "Country unavailable"}{(entry.tours ?? []).length > 0 && <> · {(entry.tours ?? []).map(t => t.toUpperCase()).join(" / ")}</>}</p>
        </div>
        <dl className="ph-facts">
          <div><dt>Rounds observed</dt><dd>{entry.n_rounds ?? "Unavailable"}</dd></div>
          <div><dt>Last observation</dt><dd title={entry.last_observation}>{shortProfileDate(entry.last_observation)}</dd></div>
          <div><dt>Profile as of</dt><dd title={profile.as_of}>{shortProfileDate(profile.as_of)} <FreshnessBadge at={profile.as_of} label="Updated" scale="catalog" /></dd></div>
        </dl>
      </div>

      <div className="ph-row ph-stats">
        <article className="ph-headline" title={ZERO_SENTENCE}>
          <span className="eyebrow">{LABELS.vsPgaAvg.short} · {ZERO_LABEL}</span>
          <strong className="ph-number" data-testid="hero-value" title={value === null ? undefined : `${signed(value, 3)} strokes per round`}>
            {value === null ? (withheld ? "Withheld" : "Unavailable") : signed(value)} {value !== null && <small>SG/round</small>}
          </strong>
          {value !== null && <HeadlineBar value={value} tick={tick} />}
          <div className="ph-badges">
            <Badge tone={support.tone}>{support.text}</Badge>
            {mix && <Badge tone="warning">{mix}</Badge>}
            {unsupported && <Badge tone="neutral">Not enough eligible R1-R2 rounds for a stable number</Badge>}
          </div>
          <p className="ph-caveat">{CAVEAT_REFERENCE_ONLY}</p>
          <p className="ph-caveat">{CAVEAT_DESCRIPTIVE}{typeof interval?.lower === "number" && typeof interval?.upper === "number" && <> · 95% interval {signed(interval.lower)} to {signed(interval.upper)} ({INTERVAL_CLAUSE})</>}</p>
          <small className="ph-zero" data-testid="zero-sentence">{ZERO_SENTENCE}</small>
          {withheld && <p className="ph-caveat" role="status">Withheld at this checkpoint: the profile is recomputed from the latest revised history, not as-was, so it would include later rounds or a reference that did not yet exist.</p>}
        </article>

        <article className="ph-week" aria-label="This week">
          <span className="eyebrow">{weekly.eventName ? `This week · ${weekly.eventName}` : "This week"}</span>
          {weekly.savedDoc && row ? (
            <>
              <div className="ph-chips">
                <span className="ph-chip"><b>{signed(model.value)}</b><small>{LABELS.vsField.short}</small></span>
                <span className="ph-chip"><b>{pct1(row.probs.win.model)}</b><small>win</small></span>
                <span className="ph-chip"><b>{pct1(row.probs.top_10.model)}</b><small>top 10</small></span>
              </div>
              {live && (
                <div className="ph-chips" data-testid="live-chips">
                  <span className="ph-chip" title={`Transient within-week latent (not skill); already netted out of the live B8 shift. Centred on all entrants, not the active field.`}><b>{signed(live.transient, 3)}</b><small>this week, transient</small></span>
                  {live.priced !== null && <span className="ph-chip" title={`Live model skill + week latent${live.contentionIncluded ? " + contention" : " (no contention shift published)"}. Week latent is centred on all entrants, not the active field.`}><b>{signed(live.priced, 3)}</b><small>priced strength, next round</small></span>}
                  <small className="ph-chip-note">{LIVE_CENTRING_NOTE}</small>
                </div>
              )}
              <p className="ph-muted">{LABELS.vsField.short} is saved model skill minus the active-field mean; a different estimator from the headline.</p>
            </>
          ) : (
            <p className="ph-muted" data-testid="week-absent">
              {weekly.savedDoc ? "Not active in this saved field." : weekly.eventName ? "No matching saved model run for this field yet." : "No saved field includes this golfer."}{" "}
              {entry.dg_id ? <a className="ph-link" href={`/weekly-players${weekly.eventUid ? `?e=${encodeURIComponent(weekly.eventUid)}` : ""}`}>Show the full field</a> : null}
            </p>
          )}
          {weekly.options.length > 0 && (
            <label className="ph-select">Comparison field
              <select value={weekly.eventUid ?? ""} onChange={e => weekly.onEvent(e.target.value)}>
                {weekly.options.map(o => <option key={o.uid} value={o.uid}>{o.label}</option>)}
              </select>
            </label>
          )}
          {entry.dg_id ? <a className="ph-link" href={`/weekly-players?player=${entry.dg_id}`}>This week&apos;s saved inputs</a> : null}
        </article>

        <TrendTeaser deep={deep} loading={deepLoading} error={deepError} anchor={reference?.adjusted_sg_mean} />
      </div>
    </section>
  );
}
