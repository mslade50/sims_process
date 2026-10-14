"use client";

/**
 * "This week" event page and "Why priced" table + the player drawer (finish shape, score distribution, why-this-price waterfall, competitor overlay).
 * Data: one explain document per run, listed in the index (golfprice/explain_export.py; week runs before the event, live runs after each round). Why priced shows the last pre-tournament run unless the owner flips to live.
 * Design-system tokens/components only (tokens.css, ui.tsx); logic in explain-rules.ts.
 */
import Link from "next/link";
import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import { ArrowDown, ArrowUp, ChevronRight, CloudSun, Search, X } from "lucide-react";
import { EmptyState, Kpi, LoadingState, PageIntro, Panel, SegmentedControl } from "./components";
import { DeltaBar, LegendGroups, MarketRows, PLAYER_COLORS, ProbPair, RangeBar, ScoreFan, ShapeBars, shapeCap, Term, Waterfall } from "./ExplainBits";
import type { FinishSeries } from "./ExplainCharts";
import {
  EDGE_MIN, MARKETS, WINNING_SCORE_MISSING, biggestGaps, eventLede, heroFacts, lastSeenRunRef, plainName, previousRunRef, ptsText, relText, spanText, whatChanged, MARKET_LABEL, MAX_OVERLAY, SHAPE_METHOD, TAG_GROUPS, american, availableMarkets, availableTags, cutLineRows, defaultCompetitors, differView, edgeBadge, edgeCellText,
  explainChoices, explainKeyFor, favourites, fieldStrengthText, filterWhy, toPar, gapText, groupSentence, leaderboard, nameMatches, orderEvents, parseExplain, pct, plainCourseBullets, plainHoleSource, plainText, shapeMap, spreadGapOf, spreadLeanNote, shapeOf, signed, sortWhy, roundsWithoutTeeTimes,
  modalFinish, modalFinishText, waterfallScale, waterfallRowScales, tagLabel, toParText, toggleCompetitor, topK, tourName, valueBets, venueClassLines, waterfall, weatherEffectLines, weatherStatus,
  type Differ, type ExPlayer, type ExplainDoc, type FieldScores, type IndexEvent, type Market, type Shape, type SortKey,
} from "./explain-rules";
import { useDashboardData } from "./data";

// Heavy parts load on demand so the summary (key facts, what changed) renders without Recharts, the weather views or the course-geometry viewer.
const FinishChart = lazy(() => import("./ExplainCharts").then((m) => ({ default: m.FinishChart })));
const HoleStrip = lazy(() => import("./ExplainCharts").then((m) => ({ default: m.HoleStrip })));
const WeatherThisRound = lazy(() => import("./ScoringView").then((m) => ({ default: m.WeatherThisRound })));
const CourseGeometry = lazy(() => import("./CourseGeometry").then((m) => ({ default: m.CourseGeometry })));
const WeatherLinkCard = lazy(() => import("./WeatherBriefView").then((m) => ({ default: m.WeatherLinkCard })));
/** Suspense with a placeholder of the final height, so nothing below jumps when the chunk arrives. */
function Lazy({ height, children }: { height: number; children: React.ReactNode }) {
  return <Suspense fallback={<div className="ex-lazy" style={{ minHeight: height }} aria-hidden="true" />}>{children}</Suspense>;
}
import { parseWeatherExtras } from "./scoring-rules";
import { parseWeather } from "./weather-rules";
import { LABELS } from "./labels";
import { etTime } from "./lib";
import { Badge, Delta, FreshnessBadge } from "./ui";
import "./explain.css";

/* ------------------------------------------------------------------ URL state (?e=<event>&p=<player>&vs=a,b,c&m=<market>) */
type View = "pre" | "live";
type Params = { e: string | null; p: number | null; vs: number[]; m: Market | null; v: View | null };
function readParams(): Params {
  if (typeof window === "undefined") return { e: null, p: null, vs: [], m: null, v: null };
  const q = new URLSearchParams(window.location.search);
  const p = Number(q.get("p"));
  const m = q.get("m") as Market | null;
  const v = q.get("v");
  return { e: q.get("e"), p: Number.isFinite(p) && p > 0 ? p : null, vs: (q.get("vs") ?? "").split(",").map(Number).filter((n) => Number.isFinite(n) && n > 0), m: m && (MARKETS as readonly string[]).includes(m) ? m : null, v: v === "pre" || v === "live" ? v : null };
}
function writeParams(next: Params) {
  if (typeof window === "undefined") return;
  const q = new URLSearchParams();
  if (next.e) q.set("e", next.e);
  if (next.p) q.set("p", String(next.p));
  if (next.vs.length) q.set("vs", next.vs.join(","));
  if (next.m && next.m !== "win") q.set("m", next.m);
  if (next.v) q.set("v", next.v);
  const s = q.toString();
  window.history.replaceState(null, "", `${window.location.pathname}${s ? `?${s}` : ""}`);
}

/* ------------------------------------------------------------------ shell: event picker, loading, empty states */
type Shell = { doc: ExplainDoc; event: IndexEvent; latest: boolean; market: Market; setMarket: (m: Market) => void; openPlayer: (id: number) => void };

function ExplainShell({ eyebrow, title, description, preEventFirst = false, defaultView = "pre", children }: { eyebrow: string; title: string; description: string; preEventFirst?: boolean; defaultView?: View; children: (s: Shell) => React.ReactNode }) {
  const index = useDashboardData<{ events?: IndexEvent[] }>("golfprice/index.json");
  const events = useMemo(() => orderEvents(index.data?.events ?? []), [index.data]);
  const [params, setParams] = useState<Params>(readParams);
  if (index.loading) return <LoadingState label="Loading golfprice events" />;
  if (index.error && index.errorKind === "outage") return <EmptyState title="Can’t reach the model data right now" detail={`${index.error} Nothing has been lost; this page retries on its own, or try again now.`} onRetry={index.refetch} />;
  if (index.error || !events.length) return <EmptyState title="No golfprice events published yet" detail={index.error ?? "Publish a run (python -m golfprice.publish_dashboard) and reload."} />;
  const current = events.find((e) => e.event_uid === params.e) ?? events.find((e) => e.explain_key) ?? events[0];
  const update = (patch: Partial<Params>) => {
    const next = { ...params, ...patch };
    setParams(next);
    writeParams(next);
  };
  return (
    <>
      <PageIntro eyebrow={eyebrow} title={title} description={description}
        controls={events.length > 1 ? (
          <div className="segmented ex-events" role="group" aria-label="Event">
            {events.slice(0, 4).map((e) => {
              const ready = !!e.explain_key || (e.distribution_runs?.length ?? 0) > 0;
              return <button key={e.event_uid} type="button" className={`${current.event_uid === e.event_uid ? "active" : ""}${ready ? "" : " empty"}`} title={ready ? undefined : "No prices to show for this event yet."}
                onClick={() => update({ e: e.event_uid, p: null, vs: [] })}>{[tourName(e.tour), e.name ?? e.event_uid].filter(Boolean).join(" · ")}</button>;
            })}
          </div>
        ) : undefined} />
      <ExplainBody key={current.event_uid} event={current} params={params} update={update} preEventFirst={preEventFirst} defaultView={defaultView}>{children}</ExplainBody>
    </>
  );
}

function ExplainBody({ event, params, update, preEventFirst, defaultView, children }: { event: IndexEvent; params: Params; update: (p: Partial<Params>) => void; preEventFirst: boolean; defaultView: View; children: (s: Shell) => React.ReactNode }) {
  const choices = useMemo(() => explainChoices(event), [event]);
  // Pages that offer the choice start on the chosen default (Why priced: pre-tournament; This week: the latest run) and only show the toggle when a live one exists.
  const offerToggle = preEventFirst && choices.hasLive && !!choices.preKey && !!choices.liveKey;
  const view: View = offerToggle ? params.v ?? defaultView : "live";
  const key = !preEventFirst || !choices.hasLive ? explainKeyFor(event) : view === "pre" && choices.preKey ? choices.preKey : choices.liveKey ?? explainKeyFor(event);
  const { data, loading, error } = useDashboardData<unknown>(key);
  const doc = useMemo(() => parseExplain(data), [data]);
  const toggle = offerToggle && (
    <div className="ex-viewbar">
      <SegmentedControl label="Which pricing to explain" value={view} onChange={(v) => update({ v, p: null, vs: [] })}
        options={[{ value: "pre", label: "Pre-tournament" }, { value: "live", label: choices.liveRound ? `Live (after round ${choices.liveRound})` : "Live" }]} />
      <span className="ex-muted">{view === "pre" ? "Prices as set before the first tee shot." : "Prices updated with scores so far."}</span>
    </div>
  );
  if (loading) return <>{toggle}<LoadingState label="Loading the pricing explanation" /></>;
  const stale = !preEventFirst && event.explain_run && doc && `${doc.kind}_${doc.run}` !== event.explain_run;
  if (error || !doc || stale) return <>{toggle}<EmptyState title={`Nothing to show for ${event.name ?? event.event_uid} yet`} detail="Prices for this event have not been run since this page was updated. They will appear after the next pricing run." /></>;
  const market = params.m ?? "win";
  const player = params.p ? doc.players.find((p) => p.id === params.p) ?? null : null;
  const competitors = params.vs;
  return (
    <>
      <RunBanner doc={doc} />
      {toggle}
      {preEventFirst && choices.hasLive && !choices.preKey && <p className="ex-muted ex-notice">The pre-tournament explanation was not kept for this event, so this shows the live pricing.</p>}
      {children({ doc, event, latest: !!event.explain_run && `${doc.kind}_${doc.run}` === event.explain_run, market, setMarket: (m) => update({ m }), openPlayer: (id) => update({ p: id, vs: [] }) })}
      {player && <PlayerDrawer doc={doc} player={player} competitors={competitors} market={market}
        setCompetitors={(vs) => update({ vs })} onSelect={(id) => update({ p: id, vs: [] })} onClose={() => update({ p: null, vs: [] })} />}
    </>
  );
}

function RunBanner({ doc }: { doc: ExplainDoc }) {
  const live = doc.kind === "live";
  return (
    <div className="ex-banner">
      <Badge tone={live ? "positive" : "model"} pulse={live}>{live ? `In play · after round ${doc.after_round}` : "Prices before the event"}</Badge>
      <span className="ex-banner-name">{doc.event.name}{doc.event.course ? ` · ${doc.event.course}` : ""}</span>
      {doc.as_of && <FreshnessBadge at={doc.as_of} label="Priced" title={`When these prices were produced: ${etTime(doc.as_of)} (US Eastern time)`} />}
      {weatherStatus(doc).short && <span title={weatherStatus(doc).long}><Badge tone="neutral">{weatherStatus(doc).short}</Badge></span>}
      {doc.model.overrides_applied.length > 0 && <Badge tone="warning">{doc.model.overrides_applied.length} manual adjustment{doc.model.overrides_applied.length > 1 ? "s" : ""}</Badge>}
      {doc.finish.method === "unavailable" && <Badge tone="warning">finish chances not available</Badge>}
    </div>
  );
}

const Tags = ({ tags, max = 4 }: { tags: string[]; max?: number }) => <span className="ex-tags">{tags.slice(0, max).map((t) => <em key={t}>{tagLabel(t)}</em>)}{tags.length > max && <em>+{tags.length - max}</em>}</span>;

/* ------------------------------------------------------------------ This week */
export function ThisWeekView() {
  return (
    <ExplainShell eyebrow="Event" title="This week" preEventFirst defaultView="live" description="The favorite, what moved, and where we differ from the sportsbook.">
      {(s) => <ThisWeek {...s} />}
    </ExplainShell>
  );
}

function ThisWeek({ doc, event, latest, market, setMarket, openPlayer }: Shell) {
  const live = doc.kind === "live";
  const weatherKey = event.weather_key || `golfprice/weather/${event.event_uid.replace(/:/g, "_")}/latest.json`;
  const weatherSource = useDashboardData<unknown>(weatherKey);
  const weatherDoc = useMemo(() => parseWeather(weatherSource.data), [weatherSource.data]);
  const weatherExtras = useMemo(() => parseWeatherExtras(weatherSource.data), [weatherSource.data]);
  const fav = useMemo(() => favourites(doc, 10), [doc]);
  const differ = useMemo(() => differView(doc), [doc]);
  const markets = useMemo(() => availableMarkets(doc), [doc]);
  const shownMarket: Market = markets.includes(market) ? market : "win";
  const cc = doc.course_card;
  const ee = [...differ.rows].sort((a, b) => Math.abs(b.shape.skill_gap) - Math.abs(a.shape.skill_gap))[0];
  const fs = doc.event.field_strength ?? doc.event.field_strength_detail?.field_offset ?? null;
  const dgGap = doc.event.field_strength_detail?.typical_ours_minus_dg?.mean ?? null;
  const fsc = doc.field_scores ?? null;
  const bullets = plainCourseBullets(cc.what_drives_scoring);
  const tech = [live ? "" : SHAPE_METHOD, cc.weather?.venue_class ? `Venue class (publisher): ${cc.weather.venue_class}` : "", cc.weather?.wind_slope_source ?? "", cc.weather?.label ?? "", cc.hole_source ? `Hole table: ${cc.hole_source}` : ""].filter(Boolean);
  const contenderIds = differ.differ.ids;
  return (
    <>
      <KeyFacts doc={doc} ids={contenderIds} openPlayer={openPlayer} />
      <WhatChangedStrip doc={doc} event={event} latest={latest} openPlayer={openPlayer} />

      <div className="ex-lede-block">
        <p className="ex-lede">{eventLede(doc)}</p>
        <div className="ex-chips-row" aria-label="Field summary">
          <span className="ex-chip"><Term k="ex_field_strength">Field strength</Term> <b>{fs == null ? "n/a" : `${signed(fs)} strokes`}</b></span>
          {!live && ee && <span className="ex-chip"><Term k="ex_skill_gap">Biggest skill gap</Term> <b className={ee.shape.skill_gap >= 0 ? "pos" : "neg"}>{plainName(ee.player.name)} {signed(ee.shape.skill_gap)}</b></span>}
          {weatherStatus(doc).short && <span className="ex-chip" title={weatherStatus(doc).long}>{weatherStatus(doc).short}</span>}
        </div>
        <details className="ex-details">
          <summary>Details: scoring by round, field strength, method</summary>
          <div className="ex-detail-grid">
            <RoundAverages fsc={fsc} cc={cc} />
            <WinningScore fsc={fsc} />
          </div>
          <p className="ex-muted">Field strength {fs == null ? "is not in this run" : fieldStrengthText(fs)}{fs != null && dgGap != null && Math.abs(dgGap) >= 0.05 ? `, about ${signed(fs - dgGap)} strokes on Data Golf's numbers` : ""}.</p>
          {tech.map((t) => <p className="ex-muted" key={t}>{t}</p>)}
        </details>
      </div>

      <Lazy height={64}><WeatherLinkCard event={event} /></Lazy>

      {live && <LiveSection doc={doc} openPlayer={openPlayer} />}

      {!live && <GapGlance doc={doc} ids={contenderIds} openPlayer={openPlayer} />}

      <div className="two-column ex-two">
        <Panel title="The course" eyebrow={cc.course ?? "Course card"}>
          <Lazy height={230}><HoleStrip holes={cc.holes} /></Lazy>
          <ul className="ex-bullets">{bullets.slice(0, 2).map((b) => <li key={b}>{b}</li>)}</ul>
          {bullets.length > 2 && <details className="ex-details"><summary>{bullets.length - 2} more notes on what drives scoring</summary><ul className="ex-bullets">{bullets.slice(2).map((b) => <li key={b}>{b}</li>)}</ul></details>}
          {cc.hole_source && <p className="ex-muted">{plainHoleSource(cc.hole_source)} Values are for the average player in this field.</p>}
        </Panel>
        <Panel title="Weather" eyebrow="Conditions and course type" actions={<Link className="ex-link" href={`/weather-effects?e=${encodeURIComponent(event.event_uid)}`}>Weather page <ChevronRight size={14} /></Link>}>
          <WeatherSnapshot doc={doc} />
        </Panel>
      </div>
      <Lazy height={120}><CourseGeometry eventUid={event.event_uid} eventName={doc.event.name ?? event.name} /></Lazy>
      <WeatherEffects doc={doc} />
      <Lazy height={160}><WeatherThisRound doc={weatherDoc} extras={weatherExtras} loading={weatherSource.loading} round={live ? (doc.after_round ?? 0) + 1 : 1} latest={latest} /></Lazy>

      {live ? (
        <p className="ex-muted ex-notice">While play is under way we do not compare our prices with the sportsbook: its prices are not live, so a gap would not mean anything. See Why priced for the comparison made before the event.</p>
      ) : (
        <DifferPanel doc={doc} rows={differ.rows} differ={differ.differ} openPlayer={openPlayer} />
      )}

      <Panel title="Headline: our chances" eyebrow={live ? "Ten best chances" : "Ten shortest-priced players"} actions={<SegmentedControl label="Bet type" value={shownMarket} onChange={setMarket} options={markets.map((m) => ({ value: m, label: MARKET_LABEL[m] }))} />}>
        <div className="table-scroll ex-scroll"><table className="ex-table ex-cards">
          <thead><tr><th>Player</th><th title="Our chance for this bet type, from the simulations.">Our chance</th>{!live && <th title="The chance implied by sportsbook odds, with the bookmaker margin removed.">Sportsbook</th>}{!live && <th title="How far our chance is from the sportsbook's, as a share of the sportsbook's chance. Plus means we are higher.">Gap</th>}<th title="The American odds that match our chance.">Our odds</th>{!live && fav.some((p) => p.probs[shownMarket]?.fair != null) && <th title="The price the odds board publishes: our chance blended with the sportsbook consensus using weights fitted on past events. The blend leans further than either source when both agree, so it can sit slightly outside both numbers.">Board price</th>}<th title="The biggest reasons for our rating of this player.">Why</th></tr></thead>
          <tbody>{fav.map((p) => {
            const e = p.probs[shownMarket];
            const b = edgeBadge(p, shownMarket);
            return (
              <tr key={p.id} className="clickable-row" onClick={() => openPlayer(p.id)}>
                <th scope="row"><button type="button" className="ex-linkbtn" onClick={() => openPlayer(p.id)}>{p.name}</button></th>
                <td data-label="Our chance">{pct(e.model)}</td>
                {!live && <td data-label="Sportsbook">{pct(e.market)}</td>}
                {!live && <td data-label="Gap" className={b.tone === "positive" ? "pos" : b.tone === "negative" ? "neg" : ""}>{b.text}</td>}
                <td data-label="Our odds">{american(e.model)}</td>{!live && fav.some((p) => p.probs[shownMarket]?.fair != null) && <td data-label="Board price">{e.fair == null ? "—" : `${pct(e.fair)} · ${american(e.fair)}`}</td>}<td data-label="Why" className="ex-why">{p.why}</td>
              </tr>
            );
          })}</tbody>
        </table></div>
      </Panel>
    </>
  );
}

/* ------------------------------------------------------------------ key facts: the three things to read first */
function KeyFacts({ doc, ids, openPlayer }: { doc: ExplainDoc; ids: number[]; openPlayer: (id: number) => void }) {
  const h = useMemo(() => heroFacts(doc, ids), [doc, ids]);
  const live = doc.kind === "live";
  const f = h.favourite;
  const s = h.second;
  const sc = h.scoring;
  return (
    <section className="ex-hero" aria-label="Key facts">
      {f && (
        <div className="ex-fact">
          <span className="ex-fact-label"><Term k="ex_our_chance">{live ? "Best chance to win" : "Favorite to win"}</Term></span>
          <strong className="ex-fact-value"><button type="button" className="ex-linkbtn" onClick={() => openPlayer(f.id)}>{f.name}</button></strong>
          <span className="ex-fact-sub">{pct(f.model)} ours{f.market !== null ? <> · {pct(f.market)} <Term k="ex_market_chance">sportsbook</Term></> : null}</span>
          <ProbPair model={f.model} market={f.market} scale={Math.max(f.model ?? 0, f.market ?? 0, 0.0001) * 1.15} />
        </div>
      )}
      {s?.kind === "gap" && (
        <div className="ex-fact">
          <span className="ex-fact-label">Biggest <Term k="ex_gap">gap to the sportsbook</Term></span>
          <strong className="ex-fact-value"><button type="button" className="ex-linkbtn" onClick={() => openPlayer(s.id)}>{s.name}</button> <em className={s.rel > 0 ? "pos" : "neg"}>{s.rel > 0 ? "▲" : "▼"} {relText(s.rel)}</em></strong>
          <span className="ex-fact-sub">Win chance {pct(s.model)} ours vs {pct(s.market)} sportsbook</span>
          <DeltaBar value={s.rel} scale={Math.max(0.25, Math.abs(s.rel))} label={`We are ${s.rel > 0 ? "higher" : "lower"} than the sportsbook by ${relText(s.rel)}`} />
        </div>
      )}
      {s?.kind === "move" && (
        <div className="ex-fact">
          <span className="ex-fact-label">Biggest mover since the last run</span>
          <strong className="ex-fact-value"><button type="button" className="ex-linkbtn" onClick={() => openPlayer(s.id)}>{s.name}</button> <em className={s.delta > 0 ? "pos" : "neg"}>{s.delta > 0 ? "▲" : "▼"} {ptsText(s.delta)}</em></strong>
          <span className="ex-fact-sub">Win chance {pct(s.before)} → {pct(s.after)}</span>
          <DeltaBar value={s.delta} scale={Math.max(1, Math.abs(s.delta))} label="Change in win chance" />
        </div>
      )}
      {!s && !live && f && (
        <div className="ex-fact is-quiet">
          <span className="ex-fact-label"><Term k="ex_in_line">In line with the sportsbook</Term></span>
          <strong className="ex-fact-value">No big gaps</strong>
          <span className="ex-fact-sub">No contender&rsquo;s win chance is 5% or more away from the sportsbook&rsquo;s.</span>
        </div>
      )}
      {sc?.kind === "winning" && (
        <div className="ex-fact">
          <span className="ex-fact-label"><Term k="ex_winning_score">Winning score</Term></span>
          <strong className="ex-fact-value">{toPar(sc.median, Number.isInteger(sc.median) ? 0 : 1)} <em>median</em></strong>
          <span className="ex-fact-sub">Most likely {toPar(sc.mode)} ({Math.round(sc.modeShare * 100)}% of sims) · 8 in 10 between {toPar(sc.p10)} and {toPar(sc.p90)}</span>
          <RangeBar p10={sc.p10} p90={sc.p90} mode={sc.mode} median={sc.median} />
        </div>
      )}
      {sc?.kind === "round" && (
        <div className="ex-fact">
          <span className="ex-fact-label"><Term k="ex_scoring_avg">{`Round ${sc.round} scoring average`}</Term></span>
          <strong className="ex-fact-value">{sc.strokes.toFixed(1)} <em>{toPar(sc.vsPar, 1)} vs par</em></strong>
          <span className="ex-fact-sub">{sc.basis === "actual" ? "Actual field average" : "Simulated field average, including the forecast"}</span>
        </div>
      )}
    </section>
  );
}

/* ------------------------------------------------------------------ what changed since the previous run (or your last visit) */
const SEEN_AT_LOAD = new Map<string, string | null>();
/** The run this browser last showed for the event (read once per page load, then moved to the current run when it is the latest). */
function useLastSeen(eventUid: string, currentId: string, latest: boolean): string | null {
  const [seen, setSeen] = useState<string | null>(null);
  useEffect(() => {
    let prior: string | null = null;
    try {
      const k = `golfprice-seen:${eventUid}`;
      if (!SEEN_AT_LOAD.has(eventUid)) SEEN_AT_LOAD.set(eventUid, window.localStorage.getItem(k));
      prior = SEEN_AT_LOAD.get(eventUid) ?? null;
      if (latest) window.localStorage.setItem(k, currentId);
    } catch { /* storage unavailable: no "since your last visit" option */ }
    queueMicrotask(() => setSeen(prior));
  }, [eventUid, currentId, latest]);
  return seen;
}

function WhatChangedStrip({ doc, event, latest, openPlayer }: { doc: ExplainDoc; event: IndexEvent; latest: boolean; openPlayer: (id: number) => void }) {
  const seen = useLastSeen(event.event_uid, `${doc.kind}_${doc.run}`, latest);
  const prevRef = useMemo(() => previousRunRef(event, doc), [event, doc]);
  const visitRef = useMemo(() => lastSeenRunRef(event, doc, seen), [event, doc, seen]);
  const [against, setAgainst] = useState<"prev" | "visit">("prev");
  const ref = against === "visit" && visitRef ? visitRef : prevRef;
  const source = useDashboardData<unknown>(ref?.key ?? null);
  const prevDoc = useMemo(() => parseExplain(source.data), [source.data]);
  const change = useMemo(() => (source.loading ? null : whatChanged(doc, prevDoc)), [doc, prevDoc, source.loading]);
  const [mk, setMk] = useState<Market>("win");
  if (!ref && !doc.since_last) return null;
  if (source.loading) return <section className="ex-changed is-loading" aria-label="What changed" aria-busy="true"><h2>What changed</h2><span className="ex-muted">Comparing with the earlier run…</span></section>;
  if (!change) return null;
  const mkts = (Object.keys(change.movers) as Market[]).filter((m) => change.movers[m]?.length);
  const shownMk: Market = mkts.includes(mk) ? mk : mkts[0] ?? "win";
  const rows = (change.movers[shownMk] ?? []).slice(0, 5);
  const scale = Math.max(0.5, ...rows.map((r) => Math.abs(r.delta)));
  const baseKind = change.baseline.after_round ? `the run after round ${change.baseline.after_round}` : "the previous run";
  const fromWhat = against === "visit" && visitRef ? "your last visit" : baseKind;
  const span = spanText(change.gapMs);
  const w = change.weather;
  const sc = change.scoring;
  return (
    <section className={`ex-changed${change.quiet ? " is-quiet" : ""}`} aria-label="What changed">
      <header>
        <h2>What changed</h2>
        <span className="ex-muted">since {fromWhat}{change.baseline.as_of ? ` (${etTime(change.baseline.as_of)}${span ? `, ${span} earlier` : ""})` : ""}</span>
        {visitRef && <SegmentedControl label="Compare with" value={against} onChange={setAgainst} options={[{ value: "prev", label: "Previous run" }, { value: "visit", label: "Your last visit" }]} />}
      </header>
      {change.quiet ? (
        <p className="ex-changed-none">Nothing material: no win chance moved by 0.1 points or more, the field and forecast are unchanged.</p>
      ) : (
        <ul className="ex-changed-chips">{change.headline.map((h) => <li key={h}>{h}</li>)}</ul>
      )}
      {change.partial && <p className="ex-muted">Price moves only: the earlier run&rsquo;s full document is not available here.</p>}
      {rows.length > 0 && (
        <>
          {mkts.length > 1 && <SegmentedControl label="Bet type" value={shownMk} onChange={setMk} options={mkts.map((m) => ({ value: m, label: MARKET_LABEL[m] }))} />}
          <ul className="ex-movers-list" aria-label={`Biggest ${MARKET_LABEL[shownMk]} price moves`}>{rows.map((m) => (
            <li key={m.id}>
              <button type="button" onClick={() => openPlayer(m.id)}>
                <span className="ex-mv-name">{m.name}</span>
                <DeltaBar value={m.delta} scale={scale} label={`${m.name} ${ptsText(m.delta)}`} />
                <span className="ex-mv-nums">{pct(m.before)} → {pct(m.after)}</span>
                <b className={m.delta > 0 ? "pos" : "neg"}>{m.delta > 0 ? "▲" : "▼"} {ptsText(m.delta)}</b>
                {m.marketDelta !== null && Math.abs(m.marketDelta) >= 0.05 && <small className="ex-mv-market">sportsbook {ptsText(m.marketDelta)}</small>}
              </button>
            </li>
          ))}</ul>
        </>
      )}
      {(change.field.added.length + change.field.removed.length + change.field.withdrawn.length + w.rounds.length + sc.rounds.length + change.gaps.length + change.overrides.added.length + change.overrides.removed.length > 0 || w.layer || sc.winning) && (
        <details className="ex-details">
          <summary>Field, forecast and market-gap changes</summary>
          <ul className="ex-changed-list">
            {change.field.withdrawn.length > 0 && <li>Withdrew: {change.field.withdrawn.join(", ")}</li>}
            {change.field.removed.length > 0 && <li>Left the field: {change.field.removed.join(", ")}</li>}
            {change.field.added.length > 0 && <li>Added to the field: {change.field.added.join(", ")}</li>}
            {w.rounds.map((r) => <li key={`w${r.round}`}>Round {r.round} forecast effect on the whole field: {signed(r.before)} → {signed(r.after)} strokes a round (<span className={r.delta < 0 ? "pos" : "neg"}>{r.delta < 0 ? "easier" : "harder"} by {Math.abs(r.delta).toFixed(2)}</span>)</li>)}
            {w.layer && <li>Weather model changed from {w.layer.from} to {w.layer.to}</li>}
            {sc.rounds.map((r) => <li key={`s${r.round}`}>Expected round {r.round} scoring average: {toPar(r.before, 1)} → {toPar(r.after, 1)}</li>)}
            {sc.winning && <li>Median winning score: {toPar(sc.winning.before, 1)} → {toPar(sc.winning.after, 1)}</li>}
            {change.gaps.slice(0, 5).map((g) => <li key={`g${g.id}`}>{g.name}: win gap to the sportsbook {relText(g.before)} → {relText(g.after)}</li>)}
            {change.overrides.added.length > 0 && <li>Manual adjustments added: {change.overrides.added.join(", ")}</li>}
            {change.overrides.removed.length > 0 && <li>Manual adjustments removed: {change.overrides.removed.join(", ")}</li>}
          </ul>
        </details>
      )}
    </section>
  );
}

/* ------------------------------------------------------------------ where we differ most: the glance list */
function GapGlance({ doc, ids, openPlayer }: { doc: ExplainDoc; ids: number[]; openPlayer: (id: number) => void }) {
  const g = useMemo(() => biggestGaps(doc, 3, ids), [doc, ids]);
  if (!g.higher.length && !g.lower.length) return null;
  const list = (rows: typeof g.higher, title: string) => rows.length > 0 && (
    <div className="ex-gap-col">
      <h3 className="ex-h3">{title}</h3>
      <ul>{rows.map((r) => (
        <li key={r.id}>
          <button type="button" onClick={() => openPlayer(r.id)}>
            <span className="ex-mv-name">{r.name}</span>
            <DeltaBar value={r.rel} scale={g.scale} label={`${r.name}: our win chance ${relText(r.rel)} against the sportsbook`} />
            <span className="ex-mv-nums">{pct(r.model)} vs {pct(r.market)}</span>
            <b className={r.rel > 0 ? "pos" : "neg"}>{r.rel > 0 ? "▲" : "▼"} {relText(r.rel)}</b>
          </button>
        </li>
      ))}</ul>
    </div>
  );
  return (
    <Panel title="Where we differ most" eyebrow="Win chance, ours vs the sportsbook">
      <div className="ex-gap-grid">
        {list(g.higher, "We are higher")}
        {list(g.lower, "We are lower")}
      </div>
      <p className="ex-muted">Percent is how far our chance is from the sportsbook&rsquo;s <Term k="ex_gap">(what is this?)</Term>. Contenders only; the full bet-by-bet table is further down.</p>
    </Panel>
  );
}

/** Expected field scoring average for each round from the simulation draws (rounds 3-4: the players who made the cut); played rounds show the actual average. */
function RoundAverages({ fsc, cc }: { fsc: FieldScores | null; cc: ExplainDoc["course_card"] }) {
  const rounds = fsc?.rounds ?? [];
  const foot = `par ${cc.par_per_round?.[0] ?? "-"} · ${cc.yardage ? `${Math.round(cc.yardage).toLocaleString()} yd` : "yardage n/a"}${rounds.some((r) => r.round > 2) ? " · rounds 3-4: players who made the cut" : ""}`;
  return (
    <div className="ex-stat">
      <span className="ex-stat-label"><Term k="ex_scoring_avg">Expected scoring average</Term></span>
      {rounds.length ? (
        <div className="ex-kpi-rounds">{rounds.map((r) => (
          <div key={r.round} className={r.basis === "actual" ? "is-actual" : ""} title={`${r.basis === "actual" ? "Actual" : "Simulated"} round ${r.round} average over ${Math.round(r.players)} players: ${r.mean_strokes.toFixed(2)} (${toPar(r.mean_vs_par, 2)})`}>
            <em>R{r.round}{r.basis === "actual" ? " ✓" : ""}</em><strong>{r.mean_strokes.toFixed(1)}</strong><small>{toPar(r.mean_vs_par, 1)}</small>
          </div>
        ))}</div>
      ) : (
        <div className="ex-stat-one"><strong>{cc.scoring_avg_vs_par == null || !cc.par_per_round?.[0] ? "n/a" : (cc.par_per_round[0] + cc.scoring_avg_vs_par).toFixed(1)}</strong></div>
      )}
      <small className="ex-muted">{rounds.length ? foot : `Course table only (no simulation draws in this run) · ${foot}`}</small>
    </div>
  );
}

/** The winning 72-hole score across the simulation draws: the most likely value, the median and the middle 80%. Without draws it says why it is empty. */
function WinningScore({ fsc }: { fsc: FieldScores | null }) {
  const w = fsc?.winning;
  return (
    <div className="ex-stat">
      <span className="ex-stat-label"><Term k="ex_winning_score">Winning score</Term></span>
      {w ? (
        <>
          <div className="ex-kpi-rounds ex-kpi-two">
            <div><em>Most likely</em><strong>{toPar(w.mode)}</strong><small>{Math.round(w.mode_share * 100)}% of sims</small></div>
            <div><em>Median</em><strong>{toPar(w.median, Number.isInteger(w.median) ? 0 : 1)}</strong><small>mean {toPar(w.mean, 1)}</small></div>
          </div>
          <RangeBar p10={w.p10} p90={w.p90} mode={w.mode} median={w.median} />
          <small className="ex-muted">8 in 10 sims won between {toPar(w.p10)} and {toPar(w.p90)}</small>
        </>
      ) : <small className="ex-muted">{WINNING_SCORE_MISSING}</small>}
    </div>
  );
}

/** Rows shown before "Show all": the whole contender list is ~14,000 px tall on a phone in card layout. */
const DIFFER_PREVIEW = 12;
const SORT_DIFFER = [{ value: "price", label: "Shortest price first" }, { value: "gap", label: "Biggest difference first" }] as const;
const MARKET_HELP: Record<Market, string> = {
  win: "Win outright.", top_5: "Finish in the top 5 (ties share the payout).", top_10: "Finish in the top 10 (ties share the payout).", top_20: "Finish in the top 20 (ties share the payout).",
  make_cut: "Make the cut. When our chance is lower, the value is on the miss-cut side, shown as 'miss cut'.",
};
/** Contenders only: bet-by-bet gaps against the sportsbook, the pattern in words and the split of the skill gap; then player types among them by summed chances. */
function DifferPanel({ doc, rows, differ, openPlayer }: { doc: ExplainDoc; rows: Array<{ player: ExPlayer; shape: Shape }>; differ: Differ; openPlayer: (id: number) => void }) {
  const [sort, setSort] = useState<"price" | "gap">("price");
  const [showAll, setShowAll] = useState(false);
  const markets = availableMarkets(doc).filter((m) => rows.some((r) => r.shape.markets[m]));
  const shown = sort === "price" ? rows : [...rows].sort((a, b) => Math.abs(b.shape.skill_gap) + Math.abs(spreadGapOf(b.shape)) - Math.abs(a.shape.skill_gap) - Math.abs(spreadGapOf(a.shape)));
  const sg = differ.spread_gap_median;
  const fieldLine = spreadLeanNote(sg, differ.spread_gap_centered === true);
  const cap = useMemo(() => shapeCap(rows.map((r) => r.shape), markets), [rows, markets]);
  if (!rows.length) return <Panel title="All contenders, bet by bet" eyebrow="Where we differ from the market"><p className="ex-muted">No contender has sportsbook prices for enough bet types in this run to compare.</p></Panel>;
  return (
    <Panel title="All contenders, bet by bet" eyebrow="Where we differ from the market" actions={<SegmentedControl label="Order" value={sort} onChange={setSort} options={SORT_DIFFER.map((o) => ({ value: o.value, label: o.label }))} />}>
      <p className="ex-muted ex-differ-rule" title="Long shots with thin records swing wildly in percentage terms and would crowd out the players a bet actually depends on.">{differ.rule}</p>
      {fieldLine && <p className="ex-differ-field">{fieldLine}</p>}
      <div className="table-scroll ex-scroll"><table className="ex-table ex-cards ex-differ">
        <thead><tr><th>Player</th>
          <th title={`One bar per bet type: up means our chance is higher than the sportsbook's, down means lower, gray means within 5% (in line). Bar height grows with the square root of the gap; a full bar is a ${Math.round(cap * 100)}% gap (bigger gaps are cut off with a notch).`}>Shape</th>
          {markets.map((m) => <th key={m} title={`${MARKET_HELP[m]} The % is how much higher (+) or lower (−) our chance is than the sportsbook's with its margin removed; under ${Math.round(EDGE_MIN * 100)}% counts as in line.`}>{MARKET_LABEL[m]}</th>)}
          <th title="The pattern across bet types in words, with the skill and the spread (how much their scores swing round to round) that the sportsbook's prices imply, compared with ours.">What we see</th></tr></thead>
        <tbody>{(showAll ? shown : shown.slice(0, DIFFER_PREVIEW)).map(({ player: p, shape: s }) => {
          const value = valueBets(s);
          return (
            <tr key={p.id} className="clickable-row" onClick={() => openPlayer(p.id)}>
              <th scope="row"><button type="button" className="ex-linkbtn" onClick={(ev) => { ev.stopPropagation(); openPlayer(p.id); }}>{p.name}</button></th>
              <td data-label="Shape" className="ex-shape-cell"><ShapeBars shape={s} markets={markets} cap={cap} /></td>
              {markets.map((m) => {
                const e = s.markets[m];
                return <td key={m} data-label={MARKET_LABEL[m]} title={e ? `Ours ${pct(e.model)}, sportsbook ${pct(e.market)} (margin removed)` : "No sportsbook price"}>
                  {e ? <span className="ex-differ-cell"><span className={e.tone === "pos" || (e.tone === "neg" && e.side === "miss_cut") ? "pos" : e.tone === "neg" ? "neg" : "ex-muted"}>{edgeCellText(e)}</span><small className="ex-differ-nums">{pct(e.model)} vs {pct(e.market)}</small></span> : "-"}
                </td>;
              })}
              <td data-label="What we see" className="ex-why ex-differ-why">
                <strong>{s.sentence}</strong>
                <span className="ex-differ-sub" title="Skill: strokes a round better than the field average. Our week-specific adjustments are course fit, travel and home country, weather and tee times, and any owner adjustment. The sportsbook does not say which parts it prices, so the rest is shown as one number.">{s.parts_sentence}</span>
                {value.length > 0 && <span className="ex-differ-value">Value against the sportsbook: {value.join(", ")}</span>}
              </td>
            </tr>
          );
        })}</tbody>
      </table></div>
      {shown.length > DIFFER_PREVIEW && (
        <button type="button" className="ex-showall" aria-expanded={showAll} onClick={() => setShowAll((v) => !v)}>
          {showAll ? `Show the first ${DIFFER_PREVIEW}` : `Show all ${shown.length} contenders`}
        </button>
      )}
      {differ.groups.length > 0 ? (
        <>
          <h3 className="ex-h3">Player types among these contenders</h3>
          <ul className="ex-differ-groups">{differ.groups.map((g) => (
            <li key={g.key}>{groupSentence({ ...g, label: g.label.charAt(0).toUpperCase() + g.label.slice(1) })} <span className="ex-muted">On average we rate them {signed(g.skill_gap)} strokes a round against the market, spread {signed(g.spread_gap)}.</span></li>
          ))}</ul>
        </>
      ) : <p className="ex-muted">No player type among these contenders stands out: every group is within 10% of the sportsbook on win and 5% on top 20.</p>}
      <p className="ex-muted">Percentages compare our chance with the sportsbook&rsquo;s after removing its margin; a real bet also has to beat the margin. The skill and spread the sportsbook implies come from running its prices back through the same curves our own chances follow, so the two are compared like for like. Groups add up chances, so a group shown above the market really does hold more chance with us.</p>
    </Panel>
  );
}

const WAVE_ORDER = ["AM", "PM"];
const waveName = (w: string): string => (w === "AM" ? "Morning wave" : w === "PM" ? "Afternoon wave" : `${w} wave`);
function WeatherSnapshot({ doc }: { doc: ExplainDoc }) {
  const w = doc.course_card.weather ?? {};
  const status = weatherStatus(doc);
  const venue = venueClassLines(w);
  if (!w || (!w.venue_class && !w.label && !status.long && !venue.length)) return <p className="ex-muted">No weather information in this run.</p>;
  const cut = cutLineRows(doc);
  const after = doc.kind === "live" ? doc.after_round ?? 0 : 0;
  const waves = (w.waves ?? []).filter((x) => x.round > after);
  const waveNames = [...new Set(waves.map((x) => x.wave))].sort((a, b) => (WAVE_ORDER.indexOf(a) < 0 ? 9 : WAVE_ORDER.indexOf(a)) - (WAVE_ORDER.indexOf(b) < 0 ? 9 : WAVE_ORDER.indexOf(b)) || a.localeCompare(b));
  const fieldRows = (w.field ?? []).filter((f) => f.round > after);
  const rounds = [...new Set([...waves.map((x) => x.round), ...fieldRows.map((f) => f.round)])].sort((a, b) => a - b);
  const noTee = roundsWithoutTeeTimes(doc);
  return (
    <div className="ex-wx">
      {w.issued_at && <p className="ex-wx-issued">Forecast issued {etTime(w.issued_at)}</p>}
      {status.long && <details className="ex-details"><summary>{status.short || "How the forecast is used in the prices"}</summary><p className="ex-muted">{status.long}</p></details>}
      {venue.length > 0 && <ul className="ex-bullets">{venue.map((v) => <li key={v}>{v}</li>)}</ul>}
      {rounds.length > 0 && (
        <>
          <div className="table-scroll ex-scroll"><table className="ex-table ex-wx-rounds">
            <caption className="ex-muted">Expected effect on scores, strokes a round (minus means easier)</caption>
            <thead><tr><th>Round</th><th title="How much the forecast changes scoring for the whole field compared with typical conditions.">Whole field</th>{waveNames.map((n) => <th key={n} title="Strokes this wave is expected to play easier (minus) or harder (plus) than the field average, from the wind and temperature its tee times face.">{waveName(n)}</th>)}{cut.length > 0 && <th title="How far the forecast is expected to move the cut line.">Cut line</th>}</tr></thead>
            <tbody>{rounds.map((r) => {
              const f = fieldRows.find((x) => x.round === r);
              const c = cut.find((x) => x.round === r);
              return <tr key={r}><th scope="row">Round {r}</th><td>{f?.common == null ? "-" : signed(f.common)}</td>
                {waveNames.map((n) => { const x = waves.find((y) => y.round === r && y.wave === n); return <td key={n} className={x?.rel == null ? "" : x.rel > 0.004 ? "neg" : x.rel < -0.004 ? "pos" : ""}>{x?.rel == null ? (noTee.includes(r) ? <span className="ex-muted" title="No tee times yet for this round">no tee times</span> : "-") : <>{signed(x.rel, 3)}{x.mean_wind != null && <small className="ex-modal"> wind {x.mean_wind} mph</small>}</>}</td>; })}
                {cut.length > 0 && <td>{c ? signed(c.shift) : "-"}</td>}</tr>;
            })}</tbody></table></div>
          <p className="ex-muted">Minus means easier than the typical or average, plus means harder.{doc.kind === "live" ? " Only the rounds still to play are listed." : ""} Wave columns appear once tee times are out; the whole-field effect comes from the forecast alone.</p>
        </>
      )}
      {!waves.length && !fieldRows.length && w.label && !status.long && <p className="ex-muted">{plainText(w.label)}</p>}
    </div>
  );
}

/** The weather effects behind the prices, one line each, with how far apart the field is on it. */
function WeatherEffects({ doc }: { doc: ExplainDoc }) {
  const lines = useMemo(() => weatherEffectLines(doc), [doc]);
  if (!lines.length) return null;
  const used = lines.filter((l) => l.spread >= 0.005);
  const biggest = Math.max(0, ...used.map((l) => l.spread));
  return (
    <Panel title="How weather enters the prices" eyebrow="Per-player effects, strokes a round">
      <p className="ex-wx-sum">{used.length ? <><b>{used.length}</b> weather {used.length === 1 ? "effect separates" : "effects separate"} players; the largest moves one player by <b>{biggest.toFixed(2)}</b> strokes a round.</> : "No player is separated by weather in this run yet; it starts to matter once tee times are out."}</p>
      <details className="ex-details">
        <summary>Effect by effect, with who it helps and hurts</summary>
        <p className="ex-muted">Each effect adds to or takes from a player&rsquo;s expected score depending on their tee time and the weather their group faces. Morning conditions such as firmness are covered by time of day; dew and humidity are not modeled as separate effects.</p>
        <div className="table-scroll ex-scroll"><table className="ex-table ex-cards">
          <thead><tr><th>Effect</th><th>What it is</th><th title="The biggest effect on any one player, in strokes a round.">Largest effect</th><th>Helped most</th><th>Hurt most</th></tr></thead>
          <tbody>{lines.map((l) => (
            <tr key={l.key}><th scope="row">{l.label}</th><td data-label="What it is" className="ex-why">{l.what}</td>
              <td data-label="Largest effect">{l.spread < 0.005 ? <span className="ex-muted">none</span> : `${l.spread.toFixed(2)} strokes`}</td>
              <td data-label="Helped most">{l.helped ? <span className="pos">{l.helped.name} {signed(l.helped.value)}</span> : <span className="ex-muted">-</span>}</td>
              <td data-label="Hurt most">{l.hurt ? <span className="neg">{l.hurt.name} {signed(l.hurt.value)}</span> : <span className="ex-muted">-</span>}</td></tr>
          ))}</tbody></table></div>
      </details>
    </Panel>
  );
}

function LiveSection({ doc, openPlayer }: { doc: ExplainDoc; openPlayer: (id: number) => void }) {
  const lb = leaderboard(doc, 12);
  const withB8 = doc.players.filter((p) => Math.abs(p.live?.d_b8_since_last ?? 0) >= 0.02).length;
  const withCon = doc.players.filter((p) => Math.abs(p.live?.d_contention_since_last ?? 0) >= 0.02).length;
  const hasCon = lb.some((p) => p.live?.contention !== null && p.live?.contention !== undefined && Math.abs(p.live.contention) > 0.0005);
  const hasB8 = lb.some((p) => p.live?.b8_delta !== null && p.live?.b8_delta !== undefined);
  const prev = doc.since_last ? (doc.since_last.prev_after_round !== null && doc.since_last.prev_after_round !== undefined ? `the run after round ${doc.since_last.prev_after_round}` : "the pre-tournament prices") : null;
  return (
    <Panel title={`In play: after round ${doc.after_round}`} eyebrow="Leaderboard and changes since the last run" actions={prev ? <span className="ex-muted">changes compared with {prev}</span> : undefined}>
      <p className="ex-muted">{withB8 > 0 ? `Form this week changed the prices of ${withB8} players since the previous run` : "No player's price moved much on form this week since the previous run"}{withCon > 0 ? `, and the title race changed ${withCon}` : ""}. Weather for the next round: see Weather effects.</p>
      <div className="table-scroll"><table className="ex-table ex-cards">
        <thead><tr><th>Pos</th><th>Player</th><th title="Score to par so far.">To par</th><th title="Our chance to win now.">Win</th><th title="Change in win chance since the previous run, in percentage points.">Win change (points)</th><th>Top 10</th><th title="Our estimate of their skill now: strokes per round better (+) or worse (-) than the field average.">Skill now</th>{hasB8 && <th title="How much their play this week has moved our rating of them, in strokes per round.">Form this week</th>}{hasCon && <th title="How much being near the lead changes their chances.">Title race</th>}</tr></thead>
        <tbody>{lb.map((p) => {
          const d = (p.live?.since_last.win ?? 0) * 100;
          return (
            <tr key={p.id} className="clickable-row" onClick={() => openPlayer(p.id)}>
              <td data-label="Pos">{p.live?.tier === 0 ? p.live?.pos ?? "-" : p.live?.tier === 2 ? "MC" : "WD"}</td>
              <th scope="row"><button type="button" className="ex-linkbtn" onClick={() => openPlayer(p.id)}>{p.name}</button></th>
              <td data-label="To par">{toParText(p.live?.to_par)}</td><td data-label="Win">{pct(p.probs.win.model)}</td>
              <td data-label="Win change">{Math.abs(d) < 0.05 ? <span className="ex-muted">no change</span> : <Delta value={d} digits={1} suffix=" pts" />}</td>
              <td data-label="Top 10">{pct(p.probs.top_10.model)}</td><td data-label="Skill now">{signed(p.live?.mu_live)}</td>
              {hasB8 && <td data-label="Form this week">{signed(p.live?.b8_delta)}</td>}{hasCon && <td data-label="Title race">{signed(p.live?.contention)}</td>}
            </tr>
          );
        })}</tbody>
      </table></div>
    </Panel>
  );
}

/* ------------------------------------------------------------------ Why priced */
export function WhyPricedView() {
  return (
    <ExplainShell eyebrow="Pricing" title="Why priced" preEventFirst description="Every player's model price against the market, sortable and filterable by player type, with the strokes that explain it.">
      {(s) => <WhyPriced {...s} />}
    </ExplainShell>
  );
}

const SORT_LABEL: Array<{ key: SortKey; label: string }> = [
  { key: "edge_sg", label: "Gap in strokes" }, { key: "rel", label: "Gap in %" }, { key: "model", label: "Our chance" }, { key: "market", label: "Sportsbook" }, { key: "mu", label: "Expected skill" }, { key: "name", label: "Name" },
];
const SORT_HELP: Partial<Record<SortKey, string>> = {
  edge_sg: "Gap in strokes: how many strokes a round better (+) or worse (-) our numbers rate a player than the sportsbook prices imply, allowing for how boom-or-bust each side sees them.",
  rel: "Gap in %: how far our chance is from the sportsbook's, as a share of the sportsbook's chance.",
  model: "Our chance for the chosen bet type.", market: "The sportsbook's chance for the chosen bet type.",
  mu: "Expected skill: strokes per round better (+) or worse (-) than the average player in this field.", name: "Alphabetical.",
};
const COLUMN_HELP = {
  pos: "Current place on the leaderboard (MC = missed the cut, WD = withdrew).",
  model: "Our chance for this bet type, from the simulations.",
  market: "The chance implied by sportsbook odds, with the bookmaker margin removed.",
  rel: "How far our chance is from the sportsbook's, as a share of the sportsbook's. Plus means we are higher than the sportsbook.",
  edge_sg: "Skill gap: how many strokes a round better (+) or worse (-) we rate them than the sportsbook prices imply, after allowing for how much their scores swing (spread). Blank when the sportsbook prices too few bet types to tell.",
  mu: "Our estimate of their skill: strokes per round better (+) or worse (-) than the average player in this field.",
  sd: "Std dev: the standard deviation of their round scores, in strokes. Higher means a streakier, less predictable player.",
} as const;

function WhyPriced({ doc, market, setMarket, openPlayer }: Shell) {
  const [query, setQuery] = useState("");
  const [tags, setTags] = useState<string[]>([]);
  const [only, setOnly] = useState<"all" | "above" | "below">("all");
  const [top, setTop] = useState<"all" | "60" | "25">("60");
  const [sort, setSort] = useState<{ key: SortKey; dir: "asc" | "desc" }>({ key: "edge_sg", dir: "desc" });
  const live = doc.kind === "live";
  const markets = useMemo(() => availableMarkets(doc), [doc]);
  const tagOptions = useMemo(() => availableTags(doc.players), [doc.players]);
  // Before the event the strokes gap and the reason come from the bet-by-bet shape comparison (skill the sportsbook's prices imply vs ours); players it cannot be computed for show no gap.
  const players = useMemo(() => {
    if (live) return doc.players;
    const map = shapeMap(doc);
    return doc.players.map((p) => { const s = p.withdrawn ? null : shapeOf(p, map, markets); return { ...p, edge_sg: s ? s.skill_gap : null, why_edge: s ? s.sentence : null }; });
  }, [doc, live, markets]);
  const rows = useMemo(() => sortWhy(filterWhy(players, { query, tags, market, onlyEdge: only, top: top === "all" ? null : Number(top) }), sort.key, sort.dir, market), [players, query, tags, market, only, top, sort]);
  const hasMarket = !live && rows.some((p) => p.probs[market].market !== null);
  const hasModel = rows.some((p) => p.probs[market].model !== null);
  const hasGap = rows.some((p) => p.edge_sg !== null);
  const hasSd = rows.some((p) => p.sd !== null && p.sd !== undefined);
  const th = (key: SortKey, label: string, help?: string) => (
    <th aria-sort={sort.key === key ? (sort.dir === "asc" ? "ascending" : "descending") : "none"} title={help}>
      <button type="button" onClick={() => setSort((s) => ({ key, dir: s.key === key && s.dir === "desc" ? "asc" : "desc" }))}>
        {label}{sort.key === key && (sort.dir === "asc" ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
      </button>
    </th>
  );
  return (
    <Panel title={`${rows.length} players`} eyebrow={MARKET_LABEL[market]} actions={<SegmentedControl label="Bet type" value={market} onChange={setMarket} options={markets.map((m) => ({ value: m, label: MARKET_LABEL[m] }))} />}>
      <p className="ex-muted ex-marketnote">The strokes gap and its reasons are the same for every bet type; only the % gap changes.{live ? " Sportsbook prices are not shown during play because they are not live." : ""}</p>
      <div className="ex-filters">
        <label className="search-box"><Search size={15} /><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search player" aria-label="Search player" /></label>
        <SegmentedControl label="Show players we rate" value={only} onChange={setOnly} options={[{ value: "all", label: "All" }, { value: "above", label: "We are higher" }, { value: "below", label: "We are lower" }]} />
        <SegmentedControl label="Show" value={top} onChange={setTop} options={[{ value: "25", label: "25 shortest prices" }, { value: "60", label: "60 shortest prices" }, { value: "all", label: "Whole field" }]} />
        <label className="ex-sort">Sort <select value={sort.key} title={SORT_HELP[sort.key]} onChange={(e) => setSort({ key: e.target.value as SortKey, dir: e.target.value === "name" ? "asc" : "desc" })}>{SORT_LABEL.map((o) => <option key={o.key} value={o.key} title={SORT_HELP[o.key]}>{o.label}</option>)}</select>
          <button type="button" className="ex-sortdir" onClick={() => setSort((s) => ({ ...s, dir: s.dir === "asc" ? "desc" : "asc" }))} title="Reverse the order" aria-label={sort.dir === "asc" ? "Sorted low to high, tap to reverse" : "Sorted high to low, tap to reverse"}>{sort.dir === "asc" ? <ArrowUp size={14} /> : <ArrowDown size={14} />}{sort.dir === "asc" ? "low to high" : "high to low"}</button></label>
      </div>
      <div className="ex-chipgroups" role="group" aria-label="Filter by player type">
        {TAG_GROUPS.map((g) => {
          const opts = g.tags.filter((t) => tagOptions.includes(t));
          return opts.length === 0 ? null : (
            <div className="ex-chipgroup" key={g.label}>
              <span className="ex-chipgroup-label" title={g.help}>{g.label}</span>
              <div className="ex-chips">{opts.map((t) => <button key={t} type="button" title={tagLabel(t)} aria-pressed={tags.includes(t)} className={tags.includes(t) ? "on" : ""} onClick={() => setTags((cur) => (cur.includes(t) ? cur.filter((x) => x !== t) : [...cur, t]))}>{tagLabel(t)}</button>)}</div>
            </div>
          );
        })}
        {tags.length > 0 && <button type="button" className="ex-clear" onClick={() => setTags([])}><X size={12} /> clear filters</button>}
      </div>
      <div className="table-scroll">
        <table className="ex-table ex-why-table ex-cards">
          <thead><tr>{th("name", "Player")}{live && th("pos", "Pos", COLUMN_HELP.pos)}{hasModel && th("model", "Our chance", COLUMN_HELP.model)}{hasMarket && th("market", "Sportsbook", COLUMN_HELP.market)}{hasMarket && th("rel", "Gap (%)", COLUMN_HELP.rel)}{hasGap && hasMarket && th("edge_sg", "Gap (strokes)", COLUMN_HELP.edge_sg)}{th("mu", "Expected skill", COLUMN_HELP.mu)}{hasSd && th("sd", "Std dev", COLUMN_HELP.sd)}<th title="The biggest reasons our price differs from the sportsbook.">Why</th></tr></thead>
          <tbody>
            {rows.map((p) => {
              const e = p.probs[market];
              const b = edgeBadge(p, market);
              return (
                <tr key={p.id} className="clickable-row" onClick={() => openPlayer(p.id)}>
                  <th scope="row" className="ex-sticky"><button type="button" className="ex-linkbtn" onClick={(ev) => { ev.stopPropagation(); openPlayer(p.id); }}>{p.name}</button><Tags tags={p.tags} max={3} /></th>
                  {live && <td data-label="Position">{p.live?.tier === 0 ? p.live?.pos ?? "-" : p.live?.tier === 2 ? "MC" : "WD"}</td>}
                  {hasModel && <td data-label="Our chance">{pct(e.model)}</td>}{hasMarket && <td data-label="Sportsbook">{pct(e.market)}</td>}
                  {hasMarket && <td data-label="Gap (%)" className={b.tone === "positive" ? "pos" : b.tone === "negative" ? "neg" : ""}>{b.text}</td>}
                  {hasGap && hasMarket && <td data-label="Gap (strokes)" className={Math.abs(p.edge_sg ?? 0) < 0.005 ? "" : (p.edge_sg ?? 0) > 0 ? "pos" : "neg"}>{gapText(p.edge_sg)}</td>}
                  <td data-label="Expected skill">{signed(p.mu_rel)}</td>{hasSd && <td data-label="Std dev">{p.sd?.toFixed(2) ?? "-"}</td>}
                  <td data-label="Why" className="ex-why">{(hasMarket ? p.why_edge : null) ?? p.why}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {rows.length === 0 && <EmptyState title="No players match" detail="Clear a filter or the search." />}
      {!hasMarket && !live && rows.length > 0 && <p className="ex-muted">No sportsbook prices are posted for {MARKET_LABEL[market].toLowerCase()}, so only our own numbers are shown.</p>}
      <details className="ex-glossary-wrap"><summary>What the reasons behind a price mean</summary><LegendGroups legend={doc.components_legend} /></details>
      <details className="ex-glossary-wrap"><summary>Technical details</summary>
        <p className="ex-muted">{doc.model.n_sims ? `Prices come from ${doc.model.n_sims.toLocaleString()} simulated tournaments. ` : ""}Priced {etTime(doc.as_of)}.</p>
        <p className="ex-muted">{doc.kind === "live" ? `In-play pricing after round ${doc.after_round}` : "Pre-tournament pricing"}; finish chances {doc.finish.method === "exact_tapes" ? "come straight from the simulations" : doc.finish.method === "unavailable" ? "are not available" : "were rebuilt by replaying the hole-by-hole model"}.</p>
      </details>
    </Panel>
  );
}

/* ------------------------------------------------------------------ Player drawer */
function PlayerDrawer({ doc, player, competitors, market, setCompetitors, onSelect, onClose }: {
  doc: ExplainDoc; player: ExPlayer; competitors: number[]; market: Market; setCompetitors: (ids: number[]) => void; onSelect: (id: number) => void; onClose: () => void;
}) {
  const [mode, setMode] = useState<"probability" | "cumulative">("probability");
  const [q, setQ] = useState("");
  const byId = useMemo(() => new Map(doc.players.map((p) => [p.id, p] as const)), [doc.players]);
  const wf = useMemo(() => waterfall(player, doc.components_legend), [player, doc.components_legend]);
  const wfScale = useMemo(() => waterfallScale(doc.players), [doc.players]);
  const wfScales = useMemo(() => waterfallRowScales(doc.players, doc.components_legend), [doc.players, doc.components_legend]);
  const shape = useMemo(() => (doc.kind === "live" || player.withdrawn ? null : shapeOf(player, shapeMap(doc), availableMarkets(doc))), [doc, player]);
  const chosen = [player, ...competitors.map((id) => byId.get(id)).filter((p): p is ExPlayer => !!p)];
  const withShape = chosen.filter((p) => p.finish);
  const series: FinishSeries[] = withShape.map((p, i) => ({ id: p.id, name: p.name, pos: p.finish!.pos, missCut: doc.event.cut_round ? p.finish!.p_miss_cut : null, color: PLAYER_COLORS[i % PLAYER_COLORS.length] }));
  const hits = q.trim() ? doc.players.filter((p) => p.finish && !p.withdrawn && p.id !== player.id && !competitors.includes(p.id) && nameMatches(p.name, q)).slice(0, 6) : [];
  const f = player.finish;
  const t = f ? topK(f.pos, [1, 5, 10, 20]) : [];
  const rounds = doc.event.rounds ?? 4;
  const live = player.live;
  const closeRef = useRef<HTMLButtonElement>(null);
  const onCloseRef = useRef(onClose);
  useEffect(() => { onCloseRef.current = onClose; });
  useEffect(() => {
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onCloseRef.current(); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [player.id]);
  return (
    <div className="ex-drawer-layer" role="dialog" aria-modal="true" aria-label={`${player.name} pricing detail`}>
      <button className="ex-drawer-backdrop" aria-label="Close player detail" onClick={onClose} />
      <aside className="ex-drawer">
        <header>
          <div>
            <span className="eyebrow">{doc.kind === "live" ? `In play, after round ${doc.after_round}` : "Before the event"}</span>
            <h2>{player.name}</h2>
            <a className="ex-link" href={`/players?player=${player.id}`}>Profile and vs PGA avg <ChevronRight size={14} /></a>
            <Tags tags={player.tags} max={8} />
          </div>
          <button type="button" className="ex-close" onClick={onClose} aria-label="Close" ref={closeRef}><X size={20} /></button>
        </header>

        <div className="ex-drawer-kpis">
          <Kpi label={`${MARKET_LABEL[market]} chance`} value={pct(player.probs[market].model)} detail={doc.kind === "live" ? "our chance now" : `sportsbook ${pct(player.probs[market].market)}`} tone="model" />
          <Kpi label={`Expected skill ${LABELS.vsField.short}`} value={`${signed(player.mu_rel)} strokes a round`} detail={`Std dev ${player.sd?.toFixed(2) ?? "-"} strokes`} tone={(player.mu_rel ?? 0) >= 0 ? "positive" : "negative"} />
          {f && <div className="ex-kpi-wrap"><Kpi label="Median finish" value={String(f.median)} detail={`Likely range: ${f.p10} to ${f.p90}`} tone="accent" /><small className="ex-modal">{modalFinishText(modalFinish(f, !!doc.event.cut_round))}</small></div>}
          {f && <Kpi label="Top 10 chance" value={pct(t[2])} detail={`Win chance ${pct(t[0])}${doc.event.cut_round ? ` · misses the cut ${pct(f.p_miss_cut, 0)}` : ""}`} tone="neutral" />}
        </div>
        {shape && <p className="ex-callout"><strong>{shape.sentence}</strong> {shape.parts_sentence}{valueBets(shape).length > 0 ? ` Value against the sportsbook: ${valueBets(shape).join(", ")}.` : ""}</p>}

        <section>
          <h3 className="ex-h3">Why this price</h3>
          <p className="ex-muted">Strokes per round above (+) or below (−) the average player in this field ({LABELS.vsField.short}); the {LABELS.vsPgaAvg.short} view is on the profile. The skill rows add up to their expected skill{doc.kind === "live" ? "; the in-play rows are part of it" : ""}; weather is shown separately because it adjusts scores on top.</p>
          <Waterfall {...wf} scale={wfScale} scales={wfScales} />
          {shape && <p className="ex-muted">The sportsbook&rsquo;s prices imply skill {signed(shape.skill_market)} and spread {shape.spread_market.toFixed(2)}; ours are {signed(shape.skill_ours)} (including weather) and {shape.spread_ours.toFixed(2)}.</p>}
        </section>

        <section>
          <h3 className="ex-h3">Our chance vs the sportsbook</h3>
          {shape && <div className="ex-shape-wrap"><ShapeBars shape={shape} markets={availableMarkets(doc)} /><span className="ex-muted">Up: our chance is higher than the sportsbook&rsquo;s. Down: lower. Gray: within {Math.round(EDGE_MIN * 100)}%.</span></div>}
          <MarketRows player={player} hasCut={!!doc.event.cut_round} />
        </section>

        <section>
          <div className="ex-sec-head"><h3 className="ex-h3">Finish shape</h3>
            <SegmentedControl label="Chart type" value={mode} onChange={setMode} options={[{ value: "probability", label: "Position" }, { value: "cumulative", label: "Cumulative" }]} /></div>
          {doc.finish.method === "unavailable" || !f ? <p className="ex-muted">Finish odds are not available for this run.</p> : (
            <>
              <Lazy height={330}><FinishChart series={series} cutTopN={doc.event.cut_round ? doc.event.cut_top_n : null} mode={mode} /></Lazy>
              <div className="ex-compare">
                <h4>Compare with</h4>
                <div className="chip-row">
                  {competitors.map((id) => <span className="chip" key={id}>{byId.get(id)?.name ?? id}<button type="button" onClick={() => setCompetitors(competitors.filter((x) => x !== id))} aria-label={`Remove ${byId.get(id)?.name}`}><X size={13} /></button></span>)}
                  {competitors.length === 0 && <span className="ex-muted">nobody selected</span>}
                </div>
                <div className="ex-compare-actions">
                  <button type="button" onClick={() => setCompetitors(defaultCompetitors(doc, player.id, 3))}>Same-odds neighbors</button>
                  <button type="button" onClick={() => setCompetitors([])}>Clear</button>
                </div>
                <label className="search-box"><Search size={15} /><input value={q} onChange={(e) => setQ(e.target.value)} placeholder={competitors.length >= MAX_OVERLAY - 1 ? "Maximum reached" : "Add a competitor"} disabled={competitors.length >= MAX_OVERLAY - 1} aria-label="Add competitor" /></label>
                {hits.length > 0 && <ul className="ex-hits">{hits.map((p) => <li key={p.id}><button type="button" onClick={() => { setCompetitors(toggleCompetitor(competitors, p.id)); setQ(""); }}>{p.name}<small>{pct(p.probs.win.model)} win</small></button></li>)}</ul>}
              </div>
              <div className="table-scroll"><table className="ex-table ex-compare-table">
                <thead><tr><th>Player</th><th title="Expected skill: strokes per round better (+) or worse (-) than the field average.">Skill</th><th>Win</th><th>Top 5</th><th>Top 10</th><th>Top 20</th>{doc.event.cut_round ? <th title="Our chance to make the cut.">Make cut</th> : null}<th title="The finish position that half of the simulations beat and half did not. The smaller note is the single most likely finish.">Median finish</th></tr></thead>
                <tbody>{withShape.map((p, i) => (
                  <tr key={p.id}><th scope="row"><i className="ex-dot" style={{ background: PLAYER_COLORS[i % PLAYER_COLORS.length] }} aria-hidden="true" />
                    {p.id === player.id ? p.name : <button type="button" className="ex-linkbtn" onClick={() => onSelect(p.id)}>{p.name}</button>}</th>
                    <td data-label="Skill">{signed(p.mu_rel)}</td><td data-label="Win">{pct(p.probs.win.model)}</td><td data-label="Top 5">{pct(p.probs.top_5.model)}</td><td data-label="Top 10">{pct(p.probs.top_10.model)}</td><td data-label="Top 20">{pct(p.probs.top_20.model)}</td>
                    {doc.event.cut_round ? <td data-label="Make cut">{pct(p.probs.make_cut.model ?? (p.finish ? 1 - p.finish.p_miss_cut : null), 0)}</td> : null}
                    <td data-label="Median finish">{p.finish!.median}<small className="ex-modal" title="The single most likely finish position">{modalFinishText(modalFinish(p.finish, !!doc.event.cut_round)).replace("Most likely finish: ", "most likely ").replace("Most likely: ", "most likely: ")}</small></td></tr>
                ))}</tbody></table></div>
              <p className="ex-muted">Finish chances come from {doc.finish.method === "exact_tapes" ? "the simulations themselves" : `${doc.finish.n_draws?.toLocaleString() ?? "many"} replayed tournaments`}{doc.finish.calibration?.top_20 ? `; on average they sit within ${pct(doc.finish.calibration.top_20.mean_abs, 1)} of the posted top-20 chances` : ""}.</p>
            </>
          )}
        </section>

        <section>
          <h3 className="ex-h3">Score distribution</h3>
          <ScoreFan rounds={rounds} items={withShape.map((p, i) => ({ id: p.id, name: p.name, color: PLAYER_COLORS[i % PLAYER_COLORS.length], scores: p.scores }))} />
        </section>

        {live && (
          <section>
            <h3 className="ex-h3">In play</h3>
            <div className="ex-live-grid">
              <span>Position <b>{live.pos ?? "-"}</b></span><span>To par <b>{toParText(live.to_par)}</b></span>
              <span>Skill before the event → now <b>{signed(live.mu_pre)} → {signed(live.mu_live)}</b></span>
              <span>Change since last run <b>{signed(live.d_mu_since_last)}</b> (form this week {signed(live.d_b8_since_last)}{Math.abs(live.d_contention_since_last ?? 0) >= 0.005 ? `, title race ${signed(live.d_contention_since_last)}` : ""})</span>
            </div>
            {live.rounds.length > 0 && <div className="table-scroll"><table className="ex-table"><thead><tr><th>Round</th><th>Score</th><th title="Strokes gained in total versus the field.">Total</th><th title="Strokes gained off the tee.">Driving</th><th title="Strokes gained on approach shots.">Approach</th><th title="Strokes gained around the green.">Short game</th><th title="Strokes gained putting.">Putting</th></tr></thead>
              <tbody>{live.rounds.map((r) => <tr key={r.round}><th scope="row">R{r.round}</th><td>{r.score ?? "-"}</td><td>{signed(r.sg_total)}</td><td>{signed(r.sg_ott)}</td><td>{signed(r.sg_app)}</td><td>{signed(r.sg_arg)}</td><td>{signed(r.sg_putt)}</td></tr>)}</tbody></table></div>}
            <div className="table-scroll"><table className="ex-table"><thead><tr><th>Market</th><th>Before the event</th><th>Now</th><th title="Change since the previous pricing run, in percentage points.">Change since last run</th></tr></thead>
              <tbody>{availableMarkets(doc).map((m) => <tr key={m}><th scope="row">{MARKET_LABEL[m]}</th><td>{pct(live.pre_event[m])}</td><td>{pct(player.probs[m].model)}</td><td><Delta value={(live.since_last[m] ?? 0) * 100} digits={1} suffix="pp" /></td></tr>)}</tbody></table></div>
          </section>
        )}
        <p className="ex-muted ex-foot"><CloudSun size={13} /> Wave, time-of-day and wind effects are in the waterfall (weather rows) and on the <Link className="ex-link" href="/weather-effects">Weather page</Link>.</p>
      </aside>
    </div>
  );
}
