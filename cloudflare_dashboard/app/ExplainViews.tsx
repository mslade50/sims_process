"use client";

/**
 * "This week" event page and "Why priced" table + the player drawer (finish shape, score distribution, why-this-price waterfall, competitor overlay).
 * Data: one explain document per run, listed in the index (golfprice/explain_export.py; week runs before the event, live runs after each round). Why priced shows the last pre-tournament run unless the owner flips to live.
 * Design-system tokens/components only (tokens.css, ui.tsx); logic in explain-rules.ts.
 */
import Link from "next/link";
import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import { useEventParam } from "./event-param";
import { ChevronRight } from "lucide-react";
import { EmptyState, LoadingState, SegmentedControl } from "./components";
import { DeltaBar, RangeBar, ShapeBars, shapeCap, Term } from "./ExplainBits";
import { MARKETS, WINNING_SCORE_MISSING, DG_GAP_LABEL, DG_GAP_KINDS, DG_PTS_MIN, DG_FILTER_DEFAULT, DG_ROUNDS_OPTIONS, DG_SKILL_OPTIONS, biggestDgSkillGap, biggestGaps, dgGaps, hasDg, eventLede, heroFacts, lastSeenRunRef, plainName, previousRunRef, ptsText, relText, spanText, MARKET_LABEL, SHAPE_METHOD, american, availableMarkets, cutLineRows, differView, edgeBadge, edgeCellText, explainChoices, explainKeyFor, favourites, fieldStrengthText, toPar, groupSentence, leaderboard, orderEvents, parseExplain, pct, changesDataName, latestChangesDataName, plainCourseBullets, plainHoleSource, plainText, spreadGapOf, spreadLeanNote, signed, roundsWithoutTeeTimes, tagLabel, toParText, valueBets, venueClassLines, weatherEffectLines, weatherStatus, type DgGapKind, type DgFilter, type Differ, type ExPlayer, type ExplainDoc, type FieldScores, type IndexEvent, type Market, type Shape } from "./explain-rules";
import { parseChangesPayload, whatChanged, type BookMove, type Cause } from "./changes-rules";
import { useDashboardData } from "./data";

// Heavy parts load on demand so the summary (key facts, what changed) renders without Recharts, the weather views or the course-geometry viewer.
const HoleStrip = lazy(() => import("./ExplainCharts").then((m) => ({ default: m.HoleStrip })));
const WeatherThisRound = lazy(() => import("./ScoringView").then((m) => ({ default: m.WeatherThisRoundFor })));
const CourseGeometry = lazy(() => import("./CourseGeometry").then((m) => ({ default: m.CourseGeometry })));
const WeatherLinkCard = lazy(() => import("./WeatherBriefView").then((m) => ({ default: m.WeatherLinkCard })));
const ShapeSpotlightCard = lazy(() => import("./ShapeViews").then((m) => ({ default: m.ShapeSpotlightCard })));
const PlayerDrawer = lazy(() => import("./ExplainDrawer").then((m) => ({ default: m.PlayerDrawer })));
/** Suspense with a placeholder of the final height, so nothing below jumps when the chunk arrives. The chunk (and its data) is requested only when the block is within
 *  600px of the viewport, so below-the-fold charts and panels never compete with the content being read (Recharts alone is ~90 KB). */
export function Lazy({ height, children }: { height: number; children: React.ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const [near, setNear] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (typeof IntersectionObserver === "undefined") { queueMicrotask(() => setNear(true)); return; }
    const io = new IntersectionObserver((entries) => { if (entries.some((e) => e.isIntersecting)) { setNear(true); io.disconnect(); } }, { rootMargin: "600px 0px" });
    io.observe(el);
    return () => io.disconnect();
  }, []);
  const placeholder = <div className="ex-lazy" style={{ minHeight: height }} aria-hidden="true" />;
  if (!near) return <div ref={ref}>{placeholder}</div>;
  return <Suspense fallback={placeholder}>{children}</Suspense>;
}
import { etTime } from "./lib";
import { Chip, Delta, GapRail, OnThisPage, PageHeader, Section } from "./ui";
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
export type Shell = { doc: ExplainDoc; event: IndexEvent; latest: boolean; market: Market; setMarket: (m: Market) => void; openPlayer: (id: number) => void };

export function ExplainShell({ title, description, preEventFirst = false, defaultView = "pre", children }: { title: string; description: string; preEventFirst?: boolean; defaultView?: View; children: (s: Shell) => React.ReactNode }) {
  const index = useDashboardData<{ events?: IndexEvent[] }>("golfprice/index.json");
  const events = useMemo(() => orderEvents(index.data?.events ?? []), [index.data]);
  const [params, setParams] = useState<Params>(readParams);
  const barEvent = useEventParam();
  if (index.loading || (!params.e && !barEvent && events.length > 0)) return <LoadingState label="Loading golfprice events" />;   // the bar's event is read after mount; the default event must not load a document first
  if (index.error && index.errorKind === "outage") return <EmptyState title="Can’t reach the model data right now" detail={`${index.error} Nothing has been lost; this page retries on its own, or try again now.`} onRetry={index.refetch} />;
  if (index.error || !events.length) return <EmptyState title="No golfprice events published yet" detail={index.error ?? "Publish a run (python -m golfprice.publish_dashboard) and reload."} />;
  const current = events.find((e) => e.event_uid === (params.e || barEvent)) ?? events.find((e) => e.explain_key) ?? events[0];
  const update = (patch: Partial<Params>) => {
    const next = { ...params, ...patch };
    setParams(next);
    writeParams(next);
  };
  return (
    <>
      <PageHeader title={title} purpose={description} />
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
  // "What changed" needs the stored diff of the latest run: ask for it now, in parallel with the run's document, so the strip arrives with the page instead of pushing it down later.
  const early = useDashboardData<unknown>(defaultView === "live" && view === "live" ? latestChangesDataName(event) : null);
  const doc = useMemo(() => parseExplain(data), [data]);
  const toggle = offerToggle && (
    <div className="ex-viewbar" title={view === "pre" ? "Prices as set before the first tee shot." : "Prices updated with scores so far."}>
      <SegmentedControl label="Prices shown" value={view} onChange={(v) => update({ v, p: null, vs: [] })}
        options={[{ value: "pre", label: "Pre-tournament prices" }, { value: "live", label: choices.liveRound ? `Prices after R${choices.liveRound}` : "Latest prices" }]} />
    </div>
  );
  if (loading || early.loading) return <>{toggle}<LoadingState label="Loading the pricing explanation" /></>;   // the small diff is waited for so "What changed" arrives with the page instead of pushing it down
  const stale = !preEventFirst && event.explain_run && doc && `${doc.kind}_${doc.run}` !== event.explain_run;
  if (error || !doc || stale) return <>{toggle}<EmptyState title={`Nothing to show for ${event.name ?? event.event_uid} yet`} detail="Prices for this event have not been run since this page was updated. They will appear after the next pricing run." /></>;
  const market = params.m ?? "win";
  const player = params.p ? doc.players.find((p) => p.id === params.p) ?? null : null;
  const competitors = params.vs;
  const isLatest = !!event.explain_run && `${doc.kind}_${doc.run}` === event.explain_run;
  return (
    <>
      {toggle}
      {!isLatest && <RunNote doc={doc} switchToLatest={offerToggle && view !== "live" ? () => update({ v: "live", p: null, vs: [] }) : null} />}
      <RunNotes doc={doc} />
      {preEventFirst && choices.hasLive && !choices.preKey && <p className="ex-muted ex-notice">The pre-tournament explanation was not kept for this event, so this shows the live pricing.</p>}
      {children({ doc, event, latest: !!event.explain_run && `${doc.kind}_${doc.run}` === event.explain_run, market, setMarket: (m) => update({ m }), openPlayer: (id) => update({ p: id, vs: [] }) })}
      {player && <Suspense fallback={null}><PlayerDrawer doc={doc} player={player} competitors={competitors} market={market}
        setCompetitors={(vs) => update({ vs })} onSelect={(id) => update({ p: id, vs: [] })} onClose={() => update({ p: null, vs: [] })} /></Suspense>}
    </>
  );
}

/** Shown only when the page is not on the latest run, so the bar's "Prices after R1" is never silently contradicted. */
function RunNote({ doc, switchToLatest }: { doc: ExplainDoc; switchToLatest: (() => void) | null }) {
  const what = doc.kind === "live" ? `the prices after round ${doc.after_round ?? "?"}` : "the pre-tournament prices";
  return (
    <p className="ex-runnote" role="status">
      <span>Showing {what}, priced {etTime(doc.as_of)}. This is not the latest run, so its gaps may no longer match the bar above.</span>
      {switchToLatest && <button type="button" className="ex-linkbtn" onClick={switchToLatest}>Show the latest prices</button>}
    </p>
  );
}

/** Only what adds meaning beyond the shell's stage chip and freshness: manual adjustments and missing finish chances. */
function RunNotes({ doc }: { doc: ExplainDoc }) {
  const n = doc.model.overrides_applied.length;
  if (!n && doc.finish.method !== "unavailable") return null;
  return (
    <div className="ex-banner" aria-label="Run notes">
      {n > 0 && <Chip tone="warning" title="Prices include hand-set adjustments from the Inputs page">{n} manual adjustment{n > 1 ? "s" : ""} in these prices</Chip>}
      {doc.finish.method === "unavailable" && <Chip tone="warning">Finish chances not available in this run</Chip>}
    </div>
  );
}

/** True on phone widths. Every caller mounts after its data has loaded in the browser (never in server markup), so the first render already reads the real width: a late flip moved whole sections (CLS 0.41 on This week). */
export function usePhone(): boolean {
  const [phone, setPhone] = useState(() => typeof window !== "undefined" && typeof window.matchMedia === "function" && window.matchMedia("(max-width: 820px)").matches);
  useEffect(() => {
    const q = window.matchMedia("(max-width: 820px)");
    const sync = () => setPhone(q.matches);
    sync();
    q.addEventListener("change", sync);
    return () => q.removeEventListener("change", sync);
  }, []);
  return phone;
}

/** A section body that is collapsed on phones (everything stays one tap away) and open on desktop. */
export function Fold({ summary, children }: { summary: string; children: React.ReactNode }) {
  const phone = usePhone();
  const [touched, setTouched] = useState<boolean | null>(null);
  const open = touched ?? !phone;
  return (
    <details className="ex-fold" open={open} onToggle={(e) => { const o = (e.currentTarget as HTMLDetailsElement).open; if (o !== open) setTouched(o); }}>
      <summary>{summary}</summary>
      {open && children}
    </details>
  );
}

/** "Show all N" for a long list: the first `limit` items, the rest on demand. */
export function useLimit<T>(items: T[], limit: number, reset: unknown = null): { shown: T[]; more: React.ReactNode } {
  const [all, setAll] = useState<{ on: boolean; key: unknown }>({ on: false, key: reset });
  const on = all.on && all.key === reset;
  const shown = on ? items : items.slice(0, limit);
  const more = items.length > limit ? (
    <button type="button" className="ex-showall" aria-expanded={on} onClick={() => setAll({ on: !on, key: reset })}>{on ? `Show the first ${limit}` : `Show all ${items.length}`}</button>
  ) : null;
  return { shown, more };
}

/** One line under a bar list that gives the zero line, the unit and what a full bar means. */
export function ScaleNote({ scale, unit, what }: { scale: number; unit: string; what: string }) {
  const end = scale.toFixed(scale < 10 ? 1 : 0);
  return <p className="ex-scale">Bars show {what} in {unit}. Centre line is 0 (no difference); each end is ±{end} {unit}.</p>;
}

/** A flat titled block inside a Section (the design system flattens cards). */
export function Panel({ title, eyebrow, actions, children }: { title: string; eyebrow?: string; actions?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="ex-sub">
      <header className="ex-sub-head">
        <div><h3>{title}</h3>{eyebrow && <span className="ex-sub-note">{eyebrow}</span>}</div>
        {actions && <div className="ex-sub-actions">{actions}</div>}
      </header>
      {children}
    </div>
  );
}

export type PriceRowData = { id: number; name: string; tags?: string[]; lead?: string; nums: Array<{ k: string; v: React.ReactNode; cls?: string; help?: string; minor?: boolean }>; rail?: { model: number; book: number } | null; relNote?: string; why?: string | null };
/** Shared price list: player, numbers, model-vs-book gap on a zero-centred rail, one-line reason. Whole row opens the player. Compact two-line rows on phone. */
export function PriceList({ rows: allRows, heads, range, label, onOpen, limit = 12 }: { rows: PriceRowData[]; heads: [string, string, string, string]; range: number; label: string; onOpen: (id: number) => void; limit?: number }) {
  const { shown: rows, more } = useLimit(allRows, limit);
  const hasRail = allRows.some((r) => r.rail);
  const hasWhy = allRows.some((r) => r.why);
  return (
    <div className={`pl${hasRail ? "" : " pl-norail"}${hasWhy ? "" : " pl-nowhy"}`}>
      <div className="pl-head" aria-hidden="true"><span>{heads[0]}</span><span>{heads[1]}</span>{hasRail && <span>{heads[2]}</span>}{hasWhy && <span>{heads[3]}</span>}</div>
      <ul aria-label={label}>{rows.map((r) => (
        // The name button is the row's one tab stop; its ::after covers the whole row (stretched target), so a click anywhere opens the player without a nested handler.
        <li key={r.id} className="pl-row">
          <span className="pl-name">
            {r.lead && <span className="pl-pos num">{r.lead}</span>}
            <span className="pl-who"><button type="button" className="ex-linkbtn pl-open" onClick={() => onOpen(r.id)}>{r.name}</button>{r.tags && r.tags.length > 0 && <Tags tags={r.tags} max={2} />}</span>
          </span>
          <span className="pl-nums">{r.nums.map((n) => <span key={n.k} className={`pl-n${n.minor ? " pl-minor" : ""}`} title={n.help}><small>{n.k}</small><b className={`num ${n.cls ?? ""}`}>{n.v}</b></span>)}</span>
          {hasRail && <span className="pl-rail">{r.rail ? <><GapRail model={r.rail.model} book={r.rail.book} range={range} />{r.relNote && <small className="pl-rel num">{r.relNote}</small>}</> : <small className="ex-muted">no book price</small>}</span>}
          {hasWhy && <span className="pl-why">{r.why ?? ""}</span>}
        </li>
      ))}</ul>
      {more}
    </div>
  );
}
export const RAIL_RANGE: Record<Market, number> = { win: 6, top_5: 10, top_10: 12, top_20: 15, make_cut: 15 };

export const Tags = ({ tags, max = 4 }: { tags: string[]; max?: number }) => <span className="ex-tags">{tags.slice(0, max).map((t) => <em key={t}>{tagLabel(t)}</em>)}{tags.length > max && <em>+{tags.length - max}</em>}</span>;

/* ------------------------------------------------------------------ This week */
export function ThisWeekView() {
  return (
    <ExplainShell title="This week" preEventFirst defaultView="live" description="Favorite, biggest moves, gaps to the book.">
      {(s) => <ThisWeek {...s} />}
    </ExplainShell>
  );
}

function ThisWeek({ doc, event, latest, market, setMarket, openPlayer }: Shell) {
  const live = doc.kind === "live";
  const weatherKey = event.weather_key || `golfprice/weather/${event.event_uid.replace(/:/g, "_")}/latest.json`;
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
  const dgSkill = useMemo(() => (live ? null : biggestDgSkillGap(doc, contenderIds)), [doc, contenderIds, live]);
  const chanceRows: PriceRowData[] = fav.map((p) => {
    const e = p.probs[shownMarket];
    const b = edgeBadge(p, shownMarket);
    const showBoard = !live && fav.some((x) => x.probs[shownMarket]?.fair != null);
    return {
      id: p.id, name: p.name, why: p.why,
      nums: [
        { k: "Our chance", v: pct(e.model), help: "Our chance for this bet type, from the simulations." },
        ...(!live ? [{ k: "Book", v: pct(e.market), help: "Chance implied by sportsbook odds with the margin removed." }] : []),
        { k: "Our odds", v: american(e.model), help: "American odds that match our chance." },
        ...(showBoard ? [{ k: "Board price", v: e.fair == null ? "—" : american(e.fair), help: "The price the odds board publishes: our chance blended with the sportsbook consensus." }] : []),
      ],
      rail: !live && e.model != null && e.market != null ? { model: e.model * 100, book: e.market * 100 } : null,
      relNote: b.tone === "neutral" ? "in line" : `${b.text} vs book`,
    };
  });
  const phone = usePhone();
  const [hasChanged, setHasChanged] = useState(false);
  const boardSection = live ? (
    <Section id="board" title="Leaderboard" purpose="Top 12 by win chance now, with the move since the last run (arrow and points).">
      <LiveSection doc={doc} openPlayer={openPlayer} />
    </Section>
  ) : (
    <Section id="board" title="Model vs sportsbook" purpose="Win chance, ours against the book with its margin removed. Plus (green) means we rate the player higher.">
      <GapGlance doc={doc} ids={contenderIds} openPlayer={openPlayer} />
    </Section>
  );
  const nav = [...(hasChanged ? [{ id: "changed", label: "What changed" }] : []), { id: "board", label: live ? "Leaderboard" : "Model vs book" }, { id: "chances", label: "Chances" }, { id: "course", label: "Course" }, { id: "weather", label: "Weather" }, { id: "shapes", label: "Shapes" }];
  return (
    <>
      <KeyFacts doc={doc} ids={contenderIds} openPlayer={openPlayer} />
      <OnThisPage sections={nav} />
      {phone && boardSection}
      <WhatChangedStrip doc={doc} event={event} latest={latest} openPlayer={openPlayer} onPresent={setHasChanged} />

      <div className="ex-lede-block">
        <p className="ex-lede">{eventLede(doc)}</p>
        <div className="ex-chips-row" aria-label="Field summary">
          <span className="ex-chip"><Term k="ex_field_strength">Field strength</Term> <b>{fs == null ? "n/a" : `${signed(fs)} strokes`}</b></span>
          {!live && ee && <span className="ex-chip"><Term k="ex_skill_gap">Biggest skill gap vs sportsbooks</Term> <b className={ee.shape.skill_gap >= 0 ? "pos" : "neg"}>{plainName(ee.player.name)} {signed(ee.shape.skill_gap)} a round</b></span>}
          {!live && dgSkill && <span className="ex-chip" title={dgSkill.stated !== undefined ? "Measured against the skill Data Golf's own odds imply. Players whose stated skill disagrees with their odds are left out." : undefined}><Term k="ex_dg_skill_gap">Biggest skill gap vs Data Golf</Term> <b className={dgSkill.gap >= 0 ? "pos" : "neg"}>{dgSkill.name} {signed(dgSkill.gap)} a round</b></span>}
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

      {!phone && boardSection}

      <Section id="chances" title={live ? "Best chances" : "Shortest prices"} purpose={live ? "Ten best chances for the selected bet type." : "Ten shortest-priced players; the rail is zero-centred on the sportsbook (right of the line = we are higher)."}
        actions={<SegmentedControl label="Bet type" value={shownMarket} onChange={setMarket} options={markets.map((m) => ({ value: m, label: MARKET_LABEL[m] }))} />}>
        <PriceList rows={chanceRows} limit={phone ? 5 : 10} range={RAIL_RANGE[shownMarket]} label={`Ten ${live ? "best" : "shortest"} ${MARKET_LABEL[shownMarket]} chances`} onOpen={openPlayer}
          heads={["Player", "Chances", "Model vs book (points)", "Main reasons"]} />
      </Section>

      <Section id="course" title="Course" purpose={`${cc.course ?? "Course card"}: strokes versus par for the average player in this field, by hole.`}>
        <ul className="ex-bullets">{bullets.slice(0, 2).map((b) => <li key={b}>{b}</li>)}</ul>
        <Fold summary="Hole-by-hole scoring and course geometry">
        <Lazy height={230}><HoleStrip holes={cc.holes} /></Lazy>
        {bullets.length > 2 && <details className="ex-details"><summary>{bullets.length - 2} more notes on what drives scoring</summary><ul className="ex-bullets">{bullets.slice(2).map((b) => <li key={b}>{b}</li>)}</ul></details>}
        {cc.hole_source && <p className="ex-muted">{plainHoleSource(cc.hole_source)} Values are for the average player in this field.</p>}
        <Lazy height={120}><CourseGeometry eventUid={event.event_uid} eventName={doc.event.name ?? event.name} /></Lazy>
        </Fold>
      </Section>

      <Section id="weather" title="Weather" purpose="Forecast effect on scoring, in strokes a round: minus is easier, plus is harder."
        actions={<Link prefetch={false} className="ex-link" href={`/weather-effects?e=${encodeURIComponent(event.event_uid)}`}>Weather page <ChevronRight size={14} /></Link>}>
        <Lazy height={64}><WeatherLinkCard event={event} /></Lazy>
        <Fold summary="Forecast detail and effect by round">
        <WeatherSnapshot doc={doc} />
        <WeatherEffects doc={doc} />
        <Lazy height={160}><WeatherThisRound weatherKey={weatherKey} round={live ? (doc.after_round ?? 0) + 1 : 1} latest={latest} /></Lazy>
        </Fold>
      </Section>

      <Section id="shapes" title="Shapes and detailed gaps" purpose={live ? "How players with the same win chance finish differently." : "Same win chance, different finishes, and bet-by-bet gaps against the book and Data Golf."}>
        <Suspense fallback={null}><ShapeSpotlightCard doc={doc} openPlayer={openPlayer} onSeeMore={() => window.location.assign("/distributions")} /></Suspense>
        <Fold summary={live ? "More on shapes" : "Gaps to Data Golf and every contender, bet by bet"}>
        {!live && hasDg(doc) && <DgGlance doc={doc} ids={contenderIds} openPlayer={openPlayer} />}
        {live ? (
          <p className="ex-muted ex-notice">While play is under way we do not compare our prices with the sportsbook: its prices are not live, so a gap would not mean anything. See Why priced for the comparison made before the event.</p>
        ) : (
          <DifferPanel doc={doc} rows={differ.rows} differ={differ.differ} openPlayer={openPlayer} />
        )}
        </Fold>
      </Section>
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
          <span className="ex-fact-sub"><b className="num">{pct(f.model)}</b> ours{f.market !== null ? <> vs <b className="num">{pct(f.market)}</b> book</> : <> · <span title="Sportsbooks have not posted a win price for this player in this run. It appears here after the next run that finds one.">no book price posted yet</span></>}</span>
        </div>
      )}
      {s?.kind === "gap" && (
        <div className="ex-fact">
          <span className="ex-fact-label">Biggest <Term k="ex_gap">gap to the sportsbook</Term></span>
          <strong className="ex-fact-value"><button type="button" className="ex-linkbtn" onClick={() => openPlayer(s.id)}>{s.name}</button> <em className={s.rel > 0 ? "pos" : "neg"}>{s.rel > 0 ? "▲" : "▼"} {relText(s.rel)}</em></strong>
          <span className="ex-fact-sub">Win chance <b className="num">{pct(s.model)}</b> ours vs <b className="num">{pct(s.market)}</b> book</span>
        </div>
      )}
      {s?.kind === "move" && (
        <div className="ex-fact">
          <span className="ex-fact-label">Biggest mover since the last run</span>
          <strong className="ex-fact-value"><button type="button" className="ex-linkbtn" onClick={() => openPlayer(s.id)}>{s.name}</button> <em data-flash className={s.delta > 0 ? "pos" : "neg"}>{s.delta > 0 ? "▲" : "▼"} {ptsText(s.delta)}</em></strong>
          <span className="ex-fact-sub">Win chance <b className="num">{pct(s.before)}</b> → <b className="num">{pct(s.after)}</b></span>
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
          <span className="ex-fact-sub">Most likely {toPar(sc.mode)} ({Math.round(sc.modeShare * 100)}% of sims) · 8 in 10: {toPar(sc.p10)} to {toPar(sc.p90)}</span>
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

/** One cause of a price move: a short chip, with the strokes and probability share in its tooltip. */
function CauseChip({ cause }: { cause: Cause }) {
  const parts = [cause.strokes !== null ? `${signed(cause.strokes)} strokes a round` : null, cause.pts !== null ? `${ptsText(cause.pts)} of the move` : null, cause.detail].filter(Boolean);
  return <span className={`ex-cause is-${cause.key}`} title={`${cause.label}${parts.length ? `: ${parts.join(", ")}` : ""}`}>{cause.chip}</span>;
}

/** Where the sportsbook moved relative to our price, and what it does to the edge. */
function BookMoves({ moves, openPlayer }: { moves: BookMove[]; openPlayer: (id: number) => void }) {
  const row = (b: BookMove) => (
    <li key={`${b.id}${b.market}`}>
      <button type="button" onClick={() => openPlayer(b.id)}>
        <span>{b.text}</span>
        {b.actionable && <b className={`ex-book-flag is-${b.actionable}`}>{b.actionable === "growing" ? "Edge growing" : "Edge closing"}</b>}
      </button>
    </li>
  );
  const first = moves.slice(0, 5);
  return (
    <div className="ex-book-moves">
      <h3 className="ex-h3">Sportsbook moves</h3>
      <ul aria-label="Sportsbook moves">{first.map(row)}</ul>
      {moves.length > first.length && <details className="ex-details"><summary>{moves.length - first.length} more</summary><ul>{moves.slice(5).map(row)}</ul></details>}
    </div>
  );
}

function WhatChangedStrip({ doc, event, latest, openPlayer, onPresent }: { doc: ExplainDoc; event: IndexEvent; latest: boolean; openPlayer: (id: number) => void; onPresent?: (present: boolean) => void }) {
  const seen = useLastSeen(event.event_uid, `${doc.kind}_${doc.run}`, latest);
  const prevRef = useMemo(() => previousRunRef(event, doc), [event, doc]);
  const visitRef = useMemo(() => lastSeenRunRef(event, doc, seen), [event, doc, seen]);
  const [against, setAgainst] = useState<"prev" | "visit">("prev");
  const ref = against === "visit" && visitRef ? visitRef : prevRef;
  // One small request for the stored diff of the two runs; the earlier run's whole document is fetched only when that request fails.
  const changesName = useMemo(() => changesDataName(event, doc, ref), [event, doc, ref]);
  const served = useDashboardData<unknown>(changesName);
  const fromServer = useMemo(() => parseChangesPayload(served.data), [served.data]);
  const needFallback = !changesName || (!served.loading && !fromServer);
  const source = useDashboardData<unknown>(needFallback ? ref?.key ?? null : null);
  const prevDoc = useMemo(() => parseExplain(source.data), [source.data]);
  const loading = changesName ? served.loading || (needFallback && !!ref && source.data === null && !source.error) : source.loading;
  const change = useMemo(() => (loading ? null : fromServer ?? whatChanged(doc, prevDoc)), [doc, prevDoc, loading, fromServer]);
  const [mk, setMk] = useState<Market>("win");
  const phone = usePhone();
  const [allMovers, setAllMovers] = useState(false);
  const present = (!!ref || !!doc.since_last) && (loading || !!change);
  useEffect(() => { onPresent?.(present); }, [present, onPresent]);
  if (!ref && !doc.since_last) return null;
  if (loading) return <section id="changed" className="ex-changed is-loading" aria-label="What changed" aria-busy="true"><h2>What changed</h2><span className="ex-muted">Comparing with the earlier run…</span></section>;
  if (!change) return null;
  const mkts = (Object.keys(change.movers) as Market[]).filter((m) => change.movers[m]?.length);
  const shownMk: Market = mkts.includes(mk) ? mk : mkts[0] ?? "win";
  const moverAll = (change.movers[shownMk] ?? []).slice(0, 5);
  const moverLimit = phone && !allMovers ? 3 : 5;
  const rows = moverAll.slice(0, moverLimit);
  const scale = Math.max(0.5, ...rows.map((r) => Math.abs(r.delta)));
  const baseKind = change.baseline.after_round ? `the run after round ${change.baseline.after_round}` : "the previous run";
  const fromWhat = against === "visit" && visitRef ? "your last visit" : baseKind;
  const span = spanText(change.gapMs);
  const w = change.weather;
  const sc = change.scoring;
  return (
    <section id="changed" className={`ex-changed${change.quiet ? " is-quiet" : ""}`} aria-label="What changed">
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
          <ScaleNote scale={scale} unit="points" what="the change in win chance since then" />
          <ul className="ex-movers-list" aria-label={`Biggest ${MARKET_LABEL[shownMk]} price moves`}>{rows.map((m) => (
            <li key={m.id}>
              <button type="button" onClick={() => openPlayer(m.id)}>
                <span className="ex-mv-name">{m.name}</span>
                <DeltaBar value={m.delta} scale={scale} label={`${m.name} ${ptsText(m.delta)}`} />
                <span className="ex-mv-nums">{pct(m.before)} → {pct(m.after)}</span>
                <b data-flash className={m.delta > 0 ? "pos" : "neg"}>{m.delta > 0 ? "▲" : "▼"} {ptsText(m.delta)}</b>
                {m.marketDelta !== null && Math.abs(m.marketDelta) >= 0.05 && <small className="ex-mv-market">sportsbook {ptsText(m.marketDelta)}</small>}
                {m.causes.length > 0 && <span className="ex-mv-causes" aria-label="Main causes">{m.causes.map((c) => <CauseChip key={c.key} cause={c} />)}</span>}
              </button>
            </li>
          ))}</ul>
          {phone && moverAll.length > 3 && <button type="button" className="ex-showall" aria-expanded={allMovers} onClick={() => setAllMovers((v) => !v)}>{allMovers ? "Show the top 3" : `Show all ${moverAll.length} moves`}</button>}
        </>
      )}
      {change.bookMoves.length > 0 && <BookMoves moves={change.bookMoves} openPlayer={openPlayer} />}
      {rows.length > 0 && (
        <details className="ex-details ex-cause-note">
          <summary>How the causes are worked out</summary>
          <p>Each chip is the share of that player&rsquo;s move we can trace to a change in the model&rsquo;s inputs, in strokes a round (plus means better for the player). Weather, course, travel, skill and manual-adjustment changes come straight from the two runs&rsquo; published inputs. &ldquo;Field and other players&rdquo; is the part of the move that is not the player&rsquo;s own rating: other players withdrawing, joining or changing, and simulation variation. &ldquo;Not separable&rdquo; means no single cause carries a clear share.</p>
        </details>
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
      <ScaleNote scale={g.scale * 100} unit="percent" what="how far our win chance is above or below the book's, as a share of the book's chance," />
      <div className="ex-gap-grid">
        {list(g.higher, "We are higher")}
        {list(g.lower, "We are lower")}
      </div>
      <p className="ex-muted">Percent is how far our chance is from the sportsbook&rsquo;s <Term k="ex_gap">(what is this?)</Term>. Contenders only; the full bet-by-bet table is further down.</p>
    </Panel>
  );
}

/** Where we differ most from Data Golf: skill estimates (strokes a round) and every bet type's chance (raw percentage points), with the owner's
 * minimum-rounds / minimum-skill filter (remembered in this browser). Reference only. */
function DgGlance({ doc, ids, openPlayer }: { doc: ExplainDoc; ids: number[]; openPlayer: (id: number) => void }) {
  const [kind, setKind] = useState<DgGapKind>("skill");
  const [filter, setFilterState] = useState<DgFilter>(DG_FILTER_DEFAULT);
  useEffect(() => {
    try {
      const raw = window.localStorage.getItem("gp.dgFilter");
      const f = raw ? JSON.parse(raw) : null;
      // eslint-disable-next-line react-hooks/set-state-in-effect -- one-time restore from localStorage after mount
      if (f && DG_ROUNDS_OPTIONS.includes(f.minRounds) && DG_SKILL_OPTIONS.includes(f.minSkill)) setFilterState({ minRounds: f.minRounds, minSkill: f.minSkill });
    } catch { /* storage unavailable: defaults */ }
  }, []);
  const setFilter = (f: DgFilter) => { setFilterState(f); try { window.localStorage.setItem("gp.dgFilter", JSON.stringify(f)); } catch { /* ignore */ } };
  const markets = useMemo(() => DG_GAP_KINDS.filter((k) => k === "skill" || (availableMarkets(doc) as string[]).includes(k)), [doc]);
  const g = useMemo(() => dgGaps(doc, kind, 5, ids, filter), [doc, kind, ids, filter]);
  const skill = kind === "skill";
  const what = skill ? "skill" : `${DG_GAP_LABEL[kind].toLowerCase()} chance`;
  const nums = (r: { ours: number; dg: number }) => (skill ? `${signed(r.ours)} vs ${signed(r.dg)}` : `${pct(r.ours)} vs ${pct(r.dg)}`);
  const checked = skill && [...g.higher, ...g.lower].some((r) => r.stated !== undefined);
  const anyFlag = skill && [...g.higher, ...g.lower].some((r) => r.flag);
  const gapTxt = (v: number) => (skill ? `${signed(v)}` : ptsText(v));
  const filtered = filter.minRounds > 0 || filter.minSkill !== null;
  const list = (rows: typeof g.higher, title: string) => (
    <div className="ex-gap-col">
      <h3 className="ex-h3">{title}</h3>
      {rows.length ? <ul>{rows.map((r) => (
        <li key={r.id}>
          <button type="button" onClick={() => openPlayer(r.id)}>
            <span className="ex-mv-name">{r.name}{r.flag && <span className="ex-dg-warn" role="img" aria-label="Data Golf's stated skill disagrees with its own odds" title={`Data Golf's odds imply ${signed(r.dg)}; its stated skill is ${r.stated != null ? signed(r.stated) : "n/a"}.`}> ⚠</span>}
              {r.rounds !== null && <small className="ex-dg-rounds" title="Rounds on record before this event">{Math.round(r.rounds)} rds</small>}</span>
            <DeltaBar value={r.gap} scale={g.scale} label={`${r.name}: our ${what} ${gapTxt(r.gap)} against Data Golf${r.flag ? " (its odds imply a different skill than it states)" : ""}`} />
            <span className="ex-mv-nums">{skill && r.flag && r.numberGap != null ? `skill numbers ${signed(r.numberGap)}, on chances ${signed(r.gap)}` : nums(r)}</span>
            <b className={r.gap > 0 ? "pos" : "neg"}>{r.gap > 0 ? "▲" : "▼"} {gapTxt(r.gap)}</b>
          </button>
        </li>
      ))}</ul> : <p className="ex-muted">Nobody{filtered ? " who passes the filter" : ""}: every gap is under {skill ? "0.1 strokes a round" : ptsText(DG_PTS_MIN[kind as Exclude<DgGapKind, "skill">]).replace("+", "")}.</p>}
    </div>
  );
  const sel = (label: string, value: string, options: Array<[string, string]>, onChange: (v: string) => void) => (
    <label className="ex-dg-filter">{label}
      <select value={value} onChange={(e) => onChange(e.target.value)}>{options.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>
    </label>
  );
  return (
    <Panel title="Where we differ from Data Golf" eyebrow={skill ? "Skill estimate, ours vs Data Golf's" : `${DG_GAP_LABEL[kind]} chance, ours vs Data Golf's, in percentage points`}
      actions={<SegmentedControl label="Compare" value={kind} onChange={setKind} options={markets.map((k) => ({ value: k, label: DG_GAP_LABEL[k] }))} />}>
      <div className="ex-dg-filters" role="group" aria-label="Filter players">
        {sel("Min rounds", String(filter.minRounds), DG_ROUNDS_OPTIONS.map((n) => [String(n), n === 0 ? "Any" : `${n}+`]), (v) => setFilter({ ...filter, minRounds: Number(v) }))}
        {sel("Min skill", filter.minSkill === null ? "any" : String(filter.minSkill), DG_SKILL_OPTIONS.map((x) => [x === null ? "any" : String(x), x === null ? "Any" : `${signed(x, 1)} vs field`]),
          (v) => setFilter({ ...filter, minSkill: v === "any" ? null : Number(v) }))}
        <span className="ex-muted">{g.pool} players compared</span>
        {filtered && <button type="button" className="ex-linkbtn" onClick={() => setFilter(DG_FILTER_DEFAULT)}>Clear</button>}
      </div>
      <ScaleNote scale={g.scale} unit={skill ? "strokes a round" : "points"} what={`our ${what} minus Data Golf's`} />
      <div className="ex-gap-grid">
        {list(g.higher, "We are higher")}
        {list(g.lower, "We are lower")}
      </div>
      {anyFlag && <p className="ex-muted ex-dg-warn-note"><span aria-hidden="true">⚠ </span><Term k="ex_dg_check">Data Golf&rsquo;s odds check</Term>: for the marked players its stated skill is out of line with its own odds, so for those players we show both gaps: the skill numbers (ours minus Data Golf&rsquo;s stated skill) and the gap on chances (what its win and top-20 odds imply), which is the one used for ranking.</p>}
      <p className="ex-muted">{skill
        ? <>Skill is strokes a round against the average of this field (before weather), ours vs {checked ? <>the skill Data Golf&rsquo;s own odds imply</> : <>Data Golf&rsquo;s final pre-tournament estimate</>}; the gap is ours minus theirs. <Term k="ex_dg_skill_gap">(what is this?)</Term> This week&rsquo;s contenders and anyone Data Golf gives at least a 5% top-20 chance.</>
        : <>The gap is our chance minus Data Golf&rsquo;s pre-tournament chance, in percentage points (pts): 5.0% vs 3.0% is +2.0 pts. Gaps under {ptsText(DG_PTS_MIN[kind as Exclude<DgGapKind, "skill">]).replace("+", "")} are not listed.</>}
        {" "}Min skill uses the higher of our skill and Data Golf&rsquo;s, in strokes a round against this field. Data Golf is a reference only; it never enters our price.</p>
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
      <div className="ex-dfl-head" aria-hidden="true">
        <span>Player</span>
        <span title={`One bar per bet type: up means our chance is higher than the sportsbook's, down means lower, gray means within 5% (in line). Bar height grows with the square root of the gap; a full bar is a ${Math.round(cap * 100)}% gap (bigger gaps are cut off with a notch).`}>Shape</span>
        <span>Ours vs book, by bet</span>
        <span>What we see</span>
      </div>
      <ul className="ex-dfl" aria-label="Contenders, bet by bet">{(showAll ? shown : shown.slice(0, DIFFER_PREVIEW)).map(({ player: p, shape: s }) => {
        const value = valueBets(s);
        return (
          <li key={p.id} className="ex-dfl-row">
            <button type="button" className="ex-linkbtn ex-dfl-name pl-open" onClick={() => openPlayer(p.id)}>{p.name}</button>
            <span className="ex-dfl-shape"><ShapeBars shape={s} markets={markets} cap={cap} /></span>
            <span className="ex-dfl-cells">{markets.map((m) => {
              const e = s.markets[m];
              if (!e) return null;
              return <span key={m} className="ex-dfl-cell" title={`${MARKET_HELP[m]} Ours ${pct(e.model)}, sportsbook ${pct(e.market)} (margin removed)`}>
                <small>{MARKET_LABEL[m]}</small>
                <b className={e.tone === "pos" || (e.tone === "neg" && e.side === "miss_cut") ? "pos" : e.tone === "neg" ? "neg" : "ex-muted"}>{edgeCellText(e)}</b>
              </span>;
            })}</span>
            <span className="ex-dfl-why">
              <strong>{s.sentence}</strong>
              {value.length > 0 && <span className="ex-differ-value">Value against the sportsbook: {value.join(", ")}</span>}
            </span>
          </li>
        );
      })}</ul>
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
  const phone = usePhone();
  const lb = leaderboard(doc, 12);
  const withB8 = doc.players.filter((p) => Math.abs(p.live?.d_b8_since_last ?? 0) >= 0.02).length;
  const withCon = doc.players.filter((p) => Math.abs(p.live?.d_contention_since_last ?? 0) >= 0.02).length;
  const hasCon = lb.some((p) => p.live?.contention !== null && p.live?.contention !== undefined && Math.abs(p.live.contention) > 0.0005);
  const hasB8 = lb.some((p) => p.live?.b8_delta !== null && p.live?.b8_delta !== undefined);
  const prev = doc.since_last ? (doc.since_last.prev_after_round !== null && doc.since_last.prev_after_round !== undefined ? `the run after round ${doc.since_last.prev_after_round}` : "the pre-tournament prices") : null;
  const rows: PriceRowData[] = lb.map((p) => {
    const d = (p.live?.since_last.win ?? 0) * 100;
    return {
      id: p.id, name: p.name, lead: String(p.live?.tier === 0 ? p.live?.pos ?? "-" : p.live?.tier === 2 ? "MC" : "WD"),
      nums: [
        { k: "To par", v: toParText(p.live?.to_par), help: "Score to par so far." },
        { k: "Win", v: pct(p.probs.win.model), help: "Our chance to win now." },
        { k: "Win change", v: Math.abs(d) < 0.05 ? <span className="ex-muted">none</span> : <span data-flash className="ex-moved"><Delta value={d} digits={1} suffix=" pts" /></span>, help: "Change in win chance since the previous run, in percentage points." },
        { k: "Top 10", v: pct(p.probs.top_10.model) },
        { k: "Skill now", v: signed(p.live?.mu_live), help: "Strokes per round better (+) or worse (-) than the field average." },
        ...(hasB8 ? [{ k: "Form", minor: true, v: signed(p.live?.b8_delta), help: "How much play this week has moved our rating, strokes per round." }] : []),
        ...(hasCon ? [{ k: "Title race", minor: true, v: signed(p.live?.contention), help: "How much being near the lead changes their chances." }] : []),
      ],
    };
  });
  return (
    <div className="ex-sub">
      <p className="ex-muted">After round {doc.after_round}{prev ? `; changes compared with ${prev}` : ""}. {withB8 > 0 ? `Form this week changed the prices of ${withB8} players` : "No player price moved much on form this week"}{withCon > 0 ? `, and the title race changed ${withCon}` : ""}.</p>
      <PriceList rows={rows} limit={phone ? 8 : 12} range={6} label="Leaderboard" onOpen={openPlayer} heads={["Pos / player", "Score and chances", "", ""]} />
    </div>
  );
}
