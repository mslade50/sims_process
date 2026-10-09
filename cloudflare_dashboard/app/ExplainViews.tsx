"use client";

/**
 * "This week" event page and "Why priced" table + the player drawer (finish shape, score distribution, why-this-price waterfall, competitor overlay).
 * Data: one explain document per run, listed in the index (golfprice/explain_export.py; week runs before the event, live runs after each round). Why priced shows the last pre-tournament run unless the owner flips to live.
 * Design-system tokens/components only (tokens.css, ui.tsx); logic in explain-rules.ts.
 */
import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowDown, ArrowUp, ChevronRight, CloudSun, Search, X } from "lucide-react";
import { EmptyState, Kpi, LoadingState, PageIntro, Panel, SegmentedControl } from "./components";
import { BiasChart, FinishChart, HoleStrip, LegendGroups, MarketRows, PLAYER_COLORS, ScoreFan, Waterfall, type FinishSeries } from "./ExplainCharts";
import {
  DIMENSION_HELP, MARKETS, MARKET_LABEL, MAX_OVERLAY, TAG_GROUPS, american, availableMarkets, availableTags, biasBaseline, biasDimensions, biasSentence, biasView, cutLineRows, defaultCompetitors, edgeBadge,
  explainChoices, explainKeyFor, favourites, filterWhy, gapText, leaderboard, nameMatches, orderEvents, parseExplain, pct, pctPair, plainCourseBullets, plainHoleSource, plainText, signed, sortWhy,
  tagLabel, toParText, toggleCompetitor, topEdges, topK, tourName, waterfall, weatherStatus,
  type ExPlayer, type ExplainDoc, type IndexEvent, type Market, type SortKey,
} from "./explain-rules";
import { useDashboardData } from "./data";
import { LABELS } from "./labels";
import { etTime } from "./lib";
import { Badge, Delta } from "./ui";
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
type Shell = { doc: ExplainDoc; market: Market; setMarket: (m: Market) => void; openPlayer: (id: number) => void };

function ExplainShell({ eyebrow, title, description, preEventFirst = false, children }: { eyebrow: string; title: string; description: string; preEventFirst?: boolean; children: (s: Shell) => React.ReactNode }) {
  const index = useDashboardData<{ events?: IndexEvent[] }>("golfprice/index.json");
  const events = useMemo(() => orderEvents(index.data?.events ?? []), [index.data]);
  const [params, setParams] = useState<Params>(readParams);
  if (index.loading) return <LoadingState label="Loading golfprice events" />;
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
      <ExplainBody key={current.event_uid} event={current} params={params} update={update} preEventFirst={preEventFirst}>{children}</ExplainBody>
    </>
  );
}

function ExplainBody({ event, params, update, preEventFirst, children }: { event: IndexEvent; params: Params; update: (p: Partial<Params>) => void; preEventFirst: boolean; children: (s: Shell) => React.ReactNode }) {
  const choices = useMemo(() => explainChoices(event), [event]);
  // Pages that offer the choice start on the pre-tournament explanation and only show the toggle when a live one exists.
  const offerToggle = preEventFirst && choices.hasLive && !!choices.preKey && !!choices.liveKey;
  const view: View = offerToggle ? params.v ?? "pre" : "live";
  const key = !preEventFirst || !choices.hasLive ? explainKeyFor(event) : view === "pre" && choices.preKey ? choices.preKey : choices.liveKey ?? explainKeyFor(event);
  const { data, loading, error } = useDashboardData<unknown>(key);
  const doc = useMemo(() => parseExplain(data), [data]);
  if (loading) return <LoadingState label="Loading the pricing explanation" />;
  const stale = !preEventFirst && event.explain_run && doc && `${doc.kind}_${doc.run}` !== event.explain_run;
  if (error || !doc || stale) return <EmptyState title={`Nothing to show for ${event.name ?? event.event_uid} yet`} detail="Prices for this event have not been run since this page was updated. They will appear after the next pricing run." />;
  const market = params.m ?? "win";
  const player = params.p ? doc.players.find((p) => p.id === params.p) ?? null : null;
  const competitors = params.vs.length || !player ? params.vs : defaultCompetitors(doc, player.id);
  return (
    <>
      <RunBanner doc={doc} />
      {offerToggle && (
        <div className="ex-viewbar">
          <SegmentedControl label="Which pricing to explain" value={view} onChange={(v) => update({ v, p: null, vs: [] })}
            options={[{ value: "pre", label: "Pre-tournament" }, { value: "live", label: choices.liveRound ? `Live (after round ${choices.liveRound})` : "Live" }]} />
          <span className="ex-muted">{view === "pre" ? "Prices as set before the first tee shot." : "Prices updated with scores so far."}</span>
        </div>
      )}
      {preEventFirst && choices.hasLive && !choices.preKey && <p className="ex-muted ex-notice">The pre-tournament explanation was not kept for this event, so this shows the live pricing.</p>}
      {children({ doc, market, setMarket: (m) => update({ m }), openPlayer: (id) => update({ p: id, vs: [] }) })}
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
      {doc.as_of && <span className="ex-muted ex-banner-time" title="When these prices were produced (US Eastern time)">Priced {etTime(doc.as_of)}</span>}
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
    <ExplainShell eyebrow="Event" title="This week" description="How we price the field: the course, where we differ from the sportsbook, and which kinds of players we favour or fade.">
      {(s) => <ThisWeek {...s} />}
    </ExplainShell>
  );
}

function ThisWeek({ doc, market, setMarket, openPlayer }: Shell) {
  const [dim, setDim] = useState("all");
  const [pickedRow, setPickedRow] = useState<string | null>(null);
  const live = doc.kind === "live";
  const fav = useMemo(() => favourites(doc, 10), [doc]);
  const edges = useMemo(() => topEdges(doc, 5), [doc]);
  const dims = useMemo(() => biasDimensions(doc), [doc]);
  const baseline = useMemo(() => biasBaseline(doc), [doc]);
  const bias = useMemo(() => biasView(doc, dim, dim === "all" ? 14 : 8, baseline), [doc, dim, baseline]);
  const markets = useMemo(() => availableMarkets(doc), [doc]);
  const shownMarket: Market = markets.includes(market) ? market : "win";
  const picked = bias.find((r) => `${r.dimension}:${r.bucket}` === pickedRow) ?? bias[0];
  const top = fav[0];
  const cc = doc.course_card;
  const ee = edges.above[0];
  const field = doc.players.filter((p) => !p.withdrawn).length;
  const avgRound = cc.scoring_avg_vs_par === null || cc.scoring_avg_vs_par === undefined || !cc.par_per_round?.[0] ? null : cc.par_per_round[0] + cc.scoring_avg_vs_par;
  const bullets = plainCourseBullets(cc.what_drives_scoring);
  const tech = [doc.edge_model.note, doc.edge_model.ridge_r2 === null ? "" : `Fit R-squared ${doc.edge_model.ridge_r2.toFixed(2)}`, cc.weather?.venue_class ? `Venue class (publisher): ${cc.weather.venue_class}` : "", cc.weather?.wind_slope_source ?? "", cc.weather?.label ?? "", cc.hole_source ? `Hole table: ${cc.hole_source}` : ""].filter(Boolean);
  return (
    <>
      <div className="kpi-grid">
        <Kpi label="Field" value={`${field} players`} detail={doc.event.cut_rule ?? "no cut"} tone="neutral" />
        {top && <Kpi label={live ? "Best chance to win" : "Favourite"} value={pct(top.probs.win.model)} detail={live ? `${top.name} · our win chance now` : `${top.name} · our win chance, sportsbook ${pct(top.probs.win.market)}`} tone="model" />}
        {!live && ee && <Kpi label="Biggest gap to the market" value={`+${(ee.edge_sg ?? 0).toFixed(2)} strokes a round`} detail={`${ee.name} · win chance ${pctPair(ee.probs.win.model, ee.probs.win.market).join(" vs ")}`} tone="positive" />}
        <Kpi label="Expected average round" value={avgRound === null ? "-" : avgRound.toFixed(1)} detail={`${signed(cc.scoring_avg_vs_par, 1)} to par · par ${cc.par_per_round?.[0] ?? "-"} · ${cc.yardage ? `${Math.round(cc.yardage).toLocaleString()} yd` : "yardage n/a"}`} tone="accent" />
      </div>

      {live && <LiveSection doc={doc} openPlayer={openPlayer} />}

      <div className="two-column">
        <Panel title="The course" eyebrow={cc.course ?? "Course card"}>
          <ul className="ex-bullets">{bullets.map((b) => <li key={b}>{b}</li>)}</ul>
          <HoleStrip holes={cc.holes} />
          {cc.hole_source && <p className="ex-muted">{plainHoleSource(cc.hole_source)} Values are for the average player in this field.</p>}
        </Panel>
        <Panel title="Weather" eyebrow="Conditions" actions={<Link className="ex-link" href="/weather-effects">Weather effects <ChevronRight size={14} /></Link>}>
          <WeatherSnapshot doc={doc} />
        </Panel>
      </div>

      {live ? (
        <p className="ex-muted ex-notice">While play is under way we do not compare our prices with the sportsbook: its prices are not live, so a gap would not mean anything. See Why priced for the comparison made before the event.</p>
      ) : (
        <>
          <Panel title="Where we differ from the market" eyebrow="Player types" actions={<span className="ex-muted" title="Positive means we rate the group higher than the market does, compared with the rest of the field.">strokes a round, compared with the field</span>}>
            <div className="ex-dimbar"><SegmentedControl label="Player-type dimension" value={dim} onChange={(v) => { setDim(v); setPickedRow(null); }}
              options={[{ value: "all", label: "Top" }, ...dims.slice(0, 7).map((d) => ({ value: d.value, label: d.label }))]} /></div>
            <p className="ex-muted ex-dimhelp">{DIMENSION_HELP[dim] ?? ""} Bars show how much higher (+) or lower (−) we rate each group than the market does, compared with the rest of the field.</p>
            <BiasChart rows={bias} active={picked ? `${picked.dimension}:${picked.bucket}` : null} onPick={(r) => setPickedRow(`${r.dimension}:${r.bucket}`)} />
            {picked && <p className="ex-bias-note"><strong>{biasSentence(picked)}</strong> {picked.p_win_model !== null && picked.p_win_market !== null && Math.sign(picked.p_win_model - picked.p_win_market) * Math.sign(picked.edge_sg ?? 0) >= 0 && <>Combined win chance: {pct(picked.p_win_model)} in our model, {pct(picked.p_win_market)} in the market. </>}</p>}
            <p className="ex-muted">Edge means how many strokes per round better (+) or worse (−) we rate a group than the market does. The reasons are estimates, not proof.{Math.abs(baseline) >= 0.03 && ` Across the whole field our numbers sit about ${Math.abs(baseline).toFixed(2)} strokes ${baseline > 0 ? "above" : "below"} the market, and the bars take that out.`}</p>
          </Panel>

          <div className="two-column">
            <EdgeList title="Where we are above the market" players={edges.above} market={shownMarket} openPlayer={openPlayer} />
            <EdgeList title="Where we are below the market" players={edges.below} market={shownMarket} openPlayer={openPlayer} />
          </div>
        </>
      )}

      <Panel title="Headline: our chances" eyebrow={live ? "Ten best chances" : "Ten shortest-priced players"} actions={<SegmentedControl label="Bet type" value={shownMarket} onChange={setMarket} options={markets.map((m) => ({ value: m, label: MARKET_LABEL[m] }))} />}>
        <div className="table-scroll"><table className="ex-table ex-cards">
          <thead><tr><th>Player</th><th title="Our chance for this bet type, from the simulations.">Our chance</th>{!live && <th title="The chance implied by sportsbook odds, with the bookmaker margin removed.">Sportsbook</th>}{!live && <th title="How far our chance is from the sportsbook's, as a share of the sportsbook's chance. Plus means we are higher.">Gap</th>}<th title="The American odds that match our chance.">Our odds</th><th title="The biggest reasons for our rating of this player.">Why</th></tr></thead>
          <tbody>{fav.map((p) => {
            const e = p.probs[shownMarket];
            const b = edgeBadge(p, shownMarket);
            return (
              <tr key={p.id} className="clickable-row" onClick={() => openPlayer(p.id)}>
                <th scope="row"><button type="button" className="ex-linkbtn" onClick={() => openPlayer(p.id)}>{p.name}</button></th>
                <td data-label="Our chance">{pct(e.model)}</td>
                {!live && <td data-label="Sportsbook">{pct(e.market)}</td>}
                {!live && <td data-label="Gap" className={b.tone === "positive" ? "pos" : b.tone === "negative" ? "neg" : ""}>{b.text}</td>}
                <td data-label="Our odds">{american(e.model)}</td><td data-label="Why" className="ex-why">{p.why}</td>
              </tr>
            );
          })}</tbody>
        </table></div>
      </Panel>
      <details className="ex-glossary-wrap"><summary>Technical details</summary>
        {tech.map((t) => <p className="ex-muted" key={t}>{t}</p>)}
      </details>
    </>
  );
}

function EdgeList({ title, players, market, openPlayer }: { title: string; players: ExPlayer[]; market: Market; openPlayer: (id: number) => void }) {
  return (
    <Panel title={title} eyebrow="Biggest gaps" actions={<span className="ex-muted" title="Players with a win price of at least 0.5% in the market.">contenders only</span>}>
      {players.length === 0 ? <p className="ex-muted">No contenders with a sportsbook price in this run.</p> : (
        <ul className="ex-edges">{players.map((p) => {
          const [a, b] = pctPair(p.probs[market].model, p.probs[market].market);
          return (
            <li key={p.id}>
              <button type="button" onClick={() => openPlayer(p.id)}>
                <span className="ex-edge-head"><strong>{p.name}</strong><Delta value={p.edge_sg ?? 0} digits={2} suffix=" strokes" /></span>
                <span className="ex-edge-nums">{MARKET_LABEL[market]} chance: {a} (ours) vs {b} (sportsbook){p.tags.length > 0 && <> · <Tags tags={p.tags} max={3} /></>}</span>
                {p.why_edge && <span className="ex-edge-why">{p.why_edge}</span>}
              </button>
            </li>
          );
        })}</ul>
      )}
    </Panel>
  );
}

function WeatherSnapshot({ doc }: { doc: ExplainDoc }) {
  const w = doc.course_card.weather ?? {};
  const status = weatherStatus(doc);
  if (!w || (!w.venue_class && !w.label && !status.long)) return <p className="ex-muted">No weather information in this run.</p>;
  const cut = cutLineRows(doc);
  return (
    <div className="ex-wx">
      <p>{status.long}{w.issued_at ? <> Forecast issued {etTime(w.issued_at)}.</> : null}</p>
      {w.wind_slope !== null && w.wind_slope !== undefined && <p>Rule of thumb for this course: each extra mph of wind adds about {w.wind_slope.toFixed(1)} strokes to the average round.</p>}
      {w.waves && w.waves.length > 0 && (
        <>
          <div className="table-scroll"><table className="ex-table"><thead><tr><th>Round</th><th>Tee time</th><th>Average wind</th><th title="Minus means easier than the field average, plus means harder.">Strokes vs field</th></tr></thead>
            <tbody>{w.waves.map((x) => <tr key={`${x.round}${x.wave}`}><th scope="row">R{x.round}</th><td>{x.wave === "AM" ? "Morning" : x.wave === "PM" ? "Afternoon" : x.wave}</td><td>{x.mean_wind ?? "-"} mph</td><td className={(x.rel ?? 0) > 0 ? "neg" : "pos"}>{signed(x.rel)}</td></tr>)}</tbody></table></div>
          <p className="ex-muted">Minus means easier than the field average, plus means harder.{doc.kind === "live" ? " Only the rounds still to play are listed." : ""}</p>
        </>
      )}
      {cut.length > 0 && <p className="ex-muted">Expected cut-line move from weather: {cut.map((f) => `round ${f.round} ${signed(f.shift)}`).join(" · ")} strokes (plus means a higher cut line).</p>}
      {!w.waves?.length && w.label && !status.long && <p className="ex-muted">{plainText(w.label)}</p>}
    </div>
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
  edge_sg: "Gap in strokes: how many strokes per round better (+) or worse (-) we rate a player than the sportsbook does.",
  rel: "Gap in %: how far our chance is from the sportsbook's, as a share of the sportsbook's chance.",
  model: "Our chance for the chosen bet type.", market: "The sportsbook's chance for the chosen bet type.",
  mu: "Expected skill: strokes per round better (+) or worse (-) than the average player in this field.", name: "Alphabetical.",
};
const COLUMN_HELP = {
  pos: "Current place on the leaderboard (MC = missed the cut, WD = withdrew).",
  model: "Our chance for this bet type, from the simulations.",
  market: "The chance implied by sportsbook odds, with the bookmaker margin removed.",
  rel: "How far our chance is from the sportsbook's, as a share of the sportsbook's. Plus means we are higher than the sportsbook.",
  edge_sg: "The same gap expressed as strokes per round of skill: how much better (+) or worse (-) we rate them than the market does.",
  mu: "Our estimate of their skill: strokes per round better (+) or worse (-) than the average player in this field.",
  sd: "How much their round scores typically bounce around, in strokes. Higher means a streakier, less predictable player.",
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
  const rows = useMemo(() => sortWhy(filterWhy(doc.players, { query, tags, market, onlyEdge: only, top: top === "all" ? null : Number(top) }), sort.key, sort.dir, market), [doc.players, query, tags, market, only, top, sort]);
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
          <thead><tr>{th("name", "Player")}{live && th("pos", "Pos", COLUMN_HELP.pos)}{hasModel && th("model", "Our chance", COLUMN_HELP.model)}{hasMarket && th("market", "Sportsbook", COLUMN_HELP.market)}{hasMarket && th("rel", "Gap (%)", COLUMN_HELP.rel)}{hasGap && hasMarket && th("edge_sg", "Gap (strokes)", COLUMN_HELP.edge_sg)}{th("mu", "Expected skill", COLUMN_HELP.mu)}{hasSd && th("sd", "Spread", COLUMN_HELP.sd)}<th title="The biggest reasons our price differs from the sportsbook.">Why</th></tr></thead>
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
                  <td data-label="Expected skill">{signed(p.mu_rel)}</td>{hasSd && <td data-label="Spread">{p.sd?.toFixed(2) ?? "-"}</td>}
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
          <Kpi label={`Expected skill ${LABELS.vsField.short}`} value={`${signed(player.mu_rel)} strokes a round`} detail={`Round-to-round spread ${player.sd?.toFixed(2) ?? "-"} strokes`} tone={(player.mu_rel ?? 0) >= 0 ? "positive" : "negative"} />
          {f && <Kpi label="Median finish" value={String(f.median)} detail={`Likely range: ${f.p10} to ${f.p90}`} tone="accent" />}
          {f && <Kpi label="Top 10 chance" value={pct(t[2])} detail={`Win chance ${pct(t[0])}${doc.event.cut_round ? ` · misses the cut ${pct(f.p_miss_cut, 0)}` : ""}`} tone="neutral" />}
        </div>
        {player.why_edge && <p className="ex-callout">{player.why_edge}</p>}

        <section>
          <h3 className="ex-h3">Why this price</h3>
          <p className="ex-muted">Strokes per round above (+) or below (−) the average player in this field ({LABELS.vsField.short}); the {LABELS.vsPgaAvg.short} view is on the profile. The skill rows add up to their expected skill{doc.kind === "live" ? "; the in-play rows are part of it" : ""}; weather is shown separately because it adjusts scores on top.</p>
          <Waterfall {...wf} />
          {Object.keys(player.attribution).length > 0 && (
            <p className="ex-muted">{doc.kind === "live" ? "" : `Our gap to the sportsbook (${gapText(player.edge_sg)} strokes a round) is linked most to: `}{doc.kind === "live" ? "" : Object.entries(player.attribution).map(([k, v]) => `${doc.components_legend[k]?.label ?? plainText(k)} ${signed(v)}`).join(", ")}{doc.kind === "live" ? "" : ". This is each piece's share of the gap, so it can differ from the size of the piece above."}</p>
          )}
        </section>

        <section>
          <h3 className="ex-h3">Our chance vs the sportsbook</h3>
          <MarketRows player={player} hasCut={!!doc.event.cut_round} />
        </section>

        <section>
          <div className="ex-sec-head"><h3 className="ex-h3">Finish shape</h3>
            <SegmentedControl label="Chart type" value={mode} onChange={setMode} options={[{ value: "probability", label: "Position" }, { value: "cumulative", label: "Cumulative" }]} /></div>
          {doc.finish.method === "unavailable" || !f ? <p className="ex-muted">Finish odds are not available for this run.</p> : (
            <>
              <FinishChart series={series} fieldSize={doc.players.filter((p) => !p.withdrawn).length} cutTopN={doc.event.cut_round ? doc.event.cut_top_n : null} mode={mode} />
              <div className="ex-compare">
                <h4>Compare with</h4>
                <div className="chip-row">
                  {competitors.map((id) => <span className="chip" key={id}>{byId.get(id)?.name ?? id}<button type="button" onClick={() => setCompetitors(competitors.filter((x) => x !== id))} aria-label={`Remove ${byId.get(id)?.name}`}><X size={13} /></button></span>)}
                  {competitors.length === 0 && <span className="ex-muted">nobody selected</span>}
                </div>
                <div className="ex-compare-actions">
                  <button type="button" onClick={() => setCompetitors(defaultCompetitors(doc, player.id, 3))}>Same-odds neighbours</button>
                  <button type="button" onClick={() => setCompetitors([])}>Clear</button>
                </div>
                <label className="search-box"><Search size={15} /><input value={q} onChange={(e) => setQ(e.target.value)} placeholder={competitors.length >= MAX_OVERLAY - 1 ? "Maximum reached" : "Add a competitor"} disabled={competitors.length >= MAX_OVERLAY - 1} aria-label="Add competitor" /></label>
                {hits.length > 0 && <ul className="ex-hits">{hits.map((p) => <li key={p.id}><button type="button" onClick={() => { setCompetitors(toggleCompetitor(competitors, p.id)); setQ(""); }}>{p.name}<small>{pct(p.probs.win.model)} win</small></button></li>)}</ul>}
              </div>
              <div className="table-scroll"><table className="ex-table ex-compare-table">
                <thead><tr><th>Player</th><th title="Expected skill: strokes per round better (+) or worse (-) than the field average.">Skill</th><th>Win</th><th>Top 10</th><th title="The finish position that half of the simulations beat and half did not.">Median finish</th>{doc.event.cut_round ? <th>Misses cut</th> : null}</tr></thead>
                <tbody>{withShape.map((p, i) => (
                  <tr key={p.id}><th scope="row"><i className="ex-dot" style={{ background: PLAYER_COLORS[i % PLAYER_COLORS.length] }} aria-hidden="true" />
                    {p.id === player.id ? p.name : <button type="button" className="ex-linkbtn" onClick={() => onSelect(p.id)}>{p.name}</button>}</th>
                    <td>{signed(p.mu_rel)}</td><td>{pct(p.probs.win.model)}</td><td>{pct(p.probs.top_10.model)}</td><td>{p.finish!.median}</td>{doc.event.cut_round ? <td>{pct(p.finish!.p_miss_cut, 0)}</td> : null}</tr>
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
        <p className="ex-muted ex-foot"><CloudSun size={13} /> Wave, time-of-day and wind effects are in the waterfall (weather rows) and in <Link className="ex-link" href="/weather-effects">Weather effects</Link>.</p>
      </aside>
    </div>
  );
}
