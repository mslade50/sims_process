"use client";

/**
 * "This week" event page and "Why priced" table + the player drawer (finish shape, score distribution, why-this-price waterfall, competitor overlay).
 * Data: golfprice/explain/<event>/latest.json (golfprice/explain_export.py; week runs before the event, live runs after each round).
 * Design-system tokens/components only (tokens.css, ui.tsx); logic in explain-rules.ts.
 */
import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowDown, ArrowUp, ChevronRight, CloudSun, Search, X } from "lucide-react";
import { EmptyState, Kpi, LoadingState, PageIntro, Panel, SegmentedControl } from "./components";
import { BiasChart, FinishChart, HoleStrip, LegendGroups, MarketRows, PLAYER_COLORS, ScoreFan, Waterfall, type FinishSeries } from "./ExplainCharts";
import {
  MARKETS, MARKET_LABEL, MAX_OVERLAY, american, availableTags, biasDimensions, biasForDimension, biasSentence, defaultCompetitors, edgeBadge, explainKeyFor, favourites, filterWhy,
  leaderboard, movers, nameMatches, orderEvents, parseExplain, pct, signed, sortWhy, toParText, toggleCompetitor, topEdges, topK, waterfall,
  type ExPlayer, type ExplainDoc, type IndexEvent, type Market, type SortKey,
} from "./explain-rules";
import { useDashboardData } from "./data";
import { Badge, Delta, FreshnessBadge } from "./ui";
import "./explain.css";

/* ------------------------------------------------------------------ URL state (?e=<event>&p=<player>&vs=a,b,c&m=<market>) */
type Params = { e: string | null; p: number | null; vs: number[]; m: Market | null };
function readParams(): Params {
  if (typeof window === "undefined") return { e: null, p: null, vs: [], m: null };
  const q = new URLSearchParams(window.location.search);
  const p = Number(q.get("p"));
  const m = q.get("m") as Market | null;
  return { e: q.get("e"), p: Number.isFinite(p) && p > 0 ? p : null, vs: (q.get("vs") ?? "").split(",").map(Number).filter((n) => Number.isFinite(n) && n > 0), m: m && (MARKETS as readonly string[]).includes(m) ? m : null };
}
function writeParams(next: Params) {
  if (typeof window === "undefined") return;
  const q = new URLSearchParams();
  if (next.e) q.set("e", next.e);
  if (next.p) q.set("p", String(next.p));
  if (next.vs.length) q.set("vs", next.vs.join(","));
  if (next.m && next.m !== "win") q.set("m", next.m);
  const s = q.toString();
  window.history.replaceState(null, "", `${window.location.pathname}${s ? `?${s}` : ""}`);
}

/* ------------------------------------------------------------------ shell: event picker, loading, empty states */
type Shell = { doc: ExplainDoc; market: Market; setMarket: (m: Market) => void; openPlayer: (id: number) => void };

function ExplainShell({ eyebrow, title, description, children }: { eyebrow: string; title: string; description: string; children: (s: Shell) => React.ReactNode }) {
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
          <SegmentedControl label="Event" value={current.event_uid} options={events.slice(0, 4).map((e) => ({ value: e.event_uid, label: `${(e.tour ?? "").toUpperCase()} ${e.name ?? e.event_uid}`.trim() }))}
            onChange={(v) => update({ e: v, p: null, vs: [] })} />
        ) : undefined} />
      <ExplainBody key={current.event_uid} event={current} params={params} update={update}>{children}</ExplainBody>
    </>
  );
}

function ExplainBody({ event, params, update, children }: { event: IndexEvent; params: Params; update: (p: Partial<Params>) => void; children: (s: Shell) => React.ReactNode }) {
  const { data, loading, error } = useDashboardData<unknown>(explainKeyFor(event));
  const doc = useMemo(() => parseExplain(data), [data]);
  if (loading) return <LoadingState label="Loading the explain document" />;
  if (error || !doc) return <EmptyState title={`No explain document for ${event.name ?? event.event_uid} yet`} detail="It is published with every run (python -m golfprice.publish_dashboard). Older runs published before this view existed have none; the next run does." />;
  const market = params.m ?? "win";
  const player = params.p ? doc.players.find((p) => p.id === params.p) ?? null : null;
  const competitors = params.vs.length || !player ? params.vs : defaultCompetitors(doc, player.id);
  return (
    <>
      <RunBanner doc={doc} />
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
      <Badge tone={live ? "positive" : "model"} pulse={live}>{live ? `In play · after round ${doc.after_round}` : "Pre-tournament prices"}</Badge>
      <span className="ex-banner-name">{doc.event.name}{doc.event.course ? ` · ${doc.event.course}` : ""}</span>
      <FreshnessBadge at={doc.as_of} label="Run" />
      {doc.model.weather_layer && <Badge tone="neutral">weather {doc.model.weather_layer}</Badge>}
      {doc.model.overrides_applied.length > 0 && <Badge tone="warning">{doc.model.overrides_applied.length} owner override{doc.model.overrides_applied.length > 1 ? "s" : ""}</Badge>}
      <Badge tone={doc.finish.method === "unavailable" ? "warning" : "neutral"}>shape: {doc.finish.method.replace("_", " ")}</Badge>
    </div>
  );
}

const Tags = ({ tags, max = 4 }: { tags: string[]; max?: number }) => <span className="ex-tags">{tags.slice(0, max).map((t) => <em key={t}>{t}</em>)}{tags.length > max && <em>+{tags.length - max}</em>}</span>;

/* ------------------------------------------------------------------ This week */
export function ThisWeekView() {
  return (
    <ExplainShell eyebrow="Event" title="This week" description="Why we price the field the way we do: the course, where we differ from the market, and which kinds of players we favour or fade.">
      {(s) => <ThisWeek {...s} />}
    </ExplainShell>
  );
}

function ThisWeek({ doc, market, setMarket, openPlayer }: Shell) {
  const [dim, setDim] = useState("all");
  const [pickedRow, setPickedRow] = useState<string | null>(null);
  const fav = useMemo(() => favourites(doc, 10), [doc]);
  const edges = useMemo(() => topEdges(doc, 5), [doc]);
  const dims = useMemo(() => biasDimensions(doc), [doc]);
  const bias = useMemo(() => biasForDimension(doc, dim).slice(0, dim === "all" ? 14 : 8), [doc, dim]);
  const picked = bias.find((r) => `${r.dimension}:${r.bucket}` === pickedRow) ?? bias[0];
  const top = fav[0];
  const live = doc.kind === "live";
  const cc = doc.course_card;
  const ee = edges.above[0];
  return (
    <>
      <div className="kpi-grid">
        <Kpi label="Field" value={String(doc.players.filter((p) => !p.withdrawn).length)} detail={doc.event.cut_rule ?? "no cut"} tone="neutral" />
        {top && <Kpi label="Favourite" value={pct(top.probs.win.model)} detail={`${top.name} · model win, market ${pct(top.probs.win.market)}`} tone="model" />}
        {ee && <Kpi label="Biggest edge up" value={`+${(ee.edge_sg ?? 0).toFixed(2)} sg`} detail={`${ee.name} · ${pct(ee.probs.win.model)} vs ${pct(ee.probs.win.market)}`} tone="positive" />}
        <Kpi label="Course scoring" value={cc.scoring_avg_vs_par === null || cc.scoring_avg_vs_par === undefined ? "-" : signed(cc.scoring_avg_vs_par)} detail={`par ${cc.par_per_round?.[0] ?? "-"} · ${cc.yardage ? `${Math.round(cc.yardage).toLocaleString()} yd` : "yardage n/a"}`} tone="accent" />
      </div>

      {live && <LiveSection doc={doc} openPlayer={openPlayer} />}

      <div className="two-column">
        <Panel title="The course" eyebrow={cc.course ?? "Course card"}>
          <ul className="ex-bullets">{cc.what_drives_scoring.map((b) => <li key={b}>{b}</li>)}</ul>
          <HoleStrip holes={cc.holes} />
          {cc.hole_source && <p className="ex-muted">Hole table: {cc.hole_source}. Values are for the field-average player.</p>}
        </Panel>
        <Panel title="Weather snapshot" eyebrow="Conditions" actions={<Link className="ex-link" href="/weather-effects">Weather effects <ChevronRight size={14} /></Link>}>
          <WeatherSnapshot doc={doc} />
        </Panel>
      </div>

      <Panel title="Where we differ from the market" eyebrow="Player types" actions={<span className="ex-muted">strokes per round, model minus market</span>}>
        <div className="ex-dimbar"><SegmentedControl label="Player-type dimension" value={dim} onChange={(v) => { setDim(v); setPickedRow(null); }}
          options={[{ value: "all", label: "Top" }, ...dims.slice(0, 7).map((d) => ({ value: d.value, label: d.label }))]} /></div>
        <BiasChart rows={bias} active={picked ? `${picked.dimension}:${picked.bucket}` : null} onPick={(r) => setPickedRow(`${r.dimension}:${r.bucket}`)} />
        {picked && <p className="ex-bias-note"><strong>{biasSentence(picked)}</strong> {picked.p_win_model !== null && picked.p_win_market !== null && <>Together they hold {pct(picked.p_win_model)} of the title in our model vs {pct(picked.p_win_market)} in the market. </>}
          {picked.drivers.length > 0 && <>Largest attributed components: {picked.drivers.map((d) => `${d.label.toLowerCase()} ${signed(d.attr_sg)}`).join(", ")}.</>}</p>}
        <p className="ex-muted">{doc.edge_model.note} Fit R&sup2; {doc.edge_model.ridge_r2 === null ? "-" : doc.edge_model.ridge_r2.toFixed(2)}.</p>
      </Panel>

      <div className="two-column">
        <EdgeList title="Where we are above the market" players={edges.above} market={market} openPlayer={openPlayer} />
        <EdgeList title="Where we are below the market" players={edges.below} market={market} openPlayer={openPlayer} />
      </div>

      <Panel title="Headline: model vs market" eyebrow="Top 10 by market" actions={<SegmentedControl label="Market" value={market} onChange={setMarket} options={MARKETS.map((m) => ({ value: m, label: MARKET_LABEL[m] }))} />}>
        <div className="table-scroll"><table className="ex-table">
          <thead><tr><th>Player</th><th>Model</th><th>Market</th><th>Edge</th><th>Fair odds</th><th>Why</th></tr></thead>
          <tbody>{fav.map((p) => {
            const e = p.probs[market];
            const b = edgeBadge(p, market);
            return (
              <tr key={p.id} className="clickable-row" onClick={() => openPlayer(p.id)}>
                <th scope="row"><button type="button" className="ex-linkbtn" onClick={() => openPlayer(p.id)}>{p.name}</button></th>
                <td>{pct(e.model)}</td><td>{pct(e.market)}</td><td className={b.tone === "positive" ? "pos" : b.tone === "negative" ? "neg" : ""}>{b.text}</td>
                <td>{american(e.fair ?? e.model)}</td><td className="ex-why">{p.why}</td>
              </tr>
            );
          })}</tbody>
        </table></div>
      </Panel>
    </>
  );
}

function EdgeList({ title, players, market, openPlayer }: { title: string; players: ExPlayer[]; market: Market; openPlayer: (id: number) => void }) {
  return (
    <Panel title={title} eyebrow="Top edges" actions={<span className="ex-muted">contenders only</span>}>
      {players.length === 0 ? <p className="ex-muted">None with a market price in this run.</p> : (
        <ul className="ex-edges">{players.map((p) => (
          <li key={p.id}>
            <button type="button" onClick={() => openPlayer(p.id)}>
              <span className="ex-edge-head"><strong>{p.name}</strong><Delta value={p.edge_sg ?? 0} digits={2} suffix=" sg" /></span>
              <span className="ex-edge-nums">{MARKET_LABEL[market]} {pct(p.probs[market].model)} vs {pct(p.probs[market].market)} · <Tags tags={p.tags} max={3} /></span>
              {p.why_edge && <span className="ex-edge-why">{p.why_edge}</span>}
            </button>
          </li>
        ))}</ul>
      )}
    </Panel>
  );
}

function WeatherSnapshot({ doc }: { doc: ExplainDoc }) {
  const w = doc.course_card.weather ?? {};
  if (!w || (!w.venue_class && !w.label)) return <p className="ex-muted">No weather layer information in this run.</p>;
  return (
    <div className="ex-wx">
      <p><Badge tone={w.applied_in_prices ? "positive" : "warning"}>{w.applied_in_prices ? "applied in prices" : "not applied in prices"}</Badge> {w.status ? <Badge tone="neutral">{w.status}</Badge> : null}</p>
      {w.venue_class && <p>Venue class <strong>{w.venue_class}</strong>{w.wind_slope !== null && w.wind_slope !== undefined ? <>; wind sensitivity <strong>{w.wind_slope}</strong> ({w.wind_slope_source})</> : null}.</p>}
      {w.waves && w.waves.length > 0 && (
        <div className="table-scroll"><table className="ex-table"><thead><tr><th>Round</th><th>Wave</th><th>Wind</th><th>Strokes vs field</th></tr></thead>
          <tbody>{w.waves.map((x) => <tr key={`${x.round}${x.wave}`}><th scope="row">R{x.round}</th><td>{x.wave}</td><td>{x.mean_wind ?? "-"} mph</td><td className={(x.rel ?? 0) > 0 ? "neg" : "pos"}>{signed(x.rel)}</td></tr>)}</tbody></table></div>
      )}
      {w.field && w.field.length > 0 && <p className="ex-muted">Cut-line shift: {w.field.map((f) => `R${f.round} ${signed(f.cut_line_shift)}`).join(" · ")} strokes (+ = tougher).</p>}
      {!w.waves?.length && w.label && <p className="ex-muted">{w.label}</p>}
    </div>
  );
}

function LiveSection({ doc, openPlayer }: { doc: ExplainDoc; openPlayer: (id: number) => void }) {
  const lb = leaderboard(doc, 12);
  const mv = movers(doc, 6);
  const withB8 = doc.players.filter((p) => Math.abs(p.live?.d_b8_since_last ?? 0) >= 0.02).length;
  const withCon = doc.players.filter((p) => Math.abs(p.live?.d_contention_since_last ?? 0) >= 0.02).length;
  return (
    <Panel title={`In play: after round ${doc.after_round}`} eyebrow="Leaderboard and changes since the last run" actions={doc.since_last ? <span className="ex-muted">vs {doc.since_last.prev_after_round !== null ? `after R${doc.since_last.prev_after_round}` : doc.since_last.prev_run}</span> : undefined}>
      <p className="ex-muted">{withB8} players moved by the in-event form shift (B8) and {withCon} by the contention layer since the previous run. Weather for the next round: see Weather effects.</p>
      <div className="table-scroll"><table className="ex-table">
        <thead><tr><th>Pos</th><th>Player</th><th>To par</th><th>Win</th><th>Δ win</th><th>Top 10</th><th>Mean</th><th>B8</th><th>Contention</th></tr></thead>
        <tbody>{lb.map((p) => (
          <tr key={p.id} className="clickable-row" onClick={() => openPlayer(p.id)}>
            <td>{p.live?.tier === 0 ? p.live?.pos ?? "-" : p.live?.tier === 2 ? "MC" : "WD"}</td>
            <th scope="row"><button type="button" className="ex-linkbtn" onClick={() => openPlayer(p.id)}>{p.name}</button></th>
            <td>{toParText(p.live?.to_par)}</td><td>{pct(p.probs.win.model)}</td>
            <td><Delta value={(p.live?.since_last.win ?? 0) * 100} digits={1} suffix="pp" /></td>
            <td>{pct(p.probs.top_10.model)}</td><td>{signed(p.live?.mu_live)}</td><td>{signed(p.live?.b8_delta)}</td><td>{signed(p.live?.contention)}</td>
          </tr>
        ))}</tbody>
      </table></div>
      {mv.length > 0 && <><h3 className="ex-h3">Biggest movers (win probability)</h3>
        <ul className="ex-movers">{mv.map((p) => <li key={p.id}><button type="button" onClick={() => openPlayer(p.id)}>{p.name} <Delta value={(p.live?.since_last.win ?? 0) * 100} digits={1} suffix="pp" /><small> now {pct(p.probs.win.model)}</small></button></li>)}</ul></>}
    </Panel>
  );
}

/* ------------------------------------------------------------------ Why priced */
export function WhyPricedView() {
  return (
    <ExplainShell eyebrow="Pricing" title="Why priced" description="Every player's model price against the market, sortable and filterable by player type, with the strokes that explain it.">
      {(s) => <WhyPriced {...s} />}
    </ExplainShell>
  );
}

const SORT_LABEL: Array<{ key: SortKey; label: string }> = [
  { key: "edge_sg", label: "Edge (strokes)" }, { key: "rel", label: "Edge (%)" }, { key: "model", label: "Model" }, { key: "market", label: "Market" }, { key: "mu", label: "Mean" }, { key: "name", label: "Name" },
];

function WhyPriced({ doc, market, setMarket, openPlayer }: Shell) {
  const [query, setQuery] = useState("");
  const [tags, setTags] = useState<string[]>([]);
  const [only, setOnly] = useState<"all" | "above" | "below">("all");
  const [top, setTop] = useState<"all" | "60" | "25">("60");
  const [sort, setSort] = useState<{ key: SortKey; dir: "asc" | "desc" }>({ key: "edge_sg", dir: "desc" });
  const live = doc.kind === "live";
  const tagOptions = useMemo(() => availableTags(doc.players), [doc.players]);
  const rows = useMemo(() => sortWhy(filterWhy(doc.players, { query, tags, market, onlyEdge: only, top: top === "all" ? null : Number(top) }), sort.key, sort.dir, market), [doc.players, query, tags, market, only, top, sort]);
  const th = (key: SortKey, label: string) => (
    <th aria-sort={sort.key === key ? (sort.dir === "asc" ? "ascending" : "descending") : "none"}>
      <button type="button" onClick={() => setSort((s) => ({ key, dir: s.key === key && s.dir === "desc" ? "asc" : "desc" }))}>
        {label}{sort.key === key && (sort.dir === "asc" ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
      </button>
    </th>
  );
  return (
    <Panel title={`${rows.length} players`} eyebrow={MARKET_LABEL[market]} actions={<SegmentedControl label="Market" value={market} onChange={setMarket} options={MARKETS.map((m) => ({ value: m, label: MARKET_LABEL[m] }))} />}>
      <div className="ex-filters">
        <label className="search-box"><Search size={15} /><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search player" aria-label="Search player" /></label>
        <SegmentedControl label="Direction" value={only} onChange={setOnly} options={[{ value: "all", label: "All" }, { value: "above", label: "We are higher" }, { value: "below", label: "We are lower" }]} />
        <SegmentedControl label="Price range" value={top} onChange={setTop} options={[{ value: "25", label: "Top 25" }, { value: "60", label: "Top 60" }, { value: "all", label: "Whole field" }]} />
        <label className="ex-sort">Sort <select value={sort.key} onChange={(e) => setSort({ key: e.target.value as SortKey, dir: e.target.value === "name" ? "asc" : "desc" })}>{SORT_LABEL.map((o) => <option key={o.key} value={o.key}>{o.label}</option>)}</select></label>
      </div>
      <div className="ex-chips" role="group" aria-label="Filter by player type">
        {tagOptions.map((t) => <button key={t} type="button" aria-pressed={tags.includes(t)} className={tags.includes(t) ? "on" : ""} onClick={() => setTags((cur) => (cur.includes(t) ? cur.filter((x) => x !== t) : [...cur, t]))}>{t}</button>)}
        {tags.length > 0 && <button type="button" className="clear" onClick={() => setTags([])}><X size={12} /> clear</button>}
      </div>
      <div className="table-scroll">
        <table className="ex-table ex-why-table">
          <thead><tr>{th("name", "Player")}{live && th("pos", "Pos")}{th("model", "Model")}{th("market", "Market")}{th("rel", "Edge")}{th("edge_sg", "Δ strokes")}{th("mu", "Mean")}{th("sd", "SD")}<th>Why</th></tr></thead>
          <tbody>
            {rows.map((p) => {
              const e = p.probs[market];
              const b = edgeBadge(p, market);
              return (
                <tr key={p.id} className="clickable-row" onClick={() => openPlayer(p.id)}>
                  <th scope="row" className="ex-sticky"><button type="button" className="ex-linkbtn" onClick={(ev) => { ev.stopPropagation(); openPlayer(p.id); }}>{p.name}</button><Tags tags={p.tags} max={3} /></th>
                  {live && <td>{p.live?.tier === 0 ? p.live?.pos ?? "-" : p.live?.tier === 2 ? "MC" : "WD"}</td>}
                  <td>{pct(e.model)}</td><td>{pct(e.market)}</td>
                  <td className={b.tone === "positive" ? "pos" : b.tone === "negative" ? "neg" : ""}>{b.text}</td>
                  <td className={(p.edge_sg ?? 0) > 0 ? "pos" : (p.edge_sg ?? 0) < 0 ? "neg" : ""}>{signed(p.edge_sg)}</td>
                  <td>{signed(p.mu_rel)}</td><td>{p.sd?.toFixed(2) ?? "-"}</td>
                  <td className="ex-why">{p.why_edge ?? p.why}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {rows.length === 0 && <EmptyState title="No players match" detail="Clear a filter or the search." />}
      <details className="ex-glossary-wrap"><summary>What the components mean</summary><LegendGroups legend={doc.components_legend} /></details>
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
            <span className="eyebrow">{doc.kind === "live" ? `After round ${doc.after_round}` : "Pre-tournament"}</span>
            <h2>{player.name}</h2>
            <Tags tags={player.tags} max={8} />
          </div>
          <button type="button" className="ex-close" onClick={onClose} aria-label="Close" ref={closeRef}><X size={20} /></button>
        </header>

        <div className="ex-drawer-kpis">
          <Kpi label={`${MARKET_LABEL[market]} model`} value={pct(player.probs[market].model)} detail={`market ${pct(player.probs[market].market)}`} tone="model" />
          <Kpi label="Mean vs field" value={`${signed(player.mu_rel)} sg`} detail={`SD ${player.sd?.toFixed(2) ?? "-"} / round`} tone={(player.mu_rel ?? 0) >= 0 ? "positive" : "negative"} />
          {f && <Kpi label="Median finish" value={String(f.median)} detail={`10th-90th: ${f.p10}-${f.p90}`} tone="accent" />}
          {f && <Kpi label="Top 10 (shape)" value={pct(t[2])} detail={`win ${pct(t[0])} · miss cut ${pct(f.p_miss_cut, 0)}`} tone="neutral" />}
        </div>
        {player.why_edge && <p className="ex-callout">{player.why_edge}</p>}

        <section>
          <h3 className="ex-h3">Why this price</h3>
          <p className="ex-muted">Strokes per round above (+) or below (−) the average player in this field. They add up to his mean skill{doc.kind === "live" ? "; in-play and weather rows adjust the next round(s)" : ""}.</p>
          <Waterfall {...wf} />
          {Object.keys(player.attribution).length > 0 && (
            <p className="ex-muted">Gap to market (strokes/round, {signed(player.edge_sg)}) attributed to: {Object.entries(player.attribution).map(([k, v]) => `${doc.components_legend[k]?.label ?? k} ${signed(v)}`).join(", ")}.</p>
          )}
        </section>

        <section>
          <h3 className="ex-h3">Model vs market</h3>
          <MarketRows player={player} />
        </section>

        <section>
          <div className="ex-sec-head"><h3 className="ex-h3">Finish shape</h3>
            <SegmentedControl label="Chart type" value={mode} onChange={setMode} options={[{ value: "probability", label: "Position" }, { value: "cumulative", label: "Cumulative" }]} /></div>
          {doc.finish.method === "unavailable" || !f ? <p className="ex-muted">No finish distribution in this run ({doc.finish.reason ?? "tapes not available"}).</p> : (
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
                <thead><tr><th>Player</th><th>Mean</th><th>Win</th><th>Top 10</th><th>Median</th><th>Miss cut</th></tr></thead>
                <tbody>{withShape.map((p, i) => (
                  <tr key={p.id}><th scope="row"><i className="ex-dot" style={{ background: PLAYER_COLORS[i % PLAYER_COLORS.length] }} aria-hidden="true" />
                    {p.id === player.id ? p.name : <button type="button" className="ex-linkbtn" onClick={() => onSelect(p.id)}>{p.name}</button>}</th>
                    <td>{signed(p.mu_rel)}</td><td>{pct(p.probs.win.model)}</td><td>{pct(p.probs.top_10.model)}</td><td>{p.finish!.median}</td><td>{pct(p.finish!.p_miss_cut, 0)}</td></tr>
                ))}</tbody></table></div>
              <p className="ex-muted">Shape from {doc.finish.method === "exact_tapes" ? "the simulation's own draws" : `a replay of the hole engine (${doc.finish.n_draws?.toLocaleString()} draws)`}; mean absolute difference to the priced top-20: {doc.finish.calibration?.top_20 ? pct(doc.finish.calibration.top_20.mean_abs, 2) : "-"}.</p>
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
              <span>Mean pre-event → now <b>{signed(live.mu_pre)} → {signed(live.mu_live)}</b></span>
              <span>Since last run <b>{signed(live.d_mu_since_last)}</b> (B8 {signed(live.d_b8_since_last)}, contention {signed(live.d_contention_since_last)})</span>
            </div>
            {live.rounds.length > 0 && <div className="table-scroll"><table className="ex-table"><thead><tr><th>Round</th><th>Score</th><th>SG total</th><th>OTT</th><th>APP</th><th>ARG</th><th>PUTT</th></tr></thead>
              <tbody>{live.rounds.map((r) => <tr key={r.round}><th scope="row">R{r.round}</th><td>{r.score ?? "-"}</td><td>{signed(r.sg_total)}</td><td>{signed(r.sg_ott)}</td><td>{signed(r.sg_app)}</td><td>{signed(r.sg_arg)}</td><td>{signed(r.sg_putt)}</td></tr>)}</tbody></table></div>}
            <div className="table-scroll"><table className="ex-table"><thead><tr><th>Market</th><th>Pre-event</th><th>Now</th><th>Since last run</th></tr></thead>
              <tbody>{MARKETS.map((m) => <tr key={m}><th scope="row">{MARKET_LABEL[m]}</th><td>{pct(live.pre_event[m])}</td><td>{pct(player.probs[m].model)}</td><td><Delta value={(live.since_last[m] ?? 0) * 100} digits={1} suffix="pp" /></td></tr>)}</tbody></table></div>
          </section>
        )}
        <p className="ex-muted ex-foot"><CloudSun size={13} /> Wave, time-of-day and wind effects are in the waterfall (weather rows) and in <Link className="ex-link" href="/weather-effects">Weather effects</Link>.</p>
      </aside>
    </div>
  );
}
