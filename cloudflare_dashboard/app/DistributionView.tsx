"use client";

import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Search, X, Layers3, ArrowRight } from "lucide-react";
import { useDashboardData } from "./data";
import { EmptyState, LoadingState, PageIntro, Panel, SegmentedControl } from "./components";
import { american, MARKETS, MARKET_LABEL, nameMatches, orderEvents, parseExplain, pct, type ExPlayer } from "./explain-rules";
import { FINISH_CHART_CAPTION, bucketRows, checkpoints, cutReference, marker, markerLabel, matchup, runLabel, type DistributionDoc, type DistributionEvent } from "./distributions-rules";
import { etTime } from "./lib";
import "./explain.css";
import "./distributions.css";

const COLORS = ["var(--chart-1)", "var(--chart-3)", "var(--chart-2)", "var(--chart-4)"];
const name = (p: ExPlayer) => p.name.includes(",") ? p.name.split(",").reverse().join(" ").trim() : p.name;
const MARKET_HELP: Record<string, string> = { win: "Chance of winning outright.", top_5: "Chance of finishing in the top 5, ties included per the market rules.", top_10: "Chance of finishing in the top 10, ties included per the market rules.", top_20: "Chance of finishing in the top 20, ties included per the market rules.", make_cut: "Chance of playing the weekend." };
const delta = (a: number | null | undefined, b: number | null | undefined) => a == null || b == null ? null : `${a - b >= 0 ? "+" : ""}${((a - b) * 100).toFixed(2)} pp`;

export function DistributionView() {
  const index = useDashboardData<{ events: DistributionEvent[] }>("golfprice/index.json");
  const [eventId, setEventId] = useState("");
  const events = orderEvents(index.data?.events ?? []) as DistributionEvent[];
  const event = events.find((e) => e.event_uid === eventId) ?? events[0];
  if (index.loading) return <LoadingState label="Loading simulation checkpoints" />;
  if (index.error || !event) return <EmptyState title="No validated simulations available" detail={index.error ?? "Run and publish an event to explore its distributions."} />;
  return <>
    <PageIntro eyebrow="Explore the simulation" title="Finish distributions" description="Compare players side by side, step back through earlier simulations, and see how the model prices their finishes and head-to-head matchups."
      controls={<label className="dist-select">Event<select aria-label="Event" value={event.event_uid} onChange={(e) => setEventId(e.target.value)}>{events.map((e) => <option key={e.event_uid} value={e.event_uid}>{e.tour?.toUpperCase()} · {e.name}</option>)}</select></label>} />
    <Explorer key={event.event_uid} event={event} />
  </>;
}

function Explorer({ event }: { event: DistributionEvent }) {
  const runs = checkpoints(event);
  const latest = runs.at(-1);
  const [runKey, setRunKey] = useState(latest?.key ?? event.explain_key ?? "");
  const [compare, setCompare] = useState("");
  const [ids, setIds] = useState<number[]>([]);
  if (!runKey) return <EmptyState title="No validated checkpoint published" detail="The next successful simulation will appear here." />;
  return <>
    <div className="dist-controls">
      <label className="dist-select">Simulation<select title="Pick which simulation run to look at: before the event or after a given round." aria-label="Simulation checkpoint" value={runKey} onChange={(e) => setRunKey(e.target.value)}>{runs.length ? runs.map((r) => <option key={r.key} value={r.key}>{runLabel(r)}{r.key === latest?.key ? " · Latest" : ""}</option>) : <option value={runKey}>Latest simulation</option>}</select></label>
      <label className="dist-select">Compare with<select title="Show an earlier or later simulation as lighter outlined bars next to the current one." aria-label="Compare shapes with" value={compare} onChange={(e) => setCompare(e.target.value)}><option value="">No comparison</option>{runs.filter((r) => r.key !== runKey).map((r) => <option key={r.key} value={r.key}>{runLabel(r)}</option>)}</select></label>
    </div>
    {runs.length > 1 && <nav className="dist-timeline" aria-label="Simulations for this event">{runs.map((r, i) => <button key={r.key} type="button" aria-current={r.key === runKey ? "step" : undefined} className={r.key === runKey ? "active" : ""} onClick={() => setRunKey(r.key)}><span>{i + 1}</span>{runLabel(r).split(" · ")[0]}<small>{runLabel(r).split(" · ")[1]}</small></button>)}</nav>}
    <Snapshot key={`${runKey}:${compare}`} runKey={runKey} compareKey={compare === runKey ? "" : compare} ids={ids} setIds={setIds} latest={runKey === latest?.key} />
  </>;
}

function Snapshot({ runKey, compareKey, ids, setIds, latest }: { runKey: string; compareKey: string; ids: number[]; setIds: (ids: number[]) => void; latest: boolean }) {
  const current = useDashboardData<unknown>(runKey);
  const previous = useDashboardData<unknown>(compareKey || runKey);
  const doc = useMemo(() => parseExplain(current.data) as DistributionDoc | null, [current.data]);
  const then = useMemo(() => compareKey ? parseExplain(previous.data) as DistributionDoc | null : null, [previous.data, compareKey]);
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("probability");
  const [finish, setFinish] = useState(10);
  if (current.loading) return <LoadingState label="Loading simulation" />;
  if (current.error || !doc) return <EmptyState title="This simulation is unavailable" detail="It could not be loaded. Try another simulation from the list above." />;
  const ranked = [...doc.players].filter((p) => !p.withdrawn).sort((a, b) => (b.probs.win.model ?? 0) - (a.probs.win.model ?? 0));
  const active = ids.length ? ids : ranked.slice(0, 3).map((p) => p.id);
  const players = active.map((id) => doc.players.find((p) => p.id === id)).filter((p): p is ExPlayer => !!p);
  const prior = then ? active.map((id) => then.players.find((p) => p.id === id)).filter((p): p is ExPlayer => !!p) : [];
  const hasCut = !!doc.event.cut_round || !!doc.event.cut_top_n;
  const fieldSize = doc.event.field_size ?? doc.players.length;
  const cumulative = mode === "cumulative";
  const rows = players.some((p) => p.finish?.pos?.length) ? bucketRows(players, prior, cumulative, fieldSize, hasCut) : [];
  const matches = query ? ranked.filter((p) => nameMatches(p.name, query) && !active.includes(p.id)).slice(0, 8) : [];
  const changePlayer = (id: number) => { setIds([...active.filter((p) => p !== id), id].slice(-4)); setQuery(""); };
  const remove = (id: number) => { if (active.length > 1) setIds(active.filter((p) => p !== id)); };
  const drawCount = doc.finish?.n_draws;
  return <div className="dist-explorer">
    <div className="dist-context"><span className="dist-status">{latest ? "Latest simulation" : "Earlier simulation"}</span><span>{doc.kind === "week" ? "Pre-event" : `After round ${doc.after_round}`} · {etTime(doc.as_of, "time unknown")}</span>{drawCount ? <span title="How many times the tournament was simulated to get these probabilities.">{drawCount.toLocaleString()} simulated tournaments</span> : null}</div>
    <Panel title="Compare players" eyebrow="Up to four on the same axes">
      <div className="dist-pickers"><div className="dist-player-chips">{players.map((p, i) => <span key={p.id} className="dist-chip" style={{ borderColor: COLORS[i] }}><i style={{ background: COLORS[i] }} />{name(p)}<button type="button" aria-label={`Remove ${name(p)}`} disabled={active.length <= 1} onClick={() => remove(p.id)}><X size={14} /></button></span>)}</div><div className="dist-search"><label><Search size={16} /><input aria-label="Find a player to overlay" placeholder={active.length >= 4 ? "Search to replace the first player…" : "Add a player…"} value={query} onChange={(e) => setQuery(e.target.value)} /></label>{query && <div className="dist-results" role="group" aria-label="Matching players">{matches.map((p) => <button key={p.id} type="button" onClick={() => changePlayer(p.id)}>{name(p)}<span>{pct(p.probs.win.model)} win</span></button>)}{!matches.length && <p>No other players match.</p>}</div>}</div></div>
    </Panel>
    <div className="dist-player-cards">{players.map((p, i) => { const old = prior.find((q) => q.id === p.id); return <article key={p.id} style={{ borderTopColor: COLORS[i] }}><span>{name(p)}</span><strong>{pct(p.probs.win.model)}<small> win</small></strong><div><b>{american(p.probs.win.model)}</b><span>{p.finish ? `Typical finish: ${p.finish.median}` : "No finish chart for this player"}</span></div>{old && <small className="dist-change">{delta(p.probs.win.model, old.probs.win.model)} win vs comparison</small>}</article>; })}</div>
    <Panel title="Where each player is likely to finish" eyebrow="Same simulation for every player">
      <div className="dist-chart-tools"><SegmentedControl label="Chart view" value={mode} onChange={setMode} options={[{ value: "probability", label: "Finish range" }, { value: "cumulative", label: "Range or better" }]} /><label className="dist-marker" title="Choose N to see each player's chance and fair odds of finishing in the top N, listed under the chart.">Show odds for top<input aria-label="Finish marker" type="number" min={1} max={doc.players.length} value={finish} onChange={(e) => setFinish(Math.max(1, Math.min(doc.players.length, Number(e.target.value) || 1)))} /></label></div>
      {!rows.length ? <EmptyState title="Finish chart unavailable" detail="The finish chart is not available for this simulation. The prices below are still valid." /> : <div className="dist-chart" role="img" aria-label={`${cumulative ? "Chance of finishing in each range or better" : "Chance of finishing in each range"} for ${players.map(name).join(", ")}`}><ResponsiveContainer width="100%" height="100%"><BarChart data={rows} barCategoryGap="14%" barGap={2} margin={{ top: 20, right: 18, left: 0, bottom: 18 }}><CartesianGrid vertical={false} stroke="var(--line)" /><XAxis dataKey="label" stroke="var(--muted)" tick={{ fontSize: 11 }} interval={0} label={{ value: "Finish position (ties shared)", position: "insideBottom", offset: -12, fill: "var(--muted)" }} /><YAxis unit="%" tickFormatter={(v) => Number(v).toFixed(0)} width={48} stroke="var(--muted)" tick={{ fontSize: 11 }} domain={cumulative ? [0, 100] : [0, "auto"]} /><Tooltip cursor={{ fill: "var(--tint-1)" }} contentStyle={{ background: "var(--popover)", border: "1px solid var(--line)", borderRadius: 10 }} formatter={(v) => `${Number(v).toFixed(1)}%`} labelFormatter={(v) => `${v === "Win" ? "Win" : v === "Missed cut" ? "Missed cut" : `Finish ${v}`}${cumulative && v !== "Win" ? " or better" : ""}`} />{players.flatMap((p, i) => [<Bar key={`now${p.id}`} name={name(p)} dataKey={`now_${p.id}`} fill={COLORS[i]} isAnimationActive={false} radius={[3, 3, 0, 0]} />, ...(prior.some((q) => q.id === p.id) ? [<Bar key={`then${p.id}`} name={`${name(p)} (comparison)`} dataKey={`then_${p.id}`} fill={COLORS[i]} fillOpacity={0.22} stroke={COLORS[i]} strokeWidth={1.5} isAnimationActive={false} radius={[3, 3, 0, 0]} />] : [])])}</BarChart></ResponsiveContainer></div>}
      <div className="dist-legend">{players.map((p, i) => <span key={p.id}><i style={{ background: COLORS[i] }} />{name(p)} · {markerLabel(finish, doc.event.cut_top_n)}: {pct(marker(p, finish, doc.event.cut_top_n))} / {american(marker(p, finish, doc.event.cut_top_n))}</span>)}{doc.event.cut_top_n && doc.event.cut_top_n !== finish ? players.map((p, i) => { const c = cutReference(p, doc.event.cut_top_n); return c ? <span key={`cut${p.id}`} data-testid="dist-cut-ref"><i style={{ background: COLORS[i] }} />{name(p)} · {c.label}: {pct(c.prob)} / {american(c.prob)}</span> : null; }) : null}{then && <span><Layers3 size={14} />Lighter outlined bars: the simulation you are comparing with</span>}</div>
      <p className="ex-muted">{FINISH_CHART_CAPTION}{doc.event.cut_top_n ? " Make cut is the model's priced chance of playing the weekend." : ""}</p>
      {compareKey && previous.loading && <p className="ex-muted">Loading the comparison simulation…</p>}{compareKey && previous.error && <p role="alert" className="ex-muted">The comparison simulation could not be loaded.</p>}
    </Panel>
    <Panel title="Finish prices" eyebrow="Probability and fair American odds">
      <div className="dist-table-scroll"><table className="ex-table dist-prices"><thead><tr><th scope="col">Player</th>{MARKETS.filter((m) => m !== "make_cut" || doc.event.cut_round).map((m) => <th scope="col" key={m} title={MARKET_HELP[m]}>{MARKET_LABEL[m]}</th>)}</tr></thead><tbody>{players.map((p, i) => <tr key={p.id}><th scope="row"><i className="ex-dot" style={{ background: COLORS[i] }} />{name(p)}</th>{MARKETS.filter((m) => m !== "make_cut" || doc.event.cut_round).map((m) => <td key={m}><strong>{pct(p.probs[m].model, 2)}</strong><small>{american(p.probs[m].model)}</small>{then && <em>{delta(p.probs[m].model, prior.find((q) => q.id === p.id)?.probs[m].model)}</em>}</td>)}</tr>)}</tbody></table></div>
      <p className="ex-muted">Model prices only, with no sportsbook odds blended in. Changes are percentage points against the simulation you are comparing with.</p>
    </Panel>
    <Panel title="Head to head" eyebrow="Row player beats column player · ties push">
      {players.length < 2 ? <p className="ex-muted">Add another player to compare matchups.</p> : <div className="dist-table-scroll"><table className="ex-table dist-prices"><thead><tr><th scope="col">Player <ArrowRight size={13} /></th>{players.map((p) => <th key={p.id} scope="col">{name(p)}</th>)}</tr></thead><tbody>{players.map((a, i) => <tr key={a.id}><th scope="row"><i className="ex-dot" style={{ background: COLORS[i] }} />{name(a)}</th>{players.map((b) => { const h = matchup(doc, a.id, b.id); const old = matchup(then, a.id, b.id); return <td key={b.id}>{a.id === b.id ? <span className="ex-muted">—</span> : h ? <><strong>{pct(h.p, 2)}</strong><small>{american(h.p)}</small>{h.tie != null && <small>{pct(h.tie, 1)} tie</small>}{old && <em>{delta(h.p, old.p)}</em>}</> : <span className="ex-muted">No price</span>}</td>; })}</tr>)}</tbody></table></div>}
      <p className="ex-muted">Matchups come from the same simulated tournaments, so they keep shared weather and tournament outcomes. They are not worked out by comparing the finish charts above.</p>
    </Panel>
  </div>;
}
