"use client";

import { useMemo, useState } from "react";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Search, X, Layers3, ArrowRight } from "lucide-react";
import { useDashboardData } from "./data";
import { EmptyState, LoadingState, PageIntro, Panel, SegmentedControl } from "./components";
import { american, MARKETS, MARKET_LABEL, nameMatches, orderEvents, parseExplain, pct, type ExPlayer } from "./explain-rules";
import { SETTLEMENT_RANK_LABEL, checkpoints, curveRows, cutReference, marker, markerLabel, matchup, runLabel, type DistributionDoc, type DistributionEvent } from "./distributions-rules";
import "./explain.css";
import "./distributions.css";

const COLORS = ["var(--chart-1)", "var(--chart-3)", "var(--chart-2)", "var(--chart-4)"];
const name = (p: ExPlayer) => p.name.includes(",") ? p.name.split(",").reverse().join(" ").trim() : p.name;
const delta = (a: number | null | undefined, b: number | null | undefined) => a == null || b == null ? null : `${a - b >= 0 ? "+" : ""}${((a - b) * 100).toFixed(2)} pp`;

export function DistributionView() {
  const index = useDashboardData<{ events: DistributionEvent[] }>("golfprice/index.json");
  const [eventId, setEventId] = useState("");
  const events = orderEvents(index.data?.events ?? []) as DistributionEvent[];
  const event = events.find((e) => e.event_uid === eventId) ?? events[0];
  if (index.loading) return <LoadingState label="Loading simulation checkpoints" />;
  if (index.error || !event) return <EmptyState title="No validated simulations available" detail={index.error ?? "Run and publish an event to explore its distributions."} />;
  return <>
    <PageIntro eyebrow="Explore the simulation" title="Finish distributions" description="Overlay players, revisit each checkpoint, and see how the same simulation prices their finishes and matchups."
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
      <label className="dist-select">Simulation checkpoint<select aria-label="Simulation checkpoint" value={runKey} onChange={(e) => setRunKey(e.target.value)}>{runs.length ? runs.map((r) => <option key={r.key} value={r.key}>{runLabel(r)}{r.key === latest?.key ? " · Latest" : ""}</option>) : <option value={runKey}>Latest validated run</option>}</select></label>
      <label className="dist-select">Compare shapes with<select aria-label="Compare shapes with" value={compare} onChange={(e) => setCompare(e.target.value)}><option value="">No comparison</option>{runs.filter((r) => r.key !== runKey).map((r) => <option key={r.key} value={r.key}>{runLabel(r)}</option>)}</select></label>
    </div>
    {runs.length > 1 && <nav className="dist-timeline" aria-label="Event checkpoints">{runs.map((r, i) => <button key={r.key} type="button" aria-current={r.key === runKey ? "step" : undefined} className={r.key === runKey ? "active" : ""} onClick={() => setRunKey(r.key)}><span>{i + 1}</span>{runLabel(r).split(" · ")[0]}<small>{runLabel(r).split(" · ")[1]}</small></button>)}</nav>}
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
  if (current.loading) return <LoadingState label="Loading validated simulation" />;
  if (current.error || !doc) return <EmptyState title="This checkpoint is unavailable" detail={current.error ?? "Publish its distribution document and retry."} />;
  const ranked = [...doc.players].filter((p) => !p.withdrawn).sort((a, b) => (b.probs.win.model ?? 0) - (a.probs.win.model ?? 0));
  const active = ids.length ? ids : ranked.slice(0, 3).map((p) => p.id);
  const players = active.map((id) => doc.players.find((p) => p.id === id)).filter((p): p is ExPlayer => !!p);
  const prior = then ? active.map((id) => then.players.find((p) => p.id === id)).filter((p): p is ExPlayer => !!p) : [];
  const rows = curveRows(players, prior, mode === "cumulative");
  const matches = query ? ranked.filter((p) => nameMatches(p.name, query) && !active.includes(p.id)).slice(0, 8) : [];
  const changePlayer = (id: number) => { setIds([...active.filter((p) => p !== id), id].slice(-4)); setQuery(""); };
  const remove = (id: number) => { if (active.length > 1) setIds(active.filter((p) => p !== id)); };
  const drawCount = doc.finish?.n_draws;
  return <div className="dist-explorer">
    <div className="dist-context"><span className="dist-status">{latest ? "Latest validated run" : "Validated checkpoint"}</span><span>{doc.kind === "week" ? "Pre-event" : `After round ${doc.after_round}`} · {doc.as_of ? new Date(doc.as_of).toLocaleString() : doc.run}</span><span>{drawCount ? `${drawCount.toLocaleString()} draws` : "Draws unavailable"}</span><span>Model probabilities</span></div>
    <Panel title="Compare players" eyebrow="Up to four on the same axes">
      <div className="dist-pickers"><div className="dist-player-chips">{players.map((p, i) => <span key={p.id} className="dist-chip" style={{ borderColor: COLORS[i] }}><i style={{ background: COLORS[i] }} />{name(p)}<button type="button" aria-label={`Remove ${name(p)}`} disabled={active.length <= 1} onClick={() => remove(p.id)}><X size={14} /></button></span>)}</div><div className="dist-search"><label><Search size={16} /><input aria-label="Find a player to overlay" placeholder={active.length >= 4 ? "Search to replace the first player…" : "Add a player…"} value={query} onChange={(e) => setQuery(e.target.value)} /></label>{query && <div className="dist-results" role="group" aria-label="Matching players">{matches.map((p) => <button key={p.id} type="button" onClick={() => changePlayer(p.id)}>{name(p)}<span>{pct(p.probs.win.model)} win</span></button>)}{!matches.length && <p>No other players match.</p>}</div>}</div></div>
    </Panel>
    <div className="dist-player-cards">{players.map((p, i) => { const old = prior.find((q) => q.id === p.id); return <article key={p.id} style={{ borderTopColor: COLORS[i] }}><span>{name(p)}</span><strong>{pct(p.probs.win.model)}<small> win</small></strong><div><b>{american(p.probs.win.model)}</b><span>{p.finish ? `Median finish ${p.finish.median}` : "Finish curve unavailable"}</span></div>{old && <small className="dist-change">{delta(p.probs.win.model, old.probs.win.model)} win vs comparison</small>}</article>; })}</div>
    <Panel title="The shape of the event" eyebrow="Same players · same simulation">
      <div className="dist-chart-tools"><SegmentedControl label="Curve mode" value={mode} onChange={setMode} options={[{ value: "probability", label: "Finish position" }, { value: "cumulative", label: "Position or better" }]} /><label className="dist-marker">Finish marker<input aria-label="Finish marker" type="number" min={1} max={doc.players.length} value={finish} onChange={(e) => setFinish(Math.max(1, Math.min(doc.players.length, Number(e.target.value) || 1)))} /></label></div>
      {!rows.length ? <EmptyState title="Exact finish draws unavailable" detail={doc.finish?.reason ?? "Finish prices below remain available from the validated simulation."} /> : <div className="dist-chart" role="img" aria-label={`${mode === "cumulative" ? "Cumulative" : "Finish-position"} distribution overlay for ${players.map(name).join(", ")}`}><ResponsiveContainer width="100%" height="100%"><LineChart data={rows} margin={{ top: 20, right: 18, left: 0, bottom: 18 }}><CartesianGrid vertical={false} stroke="var(--line)" /><XAxis dataKey="position" type="number" domain={[1, Math.max(doc.players.length, rows.length)]} stroke="var(--muted)" tick={{ fontSize: 11 }} label={{ value: "Settlement rank", position: "insideBottom", offset: -12, fill: "var(--muted)" }} /><YAxis unit="%" tickFormatter={(v) => Number(v).toFixed(mode === "cumulative" ? 0 : 1)} width={48} stroke="var(--muted)" tick={{ fontSize: 11 }} domain={mode === "cumulative" ? [0, 100] : [0, "auto"]} /><Tooltip contentStyle={{ background: "var(--popover)", border: "1px solid var(--line)", borderRadius: 10 }} formatter={(v) => `${Number(v).toFixed(2)}%`} labelFormatter={(v) => `Settlement rank ${v}${mode === "cumulative" ? " or better" : ""}`} />{doc.event.cut_top_n && doc.event.cut_top_n !== finish ? <ReferenceLine x={doc.event.cut_top_n} stroke="var(--accent)" strokeDasharray="2 3" label={{ value: markerLabel(doc.event.cut_top_n, doc.event.cut_top_n), fill: "var(--accent)", position: "insideTopLeft", fontSize: 11 }} /> : null}<ReferenceLine x={finish} stroke="var(--muted)" strokeDasharray="3 4" label={{ value: markerLabel(finish, doc.event.cut_top_n), fill: "var(--muted)", position: "insideTopRight", fontSize: 11 }} />{players.map((p, i) => <Line key={`now${p.id}`} name={name(p)} dataKey={`now_${p.id}`} stroke={COLORS[i]} strokeWidth={2.5} dot={false} isAnimationActive={false} />)}{prior.map((p) => <Line key={`then${p.id}`} name={`${name(p)} · comparison`} dataKey={`then_${p.id}`} stroke={COLORS[active.indexOf(p.id)]} strokeWidth={1.8} strokeDasharray="6 4" strokeOpacity={0.65} dot={false} isAnimationActive={false} />)}</LineChart></ResponsiveContainer></div>}
      <div className="dist-legend">{players.map((p, i) => <span key={p.id}><i style={{ background: COLORS[i] }} />{name(p)} · {markerLabel(finish, doc.event.cut_top_n)}: {pct(marker(p, finish, doc.event.cut_top_n))} / {american(marker(p, finish, doc.event.cut_top_n))}</span>)}{doc.event.cut_top_n && doc.event.cut_top_n !== finish ? players.map((p, i) => { const c = cutReference(p, doc.event.cut_top_n); return c ? <span key={`cut${p.id}`} data-testid="dist-cut-ref"><i style={{ background: COLORS[i] }} />{name(p)} · {c.label}: {pct(c.prob)} / {american(c.prob)}</span> : null; }) : null}{then && <span><Layers3 size={14} />Dashed curves: comparison checkpoint</span>}</div>
      <p className="ex-muted">Curves show {SETTLEMENT_RANK_LABEL}; they share tied positions across players. The cut-line marker is the priced make-cut probability. Win and top-5/10/20 prices below use the simulation’s exact market settlement rules; other markers read the rank curve.</p>
      {compareKey && previous.loading && <p className="ex-muted">Loading comparison checkpoint…</p>}{compareKey && previous.error && <p role="alert" className="ex-muted">Comparison unavailable: {previous.error}</p>}
    </Panel>
    <Panel title="Finish prices" eyebrow="Probability and fair American odds">
      <div className="dist-table-scroll"><table className="ex-table dist-prices"><thead><tr><th scope="col">Player</th>{MARKETS.filter((m) => m !== "make_cut" || doc.event.cut_round).map((m) => <th scope="col" key={m}>{MARKET_LABEL[m]}</th>)}</tr></thead><tbody>{players.map((p, i) => <tr key={p.id}><th scope="row"><i className="ex-dot" style={{ background: COLORS[i] }} />{name(p)}</th>{MARKETS.filter((m) => m !== "make_cut" || doc.event.cut_round).map((m) => <td key={m}><strong>{pct(p.probs[m].model, 2)}</strong><small>{american(p.probs[m].model)}</small>{then && <em>{delta(p.probs[m].model, prior.find((q) => q.id === p.id)?.probs[m].model)}</em>}</td>)}</tr>)}</tbody></table></div>
      <p className="ex-muted">Raw model prices, with no bookmaker consensus or market regression. Changes are percentage points against the selected comparison.</p>
    </Panel>
    <Panel title="Head to head" eyebrow="Row player beats column player · ties push">
      {players.length < 2 ? <p className="ex-muted">Add another player to compare matchups.</p> : <div className="dist-table-scroll"><table className="ex-table dist-prices"><thead><tr><th scope="col">Player <ArrowRight size={13} /></th>{players.map((p) => <th key={p.id} scope="col">{name(p)}</th>)}</tr></thead><tbody>{players.map((a, i) => <tr key={a.id}><th scope="row"><i className="ex-dot" style={{ background: COLORS[i] }} />{name(a)}</th>{players.map((b) => { const h = matchup(doc, a.id, b.id); const old = matchup(then, a.id, b.id); return <td key={b.id}>{a.id === b.id ? <span className="ex-muted">—</span> : h ? <><strong>{pct(h.p, 2)}</strong><small>{american(h.p)}</small>{h.tie != null && <small>{pct(h.tie, 1)} tie</small>}{old && <em>{delta(h.p, old.p)}</em>}</> : <span className="ex-muted">Unavailable</span>}</td>; })}</tr>)}</tbody></table></div>}
      <p className="ex-muted">Matchups come from the joint simulation, preserving shared weather and tournament outcomes. They are not estimated by comparing the separate finish curves.</p>
    </Panel>
  </div>;
}
