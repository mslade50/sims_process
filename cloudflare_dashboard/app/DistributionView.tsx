"use client";

import { useMemo, useState } from "react";
import { CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Search, X, Layers3, ArrowRight } from "lucide-react";
import { useDashboardData } from "./data";
import { EmptyState, LoadingState, SegmentedControl } from "./components";
import { OnThisPage, PageHeader, Section } from "./ui";
import { useEventParam } from "./event-param";
import { american, MARKETS, MARKET_LABEL, nameMatches, orderEvents, parseExplain, pct, type ExPlayer } from "./explain-rules";
import { FINISH_CHART_CAPTION, MISSED_CUT_LABEL, checkpoints, positionRows, cutReference, marker, markerLabel, matchup, runLabel, type DistributionDoc, type DistributionEvent } from "./distributions-rules";
import { playerDisplayName } from "./player-name";
import type { PositionChartSeries } from "./ExplainCharts";
import { ShapeExplorer } from "./ShapeViews";
import "./explain.css";
import "./distributions.css";

const COLORS = ["var(--chart-1)", "var(--chart-3)", "var(--chart-2)", "var(--chart-4)"];
const name = (p: ExPlayer) => playerDisplayName(p.name);
const MARKET_HELP: Record<string, string> = { win: "Chance of winning outright.", top_5: "Chance of finishing in the top 5, counting ties (a player tied for 5th counts).", top_10: "Chance of finishing in the top 10, counting ties (a player tied for 10th counts).", top_20: "Chance of finishing in the top 20, counting ties (a player tied for 20th counts).", make_cut: "Chance of playing the weekend." };
const delta = (a: number | null | undefined, b: number | null | undefined) => a == null || b == null ? null : `${a - b >= 0 ? "+" : ""}${((a - b) * 100).toFixed(2)} pp`;

const DASHES = ["", "7 3", "2 3", "9 3 2 3"];
const lastName = (n: string) => n.split(" ").slice(-1)[0];

/** Finish-position curves, one line per player with the name written on the line itself (no legend lookup), a distinct dash per player so colour is never the only signal. */
function PositionChartLabeled({ series, hasCut, cumulative, height, ariaLabel }: { series: PositionChartSeries[]; hasCut: boolean; cumulative: boolean; height: number; ariaLabel: string }) {
  const rows = positionRows(series.map((s) => ({ key: s.key, pos: s.pos, missCut: s.missCut })), cumulative, hasCut).filter((r) => r.label !== MISSED_CUT_LABEL);
  const n = rows.length;
  const main = series.filter((s) => s.style !== "dashed");
  const anchor = (s: PositionChartSeries) => {
    let best = 0, bestV = -1;
    rows.forEach((r, i) => { const v = Number(r[s.key] ?? 0); if (cumulative ? (bestV < 0 && v >= 50) : v > bestV) { best = i; bestV = v; } });
    return cumulative && bestV < 0 ? n - 1 : best;
  };
  const ticks = rows.map((r) => String(r.label)).filter((l, i) => i === 0 || Number(l) % 10 === 0);
  return <div className="dist-chart" role="img" aria-label={ariaLabel} style={{ height }}>
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={rows} margin={{ top: 22, right: 12, bottom: 18, left: 0 }}>
        <CartesianGrid stroke="var(--line)" vertical={false} />
        <XAxis dataKey="label" ticks={ticks} interval={0} stroke="var(--muted)" tick={{ fontSize: 12 }} label={{ value: "Finish position (ties shared)", position: "insideBottom", offset: -10, fill: "var(--muted)", fontSize: 12 }} />
        <YAxis stroke="var(--muted)" width={42} tick={{ fontSize: 12 }} domain={cumulative ? [0, 100] : [0, "auto"]} tickFormatter={(v: number) => `${Math.round(v)}%`} />
        <Tooltip cursor={{ stroke: "var(--line-strong)" }} content={<DistTooltip series={series} cumulative={cumulative} />} />
        {series.map((s) => {
          const idx = main.indexOf(s);
          const ai = idx >= 0 ? anchor(s) : -1;
          return <Line key={s.key} dataKey={s.key} name={s.name} type={cumulative ? "monotone" : "stepAfter"} stroke={s.color} strokeWidth={idx === 0 ? 2.5 : 2} strokeDasharray={s.style === "dashed" ? "2 3" : DASHES[Math.max(COLORS.indexOf(s.color), 0) % DASHES.length] || undefined} strokeOpacity={s.style === "dashed" ? 0.7 : 1} dot={false} activeDot={{ r: 4 }} isAnimationActive={false} connectNulls={false}
            label={idx >= 0 ? ((p: { x?: number | string; y?: number | string; index?: number }) => p.index === ai && typeof p.x === "number" && typeof p.y === "number" ? <text key={s.key} x={p.x + (ai > n * 0.6 ? -6 : 6)} y={p.y - 8 - idx * 17} textAnchor={ai > n * 0.6 ? "end" : "start"} fontSize={12} fontWeight={700} fill={s.color} stroke="var(--surface)" strokeWidth={3} paintOrder="stroke">{lastName(s.name)}</text> : <g key={`${s.key}${p.index}`} />) : undefined} />;
        })}
      </ComposedChart>
    </ResponsiveContainer>
  </div>;
}

function DistTooltip({ active, payload, label, series, cumulative }: { active?: boolean; payload?: Array<{ payload: Record<string, number | string> }>; label?: string; series: PositionChartSeries[]; cumulative: boolean }) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return <div className="dist-tip"><strong>{label === "1" ? "Win (finish 1)" : `Finish ${label}${cumulative ? " or better" : ""}`}</strong>{series.map((s) => { const v = row[s.key]; return <span key={s.key}><i style={{ background: s.color }} />{s.name}: <b>{typeof v === "number" ? `${v.toFixed(v < 1 ? 2 : 1)}%` : "n/a"}</b></span>; })}</div>;
}

export function DistributionView() {
  const index = useDashboardData<{ events: DistributionEvent[] }>("golfprice/index.json");
  const eventId = useEventParam();
  const events = orderEvents(index.data?.events ?? []) as DistributionEvent[];
  const event = events.find((e) => e.event_uid === eventId) ?? events[0];
  if (index.loading) return <LoadingState label="Loading simulation checkpoints" />;
  if (index.error || !event) return <EmptyState title="No validated simulations available" detail={index.error ?? "Run and publish an event to explore its distributions."} nextCheck="Checked after each published simulation" />;
  return <>
    <PageHeader title="Finish distributions" purpose="Where each player is likely to finish. Up to four players on one set of axes." />
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
      const setup = <details className="dist-setup"><summary>Simulation: {(() => { const r = runs.find((x) => x.key === runKey); return r ? runLabel(r) : "Latest simulation"; })()}{compare ? " · comparing with another run" : ""}<span> Change</span></summary>
    <div className="dist-controls" role="group" aria-label="Simulation to show">
      <label className="dist-select"><span>Simulation</span><select title="Pick which run to look at: before the event, or after a given round." aria-label="Simulation checkpoint" value={runKey} onChange={(e) => setRunKey(e.target.value)}>{runs.length ? runs.map((r) => <option key={r.key} value={r.key}>{runLabel(r)}{r.key === latest?.key ? " · Latest" : ""}</option>) : <option value={runKey}>Latest simulation</option>}</select></label>
      <label className="dist-select"><span>Compare with</span><select title="Show another run as faint dotted lines over the current one." aria-label="Compare shapes with" value={compare} onChange={(e) => setCompare(e.target.value)}><option value="">No comparison</option>{runs.filter((r) => r.key !== runKey).map((r) => <option key={r.key} value={r.key}>{runLabel(r)}</option>)}</select></label>
    </div></details>;
  return <>
    <Snapshot key={`${runKey}:${compare}`} setup={setup} runKey={runKey} compareKey={compare === runKey ? "" : compare} compareLabel={(() => { const r = runs.find((x) => x.key === compare); return r ? runLabel(r) : ""; })()} onStopCompare={() => setCompare("")} ids={ids} setIds={setIds} latest={runKey === latest?.key} />
  </>;
}

function Snapshot({ setup, runKey, compareKey, compareLabel, onStopCompare, ids, setIds, latest }: { setup: React.ReactNode; runKey: string; compareKey: string; compareLabel: string; onStopCompare: () => void; ids: number[]; setIds: (ids: number[]) => void; latest: boolean }) {
  const current = useDashboardData<unknown>(runKey);
  const previous = useDashboardData<unknown>(compareKey || runKey);
  const doc = useMemo(() => parseExplain(current.data) as DistributionDoc | null, [current.data]);
  const then = useMemo(() => compareKey ? parseExplain(previous.data) as DistributionDoc | null : null, [previous.data, compareKey]);
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("cumulative");
  const [finish, setFinish] = useState(10);
  if (current.loading) return <LoadingState label="Loading simulation" />;
  if (current.error || !doc) return <EmptyState title="This simulation is unavailable" detail="It could not be loaded. Try another simulation from the list above." />;
  const ranked = [...doc.players].filter((p) => !p.withdrawn).sort((a, b) => (b.probs.win.model ?? 0) - (a.probs.win.model ?? 0));
  // Default comparison: the favourite, the biggest model-vs-book win gap among the top 25, and the runner-up.
  const gapPick = ranked.slice(1, 25).filter((p) => p.probs.win.market != null && p.probs.win.model != null).sort((x, y) => Math.abs((y.probs.win.model ?? 0) - (y.probs.win.market ?? 0)) - Math.abs((x.probs.win.model ?? 0) - (x.probs.win.market ?? 0)))[0];
  const defaults = [ranked[0], gapPick, ranked[1]].filter((p, i, arr): p is ExPlayer => !!p && arr.indexOf(p) === i).concat(ranked.slice(0, 3)).filter((p, i, arr) => arr.indexOf(p) === i).slice(0, 3);
  const active = ids.length ? ids : defaults.map((p) => p.id);
  const players = active.map((id) => doc.players.find((p) => p.id === id)).filter((p): p is ExPlayer => !!p);
  const prior = then ? active.map((id) => then.players.find((p) => p.id === id)).filter((p): p is ExPlayer => !!p) : [];
  const hasCut = !!doc.event.cut_round || !!doc.event.cut_top_n;
  const cumulative = mode === "cumulative";
  const chartSeries: PositionChartSeries[] = [
    ...players.filter((p) => p.finish?.pos?.length).map((p) => ({ key: `now_${p.id}`, name: name(p), color: COLORS[players.indexOf(p)], pos: p.finish!.pos, missCut: hasCut ? p.finish!.p_miss_cut : null, style: "line" as "bar" | "line" })),
    ...prior.filter((p) => p.finish?.pos?.length).map((p) => ({ key: `then_${p.id}`, name: `${name(p)} (comparison)`, color: COLORS[active.indexOf(p.id)], pos: p.finish!.pos, missCut: hasCut ? p.finish!.p_miss_cut : null, style: "dashed" as const })),
  ];
  const rows = chartSeries.length ? chartSeries : [];
  const matches = query ? ranked.filter((p) => nameMatches(p.name, query) && !active.includes(p.id)).slice(0, 8) : [];
  const changePlayer = (id: number) => { setIds([...active.filter((p) => p !== id), id].slice(-4)); setQuery(""); };
  const remove = (id: number) => { if (active.length > 1) setIds(active.filter((p) => p !== id)); };
  const drawCount = doc.finish?.n_draws;
  const nav = [{ id: "compare", label: "Compare" }, { id: "h2h", label: "Head to head" }, { id: "shapes", label: "Why shapes differ" }];
  const jump = () => { const el = document.getElementById("compare"); if (el) window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - 70, behavior: "smooth" }); };
  return <div className="dist-explorer">
    <Section id="compare" title="Compare players">
      <div className="dist-chart-tools"><span className="dist-chart-title">Chance of finishing {cumulative ? "this position or better" : "in each position"}</span><SegmentedControl label="Chart view" value={mode} onChange={setMode} options={[{ value: "probability", label: "Each position" }, { value: "cumulative", label: "This or better" }]} /><label className="dist-marker" title="Choose N to see each player's chance and fair odds of finishing in the top N, listed under the chart.">Odds for top<input aria-label="Finish marker" type="number" min={1} max={doc.players.length} value={finish} onChange={(e) => setFinish(Math.max(1, Math.min(doc.players.length, Number(e.target.value) || 1)))} /></label></div>
      {!rows.length ? <EmptyState title="Finish chart unavailable" detail="The finish chart is not available for this simulation. The prices below are still valid." /> : <PositionChartLabeled series={chartSeries} hasCut={hasCut} cumulative={cumulative} height={320} ariaLabel={`${cumulative ? "Chance of finishing at each position or better" : "Chance of finishing at each position"} for ${players.map(name).join(", ")}`} />}
      <ul className="dist-legend" aria-label="Chart key">{players.map((p, i) => <li key={p.id}><svg width="26" height="8" aria-hidden="true"><line x1="0" y1="4" x2="26" y2="4" stroke={COLORS[i]} strokeWidth="2.5" strokeDasharray={DASHES[i % DASHES.length] || undefined} /></svg><span><b>{name(p)}</b> · {markerLabel(finish, doc.event.cut_top_n)} {pct(marker(p, finish, doc.event.cut_top_n))} <em>{american(marker(p, finish, doc.event.cut_top_n))}</em></span></li>)}{doc.event.cut_top_n && doc.event.cut_top_n !== finish ? players.map((p, i) => { const c = cutReference(p, doc.event.cut_top_n); return c ? <li key={`cut${p.id}`} data-testid="dist-cut-ref"><i style={{ background: COLORS[i] }} /><span>{name(p)} · {c.label} {pct(c.prob)} <em>{american(c.prob)}</em></span></li> : null; }) : null}{then && <li><Layers3 size={14} /><span>Faint dotted lines: the simulation you are comparing with</span></li>}</ul>
      <p className="ex-muted">{FINISH_CHART_CAPTION}{drawCount ? ` Based on ${drawCount.toLocaleString()} simulated tournaments.` : ""}{!latest ? " This is an earlier simulation, not the latest." : ""}{doc.event.cut_top_n ? " Make cut is the model's priced chance of playing the weekend." : ""}</p>
      {compareKey && previous.loading && <p className="ex-muted">Loading the comparison simulation…</p>}{compareKey && previous.error && <p role="alert" className="ex-muted">{`The comparison run${compareLabel ? ` (${compareLabel})` : ""} could not be loaded, so only the current run is shown. Choose another run in Compare with, or `}<button type="button" className="dist-linkbtn" onClick={() => onStopCompare()}>stop comparing</button>.</p>}
      <div className="dist-pickers"><div className="dist-player-chips">{players.map((p, i) => <span key={p.id} className="dist-chip" style={{ borderColor: COLORS[i] }}><i style={{ background: COLORS[i] }} />{name(p)}<button type="button" aria-label={`Remove ${name(p)}`} disabled={active.length <= 1} onClick={() => remove(p.id)}><X size={14} /></button></span>)}</div><div className="dist-search"><label><Search size={16} /><input aria-label="Find a player to overlay" placeholder={active.length >= 4 ? "Search to replace the first player…" : "Add a player…"} value={query} onChange={(e) => setQuery(e.target.value)} /></label>{query && <div className="dist-results" role="group" aria-label="Matching players">{matches.map((p) => <button key={p.id} type="button" onClick={() => changePlayer(p.id)}>{name(p)}<span>{pct(p.probs.win.model)} win</span></button>)}{!matches.length && <p>No other players match.</p>}</div>}</div></div>
      <p className="dist-table-title">Probability and fair odds, same simulations as the chart (model only, no sportsbook odds blended in)</p><div className="dist-table-scroll"><table className="ex-table dist-prices"><thead><tr><th scope="col">Player</th>{MARKETS.filter((m) => m !== "make_cut" || doc.event.cut_round).map((m) => <th scope="col" key={m} title={MARKET_HELP[m]}>{MARKET_LABEL[m]}</th>)}<th scope="col" title="The finish position that half of the simulations beat and half did not. It is not the same as the average finish.">Median finish</th></tr></thead><tbody>{players.map((p, i) => <tr key={p.id}><th scope="row"><i className="ex-dot" style={{ background: COLORS[i] }} />{name(p)}</th>{MARKETS.filter((m) => m !== "make_cut" || doc.event.cut_round).map((m) => <td key={m}><strong>{pct(p.probs[m].model, 2)}</strong><small>{american(p.probs[m].model)}</small>{then && <em>{delta(p.probs[m].model, prior.find((q) => q.id === p.id)?.probs[m].model)}</em>}</td>)}<td><strong>{p.finish ? p.finish.median : "n/a"}</strong></td></tr>)}</tbody></table></div>
      {then && <p className="ex-muted">Small italic figures are the change in percentage points against the run you are comparing with.</p>}
    </Section>
    <OnThisPage sections={nav} />
    {setup}
    <Section id="h2h" title="Head to head" purpose="Chance the row player finishes ahead of the column player.">
      {players.length < 2 ? <p className="ex-muted">Add another player to compare matchups.</p> : <div className="dist-table-scroll"><table className="ex-table dist-prices"><thead><tr><th scope="col">Player <ArrowRight size={13} /></th>{players.map((p) => <th key={p.id} scope="col">{name(p)}</th>)}</tr></thead><tbody>{players.map((a, i) => <tr key={a.id}><th scope="row"><i className="ex-dot" style={{ background: COLORS[i] }} />{name(a)}</th>{players.map((b) => { const h = matchup(doc, a.id, b.id); const old = matchup(then, a.id, b.id); return <td key={b.id}>{a.id === b.id ? <span className="ex-muted">—</span> : h ? <><strong>{pct(h.p, 2)}</strong><small>{american(h.p)}</small>{h.tie != null && <small title="How often the two finished level in the simulations.">{pct(h.tie, 1)} finish level</small>}{old && <em>{delta(h.p, old.p)}</em>}</> : <span className="ex-muted">No price</span>}</td>; })}</tr>)}</tbody></table></div>}
      <p className="ex-muted">Each pair&apos;s win chances add to 100% because a tie is a push (no bet). The finish-level figure shows how often the two tied.</p>
      <details className="ex-glossary-wrap"><summary>Technical details</summary><p className="ex-muted">Matchups come from the same simulated tournaments, so they keep shared weather and tournament outcomes. They are not worked out by comparing the finish charts above.</p></details>
    </Section>
    <Section id="shapes" title="Why the same win chance can mean different finishes" purpose="Steady players bank top-20s; boom-or-bust players win more from rare blow-out weeks. Pick a pair to load them into the chart above.">
      <ShapeExplorer doc={doc} onPick={(picked) => { setIds(picked.slice(-4)); setQuery(""); jump(); }} />
    </Section>
  </div>;
}
