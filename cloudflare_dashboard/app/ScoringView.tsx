"use client";

import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChevronRight, CircleHelp, Info, Search, X } from "lucide-react";
import Link from "next/link";
import { useDashboardData } from "./data";
import { etTime } from "./lib";
import { EmptyState, LoadingState, PageIntro, Panel } from "./components";
import { orderEvents } from "./explain-rules";
import { checkpoints, runLabel, type DistributionEvent } from "./distributions-rules";
import { betEv, breakEvenDecimal, curveRows, decimalToAmerican, evBreakEvenShift, expectationBridge, marketNoVig, normalizePmf, overUnder, parseScoring, fieldMixturePmf, pmfMean, weatherBaselineText, WEATHER_HISTORY_CAVEAT, parseWeatherExtras, waveComponentMeans, waveGapText, windSensitivityText, type WeatherExtras, plainDate, prettyName, resolvedOverShare, shiftedOverUnder, scoringEvidence, plainText, plainTeeTime, type EvidenceContext, type ScoringDoc, type ScoringPlayer } from "./scoring-rules";
import { COMPONENT_LABELS, parseWeather, runMoment, signedFixed, type WeatherDoc } from "./weather-rules";
import { define } from "./glossary";
import { CourseHistoryPanel } from "./CourseHistoryPanel";
import "./scoring.css";

const COLORS = ["var(--chart-1)", "var(--chart-3)", "var(--chart-2)", "var(--chart-4)"];
const fmt = (n: number | null | undefined, d = 2) => n == null || !Number.isFinite(n) ? "—" : n.toFixed(d);
const pct = (n: number | null | undefined, d = 1) => n == null || !Number.isFinite(n) ? "—" : `${(n * 100).toFixed(d)}%`;
const relative = (n: number | null | undefined, digits = 2) => { if (n == null) return "Unavailable"; const t = n.toFixed(digits); const zero = Number(t) === 0; return `${!zero && n > 0 ? "+" : ""}${zero ? t.replace("-", "") : t} strokes`; };
const defn = (key: string, fallback: string) => define(key) || fallback;

export function ScoringView() {
  const index = useDashboardData<{ events: DistributionEvent[] }>("golfprice/index.json");
  const events = orderEvents(index.data?.events ?? []) as DistributionEvent[];
  const [eventId, setEventId] = useState("");
  const event = events.find((e) => e.event_uid === eventId) ?? events[0];
  if (index.loading) return <LoadingState label="Loading scoring checkpoints" />;
  if (index.error || !event) return <EmptyState title="No simulations available" detail={index.error ?? "Scoring expectations appear after the next simulation is published."} />;
  return <>
    <PageIntro eyebrow="Round scoring model" title="What score should we expect?" description="See the expected round score, how widely it could vary, what the course difficulty is based on, and whether completed rounds changed it."
      controls={<label className="score-select">Event<select aria-label="Event" value={event.event_uid} onChange={(e) => setEventId(e.target.value)}>{events.map((e) => <option key={e.event_uid} value={e.event_uid}>{e.tour?.toUpperCase()} · {e.name}</option>)}</select></label>} />
    <ScoringExplorer key={event.event_uid} event={event} />
  </>;
}

function ScoringExplorer({ event }: { event: DistributionEvent }) {
  const weatherKey = (event as DistributionEvent & { weather_key?: string | null }).weather_key || `golfprice/weather/${event.event_uid.replace(/:/g, "_")}/latest.json`;
  const runs = checkpoints(event);
  const latest = runs.at(-1);
  const [runKey, setRunKey] = useState(latest?.key ?? event.explain_key ?? "");
  if (!runKey) return <EmptyState title="No scoring run yet" detail="A scoring expectation will appear after the next successful simulation is published." />;
  const selectedRun = runs.find((r) => r.key === runKey);
  return <>
    <div className="score-controls"><label className="score-select">Simulation run<select aria-label="Simulation run" value={runKey} onChange={(e) => setRunKey(e.target.value)}>{runs.length ? runs.map((r) => <option key={r.key} value={r.key}>{runLabel(r)}{r.key === latest?.key ? " · Latest" : ""}</option>) : <option value={runKey}>Latest run</option>}</select></label></div>
    {runs.length > 1 && <nav className="score-timeline" aria-label="Simulation runs">{runs.map((r, i) => <button key={r.key} type="button" aria-current={r.key === runKey ? "step" : undefined} className={r.key === runKey ? "active" : ""} onClick={() => setRunKey(r.key)}><span>{i + 1}</span>{runLabel(r).split(" · ")[0]}<small>{runLabel(r).split(" · ")[1]}</small></button>)}</nav>}
    <Snapshot key={runKey} runKey={runKey} expectedRun={selectedRun?.run ?? null} latest={runKey === latest?.key} weatherKey={weatherKey} historyKey={(event as DistributionEvent & { history_key?: string | null }).history_key ?? null} />
  </>;
}

export function Snapshot({ runKey, expectedRun, latest, weatherKey, historyKey }: { runKey: string; expectedRun: string | null; latest: boolean; weatherKey?: string; historyKey?: string | null }) {
  const source = useDashboardData<unknown>(runKey);
  const weatherSource = useDashboardData<unknown>(weatherKey ?? "golfprice/none.json");
  const weatherDoc = useMemo(() => parseWeather(weatherSource.data), [weatherSource.data]);
  const weatherExtras = useMemo(() => parseWeatherExtras(weatherSource.data), [weatherSource.data]);
  const scoring = useMemo(() => {
    if (!source.data || typeof source.data !== "object") return null;
    return parseScoring((source.data as Record<string, unknown>).scoring);
  }, [source.data]);
  const [query, setQuery] = useState("");
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [fieldSelected, setFieldSelected] = useState(true);
  const [curveMode, setCurveMode] = useState<"pmf" | "cdf">("pmf");
  const [scenarioShift, setScenarioShift] = useState(0);
  const [line, setLine] = useState(71.5);
  const [overOdds, setOverOdds] = useState(1.91);
  const [underOdds, setUnderOdds] = useState(1.91);
  if (source.loading) return <LoadingState label="Loading scoring expectation" />;
  if (source.error || !scoring) return <EmptyState title="Scoring expectation unavailable" detail={source.error ?? "This run has no scoring expectation, and the page will not guess one from finish or betting-market data."} />;
  if (expectedRun !== null && scoring.run_id !== expectedRun) return <EmptyState title="Scoring run does not match" detail="The scoring numbers belong to a different run than the one selected, so they are hidden until the published data lines up." />;
  if (scoring.status !== "available") return <EmptyState title="Scoring expectation not published" detail={scoring.source_notes.map(plainText).filter(Boolean).join(" ") || plainText(scoring.baseline.detail) || "This run did not contain enough scoring data."} />;
  const validPlayers = scoring.players.filter((p) => p.pmf.length && p.pmf.every(([, prob]) => Number.isFinite(prob)));
  const activePlayers = fieldSelected ? [] : (selectedIds.length ? selectedIds.map((id) => validPlayers.find((p) => p.id === id)).filter((p): p is ScoringPlayer => !!p) : validPlayers.slice(0, 1));
  const chosen = activePlayers[0] ?? null;
  // Field moments are published, but no joint field-average PMF is, so none is built. The field chart is a different, labeled thing: every player's round score pooled (a mixture of marginals).
  const selectedRows = chosen?.pmf ?? [];
  const selectedLabel = chosen ? prettyName(chosen.name) : "Active field average";
  const probs = selectedRows.length ? (scenarioShift ? shiftedOverUnder(selectedRows, line, scenarioShift) : overUnder(selectedRows, line)) : null;
  const noVig = marketNoVig(overOdds, underOdds);
  const allMatches = query ? validPlayers.filter((p) => prettyName(p.name).toLowerCase().includes(query.toLowerCase()) || p.name.toLowerCase().includes(query.toLowerCase())).slice(0, 8) : [];
  const shifts = Array.from({ length: 7 }, (_, i) => scenarioShift + (i - 3) * 0.5);
  const sensitivity = selectedRows.length ? shifts.map((shift) => ({ shift, over: 100 * shiftedOverUnder(selectedRows, line, shift).over, under: 100 * shiftedOverUnder(selectedRows, line, shift).under })) : [];
  const weatherOffsetsUnavailable = scoring.weather.relative_offsets_status === "unavailable";
  const bridge = expectationBridge(scoring, chosen);
  const visibleComponents = bridge.components.filter((c) => !(c.key === "weather" && weatherOffsetsUnavailable) && Math.abs(c.strokes) >= 0.005);
  const maxComponent = Math.max(0.1, ...(visibleComponents.map((c) => Math.abs(c.strokes))), Math.abs(scoring.baseline.arithmetic_residual ?? 0));
  const rawDoc = source.data as Record<string, unknown>;
  const modelMeta = rawDoc.model && typeof rawDoc.model === "object" ? rawDoc.model as Record<string, unknown> : {};
  const asOf = typeof rawDoc.as_of === "string" ? rawDoc.as_of : null;
  const courseCard = rawDoc.course_card && typeof rawDoc.course_card === "object" ? rawDoc.course_card as Record<string, unknown> : {};
  const holeProfile = Array.isArray(courseCard.holes) ? courseCard.holes.filter((h): h is Record<string, unknown> => !!h && typeof h === "object" && typeof (h as Record<string, unknown>).hole === "number" && typeof (h as Record<string, unknown>).par === "number" && typeof (h as Record<string, unknown>).exp === "number").map((h) => ({ hole: h.hole as number, par: h.par as number, exp: h.exp as number })) : [];
  const eventMeta = rawDoc.event && typeof rawDoc.event === "object" ? rawDoc.event as Record<string, unknown> : {};
  const evidenceCtx: EvidenceContext = { courseName: typeof eventMeta.course === "string" ? eventMeta.course : typeof courseCard.course === "string" ? courseCard.course : null, eventYear: typeof eventMeta.date_start === "string" ? Number(eventMeta.date_start.slice(0, 4)) || null : null, forecastTime: scoring.weather.freshness ? etTime(scoring.weather.freshness, "") || null : null };
  const evidence = scoringEvidence(scoring, evidenceCtx);
  const warnings = [...scoring.confidence.reasons.map(plainText).filter((x): x is string => !!x), ...(scoring.weather.status && !/available|applied|fresh|\bok\b|common[_ ]only/i.test(scoring.weather.status) ? [`Weather status: ${plainText(scoring.weather.status) ?? "not reported"}.`] : []), ...(weatherOffsetsUnavailable ? [plainText(scoring.weather.relative_offsets_reason) ?? "Weather effects for individual players are not available, so they are not split out below."] : []), ...(scoring.baseline.median_hole_observations == null ? ["How many earlier rounds exist on this course was not reported."] : [])];
  const notes = [...new Set([...warnings, ...scoring.source_notes.map(plainText).filter((x): x is string => !!x)])];
  const mc = chosen ? chosen.sd / Math.sqrt(chosen.n_draws) : scoring.confidence.monte_carlo_se?.field ?? null;
  const pmfs = activePlayers.map((p) => ({ id: p.id, pmf: normalizePmf(p.pmf) }));
  const fieldMix = fieldSelected ? fieldMixturePmf(validPlayers) : [];
  const series = fieldSelected ? (fieldMix.length ? [{ key: "p_0", name: "A random player's round", color: COLORS[0] }] : []) : activePlayers.slice(0, 4).map((p, i) => ({ key: `p_${p.id}`, name: prettyName(p.name), color: COLORS[i] }));
  const allRows = curveRows(fieldSelected ? [{ id: 0, pmf: fieldMix }] : pmfs, curveMode === "cdf");
  const massy = (row: Record<string, number>) => series.some((x) => (row[x.key] ?? 0) >= 0.05) || curveMode === "cdf";
  const firstIdx = allRows.findIndex((row) => curveMode === "cdf" ? (series.some((x) => (row[x.key] ?? 0) >= 0.05)) : massy(row)), lastIdx = allRows.length - 1 - [...allRows].reverse().findIndex((row) => curveMode === "cdf" ? series.some((x) => (row[x.key] ?? 0) <= 99.95) : massy(row));
  const chartRows = firstIdx >= 0 && lastIdx >= firstIdx ? allRows.slice(Math.max(0, firstIdx - 1), lastIdx + 2) : allRows;
  const tooltipStyle = { background: "var(--popover)", border: "1px solid var(--line)", borderRadius: 10 };
  const seriesName = (key: unknown) => series.find((x) => x.key === String(key))?.name ?? String(key);
  const mixMean = pmfMean(fieldMix);
  const wxRound = scoring.round != null ? weatherExtras.rounds[scoring.round] : undefined;
  const baselineWeather = weatherBaselineText(wxRound?.venueSeen ?? null);
  const playerTermsTotal = bridge.components.reduce((sum, c) => sum + c.strokes, 0);
  const showPlayerTerms = !!chosen || Math.abs(playerTermsTotal) >= 0.005;
  const resolvedOver = probs ? resolvedOverShare(probs) : null;
  const breakevenOverShift = selectedRows.length ? evBreakEvenShift(selectedRows, line, "over", overOdds) : null;
  const breakevenUnderShift = selectedRows.length ? evBreakEvenShift(selectedRows, line, "under", underOdds) : null;
  const fLo = scoring.field.active_players_min, fHi = scoring.field.active_players_max;
  const fieldSizeNote = `Field averages are for the ${fLo != null && fHi != null && fLo !== fHi ? `${fLo} to ${fHi}` : scoring.field.players} players still in the field, using the same simulated weather for everyone.`;
  const expectedScore = chosen?.mean ?? scoring.field.mean;
  const vsPar = expectedScore != null && scoring.baseline.round_par != null ? expectedScore - scoring.baseline.round_par : null;
  return <div className="score-page">
    <div className="score-context"><span className={latest ? "score-pill latest" : "score-pill"}>{latest ? "Latest run" : "Earlier run"}</span><span>Round {scoring.round}</span>{asOf && etTime(asOf, "") && <span>{`As of ${etTime(asOf)}`}</span>}{!latest && <span>Stored with its original model inputs</span>}</div>
    <section className="score-decision" aria-labelledby="score-decision-title">
      <div className="score-decision-main"><span className="score-kicker"><span className="score-pulse" />EXPECTED ROUND SCORE · {selectedLabel}</span><h2 id="score-decision-title">{fmt(expectedScore)} <small>strokes</small></h2><p>{chosen ? `Model average for ${prettyName(chosen.name)} in Round ${scoring.round}.` : `Model average of Round ${scoring.round} scores for the ${scoring.field.players} players still in the field.`}{vsPar != null ? ` ${Math.abs(vsPar).toFixed(2)} ${vsPar < 0 ? "below" : "above"} par ${scoring.baseline.round_par}.` : ""}</p><div className="score-ci" aria-label="Central 80 percent outcome interval"><span title="Eight times out of ten, the simulated score lands inside this range.">Middle 80% of simulated outcomes</span><b>{fmt(chosen?.p10 ?? scoring.field.p10)} <i>to</i> {fmt(chosen?.p90 ?? scoring.field.p90)}</b><small>How widely the score could land (std dev {fmt(chosen?.sd ?? scoring.field.sd)})</small></div></div>
      <div className="score-confidence"><div className="confidence-icon"><CircleHelp size={19} /></div><span>HOW SURE IS THE AVERAGE?</span><strong>Not yet measured</strong><p>The 80% range shows the simulated outcomes. We do not yet have a validated range for the true average.</p></div>
    </section>
    {evidence.warning && <aside className="score-alert" role="note"><Info size={17} /><div><b>Main data limitation</b><p>{plainText(evidence.warning) ?? "Course difficulty is less certain than usual for this run."}</p></div></aside>}
    <ScoringEvidencePanel scoring={scoring} context={evidenceCtx} />

    <Panel title="Pick a player or stay with the field" eyebrow="Compare up to four players">
      <div className="score-player-tools"><div className="score-player-chips"><button type="button" className={!fieldSelected && selectedIds.length === 0 ? "selected" : ""} onClick={() => { setFieldSelected(false); setSelectedIds([]); }} aria-pressed={!fieldSelected && selectedIds.length === 0}>Default player</button><button type="button" className={fieldSelected ? "selected" : ""} onClick={() => { setFieldSelected(true); }} aria-pressed={fieldSelected}>Active field average</button><span className="score-player-active">{fieldSelected ? "Showing the field average" : `${activePlayers.length} of 4 players compared`}</span></div><label className="score-search"><Search size={15} /><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Find and add a player" aria-label="Find and add a player" />{query && <span className="score-search-results" role="group" aria-label="Matching players">{allMatches.map((p) => <button type="button" key={p.id} onClick={() => { setFieldSelected(false); setSelectedIds((prev) => prev.includes(p.id) ? prev : [...prev, p.id].slice(-4)); setQuery(""); }}>{prettyName(p.name)}<small>{fmt(p.mean)} strokes</small></button>)}{!allMatches.length && <em>No matching player.</em>}</span>}</label></div>
      {!fieldSelected && activePlayers.length > 0 && <div className="score-selected-chips">{activePlayers.map((p, i) => <span key={p.id} style={{ "--player-color": COLORS[i] } as React.CSSProperties}><i />{prettyName(p.name)}<button type="button" aria-label={`Remove ${prettyName(p.name)} from overlay`} disabled={activePlayers.length <= 1} onClick={() => setSelectedIds((prev) => (prev.length ? prev : activePlayers.map((x) => x.id)).filter((id) => id !== p.id))}><X size={13} /></button></span>)}</div>}
      <div className="score-player-grid">{validPlayers.slice(0, 4).map((p, i) => <button type="button" key={p.id} className={`score-player-card ${activePlayers.some((x) => x.id === p.id) ? "chosen" : ""}`} aria-pressed={activePlayers.some((x) => x.id === p.id)} onClick={() => { setFieldSelected(false); setSelectedIds((prev) => { const base = prev.length ? prev : activePlayers.map((x) => x.id); return base.includes(p.id) ? (base.length <= 1 ? base : base.filter((id) => id !== p.id)) : [...base, p.id].slice(-4); }); }} style={{ "--player-color": COLORS[i] } as React.CSSProperties}><span className="player-dot" />{prettyName(p.name)}<b>{fmt(p.mean)} <small>strokes</small></b><span title="Good day: only 1 in 10 simulated rounds is better. Bad day: only 1 in 10 is worse. Typical is the single most likely round; the average is higher because a few very bad rounds pull it up.">Good day {fmt(p.p10)} · Typical {fmt(p.p50)} · Bad day {fmt(p.p90)}{p.data_depth != null ? ` · ${p.data_depth} prior rounds of data` : ""}{p.tee_time ? ` · Tees off ${plainTeeTime(p.tee_time)}` : ""}</span></button>)}</div>
      <p className="score-footnote">{fieldSizeNote} Typical is the most likely round; the average sits higher because a few very bad rounds pull it up.</p>
    </Panel>

      <div className="score-two-col">
      <Panel title="The full range of outcomes" eyebrow={fieldSelected ? "Every player's round score, pooled across the field" : `${selectedLabel} · chance of each round score`}>
        <div className="score-chart-tools"><div className="score-segments" role="group" aria-label="Score distribution view"><button type="button" className={curveMode === "pmf" ? "active" : ""} aria-pressed={curveMode === "pmf"} onClick={() => setCurveMode("pmf")} title="Chance of each exact round score, such as exactly 70.">Exact score</button><button type="button" title="Chance of shooting that score or lower." className={curveMode === "cdf" ? "active" : ""} aria-pressed={curveMode === "cdf"} onClick={() => setCurveMode("cdf")}>Chance of scoring this or lower</button></div><span>{fieldSelected ? "All players pooled" : `${activePlayers.length} player${activePlayers.length === 1 ? "" : "s"} compared`}</span></div>
        {chartRows.length ? <><div className="score-chart" role="img" aria-label={`${curveMode === "cdf" ? "Chance of scoring this or lower" : "Chance of each exact score"} for ${fieldSelected ? "every player in the field pooled together" : activePlayers.map((p) => prettyName(p.name)).join(", ")}`}><ResponsiveContainer width="100%" height="100%">{curveMode === "pmf" ? <BarChart data={chartRows} margin={{ top: 12, right: 16, left: 0, bottom: 8 }}><CartesianGrid vertical={false} stroke="var(--line)" /><XAxis dataKey="score" type="number" domain={["dataMin - 0.5", "dataMax + 0.5"]} allowDecimals={false} tickFormatter={(v) => fmt(Number(v), 0)} tick={{ fontSize: 11 }} stroke="var(--muted)" label={{ value: "Strokes", position: "insideBottom", offset: -2, fill: "var(--muted)" }} /><YAxis unit="%" domain={[0, "auto"]} width={45} tick={{ fontSize: 11 }} stroke="var(--muted)" tickFormatter={(v) => Number(v).toFixed(0)} /><Tooltip contentStyle={tooltipStyle} formatter={(v, key) => [`${Number(v).toFixed(1)}%`, seriesName(key)]} labelFormatter={(v) => `Exactly ${Number(v).toFixed(0)}`} />{series.map((x) => <Bar key={x.key} dataKey={x.key} name={x.name} fill={x.color} fillOpacity={series.length > 1 ? 0.6 : 0.9} radius={[3, 3, 0, 0]} isAnimationActive={false} />)}</BarChart> : <LineChart data={chartRows} margin={{ top: 12, right: 16, left: 0, bottom: 8 }}><CartesianGrid vertical={false} stroke="var(--line)" /><XAxis dataKey="score" type="number" domain={["dataMin", "dataMax"]} allowDecimals={false} tickFormatter={(v) => fmt(Number(v), 0)} tick={{ fontSize: 11 }} stroke="var(--muted)" label={{ value: "Strokes", position: "insideBottom", offset: -2, fill: "var(--muted)" }} /><YAxis unit="%" domain={[0, 100]} width={45} tick={{ fontSize: 11 }} stroke="var(--muted)" /><Tooltip contentStyle={tooltipStyle} formatter={(v, key) => [`${Number(v).toFixed(1)}%`, seriesName(key)]} labelFormatter={(v) => `Score of ${Number(v).toFixed(0)} or lower`} />{series.map((x) => <Line key={x.key} dataKey={x.key} name={x.name} type="monotone" stroke={x.color} strokeWidth={2.5} dot={false} isAnimationActive={false} />)}</LineChart>}</ResponsiveContainer></div>
        <div className="score-chart-legend">{fieldSelected ? <span><i />A random player&rsquo;s round · mean {fmt(mixMean)} strokes</span> : activePlayers.map((p, i) => <span key={p.id}><i style={{ background: COLORS[i] }} />{prettyName(p.name)} · mean {fmt(p.mean)}</span>)}</div><p className="score-footnote">{fieldSelected ? `Each bar is the share of all player-rounds expected to land on that exact score, pooling the score chances of all ${validPlayers.length} players at equal weight. It shows how scores spread across the field, not how the field average could move. Pick a player to see theirs.` : "Exact score is the chance of each round score; this or lower adds those chances up as the score rises."}</p></> : <EmptyState title={fieldSelected ? "Field chart unavailable" : "Player chart unavailable"} detail={fieldSelected ? "No player score chances were published for this run, so there is nothing honest to pool. Pick a player to see theirs." : "No score chances are available for this player."} />}
      </Panel>
      <Panel title="How the expected score adds up" eyebrow="Course difficulty + weather + player adjustments, in strokes">
        <details className="score-details"><summary>See each player adjustment</summary>{visibleComponents.length ? <div className="score-build">{visibleComponents.map((c) => <div className="score-build-row" key={c.key}><div><span>{c.label}</span><b>{relative(c.strokes)}</b></div><div className="score-build-track"><i style={{ width: `${Math.min(100, Math.abs(c.strokes) / maxComponent * 46)}%`, marginLeft: c.strokes < 0 ? `${50 - Math.min(50, Math.abs(c.strokes) / maxComponent * 46)}%` : "50%", background: c.strokes < 0 ? "var(--positive)" : "var(--warning)" }} /></div>{c.note && <small>{c.note}</small>}</div>)}{chosen?.components.some((c) => c.key === "unexplained_residual") && <p className="score-footnote">The leftover line is whatever the named adjustments do not explain, so the pieces add up to the model average. Adjustments under 0.005 strokes are hidden.</p>}</div> : fieldSelected ? <p className="score-footnote">Player skill and weather differences are centered on the field, so for the field average they cancel to zero. The field number is just the expected course difficulty plus weather. Pick a player above to see their adjustments.</p> : <EmptyState title="Component breakdown unavailable" detail="The published scoring block has no player component detail for this outcome." />}
        </details><div className="score-baseline"><span>Expected-score arithmetic</span><div title="Round par plus the average hole difficulty at this course."><span>Expected course difficulty (par-adjusted)</span><b>{fmt(scoring.baseline.course_score)}</b></div><div title={baselineWeather.hint}><span>{baselineWeather.label}</span><b>{relative(scoring.baseline.common_weather, 2)}</b></div>{showPlayerTerms && <div title={chosen ? "Skill, form, travel and other player-specific terms, including whatever the named terms leave unexplained." : "Average of the named player terms (skill, form and so on) across the field. These are centered, so they are usually close to zero."}><span>{chosen ? "This player's own terms (skill, form, travel)" : "Average of the players' own terms (skill, form)"}</span><b>{relative(playerTermsTotal, 2)}</b></div>}{!chosen && bridge.residual != null && Math.abs(bridge.residual) >= 0.005 && <div title="Small gap left over from rounding in the stored pieces."><span>Rounding leftover</span><b>{relative(bridge.residual, 3)}</b></div>}<div><span>{bridge.total == null ? "Total not available for this checkpoint" : "Expected round score"}</span><b>{fmt(bridge.total)}</b></div><small>{WEATHER_HISTORY_CAVEAT}</small></div>
      </Panel>
    </div>

    <WeatherThisRound doc={weatherDoc} extras={weatherExtras} loading={weatherSource.loading} round={scoring.round} latest={latest} />
    {historyKey !== undefined && <CourseHistoryPanel historyKey={historyKey} />}

    <details className="score-details score-investigate"><summary>See how hard each hole plays</summary><Panel title="Course difficulty by hole" eyebrow={holeProfile.length ? "Expected strokes vs par on each hole" : "Hole-by-hole course difficulty"}>
      {holeProfile.length ? <div className="score-hole-chart" role="img" aria-label="Published course expected strokes relative to par by hole"><ResponsiveContainer width="100%" height="100%"><BarChart data={holeProfile} margin={{ top: 8, right: 8, bottom: 4, left: 0 }}><CartesianGrid vertical={false} stroke="var(--line)" /><XAxis dataKey="hole" interval={0} tick={{ fontSize: 10 }} stroke="var(--muted)" label={{ value: "Hole", position: "insideBottom", offset: -3, fill: "var(--muted)" }} /><YAxis width={40} tick={{ fontSize: 10 }} stroke="var(--muted)" tickFormatter={(v) => Number(v) > 0 ? `+${v}` : String(v)} /><ReferenceLine y={0} stroke="var(--muted)" /><Tooltip contentStyle={{ background: "var(--popover)", border: "1px solid var(--line)", borderRadius: 10 }} formatter={(v) => [`${Number(v) > 0 ? "+" : ""}${Number(v).toFixed(2)} vs par`, "Expected strokes"]} labelFormatter={(h) => `Hole ${h} · par ${holeProfile.find((x) => x.hole === Number(h))?.par ?? "—"}`} /><Bar dataKey="exp" name="Expected strokes vs par" radius={[3, 3, 0, 0]} isAnimationActive={false}>{holeProfile.map((h) => <Cell key={h.hole} fill={h.par === 3 ? "var(--chart-2)" : h.par === 5 ? "var(--chart-3)" : "var(--chart-1)"} />)}</Bar></BarChart></ResponsiveContainer></div> : <EmptyState title="Hole profile unavailable" detail="No hole-by-hole expectation was published for this run." />}<p className="score-footnote">The base hole table totals approximately {fmt(holeProfile.reduce((sum, h) => sum + h.par + h.exp, 0))} strokes. Expected course difficulty for Round {scoring.round} is {fmt(scoring.baseline.course_score)} before weather. This profile does not attribute an individual player’s score.</p>
    </Panel>

    </details>
    <details className="score-details score-investigate"><summary>Compare a manual player line and test score shifts</summary><Panel title="What if the market line moved?" eyebrow="Your line and prices">
      <div className="score-market-inputs"><label>Score line<input type="number" step="0.5" value={line} onChange={(e) => setLine(Number(e.target.value))} aria-label="Manual score line" /></label><label>Over price (decimal)<input type="number" step="0.01" min="1.01" value={overOdds} onChange={(e) => setOverOdds(Number(e.target.value))} aria-label="Over decimal odds" /></label><label>Under price (decimal)<input type="number" step="0.01" min="1.01" value={underOdds} onChange={(e) => setUnderOdds(Number(e.target.value))} aria-label="Under decimal odds" /></label><span className="manual-label"><Info size={14} />Starting values are placeholders. Enter the line and prices you see.</span></div>
      <div className="score-shift-presets"><span>What if scores run higher or lower?</span>{[-1, -0.5, 0, 0.5, 1].map((s) => <button key={s} type="button" className={scenarioShift === s ? "active" : ""} aria-pressed={scenarioShift === s} onClick={() => setScenarioShift(s)}>{s > 0 ? "+" : ""}{s.toFixed(1)} strokes</button>)}</div>
      <div className="score-market-grid">{fieldSelected ? <div className="score-market-card model"><span>PICK A PLAYER</span><p>Not available for the field average. Pick one player above to compare a score line.</p></div> : <><div className="score-market-card model"><span>MODEL · {selectedLabel}</span>{probs ? <><div><b>{pct(probs.over)}</b><small>OVER</small><b>{pct(probs.under)}</b><small>UNDER</small></div><p>Chance the score goes over {line}: {pct(probs.over)}. Under: {pct(probs.under)}.{probs.push > 0.0005 ? ` Exactly ${line}: ${pct(probs.push)}.` : ""}{scenarioShift ? ` Scores shifted ${scenarioShift > 0 ? "+" : ""}${scenarioShift.toFixed(1)} strokes.` : ""}</p><small title={defn("break_even", "The price at which a bet on this side makes no profit or loss in the long run.")}>Break-even price: Over {decimalToAmerican(breakEvenDecimal(probs, "over"))} · Under {decimalToAmerican(breakEvenDecimal(probs, "under"))}</small></> : <p>Player score distribution unavailable.</p>}</div><div className="score-market-card market"><span title="The chances the betting prices imply once the bookmaker’s margin is removed.">WHAT YOUR PRICES IMPLY</span><div><b>{noVig ? pct(noVig.over) : "—"}</b><small>OVER</small><b>{noVig ? pct(noVig.under) : "—"}</b><small>UNDER</small></div><p>At over {overOdds || "—"} / under {underOdds || "—"}, with the bookmaker’s margin removed.</p><small>{noVig ? "A push is left out of these shares." : "Enter prices above 1.00 on both sides."}</small></div><div className="score-market-card ev"><span title="What you would win or lose on average for every 1 staked, over many bets. Negative means you lose money on average.">EXPECTED PROFIT PER 1 STAKED</span><div><b>{fmt(probs && betEv(probs, "over", overOdds) != null ? 100 * betEv(probs, "over", overOdds)! : null, 1)}%</b><small>OVER</small><b>{fmt(probs && betEv(probs, "under", underOdds) != null ? 100 * betEv(probs, "under", underOdds)! : null, 1)}%</b><small>UNDER</small></div><p>{probs ? "A score exactly on the line returns your stake." : "Pick a player to see this."}</p><small>Long-run average, not a stake suggestion.</small></div></>}</div>
      <p className="score-footnote">The line and prices are yours; the model does not use them. Whole-number lines can land exactly on the line (a push). Shifting scores moves the existing chances up or down; it is not a new simulation.</p>
    </Panel>

    {sensitivity.length > 0 && <div>
      <Panel title="If scores shifted" eyebrow="How the chances move">
        {sensitivity.length ? <><div className="score-sensitivity-chart" role="img" aria-label={`Over and under model probabilities at score shifts around line ${line}`}><ResponsiveContainer width="100%" height="100%"><LineChart data={sensitivity} margin={{ top: 12, right: 18, left: 0, bottom: 8 }}><CartesianGrid vertical={false} stroke="var(--line)" /><XAxis dataKey="shift" tickFormatter={(v) => `${Number(v) > 0 ? "+" : ""}${v}`} tick={{ fontSize: 10 }} stroke="var(--muted)" label={{ value: "Shift in scores (strokes)", position: "insideBottom", offset: -3, fill: "var(--muted)" }} /><YAxis unit="%" domain={[0, 100]} width={45} tick={{ fontSize: 10 }} stroke="var(--muted)" /><Tooltip contentStyle={{ background: "var(--popover)", border: "1px solid var(--line)", borderRadius: 10 }} formatter={(v, name) => [`${Number(v).toFixed(1)}%`, String(name).toUpperCase()]} labelFormatter={(v) => `Scores shifted ${Number(v) > 0 ? "+" : ""}${v} strokes`} /><ReferenceLine x={scenarioShift} stroke="var(--muted)" strokeDasharray="3 4" /><Line dataKey="over" name="Over" stroke="var(--chart-1)" strokeWidth={2.5} dot={{ r: 3 }} isAnimationActive={false} /><Line dataKey="under" name="Under" stroke="var(--chart-3)" strokeWidth={2.5} dot={{ r: 3 }} isAnimationActive={false} /></LineChart></ResponsiveContainer></div>
        <p className="score-footnote">A scenario, not a fresh simulation: every score is moved up or down by the shift shown.</p><div className="score-breakeven-shifts"><span title="How far scores would have to move before a bet at your price breaks even.">Score shift needed to break even</span><b>Over: {breakevenOverShift == null ? "more than 5 strokes away" : `${breakevenOverShift > 0 ? "+" : ""}${breakevenOverShift.toFixed(2)} strokes`}</b><b>Under: {breakevenUnderShift == null ? "more than 5 strokes away" : `${breakevenUnderShift > 0 ? "+" : ""}${breakevenUnderShift.toFixed(2)} strokes`}</b><small>Uses your current line and prices.</small></div></> : null}
      </Panel>
    </div>}
    </details>
    <details className="score-details score-investigate"><summary>Technical details</summary>
      <div className="score-source-detail">
        <TechList rows={technicalRows(scoring, rawDoc, modelMeta, chosen, mc)} />
        {noVig && resolvedOver != null && <p>Model chance of over, leaving out pushes: {pct(resolvedOver)}. Your prices imply: {pct(noVig.over)}.</p>}
        {notes.length > 0 && <ul>{notes.map((note, i) => <li key={i}>{note}</li>)}</ul>}
      </div>
    </details>
  </div>;
}


type TechRow = [label: string, value: string | null | undefined];
function TechList({ rows }: { rows: TechRow[] }) {
  const shown = rows.filter((r): r is [string, string] => !!r[1]);
  return <dl className="score-tech-list">{shown.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl>;
}
/** Lineage and precision details for the collapsed "Technical details" block. Times are US Eastern; calendar dates carry no time zone. Identifier-like publisher strings stay in the data, not on the page. */
function technicalRows(scoring: ScoringDoc, rawDoc: Record<string, unknown>, modelMeta: Record<string, unknown>, chosen: ScoringPlayer | null, mc: number | null): TechRow[] {
  void rawDoc; void modelMeta;
  const p = scoring.baseline.course_provenance, u = scoring.round_update, w = scoring.weather;
  const range = (a: string | null | undefined, b: string | null | undefined) => a && b ? (a === b ? a : `${a}–${b}`) : a ?? b ?? null;
  const when = runMoment(scoring.run_id);
  const join = (parts: Array<string | null | undefined>) => parts.map((x) => plainText(x ?? null)).filter(Boolean).join(" · ") || null;
  return [
    ["Simulation run", when ? etTime(when) : null],
    ["How it was built", "Round scores come from simulating the tournament many times, using the course, the weather and each player’s skill."],
    ["Simulations", `${scoring.n_draws.toLocaleString()} simulated tournaments`],
    ["Std dev of outcomes", `${fmt(chosen?.sd ?? scoring.field.sd)} strokes`],
    ["Simulation precision", mc != null ? `±${fmt(mc, 4)} strokes. This measures simulation noise only.` : null],
    ["How course difficulty is defined", plainText(scoring.baseline.detail)],
    ["Hole layout", p ? join([p.layout.source, p.layout.status, p.layout.year != null ? String(p.layout.year) : null]) : null],
    ["Tour-wide comparison", p ? [p.general_fit.observations != null ? `${p.general_fit.observations.toLocaleString()} holes` : null, range(p.general_fit.start, p.general_fit.end), p.general_fit.cutoff ? `data through ${plainDate(p.general_fit.cutoff)}` : null].filter(Boolean).join(" · ") : null],
    ["Skill response data", p?.general_fit.skill_response_cutoff ? `data through ${plainDate(p.general_fit.skill_response_cutoff)}` : null],
    ["Earlier rounds on this course", p ? [p.venue_history.observations != null ? (p.venue_history.observations === 0 ? "None recorded" : `${p.venue_history.observations.toLocaleString()} holes`) : null, p.venue_history.editions.length ? `editions ${p.venue_history.editions.join(", ")}` : null, p.venue_history.cutoff ? `before ${plainDate(p.venue_history.cutoff)}` : null].filter(Boolean).join(" · ") : scoring.baseline.median_hole_observations != null ? (scoring.baseline.median_hole_observations === 0 ? "None recorded" : "Earlier rounds recorded") : null],
    ["Past weather", p ? join([p.historical_weather.reference, p.historical_weather.status, p.historical_weather.field_adjustment, p.historical_weather.method, p.historical_weather.note]) : null],
    ["Field strength", p ? join([p.field_strength.status, p.field_strength.detail, p.field_strength.method]) : null],
    ["Weather", [plainText(w.source), plainText(w.status), w.freshness ? `updated ${etTime(w.freshness)}` : null].filter(Boolean).join(" · ") || null],
    ["Weather in the numbers", w.applied_in_prices === true ? "Included, using each group’s tee time and forecast wind." : w.applied_in_prices === false ? "Not included" : null],
    ["Completed-round update", u ? join([u.status, u.method, u.reason, u.source_rounds.length ? `rounds ${u.source_rounds.join(", ")}` : null, u.observations != null ? `${u.observations.toLocaleString()} observations` : null]) : null],
  ];
}

/** Plain-language account of where the course number comes from. Technical lineage lives in the collapsed block at the bottom of the page. */
export function ScoringEvidencePanel({ scoring, context = {} }: { scoring: ScoringDoc; context?: EvidenceContext }) {
  const evidence = scoringEvidence(scoring, context), update = scoring.round_update;
  const rounds = update?.observed_rounds ?? [];
  const cols = { expected: rounds.some((r) => r.field_expected != null), weather: rounds.some((r) => r.weather_strokes != null), residual: rounds.some((r) => r.residual != null) };
  const showTable = rounds.length > 0 && (cols.expected || cols.weather || cols.residual);
  const na = <span title="Not published for this round">n/a</span>;
  return <Panel title="Where the course difficulty comes from" eyebrow="In plain terms">
    <ul className="score-story">{evidence.rows.map((row) => <li key={row.label} title={row.detail}><b>{row.label}</b><span>{row.value}</span></li>)}</ul>
    {showTable && <details className="score-details"><summary>Completed rounds: actual vs expected</summary><div className="score-table-scroll"><table className="score-observed-table"><caption>Actual scores are evidence. Only a verified adjustment changes course difficulty.</caption><thead><tr><th>Round</th><th title="Average score of the players who finished the round.">Actual field average</th>{cols.expected && <th title="What the model expected the field average to be before the round.">Expected</th>}{cols.weather && <th title="Strokes the forecast weather was expected to add (+) or take off (−).">Weather</th>}{cols.residual && <th title="How far the actual average landed from what the model expected, after weather.">Surprise</th>}<th title="Players who finished the round.">Players</th></tr></thead><tbody>{rounds.map((r, i) => <tr key={`${r.round}-${i}`}><th scope="row">{r.round}</th><td>{fmt(r.field_actual)}</td>{cols.expected && <td>{r.field_expected == null ? na : fmt(r.field_expected)}</td>}{cols.weather && <td>{r.weather_strokes == null ? na : relative(r.weather_strokes, 2)}</td>}{cols.residual && <td>{r.residual == null ? na : relative(r.residual, 2)}</td>}<td>{r.players ?? na}</td></tr>)}</tbody></table></div></details>}
  </Panel>;
}

const plainComponent = (k: string) => k === "traj_wind" ? "Wind vs ball flight" : k === "tod" ? "Time of day" : COMPONENT_LABELS[k] ?? k.replace(/_/g, " ");
const COMPONENT_WHY: Record<string, string> = {
  wind: "Average wind during the group's round.", gust: "Gusts during the round.", temp: "Temperature.", rain: "Rain during the round.",
  tod: "Time of day: the course typically plays differently early versus late.", traj_wind: "How the wind interacts with this player's typical ball flight (only players with a radar profile).",
};
const KNOWN_COMPONENTS = ["wind", "gust", "temp", "rain", "tod", "traj_wind"];
const teeHourOf = (t: string): number | null => { const m = /(\d{1,2}):(\d{2})(?::\d{2})?$/.exec(t.trim()); return m ? Number(m[1]) + Number(m[2]) / 60 : null; };

/** Expected wave split and its causes, built only from what the published weather document contains. */
export function WeatherThisRound({ doc, extras, loading, round, latest }: { doc: WeatherDoc | null; extras: WeatherExtras; loading: boolean; round: number | null; latest: boolean }) {
  const [pick, setPick] = useState<number | null>(null);
  const link = <Link className="ex-link" href="/weather-effects">Full weather effects page <ChevronRight size={14} /></Link>;
  if (loading) return null;
  if (!doc) return <Panel title="Weather this round" eyebrow="Expected wave split and why" actions={link}><p className="score-footnote">No weather document is published for this event yet, so the wave split is not shown.</p></Panel>;
  const rounds = [...new Set([...doc.waves.map((w) => w.round), ...doc.field.map((f) => f.round)])].filter((x) => x > 0).sort((a, b) => a - b);
  const r = pick ?? (round != null && rounds.includes(round) ? round : rounds[0] ?? null);
  if (r == null) return <Panel title="Weather this round" eyebrow="Expected wave split and why" actions={link}><p className="score-footnote">The weather document has no round detail yet.</p></Panel>;
  const waves = doc.waves.filter((w) => w.round === r).sort((a, b) => (a.window?.[0] ?? 0) - (b.window?.[0] ?? 0));
  const field = doc.field.find((f) => f.round === r);
  const info = extras.rounds[r];
  const people = doc.players.flatMap((p) => { const pr = p.rounds[r]; return pr && Object.keys(pr.components).length ? [{ teeHour: pr.tee_time ? teeHourOf(pr.tee_time) : null, components: pr.components }] : []; });
  const split = waveComponentMeans(people, waves);
  const cols = waves.filter((w) => split.byWave[w.wave]);
  const ordered = KNOWN_COMPONENTS.filter((k) => split.keys.includes(k)).concat(split.keys.filter((k) => !KNOWN_COMPONENTS.includes(k)));
  const gap = waves.length >= 2 ? waveGapText(waves[0], waves[waves.length - 1]) : null;
  const sens = windSensitivityText(doc.venue.class, extras.venue);
  const first = cols[0], last = cols[cols.length - 1];
  const diff = (k: string) => first && last ? (split.byWave[last.wave][k] ?? 0) - (split.byWave[first.wave][k] ?? 0) : 0;
  const top = cols.length >= 2 ? [...ordered].sort((a, b) => Math.abs(diff(b)) - Math.abs(diff(a)))[0] : null;
  return <Panel title="Weather this round" eyebrow="Expected wave split and why" actions={link}>
    <div className="score-wx-head">
      {rounds.length > 1 && <label className="score-select">Round<select aria-label="Weather round" value={r} onChange={(e) => setPick(Number(e.target.value))}>{rounds.map((x) => <option key={x} value={x}>Round {x}</option>)}</select></label>}
      <span>{doc.as_of && etTime(doc.as_of, "") ? `Forecast as of ${etTime(doc.as_of)}` : "Forecast time not reported"}{!latest ? " · latest forecast, not necessarily the one this run used" : ""}</span>
    </div>
    {field?.common_strokes != null && <p className="score-wx-lead">Round {r} weather moves the whole field by <b>{signedFixed(field.common_strokes)} strokes</b> versus {info?.venueSeen === false ? "typical tour conditions" : info?.venueSeen === true ? "this course's typical conditions" : "typical conditions"}.</p>}
    {info?.status && info.status !== "ok" && <p className="score-footnote">{(info.reason && plainText(info.reason)) || `Status: ${info.status}`}. Wave detail is not available for this round.</p>}
    {waves.length ? <><div className="score-wx-waves">{waves.map((w) => <div key={`${w.round}-${w.wave}`}><span>{w.wave} wave{w.windowText ? ` · ${w.windowText.replace("-", "–")} local` : ""}</span><b>{signedFixed(w.rel, 3)} <small>strokes vs the field</small></b><small>{w.rel_p10 != null && w.rel_p90 != null ? `Range ${signedFixed(w.rel_p10, 3)} to ${signedFixed(w.rel_p90, 3)}` : ""}{w.mean_wind != null ? ` · wind ${w.mean_wind.toFixed(1)} mph` : ""}{w.mean_temp != null ? ` · ${w.mean_temp.toFixed(0)}°F` : ""}</small></div>)}</div>
      {gap && <p className="score-wx-lead">{gap}{top && cols.length >= 2 && Math.abs(diff(top)) >= 0.0005 ? ` The biggest piece of the gap is ${plainComponent(top).toLowerCase()} (${signedFixed(diff(top), 3)}).` : ""}</p>}</> : <p className="score-footnote">No tee waves are published for this round (only the field-wide weather shift).</p>}
    {cols.length > 0 && ordered.length > 0 && <div className="score-table-scroll"><table className="score-observed-table"><caption>Why: average strokes added (+) or saved (−) by each effect, by wave</caption><thead><tr><th>Effect</th>{cols.map((w) => <th key={w.wave}>{w.wave} wave ({split.counts[w.wave]} players)</th>)}{cols.length === 2 && <th>{cols[1].wave} minus {cols[0].wave}</th>}</tr></thead><tbody>{ordered.map((k) => <tr key={k} title={COMPONENT_WHY[k]}><th scope="row">{plainComponent(k)}</th>{cols.map((w) => <td key={w.wave}>{signedFixed(split.byWave[w.wave][k] ?? null, 3)}</td>)}{cols.length === 2 && <td>{signedFixed(diff(k), 3)}</td>}</tr>)}</tbody></table></div>}
    <div className="score-wx-venue"><b>Course type and wind sensitivity</b><p>{sens ?? (doc.venue.class ? `${doc.venue.class.charAt(0).toUpperCase()}${doc.venue.class.slice(1)} venue.` : "No venue type published.")}{extras.venue.global_slope != null && doc.venue.wind_slope != null ? ` Wind costs ${doc.venue.wind_slope.toFixed(3)} strokes per mph here, against ${extras.venue.global_slope.toFixed(3)} for the average course.` : ""}{extras.venue.slope_class ? ` Venue group: ${extras.venue.slope_class}.` : ""}</p>{info?.venueSeen === false && <p>This venue has no weather history in the model, so its weather is compared with typical tour conditions rather than its own.</p>}</div>
    <p className="score-footnote">Only effects published in the weather document are shown: {ordered.length ? ordered.map((k) => plainComponent(k).toLowerCase()).join(", ") : "none for this round"}. Any other condition (such as dew point) is not published as its own line.{doc.notes.some((n) => /offline mode/i.test(n)) ? " This document's forecast was captured offline, so it may not reflect the latest weather." : ""}</p>
  </Panel>;
}
