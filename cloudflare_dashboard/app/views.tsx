"use client";

import { useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { DataTable, EmptyState, ErrorState, Kpi, LoadingState, PageIntro, Panel, PlayerPicker, SegmentedControl } from "./components";
import { useDashboardData } from "./data";
import { DataRow, americanOdds, numberValue, palette, safeMean, sum, titleCase, uniqueStrings } from "./lib";

const chartMargin = { top: 16, right: 18, bottom: 8, left: 0 };

function ChartTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ name?: string; value?: unknown; color?: string }>; label?: unknown }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <strong>{titleCase(label)}</strong>
      {payload.map((item, index) => (
        <span key={`${item.name}-${index}`} style={{ color: item.color }}>
          {titleCase(item.name)}: {typeof item.value === "number" ? item.value.toFixed(2) : String(item.value ?? "—")}
        </span>
      ))}
    </div>
  );
}

function SelectControl({ label, value, options, onChange }: { label: string; value: string; options: Array<{ value: string; label: string }>; onChange: (value: string) => void }) {
  return (
    <label className="select-control">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => <option value={option.value} key={option.value}>{option.label}</option>)}
      </select>
    </label>
  );
}

function RangeControl({ label, value, min, max, step = 1, onChange }: { label: string; value: number; min: number; max: number; step?: number; onChange: (value: number) => void }) {
  return (
    <label className="range-control">
      <span>{label}<b>{value}</b></span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );
}

type RankPayload = { pre: DataRow[]; live: DataRow[]; h2h: DataRow[] };

function rankStats(rows: DataRow[], player: string) {
  const playerRows = rows.filter((row) => String(row.player_name) === player);
  const probability = (max: number, key: string) => sum(playerRows.filter((row) => numberValue(row.rank) <= max).map((row) => numberValue(row[key])));
  return {
    player,
    win: probability(1, "prob_u"),
    top5: probability(5, "prob_u"),
    top10: probability(10, "prob_u"),
    top20: probability(20, "prob_u"),
    winNdh: probability(1, "prob_ndh"),
  };
}

function DistributionExplorer({ rows, title, subtitle }: { rows: DataRow[]; title: string; subtitle: string }) {
  const [players, setPlayers] = useState<string[]>([]);
  const [maxRank, setMaxRank] = useState(40);
  const options = useMemo(() => {
    const candidates = uniqueStrings(rows, "player_name");
    return candidates.sort((a, b) => rankStats(rows, b).win - rankStats(rows, a).win);
  }, [rows]);
  const activePlayers = players.length ? players : options.slice(0, 2);
  const chartRows = useMemo(() => Array.from({ length: maxRank }, (_, index) => {
    const rank = index + 1;
    const point: DataRow = { rank };
    activePlayers.forEach((player) => {
      const row = rows.find((item) => String(item.player_name) === player && numberValue(item.rank) === rank);
      point[player] = numberValue(row?.prob_u) * 100;
    });
    return point;
  }), [activePlayers, maxRank, rows]);
  const stats = activePlayers.map((player) => rankStats(rows, player));

  if (!rows.length) return <EmptyState title="No distribution snapshot" detail="Run the simulation and publish dashboard data to populate this view." />;
  return (
    <div className="stack-lg">
      <Panel title={title} eyebrow={subtitle} actions={<RangeControl label="Ranks shown" value={maxRank} min={10} max={Math.max(40, Math.min(100, Math.max(...rows.map((row) => numberValue(row.rank))))) } step={5} onChange={setMaxRank} />}>
        <PlayerPicker options={options} value={activePlayers} onChange={setPlayers} max={5} />
        <div className="chart-large">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartRows} margin={chartMargin}>
              <defs>{activePlayers.map((player, index) => <linearGradient id={`rank-${index}`} key={player} x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={palette[index % palette.length]} stopOpacity={0.45}/><stop offset="95%" stopColor={palette[index % palette.length]} stopOpacity={0}/></linearGradient>)}</defs>
              <CartesianGrid stroke="var(--line)" vertical={false} />
              <XAxis dataKey="rank" stroke="var(--muted)" />
              <YAxis stroke="var(--muted)" tickFormatter={(value) => `${value}%`} />
              <Tooltip content={<ChartTooltip />} />
              <Legend formatter={(value) => titleCase(value)} />
              {[1, 5, 10, 20].filter((rank) => rank <= maxRank).map((rank) => <ReferenceLine key={rank} x={rank} stroke="var(--line-strong)" strokeDasharray="4 4" />)}
              {activePlayers.map((player, index) => <Area key={player} type="monotone" dataKey={player} stroke={palette[index % palette.length]} fill={`url(#rank-${index})`} strokeWidth={2} />)}
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </Panel>
      <div className="comparison-grid">
        {stats.map((stat, index) => (
          <Panel key={stat.player} eyebrow={`Fair win ${americanOdds(stat.winNdh || stat.win)}`} title={titleCase(stat.player)} className="player-summary">
            <div className="mini-stat-grid">
              {[ ["Win", stat.win], ["Top 5", stat.top5], ["Top 10", stat.top10], ["Top 20", stat.top20] ].map(([label, value]) => <div key={String(label)} style={{ borderColor: palette[index % palette.length] }}><span>{label}</span><strong>{(Number(value) * 100).toFixed(1)}%</strong></div>)}
            </div>
          </Panel>
        ))}
      </div>
    </div>
  );
}

export function DistributionsView() {
  const { data, loading, error } = useDashboardData<RankPayload>("distributions.json");
  const [mode, setMode] = useState<"pre" | "live">("live");
  if (loading) return <LoadingState label="Loading finish distributions" />;
  if (error || !data) return <ErrorState message={error ?? "Distribution data is unavailable."} />;
  const liveAvailable = data.live.length > 0;
  const activeMode = mode === "live" && !liveAvailable ? "pre" : mode;
  return (
    <>
      <PageIntro eyebrow="Finish model" title="Finish distributions" description="Compare full finish-position curves instead of relying on a single top-10 or win number." controls={<SegmentedControl label="Distribution mode" value={activeMode} onChange={setMode} options={[{ value: "pre", label: "Pre-event" }, { value: "live", label: "Live" }]} />} />
      <DistributionExplorer rows={data[activeMode]} title={activeMode === "live" ? "Live finish curves" : "Pre-event finish curves"} subtitle="Dead-heat adjusted probability by finishing rank" />
    </>
  );
}

type SgPayload = { raw: DataRow[]; adjusted: DataRow[]; predictions: DataRow[] };

export function SgDistributionsView() {
  const { data, loading, error } = useDashboardData<SgPayload>("sg-distributions.json");
  const [mode, setMode] = useState<"adjusted" | "raw">("adjusted");
  const [players, setPlayers] = useState<string[]>([]);
  if (loading) return <LoadingState label="Loading strokes-gained profiles" />;
  if (error || !data) return <ErrorState message={error ?? "SG distributions are unavailable."} />;
  const rows = mode === "adjusted" ? data.adjusted : data.raw;
  const options = uniqueStrings(rows, "player_name");
  const activePlayers = players.length ? players : options.slice(0, 2);
  const selected = rows.filter((row) => activePlayers.includes(String(row.player_name)));
  const categories = ["sg_ott", "sg_app", "sg_arg", "sg_putt"];
  const chartRows = categories.map((category) => {
    const point: DataRow = { category: titleCase(category.replace("sg_", "")) };
    activePlayers.forEach((player) => {
      const row = selected.find((item) => String(item.player_name) === player && String(item.category_clean) === category);
      point[player] = numberValue(row?.mean);
    });
    return point;
  });
  const predictionMap = new Map(data.predictions.filter((row) => numberValue(row.round) === 1).map((row) => [String(row.player_name), numberValue(row.my_pred)]));

  return (
    <>
      <PageIntro eyebrow="Skill inputs" title="Strokes-gained distributions" description="Inspect the category profile, variance, skew, and sample depth feeding the tournament model." controls={<SegmentedControl label="SG distribution mode" value={mode} onChange={setMode} options={[{ value: "adjusted", label: "Course adjusted" }, { value: "raw", label: "Raw EMA" }]} />} />
      <Panel title="Category profile" eyebrow={mode === "adjusted" ? "Course-adjusted means" : "Raw weighted means"}>
        <PlayerPicker options={options} value={activePlayers} onChange={setPlayers} max={5} />
        <div className="chart-medium">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartRows} margin={chartMargin}>
              <CartesianGrid stroke="var(--line)" vertical={false} />
              <XAxis dataKey="category" stroke="var(--muted)" />
              <YAxis stroke="var(--muted)" />
              <Tooltip content={<ChartTooltip />} />
              <Legend formatter={(value) => titleCase(value)} />
              <ReferenceLine y={0} stroke="var(--line-strong)" />
              {activePlayers.map((player, index) => <Bar key={player} dataKey={player} fill={palette[index % palette.length]} radius={[5, 5, 0, 0]} />)}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Panel>
      <div className="comparison-grid">
        {activePlayers.map((player) => {
          const playerRows = selected.filter((row) => String(row.player_name) === player);
          return <Panel key={player} title={titleCase(player)} eyebrow={`R1 prediction ${predictionMap.get(player)?.toFixed(2) ?? "—"} SG`}><div className="mini-stat-grid"><div><span>Total mean</span><strong>{sum(playerRows.map((row) => numberValue(row.mean))).toFixed(2)}</strong></div><div><span>Avg std dev</span><strong>{safeMean(playerRows.map((row) => numberValue(row.std))).toFixed(2)}</strong></div><div><span>Effective sample</span><strong>{safeMean(playerRows.map((row) => numberValue(row.n_eff))).toFixed(0)}</strong></div><div><span>Avg skew</span><strong>{safeMean(playerRows.map((row) => numberValue(row.skew))).toFixed(2)}</strong></div></div></Panel>;
        })}
      </div>
      <Panel title="Distribution moments" eyebrow="Fully customizable columns"><DataTable rows={selected} label="SG moments" preferredColumns={["player_name", "category_clean", "mean", "std", "skew", "excess_kurtosis", "n_eff", "distribution_source", "course_history_category_adjustment"]} /></Panel>
    </>
  );
}

type RoundPayload = { rounds: Record<string, DataRow[]> };

export function RoundScoresView() {
  const { data, loading, error } = useDashboardData<RoundPayload>("round-scores.json");
  const [round, setRound] = useState("1");
  const [players, setPlayers] = useState<string[]>([]);
  if (loading) return <LoadingState label="Loading simulated round scores" />;
  if (error || !data) return <ErrorState message={error ?? "Round score data is unavailable."} />;
  const rounds = Object.keys(data.rounds).sort();
  const activeRound = data.rounds[round] ? round : rounds[0];
  const rows = data.rounds[activeRound] ?? [];
  const options = uniqueStrings(rows, "player_name").sort((a, b) => {
    const mean = (player: string) => sum(rows.filter((row) => String(row.player_name) === player).map((row) => numberValue(row.score) * numberValue(row.prob)));
    return mean(a) - mean(b);
  });
  const activePlayers = players.length ? players : options.slice(0, 2);
  const scoreValues = [...new Set(rows.filter((row) => activePlayers.includes(String(row.player_name))).map((row) => numberValue(row.score)))].sort((a, b) => a - b);
  const chartRows = scoreValues.map((score) => {
    const point: DataRow = { score };
    activePlayers.forEach((player) => {
      const row = rows.find((item) => String(item.player_name) === player && numberValue(item.score) === score);
      point[player] = numberValue(row?.prob) * 100;
    });
    return point;
  });
  const stats = activePlayers.map((player) => {
    const playerRows = rows.filter((row) => String(row.player_name) === player);
    const mean = sum(playerRows.map((row) => numberValue(row.score) * numberValue(row.prob)));
    const variance = sum(playerRows.map((row) => numberValue(row.prob) * (numberValue(row.score) - mean) ** 2));
    return { player_name: player, mean, std: Math.sqrt(variance), ...(playerRows[0] ?? {}) };
  });
  let matchup: { a: number; tie: number; b: number } | null = null;
  if (activePlayers.length === 2) {
    const aRows = rows.filter((row) => String(row.player_name) === activePlayers[0]);
    const bRows = rows.filter((row) => String(row.player_name) === activePlayers[1]);
    let a = 0; let tie = 0;
    aRows.forEach((left) => bRows.forEach((right) => {
      const joint = numberValue(left.prob) * numberValue(right.prob);
      if (numberValue(left.score) < numberValue(right.score)) a += joint;
      else if (numberValue(left.score) === numberValue(right.score)) tie += joint;
    }));
    matchup = { a, tie, b: Math.max(0, 1 - a - tie) };
  }

  return (
    <>
      <PageIntro eyebrow="Round model" title="Round score distributions" description="Compare the full simulated score range, expected score, category means, and head-to-head win probability." controls={<SelectControl label="Round" value={activeRound} onChange={setRound} options={rounds.map((value) => ({ value, label: `Round ${value}` }))} />} />
      <Panel title={`Round ${activeRound} scoring curves`} eyebrow="Probability by integer score">
        <PlayerPicker options={options} value={activePlayers} onChange={setPlayers} max={4} />
        <div className="chart-large"><ResponsiveContainer width="100%" height="100%"><AreaChart data={chartRows} margin={chartMargin}><CartesianGrid stroke="var(--line)" vertical={false}/><XAxis dataKey="score" stroke="var(--muted)"/><YAxis stroke="var(--muted)" tickFormatter={(value) => `${value}%`}/><Tooltip content={<ChartTooltip />}/><Legend formatter={(value) => titleCase(value)}/>{activePlayers.map((player, index) => <Area key={player} type="monotone" dataKey={player} stroke={palette[index % palette.length]} fill={palette[index % palette.length]} fillOpacity={0.16} strokeWidth={2}/>)}</AreaChart></ResponsiveContainer></div>
      </Panel>
      {matchup && <Panel title="Head-to-head" eyebrow="No-tie fair prices"><div className="matchup-card"><div><span>{titleCase(activePlayers[0])}</span><strong>{(matchup.a * 100).toFixed(1)}%</strong><small>{americanOdds(matchup.a / (matchup.a + matchup.b))}</small></div><div><span>Tie</span><strong>{(matchup.tie * 100).toFixed(1)}%</strong></div><div><span>{titleCase(activePlayers[1])}</span><strong>{(matchup.b * 100).toFixed(1)}%</strong><small>{americanOdds(matchup.b / (matchup.a + matchup.b))}</small></div></div></Panel>}
      <Panel title="Player summary" eyebrow="Expected score and SG category profile"><DataTable rows={stats} label="Round score summary" preferredColumns={["player_name", "mean", "std", "sg_ott", "sg_app", "sg_arg", "sg_putt", "expected_avg"]} /></Panel>
    </>
  );
}

type HistoryManifest = { events: Array<{ event_id: number; event_name: string; key: string; modes: string[] }> };
type HistoryRows = { rows: DataRow[] };

export function HistoryView() {
  const manifest = useDashboardData<HistoryManifest>("history.json");
  const [eventKey, setEventKey] = useState("");
  const [mode, setMode] = useState<"pre" | "live">("pre");
  const events = manifest.data?.events ?? [];
  const activeEventKey = eventKey || events[0]?.key || "";
  const selectedEvent = events.find((event) => event.key === activeEventKey);
  const activeMode = selectedEvent?.modes.includes(mode) ? mode : (selectedEvent?.modes[0] as "pre" | "live" | undefined) ?? "pre";
  const history = useDashboardData<HistoryRows>(activeEventKey ? `history/${activeEventKey}-${activeMode}.json` : "history.json");
  if (manifest.loading) return <LoadingState label="Loading historical event index" />;
  if (manifest.error || !manifest.data) return <ErrorState message={manifest.error ?? "Historical data is unavailable."} />;
  return (
    <>
      <PageIntro eyebrow="Model archive" title="Historical distributions" description="Reopen the exact pre-event or live distribution from prior tournaments without loading the weekly pipeline." controls={<div className="control-row"><SelectControl label="Event" value={activeEventKey} onChange={setEventKey} options={events.map((event) => ({ value: event.key, label: `${titleCase(event.event_name)} · ${event.event_id}` }))}/><SegmentedControl label="Historical mode" value={activeMode} onChange={setMode} options={[{ value: "pre", label: "Pre-event" }, { value: "live", label: "Live" }]}/></div>} />
      {history.loading && <LoadingState label="Loading archived distribution" />}
      {history.error && <ErrorState message={history.error} />}
      {history.data?.rows && <DistributionExplorer rows={history.data.rows} title={titleCase(selectedEvent?.event_name)} subtitle={`${activeMode === "live" ? "Live" : "Pre-event"} archived finish probabilities`} />}
    </>
  );
}

type PerformancePayload = { bets: DataRow[] };

function unitsForBet(row: DataRow): number {
  const stored = numberValue(row.units_won, Number.NaN);
  if (Number.isFinite(stored)) return stored;
  const result = String(row.result ?? "").toLowerCase();
  if (!result || result === "push") return 0;
  const stake = String(row.bet_type ?? "").startsWith("finish_position") ? numberValue(row.kelly_stake) / 200 : 1;
  if (result === "loss") return -stake;
  const odds = numberValue(row.book_odds);
  const decimal = odds >= 0 ? odds / 100 + 1 : 100 / Math.abs(odds) + 1;
  return stake * (decimal - 1);
}

function buildCumulativeCurve(rows: DataRow[]) {
  let running = 0;
  return rows.map((row, index) => {
    running += unitsForBet(row);
    return { index: index + 1, units: running };
  });
}

export function PerformanceView() {
  const { data, loading, error } = useDashboardData<PerformancePayload>("performance.json");
  const [event, setEvent] = useState("all");
  const [type, setType] = useState("all");
  const [book, setBook] = useState("all");
  const [minEdge, setMinEdge] = useState(0);
  if (loading) return <LoadingState label="Loading performance history" />;
  if (error || !data) return <ErrorState message={error ?? "Performance data is unavailable."} />;
  const events = uniqueStrings(data.bets, "event_name");
  const types = uniqueStrings(data.bets, "bet_type");
  const books = uniqueStrings(data.bets, "bookmaker");
  const filtered = data.bets.filter((row) => (event === "all" || String(row.event_name) === event) && (type === "all" || String(row.bet_type) === type) && (book === "all" || String(row.bookmaker) === book) && numberValue(row.edge) >= minEdge);
  const resolved = filtered.filter((row) => String(row.result ?? "").trim());
  const ordered = [...resolved].sort((a, b) => String(a.run_timestamp).localeCompare(String(b.run_timestamp)));
  const curve = buildCumulativeCurve(ordered);
  const wagered = resolved.reduce((total, row) => total + (String(row.bet_type).startsWith("finish_position") ? Math.max(0.005, numberValue(row.kelly_stake) / 200) : 1), 0);
  const wins = resolved.filter((row) => String(row.result).startsWith("win")).length;
  const totalUnits = sum(resolved.map(unitsForBet));
  const byBook = books.map((name) => {
    const rows = resolved.filter((row) => String(row.bookmaker) === name);
    const won = sum(rows.map(unitsForBet));
    return { book: titleCase(name), units: won, bets: rows.length };
  }).filter((row) => row.bets).sort((a, b) => b.units - a.units);

  return (
    <>
      <PageIntro eyebrow="Bet review" title="Performance" description="A filterable, exportable record of historical results with cumulative P&L and book-level attribution." controls={<div className="control-row wrap"><SelectControl label="Event" value={event} onChange={setEvent} options={[{ value: "all", label: "All events" }, ...events.map((value) => ({ value, label: titleCase(value) }))]}/><SelectControl label="Bet type" value={type} onChange={setType} options={[{ value: "all", label: "All bet types" }, ...types.map((value) => ({ value, label: titleCase(value) }))]}/><SelectControl label="Book" value={book} onChange={setBook} options={[{ value: "all", label: "All books" }, ...books.map((value) => ({ value, label: titleCase(value) }))]}/><RangeControl label="Minimum edge" value={minEdge} min={0} max={25} onChange={setMinEdge}/></div>} />
      <div className="kpi-grid"><Kpi label="Net P&L" value={`${totalUnits >= 0 ? "+" : ""}${totalUnits.toFixed(1)}u`} detail={`${resolved.length.toLocaleString()} resolved bets`} tone={totalUnits >= 0 ? "positive" : "negative"}/><Kpi label="ROI" value={`${wagered ? (totalUnits / wagered * 100).toFixed(1) : "0.0"}%`} detail={`${wagered.toFixed(1)} units risked`} tone={totalUnits >= 0 ? "positive" : "negative"}/><Kpi label="Win rate" value={`${resolved.length ? (wins / resolved.length * 100).toFixed(1) : "0.0"}%`} detail={`${wins} wins`}/><Kpi label="Open positions" value={String(filtered.length - resolved.length)} detail="Awaiting grade"/></div>
      <div className="two-column">
        <Panel title="Cumulative P&L" eyebrow="Resolved bets in time order"><div className="chart-medium"><ResponsiveContainer width="100%" height="100%"><LineChart data={curve} margin={chartMargin}><CartesianGrid stroke="var(--line)" vertical={false}/><XAxis dataKey="index" stroke="var(--muted)"/><YAxis stroke="var(--muted)"/><Tooltip content={<ChartTooltip/>}/><ReferenceLine y={0} stroke="var(--line-strong)"/><Line type="monotone" dataKey="units" stroke="var(--accent)" dot={false} strokeWidth={2.5}/></LineChart></ResponsiveContainer></div></Panel>
        <Panel title="P&L by sportsbook" eyebrow="Current filters"><div className="chart-medium"><ResponsiveContainer width="100%" height="100%"><BarChart data={byBook.slice(0, 12)} layout="vertical" margin={{ ...chartMargin, left: 30 }}><CartesianGrid stroke="var(--line)" horizontal={false}/><XAxis type="number" stroke="var(--muted)"/><YAxis type="category" dataKey="book" stroke="var(--muted)" width={90}/><Tooltip content={<ChartTooltip/>}/><ReferenceLine x={0} stroke="var(--line-strong)"/><Bar dataKey="units" radius={[0, 6, 6, 0]}>{byBook.slice(0, 12).map((row) => <Cell key={row.book} fill={row.units >= 0 ? "var(--positive)" : "var(--negative)"}/>)}</Bar></BarChart></ResponsiveContainer></div></Panel>
      </div>
      <Panel title="Bet history" eyebrow="Search, sort, customize, export"><DataTable rows={filtered} label="Performance history" preferredColumns={["run_timestamp", "event_name", "bet_type", "round", "bet_on", "opponent", "bookmaker", "book_odds", "edge", "result", "units_won"]} pageSize={40}/></Panel>
    </>
  );
}

type DiagnosticPayload = { sg: DataRow[]; market_regression: DataRow[] };

export function DiagnosticsView() {
  const { data, loading, error } = useDashboardData<DiagnosticPayload>("diagnostics.json");
  const [event, setEvent] = useState("all");
  const [player, setPlayer] = useState("");
  if (loading) return <LoadingState label="Loading model diagnostics" />;
  if (error || !data) return <ErrorState message={error ?? "Diagnostics are unavailable."} />;
  const events = [...new Map(data.sg.map((row) => [String(row.event_id), String(row.event_name)])).entries()].sort((a, b) => numberValue(b[0]) - numberValue(a[0]));
  const rows = data.sg.filter((row) => event === "all" || String(row.event_id) === event);
  const players = uniqueStrings(rows, "player_name");
  const categories = ["ott", "app", "arg", "putt"];
  const bias = categories.map((category) => {
    const categoryRows = rows.filter((row) => String(row.category) === category);
    return { category: category.toUpperCase(), miss: safeMean(categoryRows.map((row) => numberValue(row.miss))), centered: safeMean(categoryRows.map((row) => numberValue(row.miss_centered))), sample: sum(categoryRows.map((row) => numberValue(row.rounds))) };
  });
  const archetypes = uniqueStrings(rows, "archetype").map((archetype) => {
    const archetypeRows = rows.filter((row) => String(row.archetype) === archetype);
    const point: DataRow = { archetype: titleCase(archetype) };
    categories.forEach((category) => point[category] = safeMean(archetypeRows.filter((row) => String(row.category) === category).map((row) => numberValue(row.miss_centered))));
    return point;
  });
  const recurring = players.map((name) => {
    const playerRows = rows.filter((row) => String(row.player_name) === name);
    const point: DataRow = { player_name: name, rounds: sum(playerRows.map((row) => numberValue(row.rounds))) };
    categories.forEach((category) => point[category] = safeMean(playerRows.filter((row) => String(row.category) === category).map((row) => numberValue(row.miss_centered))));
    point.max_abs_miss = Math.max(...categories.map((category) => Math.abs(numberValue(point[category]))));
    return point;
  }).sort((a, b) => numberValue(b.max_abs_miss) - numberValue(a.max_abs_miss));
  const selectedRows = rows.filter((row) => !player || String(row.player_name) === player);

  return (
    <>
      <PageIntro eyebrow="Model QA" title="Diagnostics" description="Field-adjusted prediction misses, archetype tilt, recurring player patterns, and market-regression checks in one place." controls={<div className="control-row"><SelectControl label="Event" value={event} onChange={(value) => { setEvent(value); setPlayer(""); }} options={[{ value: "all", label: "All adjusted events" }, ...events.map(([value, label]) => ({ value, label: `${titleCase(label)} · ${value}` }))]}/><SelectControl label="Player detail" value={player} onChange={setPlayer} options={[{ value: "", label: "All players" }, ...players.map((value) => ({ value, label: titleCase(value) }))]}/></div>} />
      <div className="kpi-grid">{bias.map((row) => <Kpi key={row.category} label={`${row.category} level bias`} value={`${row.miss >= 0 ? "+" : ""}${row.miss.toFixed(3)}`} detail={`${row.sample.toLocaleString()} rounds`} tone={Math.abs(row.miss) < 0.05 ? "positive" : Math.abs(row.miss) > 0.15 ? "negative" : "neutral"}/>)}</div>
      <div className="two-column">
        <Panel title="Category level bias" eyebrow="Actual minus predicted"><div className="chart-medium"><ResponsiveContainer width="100%" height="100%"><BarChart data={bias} margin={chartMargin}><CartesianGrid stroke="var(--line)" vertical={false}/><XAxis dataKey="category" stroke="var(--muted)"/><YAxis stroke="var(--muted)"/><Tooltip content={<ChartTooltip/>}/><ReferenceLine y={0} stroke="var(--line-strong)"/><Bar dataKey="miss" radius={[6, 6, 0, 0]}>{bias.map((row) => <Cell key={row.category} fill={row.miss >= 0 ? "var(--positive)" : "var(--negative)"}/>)}</Bar></BarChart></ResponsiveContainer></div></Panel>
        <Panel title="Archetype tilt" eyebrow="Field-centered category misses"><div className="chart-medium"><ResponsiveContainer width="100%" height="100%"><BarChart data={archetypes.slice(0, 10)} margin={chartMargin}><CartesianGrid stroke="var(--line)" vertical={false}/><XAxis dataKey="archetype" stroke="var(--muted)" angle={-20} textAnchor="end" height={70}/><YAxis stroke="var(--muted)"/><Tooltip content={<ChartTooltip/>}/><Legend/>{categories.map((category, index) => <Bar key={category} dataKey={category} fill={palette[index]} radius={[3,3,0,0]}/>)}</BarChart></ResponsiveContainer></div></Panel>
      </div>
      {player && <Panel title={titleCase(player)} eyebrow="Event/category prediction detail"><DataTable rows={selectedRows} label="Player diagnostics" preferredColumns={["event_name", "category", "predicted_sg", "actual_sg", "miss", "miss_centered", "rounds", "archetype"]}/></Panel>}
      <Panel title="Largest recurring misses" eyebrow="Players ranked by maximum category miss"><DataTable rows={recurring} label="Recurring misses" preferredColumns={["player_name", "archetype", "ott", "app", "arg", "putt", "max_abs_miss", "rounds"]} pageSize={30}/></Panel>
      <Panel title="Market regression" eyebrow="Prediction adjustment versus actual SG"><DataTable rows={data.market_regression} label="Market regression" preferredColumns={["event_name", "player_name", "pred", "my_pred_regressed", "actual_sg", "error_raw", "error_regressed", "regress_helped", "mkt_adj", "mu_adj"]}/></Panel>
    </>
  );
}

type WeatherPayload = { forecast: Record<string, unknown>; players: DataRow[]; matchups: DataRow[] };

export function WeatherView() {
  const { data, loading, error } = useDashboardData<WeatherPayload>("weather.json");
  if (loading) return <LoadingState label="Loading weather impact" />;
  if (error || !data) return <ErrorState message={error ?? "Weather data is unavailable."} />;
  const hours = Array.from({ length: Math.max(0, ...[1,2,3,4].flatMap((round) => [Array.isArray(data.forecast[`wind_r${round}`]) ? (data.forecast[`wind_r${round}`] as unknown[]).length : 0, Array.isArray(data.forecast[`dew_r${round}`]) ? (data.forecast[`dew_r${round}`] as unknown[]).length : 0])) }, (_, index) => {
    const point: DataRow = { hour: index + 6 };
    for (let round = 1; round <= 4; round += 1) {
      const wind = data.forecast[`wind_r${round}`]; const dew = data.forecast[`dew_r${round}`];
      if (Array.isArray(wind)) point[`R${round} wind`] = numberValue(wind[index]);
      if (Array.isArray(dew)) point[`R${round} dew`] = numberValue(dew[index]);
    }
    return point;
  });
  const impactColumns = ["wind_adv_r1_2", "dew_adv_r1_2", "total_weather_adv"].filter((column) => data.players.some((row) => row[column] !== undefined));
  const playerBars = [...data.players].map((row) => ({ ...row, total: impactColumns.reduce((total, column) => total + numberValue(row[column]), 0) })).sort((a, b) => Math.abs(numberValue(b.total)) - Math.abs(numberValue(a.total))).slice(0, 20);
  const maxWind = Math.max(0, ...hours.flatMap((row) => Object.entries(row).filter(([key]) => key.includes("wind")).map(([, value]) => numberValue(value))));
  const maxDew = Math.max(0, ...hours.flatMap((row) => Object.entries(row).filter(([key]) => key.includes("dew")).map(([, value]) => numberValue(value))));

  return (
    <>
      <PageIntro eyebrow="Course conditions" title="Weather" description="Forecast windows, field impact, and the matchups most exposed to wind and dewpoint differences." />
      <div className="kpi-grid"><Kpi label="Peak forecast wind" value={hours.length ? `${maxWind.toFixed(1)} mph` : "Not loaded"} detail="Across published round windows"/><Kpi label="Peak dewpoint" value={hours.length ? `${maxDew.toFixed(1)}°` : "Not loaded"} detail="Published hourly maximum"/><Kpi label="Players modeled" value={String(data.players.length)} detail="With weather adjustments" tone="accent"/><Kpi label="Weather matchups" value={String(data.matchups.length)} detail="With a computed differential"/></div>
      {hours.length > 0 ? <div className="two-column"><Panel title="Wind forecast" eyebrow="Hourly by round"><div className="chart-medium"><ResponsiveContainer width="100%" height="100%"><LineChart data={hours} margin={chartMargin}><CartesianGrid stroke="var(--line)" vertical={false}/><XAxis dataKey="hour" stroke="var(--muted)" tickFormatter={(value) => `${value}:00`}/><YAxis stroke="var(--muted)"/><Tooltip content={<ChartTooltip/>}/><Legend/>{[1,2,3,4].map((round, index) => <Line key={round} dataKey={`R${round} wind`} stroke={palette[index]} dot={false} strokeWidth={2}/>)}</LineChart></ResponsiveContainer></div></Panel><Panel title="Dewpoint forecast" eyebrow="Hourly by round"><div className="chart-medium"><ResponsiveContainer width="100%" height="100%"><LineChart data={hours} margin={chartMargin}><CartesianGrid stroke="var(--line)" vertical={false}/><XAxis dataKey="hour" stroke="var(--muted)" tickFormatter={(value) => `${value}:00`}/><YAxis stroke="var(--muted)"/><Tooltip content={<ChartTooltip/>}/><Legend/>{[1,2,3,4].map((round, index) => <Line key={round} dataKey={`R${round} dew`} stroke={palette[index]} dot={false} strokeWidth={2}/>)}</LineChart></ResponsiveContainer></div></Panel></div> : <EmptyState title="No hourly forecast is currently published" detail="The player-impact data below remains available from the most recent simulation." />}
      <Panel title="Largest player weather impacts" eyebrow="Absolute adjustment leaders"><div className="chart-large"><ResponsiveContainer width="100%" height="100%"><BarChart data={playerBars} layout="vertical" margin={{ ...chartMargin, left: 75 }}><CartesianGrid stroke="var(--line)" horizontal={false}/><XAxis type="number" stroke="var(--muted)"/><YAxis type="category" dataKey="player_name" stroke="var(--muted)" width={120} tickFormatter={(value) => titleCase(value)}/><Tooltip content={<ChartTooltip/>}/><ReferenceLine x={0} stroke="var(--line-strong)"/><Bar dataKey="total" radius={[0,6,6,0]}>{playerBars.map((row, index) => <Cell key={index} fill={numberValue(row.total) >= 0 ? "var(--negative)" : "var(--positive)"}/>)}</Bar></BarChart></ResponsiveContainer></div></Panel>
      <div className="stack-lg"><Panel title="Weather matchup discrepancies" eyebrow="Positive differential favors Player 1"><DataTable rows={data.matchups} label="Weather matchups" preferredColumns={["player_1", "player_2", "p1_weather_adv", "p2_weather_adv", "differential", "favored"]}/></Panel><Panel title="Field weather adjustments" eyebrow="Sortable player-level detail"><DataTable rows={data.players} label="Weather adjustments" preferredColumns={["player_name", "wind_adv_r1_2", "dew_adv_r1_2", "wind_adv_r3_4", "dew_adv_r3_4"]} pageSize={35}/></Panel></div>
    </>
  );
}
