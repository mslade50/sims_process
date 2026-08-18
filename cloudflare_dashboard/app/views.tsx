"use client";

import { useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
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

type FilterOption = { value: string; label: string };

function MultiSelectControl({ label, value, options, onChange, placeholder = "All" }: { label: string; value: string[]; options: FilterOption[]; onChange: (value: string[]) => void; placeholder?: string }) {
  const selectedLabel = value.length === 0
    ? placeholder
    : value.length === 1
      ? options.find((option) => option.value === value[0])?.label ?? titleCase(value[0])
      : `${value.length} selected`;
  return (
    <div className="filter-control multi-select-control">
      <span>{label}</span>
      <details>
        <summary>{selectedLabel}</summary>
        <div className="multi-select-menu">
          {options.map((option) => (
            <label key={option.value}>
              <input
                type="checkbox"
                checked={value.includes(option.value)}
                onChange={(event) => onChange(event.target.checked ? [...value, option.value] : value.filter((item) => item !== option.value))}
              />
              <span className="checkbox-mark">✓</span>
              <span>{option.label}</span>
            </label>
          ))}
          {value.length > 0 && <button type="button" onClick={() => onChange([])}>Clear selection</button>}
        </div>
      </details>
    </div>
  );
}

function NumberRangeControl({ label, minValue, maxValue, onMinChange, onMaxChange, min, step = 1 }: { label: string; minValue: number | null; maxValue: number | null; onMinChange: (value: number | null) => void; onMaxChange: (value: number | null) => void; min?: number; step?: number }) {
  const update = (value: string, callback: (next: number | null) => void) => callback(value === "" ? null : Number(value));
  return (
    <div className="filter-control">
      <span>{label}</span>
      <div className="number-range">
        <input type="number" min={min} step={step} value={minValue ?? ""} placeholder="Min" aria-label={`${label} minimum`} onChange={(event) => update(event.target.value, onMinChange)} />
        <input type="number" min={min} step={step} value={maxValue ?? ""} placeholder="Max" aria-label={`${label} maximum`} onChange={(event) => update(event.target.value, onMaxChange)} />
      </div>
    </div>
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
type PerformanceRow = DataRow & {
  _units_wagered: number;
  _units_won: number;
  raw_edge: number | null;
  dec_odds: number | null;
  archetype: string;
  archetype_against: string;
  bet_day: string;
};

const SHARP_BOOKS = ["pinnacle", "betonline", "betcris"];
const EXCHANGE_BOOKS = ["kalshi", "novig"];
const PERFORMANCE_TYPE_OPTIONS: FilterOption[] = [
  { value: "tournament_matchup", label: "Tournament Matchup" },
  { value: "round_matchup", label: "Round Matchup" },
  { value: "finish_position", label: "Finish Position" },
  { value: "finish_position_live", label: "Finish Position (Live)" },
  { value: "score_bet", label: "Score Bet" },
];
const PERFORMANCE_TYPE_COLORS: Record<string, string> = {
  tournament_matchup: "#54d6c8",
  round_matchup: "#ffba69",
  finish_position: "#8ca7ff",
  finish_position_live: "#f27ea9",
  score_bet: "#a58cff",
};

function americanToProbability(value: unknown): number {
  const odds = numberValue(value, Number.NaN);
  if (!Number.isFinite(odds) || odds === 0) return Number.NaN;
  return odds >= 0 ? 100 / (odds + 100) : Math.abs(odds) / (Math.abs(odds) + 100);
}

function americanToDecimal(value: unknown): number {
  const odds = numberValue(value, Number.NaN);
  if (!Number.isFinite(odds) || odds === 0) return Number.NaN;
  return odds >= 0 ? odds / 100 + 1 : 100 / Math.abs(odds) + 1;
}

function unitsWagered(row: DataRow): number {
  const betType = String(row.bet_type ?? "");
  if (betType.startsWith("finish_position")) return Math.max(0, numberValue(row.kelly_stake) / 200);
  const odds = numberValue(row.book_odds, Number.NaN);
  return Number.isFinite(odds) && odds <= -100 ? Math.abs(odds) / 100 : 1;
}

function enrichPerformanceRows(rows: DataRow[]): PerformanceRow[] {
  const archetypeLookup = new Map<string, string>();
  rows.forEach((row) => {
    const archetype = String(row.type_on ?? "").trim();
    if (archetype) archetypeLookup.set(`${String(row.event_id)}|${String(row.bet_on).trim().toLowerCase()}`, archetype);
  });
  return rows.map((row) => {
    const result = String(row.result ?? "").trim().toLowerCase();
    const stake = unitsWagered(row);
    const decimal = americanToDecimal(row.book_odds);
    const storedWon = numberValue(row.units_won, Number.NaN);
    const calculatedWon = result === "loss" ? -stake : result.startsWith("win") && Number.isFinite(decimal) ? stake * (decimal - 1) : 0;
    const marketProbability = americanToProbability(row.book_odds);
    const modelProbability = americanToProbability(row.fair_odds);
    const rawEdge = Number.isFinite(marketProbability) && Number.isFinite(modelProbability) ? (modelProbability - marketProbability) * 100 : null;
    const eventKey = String(row.event_id);
    const betOn = String(row.bet_on ?? "").trim().toLowerCase();
    const opponent = String(row.opponent ?? "").trim().toLowerCase();
    const stamped = String(row.type_on ?? "").trim();
    const date = new Date(`${String(row.run_timestamp ?? "").replace(" ", "T")}Z`);
    return {
      ...row,
      result,
      raw_edge: rawEdge,
      dec_odds: Number.isFinite(decimal) ? decimal : null,
      archetype: stamped || archetypeLookup.get(`${eventKey}|${betOn}`) || "Unknown",
      archetype_against: archetypeLookup.get(`${eventKey}|${opponent}`) || "",
      bet_day: Number.isNaN(date.getTime()) ? "" : date.toLocaleDateString("en-US", { weekday: "long", timeZone: "UTC" }),
      _units_wagered: stake,
      _units_won: Number.isFinite(storedWon) ? storedWon : calculatedWon,
    };
  });
}

function buildCumulativeCurve(rows: PerformanceRow[]) {
  let running = 0;
  return rows.map((row, index) => {
    running += row._units_won;
    return { index: index + 1, units: running };
  });
}

function bucketRoi(rows: PerformanceRow[], group: string, color: string, buckets: Array<{ label: string; min: number; max: number }>, getter: (row: PerformanceRow) => number): DataRow[] {
  return buckets.map((bucket) => {
    const selected = rows.filter((row) => {
      const value = getter(row);
      return Number.isFinite(value) && value >= bucket.min && value < bucket.max;
    });
    const wagered = sum(selected.map((row) => row._units_wagered));
    return { bucket: `${group}: ${bucket.label}`, roi: wagered ? sum(selected.map((row) => row._units_won)) / wagered * 100 : 0, bets: selected.length, color };
  });
}

export function PerformanceView() {
  const { data, loading, error } = useDashboardData<PerformancePayload>("performance.json");
  const [eventsSelected, setEventsSelected] = useState<string[]>([]);
  const [typesSelected, setTypesSelected] = useState<string[]>([]);
  const [booksSelected, setBooksSelected] = useState<string[]>([]);
  const [minEdge, setMinEdge] = useState(0);
  const [includeLive, setIncludeLive] = useState(false);
  const [roundsSelected, setRoundsSelected] = useState<string[]>([]);
  const [daysSelected, setDaysSelected] = useState<string[]>([]);
  const [sampleMin, setSampleMin] = useState<number | null>(null);
  const [sampleMax, setSampleMax] = useState<number | null>(null);
  const [predMin, setPredMin] = useState<number | null>(null);
  const [predMax, setPredMax] = useState<number | null>(null);
  const [marketsSelected, setMarketsSelected] = useState<string[]>([]);
  const [side, setSide] = useState("all");
  const [analysisMode, setAnalysisMode] = useState("all");
  const [rawEdgeMin, setRawEdgeMin] = useState<number | null>(null);
  const [rawEdgeMax, setRawEdgeMax] = useState<number | null>(null);
  const [decimalMin, setDecimalMin] = useState<number | null>(null);
  const [decimalMax, setDecimalMax] = useState<number | null>(null);
  const [archetypesSelected, setArchetypesSelected] = useState<string[]>([]);
  const [againstSelected, setAgainstSelected] = useState<string[]>([]);
  const [playersSelected, setPlayersSelected] = useState<string[]>([]);

  const rows = useMemo(() => enrichPerformanceRows(data?.bets ?? []), [data]);
  const events = useMemo(() => uniqueStrings(rows, "event_name"), [rows]);
  const books = useMemo(() => uniqueStrings(rows, "bookmaker"), [rows]);
  const players = useMemo(() => uniqueStrings(rows, "bet_on"), [rows]);
  const archetypes = useMemo(() => uniqueStrings(rows, "archetype"), [rows]);
  const archetypesAgainst = useMemo(() => uniqueStrings(rows, "archetype_against"), [rows]);

  const filtered = useMemo(() => {
    const showLive = includeLive || typesSelected.includes("finish_position_live");
    const explicitlySelectedBooks = booksSelected.map((value) => value.toLowerCase());
    let selected = rows.filter((row) => {
      const betType = String(row.bet_type ?? "");
      const bookmaker = String(row.bookmaker ?? "").toLowerCase();
      const result = String(row.result ?? "").trim().toLowerCase();
      if (result === "duplicate") return false;
      if (eventsSelected.length && !eventsSelected.includes(String(row.event_name))) return false;
      if (typesSelected.length ? !typesSelected.includes(betType) : betType === "score_bet") return false;
      if (!showLive && betType === "finish_position_live") return false;
      if (booksSelected.length && !booksSelected.includes(String(row.bookmaker))) return false;
      const hiddenExchange = EXCHANGE_BOOKS.some((exchange) => bookmaker.includes(exchange) && !explicitlySelectedBooks.some((selectedBook) => selectedBook.includes(exchange)));
      if (hiddenExchange && !(showLive && betType === "finish_position_live")) return false;
      if (numberValue(row.edge) < minEdge) return false;
      if (roundsSelected.length && !roundsSelected.includes(String(row.round).trim())) return false;
      if (daysSelected.length && !daysSelected.includes(row.bet_day)) return false;
      if (sampleMin !== null && numberValue(row.sample_on) < sampleMin) return false;
      if (sampleMax !== null && numberValue(row.sample_on) > sampleMax) return false;
      if (predMin !== null && numberValue(row.pred_on) < predMin) return false;
      if (predMax !== null && numberValue(row.pred_on) > predMax) return false;
      if (rawEdgeMin !== null && numberValue(row.raw_edge, Number.NEGATIVE_INFINITY) < rawEdgeMin) return false;
      if (rawEdgeMax !== null && numberValue(row.raw_edge, Number.POSITIVE_INFINITY) > rawEdgeMax) return false;
      if (decimalMin !== null && numberValue(row.dec_odds, Number.NEGATIVE_INFINITY) < decimalMin) return false;
      if (decimalMax !== null && numberValue(row.dec_odds, Number.POSITIVE_INFINITY) > decimalMax) return false;
      if (archetypesSelected.length && !archetypesSelected.includes(row.archetype)) return false;
      if (againstSelected.length && !againstSelected.includes(row.archetype_against)) return false;
      if (playersSelected.length && !playersSelected.includes(String(row.bet_on))) return false;
      if ((analysisMode === "sharp_only" || analysisMode === "sharp_best") && !SHARP_BOOKS.some((sharp) => bookmaker.includes(sharp))) return false;
      if (side !== "all") {
        const isFinish = betType.startsWith("finish_position");
        const isNo = String(row.opponent ?? "").trim().toLowerCase().endsWith("_no");
        if (!isFinish || (side === "no" ? !isNo : isNo)) return false;
      }
      if (marketsSelected.length) {
        if (!betType.startsWith("finish_position")) return false;
        const market = String(row.opponent ?? "").trim().toLowerCase().replace(/_no$/, "");
        const normalized = market === "winner" ? "win" : market;
        if (!marketsSelected.includes(normalized)) return false;
      }
      return true;
    });
    if (analysisMode === "best_price" || analysisMode === "sharp_best") {
      const grouped = new Map<string, PerformanceRow[]>();
      selected.forEach((row) => {
        const key = [row.event_id, row.bet_type, row.round, row.bet_on, row.opponent].join("|");
        grouped.set(key, [...(grouped.get(key) ?? []), row]);
      });
      selected = [...grouped.values()].map((group) => {
        const earliest = group.reduce((value, row) => String(row.run_timestamp) < value ? String(row.run_timestamp) : value, String(group[0]?.run_timestamp ?? ""));
        return group.filter((row) => String(row.run_timestamp) === earliest).sort((a, b) => numberValue(b.edge) - numberValue(a.edge))[0];
      }).filter(Boolean);
    }
    return selected;
  }, [againstSelected, analysisMode, archetypesSelected, booksSelected, daysSelected, decimalMax, decimalMin, eventsSelected, includeLive, marketsSelected, minEdge, playersSelected, predMax, predMin, rawEdgeMax, rawEdgeMin, roundsSelected, rows, sampleMax, sampleMin, side, typesSelected]);

  if (loading) return <LoadingState label="Loading performance history" />;
  if (error || !data) return <ErrorState message={error ?? "Performance data is unavailable."} />;

  const resolved = filtered.filter((row) => row.result && !["no_data", "unknown"].includes(String(row.result)));
  const ordered = [...resolved].sort((a, b) => String(a.run_timestamp).localeCompare(String(b.run_timestamp)));
  const curve = buildCumulativeCurve(ordered);
  const wagered = sum(resolved.map((row) => row._units_wagered));
  const wins = resolved.filter((row) => String(row.result).startsWith("win")).length;
  const losses = resolved.filter((row) => row.result === "loss").length;
  const pushes = resolved.filter((row) => row.result === "push").length;
  const totalUnits = sum(resolved.map((row) => row._units_won));
  const winRate = wins + losses ? wins / (wins + losses) * 100 : 0;

  const eventFirstSeen = new Map<string, string>();
  ordered.forEach((row) => {
    const name = String(row.event_name);
    if (!eventFirstSeen.has(name)) eventFirstSeen.set(name, String(row.run_timestamp));
  });
  const eventOrder = [...eventFirstSeen.entries()].sort((a, b) => a[1].localeCompare(b[1])).map(([name]) => name);
  const byEvent = eventOrder.map((name) => {
    const eventRows = resolved.filter((row) => String(row.event_name) === name);
    const point: DataRow = { event: titleCase(name), total: sum(eventRows.map((row) => row._units_won)) };
    PERFORMANCE_TYPE_OPTIONS.forEach((option) => point[option.value] = sum(eventRows.filter((row) => row.bet_type === option.value).map((row) => row._units_won)));
    return point;
  });
  const byBook = books.map((name) => {
    const bookRows = resolved.filter((row) => String(row.bookmaker) === name);
    const risked = sum(bookRows.map((row) => row._units_wagered));
    return { book: titleCase(name), roi: risked ? sum(bookRows.map((row) => row._units_won)) / risked * 100 : 0, bets: bookRows.length };
  }).filter((row) => row.bets >= 3).sort((a, b) => a.roi - b.roi);
  const bucketRows = [
    ...bucketRoi(resolved, "Raw edge", "#54d6c8", [{ label: "0–2%", min: 0, max: 2 }, { label: "2–4%", min: 2, max: 4 }, { label: "4–6%", min: 4, max: 6 }, { label: "6%+", min: 6, max: 1000 }], (row) => numberValue(row.raw_edge, Number.NaN)),
    ...bucketRoi(resolved, "Kelly edge", "#ffba69", [{ label: "3–5%", min: 3, max: 5 }, { label: "5–8%", min: 5, max: 8 }, { label: "8%+", min: 8, max: 1000 }], (row) => numberValue(row.edge, Number.NaN)),
    ...bucketRoi(resolved, "Odds", "#8ca7ff", [{ label: "<2.0", min: 0, max: 2 }, { label: "2.0–2.5", min: 2, max: 2.5 }, { label: "2.5–3.5", min: 2.5, max: 3.5 }, { label: "3.5–8.0", min: 3.5, max: 8 }, { label: "8.0+", min: 8, max: 1000 }], (row) => numberValue(row.dec_odds, Number.NaN)),
  ];
  const activeArchetypes = uniqueStrings(resolved, "archetype");
  const archetypeSeries = new Map(activeArchetypes.map((archetype) => {
    let running = 0;
    return [archetype, [...resolved].filter((row) => row.archetype === archetype).sort((a, b) => String(a.run_timestamp).localeCompare(String(b.run_timestamp))).map((row) => (running += row._units_won))];
  }));
  const maxArchetypeBets = Math.max(0, ...[...archetypeSeries.values()].map((values) => values.length));
  const archetypeCurve = Array.from({ length: maxArchetypeBets }, (_, index) => {
    const point: DataRow = { index: index + 1 };
    archetypeSeries.forEach((values, archetype) => { if (values[index] !== undefined) point[archetype] = values[index]; });
    return point;
  });
  const archetypePnl = activeArchetypes.map((archetype) => ({ archetype, units: sum(resolved.filter((row) => row.archetype === archetype).map((row) => row._units_won)) })).sort((a, b) => a.units - b.units);
  const againstPnl = uniqueStrings(resolved, "archetype_against").map((archetype) => ({ archetype, units: sum(resolved.filter((row) => row.archetype_against === archetype).map((row) => row._units_won)) })).sort((a, b) => a.units - b.units);
  const eventSummary = eventOrder.map((name) => {
    const eventRows = resolved.filter((row) => String(row.event_name) === name);
    const risked = sum(eventRows.map((row) => row._units_wagered));
    const won = sum(eventRows.map((row) => row._units_won));
    return { event_name: name, bets: eventRows.length, wins: eventRows.filter((row) => String(row.result).startsWith("win")).length, losses: eventRows.filter((row) => row.result === "loss").length, wagered: risked, units_won: won, roi: risked ? won / risked * 100 : 0 };
  }).sort((a, b) => b.units_won - a.units_won);
  const detailRows = filtered.map((row) => ({ ...row, units_wagered: row._units_wagered, units_won: row._units_won }));
  const activeFilterCount = [eventsSelected, typesSelected, booksSelected, roundsSelected, daysSelected, marketsSelected, archetypesSelected, againstSelected, playersSelected].filter((value) => value.length).length + [sampleMin, sampleMax, predMin, predMax, rawEdgeMin, rawEdgeMax, decimalMin, decimalMax].filter((value) => value !== null).length + (minEdge > 0 ? 1 : 0) + (includeLive ? 1 : 0) + (side !== "all" ? 1 : 0) + (analysisMode !== "all" ? 1 : 0);

  function resetFilters() {
    setEventsSelected([]); setTypesSelected([]); setBooksSelected([]); setMinEdge(0); setIncludeLive(false);
    setRoundsSelected([]); setDaysSelected([]); setSampleMin(null); setSampleMax(null); setPredMin(null); setPredMax(null);
    setMarketsSelected([]); setSide("all"); setAnalysisMode("all"); setRawEdgeMin(null); setRawEdgeMax(null);
    setDecimalMin(null); setDecimalMax(null); setArchetypesSelected([]); setAgainstSelected([]); setPlayersSelected([]);
  }

  return (
    <>
      <PageIntro eyebrow="Bet review" title="Performance" description="Historical results with the original analysis universe, inclusion rules, advanced filters, and P&L breakdowns restored." />
      <Panel title="Filters & inclusion rules" eyebrow={`${activeFilterCount} active filter${activeFilterCount === 1 ? "" : "s"}`} actions={<button type="button" className="icon-button" onClick={resetFilters}>Reset filters</button>} className="performance-filter-panel">
        <div className="inclusion-rules" aria-label="Default bet inclusion rules">
          <span className={typesSelected.includes("score_bet") ? "included" : "excluded"}><b>Score bets</b>{typesSelected.includes("score_bet") ? "Included by selection" : "Excluded by default"}</span>
          <span className={includeLive || typesSelected.includes("finish_position_live") ? "included" : "excluded"}><b>Live finish bets</b>{includeLive || typesSelected.includes("finish_position_live") ? "Included" : "Excluded by default"}</span>
          <span className={booksSelected.some((book) => EXCHANGE_BOOKS.some((exchange) => book.toLowerCase().includes(exchange))) ? "included" : "excluded"}><b>Kalshi / NoVig</b>{booksSelected.some((book) => EXCHANGE_BOOKS.some((exchange) => book.toLowerCase().includes(exchange))) ? "Included by selection" : "Hidden until selected"}</span>
        </div>
        <div className="filter-section"><h3>Bet universe</h3><div className="performance-filter-grid">
          <MultiSelectControl label="Event" value={eventsSelected} onChange={setEventsSelected} placeholder="All events" options={events.map((value) => ({ value, label: titleCase(value) }))}/>
          <MultiSelectControl label="Bet type" value={typesSelected} onChange={setTypesSelected} placeholder="Default types" options={PERFORMANCE_TYPE_OPTIONS}/>
          <MultiSelectControl label="Sportsbook" value={booksSelected} onChange={setBooksSelected} placeholder="Default books" options={books.map((value) => ({ value, label: titleCase(value) }))}/>
          <RangeControl label="Minimum Kelly edge" value={minEdge} min={0} max={25} onChange={setMinEdge}/>
          <label className="toggle-filter"><input type="checkbox" aria-label="Include live finish-position bets" checked={includeLive} onChange={(event) => setIncludeLive(event.target.checked)}/><span><b>Live bets</b><small>Include live finish positions</small></span></label>
        </div></div>
        <div className="filter-section"><h3>Timing & market</h3><div className="performance-filter-grid performance-filter-grid-six">
          <MultiSelectControl label="Round" value={roundsSelected} onChange={setRoundsSelected} placeholder="All rounds" options={[1,2,3,4].map((round) => ({ value: String(round), label: `R${round}` }))}/>
          <MultiSelectControl label="Day of bet" value={daysSelected} onChange={setDaysSelected} placeholder="All days" options={["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].map((value) => ({ value, label: value.slice(0,3) }))}/>
          <NumberRangeControl label="Sample size" minValue={sampleMin} maxValue={sampleMax} onMinChange={setSampleMin} onMaxChange={setSampleMax} min={0}/>
          <NumberRangeControl label="Pred (skill estimate)" minValue={predMin} maxValue={predMax} onMinChange={setPredMin} onMaxChange={setPredMax} min={0} step={0.1}/>
          <MultiSelectControl label="Market (finish position)" value={marketsSelected} onChange={setMarketsSelected} placeholder="All markets" options={[{value:"win",label:"Win"},{value:"top_5",label:"Top 5"},{value:"top_10",label:"Top 10"},{value:"top_20",label:"Top 20"}]}/>
          <SelectControl label="Side (finish position)" value={side} onChange={setSide} options={[{value:"all",label:"All sides"},{value:"yes",label:"YES side"},{value:"no",label:"NO side (fades)"}]}/>
        </div></div>
        <div className="filter-section"><h3>Analysis</h3><div className="performance-filter-grid performance-filter-grid-six">
          <SelectControl label="Analysis mode" value={analysisMode} onChange={setAnalysisMode} options={[{value:"all",label:"All bets"},{value:"best_price",label:"Best price"},{value:"sharp_only",label:"Sharp only"},{value:"sharp_best",label:"Sharp only, best price"}]}/>
          <NumberRangeControl label="Raw % edge" minValue={rawEdgeMin} maxValue={rawEdgeMax} onMinChange={setRawEdgeMin} onMaxChange={setRawEdgeMax} step={0.5}/>
          <NumberRangeControl label="Decimal odds" minValue={decimalMin} maxValue={decimalMax} onMinChange={setDecimalMin} onMaxChange={setDecimalMax} min={1} step={0.1}/>
          <MultiSelectControl label="Archetype" value={archetypesSelected} onChange={setArchetypesSelected} placeholder="All archetypes" options={archetypes.map((value) => ({value,label:value}))}/>
          <MultiSelectControl label="Archetype against" value={againstSelected} onChange={setAgainstSelected} placeholder="All archetypes" options={archetypesAgainst.map((value) => ({value,label:value}))}/>
          <MultiSelectControl label="Player" value={playersSelected} onChange={setPlayersSelected} placeholder="All players" options={players.map((value) => ({value,label:titleCase(value)}))}/>
        </div></div>
      </Panel>

      <div className="kpi-grid performance-kpi-grid"><Kpi label="Total bets" value={filtered.length.toLocaleString()} detail={`${resolved.length.toLocaleString()} resolved`}/><Kpi label="Record" value={`${wins}W-${losses}L-${pushes}P`} detail={`${(filtered.length - resolved.length).toLocaleString()} open or ungraded`}/><Kpi label="Win rate" value={`${winRate.toFixed(1)}%`} detail="Pushes excluded"/><Kpi label="Units won" value={`${totalUnits >= 0 ? "+" : ""}${totalUnits.toFixed(2)}u`} detail={`${wagered.toFixed(2)} units risked`} tone={totalUnits >= 0 ? "positive" : "negative"}/><Kpi label="ROI" value={`${wagered ? totalUnits / wagered * 100 >= 0 ? "+" : "" : ""}${wagered ? (totalUnits / wagered * 100).toFixed(1) : "0.0"}%`} detail="Resolved bets" tone={totalUnits >= 0 ? "positive" : "negative"}/></div>

      {filtered.length === 0 ? <EmptyState title="No bets match these filters" detail="Reset filters or broaden the selected analysis universe."/> : <>
        <div className="performance-chart-grid">
          <Panel title="Cumulative P&L" eyebrow="Resolved bets in time order"><div className="chart-medium"><ResponsiveContainer width="100%" height="100%"><LineChart data={curve} margin={chartMargin}><CartesianGrid stroke="var(--line)" vertical={false}/><XAxis dataKey="index" stroke="var(--muted)"/><YAxis stroke="var(--muted)"/><Tooltip content={<ChartTooltip/>}/><ReferenceLine y={0} stroke="var(--line-strong)"/><Line type="monotone" dataKey="units" stroke="var(--accent)" dot={false} strokeWidth={2.5}/></LineChart></ResponsiveContainer></div></Panel>
          <Panel title="P&L by event" eyebrow="Bet type attribution"><div className="chart-medium"><ResponsiveContainer width="100%" height="100%"><ComposedChart data={byEvent} margin={{...chartMargin,bottom:52}}><CartesianGrid stroke="var(--line)" vertical={false}/><XAxis dataKey="event" stroke="var(--muted)" angle={-28} textAnchor="end" interval={0} height={70}/><YAxis stroke="var(--muted)"/><Tooltip content={<ChartTooltip/>}/><Legend/>{PERFORMANCE_TYPE_OPTIONS.map((option) => <Bar key={option.value} dataKey={option.value} name={option.label} fill={PERFORMANCE_TYPE_COLORS[option.value]} radius={[3,3,0,0]}/>) }<Line type="monotone" dataKey="total" name="Total" stroke="var(--foreground)" strokeWidth={2} dot={{r:2}}/></ComposedChart></ResponsiveContainer></div></Panel>
          <Panel title="ROI by sportsbook" eyebrow="Minimum 3 resolved bets"><div className="chart-medium"><ResponsiveContainer width="100%" height="100%"><BarChart data={byBook} layout="vertical" margin={{...chartMargin,left:36}}><CartesianGrid stroke="var(--line)" horizontal={false}/><XAxis type="number" stroke="var(--muted)"/><YAxis type="category" dataKey="book" stroke="var(--muted)" width={92}/><Tooltip content={<ChartTooltip/>}/><ReferenceLine x={0} stroke="var(--line-strong)"/><Bar dataKey="roi" name="ROI %" radius={[0,6,6,0]}>{byBook.map((row) => <Cell key={row.book} fill={row.roi >= 0 ? "var(--positive)" : "var(--negative)"}/>)}</Bar></BarChart></ResponsiveContainer></div></Panel>
        </div>
        <Panel title="ROI by bucket" eyebrow="Raw edge, Kelly edge, and decimal odds"><div className="chart-large"><ResponsiveContainer width="100%" height="100%"><BarChart data={bucketRows} margin={{...chartMargin,bottom:76}}><CartesianGrid stroke="var(--line)" vertical={false}/><XAxis dataKey="bucket" stroke="var(--muted)" angle={-28} textAnchor="end" interval={0} height={92}/><YAxis stroke="var(--muted)" tickFormatter={(value) => `${value}%`}/><Tooltip content={<ChartTooltip/>}/><ReferenceLine y={0} stroke="var(--line-strong)"/><Bar dataKey="roi" name="ROI %" radius={[5,5,0,0]}>{bucketRows.map((row,index) => <Cell key={index} fill={String(row.color)}/>)}</Bar></BarChart></ResponsiveContainer></div></Panel>
        <div className="performance-chart-grid">
          <Panel title="Cumulative P&L by archetype" eyebrow="Player bet-on profile"><div className="chart-medium"><ResponsiveContainer width="100%" height="100%"><LineChart data={archetypeCurve} margin={chartMargin}><CartesianGrid stroke="var(--line)" vertical={false}/><XAxis dataKey="index" stroke="var(--muted)"/><YAxis stroke="var(--muted)"/><Tooltip content={<ChartTooltip/>}/><ReferenceLine y={0} stroke="var(--line-strong)"/><Legend/>{activeArchetypes.map((archetype,index) => <Line key={archetype} type="monotone" dataKey={archetype} stroke={palette[index % palette.length]} dot={false} strokeWidth={1.8}/>)}</LineChart></ResponsiveContainer></div></Panel>
          <Panel title="Total P&L by archetype" eyebrow="Player bet-on profile"><div className="chart-medium"><ResponsiveContainer width="100%" height="100%"><BarChart data={archetypePnl} layout="vertical" margin={{...chartMargin,left:62}}><CartesianGrid stroke="var(--line)" horizontal={false}/><XAxis type="number" stroke="var(--muted)"/><YAxis type="category" dataKey="archetype" stroke="var(--muted)" width={125}/><Tooltip content={<ChartTooltip/>}/><ReferenceLine x={0} stroke="var(--line-strong)"/><Bar dataKey="units" radius={[0,5,5,0]}>{archetypePnl.map((row) => <Cell key={row.archetype} fill={row.units >= 0 ? "var(--positive)" : "var(--negative)"}/>)}</Bar></BarChart></ResponsiveContainer></div></Panel>
          <Panel title="P&L by archetype against" eyebrow="Opponent profile"><div className="chart-medium"><ResponsiveContainer width="100%" height="100%"><BarChart data={againstPnl} layout="vertical" margin={{...chartMargin,left:62}}><CartesianGrid stroke="var(--line)" horizontal={false}/><XAxis type="number" stroke="var(--muted)"/><YAxis type="category" dataKey="archetype" stroke="var(--muted)" width={125}/><Tooltip content={<ChartTooltip/>}/><ReferenceLine x={0} stroke="var(--line-strong)"/><Bar dataKey="units" radius={[0,5,5,0]}>{againstPnl.map((row) => <Cell key={row.archetype} fill={row.units >= 0 ? "var(--positive)" : "var(--negative)"}/>)}</Bar></BarChart></ResponsiveContainer></div></Panel>
        </div>
        <Panel title="Event summary" eyebrow="Resolved performance by tournament"><DataTable rows={eventSummary} label="Event summary" preferredColumns={["event_name","bets","wins","losses","wagered","units_won","roi"]} pageSize={30}/></Panel>
        <Panel title="Filtered bets" eyebrow="Search, sort, customize, export"><DataTable rows={detailRows} label="Filtered bets" preferredColumns={["bet_on","opponent","bookmaker","bet_type","round","event_name","archetype","archetype_against","edge","raw_edge","dec_odds","pred_on","result","units_wagered","units_won"]} pageSize={40}/></Panel>
      </>}
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
