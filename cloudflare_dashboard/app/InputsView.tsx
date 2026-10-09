"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Search } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { DataTable, EmptyState, ErrorState, Kpi, LoadingState, PageIntro, Panel, SegmentedControl } from "./components";
import { useDashboardData } from "./data";
import { LABELS } from "./labels";
import { matchPlayers, matchText } from "./inputs-search";
import { DataRow, numberValue, palette, titleCase } from "./lib";
import { ALLOWED, CUT_BOUNDS, MAX_LIFETIME_DAYS, checkOverride, isExpired, isoSeconds, specFor, type FieldSpec, type OverrideRecord, type Scope } from "./overrides-rules";

/* ------------------------------------------------------------------ loose shapes of golfprice.model_inputs.v1 (see golfprice/INPUTS_SCHEMA.md) */
type Obj = Record<string, unknown>;
const obj = (value: unknown): Obj => (value && typeof value === "object" && !Array.isArray(value) ? (value as Obj) : {});
const arr = (value: unknown): Obj[] => (Array.isArray(value) ? (value as Obj[]) : []);
const num = (value: unknown): number | null => (typeof value === "number" && Number.isFinite(value) ? value : null);
const fx = (value: unknown, digits = 3): string => {
  const v = num(value);
  return v === null ? "—" : v.toFixed(digits);
};
const signed = (value: unknown, digits = 3): string => {
  const v = num(value);
  return v === null ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(digits)}`;
};
const pct = (value: unknown, digits = 1): string => {
  const v = num(value);
  return v === null ? "—" : `${(v * 100).toFixed(digits)}%`;
};

function armLabel(name: string): string {
  const known: Record<string, string> = { champion: "previous model", comparison: "alternative version", shadow: "alternative version" };
  return known[name] ?? name.replaceAll("_", " ");
}
type IndexRun = { kind: string; run: string; as_of: string | null; after_round: number | null; key: string; hole_table_key: string | null; sha256: string; bytes: number; n_players: number; overrides_applied: string[]; overrides_rejected: string[] };
type IndexEvent = { event_uid: string; name: string; tour: string; date_start: string; date_end: string; course: string; runs: IndexRun[]; latest_key: string | null };
type PublishIndex = { schema: string; events: IndexEvent[] };

const FAMILIES: Array<[string, string]> = [
  ["level_form", "Level and form"],
  ["kalman", "Kalman trend"],
  ["xtour", "Cross-tour"],
  ["category", "Category skills"],
  ["sit", "Situation"],
  ["act", "Activity"],
  ["sklv", "Skill level"],
  ["course", "Course fit and history"],
  ["thin", "Thin-data shrink"],
  ["disp", "Dispersion"],
  ["other", "Other"],
];

const TABS = [
  { value: "players", label: "Players" },
  { value: "course", label: "Course" },
  { value: "variance", label: "Variance and engine" },
  { value: "weather", label: "Weather and waves" },
  { value: "odds", label: "Odds freshness" },
  { value: "config", label: "Config" },
  { value: "features", label: "Features" },
  { value: "adjust", label: "Adjust" },
] as const;
type Tab = (typeof TABS)[number]["value"];

function ChartTip({ active, payload, label }: { active?: boolean; payload?: Array<{ name?: string; value?: unknown; color?: string }>; label?: unknown }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <strong>{String(label ?? "")}</strong>
      {payload.map((item, index) => (
        <span key={`${item.name}-${index}`} style={{ color: item.color }}>
          {item.name}: {Array.isArray(item.value) ? item.value.map((x) => Number(x).toFixed(3)).join(" to ") : typeof item.value === "number" ? item.value.toFixed(3) : String(item.value ?? "—")}
        </span>
      ))}
    </div>
  );
}

function Select({ label, value, options, onChange }: { label: string; value: string; options: Array<{ value: string; label: string }>; onChange: (value: string) => void }) {
  return (
    <label className="select-control">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option value={option.value} key={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

/** Flatten a nested object into dotted rows for a compact key-value table. */
function flatten(value: unknown, prefix = "", depth = 0, out: Array<{ key: string; value: string }> = []): Array<{ key: string; value: string }> {
  if (value && typeof value === "object" && !Array.isArray(value) && depth < 3) {
    for (const [key, child] of Object.entries(value as Obj)) flatten(child, prefix ? `${prefix}.${key}` : key, depth + 1, out);
  } else {
    const text = typeof value === "number" ? String(Math.round(value * 1e6) / 1e6) : typeof value === "string" ? value : JSON.stringify(value);
    out.push({ key: prefix, value: text === undefined ? "—" : text.length > 240 ? `${text.slice(0, 240)}…` : text });
  }
  return out;
}

function KeyValue({ data, empty = "Nothing recorded for this run." }: { data: unknown; empty?: string }) {
  const rows = flatten(data);
  if (!rows.length || (rows.length === 1 && !rows[0].key)) return <p className="inputs-muted">{empty}</p>;
  return (
    <div className="kv-table">
      {rows.map((row) => (
        <div key={row.key}>
          <span>{row.key.replaceAll("_", " ")}</span>
          <b>{row.value}</b>
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ players */
function playerRows(players: Obj[]): DataRow[] {
  return players.map((p) => {
    const ch = obj(p.challenger);
    const bd = obj(ch.breakdown);
    const fam = obj(bd.chl_families);
    const loc = obj(bd.location);
    const ovr = obj(bd.override);
    const cf = obj(p.course_fit);
    const prob = obj(ch.probabilities);
    const row: DataRow = {
      name: String(p.name ?? ""),
      mu: num(ch.mu),
      mu_untouched: num(ch.mu_untouched),
      [LABELS.thisWeekPga.short]: num(ch.mu_tour),
      sd: num(ch.sd),
      sd_untouched: num(ch.sd_untouched),
      se_kernel: num(ch.se_kernel),
      location: num(loc.total),
      course_fit: num(cf.contribution_to_mu),
      fit_rs_ddacc: num(cf.fit_rs_ddacc_lam1000),
      course_history: num(cf.course_history_resid_k80),
      course_sd_mult: num(cf.course_sd_mult_feature),
      override_total: num(ovr.total),
      prior_rounds: num(p.n_prior_rounds),
      prob_win: num(prob.p_win),
      prob_top_10: num(prob.p_top_10),
      prob_make_cut: num(prob.p_make_cut),
      country: String(p.country ?? ""),
      amateur: Boolean(p.amateur),
      dg_id: num(p.dg_id),
    };
    for (const [key] of FAMILIES) row[`chl_${key}`] = num(fam[key]);
    return row;
  });
}

function waterfall(player: Obj) {
  const ch = obj(player.challenger);
  const bd = obj(ch.breakdown);
  const fam = obj(bd.chl_families);
  const loc = obj(bd.location);
  const ovr = obj(bd.override);
  const parts: Array<{ name: string; value: number }> = [];
  for (const [key, label] of FAMILIES) parts.push({ name: label, value: num(fam[key]) ?? 0 });
  parts.push({ name: "Location (home, nationality, refit)", value: num(loc.total) ?? 0 });
  parts.push({ name: "Owner override", value: num(ovr.total) ?? 0 });
  let running = 0;
  const steps = parts.map((part) => {
    const start = running;
    running += part.value;
    return { name: part.name, value: part.value, range: [Math.min(start, running), Math.max(start, running)] as [number, number], end: running };
  });
  steps.push({ name: "Final mu", value: running, range: [Math.min(0, running), Math.max(0, running)], end: running });
  return steps;
}

type PlayerChoice = { name: string; dg_id: number };

/** Type-ahead over the field's player names ("first last" or "last, first", accent and case-insensitive); picking one selects that player. */
function PlayerSearch({ players, onPick }: { players: PlayerChoice[]; onPick: (dgId: number) => void }) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [highlight, setHighlight] = useState(0);
  const matches = useMemo(() => matchPlayers(players, query, 8), [players, query]);
  const showList = open && query.trim() !== "";
  const pick = (choice: PlayerChoice | undefined) => {
    if (!choice) return;
    onPick(choice.dg_id);
    setQuery("");
    setOpen(false);
  };
  const onKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setOpen(true);
      setHighlight((current) => Math.min(current + 1, Math.max(0, matches.length - 1)));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setHighlight((current) => Math.max(current - 1, 0));
    } else if (event.key === "Enter") {
      event.preventDefault();
      pick(matches[Math.min(highlight, matches.length - 1)]);
    } else if (event.key === "Escape") {
      setQuery("");
      setOpen(false);
    }
  };
  return (
    <div className="player-search">
      <label className="search-box">
        <Search size={15} />
        <input
          type="search"
          role="combobox"
          aria-expanded={showList}
          aria-controls="player-search-list"
          aria-autocomplete="list"
          aria-label="Find a player"
          autoComplete="off"
          value={query}
          placeholder="Find a player…"
          onChange={(event) => {
            setQuery(event.target.value);
            setHighlight(0);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => setOpen(false)}
          onKeyDown={onKeyDown}
        />
      </label>
      {showList && (
        <ul className="player-search-list" id="player-search-list" role="listbox">
          {matches.length === 0 && <li className="player-search-empty">No player matches “{query.trim()}”</li>}
          {matches.map((choice, index) => (
            <li key={choice.dg_id} role="option" aria-selected={index === highlight}>
              <button type="button" className={index === highlight ? "active" : ""} onMouseDown={(event) => event.preventDefault()} onClick={() => pick(choice)} onMouseEnter={() => setHighlight(index)}>
                {choice.name}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function PlayerDetail({ player, choices, onPick, onAdjust }: { player: Obj; choices: PlayerChoice[]; onPick: (dgId: number) => void; onAdjust: (dgId: number) => void }) {
  const ch = obj(player.challenger);
  const bd = obj(ch.breakdown);
  const steps = useMemo(() => waterfall(player), [player]);
  const arms = Object.entries(obj(player.arms));
  const prob = obj(ch.probabilities);
  const probU = obj(ch.probabilities_untouched);
  const cf = obj(player.course_fit);
  const hist = obj(player.history);
  const tee = obj(player.tee);
  const overridden = num(obj(bd.override).total) !== 0 || (Array.isArray(ch.overrides) && ch.overrides.length > 0);
  return (
    <Panel
      className="player-detail"
      eyebrow="Player detail"
      title={String(player.name ?? "")}
      actions={
        <>
          <PlayerSearch players={choices} onPick={onPick} />
          <button type="button" className="inputs-button" onClick={() => onAdjust(Number(player.dg_id))}>Adjust this player</button>
        </>
      }
    >
      <div className="mini-stat-grid">
        <div><span>{"Final mu (vs this week's field)"}</span><strong>{signed(ch.mu)}</strong></div>
        <div><span>Untouched mu</span><strong>{signed(ch.mu_untouched)}</strong></div>
        <div title={`${LABELS.thisWeekPga.long}. Reference only; prices use the field-relative number.`}><span>{LABELS.thisWeekPga.short}</span><strong>{signed(ch.mu_tour)}</strong></div>
        <div><span>Round SD</span><strong>{fx(ch.sd, 2)}</strong></div>
        <div><span>Untouched SD</span><strong>{fx(ch.sd_untouched, 2)}</strong></div>
        <div><span>Kernel SE</span><strong>{fx(ch.se_kernel, 3)}</strong></div>
        <div><span>Extra round volatility</span><strong>{fx(ch.extra_round_volatility_sd, 2)}</strong></div>
        <div><span>Prior rounds</span><strong>{fx(player.n_prior_rounds, 0)}</strong></div>
        <div><span>Overridden</span><strong>{overridden ? "Yes" : "No"}</strong></div>
      </div>
      <h3 className="inputs-h3">How the mean is built (strokes gained per round, field-centred)</h3>
      <div className="chart-medium waterfall">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={steps} layout="vertical" margin={{ top: 6, right: 24, bottom: 6, left: 8 }}>
            <CartesianGrid stroke="var(--line)" horizontal={false} />
            <XAxis type="number" tick={{ fill: "var(--muted)", fontSize: 10 }} />
            <YAxis type="category" dataKey="name" width={190} tick={{ fill: "var(--muted-strong)", fontSize: 10 }} />
            <ReferenceLine x={0} stroke="var(--line-strong)" />
            <Tooltip content={<ChartTip />} />
            <Bar dataKey="range" name="Component (start to end)" radius={3} isAnimationActive={false}>
              {steps.map((step, index) => (
                <Cell key={step.name} fill={index === steps.length - 1 ? "var(--model)" : step.value >= 0 ? "var(--positive)" : "var(--negative)"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="two-column inputs-split">
        <div>
          <h3 className="inputs-h3">Components</h3>
          <div className="kv-table">
            {steps.slice(0, -1).map((step) => (
              <div key={step.name}><span>{step.name}</span><b>{signed(step.value)}</b></div>
            ))}
            <div><span>CHL total</span><b>{signed(bd.chl_total)}</b></div>
            <div><span>Sum error vs mu (rounding)</span><b>{fx(bd.sum_error_vs_mu, 6)}</b></div>
          </div>
        </div>
        <div>
          <h3 className="inputs-h3">Course fit, history and tee</h3>
          <div className="kv-table">
            <div><span>Course contribution to mu</span><b>{signed(cf.contribution_to_mu)}</b></div>
            <div><span>Venue fit (distance and accuracy slopes x attributes)</span><b>{signed(cf.fit_rs_ddacc_lam1000, 4)}</b></div>
            <div><span>Course history residual (k=80)</span><b>{signed(cf.course_history_resid_k80, 4)}</b></div>
            <div><span>Course SD multiplier (feature)</span><b>{fx(cf.course_sd_mult_feature, 3)}</b></div>
            <div><span>Shrunk SD (feature)</span><b>{fx(cf.sd_shrunk_feature, 3)}</b></div>
            <div><span>Earlier-model course-history mu (J2)</span><b>{signed(hist.champion_mu_j2)}</b></div>
            <div><span>Earlier-model standard error</span><b>{fx(hist.se_mu, 3)}{hist.se_mu_imputed ? " (imputed)" : ""}</b></div>
            {Object.entries(tee).map(([round, info]) => (
              <div key={round}><span>Tee {round.toUpperCase()}</span><b>{String(obj(info).teetime_local ?? "—")} ({String(obj(info).wave ?? "—")})</b></div>
            ))}
          </div>
        </div>
      </div>
      <h3 className="inputs-h3">Model prices (before the market combiner)</h3>
      <div className="table-scroll">
        <table>
          <thead><tr><th><span className="th-text">Version</span></th><th><span className="th-text">Mu</span></th><th><span className="th-text">SD</span></th><th><span className="th-text">Win</span></th><th><span className="th-text">Top 5</span></th><th><span className="th-text">Top 10</span></th><th><span className="th-text">Top 20</span></th><th><span className="th-text">Make cut</span></th></tr></thead>
          <tbody>
            <tr className="active-row"><td>model (final, with overrides)</td><td>{signed(ch.mu)}</td><td>{fx(ch.sd, 2)}</td><td>{pct(prob.p_win, 2)}</td><td>{pct(prob.p_top_5)}</td><td>{pct(prob.p_top_10)}</td><td>{pct(prob.p_top_20)}</td><td>{pct(prob.p_make_cut)}</td></tr>
            {Object.keys(probU).length > 0 && <tr><td>model (untouched)</td><td>{signed(ch.mu_untouched)}</td><td>{fx(ch.sd_untouched, 2)}</td><td>{pct(probU.p_win, 2)}</td><td>{pct(probU.p_top_5)}</td><td>{pct(probU.p_top_10)}</td><td>{pct(probU.p_top_20)}</td><td>{pct(probU.p_make_cut)}</td></tr>}
            {arms.filter(([name]) => name !== "challenger").map(([name, value]) => {
              const a = obj(value);
              return <tr key={name}><td>{armLabel(name)}</td><td>{signed(a.mu)}</td><td>{fx(a.sd, 2)}</td><td>{pct(a.p_win, 2)}</td><td>{pct(a.p_top_5)}</td><td>{pct(a.p_top_10)}</td><td>{pct(a.p_top_20)}</td><td>{pct(a.p_make_cut)}</td></tr>;
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function PlayersTab({ doc, onAdjust }: { doc: Obj; onAdjust: (dgId: number) => void }) {
  const players = arr(doc.players);
  const rows = useMemo(() => playerRows(players), [players]);
  const [selected, setSelected] = useState<number | null>(null);
  const current = players.find((p) => num(p.dg_id) === selected) ?? players[0];
  const currentRow = rows.find((row) => row.dg_id === num(current?.dg_id)) ?? null;
  const choices = useMemo(() => players.flatMap((p) => (num(p.dg_id) === null ? [] : [{ name: String(p.name ?? ""), dg_id: num(p.dg_id) as number }])), [players]);
  const preferred = ["name", "mu", LABELS.thisWeekPga.short, "mu_untouched", "sd", "sd_untouched", "se_kernel", "location", "course_fit", "course_history", "prior_rounds", "prob_win", "prob_top_10", "prob_make_cut", "fit_rs_ddacc", "override_total", ...FAMILIES.map(([key]) => `chl_${key}`)];
  const mus = rows.map((row) => numberValue(row.mu)).filter(Number.isFinite);
  const strength = obj(obj(doc.event).field_strength);
  const fieldOffset = num(strength.field_offset);
  const withOverride = rows.filter((row) => numberValue(row.override_total) !== 0).length;
  const sdMean = rows.length ? rows.reduce((total, row) => total + numberValue(row.sd), 0) / rows.length : 0;
  if (!players.length) return <EmptyState title="No players in this run" detail="The run published no player objects." />;
  return (
    <div className="stack-lg">
      <div className="kpi-grid">
        <Kpi label="Players" value={String(players.length)} detail={`${players.filter((p) => p.amateur).length} amateurs`} tone="accent" />
        <Kpi label="Best mu" value={signed(Math.max(...mus), 2)} detail="strokes per round better than this week's field average (field average = 0)" />
        <Kpi
          label="Field strength"
          value={fieldOffset === null ? "—" : signed(fieldOffset, 2)}
          detail={fieldOffset === null ? "No tour-scale estimate for this run" : `this field is ${Math.abs(fieldOffset).toFixed(2)} strokes per round ${fieldOffset < 0 ? "worse" : "better"} than an average PGA Tour field (skills as of ${String(strength.vintage ?? "").slice(0, 10)}). Reference only: prices use the field-relative mu.`}
        />
        <Kpi label="Mean round SD" value={sdMean.toFixed(2)} detail="model, after overrides" />
        <Kpi label="Players with an override" value={String(withOverride)} detail="untouched numbers are kept" tone={withOverride ? "positive" : "neutral"} />
      </div>
      <Panel eyebrow="Saved model inputs" title="Every player, every component" actions={<span className="inputs-muted">{"Click a row, or search below, for the breakdown. \"" + LABELS.thisWeekPga.short + "\" is for reference; prices use mu (vs this week's field)."}</span>}>
        <DataTable rows={rows} preferredColumns={preferred} label="Model inputs players" pageSize={30} onRowClick={(row) => setSelected(num(row.dg_id))} activeRow={currentRow} />
      </Panel>
      {current && <PlayerDetail player={current} choices={choices} onPick={setSelected} onAdjust={onAdjust} />}
    </div>
  );
}

/* ------------------------------------------------------------------ course */
function CourseTab({ doc }: { doc: Obj }) {
  const course = obj(doc.course);
  const event = obj(doc.event);
  const tables = arr(course.hole_tables);
  const perRound = arr(course.per_round);
  const fit = obj(course.venue_fit);
  const slopes = obj(fit.slopes);
  const [group, setGroup] = useState(0);
  const table = tables[Math.min(group, Math.max(0, tables.length - 1))];
  const holes = arr(table?.holes);
  const holeRows: DataRow[] = holes.map((h) => ({
    hole: num(h.hole),
    par: num(h.par),
    yardage: num(h.yardage),
    expected_vs_par: num(h.exp_vs_par),
    birdie_pct: num(h.birdie_or_better),
    bogey_pct: num(h.bogey_or_worse),
    eagle_pct: num(h.p_eagle_or_better),
    par_pct: num(h.p_par),
    double_pct: num(h.p_double),
    triple_pct: num(h.p_triple_or_worse),
    sensitivity: num(h.sensitivity),
  }));
  const chartData = holes.map((h) => ({ hole: String(h.hole), expected: num(h.exp_vs_par) ?? 0, birdie: (num(h.birdie_or_better) ?? 0) * 100, bogey: (num(h.bogey_or_worse) ?? 0) * 100 }));
  const first = perRound[0];
  const sdMult = obj(fit.course_sd_mult);
  const override = obj(obj(doc.engine).owner_overrides_on_engine);
  return (
    <div className="stack-lg">
      <div className="kpi-grid">
        <Kpi label="Course" value={String(event.course_name ?? "—")} detail={`${String(event.venue_uid ?? "")} · par ${String((arr(event.par_per_round) as unknown as number[])[0] ?? "—")}`} tone="accent" />
        <Kpi label="Hole table source" value={titleCase(course.hole_table_source)} detail={String(course.table_label ?? "")} />
        <Kpi label="Scoring average vs par" value={first ? signed(first.scoring_average_vs_par, 2) : "—"} detail="round 1, field-average player" />
        <Kpi label="Course SD multiplier" value={fx(sdMult.median, 3)} detail={`field range ${fx(sdMult.min, 3)} to ${fx(sdMult.max, 3)}`} />
      </div>
      <Panel eyebrow="What the engine played" title="Hole by hole">
        <p className="inputs-note">{String(course.note ?? "")}</p>
        {tables.length > 1 && (
          <SegmentedControl label="Round group" value={String(group)} onChange={(v) => setGroup(Number(v))} options={tables.map((t, i) => ({ value: String(i), label: `Round${(t.rounds as unknown[]).length === 1 ? "" : "s"} ${(t.rounds as unknown[]).join(", ")}` }))} />
        )}
        <div className="chart-medium">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 10, right: 16, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="var(--line)" vertical={false} />
              <XAxis dataKey="hole" tick={{ fill: "var(--muted)", fontSize: 10 }} />
              <YAxis yAxisId="l" tick={{ fill: "var(--muted)", fontSize: 10 }} label={{ value: "Expected vs par", angle: -90, fill: "var(--muted)", fontSize: 10, position: "insideLeft" }} />
              <YAxis yAxisId="r" orientation="right" tick={{ fill: "var(--muted)", fontSize: 10 }} unit="%" />
              <Tooltip content={<ChartTip />} />
              <Legend />
              <ReferenceLine yAxisId="l" y={0} stroke="var(--line-strong)" />
              <Bar yAxisId="l" dataKey="expected" name="Expected vs par" fill={palette[0]} radius={3} isAnimationActive={false} />
              <Line yAxisId="r" dataKey="birdie" name="Birdie or better %" stroke={palette[1]} dot={false} isAnimationActive={false} />
              <Line yAxisId="r" dataKey="bogey" name="Bogey or worse %" stroke={palette[3]} dot={false} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <DataTable rows={holeRows} label="Hole table" pageSize={18} preferredColumns={["hole", "par", "yardage", "expected_vs_par", "birdie_pct", "bogey_pct", "eagle_pct", "par_pct", "double_pct", "triple_pct", "sensitivity"]} />
      </Panel>
      <div className="two-column">
        <Panel eyebrow="Per round" title="Course summary">
          <DataTable
            rows={perRound.map((r) => ({ round: num(r.round), par: num(r.par), yardage: num(r.yardage), scoring_avg_vs_par: num(r.scoring_average_vs_par), scoring_avg_untouched: num(r.scoring_average_vs_par_untouched), birdies_per_round: num(r.birdie_or_better_per_round), bogeys_per_round: num(r.bogey_or_worse_per_round) }))}
            label="Course per round"
            pageSize={8}
          />
          <p className="inputs-note">Hole model round SD at the average player: {fx(course.hole_model_round_sd, 3)}. The table fixes the SHAPE (birdie/bogey mix, hole variance, how skill spreads); the engine re-solves the level for every player-round, so a scoring-average override moves the displayed level and simulated scores, not the probabilities.</p>
        </Panel>
        <Panel eyebrow="Course fit" title="Venue fit and slopes">
          <div className="kv-table">
            {["fit_rs_ddacc_lam1000", "ch_resid_k80_hl1461", "course_sd_mult"].map((key) => {
              const stat = obj(fit[key]);
              return <div key={key}><span>{key.replaceAll("_", " ")}</span><b>min {fx(stat.min, 3)} · median {fx(stat.median, 3)} · max {fx(stat.max, 3)}</b></div>;
            })}
            <div><span>Owner course scoring-average delta</span><b>{course.course_scoring_avg_delta_override === null || course.course_scoring_avg_delta_override === undefined ? "none" : String(course.course_scoring_avg_delta_override)}</b></div>
            <div><span>Owner course SD multiplier</span><b>{override.course_sd_mult === null || override.course_sd_mult === undefined ? "none" : String(override.course_sd_mult)}</b></div>
            <div><span>Prior strength k (hole table)</span><b>{fx(course.prior_strength_k, 0)}</b></div>
          </div>
          <div className="inputs-banner">
            <strong>Slopes: {titleCase(slopes.status ?? "unknown")}</strong>
            <span>{String(slopes.reason ?? "")}</span>
          </div>
          <KeyValue data={Object.fromEntries(Object.entries(slopes).filter(([k]) => k !== "reason" && k !== "status"))} empty="" />
        </Panel>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ variance and engine */
function VarianceTab({ doc }: { doc: Obj }) {
  const eng = obj(doc.engine);
  const players = arr(doc.players);
  const points = players.map((p) => ({ name: String(p.name ?? ""), mu: num(obj(p.challenger).mu) ?? 0, sd: num(obj(p.challenger).sd) ?? 0 }));
  const cut = obj(obj(doc.event).cut_rule);
  const wl = obj(eng.week_latent);
  return (
    <div className="stack-lg">
      <div className="kpi-grid">
        <Kpi label="Simulations" value={Number(eng.n_sims ?? 0).toLocaleString()} detail={`${String(eng.n_seeds ?? "")} seeds from ${String(eng.seed0 ?? "")}`} tone="accent" />
        <Kpi label="Week latent SD (tau)" value={fx(wl.tau, 2)} detail={`rounds 1-2 x${fx(wl.t12, 2)}, 3-4 x${fx(wl.t34, 2)}`} />
        <Kpi label="Hole noise variance" value={fx(eng.V_hole, 2)} detail={`hole dependence a = ${fx(eng.hole_a, 3)}`} />
        <Kpi label="Seed spread (max p_win)" value={fx(eng.seed_spread_max_abs_p_win, 4)} detail="Monte Carlo noise across seeds" />
      </div>
      <div className="two-column">
        <Panel eyebrow="Spread" title="Per-round SD against mean">
          <div className="chart-medium">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 16, bottom: 18, left: 0 }}>
                <CartesianGrid stroke="var(--line)" />
                <XAxis type="number" dataKey="mu" name="mu" tick={{ fill: "var(--muted)", fontSize: 10 }} label={{ value: "mu (SG per round)", fill: "var(--muted)", fontSize: 10, position: "insideBottom", offset: -8 }} />
                <YAxis type="number" dataKey="sd" name="sd" domain={["auto", "auto"]} tick={{ fill: "var(--muted)", fontSize: 10 }} />
                <ZAxis range={[26, 26]} />
                <Tooltip content={<ChartTip />} />
                <Scatter data={points} fill={palette[0]} name="players" isAnimationActive={false} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </Panel>
        <Panel eyebrow="Rule" title="Variance rule and cut">
          <p className="inputs-note mono">{String(eng.variance_rule ?? "")}</p>
          <div className="kv-table">
            <div><span>Variance params version</span><b>{String(eng.variance_params_version ?? "—")}</b></div>
            <div><span>SD scale</span><b>{fx(eng.sd_scale, 4)}</b></div>
            <div><span>Field SD (common round conditions)</span><b>{fx(eng.field_sd_common_round_conditions, 3)}</b></div>
            <div><span>Rotation course SD</span><b>{fx(eng.rotation_course_sd, 3)} ({eng.rotation_layer_active ? "active" : "inactive"})</b></div>
            <div><span>Cut rule</span><b>{String(cut.description ?? "—")}</b></div>
            <div><span>Registry rule (before override)</span><b>{typeof cut.registry_rule === "object" ? JSON.stringify(cut.registry_rule) : String(cut.registry_rule ?? "—")}</b></div>
          </div>
        </Panel>
      </div>
      <div className="two-column">
        <Panel eyebrow="Latents" title="Week and day latent"><KeyValue data={{ week_latent: eng.week_latent, day_latent: eng.day_latent }} /></Panel>
        <Panel eyebrow="Shape" title="Mixture, kernel and SD pin check"><KeyValue data={{ shape_mixture: eng.shape_mixture, kernel_params: eng.kernel_params, sd_pin_check: eng.sd_pin_check }} /></Panel>
      </div>
      <Panel eyebrow="Owner" title="Overrides on the engine"><KeyValue data={eng.owner_overrides_on_engine} empty="No engine-level owner override." /></Panel>
    </div>
  );
}

/* ------------------------------------------------------------------ weather */
function WeatherTab({ doc }: { doc: Obj }) {
  const w = obj(doc.weather);
  const waves = arr(w.waves);
  const snapshot = obj(w.forecast_snapshot);
  const cover = obj(w.tee_sheet_coverage);
  const data = waves.map((x) => ({ label: `R${String(x.round)} ${String(x.wave)}`, wind: num(x.mean_forecast_wind_mph) ?? 0, players: num(x.n_players) ?? 0 }));
  return (
    <div className="stack-lg">
      <div className="inputs-banner warn">
        <strong>{w.applied_in_prices ? "Applied in prices" : "NOT APPLIED IN PRICES"}</strong>
        <span>{String(w.label ?? "Tee times, waves and the forecast are shown for review only.")}</span>
      </div>
      <div className="kpi-grid">
        <Kpi label="Forecast snapshot" value={String(snapshot.status ?? "—")} detail={`fetched ${String(snapshot.fetched_at ?? "—")}`} tone="accent" />
        <Kpi label="Tee sheet R1" value={pct(cover.r1, 0)} detail="players with a published time" />
        <Kpi label="Tee sheet R2" value={pct(cover.r2, 0)} detail="players with a published time" />
        <Kpi label="Waves" value={String(waves.length)} detail="round x wave groups" />
      </div>
      {data.length > 0 && (
        <Panel eyebrow="Forecast" title="Mean forecast wind by wave (mph)">
          <div className="chart-medium">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data} margin={{ top: 10, right: 16, bottom: 4, left: 0 }}>
                <CartesianGrid stroke="var(--line)" vertical={false} />
                <XAxis dataKey="label" tick={{ fill: "var(--muted)", fontSize: 10 }} />
                <YAxis tick={{ fill: "var(--muted)", fontSize: 10 }} />
                <Tooltip content={<ChartTip />} />
                <Bar dataKey="wind" name="Mean wind (mph)" fill={palette[2]} radius={3} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      )}
      <Panel eyebrow="Tee sheet" title="Waves">
        <DataTable rows={waves.map((x) => ({ round: num(x.round), wave: String(x.wave ?? ""), players: num(x.n_players), first_tee: String(x.first_tee_local ?? ""), last_tee: String(x.last_tee_local ?? ""), mean_wind_mph: num(x.mean_forecast_wind_mph) }))} label="Waves" pageSize={12} />
        <KeyValue data={snapshot} />
      </Panel>
    </div>
  );
}

/* ------------------------------------------------------------------ odds */
function OddsTab({ doc }: { doc: Obj }) {
  const odds = obj(doc.odds);
  const books = arr(odds.books);
  const stale = books.filter((b) => (num(b.age_hours_at_asof) ?? 0) > 6).length;
  return (
    <div className="stack-lg">
      <div className="kpi-grid">
        <Kpi label="As-of" value={String(odds.as_of ?? "—").replace("T", " ").replace("Z", " UTC")} detail="odds frozen at this time" tone="accent" />
        <Kpi label="Books" value={String(books.length)} detail={`${stale} older than 6 hours`} tone={stale ? "negative" : "positive"} />
        <Kpi label="Excluded as quote sources" value={arr(odds.excluded_books_config).length ? (odds.excluded_books_config as unknown as string[]).join(", ") : "none"} />
        <Kpi label="Win quotes" value={String(obj(odds.quotes_by_market).win ?? "—")} detail="all books" />
      </div>
      <p className="inputs-note">{String(odds.note ?? "")}</p>
      <Panel eyebrow="Freshness" title="Per book at the as-of">
        <DataTable
          rows={books.map((b) => ({ book: String(b.book ?? ""), age_hours: num(b.age_hours_at_asof), quotes: num(b.n_quotes), markets: Array.isArray(b.markets) ? (b.markets as unknown[]).join(", ") : String(b.markets ?? ""), latest_fetch: String(b.latest_fetch ?? ""), latest_book_update: String(b.latest_book_update ?? "") }))}
          label="Odds freshness"
          pageSize={20}
        />
      </Panel>
      <div className="two-column">
        <Panel eyebrow="Snapshots" title="Inputs by feed"><KeyValue data={odds.snapshot_inputs} /></Panel>
        <Panel eyebrow="Consensus" title="Diagnostics"><KeyValue data={{ books_by_market: odds.books_by_market, ...obj(odds.consensus_diagnostics) }} /></Panel>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ config */
function ConfigTab({ doc }: { doc: Obj }) {
  const config = obj(doc.config);
  const files = obj(config.files);
  const prov = obj(doc.provenance);
  return (
    <div className="stack-lg">
      <div className="kpi-grid">
        <Kpi label="Config files" value={String(Object.keys(files).length)} detail={String(config.config_dir ?? "")} tone="accent" />
        <Kpi label="Code" value={String(prov.code_git_sha ?? "—").slice(0, 8)} detail={`${String(prov.code_git_branch ?? "")} · ${String(prov.code_dirty_files ?? 0)} uncommitted`} />
        <Kpi label="Foundation" value={String(prov.foundation_id ?? "—").slice(0, 19)} detail="data snapshot used" />
        <Kpi label="Week key" value={String(prov.week_key ?? "—").slice(0, 10)} detail="hash of inputs and code" />
      </div>
      <Panel eyebrow="Parameters" title="Versions and run settings"><KeyValue data={{ params: config.params, run: config.run }} /></Panel>
      <Panel eyebrow="TOML" title="Config files used by this run">
        {Object.entries(files).map(([name, value]) => {
          const file = obj(value);
          return (
            <details className="config-file" key={name}>
              <summary><b>{name}</b><span>sha256 {String(file.sha256 ?? "").slice(0, 16)}</span></summary>
              <pre>{String(file.text ?? JSON.stringify(file, null, 1))}</pre>
            </details>
          );
        })}
      </Panel>
      <Panel eyebrow="Provenance" title="Where this run came from"><KeyValue data={{ ...prov, code_hashes: undefined }} /></Panel>
    </div>
  );
}

/* ------------------------------------------------------------------ features (golfprice.feature_glossary.v1, published by golfprice/feature_glossary.py) */
type WeightSet = Record<string, number | null | undefined>;
type GlossaryFeature = { base: string; family: string; family_label: string; name: string; measures: string; computed: string; sign_intuition: string; weights: { chl: WeightSet; v21: WeightSet }; abs_weight_chl: number; abs_weight_v21: number };
type GlossaryFamily = { key: string; label: string; summary: string; n_features: number; sum_base_weight_chl: number; share_abs_weight_chl: number; share_abs_weight_v21: number };
type GlossaryLocation = { base: string; part: string; name: string; measures: string; weights: { v21: WeightSet } };
type Glossary = {
  schema: string;
  season: number;
  model: Obj;
  variants: Array<{ key: string; label: string; suffix: string; text: string }>;
  standardisation: string;
  fit: string[];
  reading_weights: string;
  families: GlossaryFamily[];
  features: GlossaryFeature[];
  location: { note: string; columns: GlossaryLocation[]; n_columns_2026: number };
};

const VARIANT_SHORT: Record<string, string> = { base: "All players, all events", xeuro: "+ at DPWT events", xband0: "+ player <30 rounds", xband1: "+ player 30-100 rounds", miss: "+ if missing", miss_xeuro: "+ if missing, DPWT event" };
type WeightModel = "chl" | "v21";
type SortMode = "model" | "weight";

function WeightStrip({ weights, keys = Object.keys(VARIANT_SHORT) }: { weights: WeightSet; keys?: string[] }) {
  return (
    <div className="feature-weights">
      {keys.map((key) => {
        const value = num(weights[key]);
        return (
          <div key={key} className={value === null ? "absent" : value > 0 ? "positive" : value < 0 ? "negative" : ""} title={value === null ? "this variant column does not exist for this feature" : `${VARIANT_SHORT[key]}: ${value}`}>
            <span>{VARIANT_SHORT[key]}</span>
            <b>{value === null ? "—" : signed(value, 3)}</b>
          </div>
        );
      })}
    </div>
  );
}

function FeaturesTab() {
  const { data: glossary, loading, error } = useDashboardData<Glossary>("golfprice/feature_glossary.json");
  const [query, setQuery] = useState("");
  const [family, setFamily] = useState("all");
  const [model, setModel] = useState<WeightModel>("chl");
  const [sort, setSort] = useState<SortMode>("model");
  const features = useMemo(() => glossary?.features ?? [], [glossary]);
  const shown = useMemo(() => {
    const abs = (f: GlossaryFeature) => (model === "chl" ? f.abs_weight_chl : f.abs_weight_v21);
    const list = features.filter((f) => (family === "all" || f.family === family) && matchText(`${f.name} ${f.base} ${f.measures} ${f.computed} ${f.family_label}`, query));
    return sort === "weight" ? [...list].sort((a, b) => abs(b) - abs(a)) : list;
  }, [features, family, model, query, sort]);
  const locShown = useMemo(() => (glossary?.location.columns ?? []).filter((c) => (family === "all" || family === "location") && matchText(`${c.name} ${c.base} ${c.measures} ${c.part} location`, query)), [glossary, family, query]);
  if (loading) return <LoadingState label="Loading the feature glossary" />;
  if (error || !glossary) return <ErrorState message={error ?? "The feature glossary has not been published yet."} />;
  const m = glossary.model;
  const share = (f: GlossaryFamily) => (model === "chl" ? f.share_abs_weight_chl : f.share_abs_weight_v21);
  const visibleFamilies = glossary.families.filter((f) => shown.some((x) => x.family === f.key));
  return (
    <div className="stack-lg">
      <div className="kpi-grid">
        <Kpi label="Features" value={String(glossary.features.length)} detail={`${num(m.n_design_columns_2026) ?? "—"} design columns in ${glossary.season}`} tone="accent" />
        <Kpi label="Ridge penalty" value={fx(m.ridge_lambda_2026, 0)} detail={`chosen by rolling-origin CV, ${glossary.season} season`} />
        <Kpi label="Fitted on" value={String(m.train_seasons_2026 ?? "—")} detail={`${(num(m.n_train_2026) ?? 0).toLocaleString()} player-events, frozen all season`} />
        <Kpi label="Location columns" value={String(glossary.location.n_columns_2026)} detail="home base, travel and nationality (v2.1)" />
      </div>
      <Panel eyebrow="How to read this" title="What the weights mean">
        <p className="inputs-note">{String(m.overview ?? "")}</p>
        <p className="inputs-note">{glossary.reading_weights}</p>
        <h3 className="inputs-h3">Variants of each feature</h3>
        <div className="kv-table variant-help">
          {glossary.variants.map((v) => (
            <div key={v.key}><span>{v.label}{v.suffix ? ` (${v.suffix})` : ""}</span><b>{v.text}</b></div>
          ))}
        </div>
        <details className="config-file">
          <summary><b>Standardisation and how the weights are fitted</b><span>walk-forward ridge, per season</span></summary>
          <div className="glossary-prose">
            <p className="inputs-note">{glossary.standardisation}</p>
            <ul>{glossary.fit.map((line) => <li key={line}>{line}</li>)}</ul>
          </div>
        </details>
      </Panel>
      <Panel eyebrow="Find a feature" title="Filter">
        <div className="glossary-controls">
          <label className="search-box">
            <Search size={15} />
            <input type="search" aria-label="Search features" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search name, column or explanation…" />
          </label>
          <Select label="Family" value={family} onChange={setFamily} options={[{ value: "all", label: "All families" }, ...glossary.families.map((f) => ({ value: f.key, label: `${f.label} (${f.n_features})` })), { value: "location", label: `Location (${glossary.location.columns.length})` }]} />
          <Select label="Order" value={sort} onChange={(v) => setSort(v as SortMode)} options={[{ value: "model", label: "Model order" }, { value: "weight", label: "Largest weight first" }]} />
          <SegmentedControl label="Weights from" value={model} onChange={setModel} options={[{ value: "chl", label: "CHL (frozen)" }, { value: "v21", label: "v2.1 refit" }]} />
        </div>
        <p className="inputs-muted">{shown.length + locShown.length} of {glossary.features.length + glossary.location.columns.length} shown · weights are {glossary.season}-season standardised coefficients ({model === "chl" ? "frozen CHL ridge" : "v2.1 refit, CHL columns re-estimated jointly with the location columns"}).</p>
      </Panel>
      {visibleFamilies.map((fam) => (
        <Panel key={fam.key} eyebrow={`${fam.n_features} features · ${(share(fam) * 100).toFixed(1)}% of total absolute weight`} title={fam.label}>
          <p className="inputs-note">{fam.summary}</p>
          <div className="feature-grid">
            {shown.filter((f) => f.family === fam.key).map((f) => (
              <article className="feature-card" key={f.base} id={`feature-${f.base}`}>
                <header>
                  <h3>{f.name}</h3>
                  <code className="feature-code">{f.base}</code>
                </header>
                <p>{f.measures}</p>
                <WeightStrip weights={model === "chl" ? f.weights.chl : f.weights.v21} />
                <details>
                  <summary>How it is computed and what sign to expect</summary>
                  <h4>How it is computed</h4>
                  <p>{f.computed}</p>
                  <h4>Expected sign</h4>
                  <p>{f.sign_intuition}</p>
                </details>
              </article>
            ))}
          </div>
        </Panel>
      ))}
      {locShown.length > 0 && (
        <Panel eyebrow={`${glossary.location.columns.length} columns · v2.1 refit only`} title="Location columns">
          <p className="inputs-note">{glossary.location.note}</p>
          <div className="feature-grid">
            {locShown.map((c) => (
              <article className="feature-card" key={c.base}>
                <header>
                  <h3>{c.name}</h3>
                  <code className="feature-code">{c.base}</code>
                </header>
                <p>{c.measures}</p>
                <WeightStrip weights={c.weights.v21} keys={["base", "xeuro"]} />
                <span className="inputs-muted">{c.part}</span>
              </article>
            ))}
          </div>
        </Panel>
      )}
      {shown.length + locShown.length === 0 && <EmptyState title="No feature matches" detail="Try fewer words or clear the family filter." />}
    </div>
  );
}

/* ------------------------------------------------------------------ adjust */
type ActiveResponse = { ok: boolean; records: OverrideRecord[]; now: string; etag: string | null; error?: string };
type HistoryEntry = { action: string; at: string; by: { email: string; verified: boolean }; record: OverrideRecord };

async function apiJson<T>(url: string, init?: RequestInit): Promise<{ ok: boolean; status: number; body: T & { error?: string; problems?: string[] } }> {
  const response = await fetch(url, { ...init, headers: { accept: "application/json", ...(init?.body ? { "content-type": "application/json" } : {}), ...(init?.headers ?? {}) } });
  let body = {} as T & { error?: string; problems?: string[] };
  try {
    body = await response.json();
  } catch {
    body = { error: `HTTP ${response.status}` } as T & { error?: string };
  }
  return { ok: response.ok, status: response.status, body };
}

const localInput = (ms: number) => {
  const d = new Date(ms);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}`;
};

function describeValue(record: OverrideRecord): string {
  if (typeof record.value === "object") return Object.entries(record.value).map(([k, v]) => `${k}=${v}`).join(", ");
  return String(record.value);
}

function AdjustTab({ doc, eventUid, presetPlayer, onPresetUsed }: { doc: Obj; eventUid: string; presetPlayer: number | null; onPresetUsed: () => void }) {
  const { data: schema } = useDashboardData<{ fields?: Array<Obj> }>("golfprice/overrides_schema.json");
  const players = arr(doc.players);
  const eventObj = obj(doc.event);
  const specs: FieldSpec[] = ALLOWED; // compiled-in bounds are the enforced ones (worker); the published schema is cross-checked in tests
  const [records, setRecords] = useState<OverrideRecord[] | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [scope, setScope] = useState<Scope>("player");
  const [field, setField] = useState("skill_delta");
  const [eventKey, setEventKey] = useState(eventUid);
  const [target, setTarget] = useState<string>(presetPlayer ? String(presetPlayer) : "");
  const [value, setValue] = useState("");
  const [cut, setCut] = useState<Record<string, string>>({});
  const [reason, setReason] = useState("");
  const [expires, setExpires] = useState(() => localInput(Date.now() + 3 * 86_400_000));
  const [nowMs, setNowMs] = useState(() => Date.now());
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ tone: "ok" | "error"; text: string; problems?: string[] } | null>(null);

  const reload = useCallback(async () => {
    const [active, hist] = await Promise.all([apiJson<ActiveResponse>("/api/overrides"), apiJson<{ entries: HistoryEntry[] }>("/api/overrides/history")]);
    setNowMs(Date.now());
    if (active.ok) {
      setRecords(active.body.records);
      setLoadError(null);
    } else {
      setLoadError(active.body.error ?? `HTTP ${active.status}`);
    }
    if (hist.ok) setHistory(hist.body.entries ?? []);
  }, []);
  useEffect(() => {
    queueMicrotask(() => void reload());
  }, [reload]);
  useEffect(() => {
    if (presetPlayer) {
      queueMicrotask(() => {
        setScope("player");
        setField("skill_delta");
        setTarget(String(presetPlayer));
        onPresetUsed();
      });
    }
  }, [presetPlayer, onPresetUsed]);
  useEffect(() => {
    queueMicrotask(() => setEventKey(eventUid));
  }, [eventUid]);

  const spec = specFor(scope, field);
  const fieldsForScope = specs.filter((s) => s.scope === scope);
  const playerName = useMemo(() => new Map(players.map((p) => [String(p.dg_id), String(p.name ?? "")])), [players]);
  const effectiveCut = obj(eventObj.cut_rule);

  const draftValue = (): unknown => {
    if (!spec) return undefined;
    if (spec.kind === "number") return value.trim() === "" ? undefined : Number(value);
    if (spec.kind === "bool") return true;
    const out: Record<string, number> = {};
    for (const [k, v] of Object.entries(cut)) if (v.trim() !== "") out[k] = Number(v);
    return out;
  };
  const expiresMs = (() => {
    const ms = new Date(expires).getTime();
    return Number.isNaN(ms) ? null : ms;
  })();
  const draft = (): Record<string, unknown> => ({
    id: "draft",
    created_at: isoSeconds(nowMs),
    author: "me",
    event: eventKey,
    scope,
    ...(scope === "player" && target ? { target: Number(target) } : {}),
    field,
    value: draftValue(),
    reason: reason.trim(),
    expires_at: expiresMs === null ? "" : isoSeconds(expiresMs),
  });
  const problems = (() => {
    const found = checkOverride(draft());
    // "missing value" while the field is still blank is shown as a hint, not an error banner
    return found;
  })();
  const player = scope === "player" && target ? players.find((p) => String(p.dg_id) === target) : undefined;
  const impact = (() => {
    if (!player || !spec || spec.kind !== "number") return null;
    const ch = obj(player.challenger);
    const v = Number(value);
    if (!Number.isFinite(v) || value.trim() === "") return null;
    if (field === "skill_delta") return `mu ${signed(ch.mu)} becomes about ${signed((num(ch.mu) ?? 0) + v)} (before the field is re-centred).`;
    if (field === "sd_mult") return `Round SD ${fx(ch.sd, 2)} becomes about ${fx((num(ch.sd) ?? 0) * v, 2)}.`;
    return null;
  })();

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setMessage(null);
    if (problems.length) {
      setMessage({ tone: "error", text: "Fix these before saving", problems });
      return;
    }
    setBusy(true);
    const body = { event: eventKey, scope, ...(scope === "player" ? { target: Number(target) } : {}), field, value: draftValue(), reason: reason.trim(), expires_at: isoSeconds(expiresMs as number) };
    const result = await apiJson<{ record: OverrideRecord; records: OverrideRecord[] }>("/api/overrides", { method: "POST", body: JSON.stringify(body) });
    setBusy(false);
    if (result.ok) {
      setMessage({ tone: "ok", text: `Saved ${result.body.record.id}. golfprice applies it on its next run (as-of after ${result.body.record.created_at}); the untouched number is still stored.` });
      setValue("");
      setCut({});
      setReason("");
      await reload();
    } else {
      setMessage({ tone: "error", text: result.status === 401 ? "Not signed in through Cloudflare Access; the change was refused." : result.body.error ?? `HTTP ${result.status}`, problems: result.body.problems });
    }
  }

  async function change(record: OverrideRecord, mode: "expire" | "remove") {
    if (!window.confirm(mode === "remove" ? `Remove ${record.id}? It stops applying and is deleted from the active list (the history log keeps it).` : `Expire ${record.id} now?`)) return;
    setBusy(true);
    const result = await apiJson(`/api/overrides/${encodeURIComponent(record.id)}?mode=${mode}`, { method: "DELETE" });
    setBusy(false);
    setMessage(result.ok ? { tone: "ok", text: `${mode === "remove" ? "Removed" : "Expired"} ${record.id}.` } : { tone: "error", text: result.body.error ?? `HTTP ${result.status}`, problems: result.body.problems });
    await reload();
  }

  return (
    <div className="stack-lg">
      <div className="inputs-banner">
        <strong>One-off fixes, applied directly to the model price</strong>
        <span>Each override appears as its own named line in the player breakdown. The untouched model number is stored for the forward test. Hard bounds reject, never clip. Saving updates the shared override list and records an audit entry with your Access identity; golfprice reads it at its next run.</span>
      </div>
      <Panel eyebrow="New override" title="Adjust the model">
        <form className="override-form" onSubmit={submit}>
          <div className="form-grid">
            <Select label="Event" value={eventKey} onChange={setEventKey} options={[{ value: eventUid, label: `${String(eventObj.name ?? eventUid)} (${eventUid})` }, { value: "all", label: "All events" }]} />
            <Select label="Scope" value={scope} onChange={(v) => { setScope(v as Scope); const first = specs.find((s) => s.scope === v); setField(first?.field ?? ""); setValue(""); }} options={[{ value: "player", label: "Player" }, { value: "course", label: "Course" }, { value: "engine", label: "Engine (cut rule)" }]} />
            <Select label="Field" value={field} onChange={(v) => { setField(v); setValue(""); }} options={fieldsForScope.map((s) => ({ value: s.field, label: titleCase(s.field) }))} />
            {scope === "player" && (
              <Select label="Player" value={target} onChange={setTarget} options={[{ value: "", label: "Choose a player…" }, ...players.map((p) => ({ value: String(p.dg_id), label: `${String(p.name)} (${String(p.dg_id)})` }))]} />
            )}
          </div>
          {spec && <p className="inputs-note">{spec.what}. Unit: {spec.unit}.{spec.kind === "number" ? ` Allowed ${spec.lo} to ${spec.hi}.` : ""}</p>}
          {spec?.kind === "number" && (
            <label className="field-label">
              <span>Value</span>
              <input type="number" step="0.01" min={spec.lo} max={spec.hi} value={value} onChange={(e) => setValue(e.target.value)} placeholder={`${spec.lo} to ${spec.hi}`} />
            </label>
          )}
          {spec?.kind === "bool" && <p className="inputs-note"><b>Withdraw</b> is stored as true: the player is removed from the field and the rest is re-simulated without him.</p>}
          {spec?.kind === "cut_rule" && (
            <div className="form-grid">
              {Object.entries(CUT_BOUNDS).map(([key, [lo, hi]]) => (
                <label className="field-label" key={key}>
                  <span>{titleCase(key)} ({lo} to {hi})</span>
                  <input type="number" step="1" min={lo} max={hi} value={cut[key] ?? ""} onChange={(e) => setCut({ ...cut, [key]: e.target.value })} placeholder={`now ${String(effectiveCut[key] ?? "—")}`} />
                </label>
              ))}
            </div>
          )}
          {impact && <p className="inputs-note accent">{impact}</p>}
          <label className="field-label">
            <span>Reason (required, at least 5 characters)</span>
            <textarea value={reason} onChange={(e) => setReason(e.target.value)} rows={2} maxLength={300} placeholder="e.g. reported back injury in Tuesday presser" />
          </label>
          <div className="form-grid">
            <label className="field-label">
              <span>Expires (required, within {MAX_LIFETIME_DAYS} days)</span>
              <input type="datetime-local" value={expires} onChange={(e) => setExpires(e.target.value)} />
            </label>
            <div className="quick-expiry">
              {[1, 3, 7, 14].map((days) => (
                <button type="button" className="inputs-button subtle" key={days} onClick={() => setExpires(localInput(Date.now() + days * 86_400_000))}>+{days}d</button>
              ))}
            </div>
          </div>
          {problems.length > 0 && (reason.trim() || value.trim() || Object.keys(cut).length > 0) && (
            <ul className="problem-list">{problems.filter((p) => !p.startsWith("missing") || reason.trim()).map((p) => <li key={p}>{p}</li>)}</ul>
          )}
          <div className="form-actions">
            <button type="submit" className="inputs-button primary" disabled={busy || problems.length > 0}>{busy ? "Saving…" : "Save override"}</button>
            <span className="inputs-muted">Server re-checks every bound; times are stored in UTC.</span>
          </div>
          {message && (
            <div className={`inputs-banner ${message.tone === "error" ? "warn" : ""}`} role="status">
              <strong>{message.text}</strong>
              {message.problems && <ul className="problem-list">{message.problems.map((p) => <li key={p}>{p}</li>)}</ul>}
            </div>
          )}
        </form>
      </Panel>
      <Panel eyebrow="In the bucket" title="Active overrides" actions={<button type="button" className="inputs-button subtle" onClick={() => void reload()}>Refresh</button>}>
        {loadError && <ErrorState message={`Could not read overrides (${loadError}). Local previews have no bucket.`} />}
        {!loadError && records === null && <LoadingState label="Loading overrides" />}
        {records && records.length === 0 && <EmptyState title="No overrides" detail="golfprice prices the model untouched." />}
        {records && records.length > 0 && (
          <div className="table-scroll">
            <table>
              <thead><tr>{["Status", "Event", "Scope", "Target", "Field", "Value", "Reason", "Author", "Created", "Expires", ""].map((h) => <th key={h}><span className="th-text">{h}</span></th>)}</tr></thead>
              <tbody>
                {records.map((record) => {
                  const expired = isExpired(record, nowMs);
                  const here = record.event === "all" || record.event === eventUid || eventUid.startsWith(`${record.event}:`);
                  return (
                    <tr key={record.id} className={expired ? "dim-row" : ""}>
                      <td>{expired ? "Expired" : here ? "Active" : "Other event"}</td>
                      <td>{record.event}</td>
                      <td>{record.scope}</td>
                      <td>{record.target !== undefined ? `${playerName.get(String(record.target)) ?? ""} (${record.target})` : "—"}</td>
                      <td>{record.field}</td>
                      <td>{describeValue(record)}</td>
                      <td className="wrap-cell">{record.reason}</td>
                      <td>{record.author}</td>
                      <td>{record.created_at}</td>
                      <td>{record.expires_at}</td>
                      <td className="row-actions">
                        {!expired && <button type="button" className="inputs-button subtle" disabled={busy} onClick={() => void change(record, "expire")}>Expire now</button>}
                        <button type="button" className="inputs-button danger" disabled={busy} onClick={() => void change(record, "remove")}>Remove</button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
      {schema?.fields && <p className="inputs-muted">Bounds shown here match the published override limits ({schema.fields.length} fields).</p>}
      <Panel eyebrow="Audit" title="Recent changes">
        {history.length === 0 ? <p className="inputs-muted">No changes logged yet.</p> : (
          <div className="kv-table">
            {history.map((entry) => (
              <div key={`${entry.at}-${entry.record.id}-${entry.action}`}>
                <span>{entry.at.replace("T", " ").replace("Z", " UTC")} · {entry.action}</span>
                <b>{entry.by.email}{entry.by.verified ? "" : " (unverified)"} · {entry.record.scope}/{entry.record.field} {describeValue(entry.record)} · {entry.record.id}</b>
              </div>
            ))}
          </div>
        )}
      </Panel>
    </div>
  );
}

/* ------------------------------------------------------------------ page */
export function InputsView() {
  const { data: index, loading, error } = useDashboardData<PublishIndex>("golfprice/index.json");
  const [eventUid, setEventUid] = useState<string>("");
  const [runKey, setRunKey] = useState<string>("");
  const [tab, setTab] = useState<Tab>(() => {
    // Deep link: /inputs?tab=course|variance|weather|odds|config|features|adjust
    const wanted = typeof window === "undefined" ? null : new URLSearchParams(window.location.search).get("tab");
    return TABS.find((t) => t.value === wanted)?.value ?? "players";
  });
  const [presetPlayer, setPresetPlayer] = useState<number | null>(null);

  const events = useMemo(() => index?.events ?? [], [index]);
  const event = events.find((e) => e.event_uid === eventUid) ?? events[0];
  const runs = event?.runs ?? [];
  const run = runs.find((r) => r.key === runKey) ?? runs.find((r) => r.key === event?.latest_key) ?? runs[0];
  const { data: doc, loading: docLoading, error: docError } = useDashboardData<Obj>(run?.key ?? "golfprice/none.json");
  const clearPreset = useCallback(() => setPresetPlayer(null), []);

  if (loading) return <LoadingState label="Loading published model inputs" />;
  if (error || !event) return <div><PageIntro eyebrow="Model" title="Model inputs" description="Every input that shapes the simulations: player skill, course fit, variance, weather, odds and config." /><ErrorState message={error ?? "No golfprice run has been published yet. Run golfprice/publish_dashboard.py."} /></div>;

  const asOf = run?.as_of ? run.as_of.replace("T", " ").replace("Z", " UTC") : "—";
  return (
    <div>
      <PageIntro
        eyebrow="Model"
        title="Model inputs"
        description="What went into the simulation: player skill components, course and hole table, variance and engine settings, weather, odds and config. Features explains every skill-model feature; Adjust applies one-off owner overrides."
        controls={
          <div className="control-row wrap">
            <Select label="Event" value={event.event_uid} onChange={(v) => { setEventUid(v); setRunKey(""); }} options={events.map((e) => ({ value: e.event_uid, label: `${e.name} (${e.tour.toUpperCase()})` }))} />
            <Select label="Run" value={run?.key ?? ""} onChange={setRunKey} options={runs.map((r) => ({ value: r.key, label: `${r.kind === "live" ? `Live after R${r.after_round}` : "Week"} · ${r.as_of ?? r.run}` }))} />
          </div>
        }
      />
      <div className="inputs-meta">
        <span><b>{event.course}</b></span>
        <span>{event.date_start} to {event.date_end}</span>
        <span>as-of {asOf}</span>
        <span>{run?.n_players} players</span>
        <span>overrides applied: {run?.overrides_applied.length ?? 0}</span>
        <span className="sha">sha {run?.sha256.slice(0, 12)}</span>
      </div>
      <SegmentedControl label="Model inputs area" value={tab} onChange={setTab} options={TABS.map((t) => ({ value: t.value, label: t.label }))} />
      {docLoading && <LoadingState label="Loading run" />}
      {docError && <ErrorState message={docError} />}
      {doc && tab === "players" && <PlayersTab doc={doc} onAdjust={(id) => { setPresetPlayer(id); setTab("adjust"); }} />}
      {doc && tab === "course" && <CourseTab doc={doc} />}
      {doc && tab === "variance" && <VarianceTab doc={doc} />}
      {doc && tab === "weather" && <WeatherTab doc={doc} />}
      {doc && tab === "odds" && <OddsTab doc={doc} />}
      {doc && tab === "config" && <ConfigTab doc={doc} />}
      {tab === "features" && <FeaturesTab />}
      {doc && tab === "adjust" && <AdjustTab doc={doc} eventUid={event.event_uid} presetPlayer={presetPlayer} onPresetUsed={clearPreset} />}
    </div>
  );
}
