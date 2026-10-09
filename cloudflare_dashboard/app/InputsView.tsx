"use client";

import { Fragment, useCallback, useEffect, useMemo, useState, useSyncExternalStore, type ReactNode } from "react";
import { Search } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { DataTable, EmptyState, ErrorState, Kpi, LoadingState, PageIntro, Panel, SegmentedControl, humanValue } from "./components";
import { useDashboardData } from "./data";
import { LABELS } from "./labels";
import { matchPlayers, matchText } from "./inputs-search";
import { DataRow, etTime, numberValue, palette, titleCase } from "./lib";
import { GLOSSARY, HEADERS, define, registerFamilyLabels, titlesFor } from "./glossary";
import { ALLOWED, CUT_BOUNDS, MAX_LIFETIME_DAYS, checkOverride, isExpired, isoSeconds, specFor, type FieldSpec, type OverrideRecord, type Scope } from "./overrides-rules";

/* ------------------------------------------------------------------ loose shapes of golfprice.model_inputs.v1 (see golfprice/INPUTS_SCHEMA.md) */
type Obj = Record<string, unknown>;
const obj = (value: unknown): Obj => (value && typeof value === "object" && !Array.isArray(value) ? (value as Obj) : {});
const arr = (value: unknown): Obj[] => (Array.isArray(value) ? (value as Obj[]) : []);
const num = (value: unknown): number | null => (typeof value === "number" && Number.isFinite(value) ? value : null);
const fx = (value: unknown, digits = 2): string => {
  const v = num(value);
  if (v === null) return "—";
  const text = v.toFixed(digits);
  return /^-0(\.0+)?$/.test(text) ? text.slice(1) : text;
};
const signed = (value: unknown, digits = 2): string => {
  const v = num(value);
  if (v === null) return "—";
  const text = v.toFixed(digits);
  if (/^-?0(\.0+)?$/.test(text)) return text.replace("-", "");
  return `${v > 0 ? "+" : ""}${text}`;
};
/** A probability as a percentage; a real but tiny chance reads "<0.1%" rather than a misleading "0.0%". */
const pct = (value: unknown, digits = 1): string => {
  const v = num(value);
  if (v === null) return "—";
  if (v > 0 && v < 0.001 && digits <= 1) return "<0.1%";
  return `${(v * 100).toFixed(digits)}%`;
};

/* ------------------------------------------------------------------ plain-English helpers */
/** Hover-definition wrapper: dotted underline plus a native tooltip. */
function Term({ k, children, text }: { k?: string; children: ReactNode; text?: string }) {
  const tip = text ?? (k ? define(k) : "");
  if (!tip) return <>{children}</>;
  return <span title={tip} style={{ cursor: "help", borderBottom: "1px dotted var(--muted)" }}>{children}</span>;
}

/** A KPI card with a hover definition (the shared Kpi has no tooltip prop, so this wrapper adds one without changing layout). */
function Tip({ text, children }: { text?: string; children: ReactNode }) {
  return <div style={{ display: "contents" }} title={text}>{children}</div>;
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
/** "2026-10-05" or "2026-10-05 06:00:00" (a calendar date, no zone) -> "Oct 5". */
function ymd(value: unknown): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(value ?? ""));
  return m ? `${MONTHS[Number(m[2]) - 1]} ${Number(m[3])}` : "";
}
/** Snapshot ids like "2026-10-05T23-02-17-359780Z" are UTC clock times; show them in Eastern. */
function snapshotTime(id: unknown): string {
  const m = /^(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})/.exec(String(id ?? ""));
  return m ? etTime(`${m[1]}T${m[2]}:${m[3]}:${m[4]}Z`, "") : "";
}
/** "2026-10-07 21:51:30 UTC" -> Eastern. */
function utcText(value: unknown): string {
  const m = /^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})/.exec(String(value ?? ""));
  return m ? etTime(`${m[1]}T${m[2]}Z`, "") : "";
}

/** Publisher-supplied phrases translated for a golf bettor (display side only; the published text is unchanged). */
const PHRASES: Array<[RegExp, string]> = [
  [/kalman trend/gi, "recent form trend"],
  [/kalman/gi, "form-tracking"],
  [/cross-tour/gi, "tour strength adjustment"],
  [/\bDPWT\b/g, "DP World Tour"],
  [/market combiner/gi, "market blend"],
  [/\bseeds?\b/gi, "batches"],
];
function plain(text: unknown): string {
  let out = String(text ?? "");
  for (const [pattern, replacement] of PHRASES) out = out.replace(pattern, replacement);
  return out;
}
function joinAnd(items: string[]): string {
  return items.length <= 1 ? items.join("") : `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}
/** Hours -> "5 min" or "4 h 30 min". */
function ageText(hours: number | null): string {
  if (hours === null) return "";
  const minutes = Math.round(hours * 60);
  if (minutes < 60) return `${Math.max(minutes, 0)} min`;
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return m ? `${h} h ${m} min` : `${h} h`;
}
const BOOK_NAMES: Record<string, string> = {
  draftkings: "DraftKings", fanduel: "FanDuel", betmgm: "BetMGM", williamhill: "William Hill", skybet: "Sky Bet", bet365: "Bet365", caesars: "Caesars", pointsbet: "PointsBet", betonline: "BetOnline",
  paddypower: "Paddy Power", betrivers: "BetRivers", betfred: "Betfred", bwin: "bwin", betway: "Betway", "888sport": "888sport", unibet: "Unibet", ladbrokes: "Ladbrokes", coral: "Coral", betcris: "Betcris", bovada: "Bovada", pinnacle: "Pinnacle", circa: "Circa", espnbet: "ESPN BET",
};
const bookName = (key: unknown) => BOOK_NAMES[String(key ?? "").toLowerCase().replace(/[\s_-]/g, "")] ?? titleCase(key);
const MARKET_NAMES: Record<string, string> = { win: "Win", top_5: "Top 5", top_10: "Top 10", top_20: "Top 20", make_cut: "Make cut", mc: "Make cut", miss_cut: "Miss cut" };
const marketName = (key: string) => MARKET_NAMES[key] ?? titleCase(key);

/** Cut rule as a sentence. `rule` is the effective rule or the registry rule; both share one shape. */
function cutRuleSentence(rule: Obj): string {
  const rounds = num(rule.rounds) ?? 4;
  const cutRound = num(rule.cut_round) ?? 0;
  const source = String(rule.source ?? "");
  const where = source.startsWith("registry") ? " (from the event schedule)" : source === "dpwt_top65" ? " (standard DP World Tour rule)" : source === "remaining" ? " (rest of the event)" : "";
  if (!cutRound) return `${rounds} rounds, no cut${where}`;
  let text = `Cut after round ${cutRound}: top ${num(rule.top_n) ?? "?"} and ties`;
  const within = num(rule.within) ?? 0;
  if (within) text += `, plus anyone within ${within} shots of the lead`;
  const mdfTrigger = num(rule.mdf_trigger) ?? 0;
  if (mdfTrigger) text += `; second cut (MDF) after round ${num(rule.mdf_round) ?? 3}: top ${num(rule.mdf_top_n) ?? "?"} and ties${mdfTrigger < 0 ? " (always)" : ` (only if more than ${mdfTrigger} are left)`}`;
  return `${text}${where}`;
}

/** Tee times arrive in the course's local clock. The offset to UTC is read from the event's first round-1 tee (UTC) against the earliest wave's local time. `text` is the Eastern time; `local` is the course clock ("Thu 10:46 AM"); `both` joins them. */
type Tee = { text: string; local: string; both: string };
function teeFormatter(doc: Obj): (local: unknown) => Tee {
  const waves = arr(obj(doc.weather).waves).filter((w) => num(w.round) === 1 && typeof w.first_tee_local === "string");
  const firstUtc = Date.parse(String(obj(doc.event).first_r1_tee_utc ?? ""));
  const naive = (text: string) => Date.parse(`${text.trim().replace(" ", "T").slice(0, 19)}Z`);
  const firstLocal = waves.length ? Math.min(...waves.map((w) => naive(String(w.first_tee_local)))) : Number.NaN;
  const offset = Number.isFinite(firstUtc) && Number.isFinite(firstLocal) ? Math.round((firstLocal - firstUtc) / 900_000) * 900_000 : null;
  const clock = (ms: number) => {
    const d = new Date(ms);
    const h = d.getUTCHours();
    return `${["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][d.getUTCDay()]} ${h % 12 || 12}:${String(d.getUTCMinutes()).padStart(2, "0")} ${h < 12 ? "AM" : "PM"}`;
  };
  return (local) => {
    const text = typeof local === "string" ? local : "";
    if (!text) return { text: "", local: "", both: "" };
    const ms = naive(text);
    if (offset === null || !Number.isFinite(ms)) {
      const plainText = text.replace(/:\d{2}$/, "");
      return { text: plainText, local: "", both: plainText };
    }
    const et = etTime(new Date(ms - offset).toISOString(), "");
    const here = clock(ms);
    return { text: et, local: here, both: `${here} local (${et})` };
  };
}

/** The hole-table label is a registry string; turn it into one short sentence. The raw string goes in the Technical details box. */
function describeHoleTable(rawLabel: string, layout: Obj): string {
  if (!rawLabel || !/proxy edition at/.test(rawLabel)) return "";
  const nMed = Number(/n_med=(\d+)/.exec(rawLabel)?.[1] ?? NaN);
  if (nMed === 0) return "No past rounds here yet, so every hole is estimated from par and length.";
  let text = `Hole data is from last year's edition${Number.isFinite(nMed) ? ` (${nMed} rounds per hole)` : ""}.`;
  if (layout.current_event_confirmed === false) text += " This year's layout isn't confirmed.";
  return text;
}

/** Drop columns that carry no information: empty everywhere (or listed as constant-hidden and the same in every row). */
function dropEmptyColumns(rows: DataRow[], alsoIfConstant: string[] = [], alsoIfZero: string[] = []): DataRow[] {
  if (!rows.length) return rows;
  const keys = Object.keys(rows[0]);
  const drop = new Set<string>();
  for (const key of keys) {
    const values = rows.map((row) => row[key]);
    if (values.every((v) => v === null || v === undefined || v === "")) drop.add(key);
    else if (alsoIfConstant.includes(key) && values.every((v) => v === values[0])) drop.add(key);
    else if (alsoIfZero.includes(key) && values.every((v) => v === 0)) drop.add(key);
  }
  return drop.size ? rows.map((row) => Object.fromEntries(Object.entries(row).filter(([key]) => !drop.has(key)))) : rows;
}

function Technical({ children, label = "Technical details" }: { children: ReactNode; label?: string }) {
  return (
    <details className="config-file">
      <summary><b>{label}</b><span>for support</span></summary>
      <div className="stack-lg">{children}</div>
    </details>
  );
}

function armLabel(name: string): string {
  const known: Record<string, string> = { champion: "previous model", comparison: "alternative version", shadow: "alternative version" };
  return known[name] ?? name.replaceAll("_", " ");
}
type IndexRun = { kind: string; run: string; as_of: string | null; after_round: number | null; key: string; hole_table_key: string | null; sha256: string; bytes: number; n_players: number; overrides_applied: string[]; overrides_rejected: string[] };
type IndexEvent = { event_uid: string; name: string; tour: string; date_start: string; date_end: string; course: string; runs: IndexRun[]; latest_key: string | null };
type PublishIndex = { schema: string; events: IndexEvent[] };

const FAMILIES: Array<[string, string]> = [
  ["level_form", "Long-run skill and form"],
  ["kalman", "Current form trend"],
  ["xtour", "Results on other tours"],
  ["category", "Skill by part of the game"],
  ["sit", "Layoff and age"],
  ["act", "Recent activity"],
  ["sklv", "Skill level estimate"],
  ["course", "Course fit and history"],
  ["thin", "Small-sample adjustment"],
  ["disp", "Blow-up rounds"],
  ["other", "Everything else"],
];
registerFamilyLabels(FAMILIES);

const TABS = [
  { value: "players", label: "Players" },
  { value: "course", label: "Course" },
  { value: "variance", label: "Round swings" },
  { value: "weather", label: "Weather and waves" },
  { value: "odds", label: "Odds freshness" },
  { value: "config", label: "Run info" },
  { value: "features", label: "Features" },
  { value: "adjust", label: "Adjust" },
] as const;
type Tab = (typeof TABS)[number]["value"];

function ChartTip({ active, payload, label }: { active?: boolean; payload?: Array<{ name?: string; value?: unknown; color?: string; payload?: { name?: string; hint?: string } }>; label?: unknown }) {
  if (!active || !payload?.length) return null;
  const hint = payload[0]?.payload?.hint;
  return (
    <div className="chart-tooltip">
      <strong>{String(label ?? payload[0]?.payload?.name ?? "")}</strong>
      {hint && <span>{hint}</span>}
      {payload.map((item, index) => (
        <span key={`${item.name}-${index}`} style={{ color: item.color }}>
          {item.name}: {Array.isArray(item.value) ? item.value.map((x) => Number(x).toFixed(2)).join(" to ") : typeof item.value === "number" ? (Number.isInteger(item.value) ? String(item.value) : item.value.toFixed(2)) : String(item.value ?? "—")}
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
    const text = typeof value === "number" ? String(Math.round(value * 1e6) / 1e6) : typeof value === "string" ? value : humanValue(value);
    out.push({ key: prefix, value: text === undefined ? "—" : text.length > 240 ? `${text.slice(0, 240)}…` : text });
  }
  return out;
}

/** Timestamps in the technical list read in Eastern like everywhere else; plain text passes through. */
function plainStamp(key: string, text: string): string {
  if (/^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}/.test(text)) return snapshotTime(text) || text;
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?(\.\d+)?Z$/.test(text)) return etTime(text, text);
  if (/ UTC$/.test(text)) return utcText(text) || text;
  if (/_at$/.test(key) && /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/.test(text)) return etTime(`${text.replace(" ", "T")}Z`, text);
  return text;
}

function KeyValue({ data, empty = "Nothing recorded for this run." }: { data: unknown; empty?: string }) {
  const rows = flatten(data).filter((row) => row.value !== "—" && row.value !== "");
  if (!rows.length || (rows.length === 1 && !rows[0].key)) return empty ? <p className="inputs-muted">{empty}</p> : null;
  return (
    <div className="kv-table">
      {rows.map((row) => (
        <div key={row.key}>
          <span>{row.key.split(".").map((part) => part.replaceAll("_", " ")).join(" › ")}</span>
          <b>{plainStamp(row.key, row.value)}</b>
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ players */
function playerRows(players: Obj[], showUntouched: boolean): DataRow[] {
  const rows = players.map((p) => {
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
      [LABELS.thisWeekPga.short]: num(ch.mu_tour),
      sd: num(ch.sd),
      se_kernel: num(ch.se_kernel),
      location: num(loc.total),
      course_fit: num(cf.contribution_to_mu),
      fit_rs_ddacc: num(cf.fit_rs_ddacc_lam1000),
      course_history: num(cf.course_history_resid_k80),
      course_sd_mult: num(cf.course_sd_mult_feature),
      override_total: num(ovr.total),
      prior_rounds: num(p.n_prior_rounds),
      prob_win: num(prob.p_win),
      prob_top_5: num(prob.p_top_5),
      prob_top_10: num(prob.p_top_10),
      prob_top_20: num(prob.p_top_20),
      prob_make_cut: num(prob.p_make_cut),
      country: String(p.country ?? ""),
      amateur: Boolean(p.amateur),
      dg_id: num(p.dg_id),
    };
    if (showUntouched) {
      row.mu_untouched = num(ch.mu_untouched);
      row.sd_untouched = num(ch.sd_untouched);
    }
    for (const [key] of FAMILIES) row[`chl_${key}`] = num(fam[key]);
    return row;
  });
  return dropEmptyColumns(rows, ["course_sd_mult"], ["se_kernel", "override_total"]);
}

const differs = (a: unknown, b: unknown) => {
  const x = num(a);
  const y = num(b);
  return x !== null && y !== null && Math.abs(x - y) > 1e-9;
};
const playerHasOverride = (p: Obj) => {
  const ch = obj(p.challenger);
  return differs(ch.mu, ch.mu_untouched) || differs(ch.sd, ch.sd_untouched) || (Array.isArray(ch.overrides) && ch.overrides.length > 0) || (num(obj(obj(ch.breakdown).override).total) ?? 0) !== 0;
};

type Step = { name: string; value: number; hint: string; range: [number, number]; end: number };
type Part = { name: string; value: number; hint: string };
/** The pieces of the rating. `parts` lists every non-zero piece; `steps` is the chart version, with the small pieces grouped into one "Everything else" bar. */
function waterfall(player: Obj, overridden: boolean): { steps: Step[]; parts: Part[] } {
  const ch = obj(player.challenger);
  const bd = obj(ch.breakdown);
  const fam = obj(bd.chl_families);
  const loc = obj(bd.location);
  const ovr = obj(bd.override);
  const all: Part[] = [];
  for (const [key, label] of FAMILIES) all.push({ name: label, value: num(fam[key]) ?? 0, hint: define(`chl_${key}`) });
  all.push({ name: "Location (home, travel, nationality)", value: num(loc.total) ?? 0, hint: define("location") });
  if (overridden) all.push({ name: "Your override", value: num(ovr.total) ?? 0, hint: define("override") });
  const parts = all.filter((part) => Math.abs(part.value) >= 0.005);
  const big = parts.filter((part) => Math.abs(part.value) >= 0.03 || part.name === "Your override");
  const small = parts.filter((part) => !big.includes(part));
  const chart = [...big];
  if (small.length) chart.push({ name: "Everything else", value: small.reduce((total, part) => total + part.value, 0), hint: `Smaller pieces combined: ${small.map((part) => part.name.toLowerCase()).join(", ")}.` });
  let running = 0;
  const steps: Step[] = chart.map((part) => {
    const start = running;
    running += part.value;
    return { ...part, range: [Math.min(start, running), Math.max(start, running)], end: running };
  });
  steps.push({ name: "Final rating vs field", value: running, hint: define("mu"), range: [Math.min(0, running), Math.max(0, running)], end: running });
  return { steps, parts };
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

const THIN_ROUNDS = 50;

function PlayerDetail({ player, choices, onPick, onAdjust, fmtTee, playedRounds }: { player: Obj; choices: PlayerChoice[]; onPick: (dgId: number) => void; onAdjust: (dgId: number) => void; fmtTee: (local: unknown) => Tee; playedRounds: number }) {
  const ch = obj(player.challenger);
  const bd = obj(ch.breakdown);
  const overridden = playerHasOverride(player);
  const { steps, parts } = useMemo(() => waterfall(player, overridden), [player, overridden]);
  const arms = Object.entries(obj(player.arms)).filter(([name]) => name !== "challenger" && name !== "challenger_untouched");
  const prob = obj(ch.probabilities);
  const probU = obj(ch.probabilities_untouched);
  const cf = obj(player.course_fit);
  const hist = obj(player.history);
  const tee = obj(player.tee);
  const loc = obj(bd.location);
  const weather = obj(ch.weather);
  // Rounds already played in a live run no longer move the price, so only the rounds still to come are listed.
  const weatherRows = Object.entries(weather).filter(([round, v]) => (num(v) ?? 0) !== 0 && Number(round.replace(/\D/g, "")) > playedRounds);
  const seKernel = num(ch.se_kernel) ?? 0;
  const rounds = num(player.n_prior_rounds);
  const thin = rounds !== null && rounds < THIN_ROUNDS;
  const priceRows: Array<{ label: string; active?: boolean; mu: unknown; sd: unknown; p: Obj }> = [
    { label: overridden ? "With your override" : "Model price", active: true, mu: ch.mu, sd: ch.sd, p: prob },
    ...(overridden && Object.keys(probU).length > 0 ? [{ label: "Before your override", mu: ch.mu_untouched, sd: ch.sd_untouched, p: probU }] : []),
    ...arms.map(([name, value]) => ({ label: armLabel(name), mu: obj(value).mu, sd: obj(value).sd, p: { p_win: obj(value).p_win, p_top_5: obj(value).p_top_5, p_top_10: obj(value).p_top_10, p_top_20: obj(value).p_top_20, p_make_cut: obj(value).p_make_cut } as Obj })),
  ];
  const showCut = priceRows.some((row) => num(row.p.p_make_cut) !== null);
  const th = (label: string, key?: string) => <th><span className="th-text"><Term k={key}>{label}</Term></span></th>;
  const priceCells: Array<[string, string, string]> = [["Win", "p_win", "prob_win"], ["Top 5", "p_top_5", "prob_top_5"], ["Top 10", "p_top_10", "prob_top_10"], ["Top 20", "p_top_20", "prob_top_20"], ...(showCut ? ([["Make cut", "p_make_cut", "prob_make_cut"]] as Array<[string, string, string]>) : [])];
  const teeRows = Object.entries(tee).flatMap(([round, info]) => {
    const t = fmtTee(obj(info).teetime_local);
    if (!t.text) return [];
    const wave = obj(info).wave ? ` (${String(obj(info).wave)} wave)` : "";
    return [{ round: round.replace(/\D/g, ""), text: t.local ? `${t.local} local${wave} · ${t.text}` : `${t.text}${wave}` }];
  });
  const courseLines = [
    num(cf.fit_rs_ddacc_lam1000) !== null,
    num(cf.course_history_resid_k80) !== null,
    num(cf.course_sd_mult_feature) !== null,
    num(cf.sd_shrunk_feature) !== null,
    num(hist.champion_mu_j2) !== null,
    num(hist.se_mu) !== null,
    teeRows.length > 0,
  ].some(Boolean);
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
        <div><span><Term k="mu">Rating vs this week&apos;s field</Term></span><strong>{signed(ch.mu)}</strong></div>
        {overridden && <div><span><Term k="mu_untouched">Rating before your override</Term></span><strong>{signed(ch.mu_untouched)}</strong></div>}
        {num(ch.mu_tour) !== null && <div title={`${LABELS.thisWeekPga.long}. Reference only; prices use the field-relative number.`}><span>{LABELS.thisWeekPga.short}</span><strong>{signed(ch.mu_tour)}</strong></div>}
        <div><span><Term k="sd">Round swing</Term></span><strong>{fx(ch.sd, 2)}</strong></div>
        {overridden && <div><span><Term k="sd_untouched">Swing before your override</Term></span><strong>{fx(ch.sd_untouched, 2)}</strong></div>}
        {seKernel !== 0 && <div><span><Term k="se_kernel">Rating uncertainty (±)</Term></span><strong>{fx(seKernel, 2)}</strong></div>}
        <div><span><Term k="prior_rounds">Rounds of history</Term></span><strong>{fx(player.n_prior_rounds, 0)}{thin ? " (thin data)" : ""}</strong></div>
        {overridden && <div><span><Term k="override_total">Your override</Term></span><strong>{signed(obj(bd.override).total)}</strong></div>}
      </div>
      {thin && <p className="inputs-muted">{GLOSSARY.thin_data}</p>}
      <h3 className="inputs-h3">How the rating is built (strokes per round, relative to the field)</h3>
      <p className="inputs-muted">Hover a bar for what that piece means.</p>
      <div className="chart-medium waterfall">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={steps} layout="vertical" margin={{ top: 6, right: 24, bottom: 6, left: 8 }}>
            <CartesianGrid stroke="var(--line)" horizontal={false} />
            <XAxis type="number" tickCount={7} tickFormatter={(v: number) => (Math.round(v * 100) / 100).toFixed(2)} tick={{ fill: "var(--muted)", fontSize: 10 }} />
            <YAxis type="category" dataKey="name" width={200} tick={{ fill: "var(--muted-strong)", fontSize: 10 }} />
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
            {parts.map((part) => (
              <div key={part.name}><span><Term text={part.hint}>{part.name}</Term></span><b>{signed(part.value)}</b></div>
            ))}
            {num(loc.total) !== null && Math.abs(num(loc.total) ?? 0) >= 0.005 && (
              <>
                {Math.abs(num(loc.h2_home_base_travel) ?? 0) >= 0.005 && <div><span className="inputs-muted"><Term k="location_home_travel">of which home base and travel</Term></span><b>{signed(loc.h2_home_base_travel)}</b></div>}
                {Math.abs(num(loc.nat_nationality) ?? 0) >= 0.005 && <div><span className="inputs-muted"><Term k="location_nationality">of which nationality</Term></span><b>{signed(loc.nat_nationality)}</b></div>}
                {Math.abs(num(loc.reest_chl_refit) ?? 0) >= 0.005 && <div><span className="inputs-muted"><Term k="location_refit">of which refit correction</Term></span><b>{signed(loc.reest_chl_refit)}</b></div>}
              </>
            )}
            <div><span><Term k="skill_total">Skill before location</Term></span><b>{signed(bd.chl_total)}</b></div>
            {Math.abs(num(bd.sum_error_vs_mu) ?? 0) > 0.0005 && <div><span><Term k="rounding">Rounding difference</Term></span><b>{fx(bd.sum_error_vs_mu, 3)}</b></div>}
            {weatherRows.map(([round, value]) => (
              <div key={round}><span><Term k="weather_round">{`Weather, round ${round.replace(/\D/g, "")} (added to the rating that round)`}</Term></span><b>{signed(value)}</b></div>
            ))}
          </div>
        </div>
        {courseLines && (
          <div>
            <h3 className="inputs-h3">Course and tee times</h3>
            <div className="kv-table">
              {num(cf.fit_rs_ddacc_lam1000) !== null && <div><span><Term k="fit_rs_ddacc">Venue fit (distance and accuracy)</Term></span><b>{signed(cf.fit_rs_ddacc_lam1000, 3)}</b></div>}
              {num(cf.course_history_resid_k80) !== null && <div><span><Term k="course_history">Course history</Term></span><b>{signed(cf.course_history_resid_k80, 3)}</b></div>}
              {num(cf.course_sd_mult_feature) !== null && <div><span><Term k="course_sd_mult">Course spread factor</Term></span><b>{fx(cf.course_sd_mult_feature, 3)}</b></div>}
              {num(cf.sd_shrunk_feature) !== null && <div><span><Term k="sd_shrunk">Round swing before course adjustment</Term></span><b>{fx(cf.sd_shrunk_feature, 2)}</b></div>}
              {num(hist.champion_mu_j2) !== null && <div><span><Term text="The previous (champion) model's rating for this player, kept for comparison.">Previous model&apos;s rating</Term></span><b>{signed(hist.champion_mu_j2)}</b></div>}
              {num(hist.se_mu) !== null && <div><span><Term text="How uncertain the previous model was about this player's skill, in strokes per round.">Rating uncertainty (±, previous model)</Term></span><b>{fx(hist.se_mu, 2)}{hist.se_mu_imputed ? " (estimated: little history)" : ""}</b></div>}
              {teeRows.map((row) => <div key={row.round}><span>Round {row.round} tee time</span><b>{row.text}</b></div>)}
            </div>
          </div>
        )}
      </div>
      {priceRows.length === 1 ? (
        <p className="inputs-note">
          <b>Model price before blending with the market:</b>{" "}
          {priceCells.map(([label, key], index) => `${index ? ", " : ""}${label} ${pct(priceRows[0].p[key], key === "p_win" ? 2 : 1)}`).join("")}
          {showCut ? "" : ". No make-cut price: this event has no cut."}
        </p>
      ) : (
        <>
          <h3 className="inputs-h3">Model prices (before blending with the market)</h3>
          <div className="table-scroll">
            <table>
              <thead><tr>{th("Version")}{th("Rating", "mu")}{th("Round swing", "sd")}{priceCells.map(([label, , key]) => <Fragment key={key}>{th(label, key)}</Fragment>)}</tr></thead>
              <tbody>
                {priceRows.map((row) => (
                  <tr key={row.label} className={row.active ? "active-row" : ""}>
                    <td>{row.label}</td><td>{signed(row.mu)}</td><td>{fx(row.sd, 2)}</td>
                    {priceCells.map(([, key], index) => <td key={key}>{pct(row.p[key], index === 0 ? 2 : 1)}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {!showCut && <p className="inputs-muted">No make-cut price: this event has no cut.</p>}
        </>
      )}
    </Panel>
  );
}

const SIGNED_COLUMNS = new Set(["mu", "mu_untouched", "location", "course_fit", "override_total", "course_history", "fit_rs_ddacc", LABELS.thisWeekPga.short]);
const SWING_COLUMNS = new Set(["sd", "sd_untouched", "se_kernel", "course_sd_mult"]);

function PlayersTab({ doc, onAdjust, playedRounds }: { doc: Obj; onAdjust: (dgId: number) => void; playedRounds: number }) {
  const players = arr(doc.players);
  const showUntouched = useMemo(() => players.some(playerHasOverride), [players]);
  const rows = useMemo(() => playerRows(players, showUntouched), [players, showUntouched]);
  const [selected, setSelected] = useState<number | null>(null);
  const current = players.find((p) => num(p.dg_id) === selected) ?? players[0];
  const currentRow = rows.find((row) => row.dg_id === num(current?.dg_id)) ?? null;
  const choices = useMemo(() => players.flatMap((p) => (num(p.dg_id) === null ? [] : [{ name: String(p.name ?? ""), dg_id: num(p.dg_id) as number }])), [players]);
  const fmtTee = useMemo(() => teeFormatter(doc), [doc]);
  const present = new Set(rows.flatMap((row) => Object.keys(row)));
  const preferred = ["name", "mu", "mu_untouched", "sd", "sd_untouched", "se_kernel", "location", "course_fit", "course_history", "fit_rs_ddacc", "override_total", "prior_rounds", "prob_win", "prob_top_5", "prob_top_10", "prob_top_20", "prob_make_cut", LABELS.thisWeekPga.short, ...FAMILIES.map(([key]) => `chl_${key}`)];
  // One rating column by default; the PGA-scale version is one click away under Columns.
  const defaultColumns = ["name", "mu", "mu_untouched", "sd", "sd_untouched", "location", "course_fit", "override_total", "prior_rounds", "prob_win", "prob_top_10", "prob_make_cut"].filter((key) => present.has(key));
  const headerLabels: Record<string, string> = { ...HEADERS, [LABELS.thisWeekPga.short]: LABELS.thisWeekPga.short };
  const headerTitles = titlesFor([...present], { [LABELS.thisWeekPga.short]: GLOSSARY.mu_tour });
  const best = players.reduce<Obj | null>((top, p) => ((num(obj(p.challenger).mu) ?? -Infinity) > (num(obj(top?.challenger).mu) ?? -Infinity) ? p : top), null);
  const strength = obj(obj(doc.event).field_strength);
  const fieldOffset = num(strength.field_offset);
  const withOverride = players.filter(playerHasOverride).length;
  const sdMean = players.length ? players.reduce((total, p) => total + numberValue(obj(p.challenger).sd), 0) / players.length : 0;
  const nAmateur = players.filter((p) => p.amateur).length;
  if (!players.length) return <EmptyState title="No players in this run" detail="The run published no player objects." />;
  const gap = fieldOffset === null ? "" : Math.abs(fieldOffset).toFixed(2);
  return (
    <div className="stack-lg">
      <div className="kpi-grid">
        <Kpi label="Players" value={String(players.length)} detail={nAmateur ? `${nAmateur} amateur${nAmateur === 1 ? "" : "s"}` : "in the field"} tone="accent" />
        <Tip text={GLOSSARY.mu}><Kpi label="Strongest player" value={signed(num(obj(best?.challenger).mu), 2)} detail={`${String(best?.name ?? "")}: strokes per round better than this week's field average (the average is 0)`} /></Tip>
        {fieldOffset !== null && (
          <Tip text="How this field compares with a typical PGA Tour field, in strokes per round. Reference only: prices use the rating against this field.">
            <Kpi
              label="Field strength"
              value={signed(fieldOffset, 2)}
              detail={`This field is ${gap} strokes per round ${fieldOffset < 0 ? "weaker" : "stronger"} than a typical PGA Tour field, so every PGA-scale rating is ${gap} ${fieldOffset < 0 ? "lower" : "higher"}. Prices are not affected.${ymd(strength.vintage) ? ` Skills as of ${ymd(strength.vintage)}.` : ""}`}
            />
          </Tip>
        )}
        <Tip text="The average of the players' round swing, in strokes, including any override."><Kpi label="Average round swing" value={sdMean.toFixed(2)} detail="strokes, after overrides" /></Tip>
        {withOverride > 0 && <Tip text="How many players have a manual override on them for this event."><Kpi label="Players with an override" value={String(withOverride)} detail="the original model numbers are kept alongside" tone="positive" /></Tip>}
      </div>
      <Panel eyebrow="Saved model inputs" title="Every player, every component" actions={<span className="inputs-muted">Click a row, or search below, for the breakdown. Hover a column header for what it means.</span>}>
        <DataTable
          rows={rows}
          preferredColumns={preferred}
          defaultColumns={defaultColumns}
          headerLabels={headerLabels}
          headerTitles={headerTitles}
          renderCell={(column, value, row) => {
            if ((column === "mu_untouched" && !differs(row.mu, value)) || (column === "sd_untouched" && !differs(row.sd, value))) return "";
            if (column === "name") {
              const rounds = num(row.prior_rounds);
              return rounds !== null && rounds < THIN_ROUNDS ? <>{String(value)} <span className="inputs-muted" title={GLOSSARY.thin_data} style={{ fontStyle: "italic" }}>(thin data)</span></> : undefined;
            }
            if (column.startsWith("prob_")) return pct(value);
            if (column.startsWith("chl_") || SIGNED_COLUMNS.has(column)) return num(value) === null ? undefined : signed(value);
            if (SWING_COLUMNS.has(column)) return num(value) === null ? undefined : fx(value, 2);
            return undefined;
          }}
          label="Model inputs players"
          pageSize={30}
          onRowClick={(row) => setSelected(num(row.dg_id))}
          activeRow={currentRow}
        />
        {showUntouched && <p className="inputs-muted">The &quot;before your override&quot; columns stay empty for players you have not adjusted.</p>}
      </Panel>
      {current && <PlayerDetail player={current} choices={choices} onPick={setSelected} onAdjust={onAdjust} fmtTee={fmtTee} playedRounds={playedRounds} />}
    </div>
  );
}

/* ------------------------------------------------------------------ course */
/** Min / median / max of a list of numbers, or null when there are none. */
function spread(values: number[]): { min: number; median: number; max: number } | null {
  const v = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!v.length) return null;
  return { min: v[0], median: v[Math.floor(v.length / 2)], max: v[v.length - 1] };
}

function CourseTab({ doc, extra }: { doc: Obj; extra?: ReactNode }) {
  const course = obj(doc.course);
  const event = obj(doc.event);
  const tables = arr(course.hole_tables);
  const perRound = arr(course.per_round);
  const fit = obj(course.venue_fit);
  const slopes = obj(fit.slopes);
  const players = arr(doc.players);
  const [group, setGroup] = useState(0);
  const table = tables[Math.min(group, Math.max(0, tables.length - 1))];
  const holes = arr(table?.holes);
  const holeRows: DataRow[] = dropEmptyColumns(
    holes.map((h) => ({
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
    })),
  );
  const chartData = holes.map((h) => ({ hole: String(h.hole), expected: num(h.exp_vs_par) ?? 0 }));
  const first = perRound[0];
  const override = obj(obj(doc.engine).owner_overrides_on_engine);
  // Course spread factor: the week runs publish venue_fit.course_sd_mult; live runs do not, but every player row carries the same course feature.
  const fromFit = obj(fit.course_sd_mult);
  const sdMult = num(fromFit.median) !== null ? { min: num(fromFit.min) ?? NaN, median: num(fromFit.median) as number, max: num(fromFit.max) ?? NaN } : spread(players.map((p) => num(obj(p.course_fit).course_sd_mult_feature) ?? NaN));
  const rawLabel = String(course.table_label ?? "");
  const courseName = String(event.course_name ?? "");
  const layout = obj(obj(course.course_provenance).layout);
  const sentence = describeHoleTable(rawLabel, layout);
  const estimated = Number(/n_med=(\d+)/.exec(rawLabel)?.[1] ?? NaN) === 0;
  const sourceName = estimated ? "Estimated from par and length" : /prior edition/i.test(String(course.hole_table_source ?? "")) ? "Last year's edition" : titleCase(course.hole_table_source) || "Unknown";
  const layoutYear = num(layout.year) ?? Number(/(\d{4})-\d{2}-\d{2}\]/.exec(rawLabel)?.[1] ?? NaN);
  const showUntouchedAvg = perRound.some((r) => differs(r.scoring_average_vs_par, r.scoring_average_vs_par_untouched));
  const perRoundRows = dropEmptyColumns(
    perRound.map((r) => ({
      round: num(r.round),
      par: num(r.par),
      yardage: num(r.yardage),
      scoring_avg: num(r.scoring_average_vs_par),
      ...(showUntouchedAvg ? { scoring_avg_untouched: num(r.scoring_average_vs_par_untouched) } : {}),
      birdies_per_round: num(r.birdie_or_better_per_round),
      bogeys_per_round: num(r.bogey_or_worse_per_round),
    })),
  );
  const fitStats = ["fit_rs_ddacc_lam1000", "ch_resid_k80_hl1461"].map((key) => ({ key, stat: obj(fit[key]) })).filter(({ stat }) => num(stat.median) !== null);
  const courseOverrides = [
    num(course.course_scoring_avg_delta_override) !== null ? `scoring average ${signed(course.course_scoring_avg_delta_override, 2)} strokes` : "",
    num(override.course_sd_mult) !== null && num(override.course_sd_mult) !== 1 ? `spread factor ${fx(override.course_sd_mult, 2)}` : "",
  ].filter(Boolean);
  const showFit = fitStats.length > 0 || courseOverrides.length > 0;
  const holeKeys = ["hole", "par", "yardage", "expected_vs_par", "birdie_pct", "par_pct", "bogey_pct", "sensitivity"];
  const holeCols = ["hole", "par", "yardage", "expected_vs_par", "birdie_pct", "bogey_pct", "eagle_pct", "par_pct", "double_pct", "triple_pct", "sensitivity"];
  const roundCols = ["round", "par", "yardage", "scoring_avg", "scoring_avg_untouched", "birdies_per_round", "bogeys_per_round"];
  return (
    <div className="stack-lg">
      <div className="kpi-grid">
        <Kpi label="Course" value={courseName || "—"} detail={`par ${String((arr(event.par_per_round) as unknown as number[])[0] ?? "—")}`} tone="accent" />
        <Tip text="Where the hole-by-hole scoring numbers come from. Courses with no current data borrow last year's edition, or are estimated from par and length."><Kpi label="Hole difficulty from" value={sourceName} detail={!estimated && Number.isFinite(layoutYear) ? `${layoutYear} layout` : undefined} /></Tip>
        {first && num(first.scoring_average_vs_par) !== null && <Tip text={GLOSSARY.scoring_avg}><Kpi label="Scoring average vs par" value={signed(first.scoring_average_vs_par, 2)} detail="round 1, field-average player (other rounds in the table below)" /></Tip>}
        {sdMult && (
          <Tip text={GLOSSARY.course_sd_mult}>
            <Kpi label="Course spread factor" value={fx(sdMult.median, 3)} detail={Math.abs(sdMult.max - sdMult.min) < 0.0005 ? "1.00 is a typical course; the same for every player" : `1.00 is a typical course; field range ${fx(sdMult.min, 3)} to ${fx(sdMult.max, 3)}`} />
          </Tip>
        )}
      </div>
      <Panel eyebrow="What the simulation played" title="Hole by hole">
        <p className="inputs-note">Each hole as the simulation plays it, for a reference player of zero skill. A player&apos;s skill then moves his chances of birdies and bogeys up or down.{sentence ? ` ${sentence}` : ""}</p>
        {tables.length > 1 && (
          <SegmentedControl label="Round group" value={String(group)} onChange={(v) => setGroup(Number(v))} options={tables.map((t, i) => ({ value: String(i), label: `Round${(t.rounds as unknown[]).length === 1 ? "" : "s"} ${(t.rounds as unknown[]).join(", ")}` }))} />
        )}
        {!estimated && (
          <div className="chart-medium">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 16, bottom: 4, left: 0 }}>
                <CartesianGrid stroke="var(--line)" vertical={false} />
                <XAxis dataKey="hole" tick={{ fill: "var(--muted)", fontSize: 10 }} />
                <YAxis tick={{ fill: "var(--muted)", fontSize: 10 }} label={{ value: "Expected score vs par (red = hard, green = easy)", angle: -90, fill: "var(--muted)", fontSize: 10, position: "insideLeft" }} />
                <Tooltip content={<ChartTip />} />
                <ReferenceLine y={0} stroke="var(--line-strong)" />
                <Bar dataKey="expected" name="Expected score vs par" radius={3} isAnimationActive={false}>
                  {chartData.map((point) => <Cell key={point.hole} fill={point.expected > 0 ? "var(--negative)" : "var(--positive)"} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        <DataTable
          rows={holeRows}
          label="Hole table"
          pageSize={18}
          preferredColumns={holeCols}
          defaultColumns={holeKeys}
          headerLabels={HEADERS}
          headerTitles={titlesFor(holeCols)}
          renderCell={(column, value) => (column === "expected_vs_par" && num(value) !== null ? signed(value) : column === "sensitivity" && num(value) !== null ? fx(value) : undefined)}
        />
      </Panel>
      <div className={showFit ? "two-column" : undefined}>
        <Panel eyebrow="Per round" title="Course summary">
          <DataTable
            rows={perRoundRows}
            label="Course per round"
            pageSize={8}
            preferredColumns={roundCols}
            headerLabels={HEADERS}
            headerTitles={titlesFor(roundCols)}
            renderCell={(column, value) => ((column === "scoring_avg" || column === "scoring_avg_untouched") && num(value) !== null ? signed(value) : column === "birdies_per_round" || column === "bogeys_per_round" ? (num(value) === null ? undefined : fx(value, 1)) : undefined)}
          />
          {courseOverrides.length > 0 && <p className="inputs-note">A scoring-average override changes the displayed scores but not win odds.</p>}
        </Panel>
        {showFit && (
          <Panel eyebrow="Course fit" title="What this course changes">
            <div className="kv-table">
              {fitStats.map(({ key, stat }) => (
                <div key={key}>
                  <span><Term k={key.startsWith("fit") ? "fit_rs_ddacc" : "course_history"}>{key.startsWith("fit") ? "Venue fit (distance and accuracy)" : "Course history"}</Term></span>
                  <b>low {fx(stat.min, 3)} · typical {fx(stat.median, 3)} · high {fx(stat.max, 3)}</b>
                </div>
              ))}
              {courseOverrides.length > 0 && <div><span>Your course overrides</span><b>{courseOverrides.join("; ")}</b></div>}
            </div>
          </Panel>
        )}
      </div>
      <Technical>
        <KeyValue
          data={{
            hole_table_label: rawLabel,
            hole_table_source: course.hole_table_source,
            hole_table_prior_strength_k: course.prior_strength_k,
            hole_table_snapshot_status: course.snapshot_status,
            layout_status: layout.status,
            layout_confirmed_for_this_year: layout.current_event_confirmed,
            venue_id: event.venue_uid,
            model_version: obj(course.course_provenance).model_version,
            data_coverage_note: obj(course.course_provenance).coverage_limitations,
            slopes_status: slopes.status,
            slopes_reason: slopes.reason,
            reference_player_round_sd: course.hole_model_round_sd,
          }}
          empty=""
        />
        {extra}
      </Technical>
    </div>
  );
}

/* ------------------------------------------------------------------ variance and engine */
function VarianceTab({ doc, extra }: { doc: Obj; extra?: ReactNode }) {
  const eng = obj(doc.engine);
  const players = arr(doc.players);
  const points = players.map((p) => ({ name: String(p.name ?? ""), mu: num(obj(p.challenger).mu) ?? 0, sd: num(obj(p.challenger).sd) ?? 0 }));
  const cut = obj(obj(doc.event).cut_rule);
  const wl = obj(eng.week_latent);
  const dl = obj(eng.day_latent);
  const pin = obj(eng.sd_pin_check);
  const oo = obj(eng.owner_overrides_on_engine);
  const engineOverrides = [
    num(oo.course_scoring_avg_delta) !== null ? `course scoring average ${signed(oo.course_scoring_avg_delta, 2)} strokes` : "",
    num(oo.course_sd_mult) !== null && num(oo.course_sd_mult) !== 1 ? `course spread factor ${fx(oo.course_sd_mult, 2)}` : "",
    oo.rule_override ? `cut rule changed (${cutRuleSentence(obj(oo.rule_override))})` : "",
  ].filter(Boolean);
  const registry = obj(cut.registry_rule);
  const registryText = cutRuleSentence(registry);
  const cutText = Object.keys(cut).length ? cutRuleSentence(cut) : "";
  const wkRho = num(wl.wk_rho);
  const noise = num(eng.seed_spread_max_abs_p_win);
  const blowUp = num(dl.day_pb);
  const career = num(dl.day_ph);
  return (
    <div className="stack-lg">
      <div className="kpi-grid">
        <Tip text={GLOSSARY.n_sims}><Kpi label="Simulated tournaments" value={Number(eng.n_sims ?? 0).toLocaleString()} tone="accent" /></Tip>
        {num(wl.tau) !== null && <Tip text={GLOSSARY.tau}><Kpi label="Form drift over the week" value={`${fx(wl.tau, 1)} strokes`} detail="per round, up or down: how far a player's form can wander during a tournament" /></Tip>}
        {noise !== null && <Tip text={GLOSSARY.seed_spread}><Kpi label="Win-odds precision" value={`±${(noise * 100).toFixed(1)} pts`} detail="win odds are accurate to about this many percentage points" /></Tip>}
      </div>
      <div className="two-column">
        <Panel eyebrow="Spread" title="Rating against round swing">
          <p className="inputs-muted">Each dot is a player: how good he is (rating) against how much his rounds swing. Hover a dot for the name.</p>
          <div className="chart-medium">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 16, bottom: 18, left: 0 }}>
                <CartesianGrid stroke="var(--line)" />
                <XAxis type="number" dataKey="mu" name="Rating" tick={{ fill: "var(--muted)", fontSize: 10 }} label={{ value: "Rating (strokes per round vs field)", fill: "var(--muted)", fontSize: 10, position: "insideBottom", offset: -8 }} />
                <YAxis type="number" dataKey="sd" name="Round swing" domain={["auto", "auto"]} tick={{ fill: "var(--muted)", fontSize: 10 }} label={{ value: "Round swing", angle: -90, fill: "var(--muted)", fontSize: 10, position: "insideLeft" }} />
                <ZAxis range={[26, 26]} />
                <Tooltip content={<ChartTip />} />
                <Scatter data={points} fill={palette[0]} name="Player" isAnimationActive={false} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </Panel>
        <div className="stack-lg">
          {cutText && (
            <Panel eyebrow="Cut" title="Cut rule">
              <div className="kv-table">
                <div><span>Cut rule</span><b>{cutText}</b></div>
                {Boolean(cut.override) && <div><span>Before your override</span><b>{registryText}</b></div>}
              </div>
            </Panel>
          )}
          {(blowUp !== null || career !== null || num(wl.tau) !== null) && (
            <Panel eyebrow="Luck built into every tournament" title="Week and day swings">
              <div className="kv-table">
                {num(wl.tau) !== null && (
                  <div>
                    <span><Term k="tau">Form drift</Term></span>
                    <b>about {fx(wl.tau, 1)} strokes per round either way{wkRho !== null && wkRho >= 0.95 ? "; one shared hot or cold streak carries through the whole week" : ""}</b>
                  </div>
                )}
                {blowUp !== null && <div><span>Blow-up round</span><b>{pct(blowUp, 1)} of rounds (about 1 in {Math.round(1 / Math.max(blowUp, 0.001))}): roughly {fx(dl.day_db, 1)} strokes worse than the player&apos;s own average</b></div>}
                {career !== null && <div><span>Career round</span><b>{pct(career, 1)} of rounds: roughly {fx(dl.day_dh, 1)} strokes better than the player&apos;s own average</b></div>}
              </div>
              {(num(dl.day_db) ?? 0) > (num(dl.day_dh) ?? 0) && <p className="inputs-muted">Blow-ups are far bigger than career rounds, as in real golf.</p>}
            </Panel>
          )}
          {engineOverrides.length > 0 && (
            <Panel eyebrow="Your overrides" title="Overrides on the engine">
              <p className="inputs-note">{`Active: ${engineOverrides.join("; ")}.`}</p>
            </Panel>
          )}
        </div>
      </div>
      <Technical>
        <KeyValue
          data={{
            overall_spread_scale: eng.sd_scale,
            hole_to_hole_luck_variance: eng.V_hole,
            hole_carry_over: eng.hole_a,
            batches: eng.n_seeds,
            spread_match_average: pin.ratio_mean,
            spread_match_min: pin.ratio_min,
            spread_match_max: pin.ratio_max,
            variance_rule: eng.variance_rule,
            variance_params_version: eng.variance_params_version,
            simulator: eng.simulator,
            seed0: eng.seed0,
            field_sd_common_round_conditions: eng.field_sd_common_round_conditions,
            rotation_course_sd: eng.rotation_course_sd,
            rotation_layer_active: eng.rotation_layer_active,
            week_latent: eng.week_latent,
            day_latent: eng.day_latent,
            shape_mixture: eng.shape_mixture,
            kernel_params: eng.kernel_params,
            cut_rule_basis: cut.rule_basis,
            registry_rule: registry,
          }}
        />
        {extra}
      </Technical>
    </div>
  );
}

/* ------------------------------------------------------------------ weather */
function plainWeatherReason(text: string): string {
  const parts = text.split(";").map((part) => part.trim()).filter(Boolean);
  const noSheet: string[] = [];
  const rest: string[] = [];
  for (const part of parts) {
    const m = /^R(\d): no tee sheet: wave-free common component only$/.exec(part);
    if (m) noSheet.push(m[1]);
    else rest.push(part.replace(/^R(\d):/, "Round $1:"));
  }
  if (noSheet.length) rest.unshift(`${noSheet.length === 1 ? "Round" : "Rounds"} ${joinAnd(noSheet)}: tee times are not out yet, so only the general forecast is used`);
  return rest.join("; ");
}

function WeatherTab({ doc, runAsOf, extra }: { doc: Obj; runAsOf: string; extra?: ReactNode }) {
  const w = obj(doc.weather);
  const waves = arr(w.waves);
  const snapshot = obj(w.forecast_snapshot);
  const display = obj(w.forecast_snapshot_display);
  const cover = obj(w.tee_sheet_coverage);
  const roundStatus = obj(w.round_status);
  const fmtTee = teeFormatter(doc);
  const data = waves.map((x) => ({ label: `Round ${String(x.round)}, ${String(x.wave)}`, wind: Math.round(num(x.mean_forecast_wind_mph) ?? 0), players: num(x.n_players) ?? 0 }));
  const coverage: Array<{ round: string; value: number }> = Object.keys(roundStatus).length
    ? Object.entries(roundStatus).map(([round, info]) => ({ round, value: num(obj(info).tee_coverage) ?? 0 }))
    : Object.entries(cover).map(([round, value]) => ({ round: round.replace(/\D/g, ""), value: num(value) ?? 0 }));
  const issued = snapshot.issued_at ?? display.fetched_at;
  const hoursBefore = issued && runAsOf ? (Date.parse(runAsOf) - Date.parse(String(issued))) / 3_600_000 : Number.NaN;
  const reason = plainWeatherReason(String(w.reason ?? ""));
  const waveRows = waves.map((x) => ({
    round: num(x.round),
    wave: titleCase(x.wave),
    players: num(x.n_players),
    first_tee: fmtTee(x.first_tee_local).both,
    last_tee: fmtTee(x.last_tee_local).both,
    mean_wind_mph: num(x.mean_forecast_wind_mph) === null ? null : Math.round(num(x.mean_forecast_wind_mph) as number),
  }));
  const published = coverage.filter((item) => item.value > 0);
  return (
    <div className="stack-lg">
      <div className="inputs-banner warn">
        <strong>{w.applied_in_prices ? "Weather is in the prices" : "Weather is NOT in the prices"}</strong>
        <span>
          {w.applied_in_prices
            ? "Each player gets a small adjustment per round from his wave (early or late tee time) and the forecast wind. Realised weather is never used."
            : "Tee times, waves and the forecast are shown for review only."}
          {reason ? ` ${reason}.` : ""}
        </span>
      </div>
      <div className="kpi-grid">
        {issued ? <Kpi label="Forecast issued" value={etTime(issued)} detail={Number.isFinite(hoursBefore) && hoursBefore >= 0 ? `${ageText(hoursBefore)} before this model run` : undefined} tone="accent" /> : null}
        {published.map((item) => (
          <Tip key={item.round} text="Share of the field with a published tee time for this round. Without tee times only the shared (not wave-specific) weather effect can be applied.">
            <Kpi label={`Tee times, round ${item.round}`} value={pct(item.value, 0)} detail="of players have a published time" />
          </Tip>
        ))}
      </div>
      {data.length > 0 && (
        <Panel eyebrow="Forecast" title="Average forecast wind by wave (mph)">
          <div className="chart-medium">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data} margin={{ top: 10, right: 16, bottom: 4, left: 0 }}>
                <CartesianGrid stroke="var(--line)" vertical={false} />
                <XAxis dataKey="label" tick={{ fill: "var(--muted)", fontSize: 10 }} />
                <YAxis tick={{ fill: "var(--muted)", fontSize: 10 }} allowDecimals={false} />
                <Tooltip content={<ChartTip />} />
                <Bar dataKey="wind" name="Average wind (mph)" fill={palette[2]} radius={3} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      )}
      {waveRows.length > 0 && (
        <Panel eyebrow="Tee sheet" title="Waves (course time, with Eastern in brackets)">
          <DataTable
            rows={waveRows}
            label="Waves"
            pageSize={12}
            verbatim
            preferredColumns={["round", "wave", "players", "first_tee", "last_tee", "mean_wind_mph"]}
            headerLabels={{ round: "Round", wave: "Wave", players: "Players", first_tee: "First tee", last_tee: "Last tee", mean_wind_mph: "Average forecast wind (mph)" }}
          />
        </Panel>
      )}
      <Technical>
        <KeyValue data={{ forecast_snapshot: snapshot, forecast_file: display.file, wind_converted_from_kmh: display.wind_mph_converted_from_kmh, recipe: w.recipe }} />
        {extra}
      </Technical>
    </div>
  );
}

/* ------------------------------------------------------------------ odds */
const FEED_NAMES: Record<string, string> = {
  field_updates: "Field list",
  matchups_tournament_matchups: "Tournament matchups",
  outrights_make_cut: "Make-cut prices",
  outrights_mc: "Make-cut prices (backup source)",
  outrights_win: "Win prices",
  outrights_top_5: "Top 5 prices",
  outrights_top_10: "Top 10 prices",
  outrights_top_20: "Top 20 prices",
  pre_tournament: "Data Golf pre-tournament model",
};

/** The latest time any book's prices were pulled (ISO), falling back to the data feeds, then to the run's own odds timestamp. */
function oddsPulledAt(doc: Obj): string {
  const odds = obj(doc.odds);
  const times = [...arr(odds.books).map((b) => Date.parse(String(b.latest_fetch ?? ""))), ...Object.values(obj(odds.snapshot_inputs)).map((f) => Date.parse(String(obj(f).fetched_at ?? "")))].filter(Number.isFinite);
  return times.length ? new Date(Math.max(...times)).toISOString() : String(odds.as_of ?? "");
}

function OddsTab({ doc, runAsOf }: { doc: Obj; runAsOf: string }) {
  const odds = obj(doc.odds);
  const books = arr(odds.books);
  const stale = books.filter((b) => (num(b.age_hours_at_asof) ?? 0) > 6).length;
  const excluded = Array.isArray(odds.excluded_books_config) ? (odds.excluded_books_config as unknown as string[]) : [];
  const feedRows = Object.entries(obj(odds.snapshot_inputs)).map(([key, value]) => {
    const f = obj(value);
    return { feed: FEED_NAMES[key] ?? titleCase(key), pulled: etTime(f.fetched_at, ""), source_updated: utcText(f.payload_last_updated) };
  });
  const marketRows = Object.entries(obj(odds.books_by_market)).map(([key, value]) => ({ market: marketName(key), books: num(value) }));
  const bookRows = books.map((b) => ({
    book: bookName(b.book),
    age: ageText(num(b.age_hours_at_asof)),
    quotes: num(b.n_quotes),
    markets: Array.isArray(b.markets) ? [...new Set((b.markets as unknown[]).map((m) => marketName(String(m))))].join(", ") : String(b.markets ?? ""),
    latest_fetch: etTime(b.latest_fetch, ""),
    latest_book_update: etTime(b.latest_book_update, ""),
  }));
  const pulled = oddsPulledAt(doc);
  return (
    <div className="stack-lg">
      <div className="kpi-grid">
        <Tip text="The most recent time any book's prices were pulled. A live run can reuse earlier odds."><Kpi label="Odds last pulled" value={etTime(pulled)} detail={runAsOf ? `model run built ${etTime(runAsOf)}` : undefined} tone="accent" /></Tip>
        <Tip text={GLOSSARY.odds_age}><Kpi label="Books" value={String(books.length)} detail={stale ? `${stale} with prices older than 6 hours` : "all prices under 6 hours old when the run was built"} tone={stale ? "negative" : "positive"} /></Tip>
        {excluded.length > 0 && (
          <Tip text="Prices from these sources are compared against, never treated as a sportsbook price.">
            <Kpi label="Comparison only" value={excluded.map(bookName).join(", ")} detail="used as a model comparison, not as a sportsbook price" />
          </Tip>
        )}
      </div>
      <p className="inputs-note">Odds do not change the simulation. They only come in at the end, when the model&apos;s prices are blended with the market.</p>
      <Panel eyebrow="Freshness" title="Each book's prices">
        <DataTable
          rows={dropEmptyColumns(bookRows)}
          label="Odds freshness"
          pageSize={20}
          verbatim
          preferredColumns={["book", "age", "quotes", "markets", "latest_fetch", "latest_book_update"]}
          headerLabels={{ book: "Book", age: "Age when run was built", quotes: "Prices", markets: "Markets", latest_fetch: "Pulled (ET)", latest_book_update: "Book last changed (ET)" }}
          headerTitles={{ age: GLOSSARY.odds_age, latest_fetch: GLOSSARY.fetched, latest_book_update: GLOSSARY.book_update }}
        />
      </Panel>
      <div className="two-column">
        {feedRows.length > 0 && (
          <Panel eyebrow="Data pulls" title="Where each input came from">
            <DataTable rows={dropEmptyColumns(feedRows)} label="Odds feeds" pageSize={12} verbatim preferredColumns={["feed", "pulled", "source_updated"]} headerLabels={{ feed: "Input", pulled: "Pulled (ET)", source_updated: "Source last updated (ET)" }} />
          </Panel>
        )}
        {marketRows.length > 0 && (
          <Panel eyebrow="Consensus" title="Books quoting each market">
            <DataTable rows={marketRows} label="Books per market" pageSize={12} verbatim preferredColumns={["market", "books"]} headerLabels={{ market: "Market", books: "Books quoting it" }} />
          </Panel>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ run info */
const TOML_NAMES: Record<string, string> = {
  "books.toml": "Which books count as soft, sharp or excluded",
  "checks.toml": "Safety-check thresholds",
  "markets.toml": "Which bet types are live, and their rules",
  "run.toml": "Simulation run settings",
  "staking.toml": "Staking rules",
};

function ConfigTab({ doc, extra }: { doc: Obj; extra?: ReactNode }) {
  const config = obj(doc.config);
  const files = obj(config.files);
  const prov = obj(doc.provenance);
  const run = obj(config.run);
  const chl = obj(prov.chl);
  const guard = obj(prov.v2_guard);
  const clock = obj(prov.forward_clock);
  const snap = snapshotTime(prov.foundation_id);
  const history = ymd(chl.history_last_round_date);
  const places = Array.isArray(run.places) ? (run.places as unknown[]).join(", ") : "";
  return (
    <div className="stack-lg">
      <div className="kpi-grid">
        {(history || snap) && (
          <Tip text="The most recent round included in the players' history, and when that data was last refreshed.">
            <Kpi label="Player history" value={history ? `Rounds through ${history}` : `Refreshed ${snap}`} detail={history && snap ? `data refreshed ${snap}` : undefined} tone="accent" />
          </Tip>
        )}
        {num(run.n_sims) !== null && <Tip text={GLOSSARY.n_sims}><Kpi label="Simulated tournaments" value={(num(run.n_sims) as number).toLocaleString()} /></Tip>}
        {clock.counts_for_forward_clock !== undefined && (
          <Tip text="Only live PGA Tour events that have odds go into the live track record; other runs are for information.">
            <Kpi label="In the live track record" value={clock.counts_for_forward_clock ? "Yes" : "No"} detail={clock.counts_for_forward_clock ? "this run is scored against results" : "shown for information only"} />
          </Tip>
        )}
      </div>
      {(places || prov.location_status !== undefined) && (
        <Panel eyebrow="This run" title="How it was set up">
          <div className="kv-table">
            {places && <div><span>Finishing positions priced</span><b>Win and top {places}</b></div>}
            {prov.location_status !== undefined && <div><span>Location adjustment</span><b>{prov.location_status === "ok" ? "Applied" : titleCase(prov.location_status)}</b></div>}
          </div>
        </Panel>
      )}
      <Technical>
        <KeyValue data={{ model: obj(obj(prov.arms_status).challenger).what, spec_check: Object.keys(guard).length ? { passed: guard.ok, checks: guard.n_checked } : undefined }} empty="" />
        <KeyValue data={{ params: config.params, run: config.run }} />
        <KeyValue data={{ ...prov, code_hashes: undefined }} />
        {Object.entries(files).map(([name, value]) => {
          const file = obj(value);
          return (
            <details className="config-file" key={name}>
              <summary><b>{TOML_NAMES[name] ?? name}</b><span>{name}</span></summary>
              <pre>{String(file.text ?? JSON.stringify(file, null, 1))}</pre>
            </details>
          );
        })}
        {extra}
      </Technical>
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

const VARIANT_SHORT: Record<string, string> = { base: "All players, all events", xeuro: "+ at DP World Tour events", xband0: "+ player <30 rounds", xband1: "+ player 30-100 rounds", miss: "+ if missing", miss_xeuro: "+ if missing, DP World Tour event" };
type WeightModel = "chl" | "v21";
type SortMode = "model" | "weight";

function WeightStrip({ weights, keys = Object.keys(VARIANT_SHORT) }: { weights: WeightSet; keys?: string[] }) {
  // Variants that do not exist for this feature are left out rather than shown as blanks.
  const present = keys.filter((key) => num(weights[key]) !== null);
  if (!present.length) return <p className="inputs-muted">No weight in this model.</p>;
  return (
    <div className="feature-weights">
      {present.map((key) => {
        const value = num(weights[key]) as number;
        return (
          <div key={key} className={value > 0 ? "positive" : value < 0 ? "negative" : ""} title={`${VARIANT_SHORT[key]}: ${value}. A bigger absolute weight means the model leans on it more.`}>
            <span>{VARIANT_SHORT[key]}</span>
            <b>{signed(value, 3)}</b>
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
  if (error || !glossary) return <EmptyState title="Skill measurements are not available yet" detail="They were not saved for this run. They will appear here after the next publish." />;
  const m = glossary.model;
  const share = (f: GlossaryFamily) => (model === "chl" ? f.share_abs_weight_chl : f.share_abs_weight_v21);
  const visibleFamilies = glossary.families.filter((f) => shown.some((x) => x.family === f.key));
  return (
    <div className="stack-lg">
      <div className="kpi-grid">
        <Tip text="Distinct skill measurements the model can use to rate a player."><Kpi label="Features" value={String(glossary.features.length)} detail={`${num(m.n_design_columns_2026) ?? "—"} model inputs in ${glossary.season} once variants are counted`} tone="accent" /></Tip>
        <Tip text="How strongly the fitting shrinks every weight toward zero so no single noisy feature dominates. Bigger means more shrinkage."><Kpi label="Shrinkage strength" value={fx(m.ridge_lambda_2026, 0)} detail={`chosen by testing on later seasons, ${glossary.season} season`} /></Tip>
        <Tip text="The seasons of player results the weights were learned from. The weights are frozen for the whole season."><Kpi label="Learned from" value={String(m.train_seasons_2026 ?? "—").replace("-", " to ")} detail={`${(num(m.n_train_2026) ?? 0).toLocaleString()} player-events`} /></Tip>
        <Tip text="Inputs describing home base, travel and nationality."><Kpi label="Location inputs" value={String(glossary.location.n_columns_2026)} detail="home base, travel and nationality" /></Tip>
      </div>
      <Panel eyebrow="How to read this" title="What the weights mean">
        <p className="inputs-note">{plain(m.overview)}</p>
        <p className="inputs-note">{plain(glossary.reading_weights)}</p>
        <h3 className="inputs-h3">Variants of each feature</h3>
        <div className="kv-table variant-help">
          {glossary.variants.map((v) => (
            <div key={v.key}><span>{plain(v.label)}</span><b>{plain(v.text)}</b></div>
          ))}
        </div>
        <details className="config-file">
          <summary><b>Standardisation and how the weights are fitted</b><span>walk-forward ridge, per season</span></summary>
          <div className="glossary-prose">
            <p className="inputs-note">{glossary.standardisation}</p>
            <ul>{glossary.fit.map((line) => <li key={line}>{plain(line)}</li>)}</ul>
          </div>
        </details>
      </Panel>
      <Panel eyebrow="Find a feature" title="Filter">
        <div className="glossary-controls">
          <label className="search-box">
            <Search size={15} />
            <input type="search" aria-label="Search features" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search by name or explanation…" />
          </label>
          <Select label="Group" value={family} onChange={setFamily} options={[{ value: "all", label: "All groups" }, ...glossary.families.map((f) => ({ value: f.key, label: `${plain(f.label)} (${f.n_features})` })), { value: "location", label: `Location (${glossary.location.columns.length})` }]} />
          <Select label="Order" value={sort} onChange={(v) => setSort(v as SortMode)} options={[{ value: "model", label: "Model order" }, { value: "weight", label: "Largest weight first" }]} />
          <SegmentedControl label="Weights from" value={model} onChange={setModel} options={[{ value: "chl", label: "Skill model alone" }, { value: "v21", label: "Refit with location" }]} />
        </div>
        <p className="inputs-muted">{shown.length + locShown.length} of {glossary.features.length + glossary.location.columns.length} shown · weights are {glossary.season}-season standardised coefficients ({model === "chl" ? "the skill model on its own" : "the skill model refit together with the location inputs"}).</p>
      </Panel>
      {visibleFamilies.map((fam) => (
        <Panel key={fam.key} eyebrow={`${fam.n_features} features · ${(share(fam) * 100).toFixed(1)}% of total absolute weight`} title={plain(fam.label)}>
          <p className="inputs-note">{plain(fam.summary)}</p>
          <div className="feature-grid">
            {shown.filter((f) => f.family === fam.key).map((f) => (
              <article className="feature-card" key={f.base} id={`feature-${f.base}`}>
                <header>
                  <h3>{plain(f.name)}</h3>
                </header>
                <p>{plain(f.measures)}</p>
                <WeightStrip weights={model === "chl" ? f.weights.chl : f.weights.v21} />
                <details>
                  <summary>How it is computed and what sign to expect</summary>
                  <h4>How it is computed</h4>
                  <p>{plain(f.computed)}</p>
                  <h4>Expected sign</h4>
                  <p>{plain(f.sign_intuition)}</p>
                  <h4>Technical: model column name</h4>
                  <p><code className="feature-code">{f.base}</code></p>
                </details>
              </article>
            ))}
          </div>
        </Panel>
      ))}
      {locShown.length > 0 && (
        <Panel eyebrow={`${glossary.location.columns.length} inputs · only in the location refit`} title="Location inputs">
          <p className="inputs-note">{glossary.location.note}</p>
          <div className="feature-grid">
            {locShown.map((c) => (
              <article className="feature-card" key={c.base}>
                <header>
                  <h3>{c.name}</h3>
                </header>
                <p>{c.measures}</p>
                <WeightStrip weights={c.weights.v21} keys={["base", "xeuro"]} />
                <span className="inputs-muted">{c.part}</span>
                <details>
                  <summary>Technical: model column name</summary>
                  <p><code className="feature-code">{c.base}</code></p>
                </details>
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

const FIELD_LABELS: Record<string, string> = {
  skill_delta: "Skill change (strokes per round)",
  sd_mult: "Round swing multiplier (1.00 = no change)",
  withdraw: "Withdraw",
  scoring_avg_delta: "Course scoring average change",
  course_sd_mult: "Course spread factor (1.00 = no change)",
  cut_rule: "Cut rule",
};
const SCOPE_LABELS: Record<string, string> = { player: "Player", course: "Course", engine: "Cut rule" };

/** One plain sentence about what an override field does, with its allowed range. */
function specNote(spec: FieldSpec): string {
  const range = spec.kind === "number" ? ` (between ${spec.lo} and ${spec.hi})` : "";
  const known: Record<string, string> = {
    skill_delta: "Added to this player's rating, in strokes per round",
    sd_mult: "Multiplies this player's round swing",
    scoring_avg_delta: "Shifts the course's scoring average, in strokes per round; this changes displayed scores, not win odds",
    course_sd_mult: "Widens or narrows scores at this course",
  };
  const base = known[spec.field] ?? spec.what.replace(/\bchallenger\s+/gi, "").replace(/\.$/, "");
  return `${base}${range}.`;
}
const fieldLabel = (field: string) => FIELD_LABELS[field] ?? titleCase(field);

function describeValue(record: OverrideRecord): string {
  if (typeof record.value === "object") return Object.entries(record.value).map(([k, v]) => `${k}=${v}`).join(", ");
  return String(record.value);
}

function AdjustTab({ doc, eventUid, presetPlayer, onPresetUsed }: { doc: Obj; eventUid: string; presetPlayer: number | null; onPresetUsed: () => void }) {
  const players = arr(doc.players);
  const eventObj = obj(doc.event);
  const specs: FieldSpec[] = ALLOWED; // compiled-in bounds are the enforced ones (worker)
  const [records, setRecords] = useState<OverrideRecord[] | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [scope, setScope] = useState<Scope>("player");
  const [field, setField] = useState("skill_delta");
  const [eventKey, setEventKey] = useState(eventUid);
  const [target, setTarget] = useState<string>(presetPlayer ? String(presetPlayer) : "");
  const [playerFilter, setPlayerFilter] = useState("");
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
  const playerOptions = useMemo(() => {
    const all = players.flatMap((p) => (num(p.dg_id) === null ? [] : [{ name: String(p.name ?? ""), dg_id: num(p.dg_id) as number }]));
    const shown = playerFilter.trim() ? matchPlayers(all, playerFilter, 30) : all;
    const picked = all.find((c) => String(c.dg_id) === target);
    return picked && !shown.includes(picked) ? [picked, ...shown] : shown;
  }, [players, playerFilter, target]);

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
    if (field === "skill_delta") return `Rating ${signed(ch.mu)} becomes about ${signed((num(ch.mu) ?? 0) + v)} (before the field is re-centred).`;
    if (field === "sd_mult") return `Round swing ${fx(ch.sd, 2)} becomes about ${fx((num(ch.sd) ?? 0) * v, 2)}.`;
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
      setMessage({ tone: "ok", text: `Saved. The model applies it on its next run (prices built after ${etTime(result.body.record.created_at)}); the original model number is still stored.` });
      setValue("");
      setCut({});
      setReason("");
      await reload();
    } else {
      setMessage({ tone: "error", text: result.status === 401 ? "Not signed in through Cloudflare Access; the change was refused." : result.body.error ?? `HTTP ${result.status}`, problems: result.body.problems });
    }
  }

  async function change(record: OverrideRecord, mode: "expire" | "remove") {
    if (!window.confirm(mode === "remove" ? `Remove this override (${fieldLabel(record.field)}: ${describeValue(record)})? It stops applying and is deleted from the active list (the history log keeps it).` : `Expire this override (${fieldLabel(record.field)}: ${describeValue(record)}) now?`)) return;
    setBusy(true);
    const result = await apiJson(`/api/overrides/${encodeURIComponent(record.id)}?mode=${mode}`, { method: "DELETE" });
    setBusy(false);
    setMessage(result.ok ? { tone: "ok", text: `${mode === "remove" ? "Removed" : "Expired"} the override.` } : { tone: "error", text: result.body.error ?? `HTTP ${result.status}`, problems: result.body.problems });
    await reload();
  }

  return (
    <div className="stack-lg">
      <div className="inputs-banner">
        <strong>One-off fixes to the model price</strong>
        <span>An override changes the model&apos;s price for this event only and expires automatically. The original model number is kept. It is applied on the next run and shows as its own line in the player breakdown.</span>
      </div>
      <Panel eyebrow="New override" title="Adjust the model">
        <form className="override-form" onSubmit={submit}>
          <div className="form-grid">
            <Select label="Event" value={eventKey} onChange={setEventKey} options={[{ value: eventUid, label: String(eventObj.name ?? "This event") }, { value: "all", label: "All events" }]} />
            <Select label="Scope" value={scope} onChange={(v) => { setScope(v as Scope); const first = specs.find((s) => s.scope === v); setField(first?.field ?? ""); setValue(""); }} options={[{ value: "player", label: "Player" }, { value: "course", label: "Course" }, { value: "engine", label: "Cut rule" }]} />
            <Select label="What to change" value={field} onChange={(v) => { setField(v); setValue(""); }} options={fieldsForScope.map((s) => ({ value: s.field, label: fieldLabel(s.field) }))} />
            {scope === "player" && (
              <>
                <label className="field-label">
                  <span>Find a player</span>
                  <input type="search" autoComplete="off" value={playerFilter} onChange={(e) => setPlayerFilter(e.target.value)} placeholder="Type part of a name…" />
                </label>
                <Select label="Player" value={target} onChange={setTarget} options={[{ value: "", label: "Choose a player…" }, ...playerOptions.map((c) => ({ value: String(c.dg_id), label: c.name }))]} />
              </>
            )}
          </div>
          {spec && <p className="inputs-note">{specNote(spec)}</p>}
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
                  <span>{({ cut_round: "Cut after round", top_n: "Players who make it", within: "Plus anyone within (shots)", mdf_trigger: "Second cut applies above (players)", mdf_top_n: "Second cut keeps top", mdf_round: "Second cut after round" } as Record<string, string>)[key] ?? titleCase(key)} ({lo} to {hi})</span>
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
            <span className="inputs-muted">Times are shown in Eastern.</span>
          </div>
          {message && (
            <div className={`inputs-banner ${message.tone === "error" ? "warn" : ""}`} role="status">
              <strong>{message.text}</strong>
              {message.problems && <ul className="problem-list">{message.problems.map((p) => <li key={p}>{p}</li>)}</ul>}
            </div>
          )}
        </form>
      </Panel>
      <Panel eyebrow="Overrides" title="Current overrides" actions={<button type="button" className="inputs-button subtle" onClick={() => void reload()}>Refresh</button>}>
        {loadError && <EmptyState title="Couldn't load overrides" detail="Try again with Refresh." />}
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
                      <td>{record.event === "all" ? "All events" : here ? String(eventObj.name ?? "This event") : "Another event"}</td>
                      <td>{SCOPE_LABELS[record.scope] ?? titleCase(record.scope)}</td>
                      <td>{record.target !== undefined ? playerName.get(String(record.target)) || `Player ${record.target}` : ""}</td>
                      <td>{fieldLabel(record.field)}</td>
                      <td>{describeValue(record)}</td>
                      <td className="wrap-cell">{record.reason}</td>
                      <td>{record.author}</td>
                      <td>{etTime(record.created_at)}</td>
                      <td>{etTime(record.expires_at)}</td>
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
      {history.length > 0 && (
        <Panel title="Recent changes">
          <div className="kv-table">
            {history.map((entry) => (
              <div key={`${entry.at}-${entry.record.id}-${entry.action}`}>
                <span>{etTime(entry.at)} · {titleCase(entry.action)}</span>
                <b>{entry.by.email}{entry.by.verified ? "" : " (unverified)"} · {SCOPE_LABELS[entry.record.scope] ?? entry.record.scope}: {fieldLabel(entry.record.field)} {describeValue(entry.record)}</b>
              </div>
            ))}
          </div>
        </Panel>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ page */
const subscribeNarrow = (notify: () => void) => {
  const query = window.matchMedia("(max-width: 640px)");
  query.addEventListener("change", notify);
  return () => query.removeEventListener("change", notify);
};
/** True on a phone-width screen, where the sub-tab strip becomes a dropdown so no tab is hidden off-screen. */
function useNarrow(): boolean {
  return useSyncExternalStore(subscribeNarrow, () => window.matchMedia("(max-width: 640px)").matches, () => false);
}

const OWN_TECHNICAL: Tab[] = ["course", "variance", "weather", "config"];

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
  const narrow = useNarrow();

  const events = useMemo(() => index?.events ?? [], [index]);
  const event = events.find((e) => e.event_uid === eventUid) ?? events[0];
  const runs = event?.runs ?? [];
  const run = runs.find((r) => r.key === runKey) ?? runs.find((r) => r.key === event?.latest_key) ?? runs[0];
  const { data: doc, loading: docLoading, error: docError } = useDashboardData<Obj>(run?.key ?? "golfprice/none.json");
  const clearPreset = useCallback(() => setPresetPlayer(null), []);

  if (loading) return <LoadingState label="Loading published model inputs" />;
  if (error || !event) return <div><PageIntro eyebrow="Model" title="Model inputs" description="Every input that shapes the simulations: player skill, course fit, variance, weather, odds and config." /><ErrorState message={error ?? "No model run has been published yet. It will appear here after the next publish."} /></div>;

  const asOf = etTime(run?.as_of, "");
  const pulled = doc ? etTime(oddsPulledAt(doc), "") : "";
  const runAsOf = String(run?.as_of ?? "");
  const playedRounds = run?.kind === "live" ? run.after_round ?? 0 : 0;
  const runTech = run ? <KeyValue data={{ run_file: run.key, hole_table_file: run.hole_table_key, checksum: run.sha256, size_bytes: run.bytes, schema: index?.schema, overrides_applied: run.overrides_applied, overrides_rejected: run.overrides_rejected }} /> : null;
  const tabOptions = TABS.map((t) => ({ value: t.value, label: t.label }));
  return (
    <div>
      <PageIntro
        eyebrow="Model"
        title="Model inputs"
        description="What went into the simulation: player skill, the course, how much scores swing, weather, odds and settings. Features explains every skill measurement; Adjust applies one-off overrides."
        controls={
          <div className="control-row wrap">
            <Select label="Event" value={event.event_uid} onChange={(v) => { setEventUid(v); setRunKey(""); }} options={events.map((e) => ({ value: e.event_uid, label: `${e.name} (${e.tour.toUpperCase()})${e.runs.length ? "" : " - no saved inputs"}` }))} />
            <Select label="Run" value={run?.key ?? ""} onChange={setRunKey} options={runs.map((r) => ({ value: r.key, label: `${r.kind === "live" ? `Live, after round ${r.after_round}` : "Pre-tournament"} · ${etTime(r.as_of ?? "", "") || "time unknown"}` }))} />
          </div>
        }
      />
      <div className="inputs-meta">
        <span><b>{event.course}</b></span>
        <span>{ymd(event.date_start)} to {ymd(event.date_end)}</span>
        {asOf && <span title="When this model run was built.">model run {asOf}</span>}
        {pulled && <span title="The most recent time any book's prices were pulled. A live run can reuse earlier odds.">odds pulled {pulled}</span>}
        {run && <span>{run.n_players} players</span>}
        <span>{run?.overrides_applied.length ? `${run.overrides_applied.length} override${run.overrides_applied.length === 1 ? "" : "s"} applied` : "no overrides applied"}</span>
      </div>
      {narrow ? <Select label="Section" value={tab} onChange={(v) => setTab(v as Tab)} options={tabOptions} /> : <SegmentedControl label="Model inputs area" value={tab} onChange={setTab} options={tabOptions} />}
      {docLoading && tab !== "features" && <LoadingState label="Loading run" />}
      {docError && tab !== "features" && tab !== "adjust" && <EmptyState title="No saved inputs for this run" detail="Model inputs were not saved for this event or run. Pick another event or run above." />}
      {doc && tab === "players" && <PlayersTab key={run?.key} doc={doc} playedRounds={playedRounds} onAdjust={(id) => { setPresetPlayer(id); setTab("adjust"); }} />}
      {doc && tab === "course" && <CourseTab doc={doc} extra={runTech} />}
      {doc && tab === "variance" && <VarianceTab doc={doc} extra={runTech} />}
      {doc && tab === "weather" && <WeatherTab doc={doc} runAsOf={runAsOf} extra={runTech} />}
      {doc && tab === "odds" && <OddsTab doc={doc} runAsOf={runAsOf} />}
      {doc && tab === "config" && <ConfigTab doc={doc} extra={runTech} />}
      {tab === "features" && <FeaturesTab />}
      {doc && tab === "adjust" && <AdjustTab doc={doc} eventUid={event.event_uid} presetPlayer={presetPlayer} onPresetUsed={clearPreset} />}
      {run && !OWN_TECHNICAL.includes(tab) && (
        <div style={{ marginTop: 24 }}>
          <Technical>{runTech}</Technical>
        </div>
      )}
    </div>
  );
}
