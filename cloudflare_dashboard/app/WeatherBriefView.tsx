"use client";

/**
 * The weather page body, driven by weather_brief.json (golfprice.weather_brief.v1). Top to bottom: headline, this course and weather,
 * round by round, who it helps and hurts, what if the forecast changes, how it has moved this week, glossary. Everything is worded as
 * helps/hurts and easier/harder (weather-brief-rules.ts); the sliders interpolate the published what-if scenarios in the browser.
 */
import Link from "next/link";
import { useMemo, useState } from "react";
import { ChevronRight, CloudSun, Search, Wind } from "lucide-react";
import { Panel } from "./components";
import { useDashboardData } from "./data";
import { Badge } from "./ui";
import { etTime } from "./lib";
import {
  NEUTRAL_SLIDERS, TOP_SKILL, ageWords, buildWhatIf, componentWords, fieldWords, helpWords, historySeries, isNeutral, isTopSkill, medianProbSlopes, movementWords, orderPlayers, pointsWords,
  briefKeyFor, parseBrief, scaledProb, searchBrief, slotWords, spreadWords, statusTitle, tone, waveSide, waveWords, windSensitivity,
  type Brief, type BriefPlayer, type BriefRound, type RoundOut, type Slot, type SortMode, type Sliders, type WhatIfModel, type WhatIfResult,
} from "./weather-brief-rules";
import "./weather-brief.css";

const DEFAULT_GLOSSARY: Record<string, string> = {
  field_effect: "Field effect: how much the weather makes scoring easier or harder for everyone at once. It moves the scores and the cut line, but not who is favoured.",
  wave_split: "Wave split: how much better one tee wave has it than the other for the same player. It only moves the odds between players who tee off in different waves.",
  spread: "Spread: how much the weather widens or narrows the range of scores. Wider helps long shots, narrower helps the favourites.",
  relative_effect: "Relative effect: a player's own weather effect compared with the average player in the field. It is what moves a player's skill for the week.",
};
const GLOSSARY_ORDER = ["field_effect", "wave_split", "spread", "relative_effect"];

const dayName = (iso: string) => {
  const ms = Date.parse(`${iso}T12:00:00Z`);
  return Number.isFinite(ms) ? new Intl.DateTimeFormat("en-US", { weekday: "short", month: "short", day: "numeric", timeZone: "UTC" }).format(ms) : "";
};

/* ------------------------------------------------------------------ (a) headline */
export function HeadlineCard({ brief, version }: { brief: Brief; version: string }) {
  const level = brief.status.level;
  const badgeTone = level === "full" ? "positive" : level === "field_only" ? "warning" : "neutral";
  return (
    <section className="wb-headline" aria-label="Weather headline">
      <div className="wb-headline-top">
        <Badge tone={badgeTone}>{statusTitle(level)}</Badge>
        <small className="wb-muted">
          {brief.as_of ? `Updated ${etTime(brief.as_of)}` : ""}{brief.forecast.age_hours !== null ? ` · forecast ${ageWords(brief.forecast.age_hours)}` : ""}{version ? ` · model ${version}` : ""}
        </small>
      </div>
      {brief.status.plain && <p className="wb-status">{brief.status.plain}</p>}
      {brief.headline.length ? (
        <ul className="wb-sentences">{brief.headline.map((h, i) => <li key={i} className={i === 0 ? "first" : undefined}>{h}</li>)}</ul>
      ) : <p className="wb-sentences wb-muted">No weather headline yet.</p>}
    </section>
  );
}

/* ------------------------------------------------------------------ (b) course */
export function CoursePanel({ brief }: { brief: Brief }) {
  const c = brief.course;
  const s = windSensitivity(c);
  const top = Math.max(c.slope ?? 0, c.tourSlope ?? 0, 0.0001);
  const bar = (label: string, v: number | null, cls: string) => (
    <div className="wb-bar-row">
      <span>{label}</span>
      <div className="wb-bar-track" role="img" aria-label={`${label}: ${v === null ? "unknown" : `${v.toFixed(3)} strokes per mph`}`}><i className={cls} style={{ width: `${v === null ? 0 : Math.max(2, (v / top) * 100)}%` }} /></div>
      <b>{v === null ? "n/a" : `${v.toFixed(2)} strokes per mph`}</b>
    </div>
  );
  return (
    <Panel eyebrow="This course and weather" title={s.label} actions={s.ratio !== null ? <Badge tone="accent">{s.ratio.toFixed(1)}x the tour average</Badge> : undefined}>
      <p className="wb-lead">{c.plain || (s.perFive !== null ? `Five more mph of wind makes the field about ${s.perFive.toFixed(1)} strokes harder per round.` : "No wind sensitivity published for this course yet.")}</p>
      {c.slope !== null && (
        <div className="wb-bars">
          {bar("This course", c.slope, "me")}
          {c.tourSlope !== null && bar("Tour average", c.tourSlope, "tour")}
        </div>
      )}
      <p className="wb-muted">Long-run cost of each extra mph of average wind at this course, from past events (the sentence above re-runs the model on this week's forecast, so it can differ).{c.basis ? ` Based on: ${c.basis}.` : ""}{c.cls ? ` Course type: ${c.cls}.` : ""}</p>
    </Panel>
  );
}

/* ------------------------------------------------------------------ (c) rounds */
function shiftSlot(slot: Slot | null, which: "AM" | "PM", s: Sliders): Slot | null {
  if (!slot) return null;
  const dw = s.wind + (which === "PM" ? s.pm : 0);
  return { ...slot, wind: slot.wind === null ? null : Math.max(0, slot.wind + dw), gust: slot.gust === null ? null : Math.max(0, slot.gust + dw), temp: slot.temp === null ? null : slot.temp + s.temp, rain_prob: s.wet && slot.rain_prob !== null ? Math.max(slot.rain_prob, 0.8) : slot.rain_prob };
}

function dayAsSlot(mode: "shift" | "base", day: BriefRound["forecast"]["day"], s: Sliders): Slot | null {
  if (!day) return null;
  const slot: Slot = { window: "", wind: day.wind, gust: null, temp: day.temp, rain_prob: day.rain_prob };
  return mode === "shift" ? shiftSlot(slot, "AM", s) : slot;
}

function Faint({ now, base, render }: { now: string; base: string; render?: boolean }) {
  return <>{now}{render && now !== base ? <small className="wb-faint"> (forecast: {base})</small> : null}</>;
}

export function RoundStrip({ rounds, live, sliders }: { rounds: BriefRound[]; live: RoundOut[] | null; sliders: Sliders }) {
  const changed = live !== null && !isNeutral(sliders);
  return (
    <div className="wb-rounds">
      {rounds.map((r) => {
        const out = live?.find((x) => x.round === r.round) ?? { round: r.round, field: r.field, wave: r.wave, spread: r.spread };
        const am = changed ? shiftSlot(r.forecast.AM, "AM", sliders) : r.forecast.AM;
        const pm = changed ? shiftSlot(r.forecast.PM, "PM", sliders) : r.forecast.PM;
        const day = r.forecast.day;
        const side = waveSide(out.wave);
        return (
          <article key={r.round} className="wb-round" aria-label={`Round ${r.round}`}>
            <header><b>Round {r.round}</b><small>{dayName(r.date_local)}</small>{r.status === "field_only" && <Badge tone="warning">Tee times not out</Badge>}</header>
            <dl className="wb-fc">
              {am || pm ? (
                <>
                  <dt className="am">Morning</dt><dd>{slotWords(am)}</dd>
                  <dt className="pm">Afternoon</dt><dd>{slotWords(pm)}</dd>
                </>
              ) : (
                <><dt>Day</dt><dd>{slotWords(dayAsSlot(changed ? "shift" : "base", day, sliders))}</dd></>
              )}
            </dl>
            <div className="wb-effects">
              <div className={`wb-effect ${tone(out.field)}`}>
                <span>For everyone</span>
                <strong>Scoring {changed ? <Faint now={fieldWords(out.field)} base={fieldWords(r.field)} render /> : fieldWords(out.field)}</strong>
                <small>Moves scores and the cut line, not who wins.</small>
              </div>
              <div className={`wb-effect wave-${side}`}>
                <span>Morning vs afternoon wave</span>
                <strong>{out.wave === null ? "Tee sheet not out yet" : changed ? <Faint now={waveWords(out.wave)} base={waveWords(r.wave)} render /> : waveWords(out.wave)}</strong>
                <small>{out.wave === null ? "Wave effects arrive with the tee sheet." : (Math.abs(out.wave) < 0.02 ? "Tee times make no real difference this round." : "For the same player, the other wave plays that much harder.")}</small>
              </div>
              <div className="wb-effect flat"><span>Spread of scores</span><strong>{spreadWords(out.spread)}</strong></div>
            </div>
            {r.plain && <p className="wb-muted wb-round-plain">{r.plain}</p>}
          </article>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ (d) players */
function PlayerDetail({ p, live, slopes }: { p: BriefPlayer; live: number | null; slopes: ReturnType<typeof medianProbSlopes> }) {
  const mv = live !== null && p.total !== null && Math.abs(live - p.total) > 0.0005;
  const prob = (label: string, key: "win" | "top_10" | "top_20") => {
    const v = mv ? scaledProb(p, key, live, slopes) : p.prob[key];
    return <li key={key}>{label} chance {mv ? "≈ " : ""}{pointsWords(v)}</li>;
  };
  return (
    <div className="wb-detail">
      <p><b>{p.name}</b>: weather {helpWords(live ?? p.total)} over the rounds with a tee time{p.skill_rank ? ` (skill rank ${p.skill_rank})` : ""}.</p>
      {p.plain && !mv && <p className="wb-muted">{p.plain}</p>}
      <ul className="wb-probs">{prob("Win", "win")}{prob("Top-10", "top_10")}{prob("Top-20", "top_20")}</ul>
      <div className="table-scroll">
        <table aria-label={`Weather effect by round for ${p.name}`}>
          <thead><tr><th>Round</th><th>Draw</th><th>Effect</th><th>Where it comes from</th></tr></thead>
          <tbody>
            {p.per_round.map((r) => (
              <tr key={r.round}>
                <td>R{r.round}</td>
                <td>{r.wave === "AM" ? "Morning" : r.wave === "PM" ? "Afternoon" : "No draw yet"}{r.tee ? `, ${r.tee}` : ""}</td>
                <td className={tone(r.strokes)}>{helpWords(r.strokes)}</td>
                <td>{componentWords(r.components)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function PlayersPanel({ brief, result }: { brief: Brief; result: WhatIfResult | null }) {
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<SortMode>("helped");
  const [topOnly, setTopOnly] = useState(false);
  const [open, setOpen] = useState<number | null>(null);
  const [limit, setLimit] = useState(40);
  const slopes = useMemo(() => medianProbSlopes(brief.players), [brief.players]);
  const liveOf = (p: BriefPlayer) => (result ? (result.totals.get(p.dg_id) ?? p.total) : p.total);
  const rows = useMemo(() => {
    const withLive = brief.players.map((p) => ({ ...p, total: result ? (result.totals.get(p.dg_id) ?? p.total) : p.total }));
    const filtered = searchBrief(withLive, query).filter((p) => !topOnly || isTopSkill(p));
    return orderPlayers(filtered, sort);
  }, [brief.players, result, query, sort, topOnly]);
  const scale = Math.max(0.05, ...brief.players.flatMap((p) => [p.total, result?.totals.get(p.dg_id) ?? null]).filter((v): v is number => v !== null).map(Math.abs));
  if (!brief.players.length) return <Panel eyebrow="Who it helps and hurts" title="Player effects"><p className="wb-muted">No player effects yet: they appear once tee times are published. Until then only the field-wide effect above is in the prices.</p></Panel>;
  const shown = query ? rows : rows.slice(0, limit);
  const pos = (v: number) => `${Math.min(50, (Math.abs(v) / scale) * 50)}%`;
  return (
    <Panel eyebrow="Who it helps and hurts" title="Each player's weather effect" actions={<Badge tone="neutral">Rounds with a tee time</Badge>}>
      <p className="wb-lead">The bars show how many strokes the weather and tee time move each player&apos;s skill, compared with the average player in the field. Green bars to the right help, red bars to the left hurt.</p>
      <div className="wb-toolbar">
        <label className="wb-search"><Search size={16} aria-hidden="true" /><input type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search player" aria-label="Search player" /></label>
        <label className="wb-select"><span>Order</span>
          <select value={sort} onChange={(e) => setSort(e.target.value as SortMode)}>
            <option value="helped">Helped most first</option><option value="hurt">Hurt most first</option><option value="skill">By skill rank</option><option value="name">By name</option>
          </select>
        </label>
        <label className="wb-check"><input type="checkbox" checked={topOnly} onChange={(e) => setTopOnly(e.target.checked)} /> Top {TOP_SKILL} by skill only</label>
        <small className="wb-muted">{rows.length} of {brief.players.length} players</small>
      </div>
      <div className="wb-axis" aria-hidden="true"><span>← Hurts</span><span>Helps →</span></div>
      <ul className="wb-players">
        {shown.map((p) => {
          const base = brief.players.find((b) => b.dg_id === p.dg_id)?.total ?? null;
          const v = p.total;
          const isOpen = open === p.dg_id;
          const t = tone(v);
          return (
            <li key={p.dg_id} className={`${isTopSkill(p) ? "top" : ""} ${isOpen ? "open" : ""}`}>
              <button type="button" aria-expanded={isOpen} onClick={() => setOpen(isOpen ? null : p.dg_id)}>
                <span className="wb-pname">{p.name}{isTopSkill(p) && <em title={`Top ${TOP_SKILL} by skill`}>top {TOP_SKILL}</em>}</span>
                <span className="wb-track" aria-hidden="true">
                  <i className="mid" />
                  {v !== null && <i className={`bar ${t}`} style={t === "good" ? { left: "50%", width: pos(v) } : t === "bad" ? { right: "50%", width: pos(v) } : { left: "50%", width: "1px" }} />}
                  {result && base !== null && v !== null && Math.abs(base - v) > 0.0005 && <i className="ghost" style={{ left: `${50 - Math.max(-50, Math.min(50, (base / scale) * 50))}%` }} />}
                </span>
                <span className={`wb-val ${t}`}>{helpWords(v)}{result && result.approx.has(p.dg_id) ? " ≈" : ""}</span>
              </button>
              {isOpen && <PlayerDetail p={brief.players.find((b) => b.dg_id === p.dg_id) ?? p} live={liveOf(p)} slopes={slopes} />}
            </li>
          );
        })}
        {shown.length === 0 && <li className="wb-muted wb-none">No player matches.</li>}
      </ul>
      {!query && rows.length > limit && <button type="button" className="inputs-button" onClick={() => setLimit((l) => l + 60)}>Show more ({rows.length - limit} left)</button>}
    </Panel>
  );
}

/* ------------------------------------------------------------------ (e) what if */
function sliderWords(s: Sliders): string {
  const parts: string[] = [];
  if (s.wind !== 0) parts.push(`wind ${Math.abs(s.wind)} mph ${s.wind > 0 ? "stronger" : "lighter"}`);
  if (s.pm !== 0) parts.push(`afternoon wind another ${s.pm} mph stronger`);
  if (s.temp !== 0) parts.push(`${Math.abs(s.temp)}°F ${s.temp > 0 ? "warmer" : "colder"}`);
  if (s.wet) parts.push("wet");
  return parts.length ? `If the forecast turns out: ${parts.join(", ")}.` : "Move a slider to see how the forecast changing would move the numbers above and below.";
}

export function WhatIfPanel({ model, sliders, onChange }: { model: WhatIfModel | null; sliders: Sliders; onChange: (s: Sliders) => void }) {
  if (!model) return <Panel eyebrow="What if the forecast changes" title="Try a different forecast"><p className="wb-muted">What-if numbers are not published for this run yet.</p></Panel>;
  const set = (patch: Partial<Sliders>) => onChange({ ...sliders, ...patch });
  // The ≈ note only when some player is actually scaled (briefs with player_totals give everyone an exact number).
  const approxNote = model.evaluate({ wind: model.ranges.wind[1], pm: model.ranges.pm[1], temp: model.ranges.temp[1], wet: model.axes.wet }).approx.size > 0;
  const slide = (id: string, label: string, min: number, max: number, value: number, unitText: string, key: "wind" | "pm" | "temp", enabled: boolean) => (
    <label className={`wb-slider${enabled ? "" : " off"}`} htmlFor={id}>
      <span>{label}</span>
      <input id={id} type="range" min={min} max={max} step={1} value={value} disabled={!enabled} onChange={(e) => set({ [key]: Number(e.target.value) } as Partial<Sliders>)} />
      <b>{value === 0 ? "as forecast" : `${value > 0 ? "+" : "−"}${Math.abs(value)} ${unitText}`}</b>
    </label>
  );
  return (
    <Panel eyebrow="What if the forecast changes" title="Try a different forecast" actions={!isNeutral(sliders) ? <button type="button" className="inputs-button" onClick={() => onChange(NEUTRAL_SLIDERS)}>Reset to forecast</button> : undefined}>
      <p className="wb-lead" role="status" aria-live="polite">{sliderWords(sliders)}</p>
      <div className="wb-sliders">
        {slide("wb-wind", "Wind, all day", -10, 10, sliders.wind, "mph", "wind", model.axes.wind)}
        {slide("wb-pm", "Extra afternoon wind", 0, 10, sliders.pm, "mph", "pm", model.axes.pm)}
        {slide("wb-temp", "Temperature", -10, 10, sliders.temp, "°F", "temp", model.axes.temp)}
        <label className={`wb-check wb-wet${model.axes.wet ? "" : " off"}`}><input type="checkbox" checked={sliders.wet} disabled={!model.axes.wet} onChange={(e) => set({ wet: e.target.checked })} /> Wet weather</label>
      </div>
      <p className="wb-muted">The round cards above and the player bars update as you move these. The faint text and the faint tick marks show the current forecast. Between the published what-if runs the page blends them.{approxNote ? " Players outside the biggest movers are scaled from their current effect (shown with ≈)." : ""}</p>
    </Panel>
  );
}

/* ------------------------------------------------------------------ (f) history */
function Spark({ values, label }: { values: Array<number | null>; label: string }) {
  const v = values.map((x) => x);
  const real = v.filter((x): x is number => x !== null);
  if (real.length < 2) return null;
  const lo = Math.min(...real), hi = Math.max(...real), span = Math.max(hi - lo, 0.1);
  const W = 120, H = 36;
  const pts = v.map((x, i) => (x === null ? null : [4 + (i / (v.length - 1)) * (W - 8), 4 + (1 - (x - lo) / span) * (H - 8)] as const));
  const path = pts.filter((p): p is readonly [number, number] => p !== null).map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
  const last = pts.filter((p): p is readonly [number, number] => p !== null).at(-1)!;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="wb-spark" role="img" aria-label={label}>
      <path d={path} fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={last[0]} cy={last[1]} r="3" fill="var(--accent)" />
    </svg>
  );
}

export function HistoryPanel({ brief }: { brief: Brief }) {
  const series = useMemo(() => historySeries(brief), [brief]);
  if (!series.length) return null;
  return (
    <Panel eyebrow="How it has moved this week" title="Forecast changes since the first look">
      <div className="wb-history">
        {series.map((s) => {
          const f = movementWords(s.field, "field");
          const w = movementWords(s.wave, "wave");
          return (
            <article key={s.round} className="wb-hist">
              <b>Round {s.round}</b>
              <div><small className="wb-muted">Scoring for everyone</small><Spark values={s.field} label={`Round ${s.round} scoring effect over ${s.field.length} forecasts`} /><small>{f || "Not enough history"}</small></div>
              {s.wave.some((x) => x !== null) && <div><small className="wb-muted">Wave advantage</small><Spark values={s.wave} label={`Round ${s.round} wave split over ${s.wave.length} forecasts`} /><small>{w || "Not enough history"}</small></div>}
            </article>
          );
        })}
      </div>
    </Panel>
  );
}

/* ------------------------------------------------------------------ (g) glossary */
export function GlossaryPanel({ brief }: { brief: Brief }) {
  return (
    <Panel eyebrow="In plain English" title="What the words mean">
      <dl className="wb-gloss">
        {GLOSSARY_ORDER.map((k) => (
          <div key={k}><dt>{k === "field_effect" ? "Field effect" : k === "wave_split" ? "Wave split" : k === "spread" ? "Spread" : "Relative effect"}</dt><dd>{DEFAULT_GLOSSARY[k].replace(/^[^:]+: /, "")}</dd></div>
        ))}
      </dl>
    </Panel>
  );
}

/* ------------------------------------------------------------------ whole body */
export function WeatherBriefBody({ brief, modelVersion }: { brief: Brief; modelVersion?: string }) {
  const [sliders, setSliders] = useState<Sliders>(NEUTRAL_SLIDERS);
  const model = useMemo(() => buildWhatIf(brief), [brief]);
  const result = useMemo(() => (model && !isNeutral(sliders) ? model.evaluate(sliders) : null), [model, sliders]);
  const version = brief.model_version || modelVersion || "";
  return (
    <div className="wb-body">
      <HeadlineCard brief={brief} version={version} />
      <CoursePanel brief={brief} />
      <Panel eyebrow="Round by round" title="What the weather does each day" actions={<CloudSun size={18} aria-hidden="true" />}>
        {brief.rounds.length ? <RoundStrip rounds={brief.rounds} live={result ? result.rounds : null} sliders={sliders} /> : <p className="wb-muted">No round forecast yet.</p>}
      </Panel>
      <PlayersPanel brief={brief} result={result} />
      <WhatIfPanel model={model} sliders={sliders} onChange={setSliders} />
      <HistoryPanel brief={brief} />
      <GlossaryPanel brief={brief} />
      <p className="wb-muted wb-foot"><Wind size={13} aria-hidden="true" /> {brief.forecast.source === "archive" ? "From an archived forecast." : "From the latest live forecast."}{brief.forecast.n_scenarios ? ` ${brief.forecast.n_scenarios} forecast scenarios.` : ""} Strokes are per round.</p>
    </div>
  );
}

/** A prominent link to the weather page for an event, with the brief's first sentence when it is published. */
export function WeatherLinkCard({ event }: { event: { event_uid: string; weather_brief_key?: string | null } }) {
  const key = briefKeyFor(event);
  const { data } = useDashboardData<unknown>(key);
  const brief = useMemo(() => parseBrief(data), [data]);
  return (
    <div className="wb-link-card" data-testid="weather-link">
      <CloudSun size={20} aria-hidden="true" />
      <strong>Weather this week</strong>
      <span>{brief?.headline[0] ?? "How the forecast changes scoring at this course, who it helps and hurts, and what changes if the forecast moves."}</span>
      <Link className="ex-link" href={`/weather-effects?e=${encodeURIComponent(event.event_uid)}`}>Open the weather page <ChevronRight size={14} /></Link>
    </div>
  );
}
