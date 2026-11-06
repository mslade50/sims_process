"use client";

import { lazy, useMemo, useState } from "react";
import { useDashboardData } from "./data";
import { Block } from "./score-block";
import { ChartSlot } from "./ChartSlot";
import { DataThrough } from "./DataThrough";
import { dataThrough } from "./freshness-rules";
import { historyRoundRows, parseCourseHistory, signed, type HistoryMetric } from "./course-history-rules";

// Recharts loads after the first paint, inside a box whose height is set in scoring.css.
const HistoryChart = lazy(() => import("./CourseHistoryChart").then((m) => ({ default: m.HistoryChart })));

/** One hue, light (oldest year) to full strength (most recent): years are ordered, so a sequential ramp, never a rainbow. */
const yearColor = (i: number, n: number) => `color-mix(in oklab, var(--chart-1) ${n <= 1 ? 100 : Math.round(28 + (72 * i) / (n - 1))}%, var(--surface))`;
const fmt = (n: number | null, d = 1) => n == null ? "—" : n.toFixed(d);
const METRICS: Array<{ key: HistoryMetric; label: string; unit: string; hint: string }> = [
  { key: "vs_par", label: "Score vs par", unit: "strokes", hint: "Field average score minus par, as it actually happened." },
  { key: "adjusted_vs_par", label: "Adjusted vs par", unit: "strokes", hint: "What an average-strength tour field would have shot in typical weather: score vs par, plus field skill, minus the weather cost." },
  { key: "wind", label: "Wind", unit: "mph", hint: "Average wind during the players' rounds (DataGolf round weather)." },
  { key: "weather_strokes", label: "Weather cost", unit: "strokes", hint: "Strokes the weather model says that day's conditions added (+) or saved (−) against a typical tour day." },
  { key: "field_skill", label: "Field skill", unit: "strokes", hint: "Average skill of the players in that round, in strokes per round against an average tour field (+ = stronger)." },
];

/** Earlier editions of this event / course, round by round: field average, weather and field skill (golfprice/history/<event>.json). */
export function CourseHistoryPanel({ historyKey }: { historyKey: string | null | undefined }) {
  const src = useDashboardData<unknown>(historyKey ?? null);
  const doc = useMemo(() => parseCourseHistory(src.data), [src.data]);
  const [metric, setMetric] = useState<HistoryMetric>("vs_par");
  const title = "This event in past years", eyebrow = "Field scoring, weather and field skill by round";
  if (!historyKey) return <Block id="history" title={title} eyebrow={eyebrow}><p className="score-footnote">No course history is published for this event yet. It appears after the next publish.</p></Block>;
  if (src.loading) return null;
  if (!doc || doc.status !== "available") return <Block id="history" title={title} eyebrow={eyebrow}><p className="score-footnote">{doc?.reason ?? src.error ?? "Course history is unavailable for this event."}</p></Block>;
  const m = METRICS.find((x) => x.key === metric)!;
  const { rows, years } = historyRoundRows(doc, metric);
  const multiCourse = new Set(doc.editions.flatMap((e) => e.rounds.map((r) => r.course))).size > 1;
  const otherEvents = doc.editions.filter((e) => !e.same_event);
  return <Block id="history" title={title} eyebrow={eyebrow} fold>
    <DataThrough text={dataThrough(doc.through, "Course history through")} />
    {doc.by_round.length > 0 && <div className="score-hist-cards">{doc.by_round.map((r) => <div key={r.round}>
      <span>Round {r.round} · {r.editions} edition{r.editions === 1 ? "" : "s"}</span>
      <b title="Average field score minus par across the earlier editions at this course.">{signed(r.vs_par)} <small>vs par</small></b>
      <small title={METRICS[1].hint}>Adjusted {signed(r.adjusted_vs_par)} · weather {signed(r.weather_strokes)} · field {signed(r.field_skill)}</small>
      {(r.wind != null || r.temp != null) && <small>{[r.wind != null ? `Wind ${r.wind.toFixed(1)} mph` : null, r.temp != null ? `${r.temp.toFixed(0)}°F` : null].filter(Boolean).join(" · ")}</small>}
    </div>)}</div>}
    <div className="score-chart-tools"><div className="score-segments" role="group" aria-label="History measure">{METRICS.map((x) => <button key={x.key} type="button" title={x.hint} className={metric === x.key ? "active" : ""} aria-pressed={metric === x.key} onClick={() => setMetric(x.key)}>{x.label}</button>)}</div><span>{years[0]}–{years[years.length - 1]} · {m.unit} · hover a bar for that day</span></div>
    <div className="score-hist-chart" role="img" aria-label={`${m.label} by round, one bar per year`}><ChartSlot near><HistoryChart rows={rows} years={years} metric={metric} unit={m.unit} /></ChartSlot></div>
    <div className="score-chart-legend score-hist-legend">{years.map((y, i) => <span key={y}><i style={{ background: yearColor(i, years.length) }} />{y}</span>)}</div>
    <p className="score-footnote">{m.hint}</p>
    <details className="score-details"><summary>Every past round</summary><div className="score-table-scroll"><table className="score-observed-table">
      <caption>Newest first. Adjusted = score vs par + field skill − weather cost: the score an average tour field would post in typical weather.</caption>
      <thead><tr><th>Year</th><th>Round</th>{multiCourse && <th>Course</th>}<th title="Players who finished the round.">Players</th><th>Field avg</th><th>Vs par</th><th>Wind (mph)</th><th>Temp (°F)</th><th title="Average rain during the round (DataGolf, inches).">Rain (in)</th><th title={METRICS[3].hint}>Weather cost</th><th title={METRICS[4].hint}>Field skill</th><th title={METRICS[1].hint}>Adjusted</th></tr></thead>
      <tbody>{doc.editions.flatMap((e) => e.rounds.map((r, i) => <tr key={`${e.event_uid}-${r.round}-${r.course}-${i}`}>
        <th scope="row" title={e.name ?? undefined}>{e.year}{!e.same_event ? "*" : ""}</th><td>{r.round}</td>{multiCourse && <td>{r.course ?? "—"}</td>}<td>{r.players ?? "—"}</td><td>{fmt(r.field_avg, 2)}</td><td>{signed(r.vs_par)}</td><td>{fmt(r.wind)}</td><td>{fmt(r.temp, 0)}</td><td>{fmt(r.rain, 2)}</td><td>{signed(r.weather_strokes)}</td><td>{signed(r.field_skill)}</td><td>{signed(r.adjusted_vs_par)}</td>
      </tr>))}</tbody>
    </table></div></details>
    {otherEvents.length > 0 && <p className="score-footnote">* A different event played at this course: {[...new Set(otherEvents.map((e) => `${e.name ?? e.event_uid} (${e.year})`))].join(", ")}.</p>}
    {doc.weather_model && <p className="score-footnote">{doc.weather_model}</p>}
    <p className="score-footnote">Round averages above use this course only, weighted by players. Field skill is the model&rsquo;s skill estimate for those players at the time; weather cost comes from the weather model. Shown for context; nothing on this panel changes the prices.</p>
  </Block>;
}

/** Hover card for one bar: everything about that year's round (per course when it was played on more than one). */
