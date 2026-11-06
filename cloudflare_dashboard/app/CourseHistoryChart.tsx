"use client";

// The Course history bar chart (Recharts), loaded by CourseHistoryPanel after the first paint inside a box whose height is set in scoring.css.
import { Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { signed, type HistoryMetric, type RoundCell } from "./course-history-rules";

/** One hue, light (oldest year) to full strength (most recent): years are ordered, so a sequential ramp, never a rainbow. */
const yearColor = (i: number, n: number) => `color-mix(in oklab, var(--chart-1) ${n <= 1 ? 100 : Math.round(28 + (72 * i) / (n - 1))}%, var(--surface))`;

export function HistoryChart({ rows, years, metric, unit }: { rows: Array<Record<string, unknown>>; years: number[]; metric: HistoryMetric; unit: string }) {
  return <ResponsiveContainer width="100%" height="100%"><BarChart data={rows} margin={{ top: 10, right: 16, left: 0, bottom: 4 }} barGap={2} barCategoryGap="18%">
      <CartesianGrid vertical={false} stroke="var(--line)" /><XAxis dataKey="label" tick={{ fontSize: 11 }} stroke="var(--muted)" /><YAxis width={42} tick={{ fontSize: 11 }} stroke="var(--muted)" tickFormatter={(v) => unit === "mph" ? String(v) : signed(Number(v), 1)} domain={([lo, hi]: readonly number[]) => [Math.min(0, lo), Math.max(0, hi)]} />
      {unit !== "mph" && <ReferenceLine y={0} stroke="var(--muted)" />}
      <Tooltip shared={false} cursor={{ fill: "var(--tint-1)" }} content={<RoundCard metric={metric} />} />
      {years.map((y, i) => <Bar key={y} dataKey={`y${y}`} name={String(y)} fill={yearColor(i, years.length)} radius={[3, 3, 0, 0]} minPointSize={(v: number | null | undefined) => (v == null ? 0 : 2)} isAnimationActive={false} />)}
    </BarChart></ResponsiveContainer>;
}

type CardProps = { metric: HistoryMetric; active?: boolean; payload?: Array<{ dataKey?: unknown; payload?: Record<string, unknown> }> };
/** Hover card for one bar: everything about that year's round (per course when it was played on more than one). */
function RoundCard({ active, payload }: CardProps) {
  const hit = active && payload?.length ? payload[0] : null;
  const year = hit ? String(hit.dataKey ?? "").slice(1) : "";
  const cell = hit?.payload?.[`c${year}`] as RoundCell | undefined;
  if (!cell) return null;
  const row = (k: string, v: string) => <div><span>{k}</span><b>{v}</b></div>;
  return <div className="score-hist-card">
    <strong>{cell.year} · {String(hit?.payload?.label ?? "")}{cell.same_event ? "" : " *"}</strong>
    {row("Field average", cell.field_avg == null ? "—" : `${cell.field_avg.toFixed(2)} (${signed(cell.vs_par)} vs par)`)}
    {row("Wind", cell.wind == null ? "—" : `${cell.wind.toFixed(1)} mph`)}
    {row("Temperature", cell.temp == null ? "—" : `${cell.temp.toFixed(0)}°F`)}
    {row("Rain", cell.rain == null ? "—" : `${cell.rain.toFixed(2)} in`)}
    {row("Weather cost", `${signed(cell.weather_strokes)} strokes`)}
    {row("Field skill", `${signed(cell.field_skill)} strokes`)}
    {row("Adjusted vs par", `${signed(cell.adjusted_vs_par)} strokes`)}
    {row("Players", cell.players == null ? "—" : String(cell.players))}
    {cell.courses.length > 1 && <small>{cell.courses.map((c) => `${c.course ?? "?"}: ${signed(c.vs_par)} vs par`).join(" · ")}</small>}
  </div>;
}
