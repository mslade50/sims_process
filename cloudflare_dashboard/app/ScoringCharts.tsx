"use client";

// The three Recharts charts of the Scoring page (about 100 KB gzip). ScoringView loads this file with lazy() after the first screen has painted,
// inside boxes whose height is already set in scoring.css, so the page does not move when the charts arrive.
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const fmt = (n: number | null | undefined, d = 2) => n == null || !Number.isFinite(n) ? "—" : n.toFixed(d);
const tooltipStyle = { background: "var(--popover)", border: "1px solid var(--line)", borderRadius: 10 };

export type OutcomeSeries = { key: string; name: string; color: string };

export function OutcomeChart({ curveMode, rows, series, median, lo, hi, single, par }: {
  curveMode: "pmf" | "cdf"; rows: Array<Record<string, number>>; series: OutcomeSeries[]; median: number | null; lo: number | null; hi: number | null; single: boolean; par: number | null;
}) {
  const seriesName = (key: unknown) => series.find((x) => x.key === String(key))?.name ?? String(key);
  const band = lo != null && hi != null && single ? <ReferenceArea x1={lo} x2={hi} fill="var(--accent)" fillOpacity={0.1} stroke="none" ifOverflow="hidden" /> : null;
  const medLine = median != null && single ? <ReferenceLine x={median} stroke="var(--accent)" strokeWidth={2} label={{ value: `Median ${fmt(median, 0)}`, position: "insideTopRight", fill: "var(--foreground)", fontSize: 12 }} /> : null;
  const parLine = par != null ? <ReferenceLine x={par} stroke="var(--foreground)" strokeDasharray="4 3" label={{ value: `Par ${par}`, position: "top", fill: "var(--foreground)", fontSize: 12 }} /> : null;
  const xLabel = { value: `Round score, strokes (par ${par ?? "n/a"})`, position: "insideBottom" as const, offset: -4, fill: "var(--muted)", fontSize: 12 };
  return <ResponsiveContainer width="100%" height="100%">{curveMode === "pmf"
    ? <BarChart data={rows} margin={{ top: 18, right: 16, left: 0, bottom: 18 }}><CartesianGrid vertical={false} stroke="var(--line)" />{band}{medLine}{parLine}<XAxis dataKey="score" type="number" domain={["dataMin - 0.5", "dataMax + 0.5"]} allowDecimals={false} tickFormatter={(v) => fmt(Number(v), 0)} tick={{ fontSize: 11 }} stroke="var(--muted)" label={xLabel} /><YAxis unit="%" domain={[0, "auto"]} width={45} tick={{ fontSize: 11 }} stroke="var(--muted)" tickFormatter={(v) => Number(v).toFixed(0)} /><Tooltip contentStyle={tooltipStyle} formatter={(v, key) => [`${Number(v).toFixed(1)}%`, seriesName(key)]} labelFormatter={(v) => `Exactly ${Number(v).toFixed(0)}`} />{series.map((x) => <Bar key={x.key} dataKey={x.key} name={x.name} fill={x.color} fillOpacity={series.length > 1 ? 0.6 : 0.9} radius={[3, 3, 0, 0]} isAnimationActive={false} />)}</BarChart>
    : <LineChart data={rows} margin={{ top: 18, right: 16, left: 0, bottom: 18 }}><CartesianGrid vertical={false} stroke="var(--line)" />{band}{medLine}{parLine}<XAxis dataKey="score" type="number" domain={["dataMin", "dataMax"]} allowDecimals={false} tickFormatter={(v) => fmt(Number(v), 0)} tick={{ fontSize: 11 }} stroke="var(--muted)" label={xLabel} /><YAxis unit="%" domain={[0, 100]} width={45} tick={{ fontSize: 11 }} stroke="var(--muted)" /><Tooltip contentStyle={tooltipStyle} formatter={(v, key) => [`${Number(v).toFixed(1)}%`, seriesName(key)]} labelFormatter={(v) => `Score of ${Number(v).toFixed(0)} or lower`} />{series.map((x) => <Line key={x.key} dataKey={x.key} name={x.name} type="monotone" stroke={x.color} strokeWidth={2.5} dot={false} isAnimationActive={false} />)}</LineChart>}</ResponsiveContainer>;
}

export function HoleChart({ holes }: { holes: Array<{ hole: number; par: number; exp: number }> }) {
  return <ResponsiveContainer width="100%" height="100%"><BarChart data={holes} margin={{ top: 8, right: 8, bottom: 4, left: 0 }}><CartesianGrid vertical={false} stroke="var(--line)" /><XAxis dataKey="hole" interval={0} tick={{ fontSize: 10 }} stroke="var(--muted)" label={{ value: "Hole", position: "insideBottom", offset: -3, fill: "var(--muted)" }} /><YAxis width={40} tick={{ fontSize: 10 }} stroke="var(--muted)" tickFormatter={(v) => Number(v) > 0 ? `+${v}` : String(v)} /><ReferenceLine y={0} stroke="var(--muted)" /><Tooltip contentStyle={tooltipStyle} formatter={(v) => [`${Number(v) > 0 ? "+" : ""}${Number(v).toFixed(2)} vs par`, "Expected strokes"]} labelFormatter={(h) => `Hole ${h} · par ${holes.find((x) => x.hole === Number(h))?.par ?? "—"}`} /><Bar dataKey="exp" name="Expected strokes vs par" radius={[3, 3, 0, 0]} isAnimationActive={false}>{holes.map((h) => <Cell key={h.hole} fill={h.par === 3 ? "var(--chart-2)" : h.par === 5 ? "var(--chart-3)" : "var(--chart-1)"} />)}</Bar></BarChart></ResponsiveContainer>;
}

export function SensitivityChart({ rows, shift }: { rows: Array<{ shift: number; over: number; under: number }>; shift: number }) {
  return <ResponsiveContainer width="100%" height="100%"><LineChart data={rows} margin={{ top: 12, right: 18, left: 0, bottom: 8 }}><CartesianGrid vertical={false} stroke="var(--line)" /><XAxis dataKey="shift" tickFormatter={(v) => `${Number(v) > 0 ? "+" : ""}${v}`} tick={{ fontSize: 10 }} stroke="var(--muted)" label={{ value: "Shift in scores (strokes)", position: "insideBottom", offset: -3, fill: "var(--muted)" }} /><YAxis unit="%" domain={[0, 100]} width={45} tick={{ fontSize: 10 }} stroke="var(--muted)" /><Tooltip contentStyle={tooltipStyle} formatter={(v, name) => [`${Number(v).toFixed(1)}%`, String(name).toUpperCase()]} labelFormatter={(v) => `Scores shifted ${Number(v) > 0 ? "+" : ""}${v} strokes`} /><ReferenceLine x={shift} stroke="var(--muted)" strokeDasharray="3 4" /><Line dataKey="over" name="Over" stroke="var(--chart-1)" strokeWidth={2.5} dot={{ r: 3 }} isAnimationActive={false} /><Line dataKey="under" name="Under" stroke="var(--chart-3)" strokeWidth={2.5} dot={{ r: 3 }} isAnimationActive={false} /></LineChart></ResponsiveContainer>;
}
