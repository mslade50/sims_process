"use client";

import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { palette } from "./lib";

/** The Model inputs charts. They live in their own chunk so Recharts stays off the path to the first screen; each sits in a box whose height the page's CSS sets. */
type TipProps = { active?: boolean; payload?: Array<{ name?: string; value?: unknown; color?: string; payload?: { name?: string; hint?: string } }>; label?: unknown };

function ChartTip({ active, payload, label }: TipProps) {
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

export type WaterfallStep = { name: string; value: number; hint: string; range: [number, number]; end: number };

export function WaterfallChart({ steps }: { steps: WaterfallStep[] }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={steps} layout="vertical" margin={{ top: 6, right: 24, bottom: 6, left: 8 }}>
        <CartesianGrid stroke="var(--line)" horizontal={false} />
        <XAxis type="number" tickCount={7} tickFormatter={(v: number) => (Math.round(v * 100) / 100).toFixed(2)} tick={{ fill: "var(--muted-strong)", fontSize: 12 }} />
        <YAxis type="category" dataKey="name" width={200} tick={{ fill: "var(--muted-strong)", fontSize: 12 }} />
        <ReferenceLine x={0} stroke="var(--line-strong)" />
        <Tooltip content={<ChartTip />} />
        <Bar dataKey="range" name="Component (start to end)" radius={3} isAnimationActive={false}>
          {steps.map((step, index) => (
            <Cell key={step.name} fill={index === steps.length - 1 ? "var(--model)" : step.value >= 0 ? "var(--positive)" : "var(--negative)"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function HoleChart({ data }: { data: Array<{ hole: number | string; expected: number }> }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 10, right: 16, bottom: 4, left: 0 }}>
        <CartesianGrid stroke="var(--line)" vertical={false} />
        <XAxis dataKey="hole" tick={{ fill: "var(--muted)", fontSize: 12 }} />
        <YAxis tick={{ fill: "var(--muted)", fontSize: 12 }} label={{ value: "Expected score vs par (red = hard, green = easy)", angle: -90, fill: "var(--muted)", fontSize: 12, position: "insideLeft" }} />
        <Tooltip content={<ChartTip />} />
        <ReferenceLine y={0} stroke="var(--line-strong)" />
        <Bar dataKey="expected" name="Expected score vs par" radius={3} isAnimationActive={false}>
          {data.map((point) => <Cell key={point.hole} fill={point.expected > 0 ? "var(--negative)" : "var(--positive)"} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function SpreadChart({ points }: { points: Array<Record<string, unknown>> }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <ScatterChart margin={{ top: 10, right: 16, bottom: 18, left: 0 }}>
        <CartesianGrid stroke="var(--line)" />
        <XAxis type="number" dataKey="mu" name="Rating" tick={{ fill: "var(--muted)", fontSize: 12 }} label={{ value: "Rating (strokes per round vs field)", fill: "var(--muted)", fontSize: 12, position: "insideBottom", offset: -8 }} />
        <YAxis type="number" dataKey="sd" name="Std dev" domain={["auto", "auto"]} tick={{ fill: "var(--muted)", fontSize: 12 }} label={{ value: "Std dev", angle: -90, fill: "var(--muted)", fontSize: 12, position: "insideLeft" }} />
        <ZAxis range={[26, 26]} />
        <Tooltip content={<ChartTip />} />
        <Scatter data={points} fill={palette[0]} name="Player" isAnimationActive={false} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

export function WindChart({ data }: { data: Array<Record<string, unknown>> }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 10, right: 16, bottom: 4, left: 0 }}>
        <CartesianGrid stroke="var(--line)" vertical={false} />
        <XAxis dataKey="label" tick={{ fill: "var(--muted)", fontSize: 12 }} />
        <YAxis tick={{ fill: "var(--muted)", fontSize: 12 }} allowDecimals={false} />
        <Tooltip content={<ChartTip />} />
        <Bar dataKey="wind" name="Average wind (mph)" fill={palette[2]} radius={3} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}
