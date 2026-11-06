"use client";

import { CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { positionRows, MISSED_CUT_LABEL } from "./distributions-rules";
import type { PositionChartSeries } from "./ExplainCharts";

const COLORS = ["var(--chart-1)", "var(--chart-3)", "var(--chart-2)", "var(--chart-4)"];
const DASHES = ["", "7 3", "2 3", "9 3 2 3"];
const lastName = (n: string) => n.split(" ").slice(-1)[0];


/** Finish-position curves, one line per player with the name written on the line itself (no legend lookup), a distinct dash per player so colour is never the only signal. */
export function PositionChartLabeled({ series, hasCut, cumulative }: { series: PositionChartSeries[]; hasCut: boolean; cumulative: boolean }) {
  const rows = positionRows(series.map((s) => ({ key: s.key, pos: s.pos, missCut: s.missCut })), cumulative, hasCut).filter((r) => r.label !== MISSED_CUT_LABEL);
  const n = rows.length;
  const main = series.filter((s) => s.style !== "dashed");
  const anchor = (s: PositionChartSeries) => {
    let best = 0, bestV = -1;
    rows.forEach((r, i) => { const v = Number(r[s.key] ?? 0); if (cumulative ? (bestV < 0 && v >= 50) : v > bestV) { best = i; bestV = v; } });
    return cumulative && bestV < 0 ? n - 1 : best;
  };
  const ticks = rows.map((r) => String(r.label)).filter((l, i) => i === 0 || Number(l) % 10 === 0);
  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={rows} margin={{ top: 22, right: 12, bottom: 18, left: 0 }}>
        <CartesianGrid stroke="var(--line)" vertical={false} />
        <XAxis dataKey="label" ticks={ticks} interval={0} stroke="var(--muted)" tick={{ fontSize: 12 }} label={{ value: "Finish position (ties shared)", position: "insideBottom", offset: -10, fill: "var(--muted)", fontSize: 12 }} />
        <YAxis stroke="var(--muted)" width={42} tick={{ fontSize: 12 }} domain={cumulative ? [0, 100] : [0, "auto"]} tickFormatter={(v: number) => `${Math.round(v)}%`} />
        <Tooltip cursor={{ stroke: "var(--line-strong)" }} content={<DistTooltip series={series} cumulative={cumulative} />} />
        {series.map((s) => {
          const idx = main.indexOf(s);
          const ai = idx >= 0 ? anchor(s) : -1;
          return <Line key={s.key} dataKey={s.key} name={s.name} type={cumulative ? "monotone" : "stepAfter"} stroke={s.color} strokeWidth={idx === 0 ? 2.5 : 2} strokeDasharray={s.style === "dashed" ? "2 3" : DASHES[Math.max(COLORS.indexOf(s.color), 0) % DASHES.length] || undefined} strokeOpacity={s.style === "dashed" ? 0.7 : 1} dot={false} activeDot={{ r: 4 }} isAnimationActive={false} connectNulls={false}
            label={idx >= 0 ? ((p: { x?: number | string; y?: number | string; index?: number }) => p.index === ai && typeof p.x === "number" && typeof p.y === "number" ? <text key={s.key} x={p.x + (ai > n * 0.6 ? -6 : 6)} y={p.y - 8 - idx * 17} textAnchor={ai > n * 0.6 ? "end" : "start"} fontSize={12} fontWeight={700} fill={s.color} stroke="var(--surface)" strokeWidth={3} paintOrder="stroke">{lastName(s.name)}</text> : <g key={`${s.key}${p.index}`} />) : undefined} />;
        })}
      </ComposedChart>
    </ResponsiveContainer>
  );
}

function DistTooltip({ active, payload, label, series, cumulative }: { active?: boolean; payload?: Array<{ payload: Record<string, number | string> }>; label?: string; series: PositionChartSeries[]; cumulative: boolean }) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return <div className="dist-tip"><strong>{label === "1" ? "Win (finish 1)" : `Finish ${label}${cumulative ? " or better" : ""}`}</strong>{series.map((s) => { const v = row[s.key]; return <span key={s.key}><i style={{ background: s.color }} />{s.name}: <b>{typeof v === "number" ? `${v.toFixed(v < 1 ? 2 : 1)}%` : "n/a"}</b></span>; })}</div>;
}

