"use client";

/**
 * Charts for the explain views (This week, Why priced, player drawer). Colors only from design tokens (var(--...)); every chart has a text alternative
 * (aria-label / visible numbers) and never relies on color alone (sign glyphs, labels). Pure data shaping lives in explain-rules.ts.
 */
import { Bar, BarChart, CartesianGrid, Cell, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { MISSED_CUT_LABEL, positionRows, positionTicks } from "./distributions-rules";
import {
  pct, signed, type HoleRow,
} from "./explain-rules";

export * from "./ExplainBits";

/* ------------------------------------------------------------------ finish shape */
export type FinishSeries = { id: number; name: string; pos: number[]; missCut: number | null; color: string };
/** One series of a per-position finish histogram: "bar" = filled bars (the main player), "line" = thin step outline, "dashed" = dashed outline (an earlier run). */
export type PositionChartSeries = { key: string; name: string; color: string; pos: number[]; missCut: number | null | undefined; style: "bar" | "line" | "dashed" };

/** Finish histogram with one bar per finish position (1, 2, 3 ... last finisher); the missed-cut chance is reported beside the chart. The first "bar" series is drawn as bars; the others as thin step outlines on top, so shapes compare at a glance. */
export function PositionChart({ series, hasCut, cumulative, height, ariaLabel }: { series: PositionChartSeries[]; hasCut: boolean; cumulative: boolean; height?: number; ariaLabel: string }) {
  // The missed-cut chance is one outcome covering many players; as a bar it dwarfs every finishing position, so it is shown in the legend instead.
  const withMc = false;
  const base = positionRows(series, cumulative, hasCut).filter((r) => r.label !== MISSED_CUT_LABEL);
  const lineKeys = new Set(series.filter((s) => s.style !== "bar").map((s) => s.key));
  const rows = base.map((r) => {
    if (r.label !== MISSED_CUT_LABEL) return r;
    const o: Record<string, number | string> = { label: r.label };
    for (const [k, v] of Object.entries(r)) if (k !== "label") o[lineKeys.has(k) ? `${k}__mc` : k] = v;
    return o;
  });
  const n = rows.filter((r) => r.label !== MISSED_CUT_LABEL).length;
  const bar = series.find((s) => s.style === "bar");
  return (
    <div className="ex-chart" style={height ? { height } : undefined} role="img" aria-label={ariaLabel}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} barCategoryGap={0} margin={{ top: 10, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="var(--line)" vertical={false} />
          <XAxis dataKey="label" stroke="var(--muted)" ticks={positionTicks(n, withMc)} interval={0} tick={{ fontSize: 11 }} />
          <YAxis stroke="var(--muted)" width={38} unit="%" tick={{ fontSize: 11 }} domain={cumulative ? [0, 100] : [0, "auto"]} tickFormatter={(v: number) => String(Math.round(v))} />
          <Tooltip cursor={{ fill: "var(--tint-1)" }} content={<PositionTooltip series={series} cumulative={cumulative} />} />
          {bar && <Bar dataKey={bar.key} name={bar.name} fill={bar.color} fillOpacity={0.6} isAnimationActive={false} />}
          {series.filter((s) => s.style !== "bar").flatMap((s) => [
            <Line key={s.key} dataKey={s.key} name={s.name} type="stepAfter" stroke={s.color} strokeWidth={1.75} strokeDasharray={s.style === "dashed" ? "5 3" : undefined} dot={false} activeDot={false} isAnimationActive={false} connectNulls={false} />,
            ...(withMc ? [<Line key={`${s.key}__mc`} dataKey={`${s.key}__mc`} legendType="none" stroke="none" dot={{ r: 4, fill: s.color, stroke: s.color }} activeDot={false} isAnimationActive={false} />] : []),
          ])}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function PositionTooltip({ active, payload, label, series, cumulative }: { active?: boolean; payload?: Array<{ payload: Record<string, number | string> }>; label?: string; series: PositionChartSeries[]; cumulative: boolean }) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  const mc = label === MISSED_CUT_LABEL;
  return (
    <div className="ex-tooltip">
      <strong>{mc ? "Missed the cut" : label === "1" ? "Win (finish 1)" : `Finish ${label}${cumulative ? " or better" : ""}`}</strong>
      {series.map((s) => {
        const v = row[mc && s.style !== "bar" ? `${s.key}__mc` : s.key];
        return <span key={s.key}><i style={{ background: s.color }} aria-hidden="true" /> {s.name}: {typeof v === "number" ? `${v.toFixed(v < 1 ? 2 : 1)}%` : "n/a"}</span>;
      })}
    </div>
  );
}

export function FinishChart({ series, cutTopN, mode }: { series: FinishSeries[]; fieldSize?: number; cutTopN: number | null; mode: "probability" | "cumulative" }) {
  const cumulative = mode === "cumulative";
  const hasCut = !!cutTopN || series.some((s) => (s.missCut ?? 0) > 0);
  const ps: PositionChartSeries[] = series.map((s, j) => ({ key: `p${j}`, name: s.name, color: s.color, pos: s.pos, missCut: s.missCut, style: j === 0 ? "bar" : "line" }));
  return (
    <div>
      <PositionChart series={ps} hasCut={hasCut} cumulative={cumulative} ariaLabel={`Chance of finishing in each position ${cumulative ? "or better " : ""}for ${series.map((s) => s.name).join(", ")}`} />
      <p className="ex-muted">{cumulative ? "Each bar is the chance of finishing at that position or better (ties shared)." : "One bar per finish position (ties shared)"}{hasCut ? "; bars cover made-cut finishes only, and the chance of missing the cut is listed under the chart" : ""}{!cumulative ? "; hover for the exact figure." : ""}{series.length > 1 ? " The first player is the bars; the others are outlines on top." : ""}</p>
      <ul className="ex-legend">
        {series.map((s) => (
          <li key={s.id}><i style={{ background: s.color }} aria-hidden="true" />{s.name}{s.missCut !== null && <small> · misses the cut {pct(s.missCut, 0)}</small>}</li>
        ))}
      </ul>
    </div>
  );
}

/* ------------------------------------------------------------------ hole strip */
const PAR_COLOR: Record<number, string> = { 3: "var(--chart-2)", 4: "var(--chart-1)", 5: "var(--chart-3)" };
export function HoleStrip({ holes }: { holes: HoleRow[] }) {
  const data = holes.filter((h) => h.exp !== null).map((h) => ({ hole: h.hole, par: h.par, exp: h.exp as number, yd: h.yd, bird: h.bird, bog: h.bog }));
  if (!data.length) return null;
  return (
    <figure className="ex-holes">
      <div className="ex-chart short" role="img" aria-label="Expected score versus par for each hole of the field-average player">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 4, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="var(--line)" vertical={false} />
            <XAxis dataKey="hole" stroke="var(--muted)" tick={{ fontSize: 10 }} interval={0} />
            <YAxis stroke="var(--muted)" width={34} tick={{ fontSize: 10 }} tickFormatter={(v: number) => (v > 0 ? `+${v}` : v < 0 ? `−${Math.abs(v)}` : "0")} />
            <Tooltip content={<HoleTooltip />} />
            <Bar dataKey="exp" isAnimationActive={false} radius={[2, 2, 2, 2]}>
              {data.map((d) => <Cell key={d.hole} fill={PAR_COLOR[d.par] ?? "var(--chart-4)"} fillOpacity={0.8} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <figcaption>
        <ul className="ex-legend"><li><i style={{ background: PAR_COLOR[3] }} />par 3</li><li><i style={{ background: PAR_COLOR[4] }} />par 4</li><li><i style={{ background: PAR_COLOR[5] }} />par 5</li></ul>
        <p className="ex-muted">Strokes versus par on each hole, left to right by hole number. Below zero is an easier hole.</p>
      </figcaption>
    </figure>
  );
}
function HoleTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: { hole: number; par: number; exp: number; yd: number | null; bird: number | null; bog: number | null } }> }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return <div className="ex-tooltip"><strong>Hole {p.hole} · par {p.par}{p.yd ? ` · ${p.yd} yd` : ""}</strong><span>Average player: {signed(p.exp)} vs par</span><span>Birdie or better {pct(p.bird, 0)}, bogey or worse {pct(p.bog, 0)}</span></div>;
}

