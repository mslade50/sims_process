"use client";

/**
 * Charts for the explain views (This week, Why priced, player drawer). Colors only from design tokens (var(--...)); every chart has a text alternative
 * (aria-label / visible numbers) and never relies on color alone (sign glyphs, labels). Pure data shaping lives in explain-rules.ts.
 */
import { Bar, BarChart, CartesianGrid, Cell, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { MISSED_CUT_LABEL, positionRows, positionTicks } from "./distributions-rules";
import {
  edgeCellText, pct, signed, type BiasRow, type ExPlayer, type HoleRow, type LegendEntry, type Market, type Pctl, type Shape, type TotalScore, type WaterfallResult,
} from "./explain-rules";

export const PLAYER_COLORS = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)", "var(--chart-5)"];

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

/* ------------------------------------------------------------------ score distributions */
type Range = { lo: number; hi: number; mid: number; mean: number };
const asRange = (s: Pctl | TotalScore | undefined, total: boolean): Range | null => {
  if (!s) return null;
  if (total) {
    const t = s as TotalScore;
    return { lo: t.p5, hi: t.p95, mid: t.p50, mean: t.mean };
  }
  const r = s as Pctl;
  return { lo: r.p10, hi: r.p90, mid: r.p50, mean: r.mean };
};

/** Horizontal range bars: one block per round (p10-p90, dot = median, tick = mean) and one for the 72-hole total (p5-p95, shaded p25-p75). */
export function ScoreFan({ items, rounds }: { items: Array<{ id: number; name: string; color: string; scores: ExPlayer["scores"] }>; rounds: number }) {
  const keys = [...Array.from({ length: rounds }, (_, i) => `r${i + 1}`), "total"].filter((k) => items.some((it) => it.scores[k]));
  if (!keys.length) return <p className="ex-muted">No score distribution for this run.</p>;
  const all: number[] = [];
  keys.forEach((k) => items.forEach((it) => { const r = asRange(it.scores[k], k === "total"); if (r) all.push(r.lo, r.hi); }));
  return (
    <div className="ex-fan" role="group" aria-label="Score distribution versus par">
      {keys.map((k) => {
        const total = k === "total";
        const ks = items.map((it) => asRange(it.scores[k], total)).filter(Boolean) as Range[];
        const lo = Math.floor(Math.min(...ks.map((r) => r.lo)) - 1);
        const hi = Math.ceil(Math.max(...ks.map((r) => r.hi)) + 1);
        const x = (v: number) => ((v - lo) / (hi - lo || 1)) * 100;
        return (
          <div className="ex-fan-block" key={k}>
            <h4>{total ? "72-hole total vs par (if the player makes the cut)" : `Round ${k.slice(1)} vs par`}<small>{total ? " · middle 90% of outcomes, box = middle half" : " · middle 80% of outcomes"}</small></h4>
            {items.map((it) => {
              const r = asRange(it.scores[k], total);
              if (!r) return null;
              const t = it.scores[k] as TotalScore;
              return (
                <div className="ex-fan-row" key={it.id}>
                  <span className="ex-fan-name">{it.name.split(",")[0]}</span>
                  <svg viewBox="0 0 100 10" preserveAspectRatio="none" aria-hidden="true">
                    <line x1={x(r.lo)} x2={x(r.hi)} y1="5" y2="5" stroke={it.color} strokeWidth="1.2" vectorEffect="non-scaling-stroke" />
                    {total && t.p25 !== undefined && <rect x={x(t.p25)} width={Math.max(0.5, x(t.p75) - x(t.p25))} y="2" height="6" fill={it.color} fillOpacity="0.35" />}
                    <line x1={x(r.mean)} x2={x(r.mean)} y1="1.5" y2="8.5" stroke={it.color} strokeWidth="2" vectorEffect="non-scaling-stroke" />
                  </svg>
                  <span className="ex-fan-val">{signed(r.mean, 1)} <small>({signed(r.lo, 0)} to {signed(r.hi, 0)})</small></span>
                </div>
              );
            })}
            <div className="ex-fan-axis"><span>{lo > 0 ? `+${lo}` : lo}</span><span>par</span><span>{hi > 0 ? `+${hi}` : hi}</span></div>
          </div>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ waterfall ("why this price") */
const GROUP_TAG: Record<string, string> = { course: "Course", location: "Location", override: "Manual", live: "In play", weather: "Weather" };
export function Waterfall({ rows, small, smallCount, skillTotal, weatherTotal, smallSkill, smallSkillCount, unexplained, scale, label = "Strokes per round vs the field average" }: WaterfallResult & { label?: string; scale: number }) {
  const skillRows = rows.filter((r) => r.group !== "weather");
  const weatherRows = rows.filter((r) => r.group === "weather");
  const weatherSmall = small - smallSkill;
  const skillNet = skillTotal + unexplained;
  const max = Math.max(0.5, scale);
  const bar = (v: number) => (
    <span className="ex-wf-track" aria-hidden="true">
      <span className={`ex-wf-bar ${v >= 0 ? "pos" : "neg"}`} style={{ width: `${Math.min(1, Math.abs(v) / max) * 50}%`, [v >= 0 ? "left" : "right"]: "50%" }} />
      {Math.abs(v) > max && <span className={`ex-wf-clip ${v >= 0 ? "pos" : "neg"}`}>{v >= 0 ? "›" : "‹"}</span>}
    </span>
  );
  const row = (r: (typeof rows)[number], i: number, list: typeof rows) => (
    <div className="ex-wf-row" role="row" key={r.key} title={r.meaning}>
      <span className="ex-wf-label" role="cell">{r.label}{(i === 0 || list[i - 1].group !== r.group) && r.group !== "skill" && r.group !== "weather" && <em className={`ex-tag g-${r.group}`}>{GROUP_TAG[r.group] ?? r.group}</em>}</span>
      {bar(r.value)}
      <b className={r.value >= 0 ? "pos" : "neg"} role="cell">{signed(r.value)}</b>
    </div>
  );
  return (
    <div className="ex-wf" role="table" aria-label={label}>
      <div className="ex-wf-scale" role="row"><span role="cell" /><span className="ex-wf-axis" role="cell"><em>−{max}</em><em>0</em><em>+{max}</em></span><span role="cell" /></div>
      <p className="ex-muted ex-wf-note">Bar ends = ±{max} strokes a round, the same scale for every player in this event; a longer bar is cut off and marked with an arrow.</p>
      {skillRows.map(row)}
      {smallSkillCount > 0 && <div className="ex-wf-row ex-wf-small" role="row"><span className="ex-wf-label" role="cell">{smallSkillCount} smaller items</span>{bar(smallSkill)}<b role="cell">{signed(smallSkill)}</b></div>}
      {Math.abs(unexplained) >= 0.01 && <div className="ex-wf-row ex-wf-small" role="row" title="Rounding and items too small to list."><span className="ex-wf-label" role="cell">Rounding and other</span>{bar(unexplained)}<b role="cell">{signed(unexplained)}</b></div>}
      <div className="ex-wf-row ex-wf-total" role="row"><span className="ex-wf-label" role="cell">Expected skill</span>{bar(skillNet)}<b className={skillNet >= 0 ? "pos" : "neg"} role="cell">{signed(skillNet)}</b></div>
      {(weatherRows.length > 0 || Math.abs(weatherSmall) >= 0.005) && (
        <>
          <p className="ex-muted ex-wf-note">Weather adjusts scores on top of expected skill:</p>
          {weatherRows.map(row)}
          {smallCount - smallSkillCount > 0 && <div className="ex-wf-row ex-wf-small" role="row"><span className="ex-wf-label" role="cell">{smallCount - smallSkillCount} smaller weather items</span>{bar(weatherSmall)}<b role="cell">{signed(weatherSmall)}</b></div>}
          <div className="ex-wf-row ex-wf-total" role="row"><span className="ex-wf-label" role="cell">Weather total</span>{bar(weatherTotal)}<b className={weatherTotal >= 0 ? "pos" : "neg"} role="cell">{signed(weatherTotal)}</b></div>
        </>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ bias chart (player types we favour / fade) */
export function BiasChart({ rows, onPick, active }: { rows: BiasRow[]; onPick?: (r: BiasRow) => void; active?: string | null }) {
  if (!rows.length) return <p className="ex-muted">No player type stands out from the rest of the field in this run.</p>;
  const max = Math.max(0.1, ...rows.map((r) => Math.abs(r.edge_sg ?? 0)));
  return (
    <ul className="ex-bias" aria-label="Strokes per round we rate each player type above or below the market, compared with the rest of the field">
      {rows.map((r) => {
        const v = r.edge_sg ?? 0;
        const key = `${r.dimension}:${r.bucket}`;
        const d = r.drivers[0];
        return (
          <li key={key} className={active === key ? "active" : ""}>
            <button type="button" onClick={() => onPick?.(r)} aria-pressed={active === key}>
              <span className="ex-bias-label">{r.label}<small> · {r.n} players{d ? ` · biggest reason: ${d.label.toLowerCase()}` : ""}</small></span>
              <span className="ex-wf-track" aria-hidden="true"><span className={`ex-wf-bar ${v >= 0 ? "pos" : "neg"}`} style={{ width: `${(Math.abs(v) / max) * 50}%`, [v >= 0 ? "left" : "right"]: "50%" }} /></span>
              <b className={v >= 0 ? "pos" : "neg"}>{v > 0 ? "▲" : v < 0 ? "▼" : "•"} {Math.abs(v).toFixed(2)}</b>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

/* ------------------------------------------------------------------ hole strip */
const PAR_COLOR: Record<number, string> = { 3: "var(--chart-2)", 4: "var(--chart-1)", 5: "var(--chart-3)" };
export function HoleStrip({ holes }: { holes: HoleRow[] }) {
  const data = holes.filter((h) => h.exp !== null).map((h) => ({ hole: h.hole, par: h.par, exp: h.exp as number, yd: h.yd, bird: h.bird, bog: h.bog }));
  if (!data.length) return null;
  return (
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
      <ul className="ex-legend ex-legend-inline"><li><i style={{ background: PAR_COLOR[3] }} />par 3</li><li><i style={{ background: PAR_COLOR[4] }} />par 4</li><li><i style={{ background: PAR_COLOR[5] }} />par 5</li><li className="ex-muted">Hole number along the bottom; bars show strokes vs par (below zero = an easier hole)</li></ul>
    </div>
  );
}
function HoleTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: { hole: number; par: number; exp: number; yd: number | null; bird: number | null; bog: number | null } }> }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return <div className="ex-tooltip"><strong>Hole {p.hole} · par {p.par}{p.yd ? ` · ${p.yd} yd` : ""}</strong><span>Average player: {signed(p.exp)} vs par</span><span>Birdie or better {pct(p.bird, 0)}, bogey or worse {pct(p.bog, 0)}</span></div>;
}

/* ------------------------------------------------------------------ model vs market table for one player */
const MARKET_NAME: Record<string, string> = { win: "Win", top_5: "Top 5", top_10: "Top 10", top_20: "Top 20", make_cut: "Make cut" };
export function MarketRows({ player, hasCut = true }: { player: ExPlayer; hasCut?: boolean }) {
  const rows = (["win", "top_5", "top_10", "top_20", "make_cut"] as const).filter((m) => hasCut || m !== "make_cut").map((m) => ({ m, e: player.probs[m] }));
  const hasMarket = rows.some(({ e }) => e.market !== null);
  return (
    <>
      <div className="table-scroll"><table className="ex-table">
        <thead><tr><th>Bet type</th><th title="Our chance, from the simulations.">Our chance</th>{hasMarket && <th title="The chance implied by sportsbook odds, with the bookmaker margin removed.">Sportsbook</th>}{hasMarket && <th title="How far our chance is from the sportsbook's, as a share of the sportsbook's chance.">Gap</th>}</tr></thead>
        <tbody>
          {rows.map(({ m, e }) => (
            <tr key={m}>
              <th scope="row">{MARKET_NAME[m]}</th>
              <td>{pct(e.model)}</td>
              {hasMarket && <td title={e.n_books ? `Average of ${e.n_books} sportsbooks` : undefined}>{pct(e.market)}</td>}
              {hasMarket && <td className={e.rel === null ? "" : e.rel > 0 ? "pos" : "neg"}>{e.rel === null ? "-" : Math.round(Math.abs(e.rel * 100)) === 0 ? "0%" : `${e.rel > 0 ? "▲" : "▼"} ${Math.abs(e.rel * 100).toFixed(0)}%`}</td>}
            </tr>
          ))}
        </tbody>
      </table></div>
      <details className="ex-glossary-wrap"><summary>Technical details</summary>
        <div className="table-scroll"><table className="ex-table">
          <thead><tr><th>Bet type</th><th title="The fair price we publish for this bet type, which can differ from our chance above.">Published fair chance</th><th>Sportsbooks averaged</th></tr></thead>
          <tbody>{rows.map(({ m, e }) => <tr key={m}><th scope="row">{MARKET_NAME[m]}</th><td>{pct(e.fair)}</td><td>{e.n_books ?? "-"}</td></tr>)}</tbody>
        </table></div>
      </details>
    </>
  );
}

export function LegendGroups({ legend }: { legend: Record<string, LegendEntry> }) {
  return (
    <dl className="ex-glossary">
      {Object.entries(legend).map(([k, v]) => <div key={k}><dt>{v.label}</dt><dd>{v.meaning}</dd></div>)}
    </dl>
  );
}

/* ------------------------------------------------------------------ shape: our chance vs the sportsbook's, bet by bet, for one player */
const SHAPE_SHORT: Record<Market, string> = { win: "W", top_5: "5", top_10: "10", top_20: "20", make_cut: "MC" };
const SHAPE_NAME: Record<Market, string> = { win: "win", top_5: "top 5", top_10: "top 10", top_20: "top 20", make_cut: "make cut" };
/** Five small bars, one per bet type: up = our chance is higher than the sportsbook's (no-margin) chance, down = lower; grey = within 5% (in line). Capped at 50%. */
export function ShapeBars({ shape, markets, cap = 0.5 }: { shape: Shape; markets: readonly Market[]; cap?: number }) {
  const ms = markets.filter((m) => shape.markets[m]);
  const w = 22, h = 34, mid = 15;
  const label = ms.map((m) => `${SHAPE_NAME[m]} ${edgeCellText(shape.markets[m])}`).join(", ");
  return (
    <svg className="ex-shape" viewBox={`0 0 ${ms.length * w} ${h + 10}`} width={ms.length * w} height={h + 10} role="img" aria-label={`Our chance against the sportsbook's: ${label}`}>
      <line x1="0" x2={ms.length * w} y1={mid} y2={mid} stroke="var(--line-strong)" strokeWidth="1" />
      {ms.map((m, i) => {
        const e = shape.markets[m]!;
        const v = Math.max(-cap, Math.min(cap, e.rel));
        const len = Math.max(1.5, (Math.abs(v) / cap) * (mid - 1));
        const cls = e.tone === "pos" ? "pos" : e.tone === "neg" ? "neg" : "flat";
        return (
          <g key={m}>
            <title>{`${SHAPE_NAME[m]}: ours ${pct(e.model)} vs sportsbook ${pct(e.market)} (${edgeCellText(e)})`}</title>
            <rect className={`ex-shape-bar ${cls}`} x={i * w + 5} width={w - 10} y={v >= 0 ? mid - len : mid} height={len} rx="2" />
            <text x={i * w + w / 2} y={h + 8} textAnchor="middle" className="ex-shape-lab">{SHAPE_SHORT[m]}</text>
          </g>
        );
      })}
    </svg>
  );
}
