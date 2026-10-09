"use client";

/**
 * Charts for the explain views (This week, Why priced, player drawer). Colors only from design tokens (var(--...)); every chart has a text alternative
 * (aria-label / visible numbers) and never relies on color alone (sign glyphs, labels). Pure data shaping lives in explain-rules.ts.
 */
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { bucketProbs, finishBuckets } from "./distributions-rules";
import {
  pct, signed, type BiasRow, type ExPlayer, type HoleRow, type LegendEntry, type Pctl, type TotalScore, type WaterfallRow,
} from "./explain-rules";

export const PLAYER_COLORS = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)", "var(--chart-5)"];

/* ------------------------------------------------------------------ finish shape */
export type FinishSeries = { id: number; name: string; pos: number[]; missCut: number | null; color: string };

export function FinishChart({ series, fieldSize, cutTopN, mode }: { series: FinishSeries[]; fieldSize: number; cutTopN: number | null; mode: "probability" | "cumulative" }) {
  const cumulative = mode === "cumulative";
  const hasCut = !!cutTopN || series.some((s) => (s.missCut ?? 0) > 0);
  const all = finishBuckets(fieldSize, hasCut);
  const buckets = all.filter((b) => !(cumulative && b.missed));
  const probs = series.map((s) => bucketProbs(s.pos, s.missCut, all).filter((_, i) => !(cumulative && all[i].missed)));
  const rows = buckets.map((b, i) => {
    const row: Record<string, number | string> = { label: b.label };
    probs.forEach((pr, j) => { row[`p${j}`] = +(100 * (cumulative ? pr.slice(0, i + 1).reduce((a, v) => a + v, 0) : pr[i])).toFixed(2); });
    return row;
  });
  return (
    <div>
      <div className="ex-chart" role="img" aria-label={`Chance of finishing ${cumulative ? "in each range or better" : "in each range"} for ${series.map((s) => s.name).join(", ")}`}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} barCategoryGap="14%" barGap={2} margin={{ top: 10, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="var(--line)" vertical={false} />
            <XAxis dataKey="label" stroke="var(--muted)" interval={0} tick={{ fontSize: 11 }} />
            <YAxis stroke="var(--muted)" width={38} unit="%" tick={{ fontSize: 11 }} domain={cumulative ? [0, 100] : [0, "auto"]} tickFormatter={(v: number) => String(Math.round(v))} />
            <Tooltip cursor={{ fill: "var(--tint-1)" }} content={<FinishTooltip series={series} cumulative={cumulative} />} />
            {series.map((s, j) => <Bar key={s.id} dataKey={`p${j}`} name={s.name} fill={s.color} fillOpacity={series.length === 1 ? 0.7 : 0.85} isAnimationActive={false} radius={[3, 3, 0, 0]} />)}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="ex-muted">{cumulative ? "Each bar is the chance of finishing in that range of finish positions or better (ties shared)." : "Each bar is the chance of finishing in that range of finish positions (ties shared); hover for the exact figure."}</p>
      <ul className="ex-legend">
        {series.map((s) => (
          <li key={s.id}><i style={{ background: s.color }} aria-hidden="true" />{s.name}{s.missCut !== null && <small> · misses the cut {pct(s.missCut, 0)}</small>}</li>
        ))}
      </ul>
    </div>
  );
}

function FinishTooltip({ active, payload, label, series, cumulative }: { active?: boolean; payload?: Array<{ payload: Record<string, number> }>; label?: string; series: FinishSeries[]; cumulative: boolean }) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  const range = label === "Win" ? "Win" : label === "Missed cut" ? "Missed cut" : `Finish ${label}`;
  return (
    <div className="ex-tooltip">
      <strong>{range}{cumulative && label !== "Win" ? " or better" : ""}</strong>
      {series.map((s, j) => {
        const v = row[`p${j}`];
        return <span key={s.id}><i style={{ background: s.color }} aria-hidden="true" /> {s.name}: {v === undefined ? "n/a" : `${v.toFixed(1)}%`}</span>;
      })}
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
export function Waterfall({ rows, total, small, smallCount, label = "Strokes per round vs the field average" }: { rows: WaterfallRow[]; total: number; small: number; smallCount: number; label?: string }) {
  const max = Math.max(0.05, ...rows.map((r) => Math.abs(r.value)), Math.abs(small), Math.abs(total) * 0.6);
  const bar = (v: number) => (
    <span className="ex-wf-track" aria-hidden="true">
      <span className={`ex-wf-bar ${v >= 0 ? "pos" : "neg"}`} style={{ width: `${(Math.abs(v) / max) * 50}%`, [v >= 0 ? "left" : "right"]: "50%" }} />
    </span>
  );
  return (
    <div className="ex-wf" role="table" aria-label={label}>
      {rows.map((r, i) => (
        <div className="ex-wf-row" role="row" key={r.key} title={r.meaning}>
          <span className="ex-wf-label" role="cell">{r.label}{(i === 0 || rows[i - 1].group !== r.group) && r.group !== "skill" && <em className={`ex-tag g-${r.group}`}>{r.group}</em>}</span>
          {bar(r.value)}
          <b className={r.value >= 0 ? "pos" : "neg"} role="cell">{signed(r.value)}</b>
        </div>
      ))}
      {smallCount > 0 && <div className="ex-wf-row ex-wf-small" role="row"><span className="ex-wf-label" role="cell">{smallCount} smaller items</span>{bar(small)}<b role="cell">{signed(small)}</b></div>}
      <div className="ex-wf-row ex-wf-total" role="row"><span className="ex-wf-label" role="cell">Net (all shown)</span>{bar(total)}<b className={total >= 0 ? "pos" : "neg"} role="cell">{signed(total)}</b></div>
    </div>
  );
}

/* ------------------------------------------------------------------ bias chart (player types we favour / fade) */
export function BiasChart({ rows, onPick, active }: { rows: BiasRow[]; onPick?: (r: BiasRow) => void; active?: string | null }) {
  if (!rows.length) return <p className="ex-muted">No player-type comparison without market prices in this run.</p>;
  const max = Math.max(0.1, ...rows.map((r) => Math.abs(r.edge_sg ?? 0)));
  return (
    <ul className="ex-bias" aria-label="Strokes per round: model minus market, by player type">
      {rows.map((r) => {
        const v = r.edge_sg ?? 0;
        const key = `${r.dimension}:${r.bucket}`;
        const d = r.drivers[0];
        return (
          <li key={key} className={active === key ? "active" : ""}>
            <button type="button" onClick={() => onPick?.(r)} aria-pressed={active === key}>
              <span className="ex-bias-label">{r.label}<small> · {r.n} players{d ? ` · ${d.label.toLowerCase()}` : ""}</small></span>
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
          <YAxis stroke="var(--muted)" width={34} tick={{ fontSize: 10 }} tickFormatter={(v: number) => (v > 0 ? `+${v}` : `${v}`)} />
          <Tooltip content={<HoleTooltip />} />
          <Bar dataKey="exp" isAnimationActive={false} radius={[2, 2, 2, 2]}>
            {data.map((d) => <Cell key={d.hole} fill={PAR_COLOR[d.par] ?? "var(--chart-4)"} fillOpacity={0.8} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <ul className="ex-legend ex-legend-inline"><li><i style={{ background: PAR_COLOR[3] }} />par 3</li><li><i style={{ background: PAR_COLOR[4] }} />par 4</li><li><i style={{ background: PAR_COLOR[5] }} />par 5</li><li className="ex-muted">bars: expected score vs par, below zero = scoring hole</li></ul>
    </div>
  );
}
function HoleTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: { hole: number; par: number; exp: number; yd: number | null; bird: number | null; bog: number | null } }> }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return <div className="ex-tooltip"><strong>Hole {p.hole} · par {p.par}{p.yd ? ` · ${p.yd} yd` : ""}</strong><span>Expected {signed(p.exp)} vs par</span><span>Birdie or better {pct(p.bird, 0)}, bogey or worse {pct(p.bog, 0)}</span></div>;
}

/* ------------------------------------------------------------------ model vs market table for one player */
export function MarketRows({ player }: { player: ExPlayer }) {
  const rows = (["win", "top_5", "top_10", "top_20", "make_cut"] as const).map((m) => ({ m, e: player.probs[m] }));
  const label: Record<string, string> = { win: "Win", top_5: "Top 5", top_10: "Top 10", top_20: "Top 20", make_cut: "Make cut" };
  return (
    <div className="table-scroll"><table className="ex-table">
      <thead><tr><th>Market</th><th>Model</th><th>Market</th><th>Edge</th><th>Fair</th></tr></thead>
      <tbody>
        {rows.map(({ m, e }) => (
          <tr key={m}>
            <th scope="row">{label[m]}</th>
            <td>{pct(e.model)}</td>
            <td>{pct(e.market)}{e.n_books ? <small className="ex-muted"> {e.n_books}b</small> : null}</td>
            <td className={e.rel === null ? "" : e.rel > 0 ? "pos" : "neg"}>{e.rel === null ? "-" : `${e.rel > 0 ? "▲" : e.rel < 0 ? "▼" : "•"} ${Math.abs(e.rel * 100).toFixed(0)}%`}</td>
            <td>{pct(e.fair)}</td>
          </tr>
        ))}
      </tbody>
    </table></div>
  );
}

export function LegendGroups({ legend }: { legend: Record<string, LegendEntry> }) {
  return (
    <dl className="ex-glossary">
      {Object.entries(legend).map(([k, v]) => <div key={k}><dt>{v.label}</dt><dd>{v.meaning}</dd></div>)}
    </dl>
  );
}
