"use client";

/**
 * Chart-free pieces of the explain views (no Recharts, so This week's summary renders without it): score fan, waterfall, bias bars, per-player market rows, shape bars,
 * the defined-term popover and the small inline bars. Colors only from design tokens; never color alone. Recharts charts live in ExplainCharts.tsx.
 */
import { useEffect, useRef, useState } from "react";
import { define } from "./glossary";
import {
  edgeCellText, niceScale, pct, signed, type BiasRow, type ExPlayer, type LegendEntry, type Market, type Pctl, type Shape, type TotalScore, type WaterfallResult, type WaterfallScales,
} from "./explain-rules";

export const PLAYER_COLORS = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)", "var(--chart-5)"];

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
export function Waterfall({ rows, small, smallCount, skillTotal, weatherTotal, smallSkill, smallSkillCount, unexplained, scale, scales, label = "Strokes per round vs the field average" }: WaterfallResult & { label?: string; scale: number; scales?: WaterfallScales }) {
  const skillRows = rows.filter((r) => r.group !== "weather");
  const weatherRows = rows.filter((r) => r.group === "weather");
  const weatherSmall = small - smallSkill;
  const skillNet = skillTotal + unexplained;
  const end = (m: number) => (m < 1 ? m.toFixed(Math.round(m * 1000) % 10 !== 0 ? 3 : 2) : String(m));
  // Every row has its own scale (the printed bar end), so small items stay readable; expected skill uses the event-wide scale.
  const bar = (v: number, m: number) => {
    const max = Math.max(0.01, m);
    return (
      <>
        <span className="ex-wf-track" aria-hidden="true">
          <span className={`ex-wf-bar ${v >= 0 ? "pos" : "neg"}`} style={{ width: `${Math.min(1, Math.abs(v) / max) * 50}%`, [v >= 0 ? "left" : "right"]: "50%" }} />
          {Math.abs(v) > max && <span className={`ex-wf-clip ${v >= 0 ? "pos" : "neg"}`}>{v >= 0 ? "›" : "‹"}</span>}
        </span>
        <small className="ex-wf-end" title={`The ends of this bar are −${end(max)} and +${end(max)} strokes a round.`}>±{end(max)}</small>
      </>
    );
  };
  const rs = (k: string, fallback: number) => scales?.rows[k] ?? fallback;
  const row = (r: (typeof rows)[number], i: number, list: typeof rows) => (
    <div className="ex-wf-row" role="row" key={r.key} title={r.meaning}>
      <span className="ex-wf-label" role="cell">{r.label}{r.unavailable && <em className="ex-tag ex-unavail" title={r.unavailable === "some players" ? "Course fit or history is missing for some players this week." : `No ${r.unavailable} data for this event, so the model cannot use it.`}>{r.unavailable === "some players" ? "missing for some" : "no data this week"}</em>}{(i === 0 || list[i - 1].group !== r.group) && r.group !== "skill" && r.group !== "weather" && <em className={`ex-tag g-${r.group}`}>{GROUP_TAG[r.group] ?? r.group}</em>}</span>
      {bar(r.value, rs(r.key, niceScale(Math.abs(r.value))))}
      <b className={r.value >= 0 ? "pos" : "neg"} role="cell">{signed(r.value)}</b>
    </div>
  );
  return (
    <div className="ex-wf" role="table" aria-label={label}>
      <p className="ex-muted ex-wf-note">Every row has its own scale, printed at the end of its bar (±0.10 means the bar ends are −0.10 and +0.10 strokes). It is set to about the biggest value that item takes among players in this event, so compare a player with the field on the same row, not one row with another. Longer bars are cut off and marked with an arrow. Expected skill uses a wider scale.</p>
      {skillRows.map(row)}
      {smallSkillCount > 0 && <div className="ex-wf-row ex-wf-small" role="row"><span className="ex-wf-label" role="cell">{smallSkillCount} smaller items</span>{bar(smallSkill, scales?.smallSkill ?? niceScale(Math.abs(smallSkill)))}<b role="cell">{signed(smallSkill)}</b></div>}
      {Math.abs(unexplained) >= 0.01 && <div className="ex-wf-row ex-wf-small" role="row" title="Rounding and items too small to list."><span className="ex-wf-label" role="cell">Rounding and other</span>{bar(unexplained, scales?.unexplained ?? niceScale(Math.abs(unexplained)))}<b role="cell">{signed(unexplained)}</b></div>}
      <div className="ex-wf-row ex-wf-total" role="row"><span className="ex-wf-label" role="cell">Expected skill</span>{bar(skillNet, Math.max(0.5, scale))}<b className={skillNet >= 0 ? "pos" : "neg"} role="cell">{signed(skillNet)}</b></div>
      {(weatherRows.length > 0 || Math.abs(weatherSmall) >= 0.005) && (
        <>
          <p className="ex-muted ex-wf-note">Weather adjusts scores on top of expected skill:</p>
          {weatherRows.map(row)}
          {smallCount - smallSkillCount > 0 && <div className="ex-wf-row ex-wf-small" role="row"><span className="ex-wf-label" role="cell">{smallCount - smallSkillCount} smaller weather items</span>{bar(weatherSmall, scales?.weatherSmall ?? niceScale(Math.abs(weatherSmall)))}<b role="cell">{signed(weatherSmall)}</b></div>}
          <div className="ex-wf-row ex-wf-total" role="row"><span className="ex-wf-label" role="cell">Weather total</span>{bar(weatherTotal, scales?.weatherTotal ?? niceScale(Math.abs(weatherTotal)))}<b className={weatherTotal >= 0 ? "pos" : "neg"} role="cell">{signed(weatherTotal)}</b></div>
        </>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ bias chart (player types we favor / fade) */
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
/** Bar scale for a set of shapes: the 75th percentile of the gaps shown, between 6% and 50%. Shared by a table so rows compare. */
export function shapeCap(shapes: readonly Shape[], markets: readonly Market[]): number {
  const v = shapes.flatMap((s) => markets.map((m) => s.markets[m]?.rel)).filter((x): x is number => typeof x === "number" && Number.isFinite(x)).map(Math.abs).sort((a, b) => a - b);
  if (!v.length) return 0.5;
  return Math.min(0.5, Math.max(0.06, v[Math.min(v.length - 1, Math.floor(v.length * 0.75))]));
}
/** Five small bars, one per bet type: up = our chance is higher than the sportsbook's (no-margin) chance, down = lower; gray = within 5% (in line).
 *  `cap` is the gap that fills a bar (larger gaps are clipped and drawn with a notch); pass shapeCap(...) so the bars use the room. Square-root scale:
 *  small gaps stay visible next to big ones. */
export function ShapeBars({ shape, markets, cap }: { shape: Shape; markets: readonly Market[]; cap?: number }) {
  const ms = markets.filter((m) => shape.markets[m]);
  const w = 22, h = 44, mid = 20;
  const c = cap ?? shapeCap([shape], ms);
  const label = ms.map((m) => `${SHAPE_NAME[m]} ${edgeCellText(shape.markets[m])}`).join(", ");
  return (
    <svg className="ex-shape" viewBox={`0 0 ${ms.length * w} ${h + 10}`} width={ms.length * w} height={h + 10} role="img" aria-label={`Our chance against the sportsbook's: ${label}`}>
      <line x1="0" x2={ms.length * w} y1={mid} y2={mid} stroke="var(--line-strong)" strokeWidth="1" />
      {ms.map((m, i) => {
        const e = shape.markets[m]!;
        const v = Math.max(-c, Math.min(c, e.rel));
        const len = Math.max(e.tone === "flat" ? 1.5 : 3, Math.sqrt(Math.abs(v) / c) * (mid - 1));
        const clipped = Math.abs(e.rel) > c;
        const cls = e.tone === "pos" ? "pos" : e.tone === "neg" ? "neg" : "flat";
        return (
          <g key={m}>
            <title>{`${SHAPE_NAME[m]}: ours ${pct(e.model)} vs sportsbook ${pct(e.market)} (${edgeCellText(e)})`}</title>
            <rect className={`ex-shape-bar ${cls}`} x={i * w + 5} width={w - 10} y={v >= 0 ? mid - len : mid} height={len} rx="2" />
            {clipped && <line className="ex-shape-clip" x1={i * w + 4} x2={i * w + w - 4} y1={v >= 0 ? mid - len + 2.5 : mid + len - 2.5} y2={v >= 0 ? mid - len + 2.5 : mid + len - 2.5} />}
            <text x={i * w + w / 2} y={h + 8} textAnchor="middle" className="ex-shape-lab">{SHAPE_SHORT[m]}</text>
          </g>
        );
      })}
    </svg>
  );
}

/* ------------------------------------------------------------------ This week: small inline visuals */
/** A defined term: tap or hover for a one-sentence definition from glossary.ts (title text alone does not work on a phone). Without a definition it is plain text. */
export function Term({ k, children, text }: { k: string; children: React.ReactNode; text?: string }) {
  const def = text ?? define(k);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!open) return;
    const away = (e: Event) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("pointerdown", away);
    document.addEventListener("keydown", esc);
    return () => { document.removeEventListener("pointerdown", away); document.removeEventListener("keydown", esc); };
  }, [open]);
  if (!def) return <>{children}</>;
  return (
    <span className="ex-term" ref={ref}>
      <button type="button" className="ex-term-btn" aria-expanded={open} title={def} onClick={() => setOpen((o) => !o)}>{children}</button>
      {open && <span className="ex-term-pop" role="note">{def}</span>}
    </span>
  );
}

/** A bar growing from a centre line: right (green) for a positive value, left (red) for a negative one. `scale` is the value that fills one side. The number beside it carries the sign. */
export function DeltaBar({ value, scale, label }: { value: number; scale: number; label?: string }) {
  const w = Math.min(1, Math.abs(value) / (scale || 1)) * 50;
  const pos = value > 0;
  return (
    <span className="ex-dbar" role="img" aria-label={label ?? "change"}>
      <i className={value === 0 ? "flat" : pos ? "pos" : "neg"} style={{ width: `${w}%`, [pos ? "left" : "right"]: "50%" }} />
    </span>
  );
}

/** Two thin bars on one scale, ours over the sportsbook's, for a pair of chances. */
export function ProbPair({ model, market, scale }: { model: number | null; market: number | null; scale?: number }) {
  const top = scale ?? Math.max(model ?? 0, market ?? 0, 0.0001);
  const w = (v: number | null) => `${Math.min(100, ((v ?? 0) / top) * 100)}%`;
  return (
    <span className="ex-pair" aria-hidden="true">
      <i className="m" style={{ width: w(model) }} />
      {market !== null && <i className="k" style={{ width: w(market) }} />}
    </span>
  );
}

/** The winning score range: a track from the 10th to the 90th percentile with a marker at the most likely score and a tick at the median. Scores are to par; lower is to the left. */
export function RangeBar({ p10, p90, mode, median }: { p10: number; p90: number; mode: number; median: number }) {
  const lo = Math.floor(Math.min(p10, mode, median)) - 1;
  const hi = Math.ceil(Math.max(p90, mode, median)) + 1;
  const x = (v: number) => `${((v - lo) / (hi - lo || 1)) * 100}%`;
  return (
    <span className="ex-range" role="img" aria-label={`Winning score between ${p10} and ${p90} in 8 of 10 simulations, most likely ${mode}`}>
      <i className="band" style={{ left: x(p10), width: `calc(${x(p90)} - ${x(p10)})` }} />
      <i className="med" style={{ left: x(median) }} />
      <i className="mode" style={{ left: x(mode) }} />
    </span>
  );
}
