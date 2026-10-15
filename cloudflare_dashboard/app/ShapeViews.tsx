"use client";

/**
 * "Same chance to win, different weeks": which top-30 players have an unusual finish shape for their win chance, and why.
 * Chart-free (inline SVG only, no Recharts) so the compact card can sit on This week without adding to its bundle.
 * All numbers come from shape-rules.ts; this file only draws them.
 */
import { useEffect, useMemo, useState } from "react";
import { SegmentedControl } from "./components";
import { Term } from "./ExplainBits";
import { pct, type ExplainDoc } from "./explain-rules";
import {
  analyzeShapes, availableMarkets, curveAt, kindLabel, ladderHeadline, pairLadder, pairsForBetting, playerWhy, shortName, signedPp, SHAPE_MARKET_LABEL,
  type LadderRow, type ShapeAnalysis, type ShapeKind, type ShapeMarket, type ShapePlayer,
} from "./shape-rules";
import { AnimatedNumber, ScrollTable } from "./ui";
import "./shapes.css";

const lastName = (raw: string) => { const n = shortName(raw).split(" "); return n[n.length - 1] ?? ""; };

function Glyph({ kind }: { kind: ShapeKind }) { return <i className="shp-glyph" data-kind={kind} aria-hidden="true" />; }
function KindWord({ kind }: { kind: ShapeKind }) {
  const k = kind === "steady" ? "steady" : kind === "bust" ? "boom_bust" : "typical_shape";
  return <Term k={k}>{kindLabel(kind)}</Term>;
}


/* ------------------------------------------------------------------ bet ladder: the finding in one sentence, then one row per bet */
const pts1 = (v: number | null) => (v === null ? "—" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(1)}`);
const pct1 = (v: number | null) => (v === null ? "—" : `${(v * 100).toFixed(1)}%`);
/** Same win chance, different finishes: pair chips (best betting pair first), the finding, and a ladder whose gap bars grow on arrival and slide when the pair changes. */
export function ShapeLadder({ doc, a, onPick, onSeeMore, title = true }: { doc: ExplainDoc; a: ShapeAnalysis; onPick?: (id: number) => void; onSeeMore?: () => void; title?: boolean }) {
  const pairs = useMemo(() => pairsForBetting(doc, a.pairs, a.market), [doc, a.pairs, a.market]);
  const [ix, setIx] = useState(0);
  const [shown, setShown] = useState(false);
  const pair = pairs[Math.min(ix, pairs.length - 1)];
  const rows = useMemo<LadderRow[]>(() => (pair ? pairLadder(doc, pair) : []), [doc, pair]);
  useEffect(() => { const id = requestAnimationFrame(() => setShown(true)); return () => cancelAnimationFrame(id); }, []);
  if (!pair) return <p className="shp-empty">No two top-30 players have win chances within 20% of each other and a different shape, so there is no pair to show.</p>;
  const h = ladderHeadline(doc, pair, rows, a.market);
  const max = Math.max(1, ...rows.map((r) => Math.abs(r.gapPts ?? 0)));
  const al = lastName(pair.steady.name), bl = lastName(pair.bust.name);
  const hasBook = rows.some((r) => r.a.book !== null || r.b.book !== null);
  const name = (id: number, text: string) => onPick ? <button type="button" className="shl-name" onClick={() => onPick(id)}>{text}</button> : <span>{text}</span>;
  return (
    <div className="shl">
      {title && <div className="shl-head"><h3>Same win chance, different finishes</h3>{onSeeMore && <button type="button" className="shl-more" onClick={onSeeMore}>All 30 shapes →</button>}</div>}
      <p className="shl-lede">{h.lead}<b className="shl-key">{h.gap}</b>{h.tail}</p>
      {(h.why || h.book) && <dl className="shl-notes">
        {h.why && <div><dt>Why</dt><dd>{h.why}</dd></div>}
        {h.book && <div><dt>Books</dt><dd>{h.book}</dd></div>}
      </dl>}
      {pairs.length > 1 && (
        <div className="shl-chips" role="group" aria-label="Pair of players">
          {pairs.map((p, i) => <button key={`${p.steady.id}-${p.bust.id}`} type="button" aria-pressed={i === ix} onClick={() => setIx(i)}>{lastName(p.steady.name)} <span aria-hidden="true">·</span> {lastName(p.bust.name)}</button>)}
        </div>
      )}
      <div className="shl-table" role="table" aria-label={`Chances for ${shortName(pair.steady.name)} and ${shortName(pair.bust.name)} on each bet`} data-shown={shown ? "" : undefined}>
        <div className="shl-row shl-hdr" role="row">
          <span role="columnheader">Bet</span>
          <span role="columnheader" className="shl-n"><i className="shl-dot" data-kind="steady" aria-hidden="true" />{name(pair.steady.id, al)}</span>
          <span role="columnheader" className="shl-n"><i className="shl-dot" data-kind="bust" aria-hidden="true" />{name(pair.bust.id, bl)}</span>
          <span role="columnheader" className="shl-gaph">How much more often {al}</span>
          <span role="columnheader" className="shl-n">Points</span>
        </div>
        {rows.map((r, i) => {
          const g = r.gapPts ?? 0, k = Math.min(1, Math.abs(g) / max);
          return (
            <div key={r.market} role="row" className="shl-row" data-hl={r.market === a.market ? "" : undefined} style={{ ["--i" as string]: i }}>
              <span role="rowheader" className="shl-bet">{r.label}</span>
              <span role="cell" className="shl-n"><AnimatedNumber text={pct1(r.a.ours)} />{hasBook && <small>{r.a.book === null ? "no book" : `book ${pct1(r.a.book)}`}</small>}</span>
              <span role="cell" className="shl-n"><AnimatedNumber text={pct1(r.b.ours)} />{hasBook && <small>{r.b.book === null ? "no book" : `book ${pct1(r.b.book)}`}</small>}</span>
              <span role="cell" className="shl-track" aria-hidden="true"><i data-neg={g < 0 ? "" : undefined} style={{ ["--k" as string]: k }} /></span>
              <span role="cell" className="shl-n shl-pts">{pts1(r.gapPts)}{r.bookGapPts !== null && <small>book {pts1(r.bookGapPts)}</small>}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ scatter */
const SW = 360, SH = 290;
const niceTicks = (lo: number, hi: number): number[] => {
  const span = hi - lo;
  const step = [1, 2, 5, 10, 20].find((s) => span / s <= 6) ?? 20;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v);
  return out;
};
export function ShapeScatter({ a, selectedId, onSelect }: { a: ShapeAnalysis; selectedId: number | null; onSelect: (id: number) => void }) {
  const L = 38, R = 14, T = 12, B = 38;
  const xs = a.players.map((p) => Math.log(p.win));
  const x0 = Math.min(...xs) - 0.1, x1 = Math.max(...xs) + 0.1;
  const ys = a.players.map((p) => p.p * 100);
  const y0 = Math.max(0, Math.floor(Math.min(...ys) - 2)), y1 = Math.min(100, Math.ceil(Math.max(...ys) + 2));
  const sx = (w: number) => L + ((Math.log(w) - x0) / (x1 - x0 || 1)) * (SW - L - R);
  const sy = (v: number) => T + (1 - (v - y0) / (y1 - y0 || 1)) * (SH - T - B);
  const curve = Array.from({ length: 41 }, (_, i) => { const w = Math.exp(x0 + ((x1 - x0) * i) / 40); return `${i ? "L" : "M"}${sx(w).toFixed(1)},${sy(curveAt(a.fit, w) * 100).toFixed(1)}`; }).join(" ");
  const xt = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3].filter((w) => Math.log(w) >= x0 && Math.log(w) <= x1);
  const yt = niceTicks(y0, y1);
  const labelled: Array<{ p: ShapePlayer; x: number; y: number; end: boolean }> = [];
  for (const p of a.unusual.filter((q) => q.kind !== "typical").slice(0, 6)) {
    const x = sx(p.win), y = sy(p.p * 100);
    if (labelled.some((l) => Math.abs(l.x - x) < 54 && Math.abs(l.y - y) < 12)) continue;
    labelled.push({ p, x, y, end: x > SW * 0.62 });
    if (labelled.length >= 4) break;
  }
  const order = [...a.players].sort((p, q) => Number(p.id === selectedId) - Number(q.id === selectedId) || Number(p.kind !== "typical") - Number(q.kind !== "typical"));
  const marketLabel = SHAPE_MARKET_LABEL[a.market];
  return (
    <svg className="shp-scatter" viewBox={`0 0 ${SW} ${SH}`} role="group" aria-label={`${marketLabel} chance against win chance for the top ${a.players.length} players, with the typical shape curve`}>
      {yt.map((v) => <g key={`y${v}`}><line className="grid" x1={L} x2={SW - R} y1={sy(v)} y2={sy(v)} /><text x={L - 5} y={sy(v) + 3} textAnchor="end">{v}%</text></g>)}
      {xt.map((w) => <g key={`x${w}`}><line className="grid" x1={sx(w)} x2={sx(w)} y1={T} y2={SH - B} /><text x={sx(w)} y={SH - B + 13} textAnchor="middle">{w * 100}%</text></g>)}
      <line className="axis" x1={L} x2={SW - R} y1={SH - B} y2={SH - B} />
      <line className="axis" x1={L} x2={L} y1={T} y2={SH - B} />
      <text x={(L + SW - R) / 2} y={SH - 6} textAnchor="middle">Win chance (log scale)</text>
      <text x={10} y={(T + SH - B) / 2} textAnchor="middle" transform={`rotate(-90 10 ${(T + SH - B) / 2})`}>{marketLabel} chance</text>
      <path className="curve" d={curve} />
      {order.map((p) => {
        const x = sx(p.win), y = sy(p.p * 100);
        const sel = p.id === selectedId;
        const act = () => onSelect(p.id);
        return (
          <g key={p.id} className="pt" role="button" tabIndex={0} data-selected={sel ? "" : undefined} aria-label={`${shortName(p.name)}: win ${pct(p.win)}, ${marketLabel} ${pct(p.p)}, ${kindLabel(p.kind)}, ${signedPp(p.residualPp)} against the typical shape`}
            onClick={act} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); act(); } }}>
            <circle cx={x} cy={y} r={13} fill="transparent" />
            {p.kind === "bust" ? <rect className="pt-bust" x={x - 4.6} y={y - 4.6} width={9.2} height={9.2} transform={`rotate(45 ${x} ${y})`} /> : <circle className={p.kind === "steady" ? "pt-steady" : "pt-typical"} cx={x} cy={y} r={p.kind === "steady" ? 5 : 3.6} />}
            <circle className="ring" cx={x} cy={y} r={9} />
          </g>
        );
      })}
      {labelled.map(({ p, x, y, end }) => <text key={`l${p.id}`} className="lbl" x={end ? x - 8 : x + 8} y={y - 6} textAnchor={end ? "end" : "start"}>{lastName(p.name)}</text>)}
    </svg>
  );
}

/* ------------------------------------------------------------------ ranked list */
function SdBar({ sd, a }: { sd: number | null; a: ShapeAnalysis }) {
  const { sdMin, sdMax, sdMedian } = a.variety;
  if (sd === null || sdMin === null || sdMax === null || sdMax - sdMin < 1e-6) return <span className="shp-sdbar" aria-hidden="true" />;
  const at = (v: number) => `${Math.min(100, Math.max(0, ((v - sdMin) / (sdMax - sdMin)) * 100))}%`;
  return <span className="shp-sdbar" role="img" aria-label={`Spread ${sd.toFixed(2)} strokes; top 30 range ${sdMin.toFixed(2)} to ${sdMax.toFixed(2)}`}>{sdMedian !== null && sdMedian >= sdMin && sdMedian <= sdMax && <u style={{ left: at(sdMedian) }} />}<i style={{ left: at(sd) }} /></span>;
}
export function UnusualList({ a, selectedId, onSelect, limit = 8 }: { a: ShapeAnalysis; selectedId: number | null; onSelect: (id: number) => void; limit?: number }) {
  const rows = a.unusual.slice(0, limit);
  return (
    <div>
      <div className="shp-list-head" aria-hidden="true"><span>Player</span><span>vs typical</span><span>Spread</span></div>
      <ul className="shp-list">
        {rows.map((p) => (
          <li key={p.id}>
            <button type="button" className="shp-row" aria-pressed={p.id === selectedId} onClick={() => onSelect(p.id)}>
              <span className="nm"><span><Glyph kind={p.kind} /> {shortName(p.name)}</span><small>{pct(p.win, 1)} win · {pct(p.p, 1)} {SHAPE_MARKET_LABEL[a.market].toLowerCase()} · {kindLabel(p.kind)}</small></span>
              <span className="pp" data-kind={p.kind}>{signedPp(p.residualPp)}</span>
              <SdBar sd={p.sd} a={a} />
            </button>
          </li>
        ))}
      </ul>
      <p className="shp-fine">Percentage points above (steady) or below (boom-or-bust) the typical top-30 shape for that win chance. The spread bar runs from the lowest to the highest spread in the top 30; the thin line is the field median.</p>
    </div>
  );
}

/* ------------------------------------------------------------------ explorer panel body (Distributions page) */
const TIER_WORD = { narrow: "Alike", moderate: "Some variety", wide: "Clear variety", unknown: "" } as const;

export function ShapeExplorer({ doc, onPick }: { doc: ExplainDoc; onPick?: (ids: number[]) => void }) {
  const markets = useMemo(() => availableMarkets(doc), [doc]);
  const [market, setMarket] = useState<ShapeMarket>("top_20");
  const [selected, setSelected] = useState<number | null>(null);
  const a = useMemo(() => analyzeShapes(doc, market), [doc, market]);
  if (!a.ok) return <p className="shp-empty">{a.reason ?? "Shapes are not available for this simulation."}</p>;
  const pair = a.pairs[0];
  const sel = a.players.find((p) => p.id === selected) ?? null;
  return (
    <div className="shp-root">
      <p className="shp-variety"><span className="shp-tier" data-tier={a.variety.tier}>{TIER_WORD[a.variety.tier]}</span>{" "}{a.variety.text}</p>
      <p className="shp-intro">Yes, it is possible, and the simulation does it on purpose. Every player has an expected score and a <Term k="spread">round-to-round spread</Term>. A high-spread player wins more of his chance from rare blow-out weeks and keeps fewer steady top finishes; a low-spread player is the reverse. Two players can therefore land on the same win chance with very different top-20 chances.</p>
      <div className="shp-tabs">
        <SegmentedControl label="Market for shape" value={a.market} onChange={(m) => { setMarket(m); setSelected(null); }} options={markets.map((m) => ({ value: m, label: SHAPE_MARKET_LABEL[m] }))} />
      </div>
      {pair ? (
        <div>
          <ShapeLadder doc={doc} a={a} title={false} onPick={(id) => setSelected(id)} />
          {onPick && <div className="shp-actions"><button type="button" className="shp-btn" onClick={() => onPick([pair.steady.id, pair.bust.id])}>Compare these two in the finish chart below</button></div>}
        </div>
      ) : <p className="shp-empty">No two top-30 players have win chances within 20% of each other and a different shape, so there is no pair to show.</p>}
      <div>
        <h3 className="shp-h">The top 30 against the typical shape</h3>
        <ShapeScatter a={a} selectedId={selected} onSelect={setSelected} />
        <ul className="shp-key"><li><Glyph kind="steady" /> Steady: above the curve</li><li><Glyph kind="bust" /> Boom-or-bust: below</li><li><Glyph kind="typical" /> Typical</li><li><i className="dash" aria-hidden="true" /> Typical shape</li></ul>
        {sel ? (
          <div className="shp-detail" style={{ marginTop: 10 }} aria-live="polite">
            <p><b>{shortName(sel.name)}</b> · {kindLabel(sel.kind)} · {signedPp(sel.residualPp)}</p>
            <p>{playerWhy(a, sel)}</p>
            {onPick && <div className="shp-actions"><button type="button" className="shp-btn" onClick={() => onPick([sel.id])}>Show in the finish chart below</button></div>}
          </div>
        ) : <p className="shp-fine" style={{ marginTop: 8 }}>Tap a point or a name to see why that player sits where he does.</p>}
      </div>
      <div>
        <h3 className="shp-h">Most unusual shapes</h3>
        <UnusualList a={a} selectedId={selected} onSelect={setSelected} />
      </div>
      <details>
        <summary>All {a.players.length} players</summary>
        <ScrollTable label="Shape of every top-30 player">
          <table className="ex-table shp-table">
            <thead><tr><th scope="col">Player</th><th scope="col">Win</th><th scope="col">{SHAPE_MARKET_LABEL[a.market]}</th><th scope="col">vs typical</th><th scope="col">Spread</th>{a.cut && <th scope="col">Miss cut</th>}</tr></thead>
            <tbody>
              {a.players.map((p) => (
                <tr key={p.id}>
                  <th scope="row"><button type="button" className="shp-linkbtn" onClick={() => setSelected(p.id)}>{shortName(p.name)}</button></th>
                  <td>{pct(p.win, 1)}</td><td>{pct(p.p, 1)}</td><td>{signedPp(p.residualPp)}</td><td>{p.sd === null ? "-" : p.sd.toFixed(2)}</td>{a.cut && <td>{p.pMiss === null ? "-" : pct(p.pMiss, 0)}</td>}
                </tr>
              ))}
            </tbody>
          </table>
        </ScrollTable>
      </details>
      <details>
        <summary>How the typical shape is worked out</summary>
        <p className="shp-fine">For the top 30 by model win chance, the dashed curve is a robust fit of the logit of the {SHAPE_MARKET_LABEL[a.market].toLowerCase()} chance on the log of the win chance ({a.fit.n} players{a.fit.method === "robust" ? ", outliers down-weighted" : ""}). A player&apos;s distance from the curve in percentage points is his shape signal; under 0.3 pp (or 3%) counts as typical. Pairs must have win chances within 15% (20% if none match). Round spread is compared with the median of the whole field. Model prices only, no sportsbook odds.{a.variety.spreadCorr !== null ? ` Correlation of spread with distance from the curve this week: ${a.variety.spreadCorr.toFixed(2)}.` : ""}</p>
      </details>
    </div>
  );
}

/* ------------------------------------------------------------------ compact card for This week */
export function ShapeSpotlightCard({ doc, openPlayer, onSeeMore, market = "top_20" }: { doc: ExplainDoc | null | undefined; openPlayer?: (id: number) => void; onSeeMore?: () => void; market?: ShapeMarket }) {
  const a = useMemo(() => analyzeShapes(doc, market), [doc, market]);
  if (!doc || !a.ok || !a.pairs[0]) return null;
  return (
    <section className="shp-card shl-card" aria-label="Same win chance, different finishes">
      <ShapeLadder doc={doc} a={a} onPick={openPlayer} onSeeMore={onSeeMore} />
    </section>
  );
}

