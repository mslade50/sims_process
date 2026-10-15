"use client";

/**
 * "Same chance to win, different weeks": which top-30 players have an unusual finish shape for their win chance, and why.
 * Chart-free (inline SVG only, no Recharts) so the compact card can sit on This week without adding to its bundle.
 * All numbers come from shape-rules.ts; this file only draws them.
 */
import { useMemo, useState } from "react";
import { SegmentedControl } from "./components";
import { Term } from "./ExplainBits";
import { pct, type ExplainDoc } from "./explain-rules";
import {
  analyzeShapes, availableMarkets, curveAt, finishBuckets, kindLabel, playerWhy, shortName, signedPp, SHAPE_MARKET_LABEL,
  type FinishBucket, type ShapeAnalysis, type ShapeKind, type ShapeMarket, type ShapePair, type ShapePlayer,
} from "./shape-rules";
import { AnimatedNumber, ScrollTable } from "./ui";
import "./shapes.css";

const lastName = (raw: string) => { const n = shortName(raw).split(" "); return n[n.length - 1] ?? ""; };

function Glyph({ kind }: { kind: ShapeKind }) { return <i className="shp-glyph" data-kind={kind} aria-hidden="true" />; }
function KindWord({ kind }: { kind: ShapeKind }) {
  const k = kind === "steady" ? "steady" : kind === "bust" ? "boom_bust" : "typical_shape";
  return <Term k={k}>{kindLabel(kind)}</Term>;
}

/** The finish position that closes the shaded "market" band: top 5/10/20 or the cut line for make cut. */
function bandEnd(market: ShapeMarket, doc: ExplainDoc): number | null {
  if (market === "make_cut") return doc.event?.cut_top_n && doc.event.cut_top_n > 0 ? doc.event.cut_top_n : null;
  return market === "top_5" ? 5 : market === "top_10" ? 10 : 20;
}

/* ------------------------------------------------------------------ paired finish histogram */
const CW = 380;
/** Buckets that pay out for the market (shaded): win through top 5/10/20, or every made-cut bucket for make cut. */
function inMarket(b: FinishBucket, market: ShapeMarket, band: number | null): boolean {
  if (b.lo === Infinity) return false;
  if (market === "make_cut") return true;
  return band !== null && b.hi !== null && b.hi <= band;
}
export function PairCurves({ doc, pair, market, height = 132 }: { doc: ExplainDoc; pair: ShapePair; market: ShapeMarket; height?: number }) {
  const cut = !!doc.event?.cut_round || !!doc.event?.cut_top_n;
  const pa = doc.players.find((p) => p.id === pair.steady.id);
  const pb = doc.players.find((p) => p.id === pair.bust.id);
  const ba = pa?.finish ? finishBuckets(pa.finish.pos, pa.finish.p_miss_cut, cut) : null;
  const bb = pb?.finish ? finishBuckets(pb.finish.pos, pb.finish.p_miss_cut, cut) : null;
  if (!ba || !bb || ba.length !== bb.length) return <p className="shp-curve-note">The finish chart is not available for one of these players, so only the prices are compared.</p>;
  const L = 4, R = 4, T = 28, B = 18;
  const n = ba.length, gw = (CW - L - R) / n, bw = Math.min(18, (gw - 8) / 2);
  const yMax = Math.max(...ba.map((b) => b.p), ...bb.map((b) => b.p), 0.05) * 1.08;
  const sy = (v: number) => T + (1 - v / yMax) * (height - T - B);
  const base = height - B;
  const band = bandEnd(market, doc);
  const lbl = (v: number) => (v < 0.0005 ? "" : v < 0.1 ? (v * 100).toFixed(1) : String(Math.round(v * 100)));
  const sa = shortName(pair.steady.name), sb = shortName(pair.bust.name);
  const counted = ba.map((b) => inMarket(b, market, band));
  const firstIn = counted.indexOf(true), lastIn = counted.lastIndexOf(true);
  const mkWord = market === "make_cut" ? "Make cut" : SHAPE_MARKET_LABEL[market];
  const label = `Chance of each finish for ${sa} and ${sb}: ${ba.map((b, i) => `${b.label} ${pct(b.p, 0)} vs ${pct(bb[i].p, 0)}`).join("; ")}.`;
  return (
    <div>
      <svg className="shp-curves shp-hist" viewBox={`0 0 ${CW} ${height}`} role="img" aria-label={label}>
        {ba.map((b, i) => {
          const x0 = L + i * gw, cx = x0 + gw / 2, q = bb[i];
          return (
            <g key={b.label}>
              {counted[i] && <rect className="band" x={x0 + 1} y={T - 2} width={gw - 2} height={base - T + 2} />}
              <rect className="steady-bar" x={cx - bw - 1} y={sy(b.p)} width={bw} height={Math.max(b.p > 0 ? 1.5 : 0, base - sy(b.p))} rx={1.5}><title>{`${sa}: ${b.label} ${pct(b.p, 1)}`}</title></rect>
              <rect className="bust-bar" x={cx + 1} y={sy(q.p)} width={bw} height={Math.max(q.p > 0 ? 1.5 : 0, base - sy(q.p))} rx={1.5}><title>{`${sb}: ${q.label} ${pct(q.p, 1)}`}</title></rect>
              <text className="val" x={cx - bw / 2 - 1} y={sy(b.p) - 3} textAnchor="middle">{lbl(b.p)}</text>
              <text className="val" x={cx + bw / 2 + 1} y={sy(q.p) - 3} textAnchor="middle">{lbl(q.p)}</text>
              <text x={cx} y={height - 5} textAnchor="middle">{b.label}</text>
            </g>
          );
        })}
        {firstIn >= 0 && <g className="bracket"><path d={`M${L + firstIn * gw + 3},${T - 4} V${T - 9} H${L + (lastIn + 1) * gw - 3} V${T - 4}`} /><text x={L + ((firstIn + lastIn + 1) / 2) * gw} y={T - 11} textAnchor="middle">{mkWord}: {pct(pair.steady.p, 1)} vs {pct(pair.bust.p, 1)}</text></g>}
        <line className="axis" x1={L} x2={CW - R} y1={base} y2={base} />
      </svg>
      <p className="shp-curve-note shp-legend"><span><i className="shp-key-sw" data-kind="steady" aria-hidden="true" /> {sa}</span><span><i className="shp-key-sw" data-kind="bust" aria-hidden="true" /> {sb}</span><span>Chance of each finish, in percent; shaded buckets add up to the {mkWord.toLowerCase()} chance. Buckets cover different numbers of places (Win is one place, 31–50 is twenty), so compare the two colours within a bucket, not the bucket heights.</span></p>
    </div>
  );
}

/* ------------------------------------------------------------------ pair spotlight */
function Side({ p, market, onPick }: { p: ShapePlayer; market: ShapeMarket; onPick?: (id: number) => void }) {
  const kind: ShapeKind = p.kind === "typical" ? "typical" : p.kind;
  return (
    <div className="shp-side" data-kind={kind}>
      <span className="shp-tag"><Glyph kind={kind} /><KindWord kind={kind} /></span>
      {onPick ? <button type="button" className="shp-name" onClick={() => onPick(p.id)}>{shortName(p.name)}</button> : <span className="shp-name">{shortName(p.name)}</span>}
      <dl className="shp-stats">
        <div><dt>Win</dt><dd><AnimatedNumber text={pct(p.win, 1)} /></dd></div>
        <div><dt>{SHAPE_MARKET_LABEL[market]}</dt><dd><AnimatedNumber text={pct(p.p, 1)} /></dd></div>
        {p.sd !== null && <div className="shp-minor"><dt><Term k="spread">Spread</Term></dt><dd className="shp-minor">{p.sd.toFixed(2)} strokes</dd></div>}
        {p.pMiss !== null && <div className="shp-minor"><dt>Miss cut</dt><dd className="shp-minor">{pct(p.pMiss, 0)}</dd></div>}
      </dl>
    </div>
  );
}
export function PairSpotlight({ doc, pair, market, compact = false, onPick }: { doc: ExplainDoc; pair: ShapePair; market: ShapeMarket; compact?: boolean; onPick?: (id: number) => void }) {
  const label = SHAPE_MARKET_LABEL[market].toLowerCase();
  const maxShape = Math.max(pair.shapePartPp, ...pair.effects.map((e) => Math.abs(e.effectPp)), 0.1);
  return (
    <div className="shp-pair">
      <div className="shp-sides">
        <Side p={pair.steady} market={market} onPick={onPick} />
        <Side p={pair.bust} market={market} onPick={onPick} />
      </div>
      <PairCurves doc={doc} pair={pair} market={market} height={compact ? 120 : 140} />
      <p className="shp-gap"><b>{pair.gapPp.toFixed(1)} pp</b> apart on {label} at a {pair.winRatio < 1.06 ? "near-identical" : "similar"} win chance ({pct(pair.bust.win, 1)} vs {pct(pair.steady.win, 1)}).</p>
      <p className="shp-why">{pair.why}</p>
      {!compact && pair.effects.length > 0 && (
        <div>
          <h4 className="shp-h">What lines up with the shape gap</h4>
          <ul className="shp-effects">
            {pair.effects.map((e) => (
              <li key={e.key}>
                <span>{e.label}</span>
                <span className="bar" aria-hidden="true"><i data-neg={e.effectPp < 0 ? "" : undefined} style={{ width: `${Math.min(100, (Math.abs(e.effectPp) / maxShape) * 100)}%` }} /></span>
                <b>{e.effectPp >= 0 ? "+" : "−"}{Math.abs(e.effectPp).toFixed(1)} pp</b>
              </li>
            ))}
          </ul>
          <p className="shp-fine">Of the {pair.shapePartPp.toFixed(1)} pp shape gap. Each bar is how far the two sit apart on that input times how strongly it goes with shape across this week&apos;s top 30. It is an association, not a controlled test, and the bars need not add up.</p>
        </div>
      )}
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
  const [pairIx, setPairIx] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);
  const a = useMemo(() => analyzeShapes(doc, market), [doc, market]);
  if (!a.ok) return <p className="shp-empty">{a.reason ?? "Shapes are not available for this simulation."}</p>;
  const pair = a.pairs[Math.min(pairIx, a.pairs.length - 1)];
  const sel = a.players.find((p) => p.id === selected) ?? null;
  return (
    <div className="shp-root">
      <p className="shp-variety"><span className="shp-tier" data-tier={a.variety.tier}>{TIER_WORD[a.variety.tier]}</span>{" "}{a.variety.text}</p>
      <p className="shp-intro">Yes, it is possible, and the simulation does it on purpose. Every player has an expected score and a <Term k="spread">round-to-round spread</Term>. A high-spread player wins more of his chance from rare blow-out weeks and keeps fewer steady top finishes; a low-spread player is the reverse. Two players can therefore land on the same win chance with very different top-20 chances.</p>
      <div className="shp-tabs">
        <SegmentedControl label="Market for shape" value={a.market} onChange={(m) => { setMarket(m); setSelected(null); }} options={markets.map((m) => ({ value: m, label: SHAPE_MARKET_LABEL[m] }))} />
        {a.pairs.length > 1 && <SegmentedControl label="Pair" value={String(Math.min(pairIx, a.pairs.length - 1))} onChange={(v) => setPairIx(Number(v))} options={a.pairs.map((p, i) => ({ value: String(i), label: i === 0 ? "Biggest gap" : `${lastName(p.bust.name)} / ${lastName(p.steady.name)}` }))} />}
      </div>
      {pair ? (
        <div>
          <PairSpotlight doc={doc} pair={pair} market={a.market} onPick={(id) => setSelected(id)} />
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
    <section className="shp-card" aria-label="Same chance to win, different weeks">
      <div>
        <span className="eyebrow">Distribution shapes</span>
        <h3>Same chance to win, different weeks</h3>
      </div>
      <p className="shp-variety"><span className="shp-tier" data-tier={a.variety.tier}>{TIER_WORD[a.variety.tier]}</span>{" "}{a.variety.text}</p>
      <PairSpotlight doc={doc} pair={a.pairs[0]} market={a.market} compact onPick={openPlayer} />
      {onSeeMore && <div className="shp-actions"><button type="button" className="shp-btn" onClick={onSeeMore}>See all 30 shapes</button></div>}
    </section>
  );
}

