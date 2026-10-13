"use client";

import { useMemo, useState } from "react";
import { Panel, LoadingState } from "./components";
import { useDashboardData } from "./data";
import { playerDisplayName } from "./player-name";
import { sliceHistory, type HistoryFilters } from "./player-history-filters";
import type { DeepProfile } from "./player-deep-rules";
import "./player-dispersion.css";
import {
  DEFAULT_APP_BANDS, DEFAULT_TEE_BANDS, GREEN_HALF_DEPTH, GREEN_HALF_WIDTH, LIE_OPTIONS, ROUND_CHOICES, SIT_HELP, availableDims, bandLabel, bandStats, bucketCounts, compare,
  depthRange, dispersionKey, fairwayGuide, fmt, fmtFt, greenWindow, hasEventIds, headline, inBucket, laterality, niceStep, parseIndex, parsePayload, parseReference, referenceBand,
  roundCount, selectPoints, sideRange, situationReadout, situationStat, sliceKeys, sliceLabel, sliceRounds, sliceSpan, ticks, viewWindow, windowSpan,
  type BandStat, type Kind, type Point, type Reference, type SitResult, type SituationBucket, type SituationDim,
} from "./player-dispersion-rules";

const INDEX_KEY = "golfprice/player_profiles/dispersion/index.json";
const BAND_COLORS = ["var(--chart-1)", "var(--chart-2)", "var(--chart-6)", "var(--chart-4)", "var(--chart-3)"];
const W = 360, H = 470, M = { l: 40, r: 12, t: 14, b: 30 };

type Sit = { dim: SituationDim; bucket: SituationBucket; result: SitResult | null } | null;
type SitChoice = { dim: string; bucket: string } | null;

const shortDate = (iso: string) => {
  const d = new Date(`${iso}T12:00:00Z`);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
};

function TeeChart({ points, stats, reference, bands, sit }: { points: Point[]; stats: BandStat[]; reference: Reference | null; bands: number[][]; sit: Sit }) {
  const view = useMemo(() => viewWindow(points, "tee"), [points]);
  const { yLo, yHi, xMax } = view;
  const sx = (lat: number) => M.l + ((lat + xMax) / (2 * xMax)) * (W - M.l - M.r);
  const sy = (yd: number) => H - M.b - ((yd - yLo) / (yHi - yLo)) * (H - M.t - M.b);
  const yStep = niceStep(yHi - yLo, 7), xStep = xMax > 40 ? 20 : 10;
  const guide = useMemo(() => fairwayGuide(reference, bands, yLo, yHi), [reference, bands, yLo, yHi]);
  const exaggeration = (((W - M.l - M.r) / (2 * xMax)) / ((H - M.t - M.b) / (yHi - yLo)));
  const label = "Chart of tee shots on one straightened fairway, seen from above. The vertical line is the middle of the fairway, following the bend on doglegs, and the golfer hits towards the top. Height is distance from the tee and the sideways position is yards left or right of the middle. A shaded strip shows how wide fairways typically are. When a situation is chosen, its shots are highlighted and an arrow runs from the PGA average to this golfer's average. The table next to the chart has the numbers.";
  return (
    <svg className="dsp-chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
      <rect x={M.l} y={M.t} width={W - M.l - M.r} height={H - M.t - M.b} className="dsp-ground" />
      {guide.map((g) => <rect key={`fw${g.y0}`} x={sx(g.left)} width={Math.max(1, sx(g.right) - sx(g.left))} y={sy(g.y1)} height={Math.max(1, sy(g.y0) - sy(g.y1))} className="dsp-fairway"><title>{`Typical PGA fairway: about ${fmt(g.left, 0)} to ${fmt(g.right, 0)} yd from the middle`}</title></rect>)}
      {ticks(yLo, yHi, yStep).map((v) => (
        <g key={`y${v}`}>
          <line x1={M.l} x2={W - M.r} y1={sy(v)} y2={sy(v)} className="dsp-grid" />
          <text x={M.l - 5} y={sy(v) + 3.5} textAnchor="end" className="dsp-tick">{v}</text>
        </g>
      ))}
      {ticks(-xMax, xMax, xStep).filter((v) => v !== 0).map((v) => (
        <g key={`x${v}`}>
          <line x1={sx(v)} x2={sx(v)} y1={M.t} y2={H - M.b} className="dsp-grid dsp-grid-v" />
          <text x={sx(v)} y={H - M.b + 12} textAnchor="middle" className="dsp-tick">{Math.abs(v)}{v < 0 ? "L" : "R"}</text>
        </g>
      ))}
      <line x1={sx(0)} x2={sx(0)} y1={M.t} y2={H - M.b} className="dsp-centre" />
      <text x={sx(0)} y={H - M.b + 24} textAnchor="middle" className="dsp-axis-title">Fairway centre · yards left / right</text>
      <text transform={`translate(9 ${(M.t + H - M.b) / 2}) rotate(-90)`} textAnchor="middle" className="dsp-axis-title">Yards from the tee</text>
      <text x={M.l + 5} y={M.t + 11} className="dsp-note">▲ toward the green</text>
      {[false, true].map((pass) => points.map((p, i) => {
        if (Math.abs(p.lat) > xMax || p.y < yLo || p.y > yHi) return null;
        const hit = !sit || inBucket(p, sit.dim, sit.bucket);
        if (pass ? !(sit && hit) : !!(sit && hit)) return null;
        return <circle key={`${pass}${i}`} cx={sx(p.lat)} cy={sy(p.y)} r={sit ? (hit ? 2.6 : 1.5) : 2} fill={BAND_COLORS[p.band % BAND_COLORS.length]} className={sit && !hit ? "dsp-dot dsp-dot-dim" : sit ? "dsp-dot dsp-dot-hit" : "dsp-dot"} />;
      }))}
      {stats.map((s) => {
        const ref = referenceBand(reference, "tee", s.band);
        const y = sy(s.meanY);
        const c = BAND_COLORS[s.band % BAND_COLORS.length];
        return (
          <g key={`b${s.band}`}>
            {ref && <rect x={sx(ref.lat.q10)} width={Math.max(1, sx(ref.lat.q90) - sx(ref.lat.q10))} y={y - 9} height={5} rx={2} className="dsp-pga-bar"><title>{`Typical PGA middle 80%: ${fmt(ref.lat.q10)} to ${fmt(ref.lat.q90)} yd`}</title></rect>}
            <line x1={sx(s.q10)} x2={sx(s.q90)} y1={y - 2} y2={y - 2} stroke={c} className="dsp-bar"><title>{`Player middle 80%: ${fmt(s.q10)} to ${fmt(s.q90)} yd`}</title></line>
            <line x1={sx(s.q10)} x2={sx(s.q10)} y1={y - 6} y2={y + 2} stroke={c} className="dsp-cap" />
            <line x1={sx(s.q90)} x2={sx(s.q90)} y1={y - 6} y2={y + 2} stroke={c} className="dsp-cap" />
          </g>
        );
      })}
      {sit?.result?.showMean && sit.result.pgaLat !== null && (
        <ShiftArrow id="dsp-ah-tee" x1={sx(sit.result.pgaLat)} y1={sy(sit.result.meanDist)} x2={sx(sit.result.meanLat)} y2={sy(sit.result.meanDist)}
          title={`Average in this situation: PGA ${laterality(sit.result.pgaLat, 1)}, this golfer ${laterality(sit.result.meanLat, 1)} of the landing line`} />
      )}
      <text x={W - M.r} y={M.t + 11} textAnchor="end" className="dsp-note">{exaggeration > 1.05 ? `sideways scale ×${exaggeration.toFixed(1)}` : ""}</text>
    </svg>
  );
}

/** Arrow from the PGA average (hollow dot) to this golfer's average (filled dot) for the chosen situation. */
function ShiftArrow({ id, x1, y1, x2, y2, title }: { id: string; x1: number; y1: number; x2: number; y2: number; title: string }) {
  const long = Math.hypot(x2 - x1, y2 - y1) > 7;
  return (
    <g className="dsp-shift">
      <defs><marker id={id} viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" /></marker></defs>
      {long && <line x1={x1} y1={y1} x2={x2} y2={y2} markerEnd={`url(#${id})`} className="dsp-shift-line" />}
      <circle cx={x1} cy={y1} r={4.2} className="dsp-shift-pga" />
      <circle cx={x2} cy={y2} r={3.4} className="dsp-shift-you" />
      <title>{title}</title>
    </g>
  );
}

function SituationPicker({ kind, dims, choice, counts, onDim, onBucket }: {
  kind: Kind; dims: SituationDim[]; choice: SitChoice; counts: Record<string, Record<string, number>>; onDim: (key: string | null) => void; onBucket: (key: string) => void;
}) {
  const dim = dims.find((d) => d.key === choice?.dim) ?? null;
  const gate = kind === "tee" ? 60 : 100;
  return (
    <div className="dsp-sit" role="group" aria-label="Situation">
      <div className="dsp-sit-row">
        <span className="dsp-sit-label">Situation</span>
        <div className="segmented" role="group" aria-label="Situation type">
          <button type="button" className={!dim ? "active" : ""} aria-pressed={!dim} onClick={() => onDim(null)}>All shots</button>
          {dims.map((d) => (
            <button key={d.key} type="button" className={dim?.key === d.key ? "active" : ""} aria-pressed={dim?.key === d.key} aria-label={d.label} title={SIT_HELP[d.key]} onClick={() => onDim(d.key)}>{d.label}</button>
          ))}
        </div>
      </div>
      {dim && (
        <>
          <div className="dsp-bands dsp-sit-buckets" role="group" aria-label={`${dim.label} choices`}>
            {dim.buckets.map((b) => {
              const n = counts[dim.key]?.[b.key] ?? 0;
              return (
                <button key={b.key} type="button" aria-pressed={choice?.bucket === b.key} disabled={n === 0} className={choice?.bucket === b.key ? "on" : n < gate ? "off" : ""}
                  title={n < gate ? `Only ${n} shots: the width needs ${gate}` : `${n} shots`} onClick={() => onBucket(b.key)}>
                  {b.label}<small>{n}</small>
                </button>
              );
            })}
          </div>
          <p className="dsp-sit-help">{SIT_HELP[dim.key]}</p>
        </>
      )}
    </div>
  );
}

const G = 360, GC = G / 2;

function GreenChart({ points, stats, sit }: { points: Point[]; stats: BandStat[]; sit: Sit }) {
  const { radius, rings, outside } = useMemo(() => greenWindow(points), [points]);
  const k = (GC - 26) / radius; // px per yard, same both ways so rings are true circles
  const cx = (lat: number) => GC + lat * k, cy = (along: number) => GC - along * k; // long is up
  const label = "Top-down chart of approach shots around the pin, which sits in the middle. Shots that finished long are above the pin, short below, left and right as the golfer faced the flag. Circles mark distance from the pin. When a situation is chosen, its shots are highlighted and an arrow runs from the PGA average finish to this golfer's average. The cards next to the chart have the numbers.";
  const ringLabel = (r: number) => (r <= 5 ? "5 yd" : `${r} yd`);
  return (
    <svg className="dsp-chart dsp-green-chart" viewBox={`0 0 ${G} ${G}`} role="img" aria-label={label}>
      <rect x={1} y={1} width={G - 2} height={G - 2} className="dsp-ground" />
      <ellipse cx={GC} cy={GC} rx={GREEN_HALF_WIDTH * k} ry={GREEN_HALF_DEPTH * k} className="dsp-green"><title>Stylised green, about 22 by 30 yards, for scale only</title></ellipse>
      {rings.map((r) => (
        <g key={r}>
          <circle cx={GC} cy={GC} r={r * k} className="dsp-ring" />
        </g>
      ))}
      <line x1={GC} x2={GC} y1={8} y2={G - 8} className="dsp-centre" />
      <line x1={8} x2={G - 8} y1={GC} y2={GC} className="dsp-centre" />
      <text x={G - 8} y={GC - 4} textAnchor="end" className="dsp-axis-title">right</text>
      <text x={8} y={GC - 4} className="dsp-axis-title">left</text>
      <text x={GC + 5} y={16} className="dsp-axis-title">long ▲</text>
      <text x={GC + 5} y={G - 8} className="dsp-axis-title">short ▼ (you hit from here)</text>
      {[false, true].map((pass) => points.map((p, i) => {
        if (Math.abs(p.lat) > radius || Math.abs(p.along) > radius) return null;
        const hit = !sit || inBucket(p, sit.dim, sit.bucket);
        if (pass ? !(sit && hit) : !!(sit && hit)) return null;
        return <circle key={`${pass}${i}`} cx={cx(p.lat)} cy={cy(p.along)} r={sit ? (hit ? 2.4 : 1.4) : 1.8} fill={BAND_COLORS[p.band % BAND_COLORS.length]} className={sit && !hit ? "dsp-dot dsp-dot-dim" : sit ? "dsp-dot dsp-dot-hit" : "dsp-dot"} />;
      }))}
      {rings.map((r) => <text key={`rl${r}`} x={GC + r * k * 0.707 + 2} y={GC - r * k * 0.707 - 2} className="dsp-tick dsp-ringlabel">{ringLabel(r)}</text>)}
      {stats.filter((s) => s.proxMed <= radius).map((s) => (
        <circle key={`m${s.band}`} cx={GC} cy={GC} r={s.proxMed * k} stroke={BAND_COLORS[s.band % BAND_COLORS.length]} className="dsp-prox"><title>{`Half of these approaches finished inside ${fmtFt(s.proxMed)} of the pin`}</title></circle>
      ))}
      {sit?.result?.showMean && sit.result.pgaLat !== null && sit.result.pgaAlong !== null && (
        <ShiftArrow id="dsp-ah-app" x1={cx(sit.result.pgaLat)} y1={cy(sit.result.pgaAlong)} x2={cx(sit.result.meanLat)} y2={cy(sit.result.meanAlong)}
          title={`Average finish in this situation: PGA ${laterality(sit.result.pgaLat, 1)} and ${fmt(Math.abs(sit.result.pgaAlong), 1)} yd ${sit.result.pgaAlong < 0 ? "short" : "long"}; this golfer ${laterality(sit.result.meanLat, 1)} and ${fmt(Math.abs(sit.result.meanAlong), 1)} yd ${sit.result.meanAlong < 0 ? "short" : "long"}`} />
      )}
      <g transform={`translate(${GC} ${GC})`} className="dsp-flag"><line y1={-12} y2={2} /><path d="M0 -12 l9 4 l-9 4 z" /><circle r={2.2} /><title>Pin</title></g>
      <text x={G - 8} y={16} textAnchor="end" className="dsp-note">{outside ? `${outside} farther-out shots not shown` : ""}</text>
    </svg>
  );
}

function ApproachCards({ stats, bands, reference, lie }: { stats: BandStat[]; bands: number[][]; reference: Reference | null; lie: string }) {
  return (
    <ul className="dsp-cards" aria-label="Approach numbers by starting yardage">
      {stats.map((s) => {
        const ref = referenceBand(reference, "app", s.band, lie);
        const v = compare(s, ref);
        const refAlong = ref?.along;
        return (
          <li key={s.band} className="dsp-card">
            <h4><i style={{ background: BAND_COLORS[s.band % BAND_COLORS.length] }} />{bandLabel("app", bands, s.band)} yd <span>{s.n} shots</span></h4>
            <dl>
              <div><dt>Median distance from the pin</dt><dd>{fmtFt(s.proxMed)}<small>PGA {ref?.prox_med !== undefined ? fmtFt(ref.prox_med) : "—"}</small></dd></div>
              <div><dt>Left / right, middle 80%</dt><dd>{sideRange(s.q10, s.q90)}<small>PGA {ref ? sideRange(ref.lat.q10, ref.lat.q90) : "—"}</small></dd></div>
              <div><dt>Long / short, middle 80%</dt><dd>{depthRange(s.alongQ10, s.alongQ90)}<small>PGA {refAlong ? depthRange(refAlong.q10, refAlong.q90) : "—"}</small></dd></div>
              <div><dt>On the green</dt><dd>{s.rate === null ? "—" : `${Math.round(s.rate * 100)}%`}{ref?.green_rate !== undefined && s.rate !== null && <small>PGA {Math.round(ref.green_rate * 100)}%</small>}</dd></div>
            </dl>
            <p className={`dsp-${v.tone}`}>{v.text}</p>
          </li>
        );
      })}
    </ul>
  );
}

function BandTable({ kind, stats, bands, reference, lie }: { kind: Kind; stats: BandStat[]; bands: number[][]; reference: Reference | null; lie: string }) {
  return (
    <div className="dsp-table-wrap">
      <table className="dsp-table">
        <caption>Side-to-side spread by {kind === "tee" ? "distance travelled" : "starting yardage"}. “Middle 80%” is the width that holds the central 80% of shots.</caption>
        <thead>
          <tr><th scope="col">{kind === "tee" ? "Distance" : "Yardage"}</th><th scope="col">Shots</th><th scope="col">Middle 80%</th><th scope="col">{kind === "tee" ? "Fairway" : "Green"}</th></tr>
        </thead>
        <tbody>
          {stats.map((s) => {
            const ref = referenceBand(reference, kind, s.band, lie);
            const v = compare(s, ref);
            const refRate = kind === "tee" ? ref?.fairway_rate : ref?.green_rate;
            return (
              <tr key={s.band}>
                <th scope="row"><i style={{ background: BAND_COLORS[s.band % BAND_COLORS.length] }} />{bandLabel(kind, bands, s.band)} yd<small className={`dsp-${v.tone}`}>{v.text}</small><small>{leanText(s, kind)}</small></th>
                <td>{s.n}</td>
                <td>{fmt(s.width, 0)} yd<small>PGA {ref ? fmt(ref.lat.q90 - ref.lat.q10, 0) : "—"} yd</small></td>
                <td>{s.rate === null ? "—" : `${Math.round(s.rate * 100)}%`}{refRate !== undefined && s.rate !== null && <small>PGA {Math.round(refRate * 100)}%</small>}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function leanText(s: BandStat, kind: Kind): string {
  const side = Math.abs(s.meanLat) < 0.75 ? "centred side to side" : `leans ${laterality(s.meanLat)}`;
  if (kind === "tee") return side;
  const along = Math.abs(s.meanAlong) < 1 ? "" : `, ${Math.abs(s.meanAlong).toFixed(0)} yd ${s.meanAlong < 0 ? "short" : "long"} on average`;
  return side + along;
}

export function PlayerDispersion({ dgId, name, deep, filters }: { dgId: number | null | undefined; name?: string; deep?: DeepProfile | null; filters?: HistoryFilters }) {
  const index = useDashboardData<unknown>(INDEX_KEY);
  const idx = useMemo(() => parseIndex(index.data), [index.data]);
  const pKey = idx ? dispersionKey(idx, dgId) : null;
  const payloadLoad = useDashboardData<unknown>(pKey);
  const refLoad = useDashboardData<unknown>(pKey && idx ? idx.reference_key : null);
  const payload = useMemo(() => (dgId && payloadLoad.data ? parsePayload(payloadLoad.data, dgId) : null), [payloadLoad.data, dgId]);
  const reference = useMemo(() => parseReference(refLoad.data), [refLoad.data]);

  const [kind, setKind] = useState<Kind>("tee");
  const [rounds, setRounds] = useState<number>(50);
  const [useSlice, setUseSlice] = useState(false);
  const canSlice = !!(deep?.history?.events?.length && filters);
  const sliceOn = useSlice && canSlice;
  const sliceEvents = useMemo(() => (sliceOn && deep && filters ? sliceHistory(deep.history?.events ?? [], { ...filters, asOf: deep.as_of }) : []), [sliceOn, deep, filters]);
  const keys = useMemo(() => sliceKeys(sliceEvents), [sliceEvents]);
  const allowed = useMemo(() => (sliceOn && payload ? sliceRounds(payload, keys) : undefined), [sliceOn, payload, keys]);
  const who = playerDisplayName(name) || "this golfer";
  const [lie, setLie] = useState("all");
  const [off, setOff] = useState<Record<string, boolean>>({});
  const [sitChoice, setSitChoice] = useState<Record<Kind, SitChoice>>({ tee: null, app: null });

  const bands = (kind === "tee" ? reference?.tee_bands : reference?.app_bands) ?? (kind === "tee" ? DEFAULT_TEE_BANDS : DEFAULT_APP_BANDS);
  const all = useMemo(() => (payload ? selectPoints(payload, kind, rounds, bands, kind === "app" ? lie : "all", allowed) : []), [payload, kind, rounds, bands, lie, allowed]);
  const hidden = (b: number) => off[`${kind}${b}`] === true;
  const points = useMemo(() => all.filter((p) => !off[`${kind}${p.band}`]), [all, off, kind]);
  const stats = useMemo(() => bandStats(points, kind, bands), [points, kind, bands]);
  const present = useMemo(() => bandStats(all, kind, bands).map((s) => s.band), [all, kind, bands]);
  const dims = useMemo(() => availableDims(reference, kind, payload), [reference, kind, payload]);
  const counts = useMemo(() => Object.fromEntries(dims.map((d) => [d.key, bucketCounts(points, d)])), [dims, points]);
  const choice = sitChoice[kind];
  const sitDim = dims.find((d) => d.key === choice?.dim) ?? null;
  const sitBucket = sitDim?.buckets.find((b) => b.key === choice?.bucket) ?? null;
  const sitResult = useMemo(() => (sitDim && sitBucket ? situationStat(points, kind, sitDim, sitBucket, reference) : null), [points, kind, sitDim, sitBucket, reference]);
  const sit: Sit = sitDim && sitBucket ? { dim: sitDim, bucket: sitBucket, result: sitResult } : null;
  const pickDim = (key: string | null) => {
    const d = dims.find((x) => x.key === key);
    if (!d) return setSitChoice((c) => ({ ...c, [kind]: null }));
    const gate = kind === "tee" ? 60 : 100; // first bucket with enough shots (left before right), else the largest
    const best = d.buckets.find((b) => (counts[d.key]?.[b.key] ?? 0) >= gate) ?? [...d.buckets].sort((a, b) => (counts[d.key]?.[b.key] ?? 0) - (counts[d.key]?.[a.key] ?? 0))[0];
    setSitChoice((c) => ({ ...c, [kind]: { dim: d.key, bucket: best.key } }));
  };
  const pickBucket = (key: string) => setSitChoice((c) => ({ ...c, [kind]: c[kind] ? { ...c[kind]!, bucket: key } : c[kind] }));
  const readout = sitDim && sitBucket && sitResult ? situationReadout(kind, who, sitDim, sitBucket, sitResult) : null;

  if (index.loading) return null;
  if (index.error || !idx) return null; // not published yet: stay out of the way
  if (!pKey) return <Panel title="Shot dispersion" eyebrow="Fairway and green views"><p className="pp-muted">No published shot-by-shot data for {who} (PGA Tour rounds only).</p></Panel>;
  if (payloadLoad.loading || refLoad.loading) return <LoadingState label="Loading shot dispersion" />;
  if (payloadLoad.error || !payload) return <Panel title="Shot dispersion" eyebrow="Fairway and green views"><p className="pp-muted">Shot dispersion did not load for {who}.</p></Panel>;

  const available = roundCount(payload, kind);
  const span = sliceOn ? null : windowSpan(payload, rounds);
  const sSpan = sliceOn && allowed ? sliceSpan(payload, kind, allowed) : null;
  const sliceName = filters?.event ? deep?.history?.events.find((e) => e.id === filters.event)?.name : undefined;
  const kindWord = kind === "tee" ? "tee-shot" : "approach";
  const sliceEmpty = !hasEventIds(payload)
    ? "This shot data was published before it could be matched to the slice filters. It will work after the next profile publish."
    : keys.size === 0 ? "No rounds match the slice filters above, so there are no shots to show. Loosen a filter."
    : `None of the ${keys.size} rounds in this slice has PGA Tour ${kindWord} data. Shot detail exists only for PGA Tour events (not other tours) over roughly the last three years.`;
  const hasKind = (k: Kind) => !!(k === "tee" ? payload.tee : payload.app);

  return (
    <Panel title="Shot dispersion" eyebrow="Fairway and green views · PGA Tour shot data" className="dsp-panel">
      <div className="dsp-controls">
        <div className="segmented" role="group" aria-label="Shot type">
          {(["tee", "app"] as Kind[]).map((k) => <button key={k} type="button" className={kind === k ? "active" : ""} aria-pressed={kind === k} disabled={!hasKind(k)} onClick={() => setKind(k)}>{k === "tee" ? "Tee shots" : "Approach"}</button>)}
        </div>
        {!sliceOn && (
          <div className="segmented" role="group" aria-label="Number of recent rounds">
            {ROUND_CHOICES.map((n) => <button key={n} type="button" className={rounds === n ? "active" : ""} aria-pressed={rounds === n} onClick={() => setRounds(n)}>Last {n}</button>)}
          </div>
        )}
        {canSlice && (
          <label className="dsp-slice-toggle" title="Show only the rounds picked by “Explore a slice of their golf” above (history window, season, tour, events, courses, conditions, rounds R1–R4).">
            <input type="checkbox" checked={useSlice} onChange={(e) => setUseSlice(e.target.checked)} />Use the slice filters above
          </label>
        )}
        {kind === "app" && (
          <label className="dsp-lie">Starting lie
            <select value={lie} onChange={(e) => setLie(e.target.value)}>{LIE_OPTIONS.map((o) => <option key={o.key} value={o.key}>{o.label}</option>)}</select>
          </label>
        )}
      </div>
      {dims.length > 0 && <SituationPicker kind={kind} dims={dims} choice={choice} counts={counts} onDim={pickDim} onBucket={pickBucket} />}
      {readout && sitResult && (
        <p className="dsp-readout" aria-live="polite"><b>{readout.text}</b>{readout.width && <span> {readout.width}</span>}<small> Situations are estimated from shot coordinates, not official course maps.</small></p>
      )}
      <div className="dsp-bands" role="group" aria-label={kind === "tee" ? "Distance ranges to show" : "Starting yardages to show"}>
        {present.map((b) => (
          <button key={b} type="button" aria-pressed={!hidden(b)} className={hidden(b) ? "off" : ""} onClick={() => setOff((o) => ({ ...o, [`${kind}${b}`]: !hidden(b) }))}>
            <i style={{ background: BAND_COLORS[b % BAND_COLORS.length] }} />{bandLabel(kind, bands, b)}{kind === "app" ? " yd" : " yd"}
          </button>
        ))}
      </div>
      {sliceOn && (
        <p className="dsp-sample dsp-slice-caption">
          Slice: <b>{sliceLabel(filters ?? {}, sliceName)}</b>{sSpan ? <> · <b>{sSpan.n}</b> rounds with {kindWord} data, {shortDate(sSpan.first)} to {shortDate(sSpan.last)} · <b>{points.length.toLocaleString("en-US")}</b> shots shown</> : null}
        </p>
      )}
      {span && <p className="dsp-sample">Last <b>{Math.min(rounds, payload.rounds.length)}</b> rounds with shot data{available < rounds ? ` (all ${available} with ${kind === "tee" ? "tee-shot" : "approach"} data)` : ""}, {shortDate(span.first)} to {shortDate(span.last)} · <b>{points.length.toLocaleString("en-US")}</b> shots shown</p>}
      {points.length === 0 ? <p className="pp-muted dsp-empty" role="status">{sliceOn && all.length === 0 ? sliceEmpty : "No shots match this view."}</p> : (
        <>
          <p className="dsp-headline">{headline(kind, stats, bands, reference, kind === "app" ? lie : "all")}</p>
          <div className="dsp-body">
            <figure className="dsp-figure">
              {kind === "tee" ? <TeeChart points={points} stats={stats} reference={reference} bands={bands} sit={sit} /> : <GreenChart points={points} stats={stats} sit={sit} />}
              <figcaption>
                {kind === "tee" ? (
                  <>
                    <span><i className="dsp-key-bar" />Coloured bar: this golfer&apos;s middle 80% of shots</span>
                    <span><i className="dsp-key-pga" />Grey bar: typical PGA Tour middle 80%</span>
                    <span><i className="dsp-key-fw" />Shaded strip: typical fairway width</span>
                    {sit && <span><i className="dsp-key-arrow" />Hollow dot to filled dot: PGA average to this golfer, in the chosen situation</span>}
                  </>
                ) : (
                  <>
                    <span><i className="dsp-key-ring" />Coloured ring: half the shots finished inside it</span>
                    <span><i className="dsp-key-fw" />Green shape: stylised {GREEN_HALF_WIDTH * 2} by {GREEN_HALF_DEPTH * 2} yd green, for scale</span>
                    {sit && <span><i className="dsp-key-arrow" />Hollow dot to filled dot: PGA average finish to this golfer, in the chosen situation</span>}
                  </>
                )}
              </figcaption>
            </figure>
            {kind === "tee" ? <BandTable kind={kind} stats={stats} bands={bands} reference={reference} lie="all" /> : <ApproachCards stats={stats} bands={bands} reference={reference} lie={lie} />}
          </div>
        </>
      )}
      <details className="pp-details">
        <summary>How to read this, and what it cannot tell you</summary>
        {kind === "tee" ? (
          <p>Every dot is one tee shot on a par 4 or 5. All holes are straightened onto one fairway: the vertical line is the estimated middle of the fairway, the golfer hits towards the top, height is how far the shot travelled, and the sideways position is how far left or right of the middle it finished. The middle is the field&apos;s landing line: a smooth curve through where that day&apos;s fairway-finding drives landed at each distance on that hole, so on a dogleg it follows the bend instead of a straight line from the tee. The shaded strip is how far from the middle the field&apos;s fairway-finding drives finished (2nd to 98th percentile), a proxy for fairway width rather than a measured one. This is not the target the golfer aimed at.</p>
        ) : (
          <p>Every dot is one approach (including par-3 tee shots, 50 to 300 yards out), seen from above with the pin in the middle. Up is long (past the pin), down is short, and left and right are as the golfer faced the flag. Rings are distances from the pin, and the green outline is only a rough size guide: real greens differ in shape and size. Golfers often aim away from the pin, so this shows where shots finished relative to the flag, not ability alone.</p>
        )}
        {dims.length > 0 && <p>Situations (dogleg, water, bunkers, trees, pin position) are estimates built from where shots were hit, ended and putted from; no official course or green maps are published. They are only worth reading when the shot count is shown to be large enough, and bunkers and water mostly describe the hole, not a golfer avoiding them. Water and bunker sides are where balls have ended, so they partly reflect where the field misses.</p>}
        <p>The grey bars and PGA figures use every PGA Tour golfer&apos;s shots over the same period; the comparison words compare this golfer&apos;s side-to-side spread with the typical PGA golfer at the same distances. Greens and fairways hit are counted only for rounds from 2024, when the data provider&apos;s ground-type labels settled. DP World Tour rounds are not included.</p>
      </details>
    </Panel>
  );
}
