"use client";

import { useMemo, useState } from "react";
import { Panel, LoadingState } from "./components";
import { useDashboardData } from "./data";
import "./player-dispersion.css";
import {
  DEFAULT_APP_BANDS, DEFAULT_TEE_BANDS, LIE_OPTIONS, ROUND_CHOICES, bandLabel, bandStats, compare, dispersionKey, fmt, headline, laterality,
  niceStep, parseIndex, parsePayload, parseReference, referenceBand, roundCount, selectPoints, ticks, viewWindow, windowSpan,
  type BandStat, type Kind, type Point, type Reference,
} from "./player-dispersion-rules";

const INDEX_KEY = "golfprice/player_profiles/dispersion/index.json";
const BAND_COLORS = ["var(--chart-1)", "var(--chart-2)", "var(--chart-6)", "var(--chart-4)", "var(--chart-3)"];
const W = 360, H = 470, M = { l: 40, r: 12, t: 14, b: 30 };

const shortDate = (iso: string) => {
  const d = new Date(`${iso}T12:00:00Z`);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
};

function RangeChart({ kind, points, stats, reference, lie }: { kind: Kind; points: Point[]; stats: BandStat[]; reference: Reference | null; lie: string }) {
  const view = useMemo(() => viewWindow(points, kind), [points, kind]);
  const { yLo, yHi, xMax } = view;
  const sx = (lat: number) => M.l + ((lat + xMax) / (2 * xMax)) * (W - M.l - M.r);
  const sy = (yd: number) => H - M.b - ((yd - yLo) / (yHi - yLo)) * (H - M.t - M.b);
  const yStep = niceStep(yHi - yLo, 7), xStep = xMax > 40 ? 20 : 10;
  const exaggeration = ((H - M.t - M.b) / (yHi - yLo)) > 0 ? (((W - M.l - M.r) / (2 * xMax)) / ((H - M.t - M.b) / (yHi - yLo))) : 1;
  const label = kind === "tee"
    ? "Chart of tee shots as seen from above. The vertical line is the middle of the fairway; height is distance travelled. See the table below for the numbers."
    : "Chart of approach shots as seen from above. The vertical line runs from the ball to the pin; height is the distance the ball finished from where it was hit. See the table below for the numbers.";
  return (
    <svg className="dsp-chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
      <rect x={M.l} y={M.t} width={W - M.l - M.r} height={H - M.t - M.b} className="dsp-ground" />
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
      <text x={sx(0)} y={H - M.b + 24} textAnchor="middle" className="dsp-axis-title">{kind === "tee" ? "Fairway centre" : "Line to the pin"} · yards left / right</text>
      <text transform={`translate(9 ${(M.t + H - M.b) / 2}) rotate(-90)`} textAnchor="middle" className="dsp-axis-title">{kind === "tee" ? "Distance travelled (yd)" : "Distance from where it was hit (yd)"}</text>
      {points.map((p, i) => {
        if (Math.abs(p.lat) > xMax || p.y < yLo || p.y > yHi) return null;
        return <circle key={i} cx={sx(p.lat)} cy={sy(p.y)} r={2} fill={BAND_COLORS[p.band % BAND_COLORS.length]} className="dsp-dot" />;
      })}
      {stats.map((s) => {
        const ref = referenceBand(reference, kind, s.band, lie);
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
      {kind === "app" && stats.map((s) => <path key={`p${s.band}`} d={`M${sx(0)} ${sy(s.meanY - s.meanAlong) - 5} l4 5 l-4 5 l-4 -5 z`} className="dsp-pin"><title>{`Pin (average ${Math.round(s.meanY - s.meanAlong)} yd away)`}</title></path>)}
      <text x={W - M.r} y={M.t + 11} textAnchor="end" className="dsp-note">{exaggeration > 1.05 ? `sideways scale ×${exaggeration.toFixed(1)}` : ""}</text>
    </svg>
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

export function PlayerDispersion({ dgId, name }: { dgId: number | null | undefined; name?: string }) {
  const index = useDashboardData<unknown>(INDEX_KEY);
  const idx = useMemo(() => parseIndex(index.data), [index.data]);
  const pKey = idx ? dispersionKey(idx, dgId) : null;
  const payloadLoad = useDashboardData<unknown>(pKey);
  const refLoad = useDashboardData<unknown>(pKey && idx ? idx.reference_key : null);
  const payload = useMemo(() => (dgId && payloadLoad.data ? parsePayload(payloadLoad.data, dgId) : null), [payloadLoad.data, dgId]);
  const reference = useMemo(() => parseReference(refLoad.data), [refLoad.data]);

  const [kind, setKind] = useState<Kind>("tee");
  const [rounds, setRounds] = useState<number>(50);
  const [lie, setLie] = useState("all");
  const [off, setOff] = useState<Record<string, boolean>>({});

  const bands = (kind === "tee" ? reference?.tee_bands : reference?.app_bands) ?? (kind === "tee" ? DEFAULT_TEE_BANDS : DEFAULT_APP_BANDS);
  const all = useMemo(() => (payload ? selectPoints(payload, kind, rounds, bands, kind === "app" ? lie : "all") : []), [payload, kind, rounds, bands, lie]);
  const hidden = (b: number) => off[`${kind}${b}`] === true;
  const points = useMemo(() => all.filter((p) => !off[`${kind}${p.band}`]), [all, off, kind]);
  const stats = useMemo(() => bandStats(points, kind, bands), [points, kind, bands]);
  const present = useMemo(() => bandStats(all, kind, bands).map((s) => s.band), [all, kind, bands]);

  if (index.loading) return null;
  if (index.error || !idx) return null; // not published yet: stay out of the way
  if (!pKey) return <Panel title="Shot dispersion" eyebrow="Driving-range view"><p className="pp-muted">No published shot-by-shot data for {name ?? "this golfer"} (PGA Tour rounds only).</p></Panel>;
  if (payloadLoad.loading || refLoad.loading) return <LoadingState label="Loading shot dispersion" />;
  if (payloadLoad.error || !payload) return <Panel title="Shot dispersion" eyebrow="Driving-range view"><p className="pp-muted">Shot dispersion did not load for {name ?? "this golfer"}.</p></Panel>;

  const available = roundCount(payload, kind);
  const span = windowSpan(payload, rounds);
  const hasKind = (k: Kind) => !!(k === "tee" ? payload.tee : payload.app);

  return (
    <Panel title="Shot dispersion" eyebrow="Driving-range view · PGA Tour shot data" className="dsp-panel">
      <div className="dsp-controls">
        <div className="segmented" role="group" aria-label="Shot type">
          {(["tee", "app"] as Kind[]).map((k) => <button key={k} type="button" className={kind === k ? "active" : ""} aria-pressed={kind === k} disabled={!hasKind(k)} onClick={() => setKind(k)}>{k === "tee" ? "Tee shots" : "Approach"}</button>)}
        </div>
        <div className="segmented" role="group" aria-label="Number of recent rounds">
          {ROUND_CHOICES.map((n) => <button key={n} type="button" className={rounds === n ? "active" : ""} aria-pressed={rounds === n} onClick={() => setRounds(n)}>Last {n}</button>)}
        </div>
        {kind === "app" && (
          <label className="dsp-lie">Starting lie
            <select value={lie} onChange={(e) => setLie(e.target.value)}>{LIE_OPTIONS.map((o) => <option key={o.key} value={o.key}>{o.label}</option>)}</select>
          </label>
        )}
      </div>
      <div className="dsp-bands" role="group" aria-label={kind === "tee" ? "Distance ranges to show" : "Starting yardages to show"}>
        {present.map((b) => (
          <button key={b} type="button" aria-pressed={!hidden(b)} className={hidden(b) ? "off" : ""} onClick={() => setOff((o) => ({ ...o, [`${kind}${b}`]: !hidden(b) }))}>
            <i style={{ background: BAND_COLORS[b % BAND_COLORS.length] }} />{bandLabel(kind, bands, b)}{kind === "app" ? " yd" : " yd"}
          </button>
        ))}
      </div>
      {span && <p className="dsp-sample">Last <b>{Math.min(rounds, payload.rounds.length)}</b> rounds with shot data{available < rounds ? ` (all ${available} with ${kind === "tee" ? "tee-shot" : "approach"} data)` : ""}, {shortDate(span.first)} to {shortDate(span.last)} · <b>{points.length.toLocaleString("en-US")}</b> shots shown</p>}
      {points.length === 0 ? <p className="pp-muted">No shots match this view.</p> : (
        <>
          <p className="dsp-headline">{headline(kind, stats, bands, reference, kind === "app" ? lie : "all")}</p>
          <div className="dsp-body">
            <figure className="dsp-figure">
              <RangeChart kind={kind} points={points} stats={stats} reference={reference} lie={kind === "app" ? lie : "all"} />
              <figcaption>
                <span><i className="dsp-key-bar" />Coloured bar: this golfer&apos;s middle 80% of shots</span>
                <span><i className="dsp-key-pga" />Grey bar: typical PGA Tour middle 80%</span>
                {kind === "app" && <span><i className="dsp-key-pin" />Diamond: where the pin was</span>}
              </figcaption>
            </figure>
            <BandTable kind={kind} stats={stats} bands={bands} reference={reference} lie={kind === "app" ? lie : "all"} />
          </div>
        </>
      )}
      <details className="pp-details">
        <summary>How to read this, and what it cannot tell you</summary>
        {kind === "tee" ? (
          <p>Every dot is one tee shot on a par 4 or 5, seen from above with the ball starting at the bottom. Left and right are measured from an estimated centre of the fairway: the line from the tee to where the field&apos;s fairway-finding drives landed that day on that hole. Heights are how far each shot travelled, grouped into the colours shown. This is not a measure of the target the golfer aimed at, and holes that bend are only approximated by a straight line.</p>
        ) : (
          <p>Every dot is one approach (including par-3 tee shots, 50 to 300 yards out), seen from above. The vertical line runs from the ball to the pin, so a shot that finishes on the line is straight at the flag. A dot above the diamond finished long, below it short. Golfers often aim away from the pin, so this shows spread around the pin line, not shot-making ability alone.</p>
        )}
        <p>The grey bars use every PGA Tour golfer&apos;s shots over the same period; the comparison words compare this golfer&apos;s side-to-side spread with the typical PGA golfer at the same distances. Greens and fairways hit are counted only for rounds from 2024, when the data provider&apos;s ground-type labels settled. DP World Tour rounds are not included.</p>
      </details>
    </Panel>
  );
}
