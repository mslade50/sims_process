"use client";

import { Area, Bar, CartesianGrid, ComposedChart, Line, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { editTitle } from "./WeatherHourTable";
import { hhmm, waveName, type HourlyPoint, type Wave } from "./weather-rules";

const WIND = "var(--wx-wind)";
const GUST = "var(--wx-gust)";
const TEMP = "var(--wx-temp)";
const RAIN = "var(--wx-rain)";
const WAVE_SHADES = ["var(--wave-am-fill)", "var(--wave-pm-fill)"];

function Arrow({ cx, cy, dir }: { cx?: number; cy?: number; dir: number | null }) {
  if (cx === undefined || cy === undefined || dir === null) return <g />;
  // meteorological direction is where the wind comes FROM; the arrow points where it blows TO
  return (
    <g transform={`translate(${cx},${cy}) rotate(${dir + 180})`}>
      <path d="M0,-14 L6,5 L0,1 L-6,5 Z" fill={WIND} stroke="var(--background)" strokeWidth={1} />
    </g>
  );
}

const compass = (deg: number) => ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][Math.round((((deg % 360) + 360) % 360) / 45) % 8];

type Row = HourlyPoint & { windBand: [number, number] | null; tempBand: [number, number] | null };
const band = (lo: number | null, hi: number | null): [number, number] | null => (lo !== null && hi !== null ? [lo, hi] : null);

function WxTooltip({ active, payload, unit }: { active?: boolean; payload?: Array<{ payload: Row }>; unit: "wind" | "temp" | "rain" }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="wx-tooltip">
      <strong>{hhmm(p.hour)}</strong>
      {unit === "wind" && (
        <>
          <span>Wind {p.wind?.toFixed(1) ?? "n/a"} mph{p.wind_p10 !== null && p.wind_p90 !== null ? ` (likely range ${p.wind_p10.toFixed(1)} to ${p.wind_p90.toFixed(1)})` : ""}</span>
          {p.gust !== null && <span>Gusts {p.gust.toFixed(1)} mph</span>}
          {p.dir !== null && <span>Blowing from the {compass(p.dir)}</span>}
        </>
      )}
      {unit === "temp" && <span>Temperature {p.temp?.toFixed(1) ?? "n/a"}°F{p.temp_p10 !== null && p.temp_p90 !== null ? ` (likely range ${p.temp_p10.toFixed(1)} to ${p.temp_p90.toFixed(1)})` : ""}</span>}
      {unit === "rain" && <span>Rain {p.rain?.toFixed(2) ?? "n/a"} mm</span>}
      {(unit === "wind" ? (["wind", "gust", "dir"] as const) : unit === "temp" ? (["temp"] as const) : (["rain"] as const)).map((f) => { const t = editTitle(p, f); return t ? <span key={f} className="wx-edited-tip">{t}</span> : null; })}
    </div>
  );
}

export function HourlyCharts({ points, waves }: { points: HourlyPoint[]; waves: Wave[] }) {
  const rows: Row[] = points.map((p) => ({ ...p, windBand: band(p.wind_p10, p.wind_p90), tempBand: band(p.temp_p10, p.temp_p90) }));
  const hours = rows.map((r) => r.hour);
  const domain: [number, number] = [Math.floor(Math.min(...hours)), Math.ceil(Math.max(...hours))];
  const windows = waves.filter((w) => w.window);
  const shading = windows.map((w, i) => (
    <ReferenceArea key={`${w.wave}-${i}`} x1={w.window![0]} x2={w.window![1]} fill={WAVE_SHADES[i % 2]} stroke="none" ifOverflow="hidden"
      label={{ value: waveName(w.wave, i), position: i % 2 ? "insideTopRight" : "insideTopLeft", fill: "var(--muted)", fontSize: 10 }} />
  ));
  const axis = <XAxis type="number" dataKey="hour" domain={domain} allowDecimals={false} padding={{ left: 10 }} stroke="var(--muted)" tickFormatter={(v) => `${String(v).padStart(2, "0")}:00`} />;
  const margin = { top: 14, right: 12, bottom: 4, left: 0 };
  const hasWind = rows.some((r) => r.wind !== null);
  const hasTemp = rows.some((r) => r.temp !== null);
  const hasRain = rows.some((r) => r.rain !== null);
  return (
    <div className="wx-charts">
      {hasWind && (
        <div>
          <h3 className="inputs-h3">Wind and gusts (mph). The shaded band is the likely range; arrows point where the wind blows.</h3>
          <div className="wx-chart" role="img" aria-label="Hourly wind forecast with uncertainty band and tee windows">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={rows} margin={margin}>
                <CartesianGrid stroke="var(--line)" vertical={false} />
                {axis}
                <YAxis stroke="var(--muted)" width={34} />
                <Tooltip content={<WxTooltip unit="wind" />} />
                {shading}
                <Area dataKey="windBand" stroke="none" fill={WIND} fillOpacity={0.2} isAnimationActive={false} />
                <Line dataKey="gust" stroke={GUST} strokeDasharray="4 3" dot={false} strokeWidth={1.5} isAnimationActive={false} connectNulls />
                <Line dataKey="wind" stroke={WIND} strokeWidth={2.5} isAnimationActive={false} connectNulls
                  dot={(props: { cx?: number; cy?: number; payload?: Row; index?: number }) => <Arrow key={props.index} cx={props.cx} cy={props.cy} dir={props.payload?.dir ?? null} />} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
      {hasTemp && (
        <div>
          <h3 className="inputs-h3">Temperature (°F). The shaded band is the likely range.</h3>
          <div className="wx-chart short" role="img" aria-label="Hourly temperature forecast with uncertainty band">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={rows} margin={margin}>
                <CartesianGrid stroke="var(--line)" vertical={false} />
                {axis}
                <YAxis stroke="var(--muted)" width={34} domain={["auto", "auto"]} />
                <Tooltip content={<WxTooltip unit="temp" />} />
                {shading}
                <Area dataKey="tempBand" stroke="none" fill={TEMP} fillOpacity={0.2} isAnimationActive={false} />
                <Line dataKey="temp" stroke={TEMP} strokeWidth={2.5} dot={false} isAnimationActive={false} connectNulls />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
      {hasRain && (
        <div>
          <h3 className="inputs-h3">Rain (mm per hour)</h3>
          <div className="wx-chart short" role="img" aria-label="Hourly rain forecast">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={rows} margin={margin}>
                <CartesianGrid stroke="var(--line)" vertical={false} />
                {axis}
                <YAxis stroke="var(--muted)" width={34} />
                <Tooltip content={<WxTooltip unit="rain" />} />
                {shading}
                <Bar dataKey="rain" fill={RAIN} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}

/** Expected strokes vs the field with its p10-p90 band, drawn on one shared symmetric scale. */
