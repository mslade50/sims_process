"use client";

import { Pencil } from "lucide-react";
import { etTime } from "./lib";
import { compassLetters } from "./weather-overrides-rules";
import { hhmm, type HourlyPoint, type OwnerOverrideSummary } from "./weather-rules";
import "./weather-edits.css";

type Field = "wind" | "gust" | "temp" | "rain" | "dir";
const UNIT: Record<Field, string> = { wind: "mph", gust: "mph", temp: "°F", rain: "mm", dir: "°" };
const fixed = (v: number | null, digits = 1) => (v === null ? "n/a" : v.toFixed(digits));
const range = (lo: number | null, hi: number | null) => (lo !== null && hi !== null ? `${lo.toFixed(1)} to ${hi.toFixed(1)}` : "");

/** "Your edit: 12 mph; forecast 3.1 mph" for a cell that carries an owner edit, else undefined. */
export function editTitle(point: HourlyPoint, field: Field): string | undefined {
  const edit = point.owner_override?.[field];
  if (!edit) return undefined;
  const show = (v: number | null) => (v === null ? "not available" : field === "dir" ? `${Math.round(v)}° (${compassLetters(v)})` : `${Number(v.toFixed(2))} ${UNIT[field]}`);
  return `Your edit: ${show(edit.override)}; forecast ${show(edit.forecast)}`;
}

/** The "Includes your weather edits" note shown above the charts when the published forecast carries owner edits. */
export function OwnerEditNote({ summary, eventUid }: { summary: OwnerOverrideSummary | null; eventUid: string }) {
  if (!summary) return null;
  return (
    <p className="inputs-note accent wx-edit-note" role="status">
      <Pencil size={13} /> Includes your weather edits: {summary.count} {summary.count === 1 ? "hour" : "hours"}
      {summary.updated_at || summary.updated_by ? ` (saved ${summary.updated_at ? etTime(summary.updated_at) : "earlier"}${summary.updated_by ? ` by ${summary.updated_by}` : ""})` : ""}. The forecast values are in the tooltips.{" "}
      <a className="ex-link" href={`/weather-edits?e=${encodeURIComponent(eventUid)}`}>Change them</a>
    </p>
  );
}

function Cell({ point, field, children }: { point: HourlyPoint; field: Field; children: React.ReactNode }) {
  const title = editTitle(point, field);
  return <td className={title ? "wx-edited" : undefined} title={title}>{children}</td>;
}

/** Hour-by-hour numbers behind the charts for one round. Cells with an owner edit are highlighted. */
export function WeatherHourTable({ points }: { points: HourlyPoint[] }) {
  if (!points.length) return null;
  return (
    <div className="table-scroll wx-hour-table">
      <table>
        <thead>
          <tr>
            <th scope="col" className="wx-h">Hour</th>
            <th scope="col" className="wx-h" title="Average wind speed that hour, with the likely range (10th to 90th percentile) underneath">Wind mph</th>
            <th scope="col" className="wx-h" title="Gust speed that hour">Gusts</th>
            <th scope="col" className="wx-h" title="Where the wind comes from; the arrow points where it blows to">Direction</th>
            <th scope="col" className="wx-h" title="Temperature, with the likely range underneath">Temp °F</th>
            <th scope="col" className="wx-h" title="Rain in millimeters per hour">Rain mm</th>
            <th scope="col" className="wx-h" title="Chance that it rains at all that hour">Chance of rain</th>
          </tr>
        </thead>
        <tbody>
          {points.map((p) => (
            <tr key={`${p.round}-${p.time}`}>
              <td className="wx-hour">{hhmm(p.hour)}</td>
              <Cell point={p} field="wind">{fixed(p.wind)}{range(p.wind_p10, p.wind_p90) && <small>{range(p.wind_p10, p.wind_p90)}</small>}</Cell>
              <Cell point={p} field="gust">{fixed(p.gust)}</Cell>
              <Cell point={p} field="dir">
                {p.dir === null ? "n/a" : (
                  <span className="wx-dir"><span className="wx-dir-arrow" style={{ transform: `rotate(${p.dir + 180}deg)` }} aria-hidden="true">↑</span>{compassLetters(p.dir)}</span>
                )}
              </Cell>
              <Cell point={p} field="temp">{fixed(p.temp)}{range(p.temp_p10, p.temp_p90) && <small>{range(p.temp_p10, p.temp_p90)}</small>}</Cell>
              <Cell point={p} field="rain">{fixed(p.rain, 2)}</Cell>
              <td>{p.p_wet === null ? "n/a" : `${Math.round(p.p_wet * 100)}%`}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
