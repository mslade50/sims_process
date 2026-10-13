"use client";

import { useEffect, useMemo, useState } from "react";
import { EmptyState, Kpi, LoadingState, PageIntro, Panel, SegmentedControl } from "./components";
import { useDashboardData } from "./data";
import { etTime } from "./lib";
import { weatherKey } from "./WeatherEffectsView";
import { BOUNDS, FIELDS, MAX_NOTE, checkHourFields, chooseRepriceJob, compassLetters, eventSafe, validateWeatherOverride, type WeatherField, type WeatherOverrideDoc } from "./weather-overrides-rules";
import { hhmm, parseWeather, type HourlyPoint, type WeatherDoc } from "./weather-rules";
import "./weather-edits.css";

type IndexEvent = { event_uid: string; name: string; tour?: string; date_start?: string; weather_key?: string | null; weather_brief_key?: string | null };
type Drafts = Record<string, Partial<Record<WeatherField, string>>>;
type Loaded = { state: "loading" } | { state: "error"; message: string } | { state: "ready" };
type Notice = { tone: "ok" | "error"; text: string; link?: { href: string; label: string } } | null;

const tourWord = (tour: string) => ({ pga: "PGA Tour", euro: "DP World Tour" } as Record<string, string>)[tour.toLowerCase()] ?? tour.toUpperCase();
/** Column order on the page: wind, gusts, direction, temperature, rain. */
const COLUMNS: Array<{ field: WeatherField; head: string; hint: string }> = [
  { field: "wind", head: "Wind mph", hint: "Average wind speed for the hour (0 to 60 mph)" },
  { field: "gust", head: "Gusts mph", hint: "Gust speed for the hour (0 to 90 mph). Cannot be lower than the wind." },
  { field: "dir", head: "Direction", hint: "Degrees the wind comes FROM (0 or 360 = north, 90 = east, 180 = south, 270 = west)" },
  { field: "temp", head: "Temp °F", hint: "Air temperature for the hour (-10 to 120 °F)" },
  { field: "rain", head: "Rain mm", hint: "Rain in millimeters per hour (0 to 50)" },
];

/** What the published forecast said before any edit of yours (the top-level value already includes an edit). */
function forecastValue(point: HourlyPoint, field: WeatherField): number | null {
  const edit = point.owner_override?.[field];
  return edit ? edit.forecast : point[field];
}
const show = (field: WeatherField, v: number | null) => (v === null ? "n/a" : field === "dir" ? `${compassLetters(v)} ${Math.round(v)}°` : String(Number(v.toFixed(1))));

function draftsFromSaved(doc: WeatherOverrideDoc | null): Drafts {
  const out: Drafts = {};
  for (const h of doc?.hours ?? []) {
    const row: Partial<Record<WeatherField, string>> = {};
    for (const f of FIELDS) if (typeof h[f] === "number") row[f] = String(h[f]);
    if (Object.keys(row).length) out[h.time_local] = row;
  }
  return out;
}

function bodyFrom(drafts: Drafts, note: string) {
  const hours = Object.entries(drafts).map(([time_local, row]) => {
    const entry: Record<string, unknown> = { time_local };
    for (const f of FIELDS) { const raw = (row[f] ?? "").trim(); entry[f] = raw === "" ? null : Number(raw); }
    return entry;
  });
  return { hours, note };
}

type ApiResult<T> = { ok: boolean; status: number; body: T & { error?: string; problems?: string[] } };
async function api<T>(url: string, init?: RequestInit): Promise<ApiResult<T>> {
  try {
    const response = await fetch(url, { ...init, headers: { accept: "application/json", ...(init?.body ? { "content-type": "application/json" } : {}) } });
    let body = {} as T & { error?: string; problems?: string[] };
    try {
      body = await response.json();
    } catch {
      body = { error: response.status === 401 || response.status === 403 ? "You are not signed in. Reload the page to sign in again." : `The request failed (code ${response.status}).` } as T & { error?: string };
    }
    return { ok: response.ok, status: response.status, body };
  } catch {
    return { ok: false, status: 0, body: { error: "Could not reach the dashboard (offline?)" } as T & { error?: string } };
  }
}

function EditorBody({ event, doc }: { event: IndexEvent; doc: WeatherDoc }) {
  const safe = eventSafe(event.event_uid);
  const [loaded, setLoaded] = useState<Loaded>({ state: "loading" });
  const [drafts, setDrafts] = useState<Drafts>({});
  const [note, setNote] = useState("");
  const [saved, setSaved] = useState<{ doc: WeatherOverrideDoc | null; etag: string | null }>({ doc: null, etag: null });
  const [roundPick, setRound] = useState<number | null>(null);
  const [busy, setBusy] = useState<"save" | "clear" | "reprice" | null>(null);
  const [notice, setNotice] = useState<Notice>(null);
  const [now] = useState(() => Date.now());

  useEffect(() => {
    let active = true;
    api<{ doc?: WeatherOverrideDoc | null; etag?: string | null }>(`/api/weather-overrides/${encodeURIComponent(safe)}`).then((result) => {
      if (!active) return;
      if (!result.ok) { setLoaded({ state: "error", message: result.body.error ?? "The saved edits could not be loaded." }); return; }
      const savedDoc = result.body.doc ?? null;
      setSaved({ doc: savedDoc, etag: result.body.etag ?? null });
      setDrafts(draftsFromSaved(savedDoc));
      setNote(savedDoc?.note ?? "");
      setLoaded({ state: "ready" });
    });
    return () => { active = false; };
  }, [safe]);

  const rounds = useMemo(() => [...new Set(doc.forecast.hourly.map((h) => h.round))].sort((a, b) => a - b), [doc]);
  const round = roundPick !== null && rounds.includes(roundPick) ? roundPick : rounds[0];
  const points = doc.forecast.hourly.filter((h) => h.round === round);
  const timeOf = (p: HourlyPoint) => (p.time.length >= 13 ? `${p.time.slice(0, 13)}:00` : p.time);

  const checked = useMemo(() => validateWeatherOverride(bodyFrom(drafts, note)), [drafts, note]);
  const savedChecked = useMemo(() => validateWeatherOverride({ hours: saved.doc?.hours ?? [], note: saved.doc?.note ?? "" }), [saved]);
  const dirty = JSON.stringify([checked.hours, checked.note]) !== JSON.stringify([savedChecked.hours, savedChecked.note]) || checked.problems.length > 0;
  const editedCount = checked.hours.length;
  const savedCount = savedChecked.hours.length;
  const known = new Set(doc.forecast.hourly.map(timeOf));
  const orphaned = checked.hours.filter((h) => !known.has(h.time_local)).length;
  const ready = loaded.state === "ready";
  const choice = chooseRepriceJob(doc.teeCoverage, event.date_start, now);

  const setField = (time: string, field: WeatherField, value: string) => {
    setNotice(null);
    setDrafts((current) => ({ ...current, [time]: { ...(current[time] ?? {}), [field]: value } }));
  };

  const write = async (hours: unknown[], text: string, kind: "save" | "clear") => {
    setBusy(kind);
    setNotice(null);
    const result = await api<{ doc?: WeatherOverrideDoc; etag?: string | null }>(`/api/weather-overrides/${encodeURIComponent(safe)}`, { method: "PUT", body: JSON.stringify({ hours, note: text, etag: saved.etag }) });
    setBusy(null);
    if (result.ok && result.body.doc) {
      setSaved({ doc: result.body.doc, etag: result.body.etag ?? null });
      setDrafts(draftsFromSaved(result.body.doc));
      setNote(result.body.doc.note);
      const n = result.body.doc.hours.length;
      setNotice({ tone: "ok", text: n ? `Saved ${n} edited ${n === 1 ? "hour" : "hours"}. Prices have not changed yet: press Reprice with these edits.` : "All edits cleared. Prices have not changed yet: press Reprice to price on the plain forecast." });
    } else {
      setNotice({ tone: "error", text: result.status === 409 ? `${result.body.error ?? "The saved edits changed."} Nothing was saved.` : [result.body.error ?? "The edits could not be saved.", ...(result.body.problems ?? []).slice(1, 4)].join(" ") });
    }
  };

  const save = () => { if (!checked.problems.length) void write(checked.hours, checked.note, "save"); };
  const clearAll = () => {
    if (!window.confirm("Clear every saved weather edit for this event? The model goes back to the plain forecast on the next run.")) return;
    void write([], "", "clear");
  };
  const reprice = async () => {
    setBusy("reprice");
    setNotice(null);
    const result = await api<{ job?: { id: string } }>("/api/jobs", { method: "POST", body: JSON.stringify({ type: choice.type, params: choice.params }) });
    setBusy(null);
    if (result.ok && result.body.job) setNotice({ tone: "ok", text: `Reprice queued (job ${result.body.job.id}). It takes about 12 minutes; prices change only when it finishes.`, link: { href: "/run", label: "Watch it on the Run page" } });
    else setNotice({ tone: "error", text: [result.body.error ?? "The reprice could not be queued.", ...(result.body.problems ?? [])].join(" ") });
  };

  const stamp = saved.doc ? `Last saved ${etTime(saved.doc.updated_at)}${saved.doc.updated_by ? ` by ${saved.doc.updated_by}` : ""}.` : "Nothing has been saved for this event yet.";
  const hasEdit = (r: number) => checked.hours.some((h) => doc.forecast.hourly.some((p) => p.round === r && timeOf(p) === h.time_local));
  return (
    <div className="wx-body">
      <Panel eyebrow="How this works" title="Replace the forecast for chosen hours">
        <ul className="wx-edit-help">
          <li>An edit replaces the forecast for that hour only. Leave a box blank to keep the forecast.</li>
          <li>The weather model still adds its usual uncertainty around your number, so a single edit does not make the hour certain.</li>
          <li>Prices change only after a reprice finishes (about 12 minutes). Saving alone changes nothing.</li>
          <li>Edits stay in place for later runs until you clear them. The original forecast is kept beside each edit.</li>
        </ul>
        <div className="kpi-grid">
          <Kpi label="Saved edits" value={String(savedCount)} detail={stamp} hint="Hours with at least one number of yours, as last saved." tone={savedCount ? "accent" : "neutral"} />
          <Kpi label="On this page" value={String(editedCount)} detail={dirty ? "Unsaved changes" : "Matches what is saved"} hint="Hours with at least one number on this page right now." tone={dirty ? "negative" : "neutral"} />
          <Kpi label="Reprice runs as" value={choice.type === "after_round" ? "After a round" : choice.type === "wednesday" ? "Wednesday reprice" : "Tuesday refresh"} detail={choice.reason} hint="Chosen from the event start date and how many tee times are known." />
        </div>
      </Panel>

      {loaded.state === "loading" && <LoadingState label="Loading your saved edits" />}
      {loaded.state === "error" && <div className="inputs-banner warn" role="alert"><strong>Your saved edits could not be loaded.</strong>{loaded.message} Saving is turned off until they load, so nothing can be overwritten by mistake. Reload the page to try again.</div>}
      {orphaned > 0 && <p className="inputs-note accent">{orphaned} saved {orphaned === 1 ? "edit is" : "edits are"} for hours that are not in the current forecast. They are kept and saved with the rest.</p>}

      {rounds.length > 1 && <SegmentedControl label="Round" value={String(round)} onChange={(x) => setRound(Number(x))} options={rounds.map((r) => ({ value: String(r), label: `R${r}${hasEdit(r) ? " •" : ""}` }))} />}

      <Panel eyebrow={round ? `Round ${round}` : undefined} title="Hours" actions={<a className="inputs-button" href={`/weather-effects?e=${encodeURIComponent(event.event_uid)}`} title="See the charts with your saved edits applied">Weather page</a>}>
        <div className="table-scroll wx-edit-table">
          <table>
            <thead>
              <tr>
                <th scope="col" className="wx-h">Hour</th>
                {COLUMNS.map((c) => <th scope="col" className="wx-h" key={c.field} title={c.hint}>{c.head}</th>)}
              </tr>
            </thead>
            <tbody>
              {points.map((p) => {
                const time = timeOf(p);
                const row = drafts[time] ?? {};
                const problems = checkHourFields(row);
                return (
                  <tr key={time}>
                    <td className="wx-hour">{hhmm(p.hour)}</td>
                    {COLUMNS.map((c) => {
                      const fc = forecastValue(p, c.field);
                      const value = row[c.field] ?? "";
                      const problem = problems[c.field];
                      return (
                        <td key={c.field} className={value.trim() !== "" ? "wx-edited" : undefined}>
                          <span className="wx-fc" title="The published forecast, before any edit">Forecast {show(c.field, fc)}</span>
                          <input
                            className={`wx-input${problem ? " bad" : ""}`} type="text" inputMode="decimal" autoComplete="off" disabled={!ready || busy !== null}
                            value={value} placeholder={fc === null ? "" : String(Number(fc.toFixed(1)))} onChange={(e) => setField(time, c.field, e.target.value)}
                            aria-label={`${c.head} at ${hhmm(p.hour)}`} aria-invalid={problem ? true : undefined} title={c.hint}
                          />
                          {problem && <small className="wx-problem" role="alert">{problem}</small>}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {points.length === 0 && <EmptyState title="No hourly forecast for this round" detail="This round is too far away for the forecast to reach yet." />}
      </Panel>

      <Panel eyebrow="Save and reprice" title="Note and actions">
        <label className="wx-note-field">
          <span>Note for the record (optional)</span>
          <textarea value={note} maxLength={MAX_NOTE} rows={2} disabled={!ready || busy !== null} onChange={(e) => { setNotice(null); setNote(e.target.value); }} placeholder="For example: local radar shows the front arriving two hours early" />
          <small>{note.length} of {MAX_NOTE} characters</small>
        </label>
        {checked.problems.length > 0 && <div className="inputs-banner warn" role="alert"><strong>Fix these before saving</strong>{checked.problems.slice(0, 4).join(" ")}</div>}
        <div className="wx-actions">
          <button type="button" className="inputs-button primary" onClick={save} disabled={!ready || busy !== null || !dirty || checked.problems.length > 0} title="Save your numbers. This does not change prices.">{busy === "save" ? "Saving…" : "Save edits"}</button>
          <button type="button" className="inputs-button danger" onClick={clearAll} disabled={!ready || busy !== null || (savedCount === 0 && editedCount === 0)} title="Remove every edit for this event">{busy === "clear" ? "Clearing…" : "Clear all edits"}</button>
          <button type="button" className="inputs-button" onClick={() => void reprice()} disabled={!ready || busy !== null || dirty} title={dirty ? "Save your changes first, so the reprice uses what you see" : "Queues a pricing run that re-runs even if nothing else changed"}>{busy === "reprice" ? "Queuing…" : "Reprice with these edits"}</button>
        </div>
        {dirty && ready && <p className="inputs-muted">You have unsaved changes. Save them before repricing.</p>}
        {notice && (
          <div className={`inputs-banner ${notice.tone === "error" ? "warn" : ""}`} role="status">
            <strong>{notice.tone === "error" ? "That did not work" : "Done"}</strong>{notice.text}
            {notice.link && <a className="ex-link" href={notice.link.href}>{notice.link.label}</a>}
          </div>
        )}
      </Panel>
      <p className="inputs-muted">Allowed values: wind {BOUNDS.wind.lo} to {BOUNDS.wind.hi} mph, gusts {BOUNDS.gust.lo} to {BOUNDS.gust.hi} mph (not below the wind), temperature {BOUNDS.temp.lo} to {BOUNDS.temp.hi} °F, rain {BOUNDS.rain.lo} to {BOUNDS.rain.hi} mm per hour, direction {BOUNDS.dir.lo} to {BOUNDS.dir.hi}°.</p>
    </div>
  );
}

export function WeatherEditsView() {
  const { data: index, loading } = useDashboardData<{ events?: IndexEvent[] }>("golfprice/index.json");
  const events = useMemo(() => (index?.events ?? []).filter((e) => e.weather_key || e.weather_brief_key), [index]);
  const [pick, setPick] = useState("");
  useEffect(() => {
    queueMicrotask(() => setPick((current) => current || new URLSearchParams(window.location.search).get("e") || ""));
  }, []);
  const event = events.find((e) => e.event_uid === pick) ?? events[0];
  const { data, loading: docLoading, error } = useDashboardData<unknown>(event ? weatherKey(event) : null);
  const doc = useMemo(() => parseWeather(data), [data]);

  const intro = (
    <PageIntro
      eyebrow="Your call on the weather"
      title="Weather edits"
      description="Replace the forecast for chosen hours with your own numbers, save them, then reprice. Useful when you know something the forecast does not."
      controls={events.length > 0 ? (
        <label className="select-control">
          <span>Event</span>
          <select value={event?.event_uid ?? ""} onChange={(e) => setPick(e.target.value)}>
            {events.map((e) => <option key={e.event_uid} value={e.event_uid}>{e.name}{e.tour ? ` (${tourWord(e.tour)})` : ""}</option>)}
          </select>
        </label>
      ) : undefined}
    />
  );
  if (loading) return <div>{intro}<LoadingState label="Loading published events" /></div>;
  if (!event) return <div>{intro}<EmptyState title="No weather is published yet" detail="This page lists events once a run has been published with a forecast." /></div>;
  if (docLoading) return <div>{intro}<LoadingState label="Loading the forecast" /></div>;
  if (!doc || doc.forecast.hourly.length === 0) {
    return <div>{intro}<EmptyState title={`No hourly forecast for ${event.name}`} detail={error ? "This run has no weather forecast yet. Check back after the next run." : "There are no forecast hours to edit yet."} /></div>;
  }
  return <div>{intro}<EditorBody key={event.event_uid} event={event} doc={doc} /></div>;
}
