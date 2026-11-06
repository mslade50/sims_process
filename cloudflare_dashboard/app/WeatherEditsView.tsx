"use client";

import { useEffect, useMemo, useState } from "react";
import { useEventParam } from "./event-param";
import { EmptyState, LoadingState, Panel, SegmentedControl } from "./components";
import { PageHeader, Section as UiSection } from "./ui";
import { ChangeTray, type TrayChange, type TrayStage } from "./ChangeTray";
import { isTerminal, stateLabel } from "./jobs-rules";
import { useDashboardData } from "./data";
import { etTime } from "./lib";
import { weatherKey } from "./WeatherEffectsView";
import { BOUNDS, FIELDS, MAX_NOTE, checkHourFields, compassLetters, completedRounds, eventSafe, eventStarted, liveStatus, repriceForMode, validateWeatherOverride, type Phase, type RunLike, type Section, type WeatherField, type WeatherOverrideDocV2 } from "./weather-overrides-rules";
import { hhmm, parseWeather, type HourlyPoint, type WeatherDoc } from "./weather-rules";
import "./inputs.css";
import "./weather-edits.css";

type IndexEvent = { event_uid: string; name: string; tour?: string; date_start?: string; weather_key?: string | null; weather_brief_key?: string | null; distribution_runs?: RunLike[] | null };
type Drafts = Record<string, Partial<Record<WeatherField, string>>>;
type Loaded = { state: "loading" } | { state: "error"; message: string } | { state: "ready" };
type Applied = { id: string; state: string } | null;
type Notice = { tone: "ok" | "error"; text: string; link?: { href: string; label: string } } | null;

/** Column order on the page: wind, gusts, direction, temperature, rain. */
const UNITS: Record<WeatherField, string> = { wind: "mph", gust: "mph", dir: "°", temp: "°F", rain: "mm" };
const NAMES: Record<WeatherField, string> = { wind: "Wind", gust: "Gusts", dir: "Direction", temp: "Temp", rain: "Rain" };
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

function draftsFromSaved(doc: Section | null): Drafts {
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

type SavedState = { doc: WeatherOverrideDocV2 | null; etag: string | null };

function EditorBody({ event, doc }: { event: IndexEvent; doc: WeatherDoc }) {
  const safe = eventSafe(event.event_uid);
  const [now] = useState(() => Date.now());
  const [loaded, setLoaded] = useState<Loaded>({ state: "loading" });
  const [saved, setSaved] = useState<SavedState>({ doc: null, etag: null });
  const completed = completedRounds(event.distribution_runs);
  const started = eventStarted(event.date_start, event.distribution_runs, now);
  const [mode, setMode] = useState<Phase>(completed >= 1 ? "live" : "pre");

  useEffect(() => {
    let active = true;
    api<{ doc?: WeatherOverrideDocV2 | null; etag?: string | null }>(`/api/weather-overrides/${encodeURIComponent(safe)}`).then((result) => {
      if (!active) return;
      if (!result.ok) { setLoaded({ state: "error", message: result.body.error ?? "The saved edits could not be loaded." }); return; }
      setSaved({ doc: result.body.doc ?? null, etag: result.body.etag ?? null });
      setLoaded({ state: "ready" });
    });
    return () => { active = false; };
  }, [safe]);

  return (
    <div className="wx-body">
      <details className="wx-how">
        <summary>How weather edits work</summary>
        <ul className="wx-edit-help">
          <li>An edit replaces the forecast for that hour only. Leave a box blank to keep the forecast.</li>
          <li>The weather model still adds its usual uncertainty around your number, so a single edit does not make the hour certain.</li>
          <li>Prices change only after a reprice finishes (about 12 minutes). Saving alone changes nothing.</li>
          <li>Edits stay in place for later runs until you clear them (in-event edits also end by themselves). The original forecast is kept beside each edit.</li>
        </ul>
      </details>
      <div className="wx-kind">
        <span className="wx-kind-label">Kind of edit</span>
      <SegmentedControl label="Kind of edit" value={mode} onChange={setMode} options={[{ value: "pre", label: "Before the event" }, { value: "live", label: "During the event" }]} />
      </div>
      {loaded.state === "loading" && <LoadingState label="Loading your saved edits" />}
      {loaded.state === "error" && <div className="inputs-banner warn" role="alert"><strong>Your saved edits could not be loaded.</strong>{loaded.message} Saving is turned off until they load, so nothing can be overwritten by mistake. Reload the page to try again.</div>}
      <SectionEditor key={`${mode}-${completed}-${loaded.state}`} event={event} doc={doc} mode={mode} completed={completed} started={started} now={now} saved={saved} setSaved={setSaved} ready={loaded.state === "ready"} />
    </div>
  );
}

function SectionEditor({ event, doc, mode, completed, started, now, saved, setSaved, ready }: { event: IndexEvent; doc: WeatherDoc; mode: Phase; completed: number; started: boolean; now: number; saved: SavedState; setSaved: (s: SavedState) => void; ready: boolean }) {
  const safe = eventSafe(event.event_uid);
  const live = mode === "live";
  const status = liveStatus(saved.doc?.live ?? null, completed);
  const section: Section | null = live ? status.prefill : saved.doc?.pre ?? null;
  // Keyed on the load state in the parent, so the boxes fill from the saved edits once they arrive.
  const [drafts, setDrafts] = useState<Drafts>(() => draftsFromSaved(section));
  const [note, setNote] = useState(section?.note ?? "");
  const [roundPick, setRound] = useState<number | null>(null);
  const [busy, setBusy] = useState<"save" | "clear" | "reprice" | null>(null);
  const [notice, setNotice] = useState<Notice>(null);
  const [confirmClear, setConfirmClear] = useState(false);
  const [applied, setApplied] = useState<Applied>(null);

  const rounds = useMemo(() => [...new Set(doc.forecast.hourly.map((h) => h.round))].filter((r) => !live || r > completed).sort((a, b) => a - b), [doc, live, completed]);
  const round = roundPick !== null && rounds.includes(roundPick) ? roundPick : rounds[0];
  const points = doc.forecast.hourly.filter((h) => h.round === round);
  const timeOf = (p: HourlyPoint) => (p.time.length >= 13 ? `${p.time.slice(0, 13)}:00` : p.time);

  const checked = useMemo(() => validateWeatherOverride(bodyFrom(drafts, note)), [drafts, note]);
  const savedChecked = useMemo(() => validateWeatherOverride({ hours: section?.hours ?? [], note: section?.note ?? "" }), [section]);
  const dirty = JSON.stringify([checked.hours, checked.note]) !== JSON.stringify([savedChecked.hours, savedChecked.note]) || checked.problems.length > 0;
  const editedCount = checked.hours.length;
  const savedCount = savedChecked.hours.length;
  const known = new Set(doc.forecast.hourly.map(timeOf));
  const orphaned = checked.hours.filter((h) => !known.has(h.time_local)).length;
  const choice = repriceForMode(mode, doc.teeCoverage, event.date_start, event.distribution_runs, now);
  const preDead = !live && started;
  const locked = live && !status.open;

  const setField = (time: string, field: WeatherField, value: string) => {
    setNotice(null);
    setDrafts((current) => ({ ...current, [time]: { ...(current[time] ?? {}), [field]: value } }));
  };

  const write = async (hours: unknown[], text: string, kind: "save" | "clear") => {
    setBusy(kind);
    setNotice(null);
    const result = await api<{ doc?: WeatherOverrideDocV2; etag?: string | null }>(`/api/weather-overrides/${encodeURIComponent(safe)}`, { method: "PUT", body: JSON.stringify({ phase: mode, hours, note: text, etag: saved.etag, ...(live ? { after_round: completed } : {}) }) });
    setBusy(null);
    if (result.ok && result.body.doc) {
      const next = result.body.doc;
      setSaved({ doc: next, etag: result.body.etag ?? null });
      const sec = live ? next.live : next.pre;
      setDrafts(draftsFromSaved(sec));
      setNote(sec?.note ?? "");
      setApplied(null);
      const n = sec?.hours.length ?? 0;
      const tail = preDead ? " The event has started, so pre-tournament edits have no effect on prices now." : n ? " Prices have not changed yet: press Reprice with these edits." : " Prices have not changed yet: press Reprice to price on the plain forecast.";
      setNotice({ tone: "ok", text: (n ? `Saved ${n} edited ${n === 1 ? "hour" : "hours"}.` : "All edits cleared.") + tail });
    } else {
      setNotice({ tone: "error", text: result.status === 409 ? `${result.body.error ?? "The saved edits changed."} Nothing was saved.` : [result.body.error ?? "The edits could not be saved.", ...(result.body.problems ?? []).slice(1, 4)].join(" ") });
    }
  };

  const save = () => { if (!checked.problems.length) void write(checked.hours, checked.note, "save"); };
  const clearAll = () => { setConfirmClear(false); void write([], "", "clear"); };
  const reprice = async () => {
    if (!choice) return;
    setBusy("reprice");
    setNotice(null);
    const result = await api<{ job?: { id: string } }>("/api/jobs", { method: "POST", body: JSON.stringify({ type: choice.type, params: choice.params }) });
    setBusy(null);
    if (result.ok && result.body.job) {
      setApplied({ id: result.body.job.id, state: "queued" });
      setNotice({ tone: "ok", text: `Reprice queued (job ${result.body.job.id}). Prices change only when it finishes; this page tracks it.`, link: { href: "/run", label: "Watch it on the Run page" } });
    }
    else setNotice({ tone: "error", text: [result.body.error ?? "The reprice could not be queued.", ...(result.body.problems ?? [])].join(" ") });
  };

  // Track the queued reprice so "applied" is a fact about a finished run, never a guess.
  useEffect(() => {
    if (!applied || isTerminal(applied.state)) return;
    const jobId = applied.id;
    const timer = window.setInterval(async () => {
      const result = await api<{ jobs?: Array<{ id: string; state: string }> }>("/api/jobs");
      const found = result.body.jobs?.find((j) => j.id === jobId);
      if (!found) return;
      setApplied((current) => (current && current.id === jobId && current.state !== found.state ? { id: jobId, state: found.state } : current));
      if (found.state === "done") setNotice({ tone: "ok", text: "Reprice finished. Prices now use your saved edits." });
      else if (found.state === "failed" || found.state === "expired" || found.state === "cancelled") setNotice({ tone: "error", text: `The reprice ${stateLabel(found.state).toLowerCase()}. Prices did not change; your edits are still saved.`, link: { href: "/run", label: "See why on the Run page" } });
    }, 10_000);
    return () => window.clearInterval(timer);
  }, [applied]);

  const sec = live ? saved.doc?.live : saved.doc?.pre;
  const stamp = sec && sec.hours.length ? `Last saved ${etTime(sec.updated_at)}${sec.updated_by ? ` by ${sec.updated_by}` : ""}.` : "Nothing has been saved for this kind of edit yet.";
  const hasEdit = (r: number) => checked.hours.some((h) => doc.forecast.hourly.some((p) => p.round === r && timeOf(p) === h.time_local));
  const off = !ready || busy !== null || locked;
  const changes: TrayChange[] = [];
  for (const h of checked.hours) {
    const p = doc.forecast.hourly.find((x) => timeOf(x) === h.time_local);
    if (!p) continue;
    for (const f of FIELDS) {
      const to = h[f];
      if (typeof to !== "number") continue;
      const from = forecastValue(p, f);
      if (from !== null && Math.abs(from - to) < 0.05) continue;
      changes.push({ key: `${h.time_local}-${f}`, what: `R${p.round} ${hhmm(p.hour)} ${NAMES[f]}`, from: from === null ? "n/a" : show(f, from), to: show(f, to), unit: f === "dir" ? undefined : UNITS[f] });
    }
  }

  if (locked) {
    return <Panel eyebrow="During the event" title="In-event edits"><EmptyState title="In-event edits open after round 1 is complete" detail="Before then, use Before the event. Once a round finishes, you can edit the hours still to play and reprice the in-play prices." /></Panel>;
  }
  const appliedStage: TrayStage = !applied
    ? { label: "Applied to prices", tone: "neutral", value: "Not applied", detail: "Prices change only after a reprice finishes (about 12 minutes). Saving alone changes nothing." }
    : applied.state === "done" ? { label: "Applied to prices", tone: "positive", value: "Applied", detail: "The reprice finished. Prices use your saved edits." }
    : applied.state === "failed" || applied.state === "expired" || applied.state === "cancelled" ? { label: "Applied to prices", tone: "negative", value: `Reprice ${stateLabel(applied.state).toLowerCase()}`, detail: "Prices did not change. Your edits are still saved." }
    : { label: "Applied to prices", tone: "warning", value: applied.state === "queued" ? "Queued, waiting for a runner" : "Repricing", detail: `Job ${applied.id}. Prices change when it finishes.` };
  const stages: TrayStage[] = [
    { label: "Edited here", tone: dirty ? "warning" : "neutral", value: dirty ? "Unsaved" : editedCount ? "Matches saved" : "None", detail: `${editedCount} ${editedCount === 1 ? "hour" : "hours"} on this page${checked.problems.length ? ", with problems to fix" : ""}.` },
    { label: "Saved", tone: savedCount ? "accent" : "neutral", value: savedCount ? `${savedCount} ${savedCount === 1 ? "hour" : "hours"}` : "None", detail: `${stamp} Saving does not change prices.` },
    appliedStage,
  ];
  const eligibility = !choice ? { ok: false, text: "Pre-tournament edits no longer affect prices once the event has started." }
    : dirty ? { ok: false, text: "Save your changes first, so the reprice uses what you see." }
    : { ok: true, text: `Runs as ${choice.type === "after_round" ? "After a round" : choice.type === "wednesday" ? "Wednesday reprice" : "Tuesday refresh"}: ${choice.reason}` };
  return (
    <>
      {live ? (
        <p className="inputs-note accent">These edits apply to in-play prices after round {completed} (tonight and during round {completed + 1}) and are cleared automatically once round {completed + 1} is complete.</p>
      ) : (
        <p className="inputs-note">These edits change only the pricing runs before the event starts.</p>
      )}
      {live && status.expiredFrom !== null && <div className="inputs-banner" role="status"><strong>Your in-event edits from after round {status.expiredFrom} have expired.</strong>They no longer affect prices. The boxes below start empty.</div>}
      {preDead && <div className="inputs-banner warn" role="status"><strong>The event has started.</strong>Pre-tournament edits no longer affect any prices. You can still save or clear them, but Reprice is turned off here. Use During the event for in-play weather.</div>}
      {orphaned > 0 && <p className="inputs-note accent">{orphaned} saved {orphaned === 1 ? "edit is" : "edits are"} for hours that are not in the current forecast. They are kept and saved with the rest.</p>}

      <UiSection id="wx-hours" title={round ? `Round ${round} hours` : "Hours"} purpose="Type your number over any hour. Blank keeps the forecast; an edited hour is highlighted and shows how far it moves from the forecast." actions={<>
        {rounds.length > 1 && <SegmentedControl label="Round" value={String(round)} onChange={(x) => setRound(Number(x))} options={rounds.map((r) => ({ value: String(r), label: `R${r}${hasEdit(r) ? " •" : ""}` }))} />}
        <a className="inputs-button" href={`/weather-effects?e=${encodeURIComponent(event.event_uid)}`} title="See the charts with your saved edits applied">See the effect on Weather</a>
      </>}>
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
                  <tr key={time} className={Object.values(row).some((v) => (v ?? "").trim() !== "") ? "wx-row-edited" : undefined}>
                    <td className="wx-hour">{hhmm(p.hour)}</td>
                    {COLUMNS.map((c) => {
                      const fc = forecastValue(p, c.field);
                      const value = row[c.field] ?? "";
                      const problem = problems[c.field];
                      const typed = value.trim() === "" ? NaN : Number(value);
                      const diff = c.field !== "dir" && fc !== null && Number.isFinite(typed) ? typed - fc : null;
                      return (
                        <td key={c.field} className={value.trim() !== "" ? "wx-edited" : undefined}>
                          <span className="wx-lab">{c.head}</span>
                          <span className="wx-fc" title="The published forecast, before any edit">Forecast {show(c.field, fc)}</span>
                          <input
                            className={`wx-input${problem ? " bad" : ""}`} type="text" inputMode="decimal" autoComplete="off" disabled={off}
                            value={value} placeholder={fc === null ? "" : String(Number(fc.toFixed(1)))} onChange={(e) => setField(time, c.field, e.target.value)}
                            aria-label={`${c.head} at ${hhmm(p.hour)}`} aria-invalid={problem ? true : undefined} title={c.hint}
                          />
                          {diff !== null && Math.abs(diff) >= 0.05 && !problem && <small className="wx-diff">{diff > 0 ? "+" : "−"}{Number(Math.abs(diff).toFixed(1))} vs forecast</small>}
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
        {points.length === 0 && <EmptyState title="No hourly forecast for this round" detail={live ? "There are no hours left to play in the published forecast." : "This round is too far away for the forecast to reach yet."} />}
        <p className="inputs-muted">Allowed values: wind {BOUNDS.wind.lo} to {BOUNDS.wind.hi} mph, gusts {BOUNDS.gust.lo} to {BOUNDS.gust.hi} mph (not below the wind), temperature {BOUNDS.temp.lo} to {BOUNDS.temp.hi} °F, rain {BOUNDS.rain.lo} to {BOUNDS.rain.hi} mm per hour, direction {BOUNDS.dir.lo} to {BOUNDS.dir.hi}°.</p>
      </UiSection>

      <UiSection id="wx-save" title="Note and confirm" purpose="An optional note is stored with the saved edits.">
        <label className="wx-note-field">
          <span>Note for the record (optional)</span>
          <textarea value={note} maxLength={MAX_NOTE} rows={2} disabled={off} onChange={(e) => { setNotice(null); setNote(e.target.value); }} placeholder="For example: local radar shows the front arriving two hours early" />
          <small>{note.length} of {MAX_NOTE} characters</small>
        </label>
        {checked.problems.length > 0 && <div className="inputs-banner warn" role="alert"><strong>Fix these before saving</strong>{checked.problems.slice(0, 4).join(" ")}</div>}
        {confirmClear && (
          <div className="inputs-banner warn" role="alertdialog" aria-label="Confirm clearing edits">
            <strong>Clear every {live ? "in-event" : "pre-tournament"} edit for this event?</strong>The model goes back to the plain forecast for these hours on the next run. The other kind of edit is not touched.
            <span className="wx-actions"><button type="button" className="inputs-button danger" onClick={clearAll}>Yes, clear them</button><button type="button" className="inputs-button" onClick={() => setConfirmClear(false)}>Keep them</button></span>
          </div>
        )}
      </UiSection>

      <ChangeTray changes={changes} stages={stages} eligibility={eligibility} announce={notice} nothing="No hour differs from the forecast yet. Type over any hour above.">
        <button type="button" className="inputs-button primary" onClick={save} disabled={off || !dirty || checked.problems.length > 0} title="Save your numbers. This does not change prices.">{busy === "save" ? "Saving…" : "Save edits"}</button>
        <button type="button" className="inputs-button" onClick={() => void reprice()} disabled={off || dirty || !choice} title={!choice ? "Pre-tournament edits no longer affect prices" : dirty ? "Save your changes first, so the reprice uses what you see" : "Queues a pricing run that re-runs even if nothing else changed"}>{busy === "reprice" ? "Queuing…" : "Reprice with these edits"}</button>
        <button type="button" className="inputs-button danger" onClick={() => setConfirmClear(true)} disabled={off || confirmClear || (savedCount === 0 && editedCount === 0)} title={live ? "Remove the in-event edits for this event" : "Remove the pre-tournament edits for this event"}>{busy === "clear" ? "Clearing…" : "Clear all edits"}</button>
      </ChangeTray>
    </>
  );
}

export function WeatherEditsView() {
  const { data: index, loading } = useDashboardData<{ events?: IndexEvent[] }>("golfprice/index.json");
  const events = useMemo(() => (index?.events ?? []).filter((e) => e.weather_key || e.weather_brief_key), [index]);
  const barEvent = useEventParam();
  const pick = barEvent; // chosen in the top bar only
  const event = events.find((e) => e.event_uid === pick) ?? events[0];
  const { data, loading: docLoading, error } = useDashboardData<unknown>(event ? weatherKey(event) : null);
  const doc = useMemo(() => parseWeather(data), [data]);

  const intro = (
    <PageHeader
      title="Weather edits"
      purpose="Replace the forecast for chosen hours with your own numbers, save, then reprice. Use it when you know something the forecast does not."
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
