"use client";

import { useEffect, useRef, useState } from "react";
import { createShotMonitor, initialMonitorState, type MonitorState } from "./shot-monitor-client";
import { collectorStatus, publicationStatus, sourceFreshness, SOURCES, TOURS, TOUR_LABELS, timestamp, type Collector, type Metric, type Publication, type Source } from "./shot-monitor-rules";
import "./shot-monitor.css";
import { checkedAge, tourSummary } from "./shot-monitor-summary";
import { tourMatrix } from "./shot-monitor-matrix";

function Time({ at }: { at?: string | null }) {
  if (!at) return <span>Not reported</span>;
  if (!Number.isFinite(timestamp(at))) return <span>Invalid time: {at}</span>;
  return <time dateTime={at} title={at}>{new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit", second: "2-digit", timeZoneName: "short", timeZone: "America/New_York" }).format(new Date(at))}</time>;
}
function Metrics({ values }: { values?: Metric[] }) {
  if (!values?.length) return <p className="sm-muted">Captured / cleaned counts not reported.</p>;
  return <dl className="sm-metrics">{values.map((m, i) => <div key={`${m.label}-${i}`}><dt>{m.label}</dt><dd>{typeof m.value === "number" ? new Intl.NumberFormat("en-US").format(m.value) : m.value}</dd></div>)}</dl>;
}
function CollectorRow({ row, source, now }: { row: Collector; source: Source; now: number }) {
  const state = collectorStatus(row, source, now);
  return <article className="sm-row">
    <div className="sm-row-heading"><h4>{row.label}</h4><span className={`sm-state ${state === "error" ? "sm-bad" : ["running", "idle"].includes(state) ? "" : "sm-warn"}`}>{state === "error" ? "Needs attention" : state}</span></div>
    <p>{row.reason || "No reason reported."}</p>
    <Metrics values={row.metrics} />
    <dl className="sm-times">
      <div><dt>Latest check</dt><dd><Time at={row.observed_at} /></dd></div>
      <div><dt>Last capture / success</dt><dd><Time at={row.last_success_at} /></dd></div>
      <div><dt>Heartbeat</dt><dd><Time at={row.heartbeat_at} /></dd></div>
      <div><dt>Next run</dt><dd><Time at={row.next_run_at} /></dd></div>
      <div><dt>Schedule</dt><dd>{row.schedule || "Not reported"}</dd></div>
    </dl>
    {!!row.issues?.length && <ul className="sm-issues">{row.issues.map((issue, i) => <li key={i}>{issue}</li>)}</ul>}
  </article>;
}
function PublicationRow({ row, source, now }: { row: Publication; source: Source; now: number }) {
  const state = publicationStatus(row, source, now);
  return <article className="sm-row">
    <div className="sm-row-heading"><h4>{row.label}</h4><span className={`sm-state ${state === "Ready" ? "sm-good" : "sm-warn"}`}>{state}</span></div>
    <p>{row.validation.summary || "No validation summary reported."}</p>
    <dl className="sm-times">
      <div><dt>Published</dt><dd><Time at={row.published_at} /></dd></div>
      <div><dt>Generation</dt><dd>{row.generation || "Not reported"}</dd></div>
    </dl>
    <details><summary>Counts, validation limits and manifest</summary>
      {row.metrics?.length ? <Metrics values={row.metrics} /> : <p className="sm-muted">Published counts not reported.</p>}
      {!!row.validation.limits?.length && <ul className="sm-issues">{row.validation.limits.map((limit, i) => <li key={i}>{limit}</li>)}</ul>}
      <dl className="sm-times">
      <div><dt>Validation checked</dt><dd><Time at={row.validation.checked_at} /></dd></div>
      <div><dt>Manifest key</dt><dd>{row.manifest_key || "Not reported"}</dd></div>
      <div><dt>Published SHA-256</dt><dd className="sm-hash">{row.manifest_sha256 || "Not reported"}</dd></div>
      <div><dt>Validated SHA-256</dt><dd className="sm-hash">{row.validation.manifest_sha256 || "Not reported"}</dd></div>
    </dl>{row.notes?.map((note, i) => <p key={i}>{note}</p>)}</details>
  </article>;
}

/** Pure rendering boundary, also used by fixture tests. No fixture or packaged data in the live view. */
export function ShotDataContent({ sources, refreshing, now, onRefresh }: MonitorState & { now: number; onRefresh?: () => void }) {
  const collectors = sources.find(s => s.file === "collectors-desktop.json") ?? { file: "collectors-desktop.json" as const, status: "missing" as const };
  const publications = sources.find(s => s.file === "publications.json") ?? { file: "publications.json" as const, status: "missing" as const };
  const datasets = publications.data?.publications ?? [];
  const summaries = TOURS.map(tour => tourSummary(tour, sources, now));
  const matrix = TOURS.map(tour => tourMatrix(tour, sources, now));
  const attention = summaries.filter(s => s.attention);
  const loading = sources.length > 0 && sources.every(s => s.status === "loading");
  return <div className="shot-monitor">
    <header className="sm-intro page-intro">
      <div role="status" aria-live="polite" aria-atomic="true"><span className="eyebrow">Shot data</span><h1>{loading ? "Checking shot data…" : attention.length ? `${attention.length} ${attention.length === 1 ? "tour needs" : "tours need"} attention` : "Tour checks are clear"}</h1>
      <p className="sm-start">{loading ? "Waiting for the latest update." : attention.length ? `Start with ${TOUR_LABELS[attention[0].tour]}: see Next action below.` : "No action needed. Paused collection stays paused."}</p></div>
      <div className="sm-check"><span>{checkedAge(sources, now)}</span>{onRefresh && <button className="sm-refresh" type="button" disabled={refreshing} onClick={onRefresh}>{refreshing ? "Checking…" : "Refresh"}</button>}</div>
    </header>
    <section className="sm-matrix-wrap" aria-labelledby="sm-matrix-title">
      <h2 id="sm-matrix-title" className="sm-matrix-title">Tour status</h2>
      <p className="sm-muted">Collect = a collector is running. Check and Publish = output validated. New York time.</p>
      <table className="sm-matrix">
        <thead><tr><th scope="col">Tour</th><th scope="col">Collect <span className="sm-sub">machine</span></th><th scope="col">Check <span className="sm-sub">evidence</span></th><th scope="col">Publish <span className="sm-sub">evidence</span></th><th scope="col">Known / unknown</th><th scope="col">Next action</th></tr></thead>
        <tbody>{matrix.map(row => <tr key={row.tour} id={`sm-${row.tour}`} className={row.attention ? "sm-attn" : undefined}>
          <th scope="row" data-label="Tour">{TOUR_LABELS[row.tour]}<small>{row.badge}</small></th>
          {([["Collect", row.collect], ["Check", row.check], ["Publish", row.publish]] as const).map(([label, cell]) => <td key={label} data-label={label}><div><span className={`sm-cell sm-cell-${cell.tone}`}>{cell.state}</span><small><Time at={cell.at}/></small></div></td>)}
          <td data-label="Known / unknown"><div>{row.reason}<small>{row.available}</small></div></td>
          <td data-label="Next action"><div>{row.next}</div></td>
        </tr>)}</tbody>
      </table>
    </section>
    <div className="sm-board">{summaries.map(summary => <details className="sm-tour sm-details" key={summary.tour} aria-labelledby={`sm-d-${summary.tour}`}>
      <summary id={`sm-d-${summary.tour}`}>{TOUR_LABELS[summary.tour]} details<span className="sm-sub">collectors and published datasets</span></summary>
        <div className="sm-tour-grid">
          <div className="sm-stage"><h3>Captured and cleaned</h3>
            {(collectors.data?.collectors ?? []).filter(r => r.tour === summary.tour).map(row => <CollectorRow key={row.id} row={row} source={collectors} now={now}/>)}
            {!collectors.data?.collectors?.some(r => r.tour === summary.tour) && <p className="sm-empty">No collector reported. Capture and cleaning status unknown.</p>}
          </div>
          <div className="sm-stage"><h3>Published datasets</h3><p className="sm-muted">Ready requires a fresh monitor snapshot and validation of the published manifest. Publication age is shown separately.</p>
            {datasets.filter(r => r.tour === summary.tour).map(row => <PublicationRow key={row.id} row={row} source={publications} now={now}/>)}
            {!datasets.some(r => r.tour === summary.tour) && <p className="sm-empty">No publication reported. Dataset readiness unknown.</p>}
          </div>
        </div>
    </details>)}</div>
    <details className="sm-sources"><summary>Monitor diagnostics</summary>
      {sources.map((source) => <div className="sm-source" key={source.file}>
        <div className="sm-row-heading"><h3>{SOURCES.find((s) => s.file === source.file)!.label}</h3><span className="sm-state">{source.status === "loading" ? "Loading" : !source.data ? source.status : sourceFreshness(source, now) === "fresh" ? "Fresh snapshot" : sourceFreshness(source, now) === "stale" ? "STALE" : "Invalid check time"}</span></div>
        {source.error && <p className="sm-warn">{source.error} {source.data ? "Showing previous copy — STALE." : "No snapshot available."}</p>}
        <dl className="sm-times"><div><dt>Observed</dt><dd><Time at={source.data?.observed_at} /></dd></div><div><dt>Page last checked</dt><dd><Time at={source.checkedAt ? new Date(source.checkedAt).toISOString() : null} /></dd></div><div><dt>Host</dt><dd>{source.data?.host || "Not reported"}</dd></div><div><dt>File</dt><dd>{source.file}</dd></div></dl>
        {source.data?.notes?.map((note, i) => <p key={i}>{note}</p>)}
      </div>)}
      <p className="sm-muted">Scheduled snapshots update about every 5 minutes; this page polls every 30 seconds. Times are New York time. Collector snapshot is stale after 15 minutes; publication snapshot after 30 minutes. Check times more than 5 minutes in the future are invalid. A failed refresh marks retained data STALE immediately.</p>
    </details>
  </div>;
}

export function ShotDataView() {
  const [state, setState] = useState(initialMonitorState);
  const [now, setNow] = useState(() => Date.now());
  const monitor = useRef<ReturnType<typeof createShotMonitor> | null>(null);
  useEffect(() => {
    const client = createShotMonitor(setState);
    monitor.current = client;
    void client.refresh();
    const clock = setInterval(() => setNow(Date.now()), 1000);
    return () => { client.stop(); monitor.current = null; clearInterval(clock); };
  }, []);
  return <ShotDataContent {...state} now={now} onRefresh={() => { void monitor.current?.refresh(); }} />;
}
