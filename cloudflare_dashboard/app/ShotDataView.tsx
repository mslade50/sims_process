"use client";

import { useEffect, useRef, useState } from "react";
import { createShotMonitor, initialMonitorState, type MonitorState } from "./shot-monitor-client";
import { collectorStatus, publicationStatus, sourceFreshness, SOURCES, TOURS, TOUR_LABELS, timestamp, type Collector, type Metric, type Publication, type Source } from "./shot-monitor-rules";
import "./shot-monitor.css";
import { checkedAge, tourSummary, type Stage } from "./shot-monitor-summary";

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

function StageIcon({ stage }: { stage: string }) {
  return <svg viewBox="0 0 40 40" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {stage === "Collect" ? <><path d="M8 31h24M11 27l15-15 5 5-15 15M25 13l4-4 5 5-4 4"/><circle cx="10" cy="18" r="3"/><path d="M9 9c8-4 14-3 19 0" strokeDasharray="2 4"/></> : stage === "Check" ? <><path d="M13 7H8v26h24V7h-5M15 5h10v6H15z"/><path d="m13 22 5 5 10-12"/></> : <><ellipse cx="20" cy="10" rx="13" ry="5"/><path d="M7 10v18c0 7 26 7 26 0V10M7 19c0 7 26 7 26 0"/><path d="m16 27 3 3 6-6"/></>}
  </svg>;
}
function GolfIllustration({ tour }: { tour: string }) {
  const paths: Record<string, string> = { pga: "M38 82C72 20 120 22 157 60S219 82 252 34", dpwt: "M38 82C69 32 87 94 133 65S208 13 252 34", kft: "M38 82C75 59 106 21 152 45S225 81 252 34" };
  return <svg className="sm-golf" viewBox="0 0 300 110" role="img" aria-label="Illustrative golf hole and shot path; not venue geometry">
    <path d="M12 88C7 55 37 15 101 28S171 3 218 14 298 48 285 80 209 105 163 93 22 119 12 88Z" fill="var(--surface-soft)" stroke="var(--line-strong)"/>
    <ellipse cx="247" cy="47" rx="32" ry="19" fill="var(--surface-raised)" stroke="var(--line-strong)"/>
    <path d={paths[tour]} fill="none" stroke="var(--muted)" strokeWidth="2" strokeDasharray="4 5"/>
    <ellipse cx="252" cy="48" rx="6" ry="2.5" fill="var(--foreground)"/><path d="M252 48V12l23 8-23 8" fill="none" stroke="var(--foreground)" strokeWidth="2" strokeLinejoin="round"/>
    <circle cx="38" cy="82" r="6" fill="var(--foreground)"/><circle cx="36" cy="80" r="1" fill="var(--surface)"/>
  </svg>;
}
function StageRail({ stages }: { stages: Stage[] }) {
  return <ol className="sm-rail" aria-label="Pipeline stages">{stages.map(stage => <li key={stage.label} className={`sm-rail-${stage.tone}`}><StageIcon stage={stage.label}/><strong>{stage.label}</strong><span>{stage.state}</span></li>)}</ol>;
}

/** Pure rendering boundary, also used by fixture tests. No fixture or packaged data in the live view. */
export function ShotDataContent({ sources, refreshing, now, onRefresh }: MonitorState & { now: number; onRefresh?: () => void }) {
  const collectors = sources.find(s => s.file === "collectors-desktop.json") ?? { file: "collectors-desktop.json" as const, status: "missing" as const };
  const publications = sources.find(s => s.file === "publications.json") ?? { file: "publications.json" as const, status: "missing" as const };
  const datasets = publications.data?.publications ?? [];
  const summaries = TOURS.map(tour => tourSummary(tour, sources, now));
  const attention = summaries.filter(s => s.attention);
  const loading = sources.length > 0 && sources.every(s => s.status === "loading");
  return <div className="shot-monitor">
    <header className="sm-intro">
      <div role="status" aria-live="polite" aria-atomic="true"><p className="sm-muted">Shot data</p><h1>{loading ? "Checking shot data…" : attention.length ? `${attention.length} ${attention.length === 1 ? "tour needs" : "tours need"} attention` : "Tour checks are clear"}</h1></div>
      <p className="sm-start">{loading ? "Waiting for the latest update." : attention.length ? `Start with ${TOUR_LABELS[attention[0].tour]}. ${attention[0].next}` : "No action needed. Paused collection stays paused."}</p>
      <div className="sm-check"><span>{checkedAge(sources, now)}</span>{onRefresh && <button className="sm-refresh" type="button" disabled={refreshing} onClick={onRefresh}>{refreshing ? "Checking…" : "Refresh"}</button>}</div>
    </header>
    <div className="sm-board">{summaries.map(summary => <section className="sm-tour" key={summary.tour} aria-labelledby={`sm-${summary.tour}`}>
      <div className="sm-row-heading"><h2 id={`sm-${summary.tour}`}>{TOUR_LABELS[summary.tour]}</h2><span className={`sm-badge ${summary.attention ? "sm-warn" : "sm-muted"}`}>{summary.badge}</span></div>
      <p className="sm-headline">{summary.headline}</p>
      <GolfIllustration tour={summary.tour}/>
      <StageRail stages={summary.stages}/>
      <p className="sm-available"><strong>Data available</strong>{summary.available}</p>
      {summary.next && <p className="sm-next"><strong>Next step</strong>{summary.next}</p>}
      <details className="sm-details"><summary>View details<span className="sm-sr-only"> for {TOUR_LABELS[summary.tour]}</span></summary>
        <div className="sm-tour-grid">
          <div className="sm-stage"><h3>Captured and cleaned</h3><p className="sm-muted">Captured data does not imply cleaned or published data.</p>
            {(collectors.data?.collectors ?? []).filter(r => r.tour === summary.tour).map(row => <CollectorRow key={row.id} row={row} source={collectors} now={now}/>)}
            {!collectors.data?.collectors?.some(r => r.tour === summary.tour) && <p className="sm-empty">No collector reported. Capture and cleaning status unknown.</p>}
          </div>
          <div className="sm-stage"><h3>Published datasets</h3><p className="sm-muted">Ready requires a fresh monitor snapshot and validation of the published manifest. Publication age is shown separately.</p>
            {datasets.filter(r => r.tour === summary.tour).map(row => <PublicationRow key={row.id} row={row} source={publications} now={now}/>)}
            {!datasets.some(r => r.tour === summary.tour) && <p className="sm-empty">No publication reported. Dataset readiness unknown.</p>}
          </div>
        </div>
      </details>
    </section>)}</div>
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
