"use client";

import { useCallback, useEffect, useState } from "react";
import { EmptyState, LoadingState } from "./components";
import { Chip, OnThisPage, PageHeader, Section, type Tone } from "./ui";
import "./run.css";
import { etTime } from "./lib";
import { DataFreshnessPanel } from "./DataFreshnessPanel";
import { OddsSignalsPanel } from "./OddsSignalsPanel";
import { suggestedGroup } from "./shell-rules";
import { GROUPS, JOB_TYPES, isTerminal, machineLabel, machineRoleEvidence, machinesWithRoles, machineStatus, paramsSentence, parseUtc, requesterLabel, specFor, stateDetail, stateLabel, takeText, type JobParams, type JobRecord, type JobSpec, type JobStatus, type MachineBeat, type MachineStatus, type Roles } from "./jobs-rules";

type JobRow = JobRecord & { status: JobStatus | null; state: string };
type Heartbeat = MachineBeat & { version?: string };
type Target = "auto" | string;
type JobsResponse = { ok: boolean; now: string; jobs: JobRow[]; heartbeats: Heartbeat[]; roles?: Roles | null; roles_invalid?: boolean; roles_read_error?: boolean; error?: string };
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

function ago(iso: string | undefined, now: number): string {
  const ms = parseUtc(iso);
  if (ms === null) return "never";
  const s = Math.max(0, Math.round((now - ms) / 1000));
  if (s < 90) return `${s} s ago`;
  if (s < 5400) return `${Math.round(s / 60)} min ago`;
  if (s < 172800) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}

function duration(status: JobStatus | null, now: number): string {
  const start = parseUtc(status?.started_at ?? status?.claimed_at);
  if (start === null) return "";
  const end = parseUtc(status?.finished_at) ?? now;
  const s = Math.max(0, Math.round((end - start) / 1000));
  return s < 120 ? `${s} s` : s < 7200 ? `${Math.floor(s / 60)} min ${s % 60} s` : `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`;
}

/** Plain next step for the owner; the raw HTTP code goes to the console for developers, never into the page. */
const errorText = (result: { status: number; body: { error?: string } }) => {
  if (result.body.error && !/^The request failed \(code \d+\)\.?$/.test(result.body.error)) return result.body.error;
  if (typeof console !== "undefined") console.warn(`Dashboard request failed with code ${result.status}`);
  return "That did not go through. Try again in a minute.";
};

/** "Settle last week" for the job a machine reports it is running; "a job" when it cannot be matched. */
function runningName(id: string | null, jobs: JobRow[] | undefined): string {
  const job = id ? jobs?.find((item) => item.id === id) : undefined;
  return job ? (specFor(job.type)?.label ?? "a job") : "a job";
}

/** Which machine a job is on: the one that ran or claimed it, else the one it is addressed to, else "no machine yet" (any runner may take it). */
function jobMachine(job: JobRow): string {
  if (job.status?.machine) return machineLabel(job.status.machine);
  if (job.target_machine) return `${machineLabel(job.target_machine)} (requested)`;
  return "No machine yet";
}

const STATE_TONE: Record<string, Tone> = { done: "positive", failed: "negative", expired: "neutral", cancelled: "neutral", running: "accent", claimed: "accent", queued: "warning" };

/** Status as icon + word + colour, never colour alone. */
function StateIcon({ state }: { state: string }) {
  const common = { viewBox: "0 0 20 20", fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round" as const, strokeLinejoin: "round" as const, "aria-hidden": true, className: `run-icon ${state}` };
  if (state === "done") return <svg {...common}><circle cx="10" cy="10" r="7.5" /><path d="m6.8 10.2 2.2 2.2 4.2-4.6" /></svg>;
  if (state === "failed") return <svg {...common}><circle cx="10" cy="10" r="7.5" /><path d="m7.4 7.4 5.2 5.2M12.6 7.4l-5.2 5.2" /></svg>;
  if (state === "running" || state === "claimed") return <svg {...common}><circle cx="10" cy="10" r="7.5" opacity=".25" /><path d="M10 2.5a7.5 7.5 0 0 1 7.5 7.5" /></svg>;
  if (state === "queued") return <svg {...common}><circle cx="10" cy="10" r="7.5" /><path d="M10 6v4.2l2.8 1.6" /></svg>;
  return <svg {...common}><circle cx="10" cy="10" r="7.5" /><path d="m5.2 14.8 9.6-9.6" /></svg>;
}
const StatusChip = ({ state }: { state: string }) => <Chip tone={STATE_TONE[state] ?? "neutral"}><StateIcon state={state} />{stateLabel(state)}</Chip>;

/* ------------------------------------------------------------------ confirm sheet */
function ConfirmSheet({ spec, busy, target, took, onSubmit, onClose }: { spec: JobSpec; busy: boolean; target: Target; took: string; onSubmit: (params: JobParams) => void; onClose: () => void }) {
  const [round, setRound] = useState<"auto" | "1" | "2" | "3">("auto");
  const [noPull, setNoPull] = useState(false);
  const [supersede, setSupersede] = useState(false);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape" && !busy) onClose(); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [busy, onClose]);
  const params: JobParams = {
    ...(spec.after_round && round !== "auto" ? { after_round: Number(round) as 1 | 2 | 3 } : {}),
    ...(spec.no_pull && noPull ? { no_pull: true } : {}),
    ...(spec.supersede && supersede ? { supersede: true } : {}),
  };
  const optionCount = [spec.after_round, spec.no_pull, spec.supersede].filter(Boolean).length;
  return (
    <div className="run-sheet" role="dialog" aria-modal="true" aria-labelledby="run-sheet-title">
      <button className="run-sheet-backdrop" type="button" onClick={onClose} aria-label="Cancel" tabIndex={-1} />
      <div className="run-sheet-body">
        <span className="eyebrow">{spec.group}</span>
        <h2 id="run-sheet-title">{spec.label}</h2>
        <p>{spec.does}</p>
        <dl className="run-sheet-facts">
          <div><dt>Runs on</dt><dd>{target === "auto" ? "Automatic: the always-on desktop (the backup only if it is down)" : machineLabel(target)}</dd></div>
          <div><dt>Takes</dt><dd>{took}</dd></div>
          <div><dt>Press it</dt><dd>{spec.when}</dd></div>
        </dl>
        {optionCount > 0 && <p className="run-sheet-options">Options (the defaults are right for almost every run)</p>}
        {spec.after_round && (
          <label className="run-field">
            <span>Round just finished</span>
            <select value={round} onChange={(event) => setRound(event.target.value as typeof round)}>
              <option value="auto">Automatic (works it out)</option>
              <option value="1">After round 1</option>
              <option value="2">After round 2</option>
              <option value="3">After round 3</option>
            </select>
          </label>
        )}
        {spec.no_pull && (
          <label className="run-check">
            <input type="checkbox" checked={noPull} onChange={(event) => setNoPull(event.target.checked)} />
            <span>Skip the data download and use what the machine already has</span>
          </label>
        )}
        {spec.supersede && (
          <label className="run-check">
            <input type="checkbox" checked={supersede} onChange={(event) => setSupersede(event.target.checked)} />
            <span>Run again even if nothing has changed since the last run</span>
          </label>
        )}
        <div className="run-sheet-actions">
          <button type="button" className="inputs-button run-big" onClick={onClose} disabled={busy}>Cancel</button>
          <button type="button" className="inputs-button primary run-big" onClick={() => onSubmit(params)} disabled={busy} autoFocus // eslint-disable-line jsx-a11y/no-autofocus
>{busy ? "Sending…" : "Run it"}</button>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ the page */
export function RunView() {
  const [data, setData] = useState<JobsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const [pending, setPending] = useState<JobSpec | null>(null);
  const [busy, setBusy] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [showChecks, setShowChecks] = useState(false);
  const [target, setTarget] = useState<Target>("auto");
  const [notice, setNotice] = useState<{ tone: "ok" | "error"; text: string; problems?: string[] } | null>(null);

  const machines: MachineStatus[] = machinesWithRoles(data?.roles).map((spec) => machineStatus(spec, data?.heartbeats, now));
  const targetOffline = target !== "auto" && machines.find((m) => m.name === target)?.status === "offline";

  const reload = useCallback(async () => {
    const result = await api<JobsResponse>("/api/jobs");
    setNow(Date.now());
    if (result.ok) {
      setData(result.body);
      setError(null);
    } else {
      const text = errorText(result);
      setError(text.startsWith("That did not go through") ? "Recent runs are not available right now. Try again in a minute." : text);
    }
  }, []);

  useEffect(() => {
    queueMicrotask(() => void reload());
    const tick = () => {
      if (document.visibilityState === "visible") void reload();
    };
    const timer = window.setInterval(tick, 15_000);
    document.addEventListener("visibilitychange", tick);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", tick);
    };
  }, [reload]);

  // A one-second clock so a running job's elapsed time counts up and ages stay honest.
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const refreshNow = async () => { setRefreshing(true); await reload(); setRefreshing(false); };

  const submit = async (params: JobParams) => {
    if (!pending) return;
    const chosen = machines.find((m) => m.name === target && m.status !== "offline") ? target : "auto";   // never send a job to a machine that went offline since it was picked
    setBusy(true);
    const result = await api<{ job?: JobRecord }>("/api/jobs", { method: "POST", body: JSON.stringify({ type: pending.type, params, ...(chosen !== "auto" ? { target_machine: chosen } : {}) }) });
    setBusy(false);
    if (result.ok) {
      setNotice({ tone: "ok", text: `${pending.label} is queued${chosen === "auto" ? "" : ` for ${machineLabel(chosen)}`}. ${chosen === "auto" ? "The always-on desktop" : machineLabel(chosen)} picks it up within a minute or two. Follow it in the timeline below.` });
      setPending(null);
      void reload();
    } else {
      setNotice({ tone: "error", text: errorText(result), problems: result.body.problems });
      setPending(null);
    }
  };

  const cancel = async (id: string) => {
    const result = await api<unknown>(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST", body: "{}" });
    setNotice(result.ok ? { tone: "ok", text: "Canceled." } : { tone: "error", text: errorText(result) });
    void reload();
  };

  // The automatic 30-minute odds checks and hourly watches would bury everything else: hide the finished ones unless asked.
  const allJobs = [...(data?.jobs ?? [])].sort((a, b) => (parseUtc(b.requested_at) ?? 0) - (parseUtc(a.requested_at) ?? 0));
  const isAutoCheck = (job: JobRow) => job.requested_by === "cron" && (job.type === "odds_reprice" || job.type === "watch") && ["done", "queued", "expired", "cancelled"].includes(job.state);
  const visibleJobs = allJobs.filter((job) => showChecks || !isAutoCheck(job));
  const hiddenChecks = allJobs.length - visibleJobs.length;

  const suggested = suggestedGroup(now);
  const activeAll = allJobs.filter((job) => !isTerminal(job.state) && now - (parseUtc(job.requested_at) ?? 0) < 6 * 3_600_000);
  const busyTypes = new Set(activeAll.map((job) => job.type));
  const active = activeAll.filter((job) => !isAutoCheck(job));
  const autoWaiting = activeAll.length - active.length;
  const lastResult = allJobs.find((job) => isTerminal(job.state) && !isAutoCheck(job) && job.state !== "cancelled");
  const everyOffline = !!data && machines.length > 0 && machines.every((m) => m.status === "offline");
  /** Typical duration from real recent runs of the type; an explicitly labelled estimate when none is on record. */
  const tookFor = (type: string) => takeText(type, allJobs);

  const sections = [{ id: "now", label: "Right now" }, { id: "start", label: "Start a job" }, { id: "history", label: "Timeline" }, { id: "health", label: "Data health" }];

  return (
    <div className="run-page">
      <PageHeader title="Run" purpose="Start a pricing job from your phone. The always-on desktop checks the queue every minute, runs the job and posts the result in the timeline." actions={<button type="button" className="inputs-button" onClick={() => void refreshNow()} disabled={refreshing}>{refreshing ? "Refreshing…" : "Refresh"}</button>} />
      <OnThisPage sections={sections} />

      <Section id="now" title="Right now" purpose="What is running, the last result, and whether a machine can take work.">
        {notice && (
          <div className={`run-notice ${notice.tone === "error" ? "error" : "ok"}`} role="status">
            <span><strong>{notice.tone === "error" ? "Not started." : "Done."}</strong> {notice.text}</span>
            {notice.problems?.map((problem) => <span key={problem}>{problem}</span>)}
            <button type="button" className="inputs-button subtle" onClick={() => setNotice(null)}>Dismiss</button>
          </div>
        )}
        {!data ? (
          error ? <EmptyState tone="warning" title="The desktop runner status could not be loaded" detail="Jobs cannot be started from here right now. Try again in a minute." onRetry={() => void refreshNow()} /> : <LoadingState label="Loading machines" />
        ) : (
          <>
            {everyOffline && (
              <div className="run-notice error" role="alert"><span><strong>The desktop runner is offline{(() => { const seen = machines.map((m) => m.lastSeen).filter(Boolean).sort().at(-1); return seen ? ` (last seen ${etTime(seen)})` : ""; })()}.</strong> Jobs wait in the queue until a machine comes back online.</span></div>
            )}
            {(data.roles_invalid || data.roles_read_error) && <div className="run-notice error" role="alert"><span>Role configuration could not be verified. Built-in defaults are shown for reference; they do not establish runner ownership or backup readiness.</span></div>}
            <div className="run-now-grid">
              <div className="run-now-card">
                <h3>Running</h3>
                {active.length === 0 ? <p className="run-quiet">Nothing you started is running or waiting.</p> : (
                  <ul className="run-active">{active.map((job) => (
                    <li key={job.id}><StateIcon state={job.state} /><span><strong>{specFor(job.type)?.label ?? job.type}</strong>
                      <small>{stateDetail(job, job.state, data.heartbeats, now)}{duration(job.status, now) ? `, ${duration(job.status, now)} so far` : `, requested ${ago(job.requested_at, now)}`}</small>
                      <small className="run-job-line">Machine: {jobMachine(job)}</small></span></li>
                  ))}</ul>
                )}
                {autoWaiting > 0 && <p className="run-hint">{autoWaiting} automatic check{autoWaiting === 1 ? " is" : "s are"} waiting for a machine. They start by themselves once one is online.</p>}
              </div>
              <div className="run-now-card">
                <h3>Last result</h3>
                {lastResult ? (
                  <p className="run-last"><StatusChip state={lastResult.state} /><span><strong>{specFor(lastResult.type)?.label ?? lastResult.type}</strong><small>{ago(lastResult.status?.finished_at ?? lastResult.requested_at, now)}{duration(lastResult.status, now) ? `, took ${duration(lastResult.status, now)}` : ""}</small></span></p>
                ) : <p className="run-quiet">No finished run in the list yet.</p>}
              </div>
              <div className="run-now-card">
                <h3>Machines</h3>
                <ul className="run-machine-list">{machines.map((m) => (
                  <li key={m.name}><i aria-hidden="true" className={`run-dot ${m.status}`} /><span><strong>{m.label}</strong><small>{m.status === "offline" ? (m.lastSeen ? `Offline, last heard ${ago(m.lastSeen, now)}` : "Offline, has not reported in") : m.status === "running" ? `Running ${runningName(m.jobId, data.jobs)}` : `Online and idle, checked in ${ago(m.lastSeen ?? undefined, now)}`}</small></span></li>
                ))}</ul>
              </div>
            </div>
          </>
        )}
      </Section>

      <Section id="start" title="Start a job" purpose={`Suggested jobs for ${suggested === "During event" ? "an event day" : suggested} come first. Pressing one asks you to confirm before anything runs.`}>
        {data && (
          <fieldset className="run-target">
            <legend>Run on</legend>
            <div className="run-target-options" role="radiogroup" aria-label="Machine to run on">
              {/* eslint-disable-next-line jsx-a11y/label-has-associated-control -- radio input is nested */}
              <label className={`run-target-opt ${target === "auto" ? "selected" : ""}`}>
                <input type="radio" name="run-target" checked={target === "auto"} onChange={() => setTarget("auto")} />
                <span><strong>Automatic</strong><small>Always-on desktop; backup only if it is down</small></span>
              </label>
              {machines.map((m) => (
                // eslint-disable-next-line jsx-a11y/label-has-associated-control -- radio input is nested
                <label key={m.name} className={`run-target-opt ${m.status} ${target === m.name ? "selected" : ""}`}>
                  <input type="radio" name="run-target" checked={target === m.name} disabled={m.status === "offline"} onChange={() => setTarget(m.name)} />
                  <span><strong>{m.label}</strong><small title={machineRoleEvidence(m.name, data.roles ?? null, data.roles_read_error ? "roles_read_error" : data.roles_invalid ? "roles_invalid" : null, data.heartbeats, now).message}><i aria-hidden="true" className={`run-dot ${m.status}`} />{m.status === "offline" ? "Offline" : m.status === "running" ? "Busy" : "Online"}{m.role === "backup" ? ", backup" : ""}</small></span>
                </label>
              ))}
            </div>
            {targetOffline && <p className="run-hint">The selected machine went offline, so new jobs will use Automatic until it is back.</p>}
            {target !== "auto" && !targetOffline && machines.find((m) => m.name === target)?.role === "backup" && (
              <p className="run-hint">The backup only takes a job addressed to it when the main desktop is not busy and both run the same version of the code.</p>
            )}
          </fieldset>
        )}
        {[suggested, ...GROUPS.filter((group) => group !== suggested)].map((group) => {
          const specs = JOB_TYPES.filter((spec) => spec.group === group);
          const grid = (
            <div className="run-grid">
              {specs.map((spec) => {
                const waiting = busyTypes.has(spec.type);
                return (
                  <button type="button" key={spec.type} className="run-card" onClick={() => setPending(spec)} disabled={waiting || !data}>
                    <span className="run-card-head"><strong>{spec.label}</strong><Chip tone={waiting ? "warning" : "neutral"}>{waiting ? "Already queued or running" : tookFor(spec.type)}</Chip></span>
                    <span className="run-card-does">{spec.does}</span>
                    <small>{!data ? "Unavailable until the runner status loads" : <><b>When:</b> {spec.when}</>}</small>
                  </button>
                );
              })}
            </div>
          );
          const title = group === "During event" ? "During the event" : group === "Anytime" ? "Anytime tools" : group;
          return group === suggested ? (
            <div key={group} className="run-suggested">
              <h3><Chip tone="accent">Suggested now</Chip> {title}</h3>
              {grid}
            </div>
          ) : (
            <details key={group} className="run-collapsed"><summary>{title}<small>{specs.length} job{specs.length === 1 ? "" : "s"}</small></summary>{grid}</details>
          );
        })}
      </Section>

      <Section id="history" title="Timeline" purpose="Newest first. Each run shows its state, who asked, which machine ran it and how long it took."
        actions={<label className="run-check run-check-inline"><input type="checkbox" checked={showChecks} onChange={(event) => setShowChecks(event.target.checked)} /><span>Show automatic checks{hiddenChecks > 0 && !showChecks ? ` (${hiddenChecks} hidden)` : ""}</span></label>}>
        {error && <div className="run-notice error" role="alert"><span><strong>Could not load the recent runs.</strong> {error}</span></div>}
        {!data && !error && <LoadingState label="Loading recent runs" />}
        {data && data.jobs.length === 0 && <EmptyState title="Nothing has run yet" detail="Press one of the jobs above to start the first run." />}
        <ol className="run-timeline">
          {visibleJobs.map((job) => {
            const status = job.status;
            const spec = specFor(job.type);
            return (
              <li key={job.id} className={`run-job state-${job.state}`}>
                <span className="run-rail" aria-hidden="true"><StateIcon state={job.state} /></span>
                <div className="run-job-body">
                  <header>
                    <strong>{spec?.label ?? job.type}</strong>
                    <StatusChip state={job.state} />
                  </header>
                  {job.state === "queued" && <p className="run-summary run-waiting">{stateDetail(job, job.state, data?.heartbeats, now)}</p>}
                  <p className="run-when" title={ago(job.requested_at, now)}>
                    {etTime(job.requested_at)} by {requesterLabel(job.requested_by)}{paramsSentence(job.params) ? ` (${paramsSentence(job.params)})` : ""}
                  </p>
                  <div className="run-facts">
                    <span>Machine <b>{jobMachine(job)}</b></span>
                    <span>Status <b>{stateLabel(job.state)}</b></span>
                    {duration(status, now) && <span>{isTerminal(job.state) ? "Took" : "Running for"} <b className="num">{duration(status, now)}</b></span>}
                  </div>
                  {status?.summary && <p className="run-summary">{status.summary}</p>}
                  {job.state === "queued" && <button type="button" className="inputs-button danger" onClick={() => void cancel(job.id)}>Cancel this request</button>}
                  {(status?.log_tail || (status && status.state !== "claimed")) && (
                    <details className="tech-details">
                      <summary>Log and technical details</summary>
                      <p className="inputs-muted">Job {job.id} · {job.type}{status?.exit_code !== undefined && status?.exit_code !== null ? ` · exit code ${status.exit_code}` : ""}</p>
                      <pre className="run-log">{status?.log_tail || "The log is empty."}</pre>
                      <a href={`/api/jobs/${encodeURIComponent(job.id)}/log`} target="_blank" rel="noreferrer">Open the full log</a>
                    </details>
                  )}
                </div>
              </li>
            );
          })}
        </ol>
      </Section>

      <Section id="health" title="Data health" purpose="Is every data source current, and what the latest odds check found.">
        <DataFreshnessPanel />
        <OddsSignalsPanel />
      </Section>

      {data && (
        <details className="tech-details">
          <summary>Machine versions and names</summary>
          <ul className="inputs-muted">
            {machines.map((m) => <li key={m.name}>{m.label}: machine name {m.name}{m.commit ? `, code version ${m.commit}` : ""}{m.lastSeen ? `, last heard from ${etTime(m.lastSeen)}` : ""}</li>)}
          </ul>
        </details>
      )}

      {pending && <ConfirmSheet spec={pending} busy={busy} target={targetOffline ? "auto" : target} took={tookFor(pending.type)} onSubmit={(params) => void submit(params)} onClose={() => setPending(null)} />}
    </div>
  );
}
