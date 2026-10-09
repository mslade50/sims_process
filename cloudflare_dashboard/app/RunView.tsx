"use client";

import { useCallback, useEffect, useState } from "react";
import { EmptyState, LoadingState, PageIntro, Panel } from "./components";
import { etTime } from "./lib";
import { OddsSignalsPanel } from "./OddsSignalsPanel";
import { suggestedGroup } from "./shell-rules";
import { GROUPS, JOB_TYPES, MACHINES, isTerminal, machineLabel, machineStatus, paramsSentence, parseUtc, requesterLabel, specFor, stateLabel, waitingFor, type JobParams, type JobRecord, type JobSpec, type JobStatus, type MachineBeat, type MachineStatus } from "./jobs-rules";

type JobRow = JobRecord & { status: JobStatus | null; state: string };
type Heartbeat = MachineBeat & { version?: string };
type Target = "auto" | string;
type JobsResponse = { ok: boolean; now: string; jobs: JobRow[]; heartbeats: Heartbeat[]; error?: string };
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

const errorText = (result: { status: number; body: { error?: string } }) => result.body.error ?? `The request failed (code ${result.status}).`;

/** "Settle last week" for the job a machine reports it is running; "a job" when it cannot be matched. */
function runningName(id: string | null, jobs: JobRow[] | undefined): string {
  const job = id ? jobs?.find((item) => item.id === id) : undefined;
  return job ? (specFor(job.type)?.label ?? "a job") : "a job";
}

/* ------------------------------------------------------------------ confirm sheet */
function ConfirmSheet({ spec, busy, target, onSubmit, onClose }: { spec: JobSpec; busy: boolean; target: Target; onSubmit: (params: JobParams) => void; onClose: () => void }) {
  const [round, setRound] = useState<"auto" | "1" | "2" | "3">("auto");
  const [noPull, setNoPull] = useState(false);
  const [supersede, setSupersede] = useState(false);
  const params: JobParams = {
    ...(spec.after_round && round !== "auto" ? { after_round: Number(round) as 1 | 2 | 3 } : {}),
    ...(spec.no_pull && noPull ? { no_pull: true } : {}),
    ...(spec.supersede && supersede ? { supersede: true } : {}),
  };
  return (
    <div className="run-sheet" role="dialog" aria-modal="true" aria-label={`Confirm ${spec.label}`}>
      <button className="run-sheet-backdrop" type="button" onClick={onClose} aria-label="Cancel" />
      <div className="run-sheet-body">
        <span className="eyebrow">{spec.group}</span>
        <h2>{spec.label}</h2>
        <p>{spec.does}</p>
        <p className="inputs-muted">{spec.when}</p>
        <p className="run-target-note">Runs on: <b>{target === "auto" ? "Automatic (the always-on desktop; the backup only if it is down)" : machineLabel(target)}</b></p>
        {spec.after_round && (
          <label className="run-field">
            <span>Round just finished</span>
            <select value={round} onChange={(event) => setRound(event.target.value as typeof round)}>
              <option value="auto">Automatic</option>
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
          <button type="button" className="inputs-button primary run-big" onClick={() => onSubmit(params)} disabled={busy}>{busy ? "Sending…" : "Run it"}</button>
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
  const [showChecks, setShowChecks] = useState(false);
  const [target, setTarget] = useState<Target>("auto");
  const [notice, setNotice] = useState<{ tone: "ok" | "error"; text: string; problems?: string[] } | null>(null);

  const machines: MachineStatus[] = MACHINES.map((spec) => machineStatus(spec, data?.heartbeats, now));
  const targetOffline = target !== "auto" && machines.find((m) => m.name === target)?.status === "offline";

  const reload = useCallback(async () => {
    const result = await api<JobsResponse>("/api/jobs");
    setNow(Date.now());
    if (result.ok) {
      setData(result.body);
      setError(null);
    } else {
      setError(errorText(result));
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

  const submit = async (params: JobParams) => {
    if (!pending) return;
    const chosen = machines.find((m) => m.name === target && m.status !== "offline") ? target : "auto";   // never send a job to a machine that went offline since it was picked
    setBusy(true);
    const result = await api<{ job?: JobRecord }>("/api/jobs", { method: "POST", body: JSON.stringify({ type: pending.type, params, ...(chosen !== "auto" ? { target_machine: chosen } : {}) }) });
    setBusy(false);
    if (result.ok) {
      setNotice({ tone: "ok", text: `${pending.label} is queued${chosen === "auto" ? "" : ` for ${machineLabel(chosen)}`}. ${chosen === "auto" ? "The always-on desktop" : machineLabel(chosen)} picks it up within a minute or two.` });
      setPending(null);
      void reload();
    } else {
      setNotice({ tone: "error", text: errorText(result), problems: result.body.problems });
      setPending(null);
    }
  };

  const cancel = async (id: string) => {
    const result = await api<unknown>(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST", body: "{}" });
    setNotice(result.ok ? { tone: "ok", text: "Cancelled." } : { tone: "error", text: errorText(result) });
    void reload();
  };

  // The automatic 30-minute odds checks and hourly watches would bury everything else: hide the finished ones unless asked.
  const isAutoCheck = (job: JobRow) => job.requested_by === "cron" && (job.type === "odds_reprice" || job.type === "watch") && job.state === "done";
  const visibleJobs = (data?.jobs ?? []).filter((job) => showChecks || !isAutoCheck(job));
  const hiddenChecks = (data?.jobs ?? []).length - visibleJobs.length;

  const suggested = suggestedGroup(now);
  const busyTypes = new Set((data?.jobs ?? []).filter((job) => !isTerminal(job.state) && now - (parseUtc(job.requested_at) ?? 0) < 6 * 3_600_000).map((job) => job.type));

  return (
    <div className="run-page">
      <PageIntro eyebrow="Operate" title="Run" description="Start a pricing job from here, for example from your phone. The desktop checks for requests every minute and the results appear at the bottom of the page." />

      <Panel eyebrow="Run on" title="Machine">
        {!data ? (
          error ? <p className="inputs-muted">Machine: unavailable</p> : <LoadingState label="Loading machines" />
        ) : (
          <>
            {machines.every((m) => m.status === "offline") && (
              <div className="inputs-banner warn"><strong>No machine is reporting in.</strong>Jobs will wait in the queue until one comes back online.</div>
            )}
            <div className="run-machines" role="radiogroup" aria-label="Machine to run on">
              <label className={`run-machine ${target === "auto" ? "selected" : ""}`} aria-label="Automatic: the always-on desktop, the backup only if it is down">
                <input type="radio" name="run-target" checked={target === "auto"} onChange={() => setTarget("auto")} />
                <span>
                  <strong>Automatic</strong>
                  <small>The always-on desktop. The backup takes over only if the desktop has stopped reporting in.</small>
                </span>
              </label>
              {machines.map((m) => (
                <label key={m.name} className={`run-machine ${m.status} ${target === m.name ? "selected" : ""}`} aria-label={`${m.label}, ${m.status === "offline" ? "offline" : m.status === "running" ? "running a job" : "online, idle"}`}>
                  <input type="radio" name="run-target" checked={target === m.name} disabled={m.status === "offline"} onChange={() => setTarget(m.name)} />
                  <span>
                    <strong>{m.label}</strong>
                    <small title={m.lastSeen ? `Last heard from ${etTime(m.lastSeen)}` : undefined}>
                      <i aria-hidden="true" className={`run-dot ${m.status}`} />
                      {m.status === "offline" ? (m.lastSeen ? `Offline (last heard from ${ago(m.lastSeen, now)})` : "Offline (has not reported in)") : m.status === "running" ? `Running ${runningName(m.jobId, data?.jobs)}` : `Online and idle (checked in ${ago(m.lastSeen ?? undefined, now)})`}
                    </small>
                  </span>
                </label>
              ))}
            </div>
            {targetOffline && <p className="inputs-muted">The selected machine went offline, so new jobs will use Automatic until it is back.</p>}
            {target !== "auto" && !targetOffline && machines.find((m) => m.name === target)?.role === "backup" && (
              <p className="inputs-muted">The backup only takes a job addressed to it when the main desktop is not busy and both run the same version of the code.</p>
            )}
          </>
        )}
      </Panel>

      {notice && (
        <div className={`inputs-banner ${notice.tone === "error" ? "warn" : ""}`} role="status">
          <strong>{notice.tone === "error" ? "Not started" : "Done"}</strong>
          {notice.text}
          {notice.problems?.map((problem) => <span key={problem}>{problem}</span>)}
          <button type="button" className="inputs-button subtle run-dismiss" onClick={() => setNotice(null)}>Dismiss</button>
        </div>
      )}

      {[suggested, ...GROUPS.filter((group) => group !== suggested)].map((group) => {
        const specs = JOB_TYPES.filter((spec) => spec.group === group);
        const grid = (
          <div className="run-grid">
            {specs.map((spec) => (
              <button type="button" key={spec.type} className="run-card" onClick={() => setPending(spec)} disabled={busyTypes.has(spec.type)}>
                <strong>{spec.label}</strong>
                <span>{spec.does}</span>
                <small>{busyTypes.has(spec.type) ? "Already waiting or running" : spec.when}</small>
              </button>
            ))}
          </div>
        );
        // Today's weekday group is open and highlighted; the rest are collapsed (site audit C4).
        return group === suggested ? (
          <Panel key={group} eyebrow="Suggested now" title={group} className="run-suggested">{grid}</Panel>
        ) : (
          <details key={group} className="run-collapsed"><summary>{group}<small>{specs.length} job{specs.length === 1 ? "" : "s"}</small></summary>{grid}</details>
        );
      })}

      <OddsSignalsPanel />

      <Panel eyebrow="Latest 50" title="Recent runs" actions={<button type="button" className="inputs-button" onClick={() => void reload()}>Refresh</button>}>
        <label className="run-check">
          <input type="checkbox" checked={showChecks} onChange={(event) => setShowChecks(event.target.checked)} />
          <span>Also show finished automatic checks (odds and input watch){hiddenChecks > 0 && !showChecks ? `: ${hiddenChecks} hidden` : ""}</span>
        </label>
        {error && <div className="inputs-banner warn" role="alert"><strong>Could not load the recent runs</strong>{error}</div>}
        {!data && !error && <LoadingState label="Loading recent runs" />}
        {data && data.jobs.length === 0 && <EmptyState title="Nothing has run yet" detail="Press one of the buttons above to start the first run." />}
        <div className="run-jobs">
          {visibleJobs.map((job) => {
            const status = job.status;
            const spec = specFor(job.type);
            return (
              <article key={job.id} className={`run-job state-${job.state}`}>
                <header>
                  <strong>{spec?.label ?? job.type}</strong>
                  <span className={`run-badge ${job.state}`}>{stateLabel(job.state)}</span>
                </header>
                <p className="inputs-muted" title={`${ago(job.requested_at, now)}`}>
                  Requested {etTime(job.requested_at)} by {requesterLabel(job.requested_by)}{paramsSentence(job.params) ? ` (${paramsSentence(job.params)})` : ""}
                </p>
                {(status?.machine || duration(status, now)) && (
                  <div className="run-facts">
                    {status?.machine && <span>Ran on <b>{machineLabel(status.machine)}</b></span>}
                    {job.target_machine && !status?.machine && <span>For <b>{machineLabel(job.target_machine)}</b></span>}
                    {duration(status, now) && <span>Took <b>{duration(status, now)}</b></span>}
                  </div>
                )}
                {waitingFor(job, job.state, data?.heartbeats, now) && <p className="run-summary run-waiting">{waitingFor(job, job.state, data?.heartbeats, now)}</p>}
                {status?.summary && <p className="run-summary">{status.summary}</p>}
                {job.state === "queued" && <button type="button" className="inputs-button danger" onClick={() => void cancel(job.id)}>Cancel</button>}
                {(status?.log_tail || (status && status.state !== "claimed")) && (
                  <details className="tech-details">
                    <summary>Technical details</summary>
                    <p className="inputs-muted">Job {job.id} · {job.type}{status?.exit_code !== undefined && status?.exit_code !== null ? ` · exit code ${status.exit_code}` : ""}</p>
                    <pre className="run-log">{status?.log_tail || "The log is empty."}</pre>
                    <a className="inputs-muted" href={`/api/jobs/${encodeURIComponent(job.id)}/log`} target="_blank" rel="noreferrer">Open the full log</a>
                  </details>
                )}
              </article>
            );
          })}
        </div>
      </Panel>

      {data && (
        <details className="tech-details">
          <summary>Technical details</summary>
          <ul className="inputs-muted">
            {machines.map((m) => <li key={m.name}>{m.label}: machine name {m.name}{m.commit ? `, code version ${m.commit}` : ""}{m.lastSeen ? `, last heard from ${etTime(m.lastSeen)}` : ""}</li>)}
          </ul>
        </details>
      )}

      {pending && <ConfirmSheet spec={pending} busy={busy} target={targetOffline ? "auto" : target} onSubmit={(params) => void submit(params)} onClose={() => setPending(null)} />}
    </div>
  );
}
