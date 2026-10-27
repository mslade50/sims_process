/** Shared, side-effect-free rules for the owner operations page. Execution success and output verification are separate facts. */
import { JOB_TYPES, type MachineBeat } from "./jobs-rules.ts";
import { SLOTS, WINDOWS, type Day } from "../worker/cron-rules.ts";

export const OPS_SCHEMA = "golfops.task.v1";
export const TASK_STATES = ["planned", "in_progress", "review", "observation_pending", "verified", "blocked"] as const;
export type TaskState = typeof TASK_STATES[number];
export type Actor = { kind: "owner" | "agent"; id: string };
export type HistoryEntry = { revision: number; at: string; actor: Actor; action: string; note: string; evidence: string[]; changes?: { before: { state: TaskState; next_step: string }; after: { state: TaskState; next_step: string } } };
export type Lease = { actor: Actor; expires_at: string; token_hash: string };
export type OpsTask = { schema: typeof OPS_SCHEMA; id: string; title: string; state: TaskState; revision: number; updated_at: string | null; dependencies: string[]; next_step: string; lease: Lease | null; history: HistoryEntry[] };
export type PublicTask = Omit<OpsTask, "lease"> & { lease: Omit<Lease, "token_hash"> | null };
export type StageState = "scheduled" | "running" | "verified" | "completed_unverified" | "waiting_upstream" | "retrying" | "failed" | "needs_attention" | "skipped" | "unknown" | "stale";
export const STATE_LABEL: Record<StageState, string> = { scheduled: "Scheduled", running: "Running", verified: "Verified", completed_unverified: "Finished; output unverified", waiting_upstream: "Waiting on upstream", retrying: "Retrying", failed: "Failed", needs_attention: "Needs attention", skipped: "Intentionally skipped", unknown: "Not yet verified", stale: "Evidence is stale" };
export const TASK_LABEL: Record<TaskState, string> = { planned: "Planned", in_progress: "In progress", review: "In review", observation_pending: "Observation pending", verified: "Verified", blocked: "Blocked" };
export type StageSpec = { id: string; label: string; detail: string; dependencies: string[]; jobs: string[] };
export const STAGES: StageSpec[] = [
  { id: "inputs", label: "Data and skills", detail: "Published shot data, field, skill and course-fit inputs", dependencies: [], jobs: [] },
  { id: "weekly", label: "Weekly pricing", detail: "Sunday pricing and Monday's full weekly run", dependencies: ["inputs"], jobs: ["sunday_week", "monday_week"] },
  { id: "refresh", label: "Pre-tee refresh", detail: "Tuesday, Wednesday and final pre-tee prices", dependencies: ["weekly"], jobs: ["tuesday", "wednesday", "thursday", "thursday_close"] },
  { id: "boards", label: "Boards and reports", detail: "Correct-event PGA and Euro publication", dependencies: ["weekly", "refresh"], jobs: ["publish", "xtour_publish"] },
  { id: "checks", label: "Watch and odds", detail: "Input watch, quote refresh and daily health", dependencies: ["weekly"], jobs: ["watch", "odds_reprice", "daily_health"] },
  { id: "in_event", label: "During the event", detail: "Complete-round pricing and cross-tour refresh", dependencies: ["weekly"], jobs: ["after_round", "xtour"] },
  { id: "delivery", label: "Delivery", detail: "Emails, course-fit breakdown and OneDrive handoff", dependencies: ["boards"], jobs: [] },
  { id: "settlement", label: "Settlement", detail: "Grade finished events and alerted/voided bets", dependencies: ["in_event"], jobs: ["monday_settle"] },
];
/** Future B06 producer contract, not evidence that any producer is connected.
 * Every named check is mandatory for EACH selected event. `ok` means its described
 * content was inspected; process exit, object existence and a generic proof do not suffice.
 * Job-backed receipts use the exact latest known stage job ID as occurrence_id,
 * never a producer-chosen label or calendar slot. That job must be done, exit 0,
 * with a valid finished_at before any checks/observation. Missing execution is unknown.
 * Inputs/delivery have no independently connected expected occurrence: their receipts
 * cannot establish current state until a reviewed scoped occurrence source is added.
 */
export const STAGE_EVIDENCE_CONTRACT = {
  inputs: { occurrence: "unconnected", required_checks: {
    published_shot_data: "Published generation, validity flags, model version and completeness match the event.",
    field_inputs: "Published field contents match the event.",
    skill_inputs: "Published skill estimates match the event and input generation.",
    course_fit_inputs: "Published course-fit inputs match the event and courses.",
  } },
  weekly: { occurrence: "latest_job_id", required_checks: {
    pricing_outputs: "Pricing contents match the event and this execution.",
    edge_outputs: "Edge comparison contents match the event and this execution's pricing.",
    report_outputs: "Report contents match the event and this execution's pricing.",
    dashboard_publication: "Published dashboard contents match the event and this execution.",
  } },
  refresh: { occurrence: "latest_job_id", required_checks: {
    pricing_outputs: "Refreshed pricing contents match the event and this execution.",
    edge_outputs: "Refreshed edge contents match the event and this execution's pricing.",
    report_outputs: "Refreshed report contents match the event and this execution's pricing.",
    dashboard_publication: "Published dashboard contents match the event and this refresh.",
  } },
  boards: { occurrence: "latest_job_id", required_checks: {
    board_publication: "Live board event, simulation timestamp and source=sim rows match the execution.",
    report_publication: "Published report contents match the event and this execution.",
  } },
  checks: { occurrence: "latest_job_id", required_checks: {
    input_watch: "Field, tee-time and forecast checks cover the event and this occurrence.",
    quote_refresh: "Quote freshness and model-price comparison cover the event and this occurrence.",
    daily_health: "Expected-run and freshness checks cover the event and this occurrence.",
  } },
  in_event: { occurrence: "latest_job_id", required_checks: {
    round_pricing: "Pricing contents match the event, completed round and this execution.",
    cross_tour_refresh: "Shared baseline contents and publication match this event and execution.",
  } },
  delivery: { occurrence: "unconnected", required_checks: {
    email_delivery: "Delivered email contains the event's results and required course-fit breakdown.",
    onedrive_handoff: "Current-event export receipt and destination hashes match; local readback alone is insufficient.",
  } },
  settlement: { occurrence: "latest_job_id", required_checks: {
    event_grading: "Final-event grading contents match the event and this execution.",
    bet_grading: "Alerted and voided bet grading reconciles for the event and this execution.",
    settlement_health: "Post-settlement health checks cover the event and this execution.",
  } },
} as const;
/** All receipt states expire. Checks must be within five minutes before observation
 * (one minute clock skew allowed), after job completion, with a safe proof and the
 * same event/occurrence. Receipt lifetime is positive and capped at seven days.
 */
export const RECEIPT_TIME_LIMITS = { max_lifetime_ms: 7 * 86_400_000, check_lag_ms: 300_000, clock_skew_ms: 60_000 } as const;
export type EventScope = { uid: string; name: string; tour: string };
export type Occurrence = { id: string; type: string; label: string; scheduled_at: string; local_day: string; local_time: string; frequent: boolean; why: string; job_id?: string; execution_state?: string };
export type JobEvidence = { id: string; type: string; requested_at: string; requested_by?: string; status: { state?: string; started_at?: string; finished_at?: string; exit_code?: number; machine?: string } | null };
export type OutputCheck = { name: string; ok: boolean; event_uid: string; occurrence_id: string; checked_at: string; proof: string };
export type StageReceipt = { stage: string; event_uid: string; tour: string; occurrence_id: string; observed_at: string; expires_at: string; state: StageState; outputs: OutputCheck[]; explanation: string; actor: string };
export type StageSummary = StageSpec & { state: StageState; explanation: string; evidence_at: string | null; valid_until: string | null; proof: string[]; execution: JobEvidence | null };
export type MachineSummary = { id: string; label: string; online: boolean; state: string; role: string; at: string | null; commit: string | null; code: string | null };

const dateFormatter = new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit", weekday: "short", hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
export function etParts(ms: number) {
  const p = Object.fromEntries(dateFormatter.formatToParts(new Date(ms)).map((v) => [v.type, v.value]));
  return { date: `${p.year}-${p.month}-${p.day}`, weekday: p.weekday as Day, time: `${p.hour}:${p.minute}`, minute: Number(p.minute) };
}
export function weekStart(ms: number): string {
  const date = etParts(ms).date;
  const d = new Date(`${date}T12:00:00Z`);
  d.setUTCDate(d.getUTCDate() - d.getUTCDay());
  return d.toISOString().slice(0, 10);
}
export function shiftDate(date: string, days: number): string { const d = new Date(`${date}T12:00:00Z`); d.setUTCDate(d.getUTCDate() + days); return d.toISOString().slice(0, 10); }
export function validWeek(value: string, now: number): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const d = Date.parse(`${value}T12:00:00Z`);
  return Number.isFinite(d) && new Date(d).toISOString().slice(0, 10) === value && new Date(d).getUTCDay() === 0 && value >= shiftDate(weekStart(now), -56) && value <= shiftDate(weekStart(now), 28);
}
const scheduleCache = new Map<string, Occurrence[]>();
export function weekSchedule(start: string): Occurrence[] {
  const cached = scheduleCache.get(start); if (cached) return cached.map((v) => ({ ...v }));
  const end = shiftDate(start, 7), out: Occurrence[] = [];
  // UTC ticks naturally preserve both repeated hours and the missing hour at New York DST transitions.
  const from = Date.parse(`${start}T00:00:00Z`);
  for (let ms = from; ms < from + 8 * 86_400_000; ms += 900_000) {
    const p = etParts(ms); if (p.date < start || p.date >= end) continue;
    const due = new Map<string, string>();
    for (const s of SLOTS) if (s.days.includes(p.weekday) && s.times.includes(p.time)) due.set(s.type, s.note);
    for (const w of WINDOWS) if (w.days.includes(p.weekday) && w.minutes.includes(p.minute) && p.time >= w.from && p.time <= w.to) due.set(w.type, w.note);
    for (const [type, why] of due) {
      const at = new Date(ms).toISOString();
      out.push({ id: `${type}@${at}`, type, label: JOB_TYPES.find((v) => v.type === type)?.label ?? type, scheduled_at: at, local_day: p.date, local_time: p.time, frequent: type === "watch" || type === "odds_reprice", why });
    }
  }
  if (scheduleCache.size >= 4) scheduleCache.delete(scheduleCache.keys().next().value as string);
  scheduleCache.set(start, out);
  return out.map((v) => ({ ...v }));
}
export function attachExecutions(schedule: Occurrence[], jobs: JobEvidence[]): Occurrence[] {
  const byId = new Map(schedule.map((o) => [o.id, o]));
  for (const job of jobs) {
    if (job.requested_by !== "cron") continue;
    const requested = Date.parse(job.requested_at); if (!Number.isFinite(requested)) continue;
    const tick = Math.floor(requested / 900_000) * 900_000;
    const o = byId.get(`${job.type}@${new Date(tick).toISOString()}`);
    if (o && (!o.job_id || requested > Date.parse(jobs.find((j) => j.id === o.job_id)?.requested_at ?? ""))) { o.job_id = job.id; o.execution_state = job.status?.state ?? "queued"; }
  }
  return schedule;
}
export function safeProof(value: unknown): value is string {
  return typeof value === "string" && value.length <= 500 && ( /^https:\/\/github\.com\/mslade50\/[A-Za-z0-9_./#?=+%-]+$/.test(value) || /^\/api\/jobs\/j-[A-Za-z0-9-]+\/log$/.test(value) || /^r2:\/\/(golf-model-dashboard-data|golf-odds-board)\/[A-Za-z0-9_./:-]+$/.test(value) );
}
export function cleanText(value: unknown, max = 2000): value is string {
  return typeof value === "string" && value.trim().length > 0 && value.length <= max && !Array.from(value).some((c) => c.charCodeAt(0) < 32 && ![9,10,13].includes(c.charCodeAt(0))) && !/(?:gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{16,}|AKIA[A-Z0-9]{16}|Bearer\s+[A-Za-z0-9._-]{20,}|\d{7,}:AA[A-Za-z0-9_-]{20,})/.test(value);
}
export function stageSummaries(jobs: JobEvidence[], receipts: StageReceipt[], events: EventScope[], selected: string, start: string, now: number): StageSummary[] {
  const relevant = selected === "all" ? events : events.filter((e) => e.uid === selected);
  const end = shiftDate(start, 7);
  const { max_lifetime_ms, check_lag_ms, clock_skew_ms } = RECEIPT_TIME_LIMITS;
  return STAGES.map((spec) => {
    const known = jobs.filter((j) => {
      const at = Date.parse(j.requested_at);
      return Number.isFinite(at) && spec.jobs.includes(j.type) && etParts(at).date >= start && etParts(at).date < end;
    }).sort((a,b) => Date.parse(b.requested_at) - Date.parse(a.requested_at));
    const execution = known[0] ?? null;
    const summary: StageSummary = { ...spec, state: "unknown", explanation: "No current content-check receipt is connected for this stage.", evidence_at: null, valid_until: null, proof: [], execution };
    const contract = STAGE_EVIDENCE_CONTRACT[spec.id as keyof typeof STAGE_EVIDENCE_CONTRACT];
    if (!contract || contract.occurrence === "unconnected") {
      summary.explanation = "The independently scoped expected occurrence is not connected for this stage; receipt verification is unavailable.";
      return summary;
    }
    if (!execution) {
      summary.explanation = "No latest known job establishes the expected occurrence; receipt verification remains unknown.";
      return summary;
    }
    if (Date.parse(execution.requested_at)>now) {
      summary.explanation="The newest scoped execution is future-dated; current health is unverified. No older success is substituted.";
      return summary;
    }
    if (known.length > 1 && Date.parse(known[1].requested_at) === Date.parse(execution.requested_at)) {
      summary.explanation = "Latest execution ordering is ambiguous; receipt verification remains unknown.";
      return summary;
    }
    // Latest execution takes precedence: an earlier success cannot hide a new failure/run.
    const state = execution.status?.state;
    if (state === "running" || state === "claimed") {
      summary.state = "running"; summary.explanation = "Execution is active; delivered outputs are not verified yet."; return summary;
    }
    if (state === "failed" || (state === "done" && typeof execution.status?.exit_code === "number" && execution.status.exit_code !== 0)) {
      summary.state = "failed"; summary.explanation = "The latest observed execution failed. Open its log for the specific cause."; return summary;
    }
    if (state !== "done") {
      summary.explanation = "A queued or interrupted execution is visible; output health remains unverified."; return summary;
    }
    summary.state = "completed_unverified";
    summary.explanation = "Execution finished; correct-event delivered outputs still need content verification.";
    const finished = Date.parse(execution.status?.finished_at ?? "");
    if (execution.status?.exit_code !== 0 || !Number.isFinite(finished) || finished < Date.parse(execution.requested_at) || finished > now) {
      summary.explanation = "Successful finished execution evidence is incomplete or invalid; output verification is unavailable.";
      return summary;
    }
    if (!relevant.length) return summary;
    let ambiguous = false;
    const picked = relevant.map((event) => {
      const candidates = receipts.filter((r) => {
        const at = Date.parse(r.observed_at);
        return r.stage === spec.id && r.event_uid === event.uid && Number.isFinite(at);
      }).sort((a,b) => Date.parse(b.observed_at) - Date.parse(a.observed_at));
      if (candidates.length > 1 && Date.parse(candidates[0].observed_at) === Date.parse(candidates[1].observed_at)) ambiguous = true;
      return candidates[0];
    });
    if (ambiguous) {
      summary.explanation = "Latest receipt ordering is ambiguous; output health remains unverified."; return summary;
    }
    const current = picked.filter((r): r is StageReceipt => !!r);
    if(picked.some((r,i)=>r&&(r.tour!==relevant[i].tour||r.occurrence_id!==execution.id))) {
      summary.explanation="The newest event receipt does not match the expected tour and execution. No older success is substituted.";
      return summary;
    }
    if(current.some((r)=>{const at=Date.parse(r.observed_at);return at<finished||at>now+clock_skew_ms||etParts(at).date<start||etParts(at).date>=end;})) {
      summary.explanation="The newest scoped receipt has invalid timing; current health is unverified. No older success is substituted.";
      return summary;
    }
    if (current.length) summary.evidence_at = new Date(Math.min(...current.map((r) => Date.parse(r.observed_at)))).toISOString();
    // Evaluate freshness BEFORE adverse state, including when other events lack receipts.
    if (current.some((r) => {
      const expires = Date.parse(r.expires_at), observed = Date.parse(r.observed_at);
      return !Number.isFinite(expires) || expires <= now || expires <= observed || expires > observed + max_lifetime_ms;
    })) {
      summary.state = "stale"; summary.explanation = "Stored receipt evidence has expired or invalid freshness. Earlier state does not establish current health."; return summary;
    }
    if (current.length) summary.valid_until = new Date(Math.min(...current.map((r) => Date.parse(r.expires_at)))).toISOString();
    const adverse = current.find((r) => ["failed", "needs_attention", "waiting_upstream", "retrying", "skipped"].includes(r.state));
    if (adverse) {
      summary.state = adverse.state; summary.explanation = cleanText(adverse.explanation) ? adverse.explanation : STATE_LABEL[adverse.state]; summary.evidence_at = adverse.observed_at; return summary;
    }
    const required = Object.keys(contract.required_checks);
    const verified = picked.length > 0 && picked.every((r) => r && r.state === "verified" && Array.isArray(r.outputs) && r.outputs.length === required.length && required.every((name) => r.outputs.filter((o) => o && o.name === name).length === 1) && r.outputs.every((o) => {
      const checked = Date.parse(o.checked_at), observed = Date.parse(r.observed_at);
      return o.ok === true && o.event_uid === r.event_uid && o.occurrence_id === execution.id && safeProof(o.proof) && Number.isFinite(checked) && checked >= finished && checked >= observed - check_lag_ms && checked <= observed + clock_skew_ms && checked <= now + clock_skew_ms && checked < Date.parse(r.expires_at);
    }));
    if (verified) {
      summary.state = "verified";
      summary.explanation = "All required content checks match every selected event and the latest successfully finished job.";
      summary.proof = current.flatMap((r) => r.outputs.map((o) => o.proof));
    }
    return summary;
  });
}
export function summarizeMachines(beats: MachineBeat[], now: number): MachineSummary[] {
  return [{ id: "mckinley_home", label: "Large desktop" }, { id: "desktop-2ki41v6", label: "Small desktop" }].map((m) => {
    const b = beats.find((v) => v.machine === m.id); const age = now - Date.parse(b?.at ?? "");
    return { ...m, online: Number.isFinite(age) && age >= -60_000 && age <= 600_000, state: b?.state ?? "unknown", role: b?.role ?? "Role unverified", at: b?.at ?? null, commit: b?.commit ?? null, code: b?.code ?? null };
  });
}
export function publicTask(task: OpsTask): PublicTask { return { ...task, lease: task.lease ? { actor: task.lease.actor, expires_at: task.lease.expires_at } : null }; }
export function sameActor(a: Actor, b: Actor): boolean { return a.kind === b.kind && a.id === b.id; }
export function isTaskState(value: unknown): value is TaskState { return TASK_STATES.includes(value as TaskState); }
