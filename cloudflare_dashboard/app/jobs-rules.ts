/**
 * Phone-triggered golfprice jobs (owner authorization, October 5, 2026): the allow-list, request validation, ids and the rate limit.
 * Pure TypeScript with no runtime APIs, shared by the Worker (worker/jobs-api.ts), the Run page and tests/jobs.test.mjs.
 * The desktop runner (golfprice/jobrunner.py) holds the same allow-list and validates every job again before it runs anything.
 * A job names an ops moment from this list plus a few boolean flags. There is never free text and never a command line.
 */

export const QUEUE_KEY = "jobs/queue.json";
export const STATUS_PREFIX = "jobs/status/";
export const LOG_PREFIX = "jobs/logs/";
export const HEARTBEAT_PREFIX = "jobs/heartbeat/";
export const QUEUE_SCHEMA = "golfprice.job_queue.v1";
export const MAX_QUEUE = 200;
export const MAX_AGE_MS = 6 * 3_600_000; // a request older than this is never run, and no longer blocks a new one
export const STALE_HEARTBEAT_MS = 10 * 60_000;

/** Machines a job can be addressed to (the Worker rejects any other name; golfprice/jobrunner.py MACHINE_RE is the runner side). */
export type MachineSpec = { name: string; label: string; role: "primary" | "backup" };
export const MACHINES: MachineSpec[] = [
  { name: "desktop-2ki41v6", label: "Desktop (always-on)", role: "primary" },
  { name: "mckinley_home", label: "Desktop (big)", role: "backup" },
];
export const isKnownMachine = (name: unknown): name is string => typeof name === "string" && MACHINES.some((m) => m.name === name);
export const machineLabel = (name: string | undefined | null): string => MACHINES.find((m) => m.name === name)?.label ?? name ?? "";

export type MachineBeat = { machine?: string; at?: string; state?: string; job_id?: string | null; commit?: string | null; code?: string | null; role?: string | null };
export type MachineStatus = { name: string; label: string; role: "primary" | "backup"; status: "offline" | "idle" | "running"; lastSeen: string | null; jobId: string | null; commit: string | null };

/** Live status of one allow-listed machine from its heartbeat object: offline = no heartbeat or older than STALE_HEARTBEAT_MS. */
export function machineStatus(spec: MachineSpec, beats: MachineBeat[] | undefined, now: number): MachineStatus {
  const beat = (beats ?? []).find((b) => b && b.machine === spec.name);
  const at = parseUtc(beat?.at);
  const alive = at !== null && now - at <= STALE_HEARTBEAT_MS;
  return {
    name: spec.name,
    label: spec.label,
    role: spec.role,
    status: !alive ? "offline" : beat?.state === "running" ? "running" : "idle",
    lastSeen: beat?.at ?? null,
    jobId: alive && beat?.state === "running" ? (beat.job_id ?? null) : null,
    commit: beat?.commit ?? null,
  };
}

export type JobGroup = "Monday" | "Tuesday" | "Wednesday" | "Thursday" | "During event" | "Anytime";
export const GROUPS: JobGroup[] = ["Monday", "Tuesday", "Wednesday", "Thursday", "During event", "Anytime"];

export type JobSpec = {
  type: string;
  group: JobGroup;
  label: string;
  /** one line: what it does */
  does: string;
  /** one line: when to press it */
  when: string;
  no_pull: boolean;
  supersede: boolean;
  after_round: boolean;
};

export const JOB_TYPES: JobSpec[] = [
  { type: "monday_settle", group: "Monday", label: "Settle last week", does: "Pulls the finished event, grades last week's prices and bets, then runs a health check.", when: "Monday morning, once the event is final.", no_pull: true, supersede: false, after_round: false },
  { type: "monday_week", group: "Monday", label: "Price the new week", does: "Prices the upcoming event, then updates the edges, report and dashboard.", when: "Monday, after settling and once the field is known.", no_pull: true, supersede: true, after_round: false },
  { type: "tuesday", group: "Tuesday", label: "Tuesday refresh", does: "Prices the week again on Tuesday's data, then updates the edges, report and dashboard.", when: "Tuesday, after the market has moved.", no_pull: true, supersede: true, after_round: false },
  { type: "wednesday", group: "Wednesday", label: "Wednesday reprice", does: "Prices the week again once tee times are known, then updates the edges, report and dashboard.", when: "Wednesday, once tee times are out.", no_pull: true, supersede: true, after_round: false },
  { type: "thursday", group: "Thursday", label: "Thursday morning price", does: "The final pricing run before the first tee, then updates the edges, report and dashboard.", when: "Thursday morning before the first tee.", no_pull: true, supersede: true, after_round: false },
  { type: "thursday_close", group: "Thursday", label: "Thursday close", does: "Locks in the pre-tournament numbers and publishes them (nothing is in play yet).", when: "Starts by itself 30 minutes before each event's first tee; press it only to force it.", no_pull: false, supersede: true, after_round: false },
  { type: "after_round", group: "During event", label: "After a round", does: "Prices the live state after a finished round, then publishes it. Leave the round on Automatic and it works out which round just finished.", when: "Starts by itself when a round finishes; press it only to force it.", no_pull: false, supersede: true, after_round: true },
  { type: "daily_health", group: "Anytime", label: "Health check", does: "Checks that expected runs happened and nothing is stale or blocked. Changes nothing else.", when: "Any time you want to know if all is well.", no_pull: false, supersede: false, after_round: false },
  { type: "xtour", group: "Monday", label: "Refresh the shared skill baseline", does: "Rebuilds the baseline that puts PGA and European players on one scale and shares it with every machine. Prices nothing.", when: "Monday after the new data snapshot exists, if the Monday job did not already do it.", no_pull: false, supersede: false, after_round: false },
  { type: "xtour_publish", group: "Anytime", label: "Share the skill baseline", does: "Shares this machine's skill baseline with the other machines. Does not rebuild or price anything.", when: "If another machine says it cannot see the newest baseline.", no_pull: false, supersede: false, after_round: false },
  { type: "publish", group: "Anytime", label: "Publish model inputs", does: "Sends the latest model inputs to this dashboard. Does not price anything.", when: "If the Model inputs page looks out of date.", no_pull: false, supersede: false, after_round: false },
  { type: "odds_reprice", group: "Anytime", label: "Odds check (no re-simulation)", does: "Re-reads the latest odds and compares them with the model's prices from the last full run. Takes seconds, and alerts you if a bet appears or moves.", when: "Runs by itself every 30 minutes from Monday afternoon to the Thursday tee; press it for an immediate look.", no_pull: true, supersede: false, after_round: false },
  { type: "watch", group: "Anytime", label: "Input watch", does: "Checks the field, tee times and forecast against the last full run, and prices again only if something that moves prices has changed.", when: "Runs by itself every 15 minutes all week; press it only to check right now.", no_pull: true, supersede: false, after_round: false },
];

export const TERMINAL = ["done", "failed", "cancelled", "expired"] as const;
export const isTerminal = (state: string) => (TERMINAL as readonly string[]).includes(state);

export type JobParams = { after_round?: 1 | 2 | 3; no_pull?: boolean; supersede?: boolean };
export type JobRecord = {
  id: string;
  type: string;
  params: JobParams;
  requested_by: string;
  requested_at: string;
  verified: boolean;
  /** Optional: only this machine may run the job (absent = Auto: the desktop, the laptop only on failover). */
  target_machine?: string;
  cancelled_at?: string;
  cancelled_by?: string;
};
export type JobStatus = {
  id: string;
  state: string;
  machine?: string;
  claimed_at?: string;
  started_at?: string;
  finished_at?: string;
  exit_code?: number | null;
  summary?: string;
  log_tail?: string;
  updated_at?: string;
};

export const specFor = (type: unknown): JobSpec | undefined => JOB_TYPES.find((spec) => spec.type === type);

/** Plain words for a job state. */
const STATE_LABELS: Record<string, string> = { queued: "Waiting", claimed: "Starting", running: "Running", done: "Done", failed: "Failed", cancelled: "Cancelled", expired: "Expired" };
export const stateLabel = (state: string): string => STATE_LABELS[state] ?? state;

/** Who asked for a job: the automatic schedule, or the person's email. */
export const requesterLabel = (by: string | undefined): string => (by === "cron" ? "the automatic schedule" : by || "someone");

/** The options of a job as a short sentence ("after round 2, skipped the data pull"); empty when it ran with none. */
export function paramsSentence(params: JobParams | undefined): string {
  return [params?.after_round ? `after round ${params.after_round}` : "", params?.no_pull ? "skipped the data pull" : "", params?.supersede ? "re-ran even if nothing had changed" : ""].filter(Boolean).join(", ");
}

export function isoSeconds(ms: number): string {
  return new Date(ms).toISOString().replace(/\.\d{3}Z$/, "Z");
}

export function parseUtc(value: unknown): number | null {
  if (typeof value !== "string" || !/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z$/.test(value)) return null;
  const ms = Date.parse(value);
  return Number.isFinite(ms) ? ms : null;
}

/** Sortable by time, then random. `rand` is 6 lowercase hex characters supplied by the caller. */
export function makeJobId(now: number, rand: string): string {
  return `j-${isoSeconds(now).replace(/[-:]/g, "").replace("Z", "")}-${rand}`;
}
export const JOB_ID_RE = /^j-\d{8}T\d{6}-[0-9a-f]{6}$/;

/** Validate a create request body. Unknown keys are rejected; flags the type does not accept are rejected, never ignored. */
export function checkRequest(body: unknown): { problems: string[]; type?: string; params?: JobParams; target_machine?: string } {
  const problems: string[] = [];
  if (!body || typeof body !== "object" || Array.isArray(body)) return { problems: ["body must be a JSON object"] };
  const raw = body as Record<string, unknown>;
  for (const key of Object.keys(raw)) {
    if (!["type", "params", "target_machine"].includes(key)) problems.push(`${key} is set by the server or not allowed`);
  }
  const spec = specFor(raw.type);
  if (!spec) {
    problems.push(`type must be one of: ${JOB_TYPES.map((s) => s.type).join(", ")}`);
    return { problems };
  }
  let target: string | undefined;
  if (raw.target_machine !== undefined && raw.target_machine !== null && raw.target_machine !== "auto") {
    if (isKnownMachine(raw.target_machine)) target = raw.target_machine;
    else problems.push(`target_machine must be one of: ${MACHINES.map((m) => m.name).join(", ")} (or omitted for Auto)`);
  }
  const params = raw.params === undefined || raw.params === null ? {} : raw.params;
  if (typeof params !== "object" || Array.isArray(params)) return { problems: [...problems, "params must be an object"] };
  const clean: JobParams = {};
  for (const [key, value] of Object.entries(params as Record<string, unknown>)) {
    if (key === "after_round") {
      if (!spec.after_round) problems.push(`${spec.type} does not take after_round`);
      else if (value !== 1 && value !== 2 && value !== 3) problems.push("after_round must be 1, 2 or 3");
      else clean.after_round = value;
    } else if (key === "no_pull" || key === "supersede") {
      if (typeof value !== "boolean") problems.push(`${key} must be true or false`);
      else if (!spec[key]) problems.push(`${spec.type} does not accept ${key}`);
      else if (value) clean[key] = true;
    } else {
      problems.push(`param ${key} is not allowed`);
    }
  }
  return target ? { problems, type: spec.type, params: clean, target_machine: target } : { problems, type: spec.type, params: clean };
}

export function buildJob(body: unknown, email: string, verified: boolean, now: number, rand: string): { job?: JobRecord; problems: string[] } {
  if (!verified) return { problems: ["the Cloudflare Access identity is not verified (the Worker needs ACCESS_TEAM_DOMAIN and ACCESS_AUD)"] };
  const checked = checkRequest(body);
  if (checked.problems.length || !checked.type || !checked.params) return { problems: checked.problems };
  return {
    problems: [],
    job: { id: makeJobId(now, rand), type: checked.type, params: checked.params, requested_by: email, requested_at: isoSeconds(now), verified: true, ...(checked.target_machine ? { target_machine: checked.target_machine } : {}) },
  };
}

/** The state shown for a job: the runner's status wins; otherwise a cancel mark in the queue; no status object means still queued. */
export function effectiveState(job: JobRecord, status: JobStatus | null | undefined): string {
  if (status?.state && status.state !== "queued") return status.state;
  if (job.cancelled_at) return "cancelled";
  return "queued";
}

/** Text for a queued job addressed to a machine that is not reporting in ("waiting for Desktop (big)"), else null. Untargeted jobs never wait on a named machine. */
export function waitingFor(job: JobRecord, state: string, beats: MachineBeat[] | undefined, now: number): string | null {
  if (state !== "queued" || !job.target_machine) return null;
  const spec = MACHINES.find((m) => m.name === job.target_machine);
  if (!spec) return `Waiting for ${job.target_machine}`;
  const m = machineStatus(spec, beats, now);
  if (m.status === "offline") return `Waiting for ${m.label}, which is offline`;
  if (m.status === "running") return `Waiting for ${m.label} to finish its current job`;
  return `Waiting for ${m.label} to pick it up`;
}

/** One job of a type at a time: a new request is refused while an earlier one of the same type is queued, claimed or running and under 6 h old. */
export function rateLimitProblem(type: string, jobs: JobRecord[], statuses: Record<string, JobStatus | null | undefined>, now: number): string | null {
  for (const job of jobs) {
    if (job.type !== type) continue;
    const at = parseUtc(job.requested_at);
    if (at === null || now - at >= MAX_AGE_MS) continue;
    const state = effectiveState(job, statuses[job.id]);
    if (!isTerminal(state)) return `${specFor(type)?.label ?? "That job"} is already ${stateLabel(state).toLowerCase()}. Wait for it to finish, or cancel it.`;
  }
  return null;
}

/** Cancel is allowed only while the runner has not claimed the job. */
export function cancelProblem(job: JobRecord | undefined, status: JobStatus | null | undefined): string | null {
  if (!job) return "no such job";
  if (job.cancelled_at) return "job is already cancelled";
  const state = effectiveState(job, status);
  if (state !== "queued") return `job is already ${state}; only a queued job can be cancelled`;
  return null;
}

/** Automatic background jobs (every 15 / 30 minutes) are dropped first when the queue is full, so they never push the pricing moments off the Run page. */
const BACKGROUND_TYPES = new Set(["watch", "odds_reprice"]);

export const LIST_NON_CRON = 50;
export const LIST_CRON = 20;
export const LIST_WINDOW_MS = 48 * 3_600_000;

/**
 * Jobs the Run page lists, newest first: the last LIST_NON_CRON non-cron jobs requested in the previous 48 h plus the last LIST_CRON cron jobs
 * (the cron tick is ~190 of every 200 queue entries, so a plain slice would hide every phone-triggered job).
 */
export function listWindow(jobs: JobRecord[], now: number): JobRecord[] {
  const manual = jobs.filter((j) => j.requested_by !== "cron" && now - (parseUtc(j.requested_at) ?? 0) <= LIST_WINDOW_MS).slice(-LIST_NON_CRON);
  const cron = jobs.filter((j) => j.requested_by === "cron").slice(-LIST_CRON);
  const keep = new Set([...manual, ...cron]);
  return jobs.filter((j) => keep.has(j)).reverse();
}

export function trimQueue(jobs: JobRecord[]): JobRecord[] {
  if (jobs.length <= MAX_QUEUE) return jobs;
  let excess = jobs.length - MAX_QUEUE;
  const kept = jobs.filter((job) => {
    if (excess > 0 && BACKGROUND_TYPES.has(job.type) && job.requested_by === "cron") {
      excess -= 1;
      return false;
    }
    return true;
  });
  return kept.length > MAX_QUEUE ? kept.slice(kept.length - MAX_QUEUE) : kept;
}
