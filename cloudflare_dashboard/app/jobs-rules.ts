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
  { type: "monday_settle", group: "Monday", label: "Settle last week", does: "Pulls the finished event and settles last week's prices (arm monitoring), then a health check.", when: "Monday morning, once the event is final.", no_pull: true, supersede: false, after_round: false },
  { type: "monday_week", group: "Monday", label: "Price the new week", does: "Prices the upcoming event with both arms, then signals, report, comparison, dashboard publish and health.", when: "Monday, after the settle job and once the field is known.", no_pull: true, supersede: true, after_round: false },
  { type: "tuesday", group: "Tuesday", label: "Tuesday refresh", does: "Re-prices the week on Tuesday's data, then signals, report, comparison, publish and health.", when: "Tuesday, after the market has moved.", no_pull: true, supersede: true, after_round: false },
  { type: "wednesday", group: "Wednesday", label: "Wednesday reprice", does: "Re-prices with the tee-time gate and the Wednesday-arm view, then signals, report, comparison, publish and health.", when: "Wednesday, once tee times are out.", no_pull: true, supersede: true, after_round: false },
  { type: "thursday", group: "Thursday", label: "Thursday morning price", does: "Final pre-tee pricing run, then signals, report, comparison, publish and health.", when: "Thursday morning before the first tee.", no_pull: true, supersede: true, after_round: false },
  { type: "thursday_close", group: "Thursday", label: "Thursday close", does: "Pre-event live state (nothing in play yet), publish and health.", when: "Thursday, just before the first tee.", no_pull: false, supersede: true, after_round: false },
  { type: "after_round", group: "During event", label: "After a round", does: "Re-prices the live state after a finished round, then publishes and health. Leave the round on automatic to infer it.", when: "Each evening of the event once the round is complete.", no_pull: false, supersede: true, after_round: true },
  { type: "daily_health", group: "Anytime", label: "Health check", does: "Checks expected runs, stale prices, blocking checks and stored files. Changes nothing else.", when: "Any time you want to know if all is well.", no_pull: false, supersede: false, after_round: false },
  { type: "xtour", group: "Monday", label: "Refresh cross-tour base", does: "Rebuilds the cross-tour base from the newest foundation (paced DataGolf pull), then publishes it to R2 so every runner prices on the same base, then a health check. Prices nothing.", when: "Monday after the new foundation snapshot exists, if the Monday job did not refresh it.", no_pull: false, supersede: false, after_round: false },
  { type: "xtour_publish", group: "Anytime", label: "Publish cross-tour base", does: "Publishes the local cross-tour base to R2 (idempotent; LATEST written last). Does not rebuild or price anything.", when: "If another machine reports it cannot see the newest base.", no_pull: false, supersede: false, after_round: false },
  { type: "publish", group: "Anytime", label: "Publish model inputs", does: "Publishes the latest model inputs to this dashboard. Does not price anything.", when: "If the Model inputs page looks out of date.", no_pull: false, supersede: false, after_round: false },
  { type: "odds_reprice", group: "Anytime", label: "Odds check (no re-simulation)", does: "Re-reads the latest odds, recomputes signals against the fairs of the last full run, publishes them here and alerts on a new or moved live signal. Takes seconds.", when: "Runs by itself every 30 minutes from Monday afternoon to the Thursday tee; press it for an immediate look.", no_pull: true, supersede: false, after_round: false },
  { type: "watch", group: "Anytime", label: "Input watch", does: "Checks the field, tee times and forecast against the last full run; re-prices only if something that moves prices changed (rate limited), and runs the pre-tee close just before the first tee.", when: "Runs by itself hourly before the event; press it after a withdrawal or tee-sheet news.", no_pull: true, supersede: false, after_round: false },
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
export function checkRequest(body: unknown): { problems: string[]; type?: string; params?: JobParams } {
  const problems: string[] = [];
  if (!body || typeof body !== "object" || Array.isArray(body)) return { problems: ["body must be a JSON object"] };
  const raw = body as Record<string, unknown>;
  for (const key of Object.keys(raw)) {
    if (!["type", "params"].includes(key)) problems.push(`${key} is set by the server or not allowed`);
  }
  const spec = specFor(raw.type);
  if (!spec) {
    problems.push(`type must be one of: ${JOB_TYPES.map((s) => s.type).join(", ")}`);
    return { problems };
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
  return { problems, type: spec.type, params: clean };
}

export function buildJob(body: unknown, email: string, verified: boolean, now: number, rand: string): { job?: JobRecord; problems: string[] } {
  if (!verified) return { problems: ["the Cloudflare Access identity is not verified (the Worker needs ACCESS_TEAM_DOMAIN and ACCESS_AUD)"] };
  const checked = checkRequest(body);
  if (checked.problems.length || !checked.type || !checked.params) return { problems: checked.problems };
  return {
    problems: [],
    job: { id: makeJobId(now, rand), type: checked.type, params: checked.params, requested_by: email, requested_at: isoSeconds(now), verified: true },
  };
}

/** The state shown for a job: the runner's status wins; otherwise a cancel mark in the queue; no status object means still queued. */
export function effectiveState(job: JobRecord, status: JobStatus | null | undefined): string {
  if (status?.state && status.state !== "queued") return status.state;
  if (job.cancelled_at) return "cancelled";
  return "queued";
}

/** One job of a type at a time: a new request is refused while an earlier one of the same type is queued, claimed or running and under 6 h old. */
export function rateLimitProblem(type: string, jobs: JobRecord[], statuses: Record<string, JobStatus | null | undefined>, now: number): string | null {
  for (const job of jobs) {
    if (job.type !== type) continue;
    const at = parseUtc(job.requested_at);
    if (at === null || now - at >= MAX_AGE_MS) continue;
    const state = effectiveState(job, statuses[job.id]);
    if (!isTerminal(state)) return `a ${type} job (${job.id}) is already ${state}; wait for it to finish or cancel it`;
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

export function trimQueue(jobs: JobRecord[]): JobRecord[] {
  return jobs.length > MAX_QUEUE ? jobs.slice(jobs.length - MAX_QUEUE) : jobs;
}
