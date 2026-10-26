/**
 * Passive shadow observer on the 15-minute cron (golf_ops docs/OBSERVABILITY.md section 8). Sends NOTHING: no Telegram, no email, no repair, no
 * network call at all. It only reads R2 (ops/registry.json if present, jobs/queue.json, jobs/status/*, jobs/heartbeat/*, jobs/roles.json) and
 * writes ops/shadow/incidents/<yyyy-mm-dd>/<incident_key>.json (create-only, listed first so a known key costs no PUT) plus a rolling,
 * etag-guarded ops/shadow/summary.json that an older overlapping run can never overwrite.
 *
 * OFF unless the var OPS_SHADOW_ENABLED is exactly "1" (declared "0" in wrangler.jsonc, so enabling is a reviewed config change).
 * A failed R2 read is never treated as an absent object: a queue read failure skips the tick, a failed status/heartbeat read omits that record.
 */
import { dueJobs } from "./cron-rules.ts";
import { SHADOW_PREFIX, SHADOW_REGISTRY_KEY, SHADOW_SCHEMA, SHADOW_SUMMARY_KEY, buildSummary, evaluateShadow, type Finding, type ShadowBeat, type ShadowJob } from "./shadow-rules.ts";
import { HEARTBEAT_PREFIX, QUEUE_KEY, ROLES_KEY, STATUS_PREFIX, effectiveState, machinesWithRoles, parseRoles, parseUtc, rolesInvalid, type JobRecord, type JobStatus } from "../app/jobs-rules.ts";

export interface ShadowEnv {
  DASHBOARD_DATA?: R2Bucket;
  OPS_SHADOW_ENABLED?: string;
}

export type ShadowOutcome = { enabled: boolean; findings: Finding[]; written: number; notes: string[] };

export const MONITOR_HORIZON_MS = 48 * 3_600_000; // covers the 3 h missed-slot window plus the 6 h rate-limit lookback, and the failed-job window
export const MAX_NEW_INCIDENTS_PER_TICK = 25;
const JSON_PUT = { contentType: "application/json; charset=utf-8", cacheControl: "no-store" };

/** null = the object does not exist. Any R2 or parse error throws, so the caller can tell "absent" from "could not read". */
async function readJson(bucket: R2Bucket, key: string): Promise<{ value: unknown; etag: string } | null> {
  const object = await bucket.get(key);
  if (!object) return null;
  return { value: JSON.parse(await object.text()), etag: object.etag };
}

async function optional(bucket: R2Bucket, key: string, notes: string[]): Promise<{ value: unknown; etag: string } | null> {
  try {
    return await readJson(bucket, key);
  } catch {
    notes.push(`could not read ${key}`);
    return null;
  }
}

async function listKeys(bucket: R2Bucket, prefix: string): Promise<Set<string>> {
  const keys = new Set<string>();
  let cursor: string | undefined;
  do {
    const page = await bucket.list({ prefix, cursor, limit: 1000 });
    for (const o of page.objects) keys.add(o.key);
    cursor = page.truncated ? (page as { cursor?: string }).cursor : undefined;
  } while (cursor);
  return keys;
}

const finishedAt = (job: JobRecord, status: JobStatus | null): number | null => parseUtc(status?.finished_at) ?? parseUtc(status?.updated_at) ?? parseUtc(job.cancelled_at) ?? null;

export async function runShadow(env: ShadowEnv, now: number): Promise<ShadowOutcome> {
  const outcome: ShadowOutcome = { enabled: env.OPS_SHADOW_ENABLED === "1", findings: [], written: 0, notes: [] };
  if (!outcome.enabled) {
    outcome.notes.push("shadow observer disabled (OPS_SHADOW_ENABLED is not 1)");
    return outcome;
  }
  const bucket = env.DASHBOARD_DATA;
  if (!bucket) {
    outcome.notes.push("dashboard bucket is not bound");
    return outcome;
  }

  let records: JobRecord[];
  try {
    const queue = (await readJson(bucket, QUEUE_KEY))?.value as { jobs?: unknown } | null | undefined;
    records = Array.isArray(queue?.jobs) ? (queue.jobs as JobRecord[]) : [];
  } catch {
    outcome.notes.push("jobs/queue.json could not be read; tick skipped, nothing written");
    return outcome;
  }

  const registry = (await optional(bucket, SHADOW_REGISTRY_KEY, outcome.notes))?.value ?? null;
  const rolesValue = (await optional(bucket, ROLES_KEY, outcome.notes))?.value ?? null;
  if (rolesInvalid(rolesValue)) outcome.notes.push("roles_invalid: jobs/roles.json is not a valid {primary, backup}; using the built-in roles");
  const machines = machinesWithRoles(parseRoles(rolesValue));

  // Independent of the Run page's list limits: every queue record inside the monitoring horizon, whatever its type or requester.
  const inHorizon = records.filter((j) => now - (parseUtc(j.requested_at) ?? 0) <= MONITOR_HORIZON_MS);
  let omitted = 0;
  const omittedTypes = new Set<string>();
  const resolved = await Promise.all(
    inHorizon.map(async (job): Promise<ShadowJob | null> => {
      try {
        const status = (await readJson(bucket, `${STATUS_PREFIX}${job.id}.json`))?.value as JobStatus | null | undefined;
        const st = status && typeof status === "object" ? status : null;
        return { id: job.id, type: job.type, requested_at: job.requested_at, requested_by: job.requested_by, state: effectiveState(job, st), finished_at: finishedAt(job, st) };
      } catch {
        omitted += 1;
        omittedTypes.add(job.type);
        return null;
      }
    }),
  );
  const jobs = resolved.filter((j): j is ShadowJob => j !== null);
  if (omitted) outcome.notes.push(`${omitted} job status read(s) failed; those jobs are omitted this tick`);

  const beats: ShadowBeat[] = [];
  for (const m of machines) {
    try {
      beats.push({ machine: m.name, role: m.role, beat: (await readJson(bucket, `${HEARTBEAT_PREFIX}${m.name}.json`))?.value as { at?: string } | null });
    } catch {
      outcome.notes.push(`heartbeat read failed for ${m.name}; not evaluated`);
    }
  }

  outcome.findings = evaluateShadow({
    now,
    jobs,
    created: inHorizon.map((j) => ({ type: j.type, requested_at: j.requested_at })),
    beats,
    dueAt: dueJobs,
  });
  // An omitted job might have been the rate-limit blocker of its type, so no missed-slot finding is made for that type this tick.
  outcome.findings = outcome.findings.filter((f) => !(f.kind === "occurrence_missed" && omittedTypes.has(f.subject)));

  // Create-only: list each date prefix once and PUT just the keys that are not there yet (bounded per tick).
  const known = new Set<string>();
  for (const date of new Set(outcome.findings.map((f) => f.date))) {
    for (const key of await listKeys(bucket, `${SHADOW_PREFIX}incidents/${date}/`)) known.add(key);
  }
  const fresh = outcome.findings.filter((f) => !known.has(`${SHADOW_PREFIX}incidents/${f.date}/${f.key}.json`));
  for (const finding of fresh.slice(0, MAX_NEW_INCIDENTS_PER_TICK)) {
    const body = { schema: SHADOW_SCHEMA, ...finding, first_seen: new Date(now).toISOString() };
    const written = await bucket.put(`${SHADOW_PREFIX}incidents/${finding.date}/${finding.key}.json`, JSON.stringify(body, null, 1), { httpMetadata: JSON_PUT, onlyIf: { etagDoesNotMatch: "*" } });
    if (written !== null) outcome.written += 1;
  }
  if (fresh.length > MAX_NEW_INCIDENTS_PER_TICK) outcome.notes.push(`capped: ${fresh.length - MAX_NEW_INCIDENTS_PER_TICK} new finding(s) deferred to the next tick`);

  // Rolling summary: replace only the version we read, and never an object that is already newer than this run.
  const summary = buildSummary(now, outcome.findings, registry, { jobs: jobs.length, machines: beats.length });
  let current: { value: unknown; etag: string } | null;
  try {
    current = await readJson(bucket, SHADOW_SUMMARY_KEY);
  } catch {
    outcome.notes.push("summary.json could not be read; summary not updated");
    return outcome;
  }
  const existingAt = Date.parse(String((current?.value as { updated_at?: unknown } | undefined)?.updated_at ?? ""));
  if (Number.isFinite(existingAt) && existingAt > now) {
    outcome.notes.push("summary.json is newer than this run; not overwritten");
    return outcome;
  }
  const put = await bucket.put(SHADOW_SUMMARY_KEY, JSON.stringify(summary, null, 1), {
    httpMetadata: JSON_PUT,
    onlyIf: current ? { etagMatches: current.etag } : { etagDoesNotMatch: "*" },
  });
  if (put === null) outcome.notes.push("summary.json changed during this run (lost the race); not overwritten");
  return outcome;
}
