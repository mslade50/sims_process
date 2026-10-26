/**
 * Passive shadow observer on the 15-minute cron (golf_ops docs/OBSERVABILITY.md section 8). Sends NOTHING: no Telegram, no email, no repair, no
 * network call at all. It only reads R2 (ops/registry.json if present, jobs/queue.json, jobs/status/*, jobs/heartbeat/*, jobs/roles.json) and
 * writes ops/shadow/incidents/<yyyy-mm-dd>/<incident_key>.json (create-only, listed first so a known key costs no PUT) plus a rolling,
 * etag-guarded ops/shadow/summary.json that an older overlapping run can never overwrite.
 *
 * OFF unless the var OPS_SHADOW_ENABLED is exactly "1" (declared "0" in wrangler.jsonc, so enabling is a reviewed config change).
 * Reads are three-way: absent (null), unreadable content (JSON parse error or wrong shape: a finding is written) and an R2 failure (no
 * finding; the gap is noted as degraded coverage and, for a status object, its slot window stays suppressed). An R2 failure is never "absent".
 * Findings deferred by the per-tick write cap are kept in ops/shadow/pending.json and drained first next tick, even after their job left the scan.
 */
import { dueJobs } from "./cron-rules.ts";
import { MISSED_LOOKBACK_MS, SHADOW_PREFIX, SHADOW_REGISTRY_KEY, SHADOW_SCHEMA, SHADOW_SUMMARY_KEY, buildSummary, evaluateShadow, type Finding, type ShadowBeat, type ShadowJob } from "./shadow-rules.ts";
import { HEARTBEAT_PREFIX, QUEUE_KEY, STATUS_PREFIX, effectiveState, loadRoles, machinesWithRoles, parseUtc, type JobRecord, type JobStatus } from "../app/jobs-rules.ts";

export interface ShadowEnv {
  DASHBOARD_DATA?: R2Bucket;
  OPS_SHADOW_ENABLED?: string;
}

export type ShadowOutcome = { enabled: boolean; findings: Finding[]; written: number; notes: string[] };

export const MONITOR_HORIZON_MS = 48 * 3_600_000; // covers the 3 h missed-slot window plus the 6 h rate-limit lookback, and the failed-job window
export const MAX_NEW_INCIDENTS_PER_TICK = 25;
export const SHADOW_PENDING_KEY = "ops/shadow/pending.json";
const MAX_PENDING = 500;
const JSON_PUT = { contentType: "application/json; charset=utf-8", cacheControl: "no-store" };

type Read = { kind: "absent" } | { kind: "ok"; value: unknown; etag: string } | { kind: "parse_error"; etag: string } | { kind: "read_error" };

async function readObject(bucket: R2Bucket, key: string): Promise<Read> {
  let object: R2ObjectBody | null;
  let text = "";
  try {
    object = await bucket.get(key);
    if (object) text = await object.text();
  } catch {
    return { kind: "read_error" };
  }
  if (!object) return { kind: "absent" };
  try {
    return { kind: "ok", value: JSON.parse(text), etag: object.etag };
  } catch {
    return { kind: "parse_error", etag: object.etag };
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
const utcDay = (ms: number) => new Date(ms).toISOString().slice(0, 10);

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
  const degraded: string[] = [];
  const extra: Finding[] = [];

  const queueRead = await readObject(bucket, QUEUE_KEY);
  if (queueRead.kind === "read_error") {
    outcome.notes.push("jobs/queue.json could not be read (R2 error); tick skipped, nothing written");
    return outcome;
  }
  let records: JobRecord[] = [];
  let evaluate = true;
  if (queueRead.kind === "parse_error") {
    evaluate = false;
  } else if (queueRead.kind === "ok") {
    const queue = queueRead.value as { jobs?: unknown } | null;
    if (queue && typeof queue === "object" && Array.isArray(queue.jobs)) records = queue.jobs as JobRecord[];
    else evaluate = false; // valid JSON of the wrong shape is not an empty queue
  }
  if (!evaluate) {
    degraded.push("queue_unreadable");
    extra.push({ key: "queue-unreadable", kind: "queue_unreadable", severity: "high", date: utcDay(now), subject: QUEUE_KEY, detail: "jobs/queue.json is not a valid queue object; job and missed-slot checks are skipped (heartbeats are still checked)", expires_at: now + MISSED_LOOKBACK_MS });
  }

  const registryRead = await readObject(bucket, SHADOW_REGISTRY_KEY);
  if (registryRead.kind === "read_error" || registryRead.kind === "parse_error") outcome.notes.push(`ops/registry.json ${registryRead.kind === "read_error" ? "could not be read" : "is not valid JSON"}`);
  const registry = registryRead.kind === "ok" ? registryRead.value : null;
  const loaded = await loadRoles(bucket);
  if (loaded.problem) {
    outcome.notes.push(`${loaded.problem}: jobs/roles.json ${loaded.problem === "roles_read_error" ? "could not be read" : "is not a valid {primary, backup}"}; using the built-in roles`);
    degraded.push(loaded.problem);
  }
  const machines = machinesWithRoles(loaded.roles);

  // Independent of the Run page's list limits: every queue record inside the monitoring horizon, whatever its type or requester.
  const inHorizon = records.filter((j) => now - (parseUtc(j.requested_at) ?? 0) <= MONITOR_HORIZON_MS);
  const unreadable: { id: string; type: string; requested_at: string; corrupt: boolean }[] = [];
  const resolved = await Promise.all(
    inHorizon.map(async (job): Promise<ShadowJob | null> => {
      const read = await readObject(bucket, `${STATUS_PREFIX}${job.id}.json`);
      if (read.kind === "read_error" || read.kind === "parse_error") {
        unreadable.push({ id: job.id, type: job.type, requested_at: job.requested_at, corrupt: read.kind === "parse_error" });
        return null;
      }
      const status = read.kind === "ok" ? (read.value as JobStatus | null) : null;
      const st = status && typeof status === "object" ? status : null;
      return { id: job.id, type: job.type, requested_at: job.requested_at, requested_by: job.requested_by, state: effectiveState(job, st), finished_at: finishedAt(job, st) };
    }),
  );
  const jobs = resolved.filter((j): j is ShadowJob => j !== null);
  if (unreadable.length) {
    const corrupt = unreadable.filter((u) => u.corrupt).length;
    if (corrupt) degraded.push(`status_unreadable:${corrupt}`);
    if (unreadable.length - corrupt) degraded.push(`status_read_error:${unreadable.length - corrupt}`);
    outcome.notes.push(`${unreadable.length} job status object(s) unreadable (${corrupt} corrupt); their missed-slot windows are suppressed`);
  }

  const beats: ShadowBeat[] = [];
  for (const m of machines) {
    const read = await readObject(bucket, `${HEARTBEAT_PREFIX}${m.name}.json`);
    if (read.kind === "read_error") {
      degraded.push(`heartbeat_read_error:${m.name}`);
      outcome.notes.push(`heartbeat read failed for ${m.name}; not evaluated`);
    } else if (read.kind === "parse_error") {
      degraded.push(`heartbeat_unreadable:${m.name}`);
      extra.push({ key: `heartbeat-unreadable-${m.name}`, kind: "heartbeat_unreadable", severity: m.role === "primary" ? "high" : "warn", date: utcDay(now), subject: m.name, detail: `heartbeat object of ${m.name} is not valid JSON`, expires_at: now + MISSED_LOOKBACK_MS });
    } else {
      beats.push({ machine: m.name, role: m.role, beat: read.kind === "ok" ? (read.value as { at?: string } | null) : null });
    }
  }

  outcome.findings = [
    ...extra,
    // With an unusable queue only the heartbeat check can run: no jobs, no creation evidence, no expected occurrences.
    ...evaluateShadow(
      evaluate
        ? { now, jobs, created: inHorizon.map((j) => ({ type: j.type, requested_at: j.requested_at })), beats, unreadable, dueAt: dueJobs }
        : { now, jobs: [], created: [], beats, unreadable: [], dueAt: () => [] },
    ),
  ];

  // Create-only: list each date prefix once and PUT just the keys that are not there yet, bounded per tick. Findings that expire soonest
  // (missed slots about to leave the lookback) go first, so the cap can delay a durable finding but never lose a perishable one.
  const pendingRead = await readObject(bucket, SHADOW_PENDING_KEY);
  const carried = pendingRead.kind === "ok" && Array.isArray((pendingRead.value as { findings?: unknown })?.findings) ? ((pendingRead.value as { findings: Finding[] }).findings.filter((f) => f && typeof f.key === "string" && typeof f.date === "string")) : [];
  if (pendingRead.kind === "read_error") {
    degraded.push("pending_read_error");
    outcome.notes.push("ops/shadow/pending.json could not be read; deferred findings not drained this tick");
  }
  const byId = new Map<string, Finding>();
  for (const f of [...carried, ...outcome.findings]) byId.set(`${f.date}/${f.key}`, f);
  const candidates = [...byId.values()];
  const known = new Set<string>();
  for (const date of new Set(candidates.map((f) => f.date))) {
    for (const key of await listKeys(bucket, `${SHADOW_PREFIX}incidents/${date}/`)) known.add(key);
  }
  const fresh = candidates
    .filter((f) => !known.has(`${SHADOW_PREFIX}incidents/${f.date}/${f.key}.json`))
    .sort((a, b) => (a.expires_at ?? Number.POSITIVE_INFINITY) - (b.expires_at ?? Number.POSITIVE_INFINITY));
  for (const finding of fresh.slice(0, MAX_NEW_INCIDENTS_PER_TICK)) {
    const { expires_at: _expires, ...stored } = finding;
    const body = { schema: SHADOW_SCHEMA, ...stored, first_seen: new Date(now).toISOString() };
    const written = await bucket.put(`${SHADOW_PREFIX}incidents/${finding.date}/${finding.key}.json`, JSON.stringify(body, null, 1), { httpMetadata: JSON_PUT, onlyIf: { etagDoesNotMatch: "*" } });
    if (written !== null) outcome.written += 1;
  }
  const deferred = fresh.slice(MAX_NEW_INCIDENTS_PER_TICK, MAX_NEW_INCIDENTS_PER_TICK + MAX_PENDING);
  if (fresh.length > MAX_NEW_INCIDENTS_PER_TICK) {
    degraded.push(`write_capped:${fresh.length - MAX_NEW_INCIDENTS_PER_TICK}`);
    outcome.notes.push(`capped: ${fresh.length - MAX_NEW_INCIDENTS_PER_TICK} new finding(s) deferred to the next tick (pending object)`);
    if (fresh.length - MAX_NEW_INCIDENTS_PER_TICK > MAX_PENDING) outcome.notes.push(`pending object full; ${fresh.length - MAX_NEW_INCIDENTS_PER_TICK - MAX_PENDING} finding(s) not retained`);
  }
  // Persist what was deferred (replace the version read; skip when unchanged or when it could not be read).
  if (pendingRead.kind !== "read_error" && (deferred.length > 0 || carried.length > 0)) {
    const put = await bucket.put(SHADOW_PENDING_KEY, JSON.stringify({ schema: SHADOW_SCHEMA, updated_at: new Date(now).toISOString(), findings: deferred }, null, 1), {
      httpMetadata: JSON_PUT,
      onlyIf: pendingRead.kind === "absent" ? { etagDoesNotMatch: "*" } : { etagMatches: pendingRead.etag },
    });
    if (put === null) outcome.notes.push("pending.json changed during this run (lost the race); deferred list not updated");
  }

  // Rolling summary: replace only the version we read (a corrupt one included), and never an object that is already newer than this run.
  const summary = buildSummary(now, outcome.findings, registry, { jobs: jobs.length, machines: beats.length }, degraded);
  const current = await readObject(bucket, SHADOW_SUMMARY_KEY);
  if (current.kind === "read_error") {
    outcome.notes.push("summary.json could not be read; summary not updated");
    return outcome;
  }
  if (current.kind === "ok") {
    const existingAt = Date.parse(String((current.value as { updated_at?: unknown } | null)?.updated_at ?? ""));
    if (Number.isFinite(existingAt) && existingAt > now) {
      outcome.notes.push("summary.json is newer than this run; not overwritten");
      return outcome;
    }
  }
  const put = await bucket.put(SHADOW_SUMMARY_KEY, JSON.stringify(summary, null, 1), {
    httpMetadata: JSON_PUT,
    onlyIf: current.kind === "absent" ? { etagDoesNotMatch: "*" } : { etagMatches: current.etag },
  });
  if (put === null) outcome.notes.push("summary.json changed during this run (lost the race); not overwritten");
  return outcome;
}
