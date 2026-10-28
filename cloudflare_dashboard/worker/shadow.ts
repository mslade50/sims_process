/**
 * Passive shadow observer on the 15-minute cron (golf_ops docs/OBSERVABILITY.md section 8). Sends NOTHING: no Telegram, no email, no repair, no
 * network call at all. It only reads R2 (ops/registry.json if present, jobs/queue.json, jobs/status/*, jobs/heartbeat/*, jobs/roles.json) and
 * writes under ops/shadow/ only.
 *
 * Storage model (no cap, no deletes, bounded R2 operations per tick whatever the finding volume):
 *   - Every tick writes ALL findings it has not yet recorded in ONE create-only object ops/shadow/batches/<yyyy-mm-dd>/<tickISO>-<rand>.json
 *     (a single PUT). Findings are durable in the tick that detects them, and the observer never deletes anything.
 *   - Dedup uses a per-day index ops/shadow/index/<yyyy-mm-dd>.json = { keys: { <incident_key>: <batch key> } }. A tick reads the indexes of
 *     today and the two previous UTC days (a job finding can be re-detected for 48 h) with 3 GETs, then adds its new keys to today's index by
 *     etag compare-and-swap (union merge, up to 3 attempts). The batch is written BEFORE the index, so the index never names a missing batch.
 *   - If the index update loses all attempts, nothing is lost: the next tick may record the same incident_key again in a new batch.
 *     CONSUMERS MUST DEDUPE BY incident_key (the earliest first_seen wins). summary.json reports `index_conflict` when that happened.
 *   - ops/shadow/summary.json is an etag-guarded rolling summary (never replaced by an older run), with the count of new findings this tick.
 *
 * OFF unless the var OPS_SHADOW_ENABLED is exactly "1" (declared "0" in wrangler.jsonc, so enabling is a reviewed config change).
 * Reads are three-way: absent (null), unreadable content (JSON parse error or wrong shape: a finding is written) and an R2 failure (no
 * finding; the gap is noted as degraded coverage and, for a status object, its slot window stays suppressed). An R2 failure is never "absent".
 */
import { dueJobs } from "./cron-rules.ts";
import { MISSED_LOOKBACK_MS, SHADOW_REGISTRY_KEY, SHADOW_SCHEMA, SHADOW_SUMMARY_KEY, buildSummary, evaluateShadow, type Finding, type ShadowBeat, type ShadowJob } from "./shadow-rules.ts";
import { HEARTBEAT_PREFIX, MAX_QUEUE, JOB_ID_RE, QUEUE_KEY, STATUS_PREFIX, effectiveState, loadRoles, machinesWithRoles, parseUtc, specFor, type JobRecord, type JobStatus } from "../app/jobs-rules.ts";

export interface ShadowEnv {
  DASHBOARD_DATA?: R2Bucket;
  OPS_SHADOW_ENABLED?: string;
}

export type ShadowOutcome = { enabled: boolean; findings: Finding[]; written: number; notes: string[] };

export const MONITOR_HORIZON_MS = 48 * 3_600_000; // covers the 3 h missed-slot window plus the 6 h rate-limit lookback, and the failed-job window
export const INDEX_PREFIX = "ops/shadow/index/";
export const BATCH_PREFIX = "ops/shadow/batches/";
const INDEX_ATTEMPTS = 3;
const INDEX_DAYS = 3;
const JSON_PUT = { contentType: "application/json; charset=utf-8", cacheControl: "no-store" };

type Read = { kind: "absent" } | { kind: "ok"; value: unknown; etag: string; uploaded: unknown } | { kind: "parse_error"; etag: string } | { kind: "read_error" };

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
    return { kind: "ok", value: JSON.parse(text), etag: object.etag, uploaded: object.uploaded };
  } catch {
    return { kind: "parse_error", etag: object.etag };
  }
}

// Producers use UTC ISO timestamps. Reject calendar rollover as well as malformed dates.
const clockTolerance = 2 * 60_000;
const timestamp = (value: unknown, now: number): number | null => {
  const ms = parseUtc(value);
  if (ms === null || ms > now + clockTolerance || new Date(ms).toISOString().slice(0, 19) !== (value as string).slice(0, 19)) return null;
  return ms;
};
const object = (value: unknown): value is Record<string, unknown> => !!value && typeof value === "object" && !Array.isArray(value);
const validJob = (value: unknown, now: number): value is JobRecord => {
  if (!object(value) || typeof value.id !== "string" || !JOB_ID_RE.test(value.id) || !specFor(value.type) || typeof value.requested_by !== "string" || !value.requested_by.trim()) return false;
  const requested = timestamp(value.requested_at, now);
  if (requested === null) return false;
  if (value.cancelled_at !== undefined && (timestamp(value.cancelled_at, now) === null || (timestamp(value.cancelled_at, now) as number) < requested)) return false;
  return true;
};
const validStatus = (value: unknown, job: JobRecord, now: number): value is JobStatus => {
  if (!object(value) || value.id !== job.id || typeof value.state !== "string" || !["queued", "claimed", "running", "done", "failed", "cancelled", "expired"].includes(value.state)) return false;
  let prior = timestamp(job.requested_at, now) as number;
  for (const key of ["claimed_at", "started_at", "finished_at", "updated_at"]) {
    if (value[key] === undefined || value[key] === null) continue;
    const at = timestamp(value[key], now);
    if (at === null || at < prior) return false;
    prior = at;
  }
  return true;
};
const finishedAt = (job: JobRecord, status: JobStatus | null): number | null =>
  job.cancelled_at && (!status || status.state === "queued") ? parseUtc(job.cancelled_at) : parseUtc(status?.finished_at) ?? parseUtc(status?.updated_at);
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
  let records: JobRecord[] = [];
  let evaluate = true;
  let occurrenceCoverage = true;
  if (queueRead.kind === "read_error") {
    outcome.notes.push("jobs/queue.json could not be read (R2 error); job and occurrence checks skipped");
    degraded.push("queue_read_error");
    evaluate = false;
  }
  if (queueRead.kind === "parse_error") {
    evaluate = false;
  } else if (queueRead.kind === "ok") {
    const queue = queueRead.value as { jobs?: unknown } | null;
    if (queue && typeof queue === "object" && Array.isArray(queue.jobs)) {
      // Entries that are not job records are dropped (and reported) so the valid jobs and the heartbeats are still checked.
      const all = queue.jobs as unknown[];
      if (all.length >= MAX_QUEUE) {
        occurrenceCoverage = false;
        degraded.push("queue_history_incomplete");
        outcome.notes.push("queue reached its retention bound; removed background occurrences may be missing, so missed-occurrence checks are skipped");
      }
      records = all.filter((j): j is JobRecord => validJob(j, now));
      const bad = all.length - records.length;
      if (bad > 0) {
        occurrenceCoverage = false; // An invalid record could be the missing creation or a historical blocker.
        degraded.push(`queue_entry_invalid:${bad}`);
        extra.push({ key: `queue-entry-invalid-${utcDay(now)}`, kind: "queue_entry_invalid", severity: "warn", date: utcDay(now), subject: QUEUE_KEY, detail: `${bad} entr${bad === 1 ? "y" : "ies"} in jobs/queue.json are not job records and were skipped`, expires_at: now + MISSED_LOOKBACK_MS });
      }
    } else evaluate = false; // valid JSON of the wrong shape is not an empty queue
  }
  if (!evaluate && queueRead.kind !== "read_error") {
    degraded.push("queue_unreadable");
    extra.push({ key: `queue-unreadable-${utcDay(now)}`, kind: "queue_unreadable", severity: "high", date: utcDay(now), subject: QUEUE_KEY, detail: "jobs/queue.json is not a valid queue object; job and missed-slot checks are skipped (heartbeats are still checked)", expires_at: now + MISSED_LOOKBACK_MS });
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
      if (read.kind === "read_error" || read.kind === "parse_error" || (read.kind === "ok" && !validStatus(read.value, job, now))) {
        unreadable.push({ id: job.id, type: job.type, requested_at: job.requested_at, corrupt: read.kind !== "read_error" });
        return null;
      }
      const status = read.kind === "ok" ? (read.value as JobStatus | null) : null;
      const st = status && typeof status === "object" ? status : null;
      const finish = finishedAt(job, st);
      const fromQueue = !!job.cancelled_at && (!st || st.state === "queued");
      const publication = fromQueue ? queueRead : read;
      const uploaded = publication.kind === "ok" && publication.uploaded instanceof Date ? publication.uploaded.getTime() : NaN;
      // A terminal status cannot have been published before its own declared timestamps or the request.
      const latestStatusTime = Math.max(parseUtc(job.requested_at) as number, fromQueue ? finish ?? 0 : 0, ...[st?.claimed_at, st?.started_at, st?.finished_at, st?.updated_at].map((v) => parseUtc(v) ?? 0));
      const published = Number.isFinite(uploaded) && uploaded >= latestStatusTime && uploaded <= now + clockTolerance ? uploaded : null;
      const queueUpload = queueRead.kind === "ok" && queueRead.uploaded instanceof Date ? queueRead.uploaded.getTime() : NaN;
      const queueVisible = Number.isFinite(queueUpload) && queueUpload >= (parseUtc(job.requested_at) as number) && queueUpload <= now + clockTolerance ? queueUpload : null;
      // Either the containing queue or this job's status can prove it existed by that publication time.
      const visibility = [queueVisible, published].filter((at): at is number => at !== null);
      return { id: job.id, type: job.type, requested_at: job.requested_at, requested_by: job.requested_by, visible_at: visibility.length ? Math.min(...visibility) : null, state_published_at: fromQueue ? null : published, queue_cancelled_at: parseUtc(job.cancelled_at), state: effectiveState(job, st), finished_at: finish, terminal_source: fromQueue ? "queue" : "status", terminal_published_at: published };

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
    } else if (read.kind === "parse_error" || (read.kind === "ok" && (!object(read.value) || read.value.machine !== m.name || timestamp(read.value.at, now) === null))) {
      degraded.push(`heartbeat_unreadable:${m.name}`);
      extra.push({ key: `heartbeat-unreadable-${m.name}-${utcDay(now)}`, kind: "heartbeat_unreadable", severity: m.role === "primary" ? "high" : "warn", date: utcDay(now), subject: m.name, detail: `heartbeat object of ${m.name} is invalid`, expires_at: now + MISSED_LOOKBACK_MS });
    } else {
      beats.push({ machine: m.name, role: m.role, beat: read.kind === "ok" ? (read.value as { at?: string } | null) : null });
    }
  }

  outcome.findings = [
    ...extra,
    // With an unusable queue only the heartbeat check can run: no jobs, no creation evidence, no expected occurrences.
    ...evaluateShadow(
      evaluate
        ? { now, jobs, created: inHorizon.map((j) => ({ type: j.type, requested_at: j.requested_at, requested_by: j.requested_by })), beats, unreadable, uncertain: degraded, dueAt: occurrenceCoverage ? dueJobs : () => [] }
        : { now, jobs: [], created: [], beats, unreadable: [], dueAt: () => [] },
    ),
  ];

  // One batch object with every finding not yet in the indexes of the last three UTC days, then today's index updated by CAS.
  const today = utcDay(now);
  const days = Array.from({ length: INDEX_DAYS }, (_, n) => utcDay(now - n * 86_400_000));
  const indexes = await Promise.all(days.map((d) => readObject(bucket, `${INDEX_PREFIX}${d}.json`)));
  const recorded = new Set<string>();
  const validIndex = (read: Read): boolean => {
    if (read.kind !== "ok") return read.kind === "absent";
    const value = read.value;
    return object(value) && value.schema === SHADOW_SCHEMA && object(value.keys) && Object.values(value.keys).every((pointer) => typeof pointer === "string" && /^ops\/shadow\/batches\/\d{4}-\d{2}-\d{2}\/[^/]+\.json$/.test(pointer));
  };
  const keysOf = (read: Read): Record<string, string> => validIndex(read) && read.kind === "ok" ? (read.value as { keys: Record<string, string> }).keys : {};
  indexes.forEach((read, n) => {
    if (read.kind === "read_error") degraded.push(`index_read_error:${days[n]}`);
    if (read.kind === "parse_error" || (read.kind === "ok" && !validIndex(read))) degraded.push(`index_unreadable:${days[n]}`);
    for (const key of Object.keys(keysOf(read))) recorded.add(key);
  });
  const fresh = outcome.findings.filter((f) => !recorded.has(f.key));
  let batchKey: string | null = null;
  let indexConflict = false;
  if (fresh.length > 0) {
    batchKey = `${BATCH_PREFIX}${today}/${new Date(now).toISOString().replace(/[-:.]/g, "")}-${Math.random().toString(36).slice(2, 10)}.json`;
    const body = { schema: SHADOW_SCHEMA, tick: new Date(now).toISOString(), findings: fresh.map((f) => ({ key: f.key, kind: f.kind, severity: f.severity, date: f.date, subject: f.subject, detail: f.detail, first_seen: new Date(now).toISOString() })) };
    const stored = await bucket.put(batchKey, JSON.stringify(body, null, 1), { httpMetadata: JSON_PUT, onlyIf: { etagDoesNotMatch: "*" } });
    if (stored === null) {
      outcome.notes.push("batch object name collided; findings will be retried next tick");
      batchKey = null;
    } else {
      outcome.written = fresh.length;
      const mine: Record<string, string> = Object.fromEntries(fresh.map((f) => [f.key, batchKey as string]));
      let current: Read = indexes[0];
      // A corrupt index is never replaced with only this tick's keys (that would forget everything else it recorded): skip the update, accept duplicates.
      indexConflict = true;
      for (let attempt = 0; attempt < INDEX_ATTEMPTS && indexConflict && current.kind !== "read_error" && validIndex(current); attempt += 1) {
        const merged = { schema: SHADOW_SCHEMA, keys: { ...mine, ...keysOf(current) } };
        const put = await bucket.put(`${INDEX_PREFIX}${today}.json`, JSON.stringify(merged), { httpMetadata: JSON_PUT, onlyIf: current.kind === "absent" ? { etagDoesNotMatch: "*" } : { etagMatches: current.etag } });
        if (put !== null) {
          indexConflict = false;
          break;
        }
        if (attempt < INDEX_ATTEMPTS - 1) current = await readObject(bucket, `${INDEX_PREFIX}${today}.json`);
      }
      // A re-read can turn up a corrupt object (another writer, or damage): preserve it and say so.
      if ((current.kind === "parse_error" || (current.kind === "ok" && !validIndex(current))) && !degraded.includes(`index_unreadable:${today}`)) degraded.push(`index_unreadable:${today}`);
      if (indexConflict) {
        degraded.push("index_conflict");
        outcome.notes.push("index_conflict: today's index could not be updated; the next tick may record the same incident_key again (consumers dedupe by incident_key)");
      }
    }
  }

  // Rolling summary: replace only the version we read (a corrupt one included), and never an object that is already newer than this run.
  const summary = buildSummary(now, outcome.findings, registry, { jobs: jobs.length, machines: beats.length }, degraded, { new_findings: outcome.written, batch: batchKey, index_conflict: indexConflict });
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
