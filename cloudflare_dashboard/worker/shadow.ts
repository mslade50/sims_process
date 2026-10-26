/**
 * Passive shadow observer on the 15-minute cron (golf_ops docs/OBSERVABILITY.md section 8). Sends NOTHING: no Telegram, no email, no repair, no
 * network call at all. It only reads R2 (ops/registry.json if present, jobs/queue.json, jobs/status/*, jobs/heartbeat/*, jobs/roles.json) and
 * writes ops/shadow/incidents/<yyyy-mm-dd>/<incident_key>.json (create-only, so a finding keeps its first_seen) plus a rolling ops/shadow/summary.json.
 *
 * OFF unless the var OPS_SHADOW_ENABLED is exactly "1", so deploying this code is inert until the owner turns it on.
 */
import { dueJobs } from "./cron-rules.ts";
import { SHADOW_PREFIX, SHADOW_REGISTRY_KEY, SHADOW_SCHEMA, SHADOW_SUMMARY_KEY, buildSummary, evaluateShadow, type Finding, type ShadowBeat, type ShadowJob } from "./shadow-rules.ts";
import { HEARTBEAT_PREFIX, QUEUE_KEY, ROLES_KEY, STATUS_PREFIX, effectiveState, listWindow, machinesWithRoles, parseRoles, parseUtc, type JobRecord, type JobStatus } from "../app/jobs-rules.ts";

export interface ShadowEnv {
  DASHBOARD_DATA?: R2Bucket;
  OPS_SHADOW_ENABLED?: string;
}

export type ShadowOutcome = { enabled: boolean; findings: Finding[]; written: number; notes: string[] };

const JSON_PUT = { contentType: "application/json; charset=utf-8", cacheControl: "no-store" };

async function readJson(bucket: R2Bucket, key: string): Promise<unknown> {
  try {
    const object = await bucket.get(key);
    return object ? JSON.parse(await object.text()) : null;
  } catch {
    return null;
  }
}

export async function runShadow(env: ShadowEnv, now: number): Promise<ShadowOutcome> {
  const outcome: ShadowOutcome = { enabled: env.OPS_SHADOW_ENABLED === "1", findings: [], written: 0, notes: [] };
  if (!outcome.enabled) return outcome;
  const bucket = env.DASHBOARD_DATA;
  if (!bucket) {
    outcome.notes.push("dashboard bucket is not bound");
    return outcome;
  }

  const registry = await readJson(bucket, SHADOW_REGISTRY_KEY);
  const roles = parseRoles(await readJson(bucket, ROLES_KEY));
  const queue = (await readJson(bucket, QUEUE_KEY)) as { jobs?: unknown } | null;
  const records = Array.isArray(queue?.jobs) ? (queue.jobs as JobRecord[]) : [];
  const listed = listWindow(records, now);
  const jobs: ShadowJob[] = await Promise.all(
    listed.map(async (job) => {
      const status = (await readJson(bucket, `${STATUS_PREFIX}${job.id}.json`)) as JobStatus | null;
      return { id: job.id, type: job.type, requested_at: job.requested_at, requested_by: job.requested_by, state: effectiveState(job, status && typeof status === "object" ? status : null) };
    }),
  );
  const machines = machinesWithRoles(roles);
  const beats: ShadowBeat[] = await Promise.all(
    machines.map(async (m) => ({ machine: m.name, role: m.role, beat: (await readJson(bucket, `${HEARTBEAT_PREFIX}${m.name}.json`)) as { at?: string } | null })),
  );

  outcome.findings = evaluateShadow({
    now,
    jobs: jobs.filter((j) => now - (parseUtc(j.requested_at) ?? 0) <= 48 * 3_600_000),
    created: records.map((j) => ({ type: j.type, requested_at: j.requested_at })),
    beats,
    dueAt: dueJobs,
  });

  for (const finding of outcome.findings) {
    const body = { schema: SHADOW_SCHEMA, ...finding, first_seen: new Date(now).toISOString() };
    const written = await bucket.put(`${SHADOW_PREFIX}incidents/${finding.date}/${finding.key}.json`, JSON.stringify(body, null, 1), {
      httpMetadata: JSON_PUT,
      onlyIf: { etagDoesNotMatch: "*" },
    });
    if (written !== null) outcome.written += 1;
  }
  await bucket.put(SHADOW_SUMMARY_KEY, JSON.stringify(buildSummary(now, outcome.findings, registry, { jobs: jobs.length, machines: beats.length }), null, 1), { httpMetadata: JSON_PUT });
  return outcome;
}
