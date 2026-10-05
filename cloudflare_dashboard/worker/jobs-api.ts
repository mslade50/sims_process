/**
 * Worker endpoints for phone-triggered golfprice runs (owner authorization, October 5, 2026).
 *
 *   GET  /api/jobs                 queue + runner statuses + machine heartbeats, newest first
 *   POST /api/jobs                 create one job {type, params} (Access identity must be verified; allow-list and rate limit in app/jobs-rules.ts)
 *   POST /api/jobs/<id>/cancel     mark a still-queued job cancelled (the runner skips it)
 *   GET  /api/jobs/<id>/log        the runner's full log text for a job (jobs/logs/<id>.txt)
 *
 * This Worker owns only jobs/queue.json. The desktop runner owns jobs/status/<id>.json, jobs/logs/<id>.txt and
 * jobs/heartbeat/<machine>.json (written through wrangler, since the Worker host sits behind Cloudflare Access).
 * The Worker never executes anything; a job is a record the runner may choose to run if its type is on the allow-list.
 */
import { accessIdentity, type AccessEnv } from "./access";
import {
  HEARTBEAT_PREFIX,
  JOB_ID_RE,
  LOG_PREFIX,
  MAX_AGE_MS,
  QUEUE_KEY,
  QUEUE_SCHEMA,
  STATUS_PREFIX,
  buildJob,
  cancelProblem,
  effectiveState,
  isoSeconds,
  parseUtc,
  rateLimitProblem,
  trimQueue,
  type JobRecord,
  type JobStatus,
} from "../app/jobs-rules";

export interface JobsEnv extends AccessEnv {
  DASHBOARD_DATA?: R2Bucket;
}

const MAX_BODY_BYTES = 2048;
const LIST_LIMIT = 50;
const JSON_HEADERS = { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" };

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: JSON_HEADERS });
}
function fail(status: number, error: string, problems?: string[]): Response {
  return json({ ok: false, error, ...(problems ? { problems } : {}) }, status);
}

async function readJson(bucket: R2Bucket, key: string): Promise<{ value: unknown; etag: string | null } | null> {
  const object = await bucket.get(key);
  if (!object) return null;
  try {
    return { value: JSON.parse(await object.text()), etag: object.etag };
  } catch {
    return { value: undefined, etag: object.etag };
  }
}

async function readQueue(bucket: R2Bucket): Promise<{ jobs: JobRecord[]; etag: string | null }> {
  const got = await readJson(bucket, QUEUE_KEY);
  if (!got) return { jobs: [], etag: null };
  const value = got.value as { jobs?: unknown } | undefined;
  if (!value || !Array.isArray(value.jobs)) throw new Error("jobs/queue.json is not a valid queue; refusing to overwrite it");
  return { jobs: value.jobs as JobRecord[], etag: got.etag };
}

async function writeQueue(bucket: R2Bucket, jobs: JobRecord[], etag: string | null): Promise<boolean> {
  const body = JSON.stringify({ schema: QUEUE_SCHEMA, jobs: trimQueue(jobs) }, null, 1);
  const result = await bucket.put(QUEUE_KEY, body, {
    httpMetadata: { contentType: "application/json; charset=utf-8", cacheControl: "no-store" },
    onlyIf: etag ? { etagMatches: etag } : { etagDoesNotMatch: "*" },
  });
  return result !== null;
}

async function readStatus(bucket: R2Bucket, id: string): Promise<JobStatus | null> {
  const got = await readJson(bucket, `${STATUS_PREFIX}${id}.json`).catch(() => null);
  return got && got.value && typeof got.value === "object" ? (got.value as JobStatus) : null;
}

function randomHex(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(3));
  return [...bytes].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export async function handleJobsApi(request: Request, env: JobsEnv, now = Date.now()): Promise<Response | null> {
  const url = new URL(request.url);
  const path = url.pathname;
  if (path !== "/api/jobs" && !path.startsWith("/api/jobs/")) return null;
  if (!env.DASHBOARD_DATA) return fail(503, "dashboard bucket is not bound");
  const bucket = env.DASHBOARD_DATA;

  try {
    if (path === "/api/jobs" && request.method === "GET") {
      const { jobs } = await readQueue(bucket);
      const recent = jobs.slice(-LIST_LIMIT).reverse();
      const listing = await bucket.list({ prefix: STATUS_PREFIX, limit: 1000 });
      const present = new Set(listing.objects.map((o) => o.key));
      const statuses = await Promise.all(recent.map(async (job) => (present.has(`${STATUS_PREFIX}${job.id}.json`) ? await readStatus(bucket, job.id) : null)));
      const heartbeatList = await bucket.list({ prefix: HEARTBEAT_PREFIX, limit: 100 });
      const heartbeats = (await Promise.all(heartbeatList.objects.map(async (o) => (await readJson(bucket, o.key).catch(() => null))?.value))).filter(
        (value) => value && typeof value === "object",
      );
      return json({
        ok: true,
        now: isoSeconds(now),
        jobs: recent.map((job, index) => ({ ...job, status: statuses[index], state: effectiveState(job, statuses[index]) })),
        heartbeats,
      });
    }

    const logMatch = /^\/api\/jobs\/([^/]+)\/log$/.exec(path);
    if (logMatch && request.method === "GET") {
      const id = decodeURIComponent(logMatch[1]);
      if (!JOB_ID_RE.test(id)) return fail(400, "invalid job id");
      const object = await bucket.get(`${LOG_PREFIX}${id}.txt`);
      if (!object) return fail(404, "no log yet");
      return new Response(await object.text(), { headers: { "content-type": "text/plain; charset=utf-8", "cache-control": "no-store" } });
    }

    const cancelMatch = /^\/api\/jobs\/([^/]+)\/cancel$/.exec(path);
    const isCreate = path === "/api/jobs" && request.method === "POST";
    const isCancel = !!cancelMatch && request.method === "POST";
    if (!isCreate && !isCancel) return fail(405, "method not allowed");

    // Writes: Access identity first (verified only), then origin, then body.
    const identity = await accessIdentity(request, env, now);
    if (!identity) return fail(401, "A valid Cloudflare Access identity is required to run or cancel jobs");
    if (!identity.verified) return fail(403, "The Access identity could not be verified (the Worker needs ACCESS_TEAM_DOMAIN and ACCESS_AUD); jobs are refused");
    const origin = request.headers.get("origin");
    if (origin && new URL(origin).host !== url.host) return fail(403, "cross-origin write refused");

    if (isCreate) {
      if (!(request.headers.get("content-type") ?? "").toLowerCase().startsWith("application/json")) return fail(415, "send application/json");
      const text = await request.text();
      if (text.length > MAX_BODY_BYTES) return fail(413, "body too large");
      let body: unknown;
      try {
        body = JSON.parse(text);
      } catch {
        return fail(400, "body is not valid JSON");
      }
      const built = buildJob(body, identity.email, identity.verified, now, randomHex());
      if (!built.job) return fail(422, "job rejected", built.problems);
      const job = built.job;
      for (let attempt = 0; attempt < 3; attempt += 1) {
        const { jobs, etag } = await readQueue(bucket);
        const candidates = jobs.filter((j) => j.type === job.type && now - (parseUtc(j.requested_at) ?? 0) < MAX_AGE_MS);
        const statuses: Record<string, JobStatus | null> = {};
        await Promise.all(candidates.map(async (j) => (statuses[j.id] = await readStatus(bucket, j.id))));
        const blocked = rateLimitProblem(job.type, jobs, statuses, now);
        if (blocked) return fail(409, blocked);
        if (await writeQueue(bucket, [...jobs, job], etag)) return json({ ok: true, job }, 201);
      }
      return fail(409, "the job queue changed while saving; try again");
    }

    const id = decodeURIComponent((cancelMatch as RegExpExecArray)[1]);
    if (!JOB_ID_RE.test(id)) return fail(400, "invalid job id");
    for (let attempt = 0; attempt < 3; attempt += 1) {
      const { jobs, etag } = await readQueue(bucket);
      const job = jobs.find((j) => j.id === id);
      const problem = cancelProblem(job, job ? await readStatus(bucket, id) : null);
      if (problem) return fail(job ? 409 : 404, problem);
      const next = jobs.map((j) => (j.id === id ? { ...j, cancelled_at: isoSeconds(now), cancelled_by: identity.email } : j));
      if (await writeQueue(bucket, next, etag)) return json({ ok: true, id });
    }
    return fail(409, "the job queue changed while saving; try again");
  } catch (error) {
    return fail(500, error instanceof Error ? error.message : "job request failed");
  }
}
