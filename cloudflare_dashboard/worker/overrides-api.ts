/**
 * Worker endpoints for the golfprice "Model inputs" area (owner authorization, October 5, 2026).
 *
 *   GET    /api/golfprice/<path>        read-only: golfprice/<path> in the dashboard bucket (index, manifest, runs/..., overrides_schema.json)
 *   GET    /api/overrides               the active override list (overrides/active.json) plus its etag
 *   POST   /api/overrides               create one override (server sets id, created_at, author; bounds validated server-side)
 *   DELETE /api/overrides/<id>?mode=remove|expire   remove it, or expire it now (the record stays, golfprice ignores it after expires_at)
 *
 * Writes need a Cloudflare Access identity (see access.ts), a JSON body and a same-origin request. Every write appends an immutable
 * overrides/history/<time>-<action>-<id>.json entry carrying the Access user, then replaces overrides/active.json with an etag guard.
 * golfprice only reads active.json; nothing else in the bucket is written here.
 */
import { accessIdentity, type AccessEnv, type Identity } from "./access";
import { HISTORY_PREFIX, OVERRIDES_KEY, buildRecord, isoSeconds, isExpired, parseUtc, type OverrideRecord } from "../app/overrides-rules";

export interface ApiEnv extends AccessEnv {
  DASHBOARD_DATA?: R2Bucket;
}

const MAX_BODY_BYTES = 8192;
const MAX_ACTIVE_RECORDS = 300;
const JSON_HEADERS = { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" };

function json(body: unknown, status = 200, extra: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), { status, headers: { ...JSON_HEADERS, ...extra } });
}

function fail(status: number, error: string, problems?: string[]): Response {
  return json({ ok: false, error, ...(problems ? { problems } : {}) }, status);
}

async function readActive(bucket: R2Bucket): Promise<{ records: OverrideRecord[]; etag: string | null }> {
  const object = await bucket.get(OVERRIDES_KEY);
  if (!object) return { records: [], etag: null };
  let parsed: unknown;
  try {
    parsed = JSON.parse(await object.text());
  } catch {
    throw new Error("overrides/active.json is not valid JSON; refusing to overwrite it");
  }
  if (!Array.isArray(parsed)) throw new Error("overrides/active.json is not a list; refusing to overwrite it");
  return { records: parsed as OverrideRecord[], etag: object.etag };
}

/** Etag-guarded replace of active.json. Returns the new object's etag (needed to guard a compensating rollback), or null if the guard failed. */
async function writeActive(bucket: R2Bucket, records: OverrideRecord[], etag: string | null): Promise<{ etag: string | null } | null> {
  const result = await bucket.put(OVERRIDES_KEY, JSON.stringify(records, null, 1), {
    httpMetadata: { contentType: "application/json; charset=utf-8", cacheControl: "no-store" },
    onlyIf: etag ? { etagMatches: etag } : { etagDoesNotMatch: "*" },
  });
  return result === null ? null : { etag: (result as { etag?: string }).etag ?? null };
}

function randomSuffix(): string {
  return [...crypto.getRandomValues(new Uint8Array(3))].map((b) => b.toString(16).padStart(2, "0")).join("");
}

type Action = "create" | "remove" | "expire";

async function appendHistory(bucket: R2Bucket, action: Action, now: number, who: Identity, record: OverrideRecord, before?: OverrideRecord) {
  const stamp = new Date(now).toISOString().replace(/[-:]/g, "").replace(".", "");
  // Random suffix: two writes in the same millisecond must not collide on the key (the put is create-only).
  const key = `${HISTORY_PREFIX}${stamp}-${action}-${record.id}-${randomSuffix()}.json`;
  const entry = { schema: "golfprice.override_history.v1", action, at: isoSeconds(now), by: who, record, ...(before ? { before } : {}) };
  const put = await bucket.put(key, JSON.stringify(entry, null, 1), {
    httpMetadata: { contentType: "application/json; charset=utf-8", cacheControl: "no-store" },
    onlyIf: { etagDoesNotMatch: "*" },
  });
  if (put === null) throw new Error("history key already exists");
  return key;
}

type Mutation = { records: OverrideRecord[]; previous: OverrideRecord[]; etag: string | null };

/** Read-modify-write of active.json with an etag guard (three attempts). `change` returns the new list or a Response to stop. */
async function mutate(bucket: R2Bucket, change: (records: OverrideRecord[]) => OverrideRecord[] | Response): Promise<Mutation | Response> {
  for (let attempt = 0; attempt < 3; attempt += 1) {
    const { records, etag } = await readActive(bucket);
    const next = change(records);
    if (next instanceof Response) return next;
    const written = await writeActive(bucket, next, etag);
    if (written) return { records: next, previous: records, etag: written.etag };
  }
  return fail(409, "overrides/active.json changed while saving; reload and try again");
}

const HISTORY_ATTEMPTS = 3;

/**
 * The audit entry must exist for every live change. Retry the history put; if it still fails, restore the previous records with an etag guard
 * (only if nobody else changed active.json since) and report "change not applied". If the restore also fails, the change is live without an
 * audit entry: say so, with the record id, so the owner can reconcile.
 */
async function commitWithAudit(bucket: R2Bucket, action: Action, now: number, who: Identity, record: OverrideRecord, applied: Mutation, before?: OverrideRecord): Promise<{ historyKey: string } | Response> {
  let lastError = "history write failed";
  for (let attempt = 0; attempt < HISTORY_ATTEMPTS; attempt += 1) {
    try {
      return { historyKey: await appendHistory(bucket, action, now, who, record, before) };
    } catch (error) {
      lastError = error instanceof Error ? error.message : lastError;
    }
  }
  try {
    const restored = await bucket.put(OVERRIDES_KEY, JSON.stringify(applied.previous, null, 1), {
      httpMetadata: { contentType: "application/json; charset=utf-8", cacheControl: "no-store" },
      onlyIf: applied.etag ? { etagMatches: applied.etag } : { etagDoesNotMatch: "*" },
    });
    if (restored !== null) return fail(500, `change not applied: the audit entry could not be written (${lastError})`);
  } catch {
    // fall through to the audit-missing report
  }
  return fail(500, `audit missing; live override present: ${record.id} (${action}); the audit entry could not be written (${lastError}) and the previous list could not be restored`);
}

/** Host of an Origin header, or null when it is not a URL (e.g. the literal "null"), which never matches the request host. */
function originHost(origin: string): string | null {
  try {
    return new URL(origin).host;
  } catch {
    return null;
  }
}

export async function handleOverridesApi(request: Request, env: ApiEnv, now = Date.now()): Promise<Response | null> {
  const url = new URL(request.url);
  const path = url.pathname;

  if (path.startsWith("/api/golfprice/")) {
    if (request.method !== "GET" && request.method !== "HEAD") return fail(405, "read-only path");
    let requested: string;
    try {
      requested = decodeURIComponent(path.slice("/api/golfprice/".length));
    } catch {
      return fail(400, "Invalid golfprice path");
    }
    if (!requested || requested.includes("..") || requested.startsWith("/") || requested.includes("\\")) return fail(400, "Invalid golfprice path");
    if (!env.DASHBOARD_DATA) return fail(503, "dashboard bucket is not bound");
    const object = await env.DASHBOARD_DATA.get(`golfprice/${requested}`);
    if (!object) return fail(404, "not published yet");
    const headers = new Headers();
    object.writeHttpMetadata(headers);
    headers.set("content-type", requested.endsWith(".csv") ? "text/csv; charset=utf-8" : "application/json; charset=utf-8");
    headers.set("cache-control", /(^|\/)(manifest|index|latest|dossier-review)\.json$/.test(requested) ? "no-cache" : "public, max-age=300, stale-while-revalidate=3600");
    headers.set("etag", object.httpEtag);
    // R2 dossiers are already gzip encoded. Prevent Workers from applying
    // the same content encoding a second time to their stored bytes.
    const responseInit: ResponseInit & { encodeBody?: "manual" } = { headers };
    if (headers.has("content-encoding")) responseInit.encodeBody = "manual";
    return new Response(request.method === "HEAD" ? null : object.body, responseInit);
  }

  if (path !== "/api/overrides" && !path.startsWith("/api/overrides/")) return null;
  if (!env.DASHBOARD_DATA) return fail(503, "dashboard bucket is not bound");
  const bucket = env.DASHBOARD_DATA;

  if (path === "/api/overrides" && request.method === "GET") {
    try {
      const { records, etag } = await readActive(bucket);
      return json({ ok: true, records, etag, now: isoSeconds(now) });
    } catch (error) {
      return fail(500, error instanceof Error ? error.message : "could not read overrides");
    }
  }

  if (path === "/api/overrides/history" && request.method === "GET") {
    // Newest 25 audit entries (keys start with a UTC timestamp, so key order is time order).
    const listing = await bucket.list({ prefix: HISTORY_PREFIX, limit: 1000 });
    const keys = listing.objects.map((o) => o.key).sort().slice(-25).reverse();
    const entries = (await Promise.all(keys.map(async (key) => (await bucket.get(key))?.json().catch(() => null)))).filter(Boolean);
    return json({ ok: true, entries, truncated: listing.truncated });
  }

  const isCreate = path === "/api/overrides" && request.method === "POST";
  const isDelete = path.startsWith("/api/overrides/") && request.method === "DELETE";
  if (!isCreate && !isDelete) return fail(405, "method not allowed");

  // Writes: Access identity first, then origin, then body.
  const identity = await accessIdentity(request, env, now);
  if (!identity) return fail(401, "A valid Cloudflare Access identity is required to change overrides");
  // Overrides move prices, so only a signature-verified identity may write (the Worker needs ACCESS_TEAM_DOMAIN and ACCESS_AUD).
  if (!identity.verified) return fail(403, "The Access identity could not be verified (the Worker needs ACCESS_TEAM_DOMAIN and ACCESS_AUD); override writes are refused");
  const origin = request.headers.get("origin");
  if (origin && originHost(origin) !== url.host) return fail(403, "cross-origin write refused");

  try {
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
      const built = buildRecord(body as Record<string, unknown>, identity.email, now);
      if (!built.record) return fail(422, "override rejected by the hard bounds", built.problems);
      const record = built.record;
      const result = await mutate(bucket, (records) => {
        const live = records.filter((r) => !isExpired(r, now - 7 * 86_400_000));
        if (live.length >= MAX_ACTIVE_RECORDS) return fail(409, `too many overrides (limit ${MAX_ACTIVE_RECORDS}); remove some first`);
        return [...live, record];
      });
      if (result instanceof Response) return result;
      const audited = await commitWithAudit(bucket, "create", now, identity, record, result);
      if (audited instanceof Response) return audited;
      return json({ ok: true, record, records: result.records, history_key: audited.historyKey }, 201);
    }

    let id: string;
    try {
      id = decodeURIComponent(path.slice("/api/overrides/".length));
    } catch {
      return fail(400, "invalid override id");
    }
    const mode = url.searchParams.get("mode") ?? "remove";
    if (!id || id.length > 80 || !/^[A-Za-z0-9._:-]+$/.test(id)) return fail(400, "invalid override id");
    if (mode !== "remove" && mode !== "expire") return fail(400, "mode must be remove or expire");
    let before: OverrideRecord | undefined;
    let after: OverrideRecord | undefined;
    const result = await mutate(bucket, (records) => {
      before = records.find((r) => r.id === id);
      if (!before) return fail(404, "no such override");
      if (mode === "remove") return records.filter((r) => r.id !== id);
      if (isExpired(before, now)) return fail(409, "override is already expired");
      const created = parseUtc(before.created_at) ?? now;
      after = { ...before, expires_at: isoSeconds(Math.max(now, created + 1000)) };
      return records.map((r) => (r.id === id ? (after as OverrideRecord) : r));
    });
    if (result instanceof Response) return result;
    const audited = await commitWithAudit(bucket, mode === "remove" ? "remove" : "expire", now, identity, after ?? (before as OverrideRecord), result, before);
    if (audited instanceof Response) return audited;
    return json({ ok: true, id, mode, records: result.records, history_key: audited.historyKey });
  } catch (error) {
    return fail(500, error instanceof Error ? error.message : "override write failed");
  }
}
