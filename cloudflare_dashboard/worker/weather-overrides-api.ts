/**
 * Worker endpoints for the owner's hour-by-hour weather edits (see app/weather-overrides-rules.ts for the contract with golfprice).
 *
 *   GET /api/weather-overrides/<event_safe>   {ok, doc: <saved object as v2, or null>, etag}  (a stored v1 object is returned as {pre: ...})
 *   PUT /api/weather-overrides/<event_safe>   body {phase: "pre"|"live", hours, note, etag, after_round (live only)}: validate, replace only that phase's section,
 *                                              write the whole v2 doc to history, then replace the main object with an etag guard (409 on conflict)
 *
 * event_safe is the event uid with ":" replaced by "_" (for example euro_2026140_2026-10-15). Writes need a signature-verified Cloudflare Access identity,
 * the same as the Model inputs overrides. The server sets schema, event_uid, updated_at and updated_by; the client never does.
 */
import { accessIdentity } from "./access";
import type { ApiEnv } from "./overrides-api";
import { EVENT_SAFE_RE, WEATHER_OVERRIDE_PREFIX, historyKey, mergeSection, normalizeStored, validatePhase, validateWeatherOverride } from "../app/weather-overrides-rules";

const PREFIX = "/api/weather-overrides/";
const MAX_BODY_BYTES = 64 * 1024;
const JSON_HEADERS = { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" };
const R2_JSON = { contentType: "application/json; charset=utf-8", cacheControl: "no-store" };

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: JSON_HEADERS });
const fail = (status: number, error: string, problems?: string[]) => json({ ok: false, error, ...(problems ? { problems } : {}) }, status);

function originHost(origin: string): string | null {
  try {
    return new URL(origin).host;
  } catch {
    return null;
  }
}

export async function handleWeatherOverridesApi(request: Request, env: ApiEnv, now = Date.now()): Promise<Response | null> {
  const url = new URL(request.url);
  if (!url.pathname.startsWith(PREFIX)) return null;
  if (!env.DASHBOARD_DATA) return fail(503, "dashboard bucket is not bound");
  const bucket = env.DASHBOARD_DATA;
  let safe: string;
  try {
    safe = decodeURIComponent(url.pathname.slice(PREFIX.length));
  } catch {
    return fail(400, "Invalid event");
  }
  if (!EVENT_SAFE_RE.test(safe)) return fail(400, "Invalid event: expected something like euro_2026140_2026-10-15");
  const key = `${WEATHER_OVERRIDE_PREFIX}${safe}.json`;
  const eventUid = safe.replace(/_/g, ":");

  if (request.method === "GET") {
    const object = await bucket.get(key);
    if (!object) return json({ ok: true, doc: null, etag: null });
    let doc: unknown;
    try {
      doc = normalizeStored(JSON.parse(await object.text()), eventUid);
    } catch {
      return fail(500, "The saved weather edits are not valid JSON; refusing to show them");
    }
    return json({ ok: true, doc, etag: object.etag });
  }
  if (request.method !== "PUT") return fail(405, "method not allowed");

  const identity = await accessIdentity(request, env, now);
  if (!identity) return fail(401, "A valid Cloudflare Access identity is required to save weather edits");
  if (!identity.verified) return fail(403, "The Access identity could not be verified (the Worker needs ACCESS_TEAM_DOMAIN and ACCESS_AUD); weather edits are refused");
  const origin = request.headers.get("origin");
  if (origin && originHost(origin) !== url.host) return fail(403, "cross-origin write refused");
  if (!(request.headers.get("content-type") ?? "").toLowerCase().startsWith("application/json")) return fail(415, "send application/json");
  const text = await request.text();
  if (text.length > MAX_BODY_BYTES) return fail(413, "body too large");
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    return fail(400, "The request body is not valid JSON.");
  }
  const raw = body as Record<string, unknown> | null;
  const phase = validatePhase(body);
  if (phase.problems.length || !phase.phase) return fail(400, phase.problems[0], phase.problems);
  const checked = validateWeatherOverride(body);
  if (checked.problems.length) return fail(400, checked.problems[0], checked.problems);
  const etag = raw && "etag" in raw ? raw.etag : undefined;
  if (etag !== null && typeof etag !== "string") return fail(400, "The page's version of the saved edits is missing. Reload the page and try again.");

  try {
    // Check the guard before writing history, so a conflict leaves no history copy of an edit that never went live.
    const current = await bucket.get(key);
    if ((current?.etag ?? null) !== etag) return fail(409, "The saved edits changed since you opened this page. Reload the page, then make your changes again.");
    let stored: unknown = null;
    if (current) {
      try {
        stored = JSON.parse(await current.text());
      } catch {
        return fail(500, "The saved weather edits are not valid JSON; refusing to overwrite them");
      }
    }
    const stamp = { hours: checked.hours, note: checked.note, updated_at: new Date(now).toISOString().replace(/\.\d{3}Z$/, "Z"), updated_by: identity.email };
    const doc = mergeSection(normalizeStored(stored, eventUid), phase.phase, stamp, phase.after_round, eventUid);
    const payload = JSON.stringify(doc, null, 1);
    const history = await bucket.put(historyKey(now, safe), payload, { httpMetadata: R2_JSON, onlyIf: { etagDoesNotMatch: "*" } });
    if (history === null) return fail(409, "Another save happened in the same second. Try again.");
    const written = await bucket.put(key, payload, { httpMetadata: R2_JSON, onlyIf: etag ? { etagMatches: etag } : { etagDoesNotMatch: "*" } });
    if (written === null) return fail(409, "The saved edits changed while saving. Reload the page, then make your changes again.");
    return json({ ok: true, doc, etag: (written as { etag?: string }).etag ?? null, history_key: historyKey(now, safe) });
  } catch (error) {
    return fail(500, error instanceof Error ? error.message : "weather edit write failed");
  }
}
