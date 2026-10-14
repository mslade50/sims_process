/**
 * Persisted run diffs for the What changed strip (October 2026).
 *
 *   GET /api/golfprice/changes.json?event=<event uid>&from=<explain run key>&to=<explain run key>
 *
 * The strip used to download the earlier run's explain document (about 1 MB) just to compare two numbers per player. This endpoint loads the two documents from the
 * dashboard bucket, runs the SAME pure diff the app uses (app/changes-rules.ts: price movers, the causes behind each move, sportsbook moves, field, weather, scoring)
 * and stores the result under golfprice/explain/derived/changes/<hash of from+to>.json. Run documents are written once per run key, so the diff of two keys is
 * immutable: later requests are one small R2 read, the browser caches the answer for a year, and If-None-Match answers 304.
 *
 * Errors are JSON: 400 bad input (missing/invalid event, from or to; keys outside the event's explain folder; latest.json, which is not a fixed document; from equal to to;
 * two documents that cannot be compared), 404 when a run document is missing, 502 when the diff cannot be built. The client falls back to its own diff on any error.
 */
import { CACHE_IMMUTABLE, respondWithObject, type ServableObject } from "./data-http.ts";
import { CHANGES_LOGIC_VERSION, changesPayload, whatChanged } from "../app/changes-rules.ts";
import { parseExplain, type ExplainDoc } from "../app/explain-rules.ts";

export const CHANGES_PATH = "changes.json";
export const DERIVED_PREFIX = "golfprice/explain/derived/changes/";
const EVENT = /^[A-Za-z0-9_.:-]{1,80}$/;
const RUN_KEY = /^golfprice\/explain\/([A-Za-z0-9_.:-]{1,80})\/([A-Za-z0-9_.:-]{1,120}\.json)$/;

type Bucket = Pick<R2Bucket, "get" | "put">;
const JSON_TYPE = "application/json; charset=utf-8";

function failure(status: number, error: string): Response {
  return new Response(JSON.stringify({ ok: false, error }), { status, headers: { "content-type": JSON_TYPE, "cache-control": "no-store" } });
}

/** 20 hex characters of SHA-256 over "v<logic version>\nfrom\nto": the immutable key segment (the cache-control rule in data-http.ts recognises it).
 *  The logic version makes a code change rebuild diffs instead of serving ones cached by older code. */
export async function changesHash(from: string, to: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(`v${CHANGES_LOGIC_VERSION}\n${from}\n${to}`));
  return [...new Uint8Array(digest).slice(0, 10)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
export const derivedKey = (hash: string): string => `${DERIVED_PREFIX}${hash}.json`;

/** null = acceptable, otherwise the 400 message. */
export function validateChangesQuery(event: string | null, from: string | null, to: string | null): string | null {
  if (!event || !EVENT.test(event)) return "event is missing or invalid";
  if (!from || !to) return "from and to are required";
  const a = RUN_KEY.exec(from);
  const b = RUN_KEY.exec(to);
  if (!a || !b) return "from and to must be golfprice/explain/<event>/<run>.json keys";
  const folder = event.replace(/:/g, "_");
  if (a[1] !== folder || b[1] !== folder) return "from and to must belong to the event";
  if (a[2] === "latest.json" || b[2] === "latest.json") return "latest.json is not a fixed document; use the run keys from the index";
  if (from === to) return "from and to are the same run";
  return null;
}

class Missing extends Error {}

async function readDoc(bucket: Bucket, key: string): Promise<ExplainDoc> {
  const object = await bucket.get(key);
  if (!object) throw new Missing(key);
  let bytes = new Uint8Array(await object.arrayBuffer());
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    bytes = new Uint8Array(await new Response(stream).arrayBuffer());
  }
  const doc = parseExplain(JSON.parse(new TextDecoder().decode(bytes)));
  if (!doc) throw new Error(`${key} is not an explain document`);
  return doc;
}

export async function handleChanges(request: Request, bucket: Bucket): Promise<Response> {
  const url = new URL(request.url);
  const event = url.searchParams.get("event");
  const from = url.searchParams.get("from");
  const to = url.searchParams.get("to");
  const problem = validateChangesQuery(event, from, to);
  if (problem) return failure(400, problem);
  const hash = await changesHash(from as string, to as string);
  const key = derivedKey(hash);
  const respond = (object: ServableObject) => respondWithObject(request, object, { contentType: JSON_TYPE, cacheControl: CACHE_IMMUTABLE, extra: { "x-changes-key": key } });

  const existing = await bucket.get(key);
  if (existing) return respond(existing as unknown as ServableObject);

  let text: string;
  try {
    const [prev, cur] = await Promise.all([readDoc(bucket, from as string), readDoc(bucket, to as string)]);
    const change = whatChanged(cur, prev);
    if (!change) return failure(400, "the two documents cannot be compared (different events or the same run)");
    text = JSON.stringify(changesPayload(event as string, from as string, to as string, change));
  } catch (error) {
    if (error instanceof Missing) return failure(404, "a run document is not published");
    console.error(`changes build failed for ${from} -> ${to}: ${error instanceof Error ? error.message : String(error)}`);
    return failure(502, "the diff could not be built");
  }
  try {
    await bucket.put(key, text, { httpMetadata: { contentType: JSON_TYPE, cacheControl: CACHE_IMMUTABLE } });
  } catch (error) {
    // Serve it anyway; the next request builds again.
    console.error(`changes store failed for ${key}: ${error instanceof Error ? error.message : String(error)}`);
  }
  const headers = new Headers({ "content-type": JSON_TYPE, "cache-control": CACHE_IMMUTABLE, "x-changes-key": key });
  return new Response(request.method === "HEAD" ? null : text, { headers });
}
