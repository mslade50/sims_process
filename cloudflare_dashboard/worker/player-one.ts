/**
 * Single-golfer lookup (round 3, Players page LCP). Additive and read-only.
 *
 *   GET /api/golfprice/player_profiles/golfer/search-index.json?id=<dg id | historical-...>[&gen=<20 hex generation>]
 *
 * The Players page used to download the whole derived search index (~430 KB) before it could draw one golfer. This endpoint cuts that
 * same index down to the requested golfer and answers a document in the SAME `player_search.v1` shape with `count: 1`, so the client
 * decodes it with the unchanged `parseSearchIndex`. The path ends in `search-index.json` on purpose: the client's verbatim-name rule and
 * the derived-pointer cache class already key on that file name.
 *
 * Nothing is written. The slice is cut from the derived object that `/search-index.json` already stores; when that object does not
 * exist yet this answers 404 and the client falls back to the full index (which builds and stores it). Existing endpoints are untouched.
 */
import { CACHE_IMMUTABLE, CACHE_POINTER, respondWithObject, type ServableObject } from "./data-http.ts";
import { SEARCH_SCHEMA, derivedKey, readPointerGeneration, textChunks, type SearchIndex } from "./player-search.ts";

export const PLAYER_ONE_PATH = "player_profiles/golfer/search-index.json";
const GENERATION = /^[a-f0-9]{20}$/;
const PLAYER_ID = /^(?:[1-9][0-9]{0,9}|historical-[a-z0-9]{1,40})$/;

type Bucket = Pick<R2Bucket, "get">;

function failure(status: number, error: string): Response {
  return new Response(JSON.stringify({ ok: false, error }), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } });
}

/** Row of the golfer with this id (dg id or profile id), or -1. dg ids are stored as running deltas. */
export function findRow(index: SearchIndex, id: string): number {
  const profileIds = new Map<number, string>(index.pid.map(([row, pid]) => [row, pid]));
  let dg = 0;
  for (let row = 0; row < index.name.length; row++) {
    const delta = index.dg[row];
    let dgId: number | null = null;
    if (typeof delta === "number") { dg += delta; dgId = dg; }
    const profileId = profileIds.get(row) ?? (dgId === null ? null : String(dgId));
    if (profileId === id) return row;
  }
  return -1;
}

/** The index reduced to one row. Sparse columns are re-keyed to row 0; tour sets and keys are cut to the one used entry (the full tables are ~60 KB); statuses and meta are kept whole. */
export function sliceIndex(index: SearchIndex, row: number): SearchIndex {
  let dg = 0, dgId: number | null = null;
  for (let r = 0; r <= row; r++) {
    const delta = index.dg[r];
    if (typeof delta === "number") { dg += delta; if (r === row) dgId = dg; }
  }
  const sparse = <T>(col: Array<[number, ...unknown[]]>): T[] => col.filter((e) => e[0] === row).map((e) => [0, ...e.slice(1)] as unknown as T);
  return {
    ...index,
    count: 1,
    dg: [dgId],
    pid: sparse<[number, string]>(index.pid),
    name: [index.name[row]],
    dn: sparse<[number, string]>(index.dn),
    alias: sparse<[number, string[]]>(index.alias),
    tour_sets: [index.tour_sets[index.tours[row]] ?? []],
    tours: [0],
    geo: sparse<[number, string | null, string | null]>(index.geo),
    status: [index.status[row]],
    n: [index.n[row] ?? null],
    last: [index.last[row] ?? null],
    keys: [index.keys[index.key[row]] ?? ""],
    key: [0],
    am: index.am.includes(row) ? [0] : [],
  };
}

const cached = new Map<string, SearchIndex>();   // parsed index per generation (isolate memory; tiny)

async function loadIndex(bucket: Bucket, generation: string): Promise<{ index: SearchIndex; etag: string } | null> {
  const object = (await bucket.get(derivedKey(generation))) as unknown as (ServableObject & { httpMetadata?: { contentEncoding?: string } }) | null;
  if (!object?.body) return null;
  const known = cached.get(generation);
  const etag = object.httpEtag ?? generation;
  if (known) { void object.body.cancel?.().catch?.(() => undefined); return { index: known, etag }; }
  const gzip = (object.httpMetadata?.contentEncoding ?? "").toLowerCase() === "gzip";
  let text = "";
  for await (const chunk of textChunks(object.body as ReadableStream<Uint8Array>, gzip)) text += chunk;
  const parsed = JSON.parse(text) as SearchIndex;
  if (parsed.schema !== SEARCH_SCHEMA || !Array.isArray(parsed.name)) return null;
  if (cached.size > 2) cached.clear();
  cached.set(generation, parsed);
  return { index: parsed, etag };
}

export async function handlePlayerOne(request: Request, bucket: Bucket): Promise<Response> {
  const params = new URL(request.url).searchParams;
  const id = params.get("id");
  if (!id || !PLAYER_ID.test(id)) return failure(400, "id must be a golfer id");
  const requestedGen = params.get("gen");
  if (requestedGen !== null && !GENERATION.test(requestedGen)) return failure(400, "Invalid generation");
  const generation = requestedGen ?? (await readPointerGeneration(bucket as never));
  if (!generation) return failure(404, "not published yet");
  let loaded: { index: SearchIndex; etag: string } | null;
  try {
    loaded = await loadIndex(bucket, generation);
  } catch (error) {
    console.error(`golfer lookup failed for ${generation}: ${error instanceof Error ? error.message : String(error)}`);
    return failure(502, "golfer lookup could not be read");
  }
  if (!loaded) return failure(404, "not published yet");
  const row = findRow(loaded.index, id);
  if (row < 0) return failure(404, "golfer not in this catalog");
  const body = JSON.stringify(sliceIndex(loaded.index, row));
  const etag = `"${loaded.etag.replace(/"/g, "")}-${id}"`;
  const object: ServableObject = { httpEtag: etag, body: new Response(body).body, writeHttpMetadata: () => undefined };
  return respondWithObject(request, object, {
    contentType: "application/json; charset=utf-8",
    cacheControl: requestedGen ? CACHE_IMMUTABLE : CACHE_POINTER,
    extra: { "x-search-generation": generation },
  });
}
