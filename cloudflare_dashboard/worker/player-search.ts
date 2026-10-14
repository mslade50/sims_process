/**
 * Derived golfer search index (site reliability pass, October 2026).
 *
 *   GET /api/golfprice/player_profiles/search-index.json[?gen=<20 hex generation>]
 *
 * The published catalog (golfprice/player_profiles/index.json == <generation>/catalog.json) is ~16 MB because every one of its ~16,600
 * golfers carries a full pga_benchmark, archetype and photo block. Search, the player picker and the Players list need identity only.
 * This module reduces the catalog to a column table (a few hundred KB gzipped) WITHOUT changing what the Python publisher writes, and
 * stores the result in this dashboard's own bucket under golfprice/player_profiles/derived/<generation>/search-index.json, so the
 * work happens once per generation. Per-golfer detail (pga_benchmark etc.) is still read from the profile batches when a golfer opens.
 *
 * Memory: the catalog is never JSON.parsed whole (a 16 MB document costs far more than a 128 MB isolate can spare). `scanCatalog`
 * walks the text as it streams, hands over each top-level value and each element of `players` separately, and keeps one golfer
 * (about 1 KB) in memory at a time. CPU is one pass over the characters, paid once per generation.
 *
 * Any failure answers a JSON error (404/502) and the client falls back to the full catalog, so search never depends on this path.
 */
import { CACHE_IMMUTABLE, CACHE_POINTER, respondWithObject, type ServableObject } from "./data-http.ts";

export const SEARCH_INDEX_PATH = "player_profiles/search-index.json";
export const BENCHMARKS_PATH = "player_profiles/benchmarks.json";
export const BENCHMARKS_SCHEMA = "player_benchmarks.v1";
export const MAX_BENCHMARK_IDS = 400;
export const SEARCH_SCHEMA = "player_search.v1";
export const POINTER_KEY = "golfprice/player_profiles/index.json";
const GENERATION = /^[a-f0-9]{20}$/;

/* ------------------------------------------------------------------ streaming scanner */
export type CatalogHandlers = {
  /** Called for every top-level key except `players`, with the parsed value. Return true to stop reading. */
  onMeta?: (key: string, value: unknown) => boolean | void;
  /** Called for each element of the top-level `players` array. Return true to stop reading. */
  onPlayer?: (player: Record<string, unknown>) => boolean | void;
};

const Q = 34, BS = 92, LBRACE = 123, RBRACE = 125, LBRACK = 91, RBRACK = 93, COMMA = 44, COLON = 58;

/**
 * Walk a JSON document shaped `{ "key": value, ..., "players": [ {...}, ... ], ... }` without building it. `chunks` yields text.
 * Handles keys and values split across chunk boundaries. Returns true when a handler asked to stop (the rest is not read).
 */
export async function scanCatalog(chunks: AsyncIterable<string>, handlers: CatalogHandlers): Promise<boolean> {
  let depth = 0;
  let inStr = false, esc = false;
  let state: "key" | "value" | "after" = "key";   // meaning at depth 1
  let readingKey = false, keyParts: string[] = [], curKey = "";
  let playersArray = false;
  let capKind: "meta" | "player" | "prim" | null = null;
  let capParts: string[] = [];
  let primString = false;
  const finishMeta = (text: string): boolean => {
    let value: unknown;
    try { value = JSON.parse(text); } catch { throw new Error(`catalog: unreadable value for "${curKey}"`); }
    return handlers.onMeta?.(curKey, value) === true;
  };
  const finishPlayer = (text: string): boolean => {
    let value: unknown;
    try { value = JSON.parse(text); } catch { throw new Error("catalog: unreadable player"); }
    return !!value && typeof value === "object" && handlers.onPlayer?.(value as Record<string, unknown>) === true;
  };

  for await (const s of chunks) {
    let capStart = capKind ? 0 : -1;
    let keyStart = readingKey ? 0 : -1;
    const n = s.length;
    for (let i = 0; i < n; i++) {
      const c = s.charCodeAt(i);
      if (inStr) {
        if (esc) { esc = false; continue; }
        if (c === BS) { esc = true; continue; }
        if (c !== Q) continue;
        inStr = false;
        if (readingKey) {
          keyParts.push(s.slice(keyStart, i));
          curKey = JSON.parse(`"${keyParts.join("")}"`);
          readingKey = false; keyStart = -1; keyParts = [];
          state = "after";   // waiting for the colon
        } else if (capKind === "prim" && primString && depth === 1) {
          capParts.push(s.slice(capStart, i + 1));
          capKind = null; capStart = -1; primString = false;
          if (finishMeta(capParts.join(""))) return true;
          capParts = []; state = "after";
        }
        continue;
      }
      switch (c) {
        case Q:
          inStr = true;
          if (depth === 1 && state === "key") { readingKey = true; keyStart = i + 1; keyParts = []; }
          else if (depth === 1 && state === "value" && capKind === null) { capKind = "prim"; primString = true; capStart = i; capParts = []; }
          break;
        case LBRACE: case LBRACK:
          depth++;
          if (depth === 2 && state === "value") {
            if (curKey === "players" && c === LBRACK) playersArray = true;
            else { capKind = "meta"; capStart = i; capParts = []; }
          } else if (depth === 3 && playersArray && c === LBRACE) { capKind = "player"; capStart = i; capParts = []; }
          break;
        case RBRACE: case RBRACK:
          if (capKind === "prim" && !primString && depth === 1) {
            capParts.push(s.slice(capStart, i)); capKind = null; capStart = -1;
            if (finishMeta(capParts.join("").trim())) return true;
            capParts = [];
          }
          depth--;
          if (capKind === "meta" && depth === 1) {
            capParts.push(s.slice(capStart, i + 1)); capKind = null; capStart = -1;
            if (finishMeta(capParts.join(""))) return true;
            capParts = []; state = "after";
          } else if (capKind === "player" && depth === 2) {
            capParts.push(s.slice(capStart, i + 1)); capKind = null; capStart = -1;
            if (finishPlayer(capParts.join(""))) return true;
            capParts = [];
          } else if (playersArray && depth === 1) {
            playersArray = false; state = "after";
          }
          if (depth === 0) return false;
          break;
        case COMMA:
          if (depth === 1) {
            if (capKind === "prim" && !primString) {
              capParts.push(s.slice(capStart, i)); capKind = null; capStart = -1;
              if (finishMeta(capParts.join("").trim())) return true;
              capParts = [];
            }
            state = "key";
          }
          break;
        case COLON:
          if (depth === 1 && state === "after") state = "value";
          break;
        default:
          // Bare number / true / false / null as a top-level value.
          if (depth === 1 && state === "value" && capKind === null && c > 32) { capKind = "prim"; primString = false; capStart = i; capParts = []; }
      }
    }
    // Chunk ended mid-capture: keep what we have and continue at offset 0 of the next chunk.
    if (capKind && capStart >= 0) capParts.push(s.slice(capStart));
    if (readingKey && keyStart >= 0) keyParts.push(s.slice(keyStart));
  }
  return false;
}

/* ------------------------------------------------------------------ the compact table */
/**
 * Column-major layout (gzips ~20% better than row-major): one array per field, all the same length except the sparse ones.
 *   dg      golfer ids, delta-encoded against the previous non-null id (the catalog is id-ordered); null for name-only historical golfers
 *   pid     sparse [row, profile_id] pairs, only where profile_id differs from String(dg)
 *   name    display-order raw names
 *   dn      sparse [row, display_name]
 *   alias   sparse [row, [alias...]] with the name itself and duplicates removed
 *   tours   index into tour_sets (a table of distinct tour lists)
 *   geo     sparse [row, country, country_code]
 *   status  index into statuses;  n  rounds;  last  last observation as days since 1970-01-01 (null when none)
 *   key     index into keys (the profile batch files);  am  rows of amateurs
 */
export type SearchIndex = {
  schema: typeof SEARCH_SCHEMA;
  generation: string;
  as_of: string | null;
  count: number;
  statuses: string[];
  tour_sets: string[][];
  keys: string[];
  meta: Record<string, unknown>;
  dg: Array<number | null>;
  pid: Array<[number, string]>;
  name: string[];
  dn: Array<[number, string]>;
  alias: Array<[number, string[]]>;
  tours: number[];
  geo: Array<[number, string | null, string | null]>;
  status: number[];
  n: Array<number | null>;
  last: Array<number | string | null>;
  key: number[];
  am: number[];
  derived: { from: string; built_at: string };
};

const str = (v: unknown): string | null => (typeof v === "string" && v ? v : null);
const DAY = 86_400_000;
/** Plain dates become day numbers; anything else (a full timestamp) is kept as published. */
export const dateToDays = (v: string | null): number | string | null => {
  if (!v) return null;
  if (!/^d{4}-d{2}-d{2}$/.test(v)) return v;
  const t = Date.parse(`${v}T00:00:00Z`);
  return Number.isFinite(t) ? Math.floor(t / DAY) : v;
};
export const daysToDate = (d: number): string => new Date(d * DAY).toISOString().slice(0, 10);

export function createIndexBuilder() {
  const statuses: string[] = [];
  const tourSets: string[][] = [];
  const tourIndex = new Map<string, number>();
  const keys: string[] = [];
  const keyIndex = new Map<string, number>();
  const meta: Record<string, unknown> = {};
  const out = { dg: [] as Array<number | null>, pid: [] as Array<[number, string]>, name: [] as string[], dn: [] as Array<[number, string]>, alias: [] as Array<[number, string[]]>, tours: [] as number[], geo: [] as Array<[number, string | null, string | null]>, status: [] as number[], n: [] as Array<number | null>, last: [] as Array<number | string | null>, key: [] as number[], am: [] as number[] };
  let previousDg = 0;
  return {
    meta(key: string, value: unknown) { meta[key] = value; },
    metaValue: (key: string): unknown => meta[key],
    player(p: Record<string, unknown>) {
      const row = out.name.length;
      const dg = typeof p.dg_id === "number" && Number.isSafeInteger(p.dg_id) ? p.dg_id : null;
      if (dg === null) out.dg.push(null);
      else { out.dg.push(dg - previousDg); previousDg = dg; }
      const pid = str(p.profile_id);
      if (pid !== null && pid !== String(dg)) out.pid.push([row, pid]);
      const name = str(p.name) ?? "";
      out.name.push(name);
      const dn = str(p.display_name);
      if (dn !== null) out.dn.push([row, dn]);
      const aliases = Array.isArray(p.aliases) ? [...new Set((p.aliases as unknown[]).filter((a): a is string => typeof a === "string" && a !== name && a !== ""))] : [];
      if (aliases.length) out.alias.push([row, aliases]);
      const tours = Array.isArray(p.tours) ? (p.tours as unknown[]).filter((t): t is string => typeof t === "string") : [];
      const tk = tours.join(",");
      let ti = tourIndex.get(tk);
      if (ti === undefined) { ti = tourSets.length; tourSets.push(tours); tourIndex.set(tk, ti); }
      out.tours.push(ti);
      const country = str(p.country), cc = str(p.country_code);
      if (country !== null || cc !== null) out.geo.push([row, country, cc]);
      const status = str(p.status) ?? "unavailable";
      let si = statuses.indexOf(status);
      if (si < 0) { statuses.push(status); si = statuses.length - 1; }
      out.status.push(si);
      out.n.push(typeof p.n_rounds === "number" ? p.n_rounds : null);
      out.last.push(dateToDays(str(p.last_observation)));
      const key = str(p.profile_key) ?? "";
      let ki = keyIndex.get(key);
      if (ki === undefined) { ki = keys.length; keys.push(key); keyIndex.set(key, ki); }
      out.key.push(ki);
      if (p.amateur === true) out.am.push(row);
    },
    build(generation: string, from: string, builtAt: string): SearchIndex {
      const { generation: _generation, ...rest } = meta;   // generation is carried at the top level
      void _generation;
      return { schema: SEARCH_SCHEMA, generation, as_of: str(meta.as_of), count: out.name.length, statuses, tour_sets: tourSets, keys, meta: rest, ...out, derived: { from, built_at: builtAt } };
    },
  };
}

/* ------------------------------------------------------------------ reading R2 bodies */
export async function* textChunks(body: ReadableStream<Uint8Array>, gzip: boolean): AsyncGenerator<string> {
  const source: ReadableStream<Uint8Array> = gzip ? (body.pipeThrough(new DecompressionStream("gzip") as unknown as ReadableWritablePair<Uint8Array, Uint8Array>)) : body;
  const reader = source.getReader();
  const decoder = new TextDecoder("utf-8");
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      const text = decoder.decode(value, { stream: true });
      if (text) yield text;
    }
    const tail = decoder.decode();
    if (tail) yield tail;
  } finally {
    await reader.cancel().catch(() => undefined);
  }
}

type Bucket = Pick<R2Bucket, "get" | "put">;
type CatalogObject = { body: ReadableStream<Uint8Array>; httpMetadata?: { contentEncoding?: string } };
const isGzip = (o: CatalogObject): boolean => (o.httpMetadata?.contentEncoding ?? "").toLowerCase() === "gzip";

/** The generation named by index.json, found by scanning only until the `generation` key (it sorts before `players`). */
export async function readPointerGeneration(bucket: Bucket): Promise<string | null> {
  const object = (await bucket.get(POINTER_KEY)) as unknown as CatalogObject | null;
  if (!object?.body) return null;
  let found: string | null = null;
  await scanCatalog(textChunks(object.body, isGzip(object)), {
    onMeta: (key, value) => { if (key === "generation") { found = typeof value === "string" ? value : null; return true; } },
    onPlayer: () => true,
  });
  return found && GENERATION.test(found) ? found : null;
}

/** Scan one catalog object into an index. Throws on a malformed document or a generation mismatch. */
export async function buildFromObject(object: CatalogObject, generation: string, from: string, now: Date): Promise<SearchIndex> {
  const builder = createIndexBuilder();
  await scanCatalog(textChunks(object.body, isGzip(object)), { onMeta: (k, v) => { builder.meta(k, v); }, onPlayer: (p) => { builder.player(p); } });
  const actual = builder.metaValue("generation");
  if (typeof actual === "string" && actual !== generation) throw new NotPublished();   // the pointer moved on: that catalog is another generation
  const index = builder.build(generation, from, now.toISOString());
  const published = (index.meta as { schema_version?: unknown }).schema_version;
  if (published !== "player_profiles.v1" || index.count === 0) throw new Error("catalog is not a player_profiles.v1 document with players");
  return index;
}

export const derivedKey = (generation: string): string => `golfprice/player_profiles/derived/${generation}/search-index.json`;
const catalogKey = (generation: string): string => `golfprice/player_profiles/${generation}/catalog.json`;

class NotPublished extends Error {}

const inflight = new Map<string, Promise<{ body: string; stored: boolean }>>();

function failure(status: number, error: string): Response {
  return new Response(JSON.stringify({ ok: false, error }), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } });
}

/**
 * Serve (building and storing first when needed) the search index. With `gen` the answer is immutable; without it the generation
 * comes from the live pointer and the answer is revalidated like any other pointer.
 */
export async function handleSearchIndex(request: Request, bucket: Bucket, now: Date = new Date()): Promise<Response> {
  const requestedGen = new URL(request.url).searchParams.get("gen");
  if (requestedGen !== null && !GENERATION.test(requestedGen)) return failure(400, "Invalid generation");
  let generation = requestedGen;
  if (!generation) {
    generation = await readPointerGeneration(bucket);
    if (!generation) return failure(404, "not published yet");
  }
  const cacheControl = requestedGen ? CACHE_IMMUTABLE : CACHE_POINTER;
  const key = derivedKey(generation);
  const respond = (object: ServableObject) => respondWithObject(request, object, { contentType: "application/json; charset=utf-8", cacheControl, extra: { "x-search-generation": generation as string } });

  const existing = await bucket.get(key);
  if (existing) return respond(existing as unknown as ServableObject);

  let building = inflight.get(key);
  if (!building) {
    building = (async () => {
      // The generation folder first; the pointer object only when it is that same generation.
      let source = catalogKey(generation as string);
      let object = (await bucket.get(source)) as unknown as CatalogObject | null;
      if (!object) {
        source = POINTER_KEY;
        object = (await bucket.get(source)) as unknown as CatalogObject | null;
      }
      if (!object?.body) throw new NotPublished();
      const index = await buildFromObject(object, generation as string, source, now);
      const text = JSON.stringify(index);
      // Only the request that built it stores it; concurrent requests wait on this same promise.
      try {
        await bucket.put(key, text, { httpMetadata: { contentType: "application/json; charset=utf-8", cacheControl: CACHE_IMMUTABLE } });
        return { body: text, stored: true };
      } catch (error) {
        console.error(`search-index store failed for ${generation}: ${error instanceof Error ? error.message : String(error)}`);
        return { body: text, stored: false };   // serve it anyway; the next request builds again
      }
    })();
    inflight.set(key, building);
    void building.then(() => undefined, () => undefined).finally(() => { if (inflight.get(key) === building) inflight.delete(key); });
  }
  let built: { body: string; stored: boolean };
  try {
    built = await building;
  } catch (error) {
    if (error instanceof NotPublished) return failure(404, "not published yet");
    console.error(`search-index build failed for ${generation}: ${error instanceof Error ? error.message : String(error)}`);
    return failure(502, "search index could not be built");
  }
  const unstored = () => new Response(request.method === "HEAD" ? null : built.body, { headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } });
  if (!built.stored) return unstored();
  const stored = await bucket.get(key);   // serve the stored object so every response carries the same ETag
  return stored ? respond(stored as unknown as ServableObject) : unstored();
}


/* ------------------------------------------------------------------ benchmarks for a field */
export type BenchmarkSlice = {
  schema: typeof BENCHMARKS_SCHEMA;
  generation: string;
  as_of: string | null;
  pga_benchmark_reference: unknown;
  players: Array<{ dg_id: number; pga_benchmark: unknown }>;
};

/** Stream the catalog and keep only the requested golfers' pga_benchmark (the weekly table needs ~150 of 16,600). Stops early once all are found. */
export async function buildBenchmarkSlice(object: CatalogObject, generation: string, ids: number[]): Promise<BenchmarkSlice> {
  const wanted = new Set(ids);
  const found = new Map<number, unknown>();
  const meta: Record<string, unknown> = {};
  await scanCatalog(textChunks(object.body, isGzip(object)), {
    onMeta: (k, v) => { meta[k] = v; },
    onPlayer: (p) => {
      const id = p.dg_id;
      if (typeof id === "number" && wanted.has(id) && !found.has(id)) found.set(id, p.pga_benchmark ?? null);
      return found.size === wanted.size;
    },
  });
  if (typeof meta.generation === "string" && meta.generation !== generation) throw new NotPublished();
  // schema_version sorts after `players`, so an early stop never sees it: a real catalog is recognised by its reference block instead.
  if (found.size === 0 && !meta.pga_benchmark_reference) throw new Error("catalog has no pga_benchmark_reference and none of the requested golfers");
  return {
    schema: BENCHMARKS_SCHEMA, generation, as_of: typeof meta.as_of === "string" ? meta.as_of : null,
    pga_benchmark_reference: meta.pga_benchmark_reference ?? null,
    players: [...ids].sort((a, b) => a - b).filter((id) => found.has(id)).map((id) => ({ dg_id: id, pga_benchmark: found.get(id) })),
  };
}

/** Sorted unique positive integers from `ids=1,2,3`; null when malformed or over the limit. */
export function parseIdList(raw: string | null): number[] | null {
  if (!raw) return null;
  const parts = raw.split(",");
  if (parts.length > MAX_BENCHMARK_IDS * 2) return null;
  const out = new Set<number>();
  for (const part of parts) {
    if (!/^[1-9][0-9]{0,9}$/.test(part)) return null;
    out.add(Number(part));
  }
  return out.size > 0 && out.size <= MAX_BENCHMARK_IDS ? [...out].sort((a, b) => a - b) : null;
}

async function idsHash(ids: number[]): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-1", new TextEncoder().encode(ids.join(",")));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("").slice(0, 20);
}
export const benchmarksKey = (generation: string, hash: string): string => `golfprice/player_profiles/derived/${generation}/benchmarks-${hash}.json`;

const slicesInflight = new Map<string, Promise<string>>();

/** GET .../benchmarks.json?gen=<generation>&ids=<dg ids>: only those golfers' pga_benchmark, cached in the derived/ prefix by generation and id-list hash. */
export async function handleBenchmarks(request: Request, bucket: Bucket): Promise<Response> {
  const params = new URL(request.url).searchParams;
  const requestedGen = params.get("gen");
  if (requestedGen !== null && !GENERATION.test(requestedGen)) return failure(400, "Invalid generation");
  const ids = parseIdList(params.get("ids"));
  if (!ids) return failure(400, `ids must be 1 to ${MAX_BENCHMARK_IDS} comma-separated golfer ids`);
  const generation = requestedGen ?? (await readPointerGeneration(bucket));
  if (!generation) return failure(404, "not published yet");
  const key = benchmarksKey(generation, await idsHash(ids));
  const serve = (object: unknown) => respondWithObject(request, object as ServableObject, { contentType: "application/json; charset=utf-8", cacheControl: requestedGen ? CACHE_IMMUTABLE : CACHE_POINTER });
  const existing = await bucket.get(key);
  if (existing) return serve(existing);
  let building = slicesInflight.get(key);
  if (!building) {
    building = (async () => {
      let object = (await bucket.get(catalogKey(generation))) as unknown as CatalogObject | null;
      if (!object) object = (await bucket.get(POINTER_KEY)) as unknown as CatalogObject | null;
      if (!object?.body) throw new NotPublished();
      const text = JSON.stringify(await buildBenchmarkSlice(object, generation, ids));
      try {
        await bucket.put(key, text, { httpMetadata: { contentType: "application/json; charset=utf-8", cacheControl: CACHE_IMMUTABLE } });
      } catch (error) {
        console.error(`benchmarks store failed for ${generation}: ${error instanceof Error ? error.message : String(error)}`);
      }
      return text;
    })();
    slicesInflight.set(key, building);
    void building.then(() => undefined, () => undefined).finally(() => { if (slicesInflight.get(key) === building) slicesInflight.delete(key); });
  }
  let text: string;
  try {
    text = await building;
  } catch (error) {
    if (error instanceof NotPublished) return failure(404, "not published yet");
    console.error(`benchmarks build failed for ${generation}: ${error instanceof Error ? error.message : String(error)}`);
    return failure(502, "benchmarks could not be built");
  }
  const stored = await bucket.get(key);
  return stored ? serve(stored) : new Response(request.method === "HEAD" ? null : text, { headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } });
}
