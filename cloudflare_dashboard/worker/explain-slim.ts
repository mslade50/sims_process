/**
 * Slim explain document (R4b, phone first-screen bytes). Additive, read-only, computed in memory.
 *
 *   GET /api/golfprice/explain/<event>/<run>.json?slim=1
 *
 * The explain document is the floor of the first screen on This week and Why priced (about 100 KB compressed). Much of it is read by
 * other pages only: the round-score grid (`scoring`) and the head-to-head table (`head_to_heads`) by Round scores and Distributions (each
 * through its own request for the full document), and the per-player `scores`, `neighbors`, `form` and most of `diverge` by the player
 * drawer (the page asks for the full document when a drawer opens). This endpoint answers the SAME document without those members and
 * marks it `slim: true`; everything else (probabilities, components, shapes, finish chances, course card, field scores, legend ...) is
 * untouched, so the first screen renders identically. The full document is requested afterwards (idle prefetch) for the pages that need it.
 *
 * Nothing is written to R2 and there is no new write path. The slim body is cached in the Worker's Cache API per object etag
 * (the explain object is immutable per run, so a re-published run gets a new etag and a new cache entry). Any miss, parse problem
 * or runtime without the Cache API falls back to computing it again or to the normal full document (the caller sees the usual route).
 */
import { respondWithObject, cacheControlFor, type ServableObject } from "./data-http.ts";
import { textChunks } from "./player-search.ts";

/** Top-level members only the other pages read (each fetches the full document itself). */
export const SLIM_DROP = ["scoring", "head_to_heads"] as const;
export const EXPLAIN_PATH = /^explain\/(?!\.)[A-Za-z0-9_.:-]+\/(?!\.)[A-Za-z0-9_.:-]+\.json$/;

/** Per-player members only the player drawer reads (the page asks for the full document when a drawer opens). */
export const SLIM_PLAYER_DROP = ["scores", "neighbors", "form"] as const;

const isObj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);

/**
 * One player without the drawer-only members. `diverge` keeps `dg_check` (the Data Golf skill board reads it on This week) and an empty `edge_split`
 * when the original had one (its presence picks the wording of `why_edge`); peers, compare and sentences are drawer-only.
 */
export function slimPlayer(player: unknown): unknown {
  if (!isObj(player)) return player;
  const out: Record<string, unknown> = { ...player };
  for (const key of SLIM_PLAYER_DROP) delete out[key];
  if (isObj(player.diverge)) {
    const d: Record<string, unknown> = {};
    if (player.diverge.dg_check !== undefined) d.dg_check = player.diverge.dg_check;
    if (isObj(player.diverge.edge_split)) d.edge_split = {};
    out.diverge = d;
  }
  return out;
}

/** The document without the heavy members; a non-object is returned as is. Does not mutate the input. */
export function slimExplain(doc: unknown): unknown {
  if (!isObj(doc)) return doc;
  const out: Record<string, unknown> = { ...doc };
  for (const key of SLIM_DROP) delete out[key];
  if (Array.isArray(doc.players)) out.players = doc.players.map(slimPlayer);
  out.slim = true;
  return out;
}

type Bucket = Pick<R2Bucket, "get" | "head">;
type EdgeCache = { match(request: Request): Promise<Response | undefined>; put(request: Request, response: Response): Promise<void> };
type Ctx = { waitUntil(promise: Promise<unknown>): void };

const edgeCache = (): EdgeCache | null => {
  try {
    const c = (globalThis as unknown as { caches?: { default?: EdgeCache } }).caches?.default;
    return c ?? null;
  } catch { return null; }
};

/** The slim variant when `request` asks for one; null for every other request (the normal handlers then run). */
export async function handleExplainSlim(request: Request, bucket: Bucket | undefined, ctx?: Ctx): Promise<Response | null> {
  if (request.method !== "GET" && request.method !== "HEAD") return null;
  const url = new URL(request.url);
  if (!url.pathname.startsWith("/api/golfprice/") || url.searchParams.get("slim") !== "1" || !bucket) return null;
  let requested: string;
  try { requested = decodeURIComponent(url.pathname.slice("/api/golfprice/".length)); } catch { return null; }
  if (!EXPLAIN_PATH.test(requested)) return null;
  try {
    const head = await bucket.head(`golfprice/${requested}`);
    if (!head) return null;
    const etag = `"${head.httpEtag.replace(/^W\//, "").replace(/"/g, "")}-slim"`;
    const cacheKey = new Request(`https://slim.explain.invalid/${requested}?etag=${encodeURIComponent(etag)}`);
    const cache = edgeCache();
    let text: string | null = null;
    if (cache) {
      const hit = await cache.match(cacheKey).catch(() => undefined);
      if (hit) text = await hit.text();
    }
    if (text === null) {
      const object = await bucket.get(`golfprice/${requested}`);
      if (!object?.body) return null;
      const gzip = (object.httpMetadata?.contentEncoding ?? "").toLowerCase() === "gzip";
      let raw = "";
      for await (const chunk of textChunks(object.body as ReadableStream<Uint8Array>, gzip)) raw += chunk;
      text = JSON.stringify(slimExplain(JSON.parse(raw)));
      if (cache) {
        const put = cache.put(cacheKey, new Response(text, { headers: { "content-type": "application/json; charset=utf-8", "cache-control": "public, max-age=86400" } })).catch(() => undefined);
        if (ctx) ctx.waitUntil(put);
      }
    }
    const object: ServableObject = { httpEtag: etag, body: new Response(text).body, writeHttpMetadata: () => undefined };
    return respondWithObject(request, object, { contentType: "application/json; charset=utf-8", cacheControl: cacheControlFor(requested), extra: { "x-explain-variant": "slim" } });
  } catch (error) {
    // Anything unexpected: let the normal route answer with the full document.
    console.error(`slim explain failed for ${requested}: ${error instanceof Error ? error.message : String(error)}`);
    return null;
  }
}
