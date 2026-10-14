/**
 * HTTP caching rules shared by the two read-only data endpoints (/api/data/* and /api/golfprice/*): cache-control per key class and
 * conditional requests (If-None-Match -> 304). Pure helpers, no bindings, so tests can import them directly.
 */

/** Keys with a 20-hex generation/content segment never change once published (`44ff.../batch_0015.json`, `deep-<hash>/...`, `derived/<gen>/...`). */
const IMMUTABLE = /(^|\/)(?:[a-z]+-)*[0-9a-f]{20}(\/|\.json$|$)/;
/** Mutable pointers: a short browser cache plus background revalidation, never the 5-minute default. */
const POINTER = /(^|\/)(manifest|index|latest|brief_latest|dossier-review)\.json$/;
const DERIVED_POINTER = /(^|\/)(search-index|benchmarks)\.json$/;

export const CACHE_IMMUTABLE = "public, max-age=31536000, immutable";
/** Publish pointers (manifest, index, latest, brief_latest, dossier-review, weather latest) always revalidate: never a stale forecast. A 304 costs no body. */
export const CACHE_NO_CACHE = "no-cache";
/** Worker-derived pointer-like objects (search index / benchmarks without `gen`): short cache, background refresh. */
export const CACHE_POINTER = "public, max-age=15, stale-while-revalidate=120";
export const CACHE_DEFAULT = "public, max-age=300, stale-while-revalidate=3600";

export function cacheControlFor(requested: string): string {
  if (IMMUTABLE.test(requested)) return CACHE_IMMUTABLE;
  if (POINTER.test(requested)) return CACHE_NO_CACHE;
  if (DERIVED_POINTER.test(requested)) return CACHE_POINTER;
  return CACHE_DEFAULT;
}

/** The entity tags in an If-None-Match header, weak markers and quotes removed. `*` is kept as is. */
export function parseIfNoneMatch(header: string | null): string[] {
  if (!header) return [];
  return header.split(",").map((part) => part.trim().replace(/^W\//, "").replace(/^"|"$/g, "")).filter(Boolean);
}

export function etagMatches(header: string | null, etag: string | undefined | null): boolean {
  if (!etag) return false;
  const wanted = parseIfNoneMatch(header);
  if (wanted.length === 0) return false;
  const bare = etag.replace(/^W\//, "").replace(/^"|"$/g, "");
  return wanted.includes("*") || wanted.includes(bare);
}

/** 304 with the validators and cache policy of the 200 it stands in for. */
export function notModified(headers: Headers): Response {
  const kept = new Headers();
  for (const name of ["etag", "cache-control", "last-modified", "vary"]) {
    const value = headers.get(name);
    if (value) kept.set(name, value);
  }
  return new Response(null, { status: 304, headers: kept });
}

export type ServableObject = {
  httpEtag: string;
  body?: ReadableStream | null;
  writeHttpMetadata(headers: Headers): void;
};

/** Build the 200 (or 304) for an R2 object. `contentType` and `cacheControl` are decided by the caller. */
export function respondWithObject(request: Request, object: ServableObject, opts: { contentType: string; cacheControl: string; extra?: Record<string, string> }): Response {
  const headers = new Headers();
  object.writeHttpMetadata(headers);
  headers.set("content-type", opts.contentType);
  headers.set("cache-control", opts.cacheControl);
  headers.set("etag", object.httpEtag);
  for (const [key, value] of Object.entries(opts.extra ?? {})) headers.set(key, value);
  if (etagMatches(request.headers.get("if-none-match"), object.httpEtag)) {
    void object.body?.cancel?.().catch?.(() => undefined);
    return notModified(headers);
  }
  // R2 dossiers are already gzip encoded. Prevent Workers from applying the same content encoding a second time to their stored bytes.
  const init: ResponseInit & { encodeBody?: "manual" } = { headers };
  if (headers.has("content-encoding")) init.encodeBody = "manual";
  return new Response(request.method === "HEAD" ? null : (object.body ?? null), init);
}
