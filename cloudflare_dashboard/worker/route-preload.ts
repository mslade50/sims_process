/**
 * Early hints for the page's view code. `scripts/route-chunks.mjs` (run after the build) writes /route-chunks.json = route key -> chunk URLs.
 * For an HTML page request this adds `Link: <chunk>; rel=modulepreload` headers, so the browser fetches the view's code while the shell loads
 * instead of only after hydration lazily imports it. Additive: the body, status and every other header are untouched, and any miss
 * (no manifest, unknown route, API path) returns the response as it was.
 */
const MAX_LINKS = 12;
let routes: Record<string, string[]> | null | undefined;   // undefined = not loaded; null = unavailable (retry later)
let triedAt = 0;

export function routeKeyFor(pathname: string): string | null {
  const key = pathname.replace(/^\/+|\/+$/g, "").toLowerCase();
  if (key.includes("/") || key.includes(".")) return null;
  return key === "" ? "this-week" : key;
}

export function linkHeaderFor(files: readonly string[] | undefined): string | null {
  const list = files ?? [];
  const js = list.filter((f) => /^\/_next\/static\/chunks\/[A-Za-z0-9._-]+\.js$/.test(f)).slice(0, MAX_LINKS).map((f) => `<${f}>; rel=modulepreload; crossorigin`);
  return js.length ? js.join(", ") : null;
}

/** The route's own stylesheets as <link> tags for the end of <head>. Without them the lazy view's CSS is injected only after its code runs, so data that is already in
 *  (preloaded code, early fetches) can draw one unstyled frame and shift. The same URLs are linked again by the framework later; the browser reuses the loaded sheet. */
export function stylesheetTagsFor(files: readonly string[] | undefined): string {
  return (files ?? []).filter((f) => /^\/_next\/static\/css\/[A-Za-z0-9._-]+\.css$/.test(f)).slice(0, 4).map((f) => `<link rel="stylesheet" href="${f}">`).join("");
}

/** Streams the HTML through, inserting `tags` before the first </head>. Falls back to the untouched stream when none is seen in the first 64 KB. */
export function injectBeforeHeadEnd(body: ReadableStream<Uint8Array>, tags: string): ReadableStream<Uint8Array> {
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let pending = "";
  let done = false;
  return body.pipeThrough(new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      if (done) { controller.enqueue(chunk); return; }
      pending += decoder.decode(chunk, { stream: true });
      const at = pending.indexOf("</head>");
      if (at >= 0) { controller.enqueue(encoder.encode(pending.slice(0, at) + tags + pending.slice(at))); pending = ""; done = true; }
      else if (pending.length > 65536) { controller.enqueue(encoder.encode(pending)); pending = ""; done = true; }
    },
    flush(controller) { if (pending) controller.enqueue(encoder.encode(pending)); },
  }));
}

async function loadRoutes(assets: Fetcher, origin: string): Promise<Record<string, string[]> | null> {
  if (routes) return routes;
  if (routes === null && Date.now() - triedAt < 60_000) return null;
  triedAt = Date.now();
  try {
    const response = await assets.fetch(new Request(new URL("/route-chunks.json", origin)));
    const contentType = response.headers.get("content-type") ?? "";
    if (!response.ok || !/json/i.test(contentType)) { routes = null; return null; }
    routes = (await response.json()) as Record<string, string[]>;
    return routes;
  } catch {
    routes = null;
    return null;
  }
}

export async function withViewPreload(request: Request, response: Response, assets: Fetcher): Promise<Response> {
  try {
    if (request.method !== "GET" || response.status !== 200 || !(response.headers.get("content-type") ?? "").includes("text/html")) return response;
    const url = new URL(request.url);
    const key = routeKeyFor(url.pathname);
    if (!key) return response;
    const table = await loadRoutes(assets, url.origin);
    const link = linkHeaderFor(table?.[key]);
    const tags = stylesheetTagsFor(table?.[key]);
    if ((!link && !tags) || !response.body) return response;
    const headers = new Headers(response.headers);
    if (tags) headers.delete("content-length");
    if (link) { const existing = headers.get("link"); headers.set("link", existing ? `${existing}, ${link}` : link); }
    return new Response(tags ? injectBeforeHeadEnd(response.body, tags) : response.body, { status: response.status, statusText: response.statusText, headers });
  } catch {
    return response;
  }
}
