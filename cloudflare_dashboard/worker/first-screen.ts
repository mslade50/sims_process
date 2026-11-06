/**
 * First-screen data inlined into the HTML (round 5, cold-load LCP). Additive and read-only: no R2 write, no new data endpoint.
 *
 * Before: the page asks `/api/golfprice/index.json`, waits for it (it shares a slow link with ~330 KB of script), reads the event it names, THEN asks for the
 * slim explain document. Two sequential round trips stand between the HTML and the first answer.
 * Now: for the pages whose first screen is known from the URL (This week, Why priced, One golfer) the Worker reads those same objects while it renders the
 * HTML, through the same handlers the browser would have called, and appends them to the end of the page as a JSON script. A tiny script at the very start of
 * <head> wraps `window.fetch` so that the page's own early requests for exactly those URLs (app/shell-rules.ts earlyFetchScript, app/data.ts) are answered from the
 * inlined text instead of the network. A URL that was not inlined, or any later request, goes to the network as before; if the Worker cannot read something it
 * simply inlines less, and the page behaves as it did before this change.
 */

type Json = Record<string, unknown>;
const isObj = (v: unknown): v is Json => !!v && typeof v === "object" && !Array.isArray(v);

export const INDEX_PATH = "/api/golfprice/index.json";
const PLAYER_ONE = "/api/golfprice/player_profiles/golfer/search-index.json";
const PLAYER_FIELDS = "/api/golfprice/player_profiles/fields.json";
const PLAYER_ID = /^(?:[1-9][0-9]*|historical-[a-z0-9]+)$/;

type IndexEvent = {
  event_uid?: string; date_start?: string; tour?: string; explain_key?: string; explain_run?: string;
  distribution_runs?: Array<{ kind?: string; key?: string; as_of?: string; validated?: boolean }>;
};

/** Which page this is, or null when nothing is inlined for it. `/` is This week. */
export function firstScreenPage(pathname: string): "this-week" | "why-priced" | "players" | "weather" | "scoring" | "distributions" | "inputs" | null {
  const key = pathname.replace(/^\/+|\/+$/g, "").toLowerCase();
  if (key === "" || key === "this-week") return "this-week";
  if (key === "why-priced") return "why-priced";
  if (key === "players") return "players";
  if (key === "weather-effects") return "weather";
  if (key === "round-scores") return "scoring";
  if (key === "distributions") return "distributions";
  if (key === "inputs") return "inputs";
  return null;
}

/** The event the page will open, as app/shell-rules.ts earlyFetchScript picks it on a visit with nothing remembered: `?e=` when known, else the newest week (first tour with an explanation). */
export function defaultEvent(index: unknown, query: string | null): IndexEvent | null {
  const events = isObj(index) && Array.isArray(index.events) ? (index.events as IndexEvent[]) : null;
  if (!events) return null;
  const known = events.filter((e) => e && e.event_uid);
  if (query && known.some((e) => e.event_uid === query)) return known.find((e) => e.event_uid === query) ?? null;
  const dated = known.filter((e) => e.date_start);
  if (!dated.length) return null;
  const newest = dated.map((e) => e.date_start as string).sort().pop();
  const rows = dated.filter((e) => e.date_start === newest);
  rows.sort((a, b) => String(a.tour ?? "").toLowerCase().localeCompare(String(b.tour ?? "").toLowerCase()));
  const at = rows.findIndex((e) => e.explain_key);
  if (at > 0) rows.unshift(rows.splice(at, 1)[0]);
  return rows[0] ?? null;
}

/** API paths (with query) the page requests first for this event, in the same words as the browser's own requests. */
export function explainPaths(page: "this-week" | "why-priced", e: IndexEvent | null, view: string | null): string[] {
  if (!e || !e.explain_key) return [];
  const runs = e.distribution_runs ?? [];
  const live = runs.some((r) => r && r.kind === "live");
  if (page === "why-priced") {
    const cur = String(e.explain_run ?? "").startsWith("live_");
    if (live || cur) {
      if (view && view !== "pre") return [];
      const week = runs.filter((r) => r && r.kind === "week" && typeof r.key === "string" && r.key && r.validated !== false).sort((a, b) => String(a.as_of ?? "").localeCompare(String(b.as_of ?? ""))).pop();
      return week?.key ? [`/api/${week.key}?slim=1`] : [];
    }
    return [`/api/${e.explain_key}?slim=1`];
  }
  if (live && !String(e.explain_run ?? "").startsWith("live_")) return [];
  return [`/api/${e.explain_key}?slim=1`];
}

/** Weather: the brief (first paint) and the weather document, under the same names the view uses (briefKeyFor, weatherKey in app/WeatherEffectsView.tsx). */
export function weatherPaths(e: IndexEvent | null): string[] {
  if (!e || !e.event_uid) return [];
  const w = e as IndexEvent & { weather_brief_key?: string; weather_key?: string };
  const doc = w.weather_key || `golfprice/weather/${e.event_uid.replace(/:/g, "_")}/latest.json`;
  return [w.weather_brief_key, doc].filter((k): k is string => typeof k === "string" && /^[A-Za-z0-9_./:=-]+$/.test(k) && !k.includes("..")).map((k) => `/api/${k}`);
}

/** Scoring: the newest checkpoint document (checkpoints() in app/distributions-rules.ts: validated, a week run or after round 1+, newest as_of, a live run after a week run at a tie) and the event's weather document. */
export function scoringPaths(e: IndexEvent | null): string[] {
  if (!e || !e.event_uid) return [];
  const runs = (e.distribution_runs ?? []).filter((r) => r && r.validated && typeof r.key === "string" && r.key && (r.kind === "week" || ((r as { after_round?: number }).after_round ?? 0) >= 1));
  runs.sort((a, b) => String(a.as_of ?? "").localeCompare(String(b.as_of ?? "")) || Number(a.kind === "live") - Number(b.kind === "live"));
  const run = runs.length ? runs[runs.length - 1].key : e.explain_key;
  const w = e as IndexEvent & { weather_key?: string };
  const doc = w.weather_key || `golfprice/weather/${e.event_uid.replace(/:/g, "_")}/latest.json`;
  return [run, doc].filter((k): k is string => typeof k === "string" && /^[A-Za-z0-9_./:=-]+$/.test(k) && !k.includes("..")).map((k) => `/api/${k}`);
}

/** Finish distributions: the newest checkpoint document only (the same pick as Scoring, without the weather). */
export function distributionPaths(e: IndexEvent | null): string[] { return scoringPaths(e).slice(0, 1); }

/** Model inputs: the run the page opens (InputsView: the event's latest run, else its first). */
export function inputsPaths(e: IndexEvent | null): string[] {
  const runs = ((e as (IndexEvent & { runs?: Array<{ key?: string }>; latest_key?: string }) | null)?.runs ?? []).filter((r) => r && typeof r.key === "string" && r.key);
  const latest = (e as { latest_key?: string } | null)?.latest_key;
  const key = (runs.find((r) => r.key === latest) ?? runs[0])?.key;
  return key && /^[A-Za-z0-9_./:=-]+$/.test(key) && !key.includes("..") ? [`/api/${key}`] : [];
}

/** The profile batch a one-golfer slice names (mirrors parseSearchIndex + profileKey in the app). */
export function profileBatchPath(slice: unknown): string | null {
  if (!isObj(slice) || !Array.isArray(slice.keys) || !Array.isArray(slice.key)) return null;
  const raw = slice.keys[Number(slice.key[0])];
  if (typeof raw !== "string" || !raw || raw.includes("..")) return null;
  if (raw.startsWith("golfprice/player_profiles/")) return `/api/${raw}`;
  if (/^(?:[a-zA-Z0-9_-]+\/)*[a-zA-Z0-9_-]+\.json$/.test(raw)) return `/api/golfprice/player_profiles/${raw}`;
  return null;
}

export type FetchApi = (path: string) => Promise<Response | null>;
export type Inlined = Record<string, string>;

async function text(fetchApi: FetchApi, path: string): Promise<string | null> {
  // Only public golfprice data may be inlined; never owner-gated APIs (e.g. /api/overrides), whatever an index lists.
  if (!path.startsWith("/api/golfprice/") || path.includes("..")) return null;
  try {
    const response = await fetchApi(path);
    if (!response || response.status !== 200) return null;
    const body = await response.text();
    return body.length > 0 && body.length < 900_000 ? body : null;
  } catch { return null; }
}

/** The objects to inline for this page request (an empty object when none apply). Never throws. */
export async function collectFirstScreen(url: URL, fetchApi: FetchApi): Promise<Inlined> {
  const out: Inlined = {};
  try {
    const page = firstScreenPage(url.pathname);
    if (!page) return out;
    if (page === "players") {
      const pid = url.searchParams.get("player");
      if (!pid || !PLAYER_ID.test(pid) || url.searchParams.get("generation")) return out;
      const slicePath = `${PLAYER_ONE}?id=${pid}`;
      // fields.json (which golfers play which event, ~1 KB) is the third request the golfer page makes before it can settle its "this week" card.
      const [indexText, sliceText, fieldsText] = await Promise.all([text(fetchApi, INDEX_PATH), text(fetchApi, slicePath), text(fetchApi, PLAYER_FIELDS)]);
      if (indexText) out[INDEX_PATH] = indexText;
      if (fieldsText) out[PLAYER_FIELDS] = fieldsText;
      if (!sliceText) return out;
      out[slicePath] = sliceText;
      const batch = profileBatchPath(JSON.parse(sliceText));
      const batchText = batch ? await text(fetchApi, batch) : null;
      if (batch && batchText) out[batch] = batchText;
      return out;
    }
    const indexText = await text(fetchApi, INDEX_PATH);
    if (!indexText) return out;
    out[INDEX_PATH] = indexText;
    const picked = defaultEvent(JSON.parse(indexText), url.searchParams.get("e"));
    const paths = page === "weather" ? weatherPaths(picked) : page === "scoring" ? scoringPaths(picked) : page === "distributions" ? distributionPaths(picked) : page === "inputs" ? inputsPaths(picked) : explainPaths(page, picked, url.searchParams.get("v"));
    const bodies = await Promise.all(paths.map((p) => text(fetchApi, p)));
    paths.forEach((p, i) => { const b = bodies[i]; if (b) out[p] = b; });
  } catch { /* inline less; the page fetches the rest itself */ }
  return out;
}

/** Runs first in <head>: the page's fetches for inlined URLs wait for the data script (or 2.5 s), take the text once, and everything else is untouched. */
export const WRAPPER_SCRIPT = `<script>(function(){try{var f=window.fetch,d=null,ok=false,go,ready=new Promise(function(r){go=r});window.__inlineFirst=function(x){d=x;ok=true;go()};setTimeout(go,2500);`
  + `window.fetch=function(u,o){if(typeof u!=="string"||u.indexOf("/api/golfprice/")!==0||ok&&!d[u])return f.apply(this,arguments);var a=arguments,t=this;`
  + `return ready.then(function(){var s=d&&d[u];if(typeof s==="string"){delete d[u];return new Response(s,{status:200,headers:{"content-type":"application/json; charset=utf-8","x-inline-first-screen":"1"}})}return f.apply(t,a)})}}catch(x){}})()</script>`;

export function dataScript(inlined: Inlined): string {
  const json = JSON.stringify(inlined).replace(/</g, "\\u003c").replace(/\u2028/g, "\\u2028").replace(/\u2029/g, "\\u2029");
  return `<script>window.__inlineFirst&&window.__inlineFirst(${json})</script>`;
}

/**
 * Streams the HTML through, adding the wrapper right after `<head>` and the data just before `</body>`. Without an `</body>` in the stream the data goes at the end.
 * `inlined` is awaited only when `</body>` arrives, so rendering the page and reading the objects overlap; a read that fails or takes over 1.5 s inlines nothing.
 */
export function withFirstScreen(body: ReadableStream<Uint8Array>, inlined: Promise<Inlined>): ReadableStream<Uint8Array> {
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let pending = "";
  let head = false;
  let tail = false;
  const settled = (): Promise<Inlined> => Promise.race([inlined.catch(() => ({} as Inlined)), new Promise<Inlined>((r) => setTimeout(() => r({}), 1500))]);
  return body.pipeThrough(new TransformStream<Uint8Array, Uint8Array>({
    async transform(chunk, controller) {
      pending += decoder.decode(chunk, { stream: true });
      if (!head) {
        const at = pending.indexOf("<head>");
        if (at >= 0) { pending = pending.slice(0, at + 6) + WRAPPER_SCRIPT + pending.slice(at + 6); head = true; }
      }
      if (head && !tail) {
        const at = pending.lastIndexOf("</body>");
        if (at >= 0) { pending = pending.slice(0, at) + dataScript(await settled()) + pending.slice(at); tail = true; }
      }
      // Keep a short tail so a marker split across two chunks is still found.
      const keep = tail ? 0 : 8;
      if (pending.length > keep) { controller.enqueue(encoder.encode(pending.slice(0, pending.length - keep))); pending = pending.slice(pending.length - keep); }
    },
    async flush(controller) {
      if (head && !tail) pending += dataScript(await settled());
      if (pending) controller.enqueue(encoder.encode(pending));
    },
  }));
}
