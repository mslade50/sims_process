/**
 * Pure data-loading rules (site reliability pass, October 2026): timeouts, retries with jittered backoff, request classification,
 * last-good fallback and the "outage versus nothing published" wording. No React and no DOM: app/data.ts wires these to fetch,
 * Cache Storage and the hook, and tests/data-resilience.test.mjs drives them with fakes.
 */
import { decideFallback, isWorkerError } from "./shell-rules.ts";

/* ------------------------------------------------------------------ policy */
export const DEFAULT_TIMEOUT_MS = 12_000;
export const LARGE_TIMEOUT_MS = 45_000;
export const MAX_RETRIES = 2;
const BACKOFF_BASE_MS = 400;
/** A cached payload older than this is shown at once and revalidated in the background (mutable keys only). */
export const REVALIDATE_AFTER_MS = 2 * 60_000;

/** Known large objects: the full player catalog, the deep reference rounds and the frozen dossier review. */
const LARGE_KEY = /(^|\/)(index|dossier-review|catalog|rounds)\.json$|deep-reference|player_profiles\/(?:[a-f0-9]{20}\/)?catalog\.json$/;
export function timeoutFor(name: string): number {
  return name.startsWith("golfprice/player_profiles/") && LARGE_KEY.test(name) ? LARGE_TIMEOUT_MS : DEFAULT_TIMEOUT_MS;
}

/** A path with a 20-hex content/generation segment (`44ff.../batch_0015.json`, `deep-2d19....json`) never changes once published. */
export function isImmutableKey(name: string): boolean {
  return /(^|\/)(?:[a-z]+-)*[0-9a-f]{20}(\/|\.json|$)/.test(name.split("?")[0]) || /[?&]gen=[0-9a-f]{20}(&|$)/.test(name);
}

const POINTER = /(^|\/)(manifest|index|latest|brief_latest|dossier-review|search-index|benchmarks)\.json(\?|$)/;
/** Pointers are always revalidated (a 304 costs no body); everything else follows the response's own cache headers. */
export function fetchCacheMode(name: string): RequestCache {
  return POINTER.test(name) && !isImmutableKey(name) ? "no-cache" : "default";
}

/** Delay before retry number `attempt` (0 = first retry): exponential from 400 ms, jittered between 50% and 100%. */
export function backoffMs(attempt: number, rand: () => number = Math.random): number {
  const ceiling = BACKOFF_BASE_MS * 2 ** attempt;
  return Math.round(ceiling * (0.5 + 0.5 * rand()));
}

/** Network failures, timeouts and 5xx are retried; 4xx (including 404 "not published yet") never are. */
export function isRetryable(outcome: { kind: "network" } | { kind: "response"; status: number }): boolean {
  return outcome.kind === "network" || outcome.status >= 500;
}

/* ------------------------------------------------------------------ wording */
export type DataErrorKind = "missing" | "outage" | "invalid";

/** What a failed request means for the person looking at it. `outage` must never be worded as "nothing published". */
export function classifyFailure(status: number | null, workerError?: string): { kind: DataErrorKind; message: string } {
  if (status === null) return { kind: "outage", message: "The dashboard data service did not respond. Check the connection and try again." };
  if (status === 404) return { kind: "missing", message: `This data has not been published yet (code 404)${workerError ? `: ${workerError}` : ""}` };
  if (status >= 500) return { kind: "outage", message: `The dashboard data service is unavailable right now (code ${status}${workerError ? `: ${workerError}` : ""}). This is an outage, not missing data.` };
  return { kind: "invalid", message: `This data could not be loaded (code ${status})${workerError ? `: ${workerError}` : ""}` };
}

export function formatAge(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 90) return "moments ago";
  const m = Math.round(s / 60);
  if (m < 90) return `${m} min ago`;
  const h = Math.round(m / 60);
  if (h < 48) return `${h} h ago`;
  return `${Math.round(h / 24)} days ago`;
}

/* ------------------------------------------------------------------ fetch with timeout, retries, backoff */
export type Outcome =
  | { kind: "network"; attempts: number; reason: string }
  | { kind: "response"; attempts: number; ok: boolean; status: number; json: unknown; text: string; etag: string | null };

export type FetchLike = (url: string, init: { headers: Record<string, string>; cache: RequestCache; signal: AbortSignal }) => Promise<Pick<Response, "ok" | "status" | "text" | "headers">>;

export type FetchPlan = {
  fetchImpl: FetchLike;
  timeoutMs?: number;
  retries?: number;
  cache?: RequestCache;
  sleep?: (ms: number) => Promise<void>;
  rand?: () => number;
};

const realSleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

async function attemptOnce(url: string, plan: FetchPlan, attempts: number): Promise<Outcome> {
  const timeoutMs = plan.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
  try {
    const response = await plan.fetchImpl(url, { headers: { accept: "application/json" }, cache: plan.cache ?? "default", signal: controller.signal });
    // The body counts towards the timeout: a stalled download is aborted too.
    const text = await response.text();
    let json: unknown;
    try { json = JSON.parse(text); } catch { json = undefined; }
    return { kind: "response", attempts, ok: response.ok, status: response.status, json, text, etag: response.headers?.get?.("etag") ?? null };
  } catch (error) {
    return { kind: "network", attempts, reason: timedOut ? `timed out after ${Math.round(timeoutMs / 1000)} s` : error instanceof Error ? error.message : "network error" };
  } finally {
    clearTimeout(timer);
  }
}

/** One GET with a timeout; up to `retries` more tries (default 2) after jittered backoff, on network failure, timeout or 5xx only. */
export async function fetchWithRetry(url: string, plan: FetchPlan): Promise<Outcome> {
  const retries = plan.retries ?? MAX_RETRIES;
  const sleep = plan.sleep ?? realSleep;
  let outcome = await attemptOnce(url, plan, 1);
  for (let retry = 0; retry < retries && isRetryable(outcome); retry++) {
    await sleep(backoffMs(retry, plan.rand));
    outcome = await attemptOnce(url, plan, retry + 2);
  }
  return outcome;
}

/* ------------------------------------------------------------------ last-good store */
export type StoredCopy = { text: string; savedAt: number; etag: string | null };
export interface LastGoodStore {
  get(url: string): Promise<StoredCopy | null>;
  put(url: string, copy: StoredCopy): Promise<void>;
}

/** Persistence is best effort: every call is wrapped so a blocked, full or absent store never breaks a load. */
async function safeGet(store: LastGoodStore | undefined, url: string): Promise<StoredCopy | null> {
  try { return store ? await store.get(url) : null; } catch { return null; }
}
async function safePut(store: LastGoodStore | undefined, url: string, copy: StoredCopy): Promise<void> {
  try { if (store) await store.put(url, copy); } catch { /* ignore */ }
}

/* ------------------------------------------------------------------ the whole load */
export type LoadResult =
  | { status: "ok"; json: unknown; stale: false; savedAt: null; source: "api" | "packaged" }
  | { status: "stale"; json: unknown; stale: true; savedAt: number; failure: { kind: DataErrorKind; message: string } }
  | { status: "error"; kind: DataErrorKind; message: string; httpStatus: number | null };

export type LoadPlan = FetchPlan & {
  /** The Worker URL, then the packaged static copy (local no-Worker preview only). */
  url: string;
  /** Omit for objects that must never be served from the bundled snapshot (the legacy /api/data/* files). */
  packagedUrl?: string;
  store?: LastGoodStore;
  now?: () => number;
};

async function staleOrError(plan: LoadPlan, failure: { kind: DataErrorKind; message: string }, httpStatus: number | null): Promise<LoadResult> {
  const copy = await safeGet(plan.store, plan.url);
  if (copy) {
    try { return { status: "stale", json: JSON.parse(copy.text), stale: true, savedAt: copy.savedAt, failure }; } catch { /* corrupt copy: fall through */ }
  }
  return { status: "error", kind: failure.kind, message: failure.message, httpStatus };
}

/**
 * Load one data object. Order: the Worker (with timeout and retries) -> the packaged copy when there is no Worker at all
 * (network failure or a non-JSON body) -> the last good copy on an outage -> an error whose wording says whether this is
 * an outage ("unavailable") or genuinely nothing published ("not published yet").
 */
export async function loadResource(plan: LoadPlan): Promise<LoadResult> {
  const now = plan.now ?? Date.now;
  const api = await fetchWithRetry(plan.url, plan);
  const decision = decideFallback(api.kind === "network" ? { kind: "network" } : { kind: "response", ok: api.ok, status: api.status, json: api.json });
  if (api.kind === "response" && decision.action === "use") {
    await safePut(plan.store, plan.url, { text: api.text, savedAt: now(), etag: api.etag });
    return { status: "ok", json: api.json, stale: false, savedAt: null, source: "api" };
  }
  if (decision.action === "surface" && api.kind === "response") {
    const failure = classifyFailure(api.status, isWorkerError(api.json) ? api.json.error : undefined);
    return failure.kind === "outage" ? staleOrError(plan, failure, api.status) : { status: "error", kind: failure.kind, message: failure.message, httpStatus: api.status };
  }
  // No Worker answered in a structured way: network failure, timeout, or an HTML/empty body.
  const packaged = plan.packagedUrl ? await fetchWithRetry(plan.packagedUrl, { ...plan, retries: 0, timeoutMs: Math.min(plan.timeoutMs ?? DEFAULT_TIMEOUT_MS, 6000) }) : null;
  if (packaged && packaged.kind === "response" && packaged.ok && packaged.json !== undefined && !isWorkerError(packaged.json)) {
    return { status: "ok", json: packaged.json, stale: false, savedAt: null, source: "packaged" };
  }
  const status = api.kind === "response" ? api.status : null;
  const failure = status !== null && status < 500 ? { kind: "missing" as const, message: `This data is not available (code ${status}).` } : classifyFailure(status);
  return failure.kind === "outage" ? staleOrError(plan, failure, status) : { status: "error", kind: failure.kind, message: failure.message, httpStatus: status };
}

/* ------------------------------------------------------------------ in-flight dedupe */
/** Concurrent callers for the same key share one promise; the entry is cleared when it settles. */
export function createDeduper<T>() {
  const inflight = new Map<string, Promise<T>>();
  return {
    run(key: string, start: () => Promise<T>): Promise<T> {
      const existing = inflight.get(key);
      if (existing) return existing;
      const promise = start().finally(() => { if (inflight.get(key) === promise) inflight.delete(key); });
      inflight.set(key, promise);
      return promise;
    },
    size: () => inflight.size,
  };
}

/**
 * True when a cached entry should be shown at once but refreshed in the background. `checkedAt` is when the network last answered
 * (or failed for it). A stale entry is retried at most every STALE_RETRY_MS so a dead network is not hammered by every mount.
 */
export const STALE_RETRY_MS = 15_000;
export function needsRevalidation(name: string, checkedAt: number, stale: boolean, now: number): boolean {
  if (stale) return now - checkedAt > STALE_RETRY_MS;
  return !isImmutableKey(name) && now - checkedAt > REVALIDATE_AFTER_MS;
}
