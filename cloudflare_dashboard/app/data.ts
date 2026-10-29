"use client";

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
// decideFallback (shell-rules.ts) still decides when the packaged static copy may be used; app/data-rules.ts applies it inside loadResource.
import { americanizeDeep } from "./spelling";
import { createDeduper, fetchCacheMode, loadResource, needsRevalidation, timeoutFor, type DataErrorKind, type FetchLike, type LastGoodStore, type StoredCopy } from "./data-rules";

export type LoadState<T> = {
  data: T | null;
  loading: boolean;
  error: string | null;
  /** "outage" = the data service failed (never worded as "nothing published"); "missing" = a 404; "invalid" = another 4xx. Null when there is no error. */
  errorKind: DataErrorKind | null;
  /** True when `data` is the last good copy because the network failed; `staleAgeMs` is how old that copy is. */
  stale: boolean;
  staleAgeMs: number | null;
  /** Ask for a fresh copy now (bypasses the HTTP cache). Safe to call while a request is running. */
  refetch: () => void;
};

type Entry = { payload: unknown; checkedAt: number; stale: boolean; savedAt: number | null };
const cache = new Map<string, Entry>();
const listeners = new Set<() => void>();
let savedSnapshot: number[] = [];
/** Every write to `cache` goes through here so the page-wide saved-copy notice follows the data actually in use. */
function remember(name: string, entry: Entry): void {
  cache.set(name, entry);
  const next = [...cache.values()].filter((e) => e.stale).map((e) => e.savedAt ?? e.checkedAt).sort((a, b) => a - b);
  if (next.length === savedSnapshot.length && next.every((v, i) => v === savedSnapshot[i])) return;
  savedSnapshot = next;
  listeners.forEach((l) => l());
}
/** Epoch ms of each saved copy (Cache Storage, after a data-service outage) currently held for display, oldest first. Stable between changes. */
export function useSavedCopies(): number[] {
  return useSyncExternalStore((cb) => { listeners.add(cb); return () => { listeners.delete(cb); }; }, () => savedSnapshot, () => savedSnapshot);
}
const loads = createDeduper<Entry>();

class DataError extends Error {
  constructor(message: string, readonly kind: DataErrorKind) { super(message); }
}

/* ------------------------------------------------------------------ last-good copies in the Cache Storage API */
const STORE_NAME = "golf-dashboard-last-good-v1";
const absolute = (url: string): string => new URL(url, typeof location === "undefined" ? "https://dashboard.invalid" : location.href).href;
const metaKey = (url: string): string => absolute(`/__last-good__?u=${encodeURIComponent(url)}`);

/** Body only when its etag changed; a tiny meta record is refreshed every time so "age" means "last confirmed good". Callers wrap every call in try/catch. */
function createLastGoodStore(): LastGoodStore | undefined {
  if (typeof caches === "undefined") return undefined;
  return {
    async get(url: string): Promise<StoredCopy | null> {
      const store = await caches.open(STORE_NAME);
      const body = await store.match(absolute(url));
      if (!body) return null;
      const meta = await store.match(metaKey(url));
      const info = meta ? ((await meta.json()) as { savedAt?: number; etag?: string | null }) : {};
      return { text: await body.text(), savedAt: info.savedAt ?? (Number(body.headers.get("x-saved-at")) || 0), etag: info.etag ?? null };
    },
    async put(url: string, copy: StoredCopy): Promise<void> {
      const store = await caches.open(STORE_NAME);
      const previous = await store.match(metaKey(url));
      const known = previous ? ((await previous.json()) as { etag?: string | null }).etag : null;
      if (!copy.etag || copy.etag !== known || !(await store.match(absolute(url)))) {
        await store.put(absolute(url), new Response(copy.text, { headers: { "content-type": "application/json", "x-saved-at": String(copy.savedAt) } }));
      }
      await store.put(metaKey(url), new Response(JSON.stringify({ savedAt: copy.savedAt, etag: copy.etag }), { headers: { "content-type": "application/json" } }));
    },
  };
}
let store: LastGoodStore | undefined | null = null;   // null = not created yet
const lastGood = (): LastGoodStore | undefined => (store === null ? (store = (() => { try { return createLastGoodStore(); } catch { return undefined; } })()) : store);

/* ------------------------------------------------------------------ loading */
// Identity data (golfer names), not prose: skip the spelling pass so a golfer called "Grey" stays "Grey" (and 16,600 names are not scanned).
const VERBATIM = /(^|\/)search-index\.json(\?|$)/;

const fetchImpl: FetchLike = (url, init) => fetch(url, init);

function loadEntry(name: string, force: boolean): Promise<Entry> {
  return loads.run(name, async () => {
    // golfprice objects (published by golfprice/publish_dashboard.py) are served from R2 at /api/golfprice/.
    const isGolfprice = name.startsWith("golfprice/");
    const result = await loadResource({
      url: isGolfprice ? `/api/${name}` : `/api/data/${name}`,
      packagedUrl: isGolfprice ? `/${name}` : undefined,   // /data/* is a frozen August snapshot: never presented as current
      fetchImpl,
      timeoutMs: timeoutFor(name),
      cache: force ? "no-cache" : fetchCacheMode(name),
      store: lastGood(),
    });
    if (result.status === "error") throw new DataError(result.message, result.kind);
    const payload = VERBATIM.test(name) ? result.json : americanizeDeep(result.json);
    const entry: Entry = result.status === "ok"
      ? { payload, checkedAt: Date.now(), stale: false, savedAt: null }
      : { payload, checkedAt: Date.now(), stale: true, savedAt: result.savedAt };
    remember(name, entry);
    return entry;
  });
}

const staleAge = (entry: Entry | undefined): number | null => (entry?.stale ? Math.max(0, Date.now() - (entry.savedAt ?? entry.checkedAt)) : null);
type Core = { data: unknown; loading: boolean; error: string | null; errorKind: DataErrorKind | null; stale: boolean; staleAgeMs: number | null };
const IDLE: Core = { data: null, loading: false, error: null, errorKind: null, stale: false, staleAgeMs: null };
const ready = (entry: Entry): Core => ({ data: entry.payload, loading: false, error: null, errorKind: null, stale: entry.stale, staleAgeMs: staleAge(entry) });

/** `name` null = idle (no request); used by components that load lazily, such as the header golfer search. */
export function useDashboardData<T>(name: string | null): LoadState<T> {
  const [state, setState] = useState<Core>(() => {
    const hit = name ? cache.get(name) : undefined;
    return hit ? ready(hit) : name !== null ? { ...IDLE, loading: true } : IDLE;
  });
  const [nonce, setNonce] = useState(0);
  const forceNext = useRef(false);
  const refetch = useCallback(() => { forceNext.current = true; setNonce((n) => n + 1); }, []);

  useEffect(() => {
    let active = true;
    const force = forceNext.current;
    forceNext.current = false;
    if (name === null) {
      queueMicrotask(() => { if (active) setState(IDLE); });
      return () => { active = false; };
    }
    const cached = cache.get(name);
    if (cached && !force && !needsRevalidation(name, cached.checkedAt, cached.stale, Date.now())) {
      queueMicrotask(() => { if (active) setState(ready(cached)); });
      return () => { active = false; };
    }
    // Show what we have at once (a revalidation or a retry after an outage), otherwise show the loading state.
    queueMicrotask(() => { if (active) setState(cached ? ready(cached) : { ...IDLE, loading: true }); });
    loadEntry(name, force)
      .then((entry) => { if (active) setState(ready(entry)); })
      .catch((error: unknown) => {
        if (!active) return;
        if (cached) {
          // Keep showing the last good data, flagged stale, rather than replacing it with an error.
          const kept: Entry = { ...cached, stale: true, savedAt: cached.savedAt ?? cached.checkedAt, checkedAt: Date.now() };
          remember(name, kept);
          setState(ready(kept));
          return;
        }
        setState({ ...IDLE, error: error instanceof Error ? error.message : "Unable to load dashboard data", errorKind: error instanceof DataError ? error.kind : "outage" });
      });
    return () => { active = false; };
  }, [name, nonce]);

  return { ...state, data: state.data as T | null, refetch };
}

