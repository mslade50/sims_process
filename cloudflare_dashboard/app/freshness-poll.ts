import { CHECKER_MAX_AGE_MS, parseFreshness, type FreshnessDoc } from "../worker/freshness-rules.ts";

/** The Run page panel re-reads owner/data_freshness.json this often (the check itself runs every 15 minutes). */
export const FRESHNESS_REFRESH_MS = 5 * 60_000;

export type Phase = "loading" | "ready" | "none" | "error";
export type PanelState = { phase: Phase; doc: FreshnessDoc | null; refreshFailed: boolean };
export const INITIAL_PANEL: PanelState = { phase: "loading", doc: null, refreshFailed: false };

type Fetcher = (url: string, init: { cache: "no-store"; headers: Record<string, string> }) => Promise<Pick<Response, "ok" | "status" | "json">>;
type Result = { phase: "ready"; doc: FreshnessDoc } | { phase: "none" | "error"; doc: null };

export async function readFreshness(url: string, fetcher: Fetcher): Promise<Result> {
  try {
    const response = await fetcher(url, { cache: "no-store", headers: { accept: "application/json" } });
    if (response.status === 404) return { phase: "none", doc: null };
    if (!response.ok) return { phase: "error", doc: null };
    const doc = parseFreshness(await response.json().catch(() => null));
    return doc ? { phase: "ready", doc } : { phase: "error", doc: null };
  } catch {
    return { phase: "error", doc: null };
  }
}

/** A failed refresh keeps the last document on screen (flagged) instead of blanking it; a first-load failure shows the error state. */
export function nextPanelState(previous: PanelState, result: Result): PanelState {
  if (result.phase === "ready") return { phase: "ready", doc: result.doc, refreshFailed: false };
  if (result.phase === "error" && previous.doc) return { ...previous, refreshFailed: true };
  return { phase: result.phase, doc: null, refreshFailed: false };
}

type Timers = { setInterval: (fn: () => void, ms: number) => unknown; clearInterval: (id: never) => void };

/** Read now, then every `intervalMs`. Returns the stop function; results arriving after it are dropped. */
export function startFreshnessPolling(url: string, fetcher: Fetcher, onState: (update: (previous: PanelState) => PanelState) => void, intervalMs = FRESHNESS_REFRESH_MS, timers: Timers = globalThis as unknown as Timers): () => void {
  let live = true;
  const run = () => { void readFreshness(url, fetcher).then((result) => { if (live) onState((previous) => nextPanelState(previous, result)); }); };
  run();
  const id = timers.setInterval(run, intervalMs);
  return () => { live = false; timers.clearInterval(id as never); };
}

/** The checker itself is late when its last run is unreadable or older than CHECKER_MAX_AGE_MS at `now` (a ticking clock, never a frozen one). */
export function checkerLate(doc: FreshnessDoc | null, now: number): boolean {
  if (!doc) return false;
  const checked = Date.parse(doc.checked_at);
  return !Number.isFinite(checked) || now - checked > CHECKER_MAX_AGE_MS;
}
