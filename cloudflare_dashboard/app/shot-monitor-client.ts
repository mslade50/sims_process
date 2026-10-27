import { decodeSnapshot, initialSources, SOURCES, type Source } from "./shot-monitor-rules.ts";

export type MonitorState = { sources: Source[]; refreshing: boolean };
export const initialMonitorState = (): MonitorState => ({ sources: initialSources(), refreshing: false });

/** A single batch at a time, with independently committed results. stop() aborts and prevents late updates. */
export function createShotMonitor(onState: (state: MonitorState) => void, options: { fetcher?: typeof fetch; now?: () => number; intervalMs?: number; timeoutMs?: number } = {}) {
  const fetcher = options.fetcher ?? fetch;
  const now = options.now ?? Date.now;
  const sources = initialSources();
  let stopped = false;
  let active: Promise<void> | undefined;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const controllers = new Set<AbortController>();
  const emit = (refreshing: boolean) => { if (!stopped) onState({ sources: [...sources], refreshing }); };
  async function load(index: number) {
    const controller = new AbortController();
    controllers.add(controller);
    let timeout: ReturnType<typeof setTimeout> | undefined;
    let denied = false;
    try {
      const request = async () => {
        const response = await fetcher(`/api/owner-data/shot-monitor/${SOURCES[index].file}`, { cache: "no-store", signal: controller.signal, headers: { accept: "application/json" } });
        denied = response.status === 403;
        if (!response.ok) throw new Error(response.status === 404 ? "Missing snapshot (HTTP 404); publisher has not supplied this file." : response.status === 403 ? "Owner access denied (HTTP 403)." : `Snapshot request failed (HTTP ${response.status}).`);
        return decodeSnapshot(await response.json(), SOURCES[index].file);
      };
      const bounded = new Promise<never>((_, reject) => {
        timeout = setTimeout(() => { reject(new Error("Snapshot request timed out; next poll will retry.")); controller.abort(); }, options.timeoutMs ?? 10_000);
        controller.signal.addEventListener("abort", () => reject(new Error("Snapshot request aborted.")), { once: true });
      });
      const data = await Promise.race([request(), bounded]);
      if (!stopped) sources[index] = { file: SOURCES[index].file, status: "ready", data, checkedAt: now() };
    } catch (error) {
      if (!stopped) {
        const message = error instanceof Error ? error.message : "Snapshot request failed.";
        sources[index] = { file: SOURCES[index].file, status: message.startsWith("Missing snapshot") ? "missing" : "error", data: denied ? undefined : sources[index].data, error: message, checkedAt: now() };
      }
    } finally {
      clearTimeout(timeout);
      controllers.delete(controller);
      emit(true);
    }
  }
  function refresh(): Promise<void> {
    if (stopped) return Promise.resolve();
    if (active) return active;
    clearTimeout(timer);
    // Defer execution until active is assigned, including synchronous onState subscribers.
    active = Promise.resolve().then(async () => {
      if (stopped) return;
      emit(true);
      await Promise.all(SOURCES.map((_, index) => load(index)));
    }).finally(() => {
      active = undefined;
      emit(false);
      if (!stopped) timer = setTimeout(() => { void refresh(); }, options.intervalMs ?? 30_000);
    });
    return active;
  }
  return { refresh, stop() { stopped = true; clearTimeout(timer); for (const controller of controllers) controller.abort(); } };
}
