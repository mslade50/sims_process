"use client";

import { useEffect, useState } from "react";
import { useDashboardData } from "./data";
import { EVENT_STORAGE_KEY, resolveEventUid } from "./shell-rules";
import type { IndexEventLite } from "./ui-rules";

/** The last event chosen (localStorage, wrapped: it can throw or be empty). */
export function readRememberedEvent(): string | null {
  try { return window.localStorage.getItem(EVENT_STORAGE_KEY); } catch { return null; }
}
export function rememberEvent(uid: string): void {
  try { if (uid) window.localStorage.setItem(EVENT_STORAGE_KEY, uid); } catch { /* storage unavailable: the URL still carries the event */ }
}

/**
 * The event every event-scoped page shows, and the one the shell bar shows: resolveEventUid (shell-rules.ts) over the published index,
 * with ?e= first, then the last chosen event, then the default. Read after mount so the server render matches; empty until the index is loaded.
 */
export function useEventParam(): string {
  const { data } = useDashboardData<{ events?: IndexEventLite[] }>("golfprice/index.json");
  const [url, setUrl] = useState<{ param: string | null; remembered: string | null } | null>(null);
  useEffect(() => {
    const read = () => setUrl({ param: new URLSearchParams(window.location.search).get("e"), remembered: readRememberedEvent() });
    queueMicrotask(read);
    window.addEventListener("golf-event-changed", read);
    return () => window.removeEventListener("golf-event-changed", read);
  }, []);
  if (!url || !data) return "";
  return resolveEventUid(data, url.param, url.remembered, 0);
}

/** A page's own event picker changed: remember it and put it in the URL (no reload) so the shell bar, the other pages and a shared link all name the same event. */
export function switchBarEvent(uid: string): void {
  if (!uid) return;
  rememberEvent(uid);
  const q = new URLSearchParams(window.location.search);
  q.set("e", uid);
  window.history.replaceState(null, "", `${window.location.pathname}?${q.toString()}${window.location.hash}`);
  window.dispatchEvent(new Event("golf-event-changed"));
}
