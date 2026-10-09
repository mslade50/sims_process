"use client";

import { useEffect, useState } from "react";
import { decideFallback, type FallbackDecision } from "./shell-rules";

type LoadState<T> = { data: T | null; loading: boolean; error: string | null };

const cache = new Map<string, unknown>();

/** `name` null = idle (no request); used by components that load lazily, such as the header golfer search. */
export function useDashboardData<T>(name: string | null): LoadState<T> {
  const [state, setState] = useState<LoadState<T>>(() => ({
    data: name ? ((cache.get(name) as T | undefined) ?? null) : null,
    loading: name !== null && !cache.has(name),
    error: null,
  }));

  useEffect(() => {
    let active = true;
    if (name === null) {
      queueMicrotask(() => {
        if (active) setState({ data: null, loading: false, error: null });
      });
      return () => {
        active = false;
      };
    }
    if (cache.has(name)) {
      queueMicrotask(() => {
        if (active) setState({ data: cache.get(name) as T, loading: false, error: null });
      });
      return () => {
        active = false;
      };
    }

    queueMicrotask(() => {
      if (active) setState({ data: null, loading: true, error: null });
    });
    // The packaged static copy (public/golfprice, local preview only) is used ONLY when the API is absent: a network failure or a non-JSON
    // body. A structured Worker error (404 "not published yet", 503 bucket error) is shown as is (site audit D2; decision in shell-rules.ts).
    const readFrom = async (url: string): Promise<{ decision: FallbackDecision; json: unknown }> => {
      let response: Response;
      try {
        response = await fetch(url, { headers: { accept: "application/json" }, cache: name === "golfprice/player_profiles/dossier-review.json" ? "no-cache" : "default" });
      } catch {
        return { decision: decideFallback({ kind: "network" }), json: undefined };
      }
      let json: unknown;
      try {
        json = await response.json();
      } catch {
        json = undefined;
      }
      return { decision: decideFallback({ kind: "response", ok: response.ok, status: response.status, json }), json };
    };
    const readJson = async (): Promise<T> => {
      // golfprice objects (published by golfprice/publish_dashboard.py) are served from R2 at /api/golfprice/.
      const isGolfprice = name.startsWith("golfprice/");
      const api = await readFrom(isGolfprice ? `/api/${name}` : `/api/data/${name}`);
      if (api.decision.action === "use") return api.json as T;
      if (api.decision.action === "surface") throw new Error(api.decision.message);
      const packaged = await readFrom(isGolfprice ? `/${name}` : `/data/${name}`);
      if (packaged.decision.action === "use") return packaged.json as T;
      throw new Error(packaged.decision.action === "surface" ? packaged.decision.message : "Data request failed (no response)");
    };
    readJson()
      .then((payload) => {
        cache.set(name, payload);
        if (active) setState({ data: payload, loading: false, error: null });
      })
      .catch((error: unknown) => {
        if (active) {
          setState({
            data: null,
            loading: false,
            error: error instanceof Error ? error.message : "Unable to load dashboard data",
          });
        }
      });

    return () => {
      active = false;
    };
  }, [name]);

  return state;
}
