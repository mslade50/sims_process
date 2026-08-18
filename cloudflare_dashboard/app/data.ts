"use client";

import { useEffect, useState } from "react";

type LoadState<T> = { data: T | null; loading: boolean; error: string | null };

const cache = new Map<string, unknown>();

export function useDashboardData<T>(name: string): LoadState<T> {
  const [state, setState] = useState<LoadState<T>>(() => ({
    data: (cache.get(name) as T | undefined) ?? null,
    loading: !cache.has(name),
    error: null,
  }));

  useEffect(() => {
    let active = true;
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
    const readJson = async (url: string) => {
      const response = await fetch(url, { headers: { accept: "application/json" } });
      if (!response.ok) throw new Error(`Data request failed (${response.status})`);
      return (await response.json()) as T;
    };
    readJson(`/api/data/${name}`)
      .catch(() => readJson(`/data/${name}`))
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
