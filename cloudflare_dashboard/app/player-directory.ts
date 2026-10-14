"use client";

import { useMemo } from "react";
import { useDashboardData } from "./data";
import { parseBenchmarks, parseCatalog, parseSearchIndex, type ProfileCatalog } from "./player-profile-rules";
import { PLAYER_BENCHMARKS_KEY, PLAYER_CATALOG_KEY, PLAYER_SEARCH_KEY } from "./shell-rules";

const parsedIndex = new WeakMap<object, ProfileCatalog | null>();
function parseSearchIndexOnce(raw: unknown): ProfileCatalog | null {
  if (!raw || typeof raw !== "object") return null;
  if (!parsedIndex.has(raw)) parsedIndex.set(raw, parseSearchIndex(raw));
  return parsedIndex.get(raw) ?? null;
}

export type PlayerDirectory = { catalog: ProfileCatalog | null; loading: boolean; error: string | null; stale: boolean; /** True when only the full 16 MB catalog could be used. */ fullCatalog: boolean };

/**
 * The golfer list for search, the picker and the Players page. Reads the Worker's derived search index (identity only, ~0.3 MB gzipped) and
 * downloads the full catalog ONLY if `allowFull` is set and the index is unavailable (old Worker, build error, local preview).
 * Per-golfer detail (pga_benchmark and the rest) still comes from the profile batches when a golfer is opened.
 */
export function usePlayerDirectory({ load, allowFull, generation = null }: { load: boolean; allowFull: boolean; generation?: string | null }): PlayerDirectory {
  const slimKey = generation ? `${PLAYER_SEARCH_KEY}?gen=${generation}` : PLAYER_SEARCH_KEY;
  const slim = useDashboardData<unknown>(load ? slimKey : null);
  const slimCatalog = useMemo(() => parseSearchIndexOnce(slim.data), [slim.data]);
  const slimFailed = load && !slim.loading && slimCatalog === null && (slim.error !== null || slim.data !== null);
  const fullKey = generation ? `golfprice/player_profiles/${generation}/catalog.json` : PLAYER_CATALOG_KEY;
  const full = useDashboardData<unknown>(slimFailed && allowFull ? fullKey : null);
  const fullCatalog = useMemo(() => parseCatalog(full.data), [full.data]);
  const catalog = slimCatalog ?? fullCatalog;
  const loading = load && !catalog && (slim.loading || (slimFailed && allowFull && full.loading));
  const error = catalog ? null : slimFailed ? (allowFull ? (full.error ?? slim.error ?? (full.loading ? null : "The golfer list is unavailable.")) : slim.error) : null;
  return { catalog, loading, error, stale: slimCatalog ? slim.stale : full.stale, fullCatalog: !slimCatalog && !!fullCatalog };
}

/**
 * Recent-form ratings (pga_benchmark) for one field, e.g. the ~150 players of a saved run. Asks the Worker for just those golfers
 * (a few tens of KB) and falls back to the full 16 MB catalog only if that endpoint errors. Catalog-shaped result: players carry dg_id and pga_benchmark only.
 */
export function usePlayerBenchmarks(ids: number[] | null): { catalog: ProfileCatalog | null; loading: boolean } {
  const sorted = useMemo(() => (ids && ids.length ? [...new Set(ids)].sort((a, b) => a - b) : null), [ids]);
  const slim = useDashboardData<unknown>(sorted ? `${PLAYER_BENCHMARKS_KEY}?ids=${sorted.join(",")}` : null);
  const slimCatalog = useMemo(() => parseBenchmarks(slim.data), [slim.data]);
  const failed = !!sorted && !slim.loading && slimCatalog === null;
  const full = useDashboardData<unknown>(failed ? PLAYER_CATALOG_KEY : null);
  const fullCatalog = useMemo(() => parseCatalog(full.data), [full.data]);
  return { catalog: slimCatalog ?? fullCatalog, loading: !!sorted && !slimCatalog && (slim.loading || (failed && full.loading)) };
}
