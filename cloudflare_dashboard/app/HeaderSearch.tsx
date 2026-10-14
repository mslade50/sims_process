"use client";

/**
 * Header type-ahead for golfers, then navigates to /players?player=<id>. Reads the Worker's slim search index (identity only, ~0.3 MB gzipped)
 * rather than the 16 MB catalog: it is fetched once the browser is idle (so the first keystroke is instant) or on first focus, and the full
 * catalog is used only as a fallback after the person actually interacts with the box and the slim index is unavailable.
 */
import { useEffect, useMemo, useState } from "react";
import { Search } from "lucide-react";
import { playerDisplayName } from "./player-name";
import { usePlayerDirectory } from "./player-directory";
import type { ProfileEntry } from "./player-profile-rules";
import { playerHref, searchPlayers } from "./shell-rules";

function Results({ query, active, hits, loading, available, onNavigate }: { query: string; active: number; hits: ProfileEntry[]; loading: boolean; available: boolean; onNavigate: (event: React.MouseEvent<HTMLAnchorElement>, href: string) => void }) {
  if (query.trim().length < 2) return null;
  return (
    <ul className="header-search-results" role="listbox" aria-label="Golfer matches">
      {loading && <li><span>Loading golfers…</span></li>}
      {!loading && !available && <li><span>Golfer search is unavailable right now.</span></li>}
      {!loading && available && hits.length === 0 && <li><span>No golfer matches “{query.trim()}”.</span></li>}
      {hits.map((p, i) => (
        <li key={p.profile_key} role="option" aria-selected={i === active}>
          <a className={i === active ? "hit-active" : ""} href={playerHref(p)} onClick={(event) => onNavigate(event, playerHref(p))}>
            <strong>{playerDisplayName(p.display_name ?? p.name)}</strong><small>{(p.tours ?? []).join(" · ").toUpperCase() || "Tour unavailable"}{p.country ? ` · ${p.country}` : ""}</small>
          </a>
        </li>
      ))}
    </ul>
  );
}

export function HeaderSearch({ onNavigate }: { onNavigate: (event: React.MouseEvent<HTMLAnchorElement>, href: string) => void }) {
  const [query, setQuery] = useState("");
  const [armed, setArmed] = useState(false);
  const [prefetch, setPrefetch] = useState(false);
  const [active, setActive] = useState(0);
  // Warm the slim index when the browser is idle, unless the visitor asked to save data. Never falls back to the 16 MB catalog by itself.
  useEffect(() => {
    const saveData = (navigator as Navigator & { connection?: { saveData?: boolean } }).connection?.saveData === true;
    if (saveData) return;
    const idle = (window as Window & { requestIdleCallback?: (cb: () => void, opts?: { timeout: number }) => number; cancelIdleCallback?: (id: number) => void });
    if (idle.requestIdleCallback) {
      const id = idle.requestIdleCallback(() => setPrefetch(true), { timeout: 4000 });
      return () => idle.cancelIdleCallback?.(id);
    }
    const timer = window.setTimeout(() => setPrefetch(true), 2500);
    return () => window.clearTimeout(timer);
  }, []);
  const { catalog, loading } = usePlayerDirectory({ load: armed || prefetch, allowFull: armed });
  const hits = useMemo(() => (catalog ? searchPlayers(catalog.players, query) : []), [catalog, query]);
  return (
    <div className="header-search" role="search">
      <label>
        <Search size={16} aria-hidden="true" />
        <input
          type="search" aria-label="Search golfers" placeholder="Search golfers…" autoComplete="off" value={query}
          onFocus={() => setArmed(true)}
          onChange={(event) => { setQuery(event.target.value); setActive(0); setArmed(true); }}
          onKeyDown={(event) => {
            if (event.key === "ArrowDown") { event.preventDefault(); setActive((i) => Math.min(i + 1, Math.max(0, hits.length - 1))); }
            else if (event.key === "ArrowUp") { event.preventDefault(); setActive((i) => Math.max(0, i - 1)); }
            else if (event.key === "Escape") { setQuery(""); }
            else if (event.key === "Enter" && hits[active]) { event.preventDefault(); window.location.assign(playerHref(hits[active])); }
          }}
        />
      </label>
      {armed && <Results query={query} active={active} hits={hits} loading={loading} available={!!catalog} onNavigate={onNavigate} />}
    </div>
  );
}
