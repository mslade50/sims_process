"use client";

/** Header type-ahead for golfers. The catalog loads lazily, only after the box is first focused or typed in, then navigates to /players?player=<id>. */
import { useMemo, useState } from "react";
import { Search } from "lucide-react";
import { useDashboardData } from "./data";
import { parseCatalog, type ProfileEntry } from "./player-profile-rules";
import { PLAYER_CATALOG_KEY, playerHref, searchPlayers } from "./shell-rules";

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
            <strong>{p.name}</strong><small>{(p.tours ?? []).join(" · ").toUpperCase() || "Tour unavailable"}{p.country ? ` · ${p.country}` : ""}</small>
          </a>
        </li>
      ))}
    </ul>
  );
}

export function HeaderSearch({ onNavigate }: { onNavigate: (event: React.MouseEvent<HTMLAnchorElement>, href: string) => void }) {
  const [query, setQuery] = useState("");
  const [armed, setArmed] = useState(false);
  const [active, setActive] = useState(0);
  const { data, loading } = useDashboardData<unknown>(armed ? PLAYER_CATALOG_KEY : null);
  const catalog = useMemo(() => parseCatalog(data), [data]);
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
