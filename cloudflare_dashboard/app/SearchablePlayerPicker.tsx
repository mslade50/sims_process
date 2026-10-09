"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import { Search } from "lucide-react";
import { playerId, profileMatches, sortProfiles, type ProfileEntry } from "./player-profile-rules";

export function SearchablePlayerPicker({ players, excludedIds, disabled, onSelect }: {
  players: ProfileEntry[]; excludedIds: string[]; disabled: boolean; onSelect: (id: string) => void;
}) {
  const id = useId();
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [limit, setLimit] = useState(80);
  const wrapper = useRef<HTMLDivElement>(null);
  const list = useRef<HTMLDivElement>(null);
  const sorted = useMemo(() => sortProfiles(players), [players]);
  const matches = sorted.filter(p => !excludedIds.includes(playerId(p)) && (profileMatches(p,query) || playerId(p) === query.trim()));
  const visible = matches.slice(0,limit);
  const expanded = open && !disabled;
  const current = Math.min(active, visible.length - 1);
  useEffect(() => {
    if(expanded) list.current?.querySelector('[aria-selected="true"]')?.scrollIntoView({block:"nearest"});
  }, [current, expanded]);
  function choose(player: ProfileEntry) {
    onSelect(playerId(player)); setQuery(""); setOpen(false); setActive(0); setLimit(80);
  }
  return <div className="pp-player-picker" ref={wrapper} onBlur={event => {
    if(!event.currentTarget.contains(event.relatedTarget as Node | null)) setOpen(false);
  }}>
    <label htmlFor={`${id}-input`}>Compare with other players</label>
    <div className="pp-player-picker-input"><Search size={16} aria-hidden="true"/>
      <input id={`${id}-input`} role="combobox" aria-autocomplete="list" aria-expanded={expanded}
        aria-controls={expanded ? `${id}-list` : undefined} aria-activedescendant={expanded && current >= 0 ? `${id}-option-${current}` : undefined}
        aria-describedby={`${id}-hint`} autoComplete="off" disabled={disabled} value={query}
        placeholder={disabled ? "Two comparisons selected" : "Search by player name…"}
        onFocus={()=>setOpen(true)} onClick={()=>setOpen(true)}
        onChange={event=>{setQuery(event.target.value);setOpen(true);setActive(0);setLimit(80);}}
        onKeyDown={event=>{
          if(event.key === "Escape"){setOpen(false);event.preventDefault();}
          if(event.key === "ArrowDown" || event.key === "ArrowUp"){
            event.preventDefault();setOpen(true);
            const next = event.key === "ArrowDown" ? Math.min(Math.max(current,0)+1,matches.length-1) : Math.max(0,current-1);
            if(next >= limit) setLimit(limit+80);
            setActive(expanded ? next : 0);
          }
          if(event.key === "Enter" && expanded){event.preventDefault();if(visible[current])choose(visible[current]);}
        }}/>
    </div>
    <span id={`${id}-hint`} className="sr-only">Add up to two players. Search by name. Use the up and down arrows, then Enter to select. Escape closes the list.</span>
    {expanded && <div className="pp-player-options">
      <div id={`${id}-list`} role="listbox" aria-label="Players" ref={list}>
        {visible.map((p,index)=><div key={playerId(p)} id={`${id}-option-${index}`} role="option" tabIndex={-1} aria-selected={index===current} onKeyDown={event=>{if(event.key==="Enter" || event.key===" "){event.preventDefault();choose(p);}}}
          onMouseDown={event=>event.preventDefault()} onMouseMove={()=>setActive(index)} onClick={()=>choose(p)}>
          <strong>{p.name}</strong>{(p.tours ?? []).length>0 && <small>{(p.tours ?? []).join(" · ").toUpperCase()}</small>}
        </div>)}
        {!matches.length && <p className="pp-player-empty">No players match “{query}”. Try a different spelling.</p>}
      </div>
      <div className="pp-player-options-footer"><span role="status">Showing {visible.length} of {matches.length}</span>
        {matches.length>visible.length && <button type="button" onMouseDown={event=>event.preventDefault()} onClick={()=>setLimit(limit+80)}>Show more</button>}
      </div>
    </div>}
  </div>;
}
