"use client";

/**
 * Shell pieces for design system v2: the event switcher (one event identity), the phone tab bar and the context-bar controls.
 * Pure navigation helpers live in shell-rules.ts; DashboardApp owns state and routing.
 */
import { useEffect, useRef, useState } from "react";
import { CircleGauge, CloudSun, Ellipsis, Moon, Search, Sun, Ticket, Users, Scale, ChevronDown } from "lucide-react";
import { eventHref, primaryTabKeys, tourShort, type ShellEvent } from "./shell-rules";
import { Chip, FreshnessBadge } from "./ui";

type Navigate = (event: React.MouseEvent<HTMLAnchorElement>, href: string) => void;

function stageTone(e: ShellEvent) { return e.stageTone === "accent" ? "accent" : "neutral"; }

/** Current event as the single identity in the bar; opens a list of this week's events (grouped by tour) that swaps ?e= on the current page. */
export function EventSwitcher({ events, selected, onNavigate }: { events: ShellEvent[]; selected: ShellEvent; onNavigate: Navigate }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (event: Event) => { if (event instanceof KeyboardEvent ? event.key === "Escape" : !ref.current?.contains(event.target as Node)) setOpen(false); };
    document.addEventListener("keydown", close);
    document.addEventListener("pointerdown", close);
    return () => { document.removeEventListener("keydown", close); document.removeEventListener("pointerdown", close); };
  }, [open]);
  const tours = [...new Set(events.map((e) => e.tour))];
  const identity = (
    <>
      <span className="evsw-tour">{tourShort(selected.tour)}</span>
      <span className="evsw-name"><strong title={selected.name}>{selected.name}</strong>{selected.course && <small title={selected.course}>{selected.course}</small>}</span>
    </>
  );
  if (events.length < 2) return <div className="evsw"><div className="ctx-static evsw-static" style={{ display: "flex", alignItems: "center", gap: 8 }}>{identity}</div></div>;
  return (
    <div className="evsw" ref={ref}>
      <button type="button" aria-expanded={open} aria-haspopup="true" aria-label={`Event: ${selected.name}. Switch event`} onClick={() => setOpen((o) => !o)}>
        {identity}<ChevronDown size={16} aria-hidden="true" />
      </button>
      {open && (
        <div className="evsw-pop" role="group" aria-label="This week's events">
          {tours.map((tour) => (
            <div key={tour || "other"}>
              <h3>{events.find((e) => e.tour === tour)?.tourLabel}</h3>
              {events.filter((e) => e.tour === tour).map((e) => {
                const href = eventHref(window.location.pathname, window.location.search, e.uid);
                return (
                  <a key={e.uid} href={href} aria-current={e.uid === selected.uid ? "true" : undefined} onClick={(event) => onNavigate(event, href)}>
                    <strong>{e.name}</strong>
                    {e.course && <small>{e.course}</small>}
                    <span className="evsw-row"><Chip tone={stageTone(e)}>{e.stage}</Chip><FreshnessBadge at={e.asOf} label="Run" /></span>
                  </a>
                );
              })}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function SearchToggle({ open, onClick }: { open: boolean; onClick: () => void }) {
  return <button type="button" className="ctx-icon-btn ctx-search-btn" aria-label="Search golfers" aria-expanded={open} onClick={onClick}><Search size={18} aria-hidden="true" /></button>;
}

export function ThemeToggle({ onToggle }: { onToggle: () => void }) {
  return (
    <button type="button" className="ctx-icon-btn ctx-theme" aria-label="Switch between light and dark theme" onClick={onToggle}>
      <Sun className="i-sun" size={18} aria-hidden="true" /><Moon className="i-moon" size={18} aria-hidden="true" />
    </button>
  );
}

const TAB_ICONS: Record<string, typeof CircleGauge> = { "this-week": CircleGauge, "weekly-players": Users, "bet-card": Ticket, "why-priced": Scale, "weather-effects": CloudSun };
const TAB_LABELS: Record<string, string> = { "this-week": "This week", "weekly-players": "Players", "bet-card": "Bet card", "why-priced": "Why priced", "weather-effects": "Weather" };

/** Phone bottom bar: four primary destinations plus More, which opens the full grouped navigation sheet. */
export function TabBar({ activeKey, owner, moreOpen, onMore, onNavigate, onPrefetch }: { activeKey: string; owner: boolean; moreOpen: boolean; onMore: () => void; onNavigate: Navigate; onPrefetch: (key: string) => void }) {
  const keys = primaryTabKeys(owner);
  const owns = (key: string) => activeKey === key || (key === "weekly-players" && activeKey === "players");
  return (
    <nav className="tabbar" aria-label="Primary">
      {keys.map((key) => {
        const Icon = TAB_ICONS[key] ?? CircleGauge;
        const href = `/${key}`;
        return (
          <a key={key} href={href} aria-current={owns(key) ? "page" : undefined} onFocus={() => onPrefetch(key)} onClick={(event) => onNavigate(event, href)}>
            <Icon size={22} aria-hidden="true" /><span>{TAB_LABELS[key]}</span>
          </a>
        );
      })}
      <button type="button" aria-expanded={moreOpen} aria-haspopup="dialog" onClick={onMore}><Ellipsis size={22} aria-hidden="true" /><span>More</span></button>
    </nav>
  );
}
