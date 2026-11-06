"use client";

/**
 * Shell pieces for design system v2: the event switcher (one event identity), the event status (freshness as usability),
 * the phone tab bar, the context-bar controls and the shared modal-dialog focus contract.
 * Pure navigation helpers live in shell-rules.ts; DashboardApp owns state and routing.
 */
import Link from "next/link";
import { useEffect, useId, useRef, useState, type RefObject } from "react";
import { CircleGauge, CloudSun, Ellipsis, Info, Moon, Search, Sun, Ticket, TriangleAlert, Users, Scale, ChevronDown } from "lucide-react";
import { eventHref, primaryTabKeys, tourShort, type ShellEvent } from "./shell-rules";
import { etStamp, parseTimestamp, relativeAge } from "./ui-rules";
import { useNow } from "./ui";

type Navigate = (event: React.MouseEvent<HTMLAnchorElement>, href: string) => void;

const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';
const visibleFocusables = (root: HTMLElement) => [...root.querySelectorAll<HTMLElement>(FOCUSABLE)].filter((el) => el.getClientRects().length > 0);

/**
 * Modal dialog contract: focus moves in, Tab and Shift+Tab stay inside, Escape closes, everything matched by `inertSelectors`
 * is inert while open, and focus returns to the element that opened it (or `returnSelector` when the browser never focused it).
 */
export function useDialogFocus(open: boolean, ref: RefObject<HTMLElement | null>, onClose: () => void, inertSelectors: string[], opts: { initial?: string; returnSelector?: string } = {}) {
  const close = useRef(onClose);
  useEffect(() => { close.current = onClose; });
  const { initial, returnSelector } = opts;
  const selectors = inertSelectors.join("|");
  useEffect(() => {
    const root = ref.current;
    if (!open || !root) return;
    const opener = document.activeElement instanceof HTMLElement && document.activeElement !== document.body ? document.activeElement : null;
    const inert = selectors.split("|").filter(Boolean).flatMap((sel) => [...document.querySelectorAll<HTMLElement>(sel)]).filter((el) => el !== root && !el.contains(root) && !root.contains(el));
    inert.forEach((el) => el.setAttribute("inert", ""));
    const frame = requestAnimationFrame(() => (root.querySelector<HTMLElement>(initial ?? "") ?? visibleFocusables(root)[0] ?? root).focus());
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") { event.preventDefault(); close.current(); return; }
      if (event.key !== "Tab") return;
      const items = visibleFocusables(root);
      if (!items.length) { event.preventDefault(); return; }
      const first = items[0], last = items[items.length - 1];
      if (!root.contains(document.activeElement)) { event.preventDefault(); first.focus(); }
      else if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      cancelAnimationFrame(frame);
      document.removeEventListener("keydown", onKey);
      inert.forEach((el) => el.removeAttribute("inert"));
      (opener && opener.isConnected ? opener : returnSelector ? document.querySelector<HTMLElement>(returnSelector) : null)?.focus();
    };
  }, [open, ref, selectors, initial, returnSelector]);
}

/** Closes a popover on Escape (focus back to its trigger) and on a press outside it. */
function usePopover(open: boolean, setOpen: (open: boolean) => void, ref: RefObject<HTMLElement | null>, triggerRef: RefObject<HTMLElement | null>) {
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") { setOpen(false); triggerRef.current?.focus(); } };
    const onPress = (event: Event) => { if (!ref.current?.contains(event.target as Node)) setOpen(false); };
    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onPress);
    return () => { document.removeEventListener("keydown", onKey); document.removeEventListener("pointerdown", onPress); };
  }, [open, setOpen, ref, triggerRef]);
}

/* ------------------------------------------------------------------ freshness = usability */
export type StatusTone = "neutral" | "warning" | "negative";
/** Colour rule: age alone is never coloured. Amber when prices lag the event stage; red only when the data itself is broken (a saved copy). */
export function statusTone(event: ShellEvent, savedCopy: boolean): StatusTone {
  if (savedCopy) return "negative";
  return event.status.state === "behind" ? "warning" : "neutral";
}
function ageWords(asOf: string | null, now: number): string {
  const at = parseTimestamp(asOf);
  return at === null ? "no run yet" : relativeAge(now - at);
}

/**
 * Context-bar status, e.g. "R2 in play | Prices after R1 | Run 13 h ago". Three separate facts and one disclosure that says what it means
 * for the numbers and when they next change. Neutral by default; amber when the prices lag the event; red only for a saved copy.
 */
export function EventStatusBar({ event, savedCopy, owner }: { event: ShellEvent; savedCopy: boolean; owner: boolean }) {
  const now = useNow(30_000);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const panelId = useId();
  usePopover(open, setOpen, ref, trigger);
  const tone = statusTone(event, savedCopy);
  const { status } = event;
  const age = ageWords(event.asOf, now);
  const at = parseTimestamp(event.asOf);
  const Icon = tone === "neutral" ? Info : TriangleAlert;
  const consequence = savedCopy ? "The data service could not be reached, so this is a saved copy and may be out of date. Reload to retry." : status.consequence;
  const warning = tone === "warning" ? "Prices lag the event." : tone === "negative" ? "Saved copy." : "";
  return (
    <div className="evstat" ref={ref} data-tone={tone}>
      <button ref={trigger} type="button" className="evstat-btn" aria-expanded={open} aria-controls={open ? panelId : undefined} aria-label={`${status.stageNow}. ${status.pricesLabel}. ${at === null ? "No run yet." : `Run ${age}.`} ${warning} Show what this means`.replace(/\s+/g, " ")} onClick={() => setOpen((o) => !o)}>
        <Icon size={15} aria-hidden="true" />
        <span className={`evstat-stage${status.inPlay ? " is-live" : ""}`}>{status.stageNow}</span>
        <span className="evstat-prices">{status.pricesLabel}</span>
        <span className="evstat-age">{at === null ? "No run yet" : `Run ${age}`}</span>
      </button>
      {open && (
        <div id={panelId} className="evstat-pop" role="region" aria-label="What the prices reflect">
          <dl>
            <div><dt>Event now</dt><dd>{status.stageNow}</dd></div>
            <div><dt>Prices reflect</dt><dd>{status.pricesLabel}</dd></div>
            <div><dt>Last run</dt><dd>{at === null ? "None published" : `${etStamp(at)} (${age})`}</dd></div>
          </dl>
          <p className="evstat-consequence">{consequence}</p>
          <p className="evstat-next"><strong>Next update.</strong> {status.next}{owner && status.state !== "final" && <> <Link className="ex-link" href="/run" prefetch={false}>Open the Run page</Link></>}</p>
        </div>
      )}
    </div>
  );
}

/** Current event as the single identity in the bar; opens a list of this week's events (grouped by tour) that swaps ?e= on the current page. Fully keyboard operable. */
export function EventSwitcher({ events, selected, onNavigate }: { events: ShellEvent[]; selected: ShellEvent; onNavigate: Navigate }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const popId = useId();
  const now = useNow(30_000);
  usePopover(open, setOpen, ref, trigger);
  useEffect(() => {
    if (!open) return;
    (ref.current?.querySelector<HTMLElement>('.evsw-pop a[aria-current="true"]') ?? ref.current?.querySelector<HTMLElement>(".evsw-pop a"))?.focus();
  }, [open]);
  const tours = [...new Set(events.map((e) => e.tour))];
  const identity = (
    <>
      <span className="evsw-tour">{tourShort(selected.tour)}</span>
      <span className="evsw-name"><strong title={selected.name}>{selected.name}</strong>{selected.course && <small title={selected.course}>{selected.course}</small>}</span>
    </>
  );
  if (events.length < 2) return <div className="evsw"><div className="ctx-static evsw-static" style={{ display: "flex", alignItems: "center", gap: 8 }}>{identity}</div></div>;
  const move = (event: React.KeyboardEvent<HTMLDivElement>) => {
    const items = [...event.currentTarget.querySelectorAll<HTMLElement>("a")];
    const at = items.indexOf(document.activeElement as HTMLElement);
    const go = (i: number) => { event.preventDefault(); items[(i + items.length) % items.length]?.focus(); };
    if (event.key === "ArrowDown") go(at + 1);
    else if (event.key === "ArrowUp") go(at - 1);
    else if (event.key === "Home") go(0);
    else if (event.key === "End") go(items.length - 1);
    else if (event.key === " " && at >= 0) { event.preventDefault(); items[at].click(); }
  };
  return (
    <div className="evsw" ref={ref} onBlur={(event) => { const next = event.relatedTarget as Node | null; if (open && next && !event.currentTarget.contains(next)) setOpen(false); }}>
      <button ref={trigger} type="button" aria-expanded={open} aria-haspopup="true" aria-controls={open ? popId : undefined} aria-label={`Event: ${selected.name}. Switch event`} onClick={() => setOpen((o) => !o)} onKeyDown={(event) => { if (!open && (event.key === "ArrowDown" || event.key === "ArrowUp")) { event.preventDefault(); setOpen(true); } }}>
        {identity}<ChevronDown size={16} aria-hidden="true" />
      </button>
      {open && (
        /* eslint-disable-next-line jsx-a11y/no-noninteractive-element-interactions -- arrow-key roving focus for the buttons inside this labelled group */
        <div id={popId} className="evsw-pop" role="group" aria-label="This week's events" onKeyDown={move}>
          {tours.map((tour) => (
            <div key={tour || "other"}>
              <h3>{events.find((e) => e.tour === tour)?.tourLabel}</h3>
              {events.filter((e) => e.tour === tour).map((e) => {
                const href = eventHref(window.location.pathname, window.location.search, e.uid);
                return (
                  <a key={e.uid} href={href} aria-current={e.uid === selected.uid ? "true" : undefined} onClick={(event) => onNavigate(event, href)}>
                    <strong>{e.name}</strong>
                    {e.course && <small>{e.course}</small>}
                    <span className="evsw-row"><span className={`evsw-stage${e.status.inPlay ? " is-live" : ""}`}>{e.status.stageNow}</span><span>{e.status.pricesLabel}</span><span className="evsw-age">Run {ageWords(e.asOf, now)}</span></span>
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

/** Phone bottom bar: four primary destinations plus More, which opens the full grouped navigation sheet (a modal dialog on phones). */
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
      <button type="button" className="tabbar-more" aria-expanded={moreOpen} aria-haspopup="dialog" onClick={onMore}><Ellipsis size={22} aria-hidden="true" /><span>More</span></button>
    </nav>
  );
}
