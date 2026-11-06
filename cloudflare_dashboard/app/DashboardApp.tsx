"use client";

import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Archive, BookOpen, ChartColumn, ChartNoAxesCombined, ChevronLeft, ChevronRight, CircleGauge, CloudRain, CloudSun, Crosshair, Flag, LayoutDashboard, ListChecks, Microscope, Play, Scale, Settings2, SlidersHorizontal, Ticket, TrendingUp, Users, X } from "lucide-react";
import { readRememberedEvent, rememberEvent } from "./event-param";
import { ArchivedStatus, EventStatusBar, EventSwitcher, SearchToggle, TabBar, ThemeToggle, useDialogFocus } from "./ShellParts";
import { useDashboardData, useSavedCopies, warmDashboardData } from "./data";
import { LEGACY_VIEWS, explainWarmKey, slimExplainKey, isChunkLoadError, OWNER_ONLY_VIEWS, type NavGroup, groupOf, navEntryFor, navigationFor, resolveEventUid, resolveRoute, selectedShellEvent, shellEventFor, shellEvents, unknownEventParam, type NavItem } from "./shell-rules";
import { EmptyState, FreshnessBadge, PageGroupContext, PageHeader, useNow } from "./ui";
import { ACCENTS, ACCENT_STORAGE_KEY, THEME_STORAGE_KEY, savedCopyNotice, summarizeWeek, type AccentKey, type IndexEventLite, type ThemeMode } from "./ui-rules";
import { ViewBoundary, reloadOnceForStaleChunk, useChunkLoadRecovery } from "./ViewBoundary";
import { useViewer } from "./viewer";
import "./shell.css";

export type ViewKey = "model-walkthrough" | "shot-data" | "ops" | "players" | "weekly-players" | "run" | "inputs" | "research" | "distributions" | "sg-distributions" | "round-scores" | "history" | "performance" | "diagnostics" | "weather-effects" | "weather-edits" | "this-week" | "why-priced" | "bet-breakdowns" | "bet-card";

type Manifest = {
  generated_at: string;
  event: string;
  event_id: number | string | null;
  course: string;
  course_id: number | string | null;
  par: number | null;
  rounds: number[];
};

/** Icons by view key; the labels, groups and order live in shell-rules.ts (NAVIGATION) so tests can read them. */
const NAV_ICONS: Record<string, typeof CircleGauge> = {
  "this-week": LayoutDashboard, "weekly-players": Users, "why-priced": Scale, distributions: ChartNoAxesCombined, "round-scores": Flag, "weather-effects": CloudSun, "weather-edits": CloudRain,
  "model-walkthrough": BookOpen, "shot-data": Crosshair, ops: ListChecks, inputs: SlidersHorizontal, run: Play, performance: TrendingUp, "bet-breakdowns": ChartColumn, "bet-card": Ticket, history: Archive, diagnostics: Microscope,
};

/** The golfer search (slim index, directory parsing) loads after hydration; the server-rendered input stays in place meanwhile, so nothing shifts. */
const HeaderSearch = lazy(() => import("./HeaderSearch").then((m) => ({ default: m.HeaderSearch })));

/** Views load on demand: each file is its own chunk, so the first route ships the shell plus only the view being opened. */
const PlayersView = lazy(() => import("./PlayerProfilesView").then((m) => ({ default: m.PlayersView })));
const WeeklyPlayersView = lazy(() => import("./WeeklyPlayersView").then((m) => ({ default: m.WeeklyPlayersView })));
const ShotDataView = lazy(() => import("./ShotDataView").then((m) => ({ default: m.ShotDataView })));
const OpsView = lazy(() => import("./OpsView").then((m) => ({ default: m.OpsView })));
const WalkthroughView = lazy(() => import("./WalkthroughView").then((m) => ({ default: m.WalkthroughView })));
const RunView = lazy(() => import("./RunView").then((m) => ({ default: m.RunView })));
const InputsView = lazy(() => import("./InputsView").then((m) => ({ default: m.InputsView })));
const ResearchView = lazy(() => import("./ResearchView").then((m) => ({ default: m.ResearchView })));
const DistributionView = lazy(() => import("./DistributionView").then((m) => ({ default: m.DistributionView })));
const SgDistributionsView = lazy(() => import("./views").then((m) => ({ default: m.SgDistributionsView })));
const ScoringView = lazy(() => import("./ScoringView").then((m) => ({ default: m.ScoringView })));
const HistoryView = lazy(() => import("./views").then((m) => ({ default: m.HistoryView })));
const PerformanceView = lazy(() => import("./PerformanceView").then((m) => ({ default: m.PerformanceView })));
const DiagnosticsView = lazy(() => import("./views").then((m) => ({ default: m.DiagnosticsView })));
const WeatherEffectsView = lazy(() => import("./WeatherEffectsView").then((m) => ({ default: m.WeatherEffectsView })));
const WeatherEditsView = lazy(() => import("./WeatherEditsView").then((m) => ({ default: m.WeatherEditsView })));
const ThisWeekView = lazy(() => import("./ExplainViews").then((m) => ({ default: m.ThisWeekView })));
const BetBreakdownsView = lazy(() => import("./BetBreakdownsView").then((m) => ({ default: m.BetBreakdownsView })));
const BetCardView = lazy(() => import("./BetCardView").then((m) => ({ default: m.BetCardView })));
const WhyPricedView = lazy(() => import("./ExplainWhy").then((m) => ({ default: m.WhyPricedView })));

const loaders: Record<string, () => Promise<unknown>> = {
  "model-walkthrough": () => import("./WalkthroughView"),
  "shot-data": () => import("./ShotDataView"),
  "players": () => import("./PlayerProfilesView"),
  "weekly-players": () => import("./WeeklyPlayersView"),
  "ops": () => import("./OpsView"),
  "run": () => import("./RunView"),
  "inputs": () => import("./InputsView"),
  "research": () => import("./ResearchView"),
  "distributions": () => import("./DistributionView"),
  "sg-distributions": () => import("./views"),
  "round-scores": () => import("./ScoringView"),
  "history": () => import("./views"),
  "performance": () => import("./PerformanceView"),
  "diagnostics": () => import("./views"),
  "weather-effects": () => import("./WeatherEffectsView"),
  "weather-edits": () => import("./WeatherEditsView"),
  "this-week": () => import("./ExplainViews"),
  "why-priced": () => import("./ExplainWhy"),
  "bet-breakdowns": () => import("./BetBreakdownsView"),
  "bet-card": () => import("./BetCardView"),
};

const warmed = new Set<string>();
/** Warm a view's chunk (no render). Safe to call repeatedly; a failure is ignored and retried on the real navigation. */
function prefetchView(key: string) {
  const load = loaders[key];
  if (!load || warmed.has(key)) return;
  warmed.add(key);
  load().catch(() => warmed.delete(key));
}
// The view for the address being opened is requested as soon as this module runs (before hydration finishes), not when React first renders it.
if (typeof window !== "undefined") {
  const first = window.location.pathname.replace(/^\/+|\/+$/g, "").split("/")[0] || "this-week";
  if (Object.hasOwn(loaders, first)) prefetchView(first);
}
const IDLE_PREFETCH = ["this-week", "weekly-players", "why-priced"];
const PREFETCH_DELAY_MS = 8000;

const views: Record<ViewKey, React.ComponentType> = {
  "model-walkthrough": WalkthroughView,
  "shot-data": ShotDataView,
  players: PlayersView,
  "weekly-players": WeeklyPlayersView,
  ops: OpsView,
  run: RunView,
  inputs: InputsView,
  research: ResearchView,
  distributions: DistributionView,
  "sg-distributions": SgDistributionsView,
  "round-scores": ScoringView,
  history: HistoryView,
  performance: PerformanceView,
  diagnostics: DiagnosticsView,
  "weather-effects": WeatherEffectsView,
  "weather-edits": WeatherEditsView,
  "this-week": ThisWeekView,
  "why-priced": WhyPricedView,
  "bet-breakdowns": BetBreakdownsView,
  "bet-card": BetCardView,
};

function navigateWithReload(event: React.MouseEvent<HTMLAnchorElement>, href: string) {
  if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  event.preventDefault();
  window.location.assign(href);
}

function Moved({ to }: { to: string }) {
  useEffect(() => {
    window.location.replace(`${to}${window.location.search}`);
  }, [to]);
  return (
    <div>
      <PageHeader title="This page moved" purpose="The older page was replaced by a newer one." />
      <EmptyState title="Taking you to the new page" detail="If nothing happens, use the link below or the navigation." />
      <p><a className="ex-link" href={to}>Go to the new page</a></p>
    </div>
  );
}

function NotFound({ requested, navigation }: { requested: string; navigation: NavGroup[] }) {
  return (
    <div className="route-notice">
      <PageHeader title="That page does not exist" purpose={requested ? `There is no page at /${requested}.` : "There is no page at this address."} />
      <EmptyState title="Pick a page from the navigation" detail="Or start with one of these." />
      <ul>
        {navigation.filter((group) => !group.collapsed).flatMap((group) => group.items).map((item) => (
          <li key={item.key}><a className="ex-link" href={item.href}>{item.label}</a> <small>{item.description}</small></li>
        ))}
      </ul>
    </div>
  );
}


function NavLink({ item, activeView, collapsed }: { item: NavItem; activeView: string; collapsed: boolean }) {
  const Icon = NAV_ICONS[item.key] ?? CircleGauge;
  const owns = activeView === item.key || !!item.toggle?.some((t) => t.key === activeView);
  return (
    <>
      <a className={activeView === item.key ? "active" : owns ? "has-active" : ""} aria-current={owns ? (item.toggle ? "true" : "page") : undefined} href={item.href} onMouseEnter={() => prefetchView(item.key)} onFocus={() => prefetchView(item.key)} onClick={(event) => navigateWithReload(event, item.href)} title={collapsed ? item.label : item.description}>
        <Icon size={18} />
        <span className="nav-text"><strong>{item.label}</strong></span>
      </a>
      {item.toggle && (
        <div className="nav-toggle nav-text" role="group" aria-label={`${item.label} views`}>
          {item.toggle.map((t) => (
            <a key={t.key} className={activeView === t.key ? "active" : ""} href={t.href} onMouseEnter={() => prefetchView(t.key)} onFocus={() => prefetchView(t.key)} aria-current={activeView === t.key ? "page" : undefined} onClick={(event) => navigateWithReload(event, t.href)}>{t.label}</a>
          ))}
        </div>
      )}
    </>
  );
}

export function DashboardApp({ initialView }: { initialView: string }) {
  useChunkLoadRecovery();
  const route = resolveRoute(initialView, Object.keys(views));
  const activeKey = route.kind === "view" || route.kind === "redirect" || route.kind === "retired" ? route.key : "";
  const ActiveView = route.kind === "view" ? views[route.key as ViewKey] : route.kind === "retired" ? ResearchView : null;
  // Colleagues see everything but the owner-only areas (worker/owner.ts enforces it; this only hides the links).
  const viewer = useViewer();
  const navigation = navigationFor(viewer?.owner === true);
  const ownerOnly = route.kind === "view" && OWNER_ONLY_VIEWS.includes(route.key);
  const { data: manifest } = useDashboardData<Manifest>("manifest.json");
  // golfprice is the production model since 2026-10-06: the header shows its current week (all events sharing the newest start date),
  // with no run shown when the golfprice index is unavailable (the legacy manifest is only the badge on the Legacy views, never a stand-in for the current week). Freshness is the OLDEST event's
  // newest run, so a stale event cannot hide behind a fresh one (site audit C3).
  const { data: gpIndex, stale: indexSaved } = useDashboardData<{ events?: IndexEventLite[] }>("golfprice/index.json");
  const now = useNow(30_000);
  const savedNotice = savedCopyNotice(useSavedCopies(), now);
  const week = useMemo(() => summarizeWeek(gpIndex, now), [gpIndex, now]);
  const events = useMemo(() => shellEvents(gpIndex, now), [gpIndex, now]);
  useEffect(() => {
    // Warm the most used views only after the page has loaded and settled: on a phone connection these downloads (the Players view alone is ~55 KB)
    // otherwise share bandwidth with the data the visitor is waiting to read.
    const run = () => IDLE_PREFETCH.forEach(prefetchView);
    const w = window as Window & { requestIdleCallback?: (cb: () => void, opts?: { timeout: number }) => number; cancelIdleCallback?: (id: number) => void };
    const saveData = (navigator as Navigator & { connection?: { saveData?: boolean } }).connection?.saveData === true;
    if (saveData) return;
    let idleId: number | undefined;
    const timer = window.setTimeout(() => { if (w.requestIdleCallback) idleId = w.requestIdleCallback(run, { timeout: 4000 }); else run(); }, PREFETCH_DELAY_MS);
    return () => { window.clearTimeout(timer); if (idleId !== undefined) w.cancelIdleCallback?.(idleId); };
  }, []);
  // The web font is requested once the page has settled: the class below makes the CSS rule use it (a boot flag applies it at once on later visits, when it is cached).
  useEffect(() => {
    const root = document.documentElement;
    if (root.classList.contains("fonts-ready")) return;
    const apply = () => window.setTimeout(() => { root.classList.add("fonts-ready"); try { localStorage.setItem("golf-fonts", "1"); } catch { /* flag is only an optimisation */ } }, 300);
    if (document.readyState === "complete") apply(); else window.addEventListener("load", apply, { once: true });
    return () => window.removeEventListener("load", apply);
  }, []);
  const isLegacy = LEGACY_VIEWS.includes(activeKey);
  // An invalid ?e= used to fall back to the default event with no word of it (site audit B5). Read after mount so the server render matches.
  const [eventParam, setEventParam] = useState<string | null>(null);
  const [rememberedEvent, setRememberedEvent] = useState<string | null>(null);
  useEffect(() => {
    const read = () => { setEventParam(new URLSearchParams(window.location.search).get("e")); setRememberedEvent(readRememberedEvent()); };
    queueMicrotask(read);
    window.addEventListener("golf-event-changed", read);
    return () => window.removeEventListener("golf-event-changed", read);
  }, []);
  const badEvent = unknownEventParam(eventParam, gpIndex?.events);
  // The same resolver the pages use (useEventParam), so the bar and the page can never name different events.
  const resolvedUid = resolveEventUid(gpIndex, eventParam, rememberedEvent, now);
  // This week reads the selected event's latest explanation document. Ask for it as soon as the index names it, rather than after the page's own code has loaded.
  const warmKey = activeKey === "this-week" ? explainWarmKey(gpIndex?.events, resolvedUid) : null;
  useEffect(() => { if (warmKey) warmDashboardData(slimExplainKey(warmKey)); }, [warmKey]);
  const selectedEvent = shellEventFor(gpIndex, resolvedUid, now) ?? selectedShellEvent(events, null);
  useEffect(() => { if (eventParam && !badEvent) rememberEvent(eventParam); }, [eventParam, badEvent]);
  const [searchOpen, setSearchOpen] = useState(false);
  useEffect(() => {
    if (searchOpen) document.querySelector<HTMLInputElement>(".header-search input")?.focus();
  }, [searchOpen]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  // On phones the sidebar is the "More" sheet, a modal dialog; on larger screens it is plain navigation.
  const [phone, setPhone] = useState(false);
  useEffect(() => {
    const query = window.matchMedia("(max-width: 820px)");
    const read = () => { setPhone(query.matches); if (!query.matches) setSidebarOpen(false); };
    read();
    query.addEventListener("change", read);
    return () => query.removeEventListener("change", read);
  }, []);
  const sheetOpen = sidebarOpen && phone;
  const sidebarRef = useRef<HTMLElement>(null);
  const settingsRef = useRef<HTMLDivElement>(null);
  useDialogFocus(sheetOpen, sidebarRef, () => setSidebarOpen(false), ["main", ".tabbar", ".skip-link", ".sidebar-backdrop"], { initial: ".mobile-close", returnSelector: ".tabbar-more" });
  const [collapsed, setCollapsed] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  // The dialog code is fetched on the first click and the panel opens only once it is in hand, so focus handling finds it mounted.
  const [SettingsPanel, setSettingsPanel] = useState<typeof import("./ShellExtras").SettingsPanel | null>(null);
  const openSettings = () => { import("./ShellExtras").then((m) => { setSettingsPanel(() => m.SettingsPanel); setSettingsOpen(true); }).catch((error) => { if (isChunkLoadError(error)) reloadOnceForStaleChunk(); }); };
  useDialogFocus(settingsOpen, settingsRef, () => setSettingsOpen(false), [".sidebar", "main", ".tabbar", ".skip-link", ".sidebar-backdrop"], { returnSelector: ".sidebar-footer button" });
  // Saved preferences are read after mount: the server render has no localStorage, so a lazy initial read would differ from it on
  // hydration (React error #418). The boot script in <head> has already applied the saved theme/accent before first paint.
  const [density, setDensity] = useState<"comfortable" | "compact">("comfortable");
  const [accent, setAccent] = useState<AccentKey>(ACCENTS[0].key);
  const [theme, setTheme] = useState<ThemeMode>("auto");
  const [prefsRead, setPrefsRead] = useState(false);
  useEffect(() => {
    const read = () => {
      try {
        setDensity(localStorage.getItem("golf-dashboard-density") === "compact" ? "compact" : "comfortable");
        const savedAccent = localStorage.getItem(ACCENT_STORAGE_KEY);
        setAccent(ACCENTS.find((option) => option.key === savedAccent)?.key ?? ACCENTS[0].key);
        const savedTheme = localStorage.getItem(THEME_STORAGE_KEY);
        setTheme(savedTheme === "light" || savedTheme === "dark" ? savedTheme : "auto");
      } catch { /* storage unavailable: defaults stay */ }
      setPrefsRead(true);
    };
    queueMicrotask(read);
  }, []);

  useEffect(() => {
    if (!prefsRead) return;
    const root = document.documentElement;
    const option = ACCENTS.find((item) => item.key === accent) ?? ACCENTS[0];
    root.dataset.density = density;
    root.style.setProperty("--accent-d", option.d);
    root.style.setProperty("--accent-l", option.l);
    if (theme === "auto") delete root.dataset.theme;
    else root.dataset.theme = theme;
    try {
      localStorage.setItem("golf-dashboard-density", density);
      localStorage.setItem(ACCENT_STORAGE_KEY, accent);
      localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch { /* storage unavailable: the choice lasts for this visit */ }
  }, [accent, density, theme, prefsRead]);

  const activeMeta = (ownerOnly && viewer && !viewer.owner ? null : navEntryFor(activeKey)) ?? (route.kind === "notfound" || ownerOnly ? { label: "Page not found", description: "Choose a page from the navigation" } : route.kind === "retired" ? { label: "Retired", description: "This page was retired" } : null);
  const toggleTheme = () => setTheme(theme === "dark" || (theme === "auto" && window.matchMedia("(prefers-color-scheme: dark)").matches) ? "light" : "dark");
  const pinned = navigation.flatMap((group) => group.items).find((item) => item.key === "bet-card");
  // Until /api/whoami answers, hold the owner's pinned row open (invisible, same markup) so the groups below do not drop 69 px when it appears. A colleague's page closes it once.
  const pinnedHold = !viewer ? navigationFor(true).flatMap((group) => group.items).find((item) => item.key === "bet-card") : undefined;
  const archiveActive = isLegacy;
  const [archiveOpen, setArchiveOpen] = useState(archiveActive);

  return (
    <PageGroupContext.Provider value={{ group: activeKey === "players" ? "Players" : groupOf(activeKey), ownerOnly: ownerOnly }}>   {/* a golfer profile is not week-scoped: the field board stays "Week", the golfer page is "Players" */}
    <div className={`app-shell ${collapsed ? "sidebar-collapsed" : ""}`}>
      <a className="skip-link" href="#main-content" onClick={(event) => { event.preventDefault(); const target = document.getElementById("main-content"); target?.focus(); target?.scrollIntoView({ block: "start" }); }}>Skip to content</a>
      <aside ref={sidebarRef} className={`sidebar ${sidebarOpen ? "mobile-open" : ""}`} aria-label="Site menu" {...(sheetOpen ? { role: "dialog", "aria-modal": true } : {})}>
        <div className="brand-row">
          {/* eslint-disable-next-line @next/next/no-html-link-for-pages -- Native navigation avoids the deployed vinext client-router failure. */}
          <a className="brand" href="/this-week" onClick={(event) => navigateWithReload(event, "/this-week")} aria-label="Golf Model home">
            <span className="brand-mark"><i /><i /><i /></span>
            <span className="nav-text"><strong>Golf Model</strong><small>Simulation intelligence</small></span>
          </a>
          <button className="mobile-close" type="button" onClick={() => setSidebarOpen(false)} aria-label="Close navigation"><X size={19} /></button>
        </div>
        <nav aria-label="Dashboard navigation">
          {pinnedHold && <div className="nav-group nav-pinned" aria-hidden="true" style={{ visibility: "hidden" }}><NavLink item={pinnedHold} activeView={activeKey} collapsed={collapsed} /></div>}
          {pinned && <div className="nav-group nav-pinned"><NavLink item={pinned} activeView={activeKey} collapsed={collapsed} /></div>}
          {navigation.map((group) => {
            const items = group.items.filter((item) => item.key !== pinned?.key);
            if (!items.length) return null;
            return group.collapsed && !collapsed ? (
              <details className="nav-group nav-archive" key={group.label} open={archiveOpen} onToggle={(event) => setArchiveOpen(event.currentTarget.open)}>
                <summary><span className="nav-label">{group.label}</span></summary>
                {items.map((item) => <NavLink item={item} activeView={activeKey} collapsed={collapsed} key={item.key} />)}
              </details>
            ) : (
              <div className="nav-group" key={group.label}>
                {!collapsed && <span className="nav-label">{group.label}</span>}
                {items.map((item) => <NavLink item={item} activeView={activeKey} collapsed={collapsed} key={item.key} />)}
              </div>
            );
          })}
        </nav>
        <div className="sidebar-footer">
          <button type="button" onClick={openSettings}><Settings2 size={18} /><span className="nav-text">Customize</span></button>
          <button className="collapse-button" type="button" onClick={() => setCollapsed((value) => !value)} aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}>{collapsed ? <ChevronRight size={17} /> : <><ChevronLeft size={17} /><span>Collapse</span></>}</button>
        </div>
      </aside>

      {sidebarOpen && <button className="sidebar-backdrop" aria-hidden="true" tabIndex={-1} onClick={() => setSidebarOpen(false)} />}

      <main>
        <header className={`contextbar ${searchOpen ? "search-open" : ""}`}>
          <span className="ctx-page">{activeMeta?.label}</span>
          <span className="ctx-sep" aria-hidden="true">/</span>
          <div className="ctx-identity">
            {activeKey === "shot-data" ? <div className="ctx-static"><strong>Scheduled shot snapshots</strong></div> : isLegacy ? (
              <>
                <span className="legacy-pill">Legacy data</span>
                <ArchivedStatus lastData={manifest?.generated_at} />
              </>
            ) : selectedEvent ? (
              <EventSwitcher events={events} selected={selectedEvent} onNavigate={navigateWithReload} allEvents={activeKey === "bet-card"} />
            ) : (
              <div className="ctx-static"><strong>This week</strong><small>Run data unavailable</small></div>
            )}
          </div>
          {!isLegacy && activeKey !== "shot-data" && activeKey !== "bet-card" && (
            <div className="ctx-meta">
              {selectedEvent ? <EventStatusBar event={selectedEvent} savedCopy={indexSaved} owner={viewer?.owner === true} /> : <FreshnessBadge at={week?.oldestAsOf ?? null} label="Run" title={week?.tooltip || "When the model last priced this event"} savedCopy={indexSaved} />}
              {week?.heldBackNote && <span className="evstat-held" role="status" title={week.heldBackNote}>Newer run held back</span>}
            </div>
          )}
          <Suspense fallback={null}><HeaderSearch onNavigate={navigateWithReload} /></Suspense>
          <div className="ctx-actions">
            <SearchToggle open={searchOpen} onClick={() => setSearchOpen((o) => !o)} />
            <ThemeToggle onToggle={toggleTheme} />
          </div>
        </header>
        <div className="content-frame" id="main-content" tabIndex={-1}>
          {savedNotice && <p className="link-notice saved-copy-notice" role="alert" data-testid="saved-copy-notice">{savedNotice}</p>}
          {badEvent && <p className="link-notice" role="status">The event in this link is not in the published list, so the default event is shown instead.</p>}
          {route.kind === "notfound" ? <NotFound requested={route.requested} navigation={navigation} /> : ownerOnly && !viewer ? null : ownerOnly && !viewer?.owner ? <NotFound requested={activeKey} navigation={navigation} /> : route.kind === "redirect" ? <Moved to={route.to} /> : ActiveView ? <ViewBoundary viewKey={activeKey}><ActiveView /></ViewBoundary> : null}
        </div>
      </main>

      <TabBar activeKey={activeKey} owner={viewer?.owner === true} moreOpen={sidebarOpen} onMore={() => setSidebarOpen(true)} onNavigate={navigateWithReload} onPrefetch={prefetchView} />

      {settingsOpen && SettingsPanel && <SettingsPanel panelRef={settingsRef} onClose={() => setSettingsOpen(false)} density={density} setDensity={setDensity} theme={theme} setTheme={setTheme} accent={accent} setAccent={setAccent} />}
    </div>
    </PageGroupContext.Provider>
  );
}
