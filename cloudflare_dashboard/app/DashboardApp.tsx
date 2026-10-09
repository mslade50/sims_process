"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Archive,
  ChartNoAxesCombined,
  ChevronLeft,
  ChevronRight,
  CircleGauge,
  CloudSun,
  Menu,
  Microscope,
  Play,
  Settings2,
  SlidersHorizontal,
  TrendingUp,
  Users,
  X,
} from "lucide-react";
import { ThisWeekView, WhyPricedView } from "./ExplainViews";
import { DistributionView } from "./DistributionView";
import { ScoringView } from "./ScoringView";
import { PlayersView } from "./PlayerProfilesView";
import { WeeklyPlayersView } from "./WeeklyPlayersView";
import { InputsView } from "./InputsView";
import { ResearchView } from "./ResearchView";
import { RunView } from "./RunView";
import { WeatherEffectsView } from "./WeatherEffectsView";
import { HeaderSearch } from "./HeaderSearch";
import { EmptyState, PageIntro } from "./components";
import { useDashboardData } from "./data";
import { titleCase } from "./lib";
import { LEGACY_VIEWS, NAVIGATION, navEntryFor, resolveRoute, unknownEventParam, type NavItem } from "./shell-rules";
import { FreshnessBadge, useNow } from "./ui";
import { ACCENTS, ACCENT_STORAGE_KEY, THEME_STORAGE_KEY, summarizeWeek, type AccentKey, type IndexEventLite, type ThemeMode } from "./ui-rules";
import {
  DiagnosticsView,
  HistoryView,
  PerformanceView,
  SgDistributionsView,
  WeatherView,
} from "./views";
import "./shell.css";

export type ViewKey = "players" | "weekly-players" | "run" | "inputs" | "research" | "distributions" | "sg-distributions" | "round-scores" | "history" | "performance" | "diagnostics" | "weather" | "weather-effects" | "this-week" | "why-priced";

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
  "this-week": CircleGauge, "weekly-players": Users, "why-priced": SlidersHorizontal, distributions: ChartNoAxesCombined, "round-scores": CloudSun,
  inputs: SlidersHorizontal, run: Play, performance: TrendingUp, history: Archive, diagnostics: Microscope,
};

const views: Record<ViewKey, React.ComponentType> = {
  players: PlayersView,
  "weekly-players": WeeklyPlayersView,
  run: RunView,
  inputs: InputsView,
  research: ResearchView,
  distributions: DistributionView,
  "sg-distributions": SgDistributionsView,
  "round-scores": ScoringView,
  history: HistoryView,
  performance: PerformanceView,
  diagnostics: DiagnosticsView,
  weather: WeatherView,
  "weather-effects": WeatherEffectsView,
  "this-week": ThisWeekView,
  "why-priced": WhyPricedView,
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
      <PageIntro eyebrow="Moved" title="This page moved" description="The older page was replaced by a newer one." />
      <EmptyState title={`Now at ${to}`} detail="You are being taken there. If nothing happens, use the link in the navigation." />
      <p><a className="ex-link" href={to}>Go to {to}</a></p>
    </div>
  );
}

function NotFound({ requested }: { requested: string }) {
  return (
    <div className="route-notice">
      <PageIntro eyebrow="Not found" title="That page does not exist" description={requested ? `There is no page at /${requested}.` : "There is no page at this address."} />
      <EmptyState title="Pick a page from the navigation" detail="Or start with one of these." />
      <ul>
        {NAVIGATION.filter((group) => !group.collapsed).flatMap((group) => group.items).map((item) => (
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
      <a className={owns ? "active" : ""} href={item.href} onClick={(event) => navigateWithReload(event, item.href)} title={collapsed ? item.label : undefined}>
        <Icon size={18} />
        {!collapsed && <span><strong>{item.label}</strong><small>{item.description}</small></span>}
      </a>
      {!collapsed && item.toggle && (
        <div className="nav-toggle" role="group" aria-label={`${item.label} views`}>
          {item.toggle.map((t) => (
            <a key={t.key} className={activeView === t.key ? "active" : ""} href={t.href} aria-current={activeView === t.key ? "page" : undefined} onClick={(event) => navigateWithReload(event, t.href)}>{t.label}</a>
          ))}
        </div>
      )}
    </>
  );
}

export function DashboardApp({ initialView }: { initialView: string }) {
  const route = resolveRoute(initialView, Object.keys(views));
  const activeKey = route.kind === "view" || route.kind === "redirect" || route.kind === "retired" ? route.key : "";
  const ActiveView = route.kind === "view" ? views[route.key as ViewKey] : route.kind === "retired" ? ResearchView : null;
  const { data: manifest } = useDashboardData<Manifest>("manifest.json");
  // golfprice is the production model since 2026-10-06: the header shows its current week (all events sharing the newest start date),
  // falling back to the legacy sims_process manifest only when the golfprice index is unavailable. Freshness is the OLDEST event's
  // newest run, so a stale event cannot hide behind a fresh one (site audit C3).
  const { data: gpIndex } = useDashboardData<{ events?: IndexEventLite[] }>("golfprice/index.json");
  const now = useNow(30_000);
  const week = useMemo(() => summarizeWeek(gpIndex, now), [gpIndex, now]);
  const isLegacy = LEGACY_VIEWS.includes(activeKey);
  // An invalid ?e= used to fall back to the default event with no word of it (site audit B5). Read after mount so the server render matches.
  const [eventParam, setEventParam] = useState<string | null>(null);
  useEffect(() => {
    queueMicrotask(() => setEventParam(new URLSearchParams(window.location.search).get("e")));
  }, []);
  const badEvent = unknownEventParam(eventParam, gpIndex?.events);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [density, setDensity] = useState<"comfortable" | "compact">(() => {
    if (typeof window === "undefined") return "comfortable";
    return localStorage.getItem("golf-dashboard-density") === "compact" ? "compact" : "comfortable";
  });
  const [accent, setAccent] = useState<AccentKey>(() => {
    if (typeof window === "undefined") return ACCENTS[0].key;
    const saved = localStorage.getItem(ACCENT_STORAGE_KEY);
    return ACCENTS.find((option) => option.key === saved)?.key ?? ACCENTS[0].key;
  });
  const [theme, setTheme] = useState<ThemeMode>(() => {
    if (typeof window === "undefined") return "auto";
    const saved = localStorage.getItem(THEME_STORAGE_KEY);
    return saved === "light" || saved === "dark" ? saved : "auto";
  });

  useEffect(() => {
    const root = document.documentElement;
    const option = ACCENTS.find((item) => item.key === accent) ?? ACCENTS[0];
    root.dataset.density = density;
    root.style.setProperty("--accent-d", option.d);
    root.style.setProperty("--accent-l", option.l);
    if (theme === "auto") delete root.dataset.theme;
    else root.dataset.theme = theme;
    localStorage.setItem("golf-dashboard-density", density);
    localStorage.setItem(ACCENT_STORAGE_KEY, accent);
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [accent, density, theme]);

  const activeMeta = navEntryFor(activeKey) ?? (route.kind === "notfound" ? { label: "Page not found", description: "Choose a page from the navigation" } : route.kind === "retired" ? { label: "Retired", description: "This page was retired" } : null);
  const archiveActive = isLegacy;
  const [archiveOpen, setArchiveOpen] = useState(archiveActive);

  return (
    <div className={`app-shell ${collapsed ? "sidebar-collapsed" : ""}`}>
      <aside className={`sidebar ${sidebarOpen ? "mobile-open" : ""}`}>
        <div className="brand-row">
          {/* eslint-disable-next-line @next/next/no-html-link-for-pages -- Native navigation avoids the deployed vinext client-router failure. */}
          <a className="brand" href="/this-week" onClick={(event) => navigateWithReload(event, "/this-week")} aria-label="Golf Model home">
            <span className="brand-mark"><i /><i /><i /></span>
            {!collapsed && <span><strong>Golf Model</strong><small>Simulation intelligence</small></span>}
          </a>
          <button className="mobile-close" type="button" onClick={() => setSidebarOpen(false)} aria-label="Close navigation"><X size={19} /></button>
        </div>
        <nav aria-label="Dashboard navigation">
          {NAVIGATION.map((group) =>
            group.collapsed && !collapsed ? (
              <details className="nav-group nav-archive" key={group.label} open={archiveOpen} onToggle={(event) => setArchiveOpen(event.currentTarget.open)}>
                <summary><span className="nav-label">{group.label}</span></summary>
                {group.items.map((item) => <NavLink item={item} activeView={activeKey} collapsed={collapsed} key={item.key} />)}
              </details>
            ) : (
              <div className="nav-group" key={group.label}>
                {!collapsed && <span className="nav-label">{group.label}</span>}
                {group.items.map((item) => <NavLink item={item} activeView={activeKey} collapsed={collapsed} key={item.key} />)}
              </div>
            ),
          )}
        </nav>
        <div className="sidebar-footer">
          <button type="button" onClick={() => setSettingsOpen(true)}><Settings2 size={18} />{!collapsed && <span>Customize</span>}</button>
          <button className="collapse-button" type="button" onClick={() => setCollapsed((value) => !value)} aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}>{collapsed ? <ChevronRight size={17} /> : <><ChevronLeft size={17} /><span>Collapse</span></>}</button>
        </div>
      </aside>

      {sidebarOpen && <button className="sidebar-backdrop" aria-label="Close navigation" onClick={() => setSidebarOpen(false)} />}

      <main>
        <header className="topbar">
          <div className="topbar-left">
            <button className="menu-button" type="button" onClick={() => setSidebarOpen(true)} aria-label="Open navigation"><Menu size={20} /></button>
            <div><span>{activeMeta?.label}</span><small>{activeMeta?.description}</small></div>
          </div>
          <HeaderSearch onNavigate={navigateWithReload} />
          <div className="event-context">
            {isLegacy ? (
              <>
                <span className="legacy-pill">Legacy data</span>
                <FreshnessBadge at={manifest?.generated_at} label="Data" />
              </>
            ) : week ? (
              <>
                <div><strong>{week.title}</strong><small>{week.sub || "golfprice"}</small></div>
                {week.publishedLine && <span className="published-line">{week.publishedLine}</span>}
                {week.heldBackNote && <p className="held-note" role="status">{week.heldBackNote}</p>}
                <FreshnessBadge at={week.oldestAsOf} label={week.events.length > 1 ? "Oldest run" : "Last run"} title={week.tooltip} />
              </>
            ) : (
              <>
                <div><strong>{titleCase(manifest?.event || "Tournament")}</strong><small>{manifest?.par ? `Par ${manifest.par}` : "Course model"}{manifest?.event_id ? ` · Event ${manifest.event_id}` : ""}</small></div>
                <FreshnessBadge at={manifest?.generated_at} label="Data" />
              </>
            )}
          </div>
        </header>
        <div className="content-frame">
          {badEvent && <p className="link-notice" role="status">The event in this link (<code>{eventParam}</code>) is not in the published index, so the default event is shown instead.</p>}
          {route.kind === "notfound" ? <NotFound requested={route.requested} /> : route.kind === "redirect" ? <Moved to={route.to} /> : ActiveView ? <ActiveView /> : null}
        </div>
      </main>

      {settingsOpen && (
        <div className="settings-layer" role="dialog" aria-modal="true" aria-label="Customize dashboard">
          <button className="settings-backdrop" onClick={() => setSettingsOpen(false)} aria-label="Close customization" />
          <div className="settings-panel">
            <div className="settings-heading"><div><span className="eyebrow">Your workspace</span><h2>Customize</h2></div><button type="button" onClick={() => setSettingsOpen(false)} aria-label="Close customization"><X size={19}/></button></div>
            <section><h3>Information density</h3><p>Choose how much data fits on screen. Your preference stays on this device.</p><div className="choice-grid"><button className={density === "comfortable" ? "active" : ""} onClick={() => setDensity("comfortable")}><span className="density-preview comfortable"><i/><i/><i/></span><strong>Comfortable</strong><small>More breathing room</small></button><button className={density === "compact" ? "active" : ""} onClick={() => setDensity("compact")}><span className="density-preview compact"><i/><i/><i/><i/></span><strong>Compact</strong><small>More rows at once</small></button></div></section>
            <section><h3>Theme</h3><p>Auto follows your phone or computer. Your choice stays on this device.</p><div className="theme-picker" role="group" aria-label="Theme">{(["auto", "dark", "light"] as const).map((mode) => <button type="button" className={theme === mode ? "active" : ""} aria-pressed={theme === mode} key={mode} onClick={() => setTheme(mode)}><span className={`theme-swatch ${mode}`}/><span>{mode === "auto" ? "Auto" : mode === "dark" ? "Dark" : "Light"}</span></button>)}</div></section>
            <section><h3>Accent color</h3><p>Use color to make key model signals easier to spot.</p><div className="accent-picker">{ACCENTS.map((option) => <button type="button" className={accent === option.key ? "active" : ""} aria-pressed={accent === option.key} key={option.key} onClick={() => setAccent(option.key)}><i style={{ backgroundColor: option.d }}/><span>{option.name}</span></button>)}</div></section>
            <section className="settings-note"><CircleGauge size={18}/><div><strong>Tables remember what matters</strong><p>Every table has its own sortable columns, visibility controls, search, and CSV export.</p></div></section>
          </div>
        </div>
      )}
    </div>
  );
}
