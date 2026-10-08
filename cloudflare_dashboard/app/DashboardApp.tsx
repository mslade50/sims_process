"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  Activity,
  Archive,
  ChartNoAxesCombined,
  ChevronLeft,
  ChevronRight,
  CircleGauge,
  CloudSun,
  Menu,
  Play,
  Microscope,
  Settings2,
  SlidersHorizontal,
  TrendingUp,
  Users,
  X,
} from "lucide-react";
import { ThisWeekView, WhyPricedView } from "./ExplainViews";
import { DistributionView } from "./DistributionView";
import { ScoringView } from "./ScoringView";
import { PlayersView, WeeklyPlayersView } from "./PlayerProfilesView";
import { InputsView } from "./InputsView";
import { ResearchView } from "./ResearchView";
import { RunView } from "./RunView";
import { WeatherEffectsView } from "./WeatherEffectsView";
import { useDashboardData } from "./data";
import { displayDate, titleCase } from "./lib";
import { FreshnessBadge } from "./ui";
import { ACCENTS, ACCENT_STORAGE_KEY, THEME_STORAGE_KEY, type AccentKey, type ThemeMode } from "./ui-rules";
import {
  DiagnosticsView,
  HistoryView,
  PerformanceView,
  SgDistributionsView,
  WeatherView,
} from "./views";

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

const navigation: Array<{ label: string; items: Array<{ key: ViewKey; label: string; description: string; icon: typeof CircleGauge }> }> = [
  {
    label: "Operate",
    items: [{ key: "run", label: "Run", description: "Start a golfprice job from your phone", icon: Play }],
  },
  {
    label: "Week",
    items: [
      { key: "this-week", label: "This week", description: "Course, model vs market, who we favour and why", icon: CircleGauge },
      { key: "weekly-players", label: "Field profiles", description: "This week’s players and every skill component", icon: Users },
      { key: "why-priced", label: "Why priced", description: "Every player's price, shape and drivers", icon: SlidersHorizontal },
    ],
  },
  {
    label: "Live",
    items: [
      { key: "round-scores", label: "Scoring expectation", description: "Expected score, drivers and uncertainty", icon: CircleGauge },
      { key: "weather", label: "Weather", description: "Forecast and impact", icon: CloudSun },
      { key: "weather-effects", label: "Weather effects", description: "Forecast, mean and variance by tee time", icon: CloudSun },
    ],
  },
  {
    label: "Model",
    items: [
      { key: "players", label: "Player directory", description: "Strengths, playing style and career coverage", icon: Users },
      { key: "inputs", label: "Model inputs", description: "Skill, course fit, variance, adjust", icon: SlidersHorizontal },
      { key: "distributions", label: "Finish distributions", description: "Rank probability curves", icon: ChartNoAxesCombined },
      { key: "sg-distributions", label: "SG distributions", description: "Category inputs", icon: Activity },
      { key: "history", label: "History", description: "Archived simulations", icon: Archive },
    ],
  },
  {
    label: "Review",
    items: [
      { key: "performance", label: "Performance", description: "P&L and attribution", icon: TrendingUp },
      { key: "research", label: "Betting backtests", description: "Model experiments", icon: Microscope },
      { key: "diagnostics", label: "Diagnostics", description: "Model quality", icon: Microscope },
    ],
  },
];

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

export function DashboardApp({ initialView }: { initialView: ViewKey }) {
  const activeView = views[initialView] ? initialView : "performance";
  const ActiveView = views[activeView];
  const { data: manifest } = useDashboardData<Manifest>("manifest.json");
  // golfprice is the production model since 2026-10-06: the header shows its current week (all events sharing the newest start date),
  // falling back to the legacy sims_process manifest only when the golfprice index is unavailable.
  const { data: gpIndex } = useDashboardData<{ events?: Array<{ name?: string; course?: string; date_start?: string; event_uid?: string; runs?: Array<{ as_of?: string }> }> }>("golfprice/index.json");
  const gpCurrent = useMemo(() => {
    const evs = (gpIndex?.events ?? []).filter((e) => e.date_start);
    if (!evs.length) return null;
    const newest = evs.map((e) => e.date_start as string).sort().at(-1);
    const week = evs.filter((e) => e.date_start === newest);
    const asOf = week.flatMap((e) => (e.runs ?? []).map((r) => r.as_of ?? "")).sort().at(-1);
    return { title: week.map((e) => e.name ?? e.event_uid ?? "").join(" · "), sub: week.map((e) => e.course ?? "").filter(Boolean).join(" · "), asOf };
  }, [gpIndex]);
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

  const activeMeta = useMemo(() => navigation.flatMap((group) => group.items).find((item) => item.key === activeView), [activeView]);

  return (
    <div className={`app-shell ${collapsed ? "sidebar-collapsed" : ""}`}>
      <aside className={`sidebar ${sidebarOpen ? "mobile-open" : ""}`}>
        <div className="brand-row">
          <Link className="brand" href="/performance" onClick={(event) => navigateWithReload(event, "/performance")} aria-label="Golf Model home">
            <span className="brand-mark"><i /><i /><i /></span>
            {!collapsed && <span><strong>Golf Model</strong><small>Simulation intelligence</small></span>}
          </Link>
          <button className="mobile-close" type="button" onClick={() => setSidebarOpen(false)} aria-label="Close navigation"><X size={19} /></button>
        </div>
        <nav aria-label="Dashboard navigation">
          {navigation.map((group) => (
            <div className="nav-group" key={group.label}>
              {!collapsed && <span className="nav-label">{group.label}</span>}
              {group.items.map((item) => {
                const Icon = item.icon;
                return (
                  <Link className={activeView === item.key ? "active" : ""} href={`/${item.key}`} onClick={(event) => navigateWithReload(event, `/${item.key}`)} key={item.key} title={collapsed ? item.label : undefined}>
                    <Icon size={18} />
                    {!collapsed && <span><strong>{item.label}</strong><small>{item.description}</small></span>}
                  </Link>
                );
              })}
            </div>
          ))}
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
          <div className="event-context">
            <span className="live-indicator"><i /> Published</span>
            {gpCurrent ? (
              <div><strong>{gpCurrent.title}</strong><small>{gpCurrent.sub || "golfprice"}</small></div>
            ) : (
              <div><strong>{titleCase(manifest?.event || "Tournament")}</strong><small>{manifest?.par ? `Par ${manifest.par}` : "Course model"}{manifest?.event_id ? ` · Event ${manifest.event_id}` : ""}</small></div>
            )}
            <div className="freshness"><strong>{displayDate(gpCurrent?.asOf || manifest?.generated_at)}</strong><small>{gpCurrent ? "Latest golfprice run" : "Data snapshot"}</small></div>
            <FreshnessBadge at={gpCurrent?.asOf || manifest?.generated_at} label={gpCurrent ? "Last run" : "Data"} />
          </div>
        </header>
        <div className="content-frame"><ActiveView /></div>
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
