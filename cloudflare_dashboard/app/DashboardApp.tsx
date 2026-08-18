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
  Microscope,
  Settings2,
  Trophy,
  TrendingUp,
  X,
} from "lucide-react";
import { useDashboardData } from "./data";
import { displayDate, titleCase } from "./lib";
import {
  DiagnosticsView,
  DistributionsView,
  HistoryView,
  OutrightsView,
  PerformanceView,
  RoundScoresView,
  SgDistributionsView,
  WeatherView,
} from "./views";

export type ViewKey = "outrights" | "distributions" | "sg-distributions" | "round-scores" | "history" | "performance" | "diagnostics" | "weather";

type Manifest = {
  generated_at: string;
  event: string;
  event_id: number | string | null;
  course: string;
  course_id: number | string | null;
  par: number | null;
  rounds: number[];
};

const navigation: Array<{ label: string; items: Array<{ key: ViewKey; label: string; description: string; icon: typeof Trophy }> }> = [
  {
    label: "Live",
    items: [
      { key: "outrights", label: "Outrights", description: "Win and finish equity", icon: Trophy },
      { key: "round-scores", label: "Round scores", description: "Score distributions", icon: CircleGauge },
      { key: "weather", label: "Weather", description: "Forecast and impact", icon: CloudSun },
    ],
  },
  {
    label: "Model",
    items: [
      { key: "distributions", label: "Finish distributions", description: "Rank probability curves", icon: ChartNoAxesCombined },
      { key: "sg-distributions", label: "SG distributions", description: "Category inputs", icon: Activity },
      { key: "history", label: "History", description: "Archived simulations", icon: Archive },
    ],
  },
  {
    label: "Review",
    items: [
      { key: "performance", label: "Performance", description: "P&L and attribution", icon: TrendingUp },
      { key: "diagnostics", label: "Diagnostics", description: "Model quality", icon: Microscope },
    ],
  },
];

const views: Record<ViewKey, React.ComponentType> = {
  outrights: OutrightsView,
  distributions: DistributionsView,
  "sg-distributions": SgDistributionsView,
  "round-scores": RoundScoresView,
  history: HistoryView,
  performance: PerformanceView,
  diagnostics: DiagnosticsView,
  weather: WeatherView,
};

const accents = [
  { name: "Turf", value: "#54d6c8" },
  { name: "Sky", value: "#8ca7ff" },
  { name: "Citrus", value: "#d3e86b" },
  { name: "Rose", value: "#f27ea9" },
];

export function DashboardApp({ initialView }: { initialView: ViewKey }) {
  const activeView = views[initialView] ? initialView : "outrights";
  const ActiveView = views[activeView];
  const { data: manifest } = useDashboardData<Manifest>("manifest.json");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [density, setDensity] = useState<"comfortable" | "compact">(() => {
    if (typeof window === "undefined") return "comfortable";
    return localStorage.getItem("golf-dashboard-density") === "compact" ? "compact" : "comfortable";
  });
  const [accent, setAccent] = useState(() => {
    if (typeof window === "undefined") return accents[0].value;
    return localStorage.getItem("golf-dashboard-accent") || accents[0].value;
  });

  useEffect(() => {
    document.documentElement.dataset.density = density;
    document.documentElement.style.setProperty("--accent", accent);
    localStorage.setItem("golf-dashboard-density", density);
    localStorage.setItem("golf-dashboard-accent", accent);
  }, [accent, density]);

  const activeMeta = useMemo(() => navigation.flatMap((group) => group.items).find((item) => item.key === activeView), [activeView]);

  return (
    <div className={`app-shell ${collapsed ? "sidebar-collapsed" : ""}`}>
      <aside className={`sidebar ${sidebarOpen ? "mobile-open" : ""}`}>
        <div className="brand-row">
          <Link className="brand" href="/outrights" aria-label="Golf Model home">
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
                  <Link className={activeView === item.key ? "active" : ""} href={`/${item.key}`} key={item.key} title={collapsed ? item.label : undefined}>
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
            <div><strong>{titleCase(manifest?.event || "Tournament")}</strong><small>{manifest?.par ? `Par ${manifest.par}` : "Course model"}{manifest?.event_id ? ` · Event ${manifest.event_id}` : ""}</small></div>
            <div className="freshness"><strong>{displayDate(manifest?.generated_at)}</strong><small>Data snapshot</small></div>
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
            <section><h3>Accent color</h3><p>Use color to make key model signals easier to spot.</p><div className="accent-picker">{accents.map((option) => <button type="button" className={accent === option.value ? "active" : ""} key={option.value} onClick={() => setAccent(option.value)}><i style={{ backgroundColor: option.value }}/><span>{option.name}</span></button>)}</div></section>
            <section className="settings-note"><CircleGauge size={18}/><div><strong>Tables remember what matters</strong><p>Every table has its own sortable columns, visibility controls, search, and CSV export.</p></div></section>
          </div>
        </div>
      )}
    </div>
  );
}
