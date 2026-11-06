"use client";

/** Rarely used shell pieces (the customize dialog). Loaded on demand so the first route does not ship them. */
import type { RefObject } from "react";
import { CircleGauge, X } from "lucide-react";
import { ACCENTS, type AccentKey, type ThemeMode } from "./ui-rules";

export function SettingsPanel({ panelRef, onClose, density, setDensity, theme, setTheme, accent, setAccent }: { panelRef: RefObject<HTMLDivElement | null>; onClose: () => void; density: "comfortable" | "compact"; setDensity: (d: "comfortable" | "compact") => void; theme: ThemeMode; setTheme: (t: ThemeMode) => void; accent: AccentKey; setAccent: (a: AccentKey) => void }) {
  return (
        <div className="settings-layer" ref={panelRef} role="dialog" aria-modal="true" aria-label="Customize dashboard">
          <button className="settings-backdrop" onClick={() => onClose()} aria-label="Close customization" />
          <div className="settings-panel">
            <div className="settings-heading"><div><span className="eyebrow">Your workspace</span><h2>Customize</h2></div><button type="button" onClick={() => onClose()} aria-label="Close customization"><X size={19}/></button></div>
            <section><h3>Information density</h3><p>Choose how much data fits on screen. Your preference stays on this device.</p><div className="choice-grid"><button className={density === "comfortable" ? "active" : ""} onClick={() => setDensity("comfortable")}><span className="density-preview comfortable"><i/><i/><i/></span><strong>Comfortable</strong><small>More breathing room</small></button><button className={density === "compact" ? "active" : ""} onClick={() => setDensity("compact")}><span className="density-preview compact"><i/><i/><i/><i/></span><strong>Compact</strong><small>More rows at once</small></button></div></section>
            <section><h3>Theme</h3><p>Auto follows your phone or computer. Your choice stays on this device.</p><div className="theme-picker" role="group" aria-label="Theme">{(["auto", "dark", "light"] as const).map((mode) => <button type="button" className={theme === mode ? "active" : ""} aria-pressed={theme === mode} key={mode} onClick={() => setTheme(mode)}><span className={`theme-swatch ${mode}`}/><span>{mode === "auto" ? "Auto" : mode === "dark" ? "Dark" : "Light"}</span></button>)}</div></section>
            <section><h3>Accent color</h3><p>Use color to make key model signals easier to spot.</p><div className="accent-picker">{ACCENTS.map((option) => <button type="button" className={accent === option.key ? "active" : ""} aria-pressed={accent === option.key} key={option.key} onClick={() => setAccent(option.key)}><i style={{ backgroundColor: option.d }}/><span>{option.name}</span></button>)}</div></section>
            <section className="settings-note"><CircleGauge size={18}/><div><strong>Tables remember what matters</strong><p>Every table has its own sortable columns, visibility controls, search, and CSV export.</p></div></section>
          </div>
        </div>
  );
}
