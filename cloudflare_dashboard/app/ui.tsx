"use client";

/**
 * Design-system building blocks (see engine_notes/dash_design.md). Colors come only from CSS tokens (tokens.css);
 * motion respects prefers-reduced-motion (CSS media query plus useReducedMotion for JS-driven animation).
 */
import { useEffect, useId, useState, useSyncExternalStore } from "react";
import { etTime } from "./lib";
import { catalogFreshnessTone, countUpText, freshnessTone, parseTimestamp, relativeAge, type FreshnessTone } from "./ui-rules";

export type Tone = "positive" | "negative" | "warning" | "model" | "market" | "am" | "pm" | "accent" | "neutral";
export type SegmentTone = Tone | "chart-1" | "chart-2" | "chart-3" | "chart-4" | "chart-5" | "chart-6" | "chart-7" | "chart-8";

const REDUCED_QUERY = "(prefers-reduced-motion: reduce)";
function subscribeReduced(callback: () => void) {
  const query = window.matchMedia(REDUCED_QUERY);
  query.addEventListener("change", callback);
  return () => query.removeEventListener("change", callback);
}
/** True when the user prefers reduced motion (false during server render). */
export function useReducedMotion(): boolean {
  return useSyncExternalStore(subscribeReduced, () => window.matchMedia(REDUCED_QUERY).matches, () => false);
}

/** A clock that ticks every `intervalMs` (default 30 s) and pauses while the tab is hidden. */
export function useNow(intervalMs = 30_000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const tick = () => { if (!document.hidden) setNow(Date.now()); };
    const id = window.setInterval(tick, intervalMs);
    document.addEventListener("visibilitychange", tick);
    return () => { window.clearInterval(id); document.removeEventListener("visibilitychange", tick); };
  }, [intervalMs]);
  return now;
}

/** Renders `text` (e.g. "$1,234.5", "12.3%") counting up from zero once on mount and whenever it changes. */
export function AnimatedNumber({ text, duration = 750 }: { text: string; duration?: number }) {
  const reduced = useReducedMotion();
  const [shown, setShown] = useState<{ for: string; text: string }>({ for: text, text });
  useEffect(() => {
    if (reduced) return;
    let frame = 0;
    const start = performance.now();
    const step = (time: number) => {
      const progress = Math.min(1, (time - start) / duration);
      setShown({ for: text, text: countUpText(text, progress) });
      if (progress < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [text, reduced, duration]);
  return <>{reduced || shown.for !== text ? text : shown.text}</>;
}

export function LiveDot({ tone = "positive", pulse = true }: { tone?: Tone; pulse?: boolean }) {
  return <i className={`live-dot tone-${tone} ${pulse ? "pulse" : ""}`} aria-hidden="true" />;
}

export function Badge({ tone = "neutral", pulse = false, children }: { tone?: Tone; pulse?: boolean; children: React.ReactNode }) {
  return <span className={`badge tone-${tone}`}>{pulse && <LiveDot tone={tone} />}{children}</span>;
}

/** Signed change with a direction glyph so the sign never relies on color alone. */
export function Delta({ value, digits = 1, suffix = "" }: { value: number; digits?: number; suffix?: string }) {
  if (!Number.isFinite(value)) return <span className="delta">—</span>;
  const tone = value > 0 ? "positive" : value < 0 ? "negative" : "neutral";
  const glyph = value > 0 ? "▲" : value < 0 ? "▼" : "•";
  return (
    <span className={`delta tone-${tone}`}>
      <span aria-hidden="true">{glyph}</span> {Math.abs(value).toFixed(digits)}{suffix}
      <span className="sr-only">{value > 0 ? " up" : value < 0 ? " down" : " flat"}</span>
    </span>
  );
}

export function Sparkline({ values, tone = "accent", width = 96, height = 28, label = "Trend" }: { values: number[]; tone?: Tone; width?: number; height?: number; label?: string }) {
  const finite = values.filter(Number.isFinite);
  if (finite.length < 2) return null;
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const span = max - min || 1;
  const pad = 2;
  const points = finite.map((value, index) => [pad + (index / (finite.length - 1)) * (width - pad * 2), height - pad - ((value - min) / span) * (height - pad * 2)] as const);
  const line = points.map(([x, y], index) => `${index ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const last = points[points.length - 1];
  return (
    <svg className={`sparkline tone-${tone}`} width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label}>
      <path className="sparkline-area" d={`${line} L${last[0]},${height} L${points[0][0]},${height} Z`} />
      <path className="sparkline-line" d={line} pathLength={1} />
      <circle className="sparkline-end" cx={last[0]} cy={last[1]} r={2.4} />
    </svg>
  );
}

export type Segment = { label: string; value: number; tone?: SegmentTone };

/** A stacked horizontal bar (shares of a whole, model vs market, AM vs PM wave splits) with a legend. */
export function DistributionBar({ segments, label, format = (v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 1 }) }: { segments: Segment[]; label: string; format?: (value: number) => string }) {
  const total = segments.reduce((sum, s) => sum + (Number.isFinite(s.value) && s.value > 0 ? s.value : 0), 0);
  if (!total) return null;
  const toneOf = (s: Segment, index: number): SegmentTone => s.tone ?? (`chart-${(index % 8) + 1}` as SegmentTone);
  return (
    <div className="dist-bar" role="group" aria-label={label}>
      <div className="dist-track">
        {segments.map((s, index) => s.value > 0 && (
          <span key={s.label} className={`dist-seg tone-${toneOf(s, index)}`} style={{ flexGrow: s.value, animationDelay: `${index * 60}ms` }} title={`${s.label}: ${format(s.value)}`} />
        ))}
      </div>
      <ul className="dist-legend">
        {segments.map((s, index) => <li key={s.label}><i className={`tone-${toneOf(s, index)}`} aria-hidden="true" />{s.label} <b>{format(s.value)}</b></li>)}
      </ul>
    </div>
  );
}

export function Skeleton({ lines = 3, height = 14 }: { lines?: number; height?: number }) {
  return <div className="skeleton-stack" aria-hidden="true">{Array.from({ length: lines }, (_, i) => <span key={i} className="skeleton" style={{ height, width: `${100 - (i % 3) * 14}%` }} />)}</div>;
}

/** Placeholder shaped like a page (KPI row plus a panel) shown while data loads. */
export function SkeletonPage({ label = "Loading dashboard data" }: { label?: string }) {
  const id = useId();
  return (
    <div className="skeleton-page" role="status" aria-live="polite" aria-labelledby={id}>
      <div className="kpi-grid" aria-hidden="true">
        {Array.from({ length: 4 }, (_, i) => <div key={i} className="kpi skeleton-card"><span className="skeleton" style={{ height: 10, width: "40%" }} /><span className="skeleton" style={{ height: 26, width: "60%" }} /></div>)}
      </div>
      <div className="panel" aria-hidden="true"><Skeleton lines={5} height={16} /></div>
      <span className="sr-only" id={id}>{label}</span>
    </div>
  );
}

const FRESH_LABEL: Record<FreshnessTone, string> = { live: "Live", fresh: "Fresh", aging: "Aging", stale: "Stale", unknown: "No run yet" };

/** "Last run N minutes ago" that keeps ticking. `at` is an ISO (or golfprice folder-style) timestamp. */
export function FreshnessBadge({ at, label = "Last run", scale = "run", title }: { at: unknown; label?: string; scale?: "run" | "catalog"; title?: string }) {
  const now = useNow(30_000);
  const parsed = parseTimestamp(at);
  const age = parsed === null ? null : now - parsed;
  const tone = scale === "catalog" ? catalogFreshnessTone(age) : freshnessTone(age);
  const text = age === null ? FRESH_LABEL.unknown : `${label} ${relativeAge(age)}${scale === "catalog" && tone === "aging" ? " · stale" : ""}`;
  return (
    <span className={`freshness-badge fresh-${tone}`} title={title ?? (parsed === null ? undefined : `${label}: ${etTime(new Date(parsed).toISOString())}`)}>
      <LiveDot tone={tone === "aging" ? "warning" : tone === "stale" || tone === "unknown" ? "negative" : "positive"} pulse={tone === "live" || tone === "fresh"} />
      <span>{text}</span>
    </span>
  );
}
