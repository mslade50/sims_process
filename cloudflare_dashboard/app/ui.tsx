"use client";

/**
 * Design-system building blocks (docs/ui-overhaul/DESIGN_SYSTEM.md). Colors come only from CSS tokens (tokens.css, design.css);
 * motion respects prefers-reduced-motion (CSS media query plus useReducedMotion for JS-driven animation).
 *
 * Page primitives (design system v2):
 *   PageHeader      eyebrow, h1, one-line purpose, optional actions. One per page.
 *   Section         anchored section (id) with a rule + h2, optional purpose line and actions. Replaces a card per section.
 *   OnThisPage      sticky in-page section nav (horizontal scroller on phone); pass the same ids as the Sections.
 *   Stat            KPI tile: label, value, unit, signed Delta (arrow + colour), note. Same look as components.Kpi.
 *   Delta           signed change with arrow glyph; favourable green / adverse red; `inverse` when lower is better.
 *   GapRail         zero-centred model-vs-book bar with direct label "model 24.0% vs book 19.7% = +4.3 pts".
 *   Chip            small status/stage tag (neutral, positive, negative, warning, accent).
 *   Badge           pill with optional live dot (older API, same look).
 *   SegmentedControl  exclusive choice with aria-pressed buttons.
 *   EmptyState      compact honest empty/unavailable note: title, reason, optional next-check time, optional retry.
 *   Skeleton / SkeletonPage  shaped loading placeholders (no spinners).
 *   InfoTip         jargon explainer: dotted term that opens a one-sentence definition (click, Escape closes).
 *   AnimatedNumber  counts up once, tweens with an accent wash on later changes.
 */
import { createContext, useContext, useEffect, useId, useRef, useState, useSyncExternalStore } from "react";
import { Lock } from "lucide-react";
import { etTime } from "./lib";
import { badgeView, parseTimestamp, tweenText } from "./ui-rules";

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

/**
 * Renders `text` (e.g. "$1,234.5", "12.3%") counting up from zero on mount and tweening from the previous value when it changes
 * (with a brief accent wash). Tabular digits, requestAnimationFrame only, instant under prefers-reduced-motion.
 */
export function AnimatedNumber({ text, duration = 650 }: { text: string; duration?: number }) {
  const reduced = useReducedMotion();
  const [track, setTrack] = useState({ text, from: "0", changes: 0 });
  if (track.text !== text) setTrack({ text, from: track.text, changes: track.changes + 1 });
  const [frame, setFrame] = useState<{ key: string; text: string } | null>(null);
  useEffect(() => {
    if (reduced) return;
    let id = 0;
    const start = performance.now();
    const length = track.changes ? duration * 0.65 : duration;
    const step = (now: number) => {
      const progress = Math.min(1, (now - start) / length);
      setFrame({ key: text, text: tweenText(track.from, text, progress) });
      if (progress < 1) id = requestAnimationFrame(step);
    };
    id = requestAnimationFrame(step);
    return () => cancelAnimationFrame(id);
  }, [text, track.from, track.changes, reduced, duration]);
  return <span key={track.changes} className="anim-num" data-tick={track.changes || undefined}>{!reduced && frame?.key === text ? frame.text : text}</span>;
}

export type PulseStatus = "fresh" | "aging" | "stale";

/** A status dot for freshness: fresh rings out softly, aging breathes, stale is a still red dot. Decorative unless `label` is given. */
export function LivePulse({ status = "fresh", label }: { status?: PulseStatus; label?: string }) {
  return <i className={`live-pulse live-pulse--${status}`} role={label ? "img" : undefined} aria-label={label} aria-hidden={label ? undefined : true} />;
}

/** A horizontally scrolling region with edge shadows on the sides that have more content; keyboard focusable. Wrap any wide table in it. */
export function ScrollTable({ children, label, className = "" }: { children: React.ReactNode; label: string; className?: string }) {
  // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex -- a focusable scroll region lets keyboard users scroll wide tables
  return <div className={`scroll-x ${className}`.trim()} role="region" aria-label={label} tabIndex={0}>{children}</div>;
}

export function LiveDot({ tone = "positive", pulse = true }: { tone?: Tone; pulse?: boolean }) {
  return <i className={`live-dot tone-${tone} ${pulse ? "pulse" : ""}`} aria-hidden="true" />;
}

export function Badge({ tone = "neutral", pulse = false, children }: { tone?: Tone; pulse?: boolean; children: React.ReactNode }) {
  return <span className={`badge tone-${tone}`}>{pulse && <LiveDot tone={tone} />}{children}</span>;
}

/** Signed change with a direction glyph so the sign never relies on color alone. */
export function Delta({ value, digits = 1, suffix = "", inverse = false }: { value: number; digits?: number; suffix?: string; /** Lower is better (scores, error): a fall is shown as good. */ inverse?: boolean }) {
  if (!Number.isFinite(value)) return <span className="delta">—</span>;
  const good = inverse ? value < 0 : value > 0;
  const tone = value === 0 ? "neutral" : good ? "positive" : "negative";
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

/** "Last run N minutes ago" that keeps ticking. `at` is an ISO (or golfprice folder-style) timestamp. */
export function FreshnessBadge({ at, label = "Last run", scale = "run", title, savedCopy = false }: { at: unknown; label?: string; scale?: "run" | "catalog"; title?: string; savedCopy?: boolean }) {
  const now = useNow(30_000);
  const parsed = parseTimestamp(at);
  const age = parsed === null ? null : now - parsed;
  const { tone, text } = badgeView(age, label, scale, savedCopy);
  return (
    <span className={`freshness-badge fresh-${tone}`} title={savedCopy ? `Saved copy: the data service could not be reached. ${title ?? ""}`.trim() : title ?? (parsed === null ? undefined : `${label}: ${etTime(new Date(parsed).toISOString())}`)}>
      <LivePulse status={tone === "live" || tone === "fresh" ? "fresh" : tone === "aging" ? "aging" : "stale"} />
      <span>{text}</span>
    </span>
  );
}


/* ------------------------------------------------------------------ design system v2 primitives */

/**
 * Eyebrow rule (one rule for every page): the eyebrow is the SECTION GROUP the page lives in (Week, Model, Operate, Review, Archive),
 * supplied by the shell through PageGroupContext, never written per page. Owner-only pages add a small lock chip beside it.
 * Event names, tours and ad-hoc labels do not belong in the eyebrow (the event is in the bar). The `eyebrow` prop is only a fallback
 * for a page rendered outside the shell (tests, previews).
 */
export type PageGroup = { group: string | null; ownerOnly: boolean };
export const PageGroupContext = createContext<PageGroup | null>(null);

export function Eyebrow({ fallback }: { fallback?: string }) {
  const ctx = useContext(PageGroupContext);
  const text = ctx?.group ?? fallback;
  if (!text && !ctx?.ownerOnly) return null;
  return (
    <span className="eyebrow-row">
      {text && <span className="eyebrow">{text}</span>}
      {ctx?.ownerOnly && <span className="owner-chip" title="Only you can see this page"><Lock size={11} aria-hidden="true" />Owner only</span>}
    </span>
  );
}

export function PageHeader({ eyebrow, title, purpose, actions }: { /** Fallback only; the shell's group name wins (see Eyebrow). */ eyebrow?: string; title: string; purpose?: string; actions?: React.ReactNode }) {
  return (
    <div className="page-intro">
      <div>
        <Eyebrow fallback={eyebrow} />
        <h1>{title}</h1>
        {purpose && <p>{purpose}</p>}
      </div>
      {actions && <div className="page-controls">{actions}</div>}
    </div>
  );
}

/** An anchored section: rule + h2 (+ purpose line, actions), no card. Pair with OnThisPage by id. */
export function Section({ id, title, purpose, actions, children, className = "" }: { id: string; title: string; purpose?: string; actions?: React.ReactNode; children: React.ReactNode; className?: string }) {
  const headingId = `${id}-h`;
  return (
    <section id={id} className={`ds-section ${className}`.trim()} aria-labelledby={headingId}>
      <header>
        <div><h2 id={headingId}>{title}</h2>{purpose && <p>{purpose}</p>}</div>
        {actions && <div className="section-actions">{actions}</div>}
      </header>
      {children}
    </section>
  );
}

/**
 * Sticky in-page navigation for long pages; scrolls horizontally on phones. Scroll-position spy: the active item is the last section whose top
 * has passed the sticky header, the first section while the page is at the top, and the last one at the very bottom. The active item is
 * kept in view inside the strip.
 */
export function OnThisPage({ sections, label = "On this page", value, onSelect }: { sections: Array<{ id: string; label: string }>; label?: string; /** Tab mode: the selected id (controlled). Omit both for anchor mode. */ value?: string; onSelect?: (id: string) => void }) {
  const tabs = onSelect !== undefined;
  const [current, setCurrent] = useState(sections[0]?.id ?? "");
  const navRef = useRef<HTMLElement>(null);
  const key = sections.map((s) => s.id).join("|");
  useEffect(() => {
    if (tabs) return;
    const ids = key.split("|");
    let frame = 0;
    const update = () => {
      frame = 0;
      const atBottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4;
      let active = ids[0];
      if (window.scrollY > 8) for (const id of ids) { const el = document.getElementById(id); if (el && el.getBoundingClientRect().top <= 140) active = id; }
      if (atBottom && window.scrollY > 8) active = ids[ids.length - 1];
      setCurrent(active);
    };
    const onScroll = () => { if (!frame) frame = requestAnimationFrame(update); };
    update();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => { window.removeEventListener("scroll", onScroll); window.removeEventListener("resize", onScroll); if (frame) cancelAnimationFrame(frame); };
  }, [key, tabs]);
  const active = tabs ? value : current;
  useEffect(() => {
    const strip = navRef.current;
    const link = strip?.querySelector<HTMLElement>('[aria-current="true"], [aria-selected="true"]');
    if (!strip || !link || strip.scrollWidth <= strip.clientWidth) return;
    strip.scrollTo({ left: link.offsetLeft - (strip.clientWidth - link.offsetWidth) / 2, behavior: "auto" });
  }, [active]);
  useEffect(() => {
    // Scroll cue: mark which edges of the strip are clipped so CSS can fade them (phone only, where the strip scrolls).
    const strip = navRef.current;
    if (!strip) return;
    const mark = () => {
      const more = strip.scrollWidth - strip.clientWidth > 2;
      const start = more && strip.scrollLeft > 2, end = more && strip.scrollLeft + strip.clientWidth < strip.scrollWidth - 2;
      if (start && end) strip.dataset.clip = "both"; else if (start) strip.dataset.clip = "start"; else if (end) strip.dataset.clip = "end"; else delete strip.dataset.clip;
    };
    mark();
    strip.addEventListener("scroll", mark, { passive: true });
    window.addEventListener("resize", mark);
    return () => { strip.removeEventListener("scroll", mark); window.removeEventListener("resize", mark); };
  }, [key]);
  if (tabs) {
    // Tab mode: the same strip, as a WAI-ARIA tablist (roving tabindex, arrow keys, Home/End). The page owns the panels.
    const move = (event: React.KeyboardEvent<HTMLElement>) => {
      const ids = sections.map((s) => s.id);
      const at = Math.max(0, ids.indexOf(active ?? ""));
      const next = event.key === "ArrowRight" ? at + 1 : event.key === "ArrowLeft" ? at - 1 : event.key === "Home" ? 0 : event.key === "End" ? ids.length - 1 : null;
      if (next === null) return;
      event.preventDefault();
      const id = ids[(next + ids.length) % ids.length];
      onSelect(id);
      requestAnimationFrame(() => navRef.current?.querySelector<HTMLElement>(`[data-section="${id}"]`)?.focus());
    };
    return (
      <nav className="on-this-page" aria-label={label} ref={navRef}>
        {/* eslint-disable-next-line jsx-a11y/interactive-supports-focus -- the tab buttons inside carry the roving tabindex */}
        <div role="tablist" aria-label={label} className="otp-tablist" onKeyDown={move}>
          {sections.map((s) => <button key={s.id} type="button" role="tab" data-section={s.id} aria-selected={active === s.id} tabIndex={active === s.id ? 0 : -1} onClick={() => onSelect(s.id)}>{s.label}</button>)}
        </div>
      </nav>
    );
  }
  return (
    <nav className="on-this-page" aria-label={label} ref={navRef}>
      {sections.map((s) => <a key={s.id} href={`#${s.id}`} aria-current={current === s.id ? "true" : undefined}>{s.label}</a>)}
    </nav>
  );
}

/** KPI tile. `value` is pre-formatted text; `unit` sits beside it; `delta` is signed (use inverse when lower is better). */
export function Stat({ label, value, unit, delta, deltaSuffix = "", deltaDigits = 1, inverse = false, note, tone = "neutral", spark, hint }: {
  label: string; value: string; unit?: string; delta?: number; deltaSuffix?: string; deltaDigits?: number; inverse?: boolean; note?: string;
  tone?: "neutral" | "positive" | "negative" | "accent"; spark?: number[]; hint?: string;
}) {
  const toneClass = tone === "positive" ? "stat-pos" : tone === "negative" ? "stat-neg" : "";
  return (
    <div className={`stat ${toneClass}`.trim()}>
      <span className="stat-label" title={hint}>{label}</span>
      <div className="kpi-row">
        <strong className="stat-value"><AnimatedNumber text={value} />{unit && <small>{unit}</small>}</strong>
        {spark && spark.length > 1 && <Sparkline values={spark} tone={tone === "neutral" ? "accent" : tone} label={`${label} trend`} />}
      </div>
      {(note || delta !== undefined) && (
        <div className="stat-foot">
          {delta !== undefined && <Delta value={delta} suffix={deltaSuffix} digits={deltaDigits} inverse={inverse} />}
          {note && <span className="stat-note">{note}</span>}
        </div>
      )}
    </div>
  );
}

/** Small tag for a stage, status or category. Meaning is in the words; colour only reinforces it. */
export function Chip({ tone = "neutral", children, title }: { tone?: Tone; children: React.ReactNode; title?: string }) {
  return <span className={`tag tone-${tone}`} title={title}>{children}</span>;
}

/**
 * Model vs book on a shared zero. `model` and `book` are probabilities in percent (24 = 24%). The bar spans `range` points each side of zero
 * (default 10) and clips beyond it. A positive gap (model above book) is favourable for backing; the label states the numbers, the sign and an arrow.
 */
export function GapRail({ model, book, range = 10, digits = 1, unit = "%", modelLabel = "model", bookLabel = "book" }: { model: number; book: number; range?: number; digits?: number; unit?: string; modelLabel?: string; bookLabel?: string }) {
  if (!Number.isFinite(model) || !Number.isFinite(book)) return <span className="num">—</span>;
  const raw = model - book;
  const gap = Number(Math.abs(raw).toFixed(digits)) === 0 ? 0 : raw;
  const half = Math.min(50, (Math.abs(gap) / range) * 50);
  const side = gap > 0 ? "pos" : gap < 0 ? "neg" : "";
  const signed = `${gap > 0 ? "+" : gap < 0 ? "−" : ""}${Math.abs(gap).toFixed(digits)}`;
  const text = `${modelLabel} ${model.toFixed(digits)}${unit} vs ${bookLabel} ${book.toFixed(digits)}${unit} = ${signed} pts`;
  return (
    <div className="gap-rail" role="img" aria-label={text}>
      <div className="gap-rail-track" aria-hidden="true">{side && <i className={`gap-rail-bar ${side}`} style={{ width: `${half}%` }} />}</div>
      <div className="gap-rail-label" aria-hidden="true">
        <span>{modelLabel} <b className="gap-model">{model.toFixed(digits)}{unit}</b></span>
        <span>vs {bookLabel} <b className="gap-book">{book.toFixed(digits)}{unit}</b></span>
        <span className={`gap-pts ${side}`}>{gap > 0 ? "▲ " : gap < 0 ? "▼ " : ""}{signed} pts</span>
      </div>
    </div>
  );
}

/** Exclusive choice. Buttons carry aria-pressed; the group is labelled. */
export function SegmentedControl<T extends string>({ value, options, onChange, label }: { value: T; options: Array<{ value: T; label: string }>; onChange: (value: T) => void; label: string }) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map((option) => (
        <button key={option.value} type="button" className={value === option.value ? "active" : ""} aria-pressed={value === option.value} onClick={() => onChange(option.value)}>{option.label}</button>
      ))}
    </div>
  );
}

/** Compact, honest note for missing or unavailable content. Say why, and when it will be checked again. Sentence case. */
export function EmptyState({ title, detail, nextCheck, tone, onRetry }: { title: string; detail?: string; /** e.g. "Next check: Thursday 6:00 ET" */ nextCheck?: string; tone?: "warning"; onRetry?: () => void }) {
  return (
    <div className={`empty-state${tone ? ` tone-${tone}` : ""}`} role={onRetry ? "alert" : undefined}>
      <div className="empty-body">
        <strong>{title}</strong>
        {detail && <span>{detail}</span>}
        {nextCheck && <span className="empty-next">{nextCheck}</span>}
        {onRetry && <button type="button" className="ex-link" onClick={onRetry}>Try again</button>}
      </div>
    </div>
  );
}

/** Jargon explainer: the term stays in the sentence, a click shows a one-sentence definition. Escape or an outside click closes it. */
export function InfoTip({ children, definition }: { children: React.ReactNode; definition: string }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (event: Event) => { if (event instanceof KeyboardEvent ? event.key === "Escape" : !ref.current?.contains(event.target as Node)) setOpen(false); };
    document.addEventListener("keydown", close);
    document.addEventListener("pointerdown", close);
    return () => { document.removeEventListener("keydown", close); document.removeEventListener("pointerdown", close); };
  }, [open]);
  return (
    <span className="info-tip" ref={ref}>
      <button type="button" aria-expanded={open} aria-describedby={open ? id : undefined} onClick={() => setOpen((o) => !o)}>{children}<span className="info-i" aria-hidden="true">i</span></button>
      {open && <span className="info-tip-pop" role="note" id={id}>{definition}</span>}
    </span>
  );
}
