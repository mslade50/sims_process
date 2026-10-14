// Pure rules for scripts/visual-check.mjs: geometry, colour/contrast math, allowlists, severity. No DOM, no browser.
// Tested by tests/visual-rules.test.mjs.

/* ------------------------------------------------------------------ geometry */
/** Box = {x, y, w, h} in CSS px. */
export const right = (b) => b.x + b.w;
export const bottom = (b) => b.y + b.h;
export function intersection(a, b) {
  const x = Math.max(a.x, b.x), y = Math.max(a.y, b.y);
  const w = Math.min(right(a), right(b)) - x, h = Math.min(bottom(a), bottom(b)) - y;
  return w > 0 && h > 0 ? { x, y, w, h } : null;
}
export const area = (b) => Math.max(0, b.w) * Math.max(0, b.h);
/** Overlap that a reader would notice: not a hairline touch, and a real share of the smaller box. */
export function meaningfulOverlap(a, b, { minW = 2, minH = 3, minRatio = 0.25 } = {}) {
  const i = intersection(a, b);
  if (!i || i.w < minW || i.h < minH) return null;
  const smaller = Math.min(area(a), area(b));
  if (smaller <= 0) return null;
  const ratio = area(i) / smaller;
  return ratio >= minRatio ? { box: i, ratio } : null;
}
/** Visible part of `box` after clipping by ancestor clip boxes; null when fully clipped away. */
export function clipTo(box, clips) {
  let cur = box;
  for (const c of clips) { cur = intersection(cur, c); if (!cur) return null; }
  return cur;
}

/* ------------------------------------------------------------------ colour */
const clamp01 = (v) => Math.min(1, Math.max(0, v));
const srgbEncode = (c) => (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);
const num = (s) => (s.endsWith("%") ? parseFloat(s) / 100 : s === "none" ? 0 : parseFloat(s));

function oklabToRgb(L, a, b) {
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b, m_ = L - 0.1055613458 * a - 0.0638541728 * b, s_ = L - 0.0894841775 * a - 1.2914855480 * b;
  const l = l_ ** 3, m = m_ ** 3, s = s_ ** 3;
  const r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
  const g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
  const bl = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;
  return [clamp01(srgbEncode(r)) * 255, clamp01(srgbEncode(g)) * 255, clamp01(srgbEncode(bl)) * 255];
}

/** Parse a computed CSS colour. Returns {r,g,b,a} (0-255, a 0-1) or null. Handles rgb(), rgba(), #hex, color(srgb ...), oklab(), oklch(), transparent. */
export function parseColor(input) {
  if (!input) return null;
  const s = String(input).trim().toLowerCase();
  if (s === "transparent") return { r: 0, g: 0, b: 0, a: 0 };
  let m = s.match(/^#([0-9a-f]{3,8})$/);
  if (m) {
    let h = m[1];
    if (h.length <= 4) h = [...h].map((c) => c + c).join("");
    if (h.length !== 6 && h.length !== 8) return null;
    return { r: parseInt(h.slice(0, 2), 16), g: parseInt(h.slice(2, 4), 16), b: parseInt(h.slice(4, 6), 16), a: h.length === 8 ? parseInt(h.slice(6, 8), 16) / 255 : 1 };
  }
  m = s.match(/^(rgba?|color|oklab|oklch)\((.*)\)$/);
  if (!m) return null;
  const fn = m[1];
  const [body, alphaRaw] = m[2].split("/").map((x) => x.trim());
  const parts = body.replace(/,/g, " ").split(/\s+/).filter(Boolean);
  let alpha = alphaRaw !== undefined ? num(alphaRaw) : 1;
  if (fn === "rgb" || fn === "rgba") {
    if (parts.length === 4 && alphaRaw === undefined) alpha = num(parts[3]);
    const ch = parts.slice(0, 3).map((p) => (p.endsWith("%") ? (parseFloat(p) / 100) * 255 : parseFloat(p)));
    if (ch.some(Number.isNaN)) return null;
    return { r: ch[0], g: ch[1], b: ch[2], a: clamp01(alpha) };
  }
  if (fn === "color") {
    if (parts[0] !== "srgb" && parts[0] !== "srgb-linear") return null;
    let ch = parts.slice(1, 4).map(num);
    if (ch.some(Number.isNaN)) return null;
    if (parts[0] === "srgb-linear") ch = ch.map(srgbEncode);
    return { r: clamp01(ch[0]) * 255, g: clamp01(ch[1]) * 255, b: clamp01(ch[2]) * 255, a: clamp01(alpha) };
  }
  const p = parts.map(num);
  if (p.some(Number.isNaN) || p.length < 3) return null;
  if (fn === "oklab") { const [r, g, b] = oklabToRgb(p[0], p[1], p[2]); return { r, g, b, a: clamp01(alpha) }; }
  const hue = (p[2] * Math.PI) / 180; // oklch
  const [r, g, b] = oklabToRgb(p[0], p[1] * Math.cos(hue), p[1] * Math.sin(hue));
  return { r, g, b, a: clamp01(alpha) };
}

/** Source-over composite of fg (any alpha) on an opaque bg. */
export function composite(fg, bg) {
  const a = fg.a;
  return { r: fg.r * a + bg.r * (1 - a), g: fg.g * a + bg.g * (1 - a), b: fg.b * a + bg.b * (1 - a), a: 1 };
}
export function relativeLuminance({ r, g, b }) {
  const f = (v) => { const c = v / 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
}
export function contrastRatio(c1, c2) {
  const l1 = relativeLuminance(c1), l2 = relativeLuminance(c2);
  const hi = Math.max(l1, l2), lo = Math.min(l1, l2);
  return (hi + 0.05) / (lo + 0.05);
}
/** WCAG large text: >= 24px, or >= 18.66px and bold (700+). */
export const isLargeText = (fontPx, weight) => fontPx >= 24 || (fontPx >= 18.66 && Number(weight) >= 700);
export const requiredContrast = (fontPx, weight) => (isLargeText(fontPx, weight) ? 3 : 4.5);

/**
 * Effective background from a chain of computed background colours, nearest ancestor first. Translucent layers are composited
 * over whatever lies beneath; the page colour is the base. Returns null when any layer is an image/gradient ("unknown").
 */
export function effectiveBackground(chain, base = { r: 255, g: 255, b: 255, a: 1 }) {
  const layers = [];
  for (const entry of chain) {
    if (entry && entry.unknown) return null;
    const c = typeof entry === "string" ? parseColor(entry) : entry;
    if (!c || c.a === 0) continue;
    layers.push(c);
    if (c.a >= 1) break;
  }
  let bg = layers.length && layers[layers.length - 1].a >= 1 ? layers.pop() : base;
  for (let i = layers.length - 1; i >= 0; i--) bg = composite(layers[i], bg);
  return bg;
}

/** Contrast verdict for one text item: {color, chain, fontPx, weight, opacity}. null when it cannot be judged. */
export function evaluateContrast({ color, chain, fontPx, weight, opacity = 1 }, base) {
  const fg = parseColor(color);
  const bg = effectiveBackground(chain, base);
  if (!fg || !bg) return null;
  const fgEff = composite({ ...fg, a: fg.a * opacity }, bg);
  const ratio = contrastRatio(fgEff, bg);
  const need = requiredContrast(fontPx, weight);
  return { ratio, need, pass: ratio >= need };
}

/* ------------------------------------------------------------------ allowlists */
/** Elements deliberately layered on other text (badges, tooltips, chart overlays). Matched against "tag.class1.class2" descriptors. */
export const OVERLAP_ALLOW = [
  /\brecharts-tooltip/, /\brecharts-active/, /\bsr-only\b/, /\bvisually-hidden\b/, /\btooltip\b/, /\bbadge-float\b/, /\bpopover\b/, /\bsticky-corner\b/,
];
export const CONTRAST_ALLOW = [/\bsr-only\b/, /\bvisually-hidden\b/];
/** API paths where a 404 means "object not published yet": listed separately instead of failing. */
export const KNOWN_404 = [/^\/api\/data\/.+/, /^\/api\/golfprice\/.+/, /^\/golfprice\/.+/, /^\/api\/(weather-)?overrides?/, /^\/api\/jobs/];

export const matchesAny = (descriptor, patterns) => patterns.some((p) => p.test(descriptor));

/* ------------------------------------------------------------------ network / console */
export function classifyRequest({ url, status, base }) {
  let u; try { u = new URL(url); } catch { return "ignore"; }
  if (u.origin !== new URL(base).origin) return "ignore";
  if (status === 0) return "failed";
  if ((status >= 200 && status < 300) || status === 304) return "ok";
  if (status === 404 && matchesAny(u.pathname, KNOWN_404)) return "known404";
  return "failed";
}
export const isIgnorableConsole = (text) => /favicon|Download the React DevTools|\[vite\] (connecting|connected)/i.test(text);

/* ------------------------------------------------------------------ tap targets, page height */
export function tapTargetIssues(targets, min = 40) {
  const small = targets.filter((t) => t.w < min || t.h < min);
  small.sort((a, b) => Math.min(a.w, a.h) - Math.min(b.w, b.h));
  return { total: targets.length, count: small.length, worst: small.slice(0, 8) };
}
export const pageScreens = (docHeight, viewportHeight) => docHeight / viewportHeight;

/* ------------------------------------------------------------------ severity */
/** Check ids -> severity. "error" fails the run; "warn" and "info" are reported only. */
export const SEVERITY = {
  "page-overflow": "error",
  "offscreen-element": "warn",
  "text-overlap": "error",
  "clipped-content": "error",
  "truncated-ellipsis": "info",
  "contrast": "error",
  "tap-target": "warn",
  "page-height": "warn",
  "title-wrap": "warn",
  "console-error": "error",
  "failed-request": "error",
  "known-404": "info",
  "long-task": "warn",
  "load-failure": "error",
};
export const severityOf = (check) => SEVERITY[check] ?? "warn";
export const PHONE_MAX_SCREENS = 12;
export const LONG_TASK_MS = 200;

export function summarize(cases) {
  const counts = { error: 0, warn: 0, info: 0 };
  for (const c of cases) for (const i of c.issues) counts[i.severity] = (counts[i.severity] ?? 0) + 1;
  return { cases: cases.length, ...counts, failed: counts.error > 0 };
}
