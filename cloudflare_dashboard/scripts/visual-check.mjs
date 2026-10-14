// Visual-regression and layout-health harness for the dashboard. Zero dependencies: drives headless Chrome/Edge over the DevTools Protocol
// with Node's built-in WebSocket. Opens every route x viewport (phone 375x812 touch, desktop 1280x900) x theme (dark, light), runs the
// layout checks in scripts/visual-rules.mjs, writes dist-visual/<timestamp>/{report.json,report.html,shots/*.png}.
//
//   node scripts/visual-check.mjs [--base http://localhost:3111] [--routes this-week,run] [--viewports phone,desktop] [--themes dark,light]
//                                 [--jobs 3] [--browser <path-to-chrome-or-edge>] [--out <dir>] [--no-shots] [--keep-going]
// Exit code: 0 no "error" issues, 1 at least one error issue, 2 setup failure (server or browser unreachable).
import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  CONTRAST_ALLOW, LONG_TASK_MS, OVERLAP_ALLOW, PHONE_MAX_SCREENS, classifyRequest, evaluateContrast, isIgnorableConsole, matchesAny,
  meaningfulOverlap, parseColor, severityOf, summarize, tapTargetIssues,
} from "./visual-rules.mjs";

const ROOT = resolve(fileURLToPath(new URL("..", import.meta.url)));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* ------------------------------------------------------------------ args */
function parseArgs(argv) {
  const o = { base: "http://localhost:3111", routes: null, viewports: ["phone", "desktop"], themes: ["dark", "light"], jobs: 3, browser: null, out: null, shots: true };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i], next = () => argv[++i];
    if (a === "--base") o.base = next();
    else if (a === "--routes") o.routes = next().split(",").map((s) => s.trim().replace(/^\//, ""));
    else if (a === "--viewports") o.viewports = next().split(",");
    else if (a === "--themes") o.themes = next().split(",");
    else if (a === "--jobs") o.jobs = Math.max(1, Number(next()) || 1);
    else if (a === "--browser") o.browser = next();
    else if (a === "--out") o.out = resolve(next());
    else if (a === "--no-shots") o.shots = false;
    else if (a === "--help" || a === "-h") { console.log(readFileSync(fileURLToPath(import.meta.url), "utf8").split("\n").slice(0, 9).join("\n")); process.exit(0); }
    else { console.error(`Unknown argument ${a}`); process.exit(2); }
  }
  o.base = o.base.replace(/\/+$/, "");
  return o;
}

const VIEWPORTS = {
  phone: { name: "phone", width: 375, height: 812, mobile: true, touch: true },
  desktop: { name: "desktop", width: 1280, height: 900, mobile: false, touch: false },
};
const MAX_SHOT_HEIGHT = 16000; // Chrome's texture limit is 16384

/** Routes come from the app's own NAVIGATION table (plus toggle targets), so a new nav entry is covered automatically. */
function loadRoutes() {
  const src = readFileSync(join(ROOT, "app", "shell-rules.ts"), "utf8");
  const block = src.slice(src.indexOf("export const NAVIGATION"), src.indexOf("export const NAV_ITEMS"));
  const hrefs = [...block.matchAll(/href:\s*"(\/[a-z0-9-]*)"/g)].map((m) => m[1]);
  return [...new Set(hrefs)];
}
function themeKey() {
  const src = readFileSync(join(ROOT, "app", "ui-rules.ts"), "utf8");
  return src.match(/THEME_STORAGE_KEY\s*=\s*"([^"]+)"/)?.[1] ?? "golf-dashboard-theme";
}

/* ------------------------------------------------------------------ browser + CDP */
function findBrowser(explicit) {
  const candidates = [explicit, process.env.CHROME_PATH, "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe", "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe", "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"].filter(Boolean);
  return candidates.find((p) => existsSync(p));
}

async function launchBrowser(path) {
  const userDir = mkdtempSync(join(tmpdir(), "visual-check-"));
  const child = spawn(path, ["--headless=new", "--remote-debugging-port=0", `--user-data-dir=${userDir}`, "--no-first-run", "--no-default-browser-check",
    "--disable-extensions", "--disable-background-timer-throttling", "--disable-renderer-backgrounding", "--disable-backgrounding-occluded-windows", "--force-color-profile=srgb", "about:blank"],
  { stdio: "ignore", windowsHide: true });
  let exited = false;
  child.on("exit", () => { exited = true; });
  const portFile = join(userDir, "DevToolsActivePort");
  for (let i = 0; i < 150 && !existsSync(portFile) && !exited; i++) await sleep(100);
  if (!existsSync(portFile)) { child.kill(); throw new Error("Browser did not publish DevToolsActivePort"); }
  await sleep(100);
  const [port, wsPath] = readFileSync(portFile, "utf8").trim().split(/\r?\n/);
  const close = () => {
    try { if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(child.pid), "/T", "/F"], { stdio: "ignore" }); else child.kill("SIGKILL"); } catch { /* already gone */ }
    for (let i = 0; i < 5; i++) { try { rmSync(userDir, { recursive: true, force: true }); break; } catch { Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 200); } }
  };
  return { wsUrl: `ws://127.0.0.1:${port}${wsPath}`, close };
}

class Cdp {
  constructor(ws) { this.ws = ws; this.id = 0; this.pending = new Map(); this.handlers = new Map(); ws.onmessage = (ev) => this.#onMessage(ev); }
  static async connect(url) {
    const ws = new WebSocket(url);
    await new Promise((res, rej) => { ws.onopen = res; ws.onerror = () => rej(new Error("DevTools WebSocket failed to open")); });
    return new Cdp(ws);
  }
  #onMessage(ev) {
    const msg = JSON.parse(typeof ev.data === "string" ? ev.data : Buffer.from(ev.data).toString());
    if (msg.id) {
      const p = this.pending.get(msg.id); if (!p) return;
      this.pending.delete(msg.id);
      if (msg.error) p.rej(new Error(`${p.method}: ${msg.error.message}`)); else p.res(msg.result);
    } else if (msg.method) this.handlers.get(msg.sessionId ?? "")?.(msg.method, msg.params);
  }
  send(method, params = {}, sessionId) {
    const id = ++this.id;
    return new Promise((res, rej) => {
      const timer = setTimeout(() => { this.pending.delete(id); rej(new Error(`${method}: timed out`)); }, 60000);
      this.pending.set(id, { method, res: (v) => { clearTimeout(timer); res(v); }, rej: (e) => { clearTimeout(timer); rej(e); } });
      this.ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
    });
  }
  on(sessionId, fn) { this.handlers.set(sessionId, fn); }
  off(sessionId) { this.handlers.delete(sessionId); }
}

/* ------------------------------------------------------------------ in-page collector (serialised into the page; must be self-contained) */
function collect(opts) {
  const vw = innerWidth, vh = innerHeight, de = document.documentElement;
  const styles = new Map();
  const cs = (el) => { let s = styles.get(el); if (!s) { s = getComputedStyle(el); styles.set(el, s); } return s; };
  const box = (r) => ({ x: Math.round((r.left + scrollX) * 10) / 10, y: Math.round((r.top + scrollY) * 10) / 10, w: Math.round(r.width * 10) / 10, h: Math.round(r.height * 10) / 10 });
  const part = (e) => { let s = e.localName; if (e.id) return `${s}#${e.id}`; const c = [...e.classList].slice(0, 3).join("."); return c ? `${s}.${c}` : s; };
  const sel = (el) => { const out = []; let e = el; for (let i = 0; e && e.nodeType === 1 && i < 3; i++, e = e.parentElement) { out.unshift(part(e)); if (e.id) break; } return out.join(" > "); };
  const ancClasses = (el) => { let s = ""; for (let e = el; e && s.length < 400; e = e.parentElement) s += ` ${e.getAttribute("class") || ""}`; return s; };
  const indexPath = (el) => { const p = []; for (let e = el; e && e !== document.body; e = e.parentElement) p.unshift([...e.parentElement.children].indexOf(e)); return p.join("."); };
  const inClosedDetails = (el) => { for (let d = el.closest("details"); d; d = d.parentElement && d.parentElement.closest("details")) { if (d.open) continue; const sm = el.closest("summary"); if (!(sm && sm.parentElement === d)) return true; } return false; };
  const rotated = (el) => { if (el instanceof SVGElement) { const m = el.getScreenCTM && el.getScreenCTM(); if (m && (Math.abs(m.b) > 0.01 || Math.abs(m.c) > 0.01)) return true; } for (let e = el; e && e !== document.body; e = e.parentElement) { const t = cs(e).transform; if (t && t !== "none") { const m = t.match(/matrix(([^)]+))/); if (m) { const v = m[1].split(",").map(Number); if (Math.abs(v[1]) > 0.01 || Math.abs(v[2]) > 0.01) return true; } } } return false; };
  const decorative = (el) => !!el.closest('[aria-hidden="true"],[inert],[hidden]') || inClosedDetails(el);
  const offscreenX = (b) => b.x + b.w <= 0 || b.x >= vw;
  const scrollX_ = (e) => { const s = cs(e); return s.overflowX !== "visible" || s.overflowY !== "visible"; };
  const inter = (a, b) => { const x = Math.max(a.x, b.x), y = Math.max(a.y, b.y), w = Math.min(a.x + a.w, b.x + b.w) - x, h = Math.min(a.y + a.h, b.y + b.h) - y; return w > 0 && h > 0 ? { x, y, w, h } : null; };
  /** Boxes of ancestors that clip this element (overflow != visible), honouring absolute/fixed escape from non-positioned ancestors. */
  const clipsFor = (el) => {
    const out = []; const own = cs(el).position;
    if (own === "fixed") return out;
    let abs = own === "absolute";
    for (let e = el.parentElement; e && e !== document.body && e !== de; e = e.parentElement) {
      const s = cs(e);
      if (abs && s.position === "static") continue;
      abs = false;
      if (scrollX_(e)) out.push(box(e.getBoundingClientRect()));
      if (s.position === "absolute") abs = true; else if (s.position === "fixed") break;
    }
    return out;
  };
  const visibleStyle = (el) => { const s = cs(el); return s.display !== "none" && s.visibility !== "hidden"; };

  /* ---- text items */
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const owners = new Map();
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    if (!n.nodeValue.trim()) continue;
    const p = n.parentElement;
    if (!p || /^(SCRIPT|STYLE|NOSCRIPT|TEXTAREA|OPTION)$/.test(p.tagName)) continue;
    if (!owners.has(p)) owners.set(p, []);
    owners.get(p).push(n);
  }
  const items = [];
  for (const [el, nodes] of owners) {
    if (!visibleStyle(el) || decorative(el)) continue;
    let opacity = 1; for (let e = el; e; e = e.parentElement) opacity *= parseFloat(cs(e).opacity);
    if (opacity < 0.05) continue;
    const clips = clipsFor(el);
    const rects = [];
    for (const n of nodes) {
      const r = document.createRange(); r.selectNodeContents(n);
      for (const q of r.getClientRects()) {
        if (q.width < 1 || q.height < 1) continue;
        let b = box(q); for (const c of clips) { b = b && inter(b, c); }
        if (b && !offscreenX(b)) rects.push(b);
      }
    }
    if (!rects.length) continue;
    const s = cs(el);
    const svg = el instanceof SVGElement;
    let fontPx = parseFloat(s.fontSize);
    if (svg) { const m = el.getScreenCTM?.(); if (m) fontPx *= Math.abs(m.a) || 1; }
    const chain = []; let unknown = false;
    for (let e = el; e && !unknown; e = e.parentElement) {
      const t = cs(e);
      if (t.backgroundImage !== "none") { chain.push({ unknown: true }); unknown = true; break; }
      chain.push(t.backgroundColor);
      const bg = t.backgroundColor;
      if (!bg.startsWith("rgba") && !bg.includes("/") && bg !== "transparent") break;
    }
    items.push({ sel: sel(el), cls: ancClasses(el), path: indexPath(el), rects, color: svg ? s.fill : s.color, fontPx, weight: s.fontWeight, opacity, chain,
      txt: nodes[0].nodeValue.trim().slice(0, 40), disabled: !!el.closest(":disabled,[aria-disabled='true']"), rotated: rotated(el), rect0: rects[0] });
  }

  /* ---- clipped content */
  const clipped = [], ellipsis = [];
  const all = document.body.querySelectorAll("*");
  for (const el of all) {
    const s = cs(el);
    if (s.overflowX !== "hidden" && s.overflowX !== "clip") continue;
    if (el.clientWidth < 4 || el.scrollWidth <= el.clientWidth + 1) continue;
    if (!visibleStyle(el) || decorative(el)) continue;
    const b = box(el.getBoundingClientRect());
    if (b.w < 4 || offscreenX(b) || clipsFor(el).some((c) => !inter(b, c))) continue;
    if (!el.textContent.trim() && !el.querySelector("img,svg,canvas,table")) continue;
    const row = { sel: sel(el), cls: ancClasses(el), box: b, scrollWidth: el.scrollWidth, clientWidth: el.clientWidth, hasTable: !!el.querySelector("table"), txt: el.textContent.trim().slice(0, 40) };
    (s.textOverflow === "ellipsis" ? ellipsis : clipped).push(row);
  }

  /* ---- elements past the viewport edge that no scroll container explains */
  const offscreen = [];
  for (const el of all) {
    if (el.closest("svg") && el.localName !== "svg") continue;
    const s = cs(el); if (s.display === "none" || s.visibility === "hidden" || s.position === "fixed") continue;
    const r = el.getBoundingClientRect(); if (r.width < 1 || r.height < 1) continue;
    if (r.right <= vw + 1 && r.left >= -1) continue;
    if (decorative(el) || clipsFor(el).length) continue;
    let fixedAnc = false; for (let e = el.parentElement; e; e = e.parentElement) if (cs(e).position === "fixed") { fixedAnc = true; break; }
    if (fixedAnc) continue;
    offscreen.push({ sel: sel(el), cls: ancClasses(el), box: box(r), over: Math.round(Math.max(r.right - vw, -r.left)), parentOver: !!(el.parentElement && el.parentElement.getBoundingClientRect().right > vw + 1) });
  }

  /* ---- tap targets (phone) */
  const taps = [];
  if (opts.phone) {
    for (const el of document.querySelectorAll("a[href],button,[role=button],[role=tab],[role=switch],summary,select,textarea,input:not([type=hidden])")) {
      const s = cs(el); if (s.display === "none" || s.visibility === "hidden" || decorative(el) || el.disabled) continue;
      if (s.display === "inline" && el.localName === "a") continue; // inline links inside prose are exempt
      const r = el.getBoundingClientRect(); if (r.width < 1 || r.height < 1 || r.right <= 0 || r.left >= vw) continue;
      if (clipsFor(el).some((c) => !inter(box(r), c))) continue;
      taps.push({ w: Math.round(r.width), h: Math.round(r.height), sel: sel(el), box: box(r), txt: (el.getAttribute("aria-label") || el.textContent || "").trim().slice(0, 30) });
    }
  }

  /* ---- header title wrap */
  let titleLines = 0, titleSel = "", titleBox = null;
  const title = document.querySelector(".topbar-title span") || document.querySelector(".topbar-title");
  if (title) {
    const r = document.createRange(); r.selectNodeContents(title);
    titleLines = new Set([...r.getClientRects()].filter((q) => q.width > 1).map((q) => Math.round(q.top / 4))).size;
    titleSel = sel(title); titleBox = box(title.getBoundingClientRect());
  }

  const docHeight = Math.max(de.scrollHeight, document.body.scrollHeight);
  return {
    vw, vh, docHeight, scrollWidth: Math.max(de.scrollWidth, document.body.scrollWidth), clientWidth: de.clientWidth,
    theme: de.dataset.theme || null, pageBg: cs(document.body).backgroundColor, htmlBg: cs(de).backgroundColor,
    items, clipped, ellipsis, offscreen, taps, titleLines, titleSel, titleBox,
    longTasks: window.__vcLongTasks || [], title: document.title,
  };
}

const INIT_SCRIPT = (key, theme) => `
try { localStorage.setItem(${JSON.stringify(key)}, ${JSON.stringify(theme)}); } catch (e) {}
window.__vcLongTasks = [];
try { new PerformanceObserver((l) => { for (const e of l.getEntries()) window.__vcLongTasks.push(Math.round(e.duration)); }).observe({ type: "longtask", buffered: true }); } catch (e) {}
`;

/* ------------------------------------------------------------------ one case */
async function runCase(cdp, base, c, opts, shotsDir) {
  const out = { route: c.route, viewport: c.viewport.name, theme: c.theme, id: c.id, issues: [], stats: {}, shot: null, thumb: null, thumbSize: null };
  const add = (check, message, extra = {}) => out.issues.push({ check, severity: severityOf(check), message, ...extra });
  // own browser context per case: the app syncs theme across tabs through localStorage, so parallel cases must not share storage
  const { browserContextId } = await cdp.send("Target.createBrowserContext");
  const { targetId } = await cdp.send("Target.createTarget", { url: "about:blank", browserContextId });
  const { sessionId } = await cdp.send("Target.attachToTarget", { targetId, flatten: true });
  const S = (m, p) => cdp.send(m, p, sessionId);
  const consoleErrors = [], requests = new Map(), failures = [];
  let inflight = 0, lastNet = Date.now(), loaded = false, docStatus = null;
  cdp.on(sessionId, (method, p) => {
    if (method === "Runtime.consoleAPICalled" && p.type === "error") {
      const text = p.args.map((a) => a.value ?? a.description ?? "").join(" ");
      if (!isIgnorableConsole(text)) consoleErrors.push(text.slice(0, 300));
    } else if (method === "Runtime.exceptionThrown") {
      const d = p.exceptionDetails; consoleErrors.push(`Uncaught: ${(d.exception?.description || d.text || "").split("\n")[0].slice(0, 300)}`);
    } else if (method === "Log.entryAdded" && p.entry.level === "error" && p.entry.source !== "network") {
      if (!isIgnorableConsole(p.entry.text)) consoleErrors.push(`${p.entry.source}: ${p.entry.text.slice(0, 300)}`);
    } else if (method === "Network.requestWillBeSent") { inflight++; lastNet = Date.now(); requests.set(p.requestId, { url: p.request.url, type: p.type }); }
    else if (method === "Network.responseReceived") {
      const r = requests.get(p.requestId); if (r) { r.status = p.response.status; if (p.type === "Document" && !docStatus) docStatus = p.response.status; }
    } else if (method === "Network.loadingFinished") { inflight = Math.max(0, inflight - 1); lastNet = Date.now(); }
    else if (method === "Network.loadingFailed") {
      inflight = Math.max(0, inflight - 1); lastNet = Date.now();
      const r = requests.get(p.requestId);
      if (r && !p.canceled && p.errorText !== "net::ERR_ABORTED") failures.push({ url: r.url, status: 0, error: p.errorText });
    } else if (method === "Page.loadEventFired") loaded = true;
  });
  try {
    const v = c.viewport;
    await Promise.all(["Page.enable", "Runtime.enable", "Network.enable", "Log.enable"].map((m) => S(m)));
    await S("Network.setCacheDisabled", { cacheDisabled: true });
    await S("Emulation.setDeviceMetricsOverride", { width: v.width, height: v.height, deviceScaleFactor: 1, mobile: v.mobile });
    if (v.touch) await S("Emulation.setTouchEmulationEnabled", { enabled: true, maxTouchPoints: 5 });
    await S("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: c.theme }, { name: "prefers-reduced-motion", value: "reduce" }] });
    await S("Page.addScriptToEvaluateOnNewDocument", { source: INIT_SCRIPT(opts.themeKey, c.theme) });
    const nav = await S("Page.navigate", { url: `${base}${c.route}` });
    if (nav.errorText) { add("load-failure", `Navigation failed: ${nav.errorText}`); return out; }
    for (let i = 0; i < 300 && !loaded; i++) await sleep(100);
    // settle: network quiet for 800 ms (lazy view chunks and data), fonts ready, then a scroll pass to wake lazy/observed content
    for (let i = 0; i < 300; i++) { if (inflight === 0 && Date.now() - lastNet > 800) break; await sleep(100); }
    await S("Runtime.evaluate", { expression: "document.fonts && document.fonts.ready.then(() => 1)", awaitPromise: true });
    await S("Runtime.evaluate", {
      awaitPromise: true,
      expression: `(async () => { const h = Math.max(document.documentElement.scrollHeight, document.body.scrollHeight); const step = innerHeight * 0.9;
        for (let y = 0, n = 0; y < h && n < 40; y += step, n++) { scrollTo(0, y); await new Promise((r) => setTimeout(r, 120)); } scrollTo(0, 0); await new Promise((r) => setTimeout(r, 700)); return 1; })()`,
    });
    for (let i = 0; i < 100; i++) { if (inflight === 0 && Date.now() - lastNet > 500) break; await sleep(100); }
    if (docStatus && docStatus >= 400) add("load-failure", `Document responded ${docStatus}`);

    const res = await S("Runtime.evaluate", { expression: `(${collect.toString()})(${JSON.stringify({ phone: v.name === "phone" })})`, returnByValue: true });
    if (res.exceptionDetails) { add("load-failure", `Collector threw: ${res.exceptionDetails.exception?.description ?? res.exceptionDetails.text}`); return out; }
    const d = res.result.value;
    analyse(d, c, add, out);

    for (const [, r] of requests) if (r.status !== undefined) { const k = classifyRequest({ url: r.url, status: r.status, base }); if (k === "failed") failures.push({ url: r.url, status: r.status }); else if (k === "known404") out.stats.known404 = [...(out.stats.known404 ?? []), r.url.replace(base, "")]; }
    const seen = new Set();
    for (const f of failures) { const k = `${f.status}${f.url}`; if (seen.has(k)) continue; seen.add(k); add("failed-request", `${f.status || f.error} ${f.url.replace(base, "")}`); }
    for (const u of [...new Set(out.stats.known404 ?? [])]) add("known-404", `404 (not published) ${u}`);
    [...new Set(consoleErrors)].slice(0, 10).forEach((t) => add("console-error", t));
    const slow = d.longTasks.filter((t) => t > LONG_TASK_MS);
    if (slow.length) add("long-task", `${slow.length} long task(s), max ${Math.max(...slow)} ms`);
    out.stats.longTasks = d.longTasks.length;

    if (opts.shots) {
      const H = Math.min(d.docHeight, MAX_SHOT_HEIGHT);
      const clip = { x: 0, y: 0, width: d.vw, height: H, scale: 1 };
      const png = await S("Page.captureScreenshot", { format: "png", captureBeyondViewport: true, clip });
      const file = `${c.id}.png`;
      writeFileSync(join(shotsDir, file), Buffer.from(png.data, "base64"));
      out.shot = `shots/${file}`;
      const jpg = await S("Page.captureScreenshot", { format: "jpeg", quality: 55, captureBeyondViewport: true, clip: { ...clip, scale: 0.5 } });
      out.thumb = jpg.data; out.thumbSize = { w: d.vw, h: H, full: d.docHeight };
    }
  } catch (err) {
    add("load-failure", `Case crashed: ${err.message}`);
  } finally {
    cdp.off(sessionId);
    try { await cdp.send("Target.closeTarget", { targetId }); } catch { /* tab already gone */ }
    try { await cdp.send("Target.disposeBrowserContext", { browserContextId }); } catch { /* context already gone */ }
  }
  return out;
}

/* ------------------------------------------------------------------ turn raw page data into issues */
const MAX_PER_CHECK = 12;
function analyse(d, c, add, out) {
  const phone = c.viewport.name === "phone";
  const capped = (list, check, make) => { list.slice(0, MAX_PER_CHECK).forEach(make); if (list.length > MAX_PER_CHECK) add(check, `... and ${list.length - MAX_PER_CHECK} more`); };
  out.stats = { ...out.stats, docHeight: d.docHeight, screens: Math.round((d.docHeight / d.vh) * 10) / 10, textElements: d.items.length, themeApplied: d.theme };

  if (d.theme !== c.theme) add("load-failure", `Theme not applied: data-theme=${d.theme}, wanted ${c.theme}`);

  if (d.scrollWidth > d.clientWidth + 1) {
    const culprits = (d.offscreen.some((o) => !o.parentOver) ? d.offscreen.filter((o) => !o.parentOver) : d.offscreen).sort((a, b) => b.over - a.over);
    add("page-overflow", `Page scrolls horizontally: scrollWidth ${d.scrollWidth} > ${d.clientWidth}${culprits[0] ? `; widest offender ${culprits[0].sel} (+${culprits[0].over}px)` : ""}`, culprits[0] ? { selector: culprits[0].sel, box: culprits[0].box } : {});
  }
  const topOff = d.offscreen.filter((o) => !o.parentOver);
  capped(topOff, "offscreen-element", (o) => add("offscreen-element", `Extends ${o.over}px past the viewport edge, not inside a scroll container`, { selector: o.sel, box: o.box }));

  // overlapping text
  const items = d.items.filter((i) => !i.rotated && !matchesAny(`${i.sel} ${i.cls}`, OVERLAP_ALLOW));
  const flat = [];
  items.forEach((it, idx) => it.rects.forEach((r) => flat.push({ idx, r })));
  flat.sort((a, b) => a.r.x - b.r.x);
  const pairs = new Map();
  for (let i = 0; i < flat.length; i++) {
    for (let j = i + 1; j < flat.length && flat[j].r.x < flat[i].r.x + flat[i].r.w; j++) {
      const A = items[flat[i].idx], B = items[flat[j].idx];
      if (A === B || A.path === B.path) continue;
      if (A.path.startsWith(`${B.path}.`) || B.path.startsWith(`${A.path}.`)) continue; // ancestor / descendant
      const o = meaningfulOverlap(flat[i].r, flat[j].r, { minRatio: 0.3 });
      if (!o) continue;
      const key = [A.sel + "|" + A.txt, B.sel + "|" + B.txt].sort().join(" <> ");
      if (!pairs.has(key)) pairs.set(key, { a: A, b: B, box: o.box, ratio: o.ratio });
    }
  }
  capped([...pairs.values()].sort((p, q) => q.ratio - p.ratio), "text-overlap", (p) =>
    add("text-overlap", `"${p.a.txt}" overlaps "${p.b.txt}" (${Math.round(p.ratio * 100)}% of the smaller box)`, { selector: `${p.a.sel}  x  ${p.b.sel}`, box: p.box }));

  // clipped content
  capped(d.clipped.filter((k) => !matchesAny(k.cls, [/\bsr-only\b/])), "clipped-content", (k) =>
    add("clipped-content", `${k.hasTable ? "Table/content" : "Content"} clipped: scrollWidth ${k.scrollWidth} > clientWidth ${k.clientWidth} with overflow hidden ("${k.txt}")`, { selector: k.sel, box: k.box }));
  if (d.ellipsis.length) add("truncated-ellipsis", `${d.ellipsis.length} element(s) truncated with an ellipsis, e.g. ${d.ellipsis[0].sel}`, { selector: d.ellipsis[0].sel, box: d.ellipsis[0].box });

  // contrast, grouped by identical colour signature
  const base = parseColor(d.pageBg)?.a === 1 ? parseColor(d.pageBg) : parseColor(d.htmlBg)?.a === 1 ? parseColor(d.htmlBg) : { r: 255, g: 255, b: 255, a: 1 };
  const groups = new Map();
  let judged = 0, skipped = 0;
  for (const it of d.items) {
    if (it.disabled || matchesAny(`${it.sel} ${it.cls}`, CONTRAST_ALLOW) || !it.color || it.color.startsWith("url")) { skipped++; continue; }
    const r = evaluateContrast({ color: it.color, chain: it.chain, fontPx: it.fontPx, weight: it.weight, opacity: it.opacity }, base);
    if (!r) { skipped++; continue; }
    judged++;
    if (r.pass) continue;
    const key = `${it.color}|${JSON.stringify(it.chain)}|${it.opacity}|${it.fontPx >= 18 ? "L" : "S"}`;
    if (!groups.has(key)) groups.set(key, { r, n: 0, first: it });
    groups.get(key).n++;
  }
  out.stats.contrastJudged = judged; out.stats.contrastSkipped = skipped;
  capped([...groups.values()].sort((a, b) => a.r.ratio - b.r.ratio), "contrast", (g) =>
    add("contrast", `Contrast ${g.r.ratio.toFixed(2)}:1 (needs ${g.r.need}:1), ${g.n} text element(s), e.g. "${g.first.txt}" ${g.first.color}`, { selector: g.first.sel, box: g.first.rect0 }));

  if (phone) {
    const t = tapTargetIssues(d.taps);
    out.stats.tapTargets = { total: t.total, small: t.count };
    if (t.count) add("tap-target", `${t.count} of ${t.total} tap targets under 40x40; worst: ${t.worst.slice(0, 4).map((w) => `${w.w}x${w.h} ${w.sel}${w.txt ? ` "${w.txt}"` : ""}`).join("; ")}`, { selector: t.worst[0].sel, box: t.worst[0].box, worst: t.worst });
    if (d.docHeight / d.vh > PHONE_MAX_SCREENS) add("page-height", `Page is ${d.docHeight}px tall = ${(d.docHeight / d.vh).toFixed(1)} screens (limit ${PHONE_MAX_SCREENS})`);
  }
  if (d.titleLines > 1) add("title-wrap", `Header title wraps onto ${d.titleLines} lines`, { selector: d.titleSel, box: d.titleBox });
}

/* ------------------------------------------------------------------ report */
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));
function renderHtml(report) {
  const { cases, summary } = report;
  const routes = [...new Set(cases.map((x) => x.route))];
  const combos = [...new Set(cases.map((x) => `${x.viewport}/${x.theme}`))];
  const cell = (rt, cb) => {
    const x = cases.find((k) => k.route === rt && `${k.viewport}/${k.theme}` === cb); if (!x) return "<td></td>";
    const e = x.issues.filter((i) => i.severity === "error").length, w = x.issues.filter((i) => i.severity === "warn").length;
    return `<td class="${e ? "bad" : w ? "meh" : "ok"}"><a href="#${x.id}">${e ? `${e} err` : ""}${e && w ? " / " : ""}${w ? `${w} warn` : ""}${!e && !w ? "ok" : ""}</a></td>`;
  };
  const table = `<table class="sum"><tr><th>Route</th>${combos.map((cb) => `<th>${esc(cb)}</th>`).join("")}</tr>${routes.map((rt) => `<tr><th>${esc(rt)}</th>${combos.map((cb) => cell(rt, cb)).join("")}</tr>`).join("")}</table>`;
  const sections = cases.map((x) => {
    const numbered = x.issues.filter((i) => i.box);
    const overlays = x.thumbSize ? numbered.map((i, n) => {
      const b = i.box, W = x.thumbSize.w, H = x.thumbSize.h;
      if (b.y > H) return "";
      return `<div class="box ${i.severity}" style="left:${(b.x / W) * 100}%;top:${(b.y / H) * 100}%;width:${Math.max(0.5, (b.w / W) * 100)}%;height:${Math.max(0.3, (Math.min(b.h, H - b.y) / H) * 100)}%"><b>${n + 1}</b></div>`;
    }).join("") : "";
    const list = x.issues.map((i) => { const n = numbered.indexOf(i); return `<li class="${i.severity}"><span class="tag">${esc(i.check)}</span>${n >= 0 ? `<b class="n">${n + 1}</b>` : ""} ${esc(i.message)}${i.selector ? `<br><code>${esc(i.selector)}</code>` : ""}${i.box ? ` <small>[${i.box.x},${i.box.y} ${i.box.w}x${i.box.h}]</small>` : ""}</li>`; }).join("");
    const e = x.issues.filter((i) => i.severity === "error").length;
    return `<details id="${x.id}" ${e ? "open" : ""}><summary>${esc(x.route)} <i>${esc(x.viewport)} / ${esc(x.theme)}</i> <span class="${e ? "bad" : "ok"}">${e} error, ${x.issues.filter((i) => i.severity === "warn").length} warn</span> <small>${x.stats.screens ?? "?"} screens</small></summary>
<div class="case"><div class="shot ${esc(x.viewport)}">${x.thumb ? `<div class="wrap"><img loading="lazy" src="data:image/jpeg;base64,${x.thumb}">${overlays}</div>${x.thumbSize.full > x.thumbSize.h ? `<p class="note">Screenshot truncated at ${x.thumbSize.h}px of ${x.thumbSize.full}px</p>` : ""}` : "<em>no screenshot</em>"}</div>
<ol class="issues">${list || "<li class=ok>No issues</li>"}</ol></div></details>`;
  }).join("\n");
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Visual check ${esc(report.startedAt)}</title><style>
:root{color-scheme:light dark;--bg:#fff;--fg:#1a1d21;--mut:#6a7380;--line:#d9dde3;--bad:#c62828;--warn:#b26a00;--ok:#2e7d32}
@media(prefers-color-scheme:dark){:root{--bg:#14171b;--fg:#e6e9ed;--mut:#9aa4b0;--line:#2b3138;--bad:#ff6b6b;--warn:#ffb454;--ok:#6fcf7f}}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}h1{font-size:20px;margin:0 0 4px}.mut{color:var(--mut)}
table.sum{border-collapse:collapse;margin:12px 0;font-size:13px}table.sum th,table.sum td{border:1px solid var(--line);padding:4px 8px;text-align:left}table.sum a{color:inherit;text-decoration:none}
td.bad{background:color-mix(in srgb,var(--bad) 22%,transparent)}td.meh{background:color-mix(in srgb,var(--warn) 18%,transparent)}td.ok{background:color-mix(in srgb,var(--ok) 14%,transparent)}
details{border:1px solid var(--line);border-radius:6px;margin:8px 0}summary{cursor:pointer;padding:8px 10px;font-weight:600}summary i{font-weight:400;color:var(--mut)}span.bad{color:var(--bad)}span.ok{color:var(--ok)}
.case{display:flex;gap:16px;padding:10px;flex-wrap:wrap}.shot{flex:0 0 auto;max-width:100%}.shot.phone{width:200px}.shot.desktop{width:640px;max-width:100%}.wrap{position:relative;line-height:0;border:1px solid var(--line)}.wrap img{width:100%;height:auto}
.box{position:absolute;box-sizing:border-box;border:2px solid var(--warn);background:rgba(255,160,0,.15)}.box.error{border-color:var(--bad);background:rgba(255,0,0,.18)}.box.info{border-color:#58a;background:rgba(80,130,200,.12)}.box b{position:absolute;top:-1px;left:-1px;background:var(--bad);color:#fff;font:10px/1 sans-serif;padding:1px 3px}
ol.issues{flex:1 1 360px;margin:0;padding-left:22px}ol.issues li{margin:0 0 8px}li.error .tag{background:var(--bad)}li.warn .tag{background:var(--warn)}li.info .tag{background:#58a}.tag{color:#fff;font-size:11px;padding:1px 6px;border-radius:3px;margin-right:4px}.n{color:var(--bad)}
code{font-size:12px;color:var(--mut);word-break:break-all}.note{font-size:12px;color:var(--mut)}</style></head><body>
<h1>Visual check</h1><div class="mut">${esc(report.base)} - ${esc(report.startedAt)} - ${summary.cases} cases, ${summary.error} errors, ${summary.warn} warnings, ${summary.info} info - ${report.durationSec}s</div>
${table}${sections}</body></html>`;
}

/* ------------------------------------------------------------------ main */
async function main() {
  const args = parseArgs(process.argv.slice(2));
  try { const r = await fetch(`${args.base}/`, { signal: AbortSignal.timeout(8000) }); if (r.status >= 500) throw new Error(`HTTP ${r.status}`); }
  catch (err) { console.error(`Server not reachable at ${args.base} (${err.cause?.code ?? err.message}). Start the dev server (e.g. npm run dev) or pass --base <url>.`); process.exit(2); }
  const browserPath = findBrowser(args.browser);
  if (!browserPath) { console.error("No Chrome/Edge found. Pass --browser <path> or set CHROME_PATH."); process.exit(2); }

  const allRoutes = loadRoutes();
  const routes = args.routes ? args.routes.map((r) => `/${r}`) : allRoutes;
  const key = themeKey();
  const cases = [];
  for (const route of routes) for (const vp of args.viewports) for (const theme of args.themes) {
    if (!VIEWPORTS[vp]) { console.error(`Unknown viewport ${vp}`); process.exit(2); }
    cases.push({ route, viewport: VIEWPORTS[vp], theme, id: `${route.slice(1) || "root"}__${vp}__${theme}` });
  }
  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const outDir = args.out ?? join(ROOT, "dist-visual", stamp);
  const shotsDir = join(outDir, "shots");
  mkdirSync(shotsDir, { recursive: true });

  const started = Date.now();
  const browser = await launchBrowser(browserPath);
  const cleanup = () => browser.close();
  process.on("SIGINT", () => { cleanup(); process.exit(130); });
  let results = [];
  try {
    const cdp = await Cdp.connect(browser.wsUrl);
    console.log(`Checking ${cases.length} cases against ${args.base} with ${args.jobs} parallel tabs (${browserPath.split(/[\\/]/).pop()})`);
    let next = 0;
    const done = [];
    await Promise.all(Array.from({ length: args.jobs }, async () => {
      for (;;) {
        const i = next++; if (i >= cases.length) return;
        const r = await runCase(cdp, args.base, cases[i], { themeKey: key, shots: args.shots }, shotsDir);
        done[i] = r;
        const e = r.issues.filter((x) => x.severity === "error").length, w = r.issues.filter((x) => x.severity === "warn").length;
        console.log(`  [${String(++results.length).padStart(2)}/${cases.length}] ${r.route.padEnd(18)} ${r.viewport.padEnd(7)} ${r.theme.padEnd(5)} ${e} error ${w} warn`);
      }
    }));
    results = done;
  } finally { cleanup(); }

  const summary = summarize(results);
  const report = { base: args.base, startedAt: new Date(started).toISOString(), durationSec: Math.round((Date.now() - started) / 1000), summary,
    cases: results.map((r) => { const rest = { ...r }; delete rest.thumb; return rest; }) };
  writeFileSync(join(outDir, "report.json"), JSON.stringify(report, null, 1));
  writeFileSync(join(outDir, "report.html"), renderHtml({ ...report, cases: results }));
  console.log(`\n${summary.cases} cases: ${summary.error} errors, ${summary.warn} warnings, ${summary.info} info in ${report.durationSec}s`);
  console.log(`Report: ${join(outDir, "report.html")}`);
  process.exit(summary.failed ? 1 : 0);
}

main().catch((err) => { console.error(err); process.exit(2); });
