// Perceived-speed harness: cold-loads the key routes in headless Chrome over the DevTools Protocol (Node's built-in WebSocket, no dependencies)
// on a throttled phone (4x CPU, fast 4G, 375x812 touch) and an unthrottled desktop (1280x900), and records
//   LCP, FCP, CLS (shifts not caused by input), TBT (long tasks over 50 ms between FCP and load + settle), longest task,
//   INP proxy (click a tab / row / toggle, take the Event Timing duration and the time to the next two frames),
//   route-change time (click a nav link; full page load, warm cache, to its LCP) and transferred JS/CSS bytes.
//
//   node scripts/perf-check.mjs [--base http://localhost:3111] [--routes this-week,why-priced] [--viewports phone,desktop] [--runs 1]
//                               [--browser <chrome>] [--out docs/ui-overhaul/perf-latest.json] [--event pga%3A457%3A2026-11-05]
// Needs a running preview (node scripts/ui-preview.mjs --detach --no-build). Targets (phone throttled): LCP < 2.0 s, CLS < 0.05, no task > 200 ms on interaction.
// Exit code 0 always unless setup fails (2); the table flags misses with "!".
import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const args = process.argv.slice(2);
const opt = (n, d) => (args.includes(n) ? args[args.indexOf(n) + 1] : d);
const BASE = opt("--base", "http://localhost:3111").replace(/\/+$/, "");
const ROUTES = opt("--routes", "this-week,why-priced,players?player=10091,weekly-players,bet-breakdowns,weather-effects").split(",");
const VIEWPORTS = opt("--viewports", "phone,desktop").split(",");
const RUNS = Math.max(1, Number(opt("--runs", "1")));
const OUT = opt("--out", null);
const EVENT = opt("--event", null);
const WATERFALL = args.includes("--waterfall");
const PROBE = opt("--probe", null); // CSS selector: prints the height of each match after settle (to size skeleton placeholders)

const PROFILES = {
  phone: { width: 375, height: 812, mobile: true, dpr: 2, cpu: 4, net: { latency: 150, down: (1.6 * 1024 * 1024) / 8, up: (750 * 1024) / 8 }, label: "phone (4x CPU, fast 4G)" },
  desktop: { width: 1280, height: 900, mobile: false, dpr: 1, cpu: 1, net: null, label: "desktop (no throttle)" },
};
// What to click for the INP proxy on each route, first match wins. Reading never changes data; these are view toggles / row expanders.
const INTERACT = ['[role="tab"]:not([aria-selected="true"])', ".segmented button:not([aria-pressed=\"true\"])", "button[aria-pressed=\"false\"]", "details:not([open]) > summary", "table tbody tr", "button:not([disabled])"];
// Nav link to click for the route-change measurement (a different page than the one loaded).
const NAV_TARGET = { "this-week": "/why-priced", "why-priced": "/this-week", "players?player=10091": "/this-week", "weekly-players": "/why-priced", "bet-breakdowns": "/this-week", "weather-effects": "/this-week" };

function findBrowser(explicit) {
  const candidates = [explicit, process.env.CHROME_PATH, "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe", "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe", "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"].filter(Boolean);
  return candidates.find((p) => existsSync(p));
}

async function launchBrowser(path) {
  const userDir = mkdtempSync(join(tmpdir(), "perf-check-"));
  const child = spawn(path, ["--headless=new", "--remote-debugging-port=0", `--user-data-dir=${userDir}`, "--no-first-run", "--no-default-browser-check", "--disable-extensions",
    "--disable-background-timer-throttling", "--disable-renderer-backgrounding", "--disable-backgrounding-occluded-windows", "--force-color-profile=srgb", "about:blank"], { stdio: "ignore", windowsHide: true });
  let exited = false;
  child.on("exit", () => { exited = true; });
  const portFile = join(userDir, "DevToolsActivePort");
  for (let i = 0; i < 150 && !existsSync(portFile) && !exited; i++) await sleep(100);
  if (!existsSync(portFile)) { child.kill(); throw new Error("Browser did not publish DevToolsActivePort"); }
  await sleep(100);
  const [port, wsPath] = readFileSync(portFile, "utf8").trim().split(/\r?\n/);
  const close = () => {
    try { if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(child.pid), "/T", "/F"], { stdio: "ignore" }); else child.kill("SIGKILL"); } catch { /* gone */ }
    for (let i = 0; i < 5; i++) { try { rmSync(userDir, { recursive: true, force: true }); break; } catch { Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 200); } }
  };
  return { wsUrl: `ws://127.0.0.1:${port}${wsPath}`, close };
}

class Cdp {
  constructor(ws) { this.ws = ws; this.id = 0; this.pending = new Map(); this.handlers = new Map(); ws.onmessage = (ev) => this.#msg(ev); }
  static async connect(url) {
    const ws = new WebSocket(url);
    await new Promise((res, rej) => { ws.onopen = res; ws.onerror = () => rej(new Error("DevTools WebSocket failed to open")); });
    return new Cdp(ws);
  }
  #msg(ev) {
    const m = JSON.parse(typeof ev.data === "string" ? ev.data : Buffer.from(ev.data).toString());
    if (m.id) { const p = this.pending.get(m.id); if (!p) return; this.pending.delete(m.id); if (m.error) p.rej(new Error(`${p.method}: ${m.error.message}`)); else p.res(m.result); }
    else if (m.method) this.handlers.get(m.sessionId ?? "")?.(m.method, m.params);
  }
  send(method, params = {}, sessionId) {
    const id = ++this.id;
    return new Promise((res, rej) => {
      const timer = setTimeout(() => { this.pending.delete(id); rej(new Error(`${method}: timed out`)); }, 90000);
      this.pending.set(id, { method, res: (v) => { clearTimeout(timer); res(v); }, rej: (e) => { clearTimeout(timer); rej(e); } });
      this.ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
    });
  }
  on(sid, fn) { this.handlers.set(sid, fn); }
}

/** Installed before any page script: collects the web-vitals style entries in window.__perf. */
const OBSERVER = `(() => {
  const P = (window.__perf = { lcp: 0, lcpEl: "", fcp: 0, cls: 0, shifts: [], longTasks: [], events: [], clicks: [] });
  const sel = (el) => { if (!el || !el.localName) return ""; const c = [...(el.classList || [])].slice(0, 2).join("."); return el.localName + (el.id ? "#" + el.id : "") + (c ? "." + c : ""); };
  const obs = (type, fn, extra) => { try { new PerformanceObserver((l) => l.getEntries().forEach(fn)).observe({ type, buffered: true, ...extra }); } catch (e) {} };
  obs("largest-contentful-paint", (e) => { P.lcp = e.startTime; P.lcpEl = sel(e.element) + (e.url ? " (img)" : ""); });
  obs("paint", (e) => { if (e.name === "first-contentful-paint") P.fcp = e.startTime; });
  obs("layout-shift", (e) => { if (e.hadRecentInput) return; P.cls += e.value; P.shifts.push({ t: Math.round(e.startTime), v: +e.value.toFixed(4), el: (e.sources || []).slice(0, 2).map((s) => sel(s.node) + " " + Math.round(s.previousRect.y) + "->" + Math.round(s.currentRect.y)).join(" | ") }); });
  P.series = []; let last = ""; const sample = () => { const h = [...document.querySelectorAll("#main-content > * > *")].map((n) => sel(n) + ":" + Math.round(n.getBoundingClientRect().height)).join(" "); if (h !== last) { last = h; P.series.push({ t: Math.round(performance.now()), h }); } }; setInterval(sample, 100);
  obs("longtask", (e) => P.longTasks.push({ t: Math.round(e.startTime), d: Math.round(e.duration) }));
  obs("event", (e) => { if (e.interactionId) P.events.push({ type: e.name, t: Math.round(e.startTime), d: Math.round(e.duration), input: Math.round(e.processingStart - e.startTime), proc: Math.round(e.processingEnd - e.processingStart) }); }, { durationThreshold: 16 });
})();`;

async function newPage(cdp, profile) {
  const { browserContextId } = await cdp.send("Target.createBrowserContext", { disposeOnDetach: true }); // cold: empty HTTP cache
  const { targetId } = await cdp.send("Target.createTarget", { url: "about:blank", browserContextId });
  const { sessionId } = await cdp.send("Target.attachToTarget", { targetId, flatten: true });
  const s = (m, p) => cdp.send(m, p, sessionId);
  await s("Page.enable"); await s("Runtime.enable"); await s("Network.enable");
  await s("Emulation.setDeviceMetricsOverride", { width: profile.width, height: profile.height, deviceScaleFactor: profile.dpr, mobile: profile.mobile });
  if (profile.mobile) await s("Emulation.setTouchEmulationEnabled", { enabled: true, maxTouchPoints: 5 });
  await s("Emulation.setCPUThrottlingRate", { rate: profile.cpu });
  await s("Network.emulateNetworkConditions", { offline: false, latency: profile.net?.latency ?? 0, downloadThroughput: profile.net?.down ?? -1, uploadThroughput: profile.net?.up ?? -1 });
  await s("Page.addScriptToEvaluateOnNewDocument", { source: OBSERVER });
  const bytes = { js: 0, css: 0, other: 0, requests: 0 };
  const types = new Map();
  let loads = 0; const loadWaiters = [];
  cdp.on(sessionId, (method, params) => {
    if (method === "Page.loadEventFired") { loads++; loadWaiters.splice(0).forEach((f) => f()); }
    if (method === "Network.responseReceived") types.set(params.requestId, params.type);
    if (method === "Network.loadingFinished") { const t = types.get(params.requestId); bytes.requests++; if (t === "Script") bytes.js += params.encodedDataLength; else if (t === "Stylesheet") bytes.css += params.encodedDataLength; else bytes.other += params.encodedDataLength; }
  });
  const evalIn = async (expression) => { if (process.env.PERF_DEBUG) console.log("eval", expression.slice(0, 70)); const r = await s("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true }); if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? "evaluate failed"); return r.result.value; };
  const nextLoad = (fromCount) => new Promise((res) => { if (loads > fromCount) res(); else { loadWaiters.push(res); setTimeout(res, 40000); } });
  return { s, evalIn, bytes, loadCount: () => loads, nextLoad, close: () => cdp.send("Target.closeTarget", { targetId }).catch(() => undefined) };
}

const url = (route) => `${BASE}/${route}${EVENT ? (route.includes("?") ? "&" : "?") + "e=" + EVENT : ""}`;
const settle = async (page, ms, fromCount) => { // wait for the load event, then let late fonts / data / images / charts arrive
  await page.nextLoad(fromCount);
  await sleep(ms);
};
const snapshot = (page) => page.evalIn(`(() => { const p = window.__perf; const res = performance.getEntriesByType('resource').map((r) => ({ n: r.name.replace(location.origin, '').slice(0, 90), t: r.initiatorType, s: Math.round(r.startTime), e: Math.round(r.responseEnd), kb: +(r.encodedBodySize / 1024).toFixed(1) })); const nav = performance.getEntriesByType("navigation")[0]; return { lcp: p.lcp, lcpEl: p.lcpEl, fcp: p.fcp, cls: p.cls, shifts: p.shifts, longTasks: p.longTasks, events: p.events, series: p.series, load: nav ? nav.loadEventEnd : 0, dcl: nav ? nav.domContentLoadedEventEnd : 0, ttfb: nav ? nav.responseStart : 0, res }; })()`);

async function measureRoute(cdp, route, profile) {
  const page = await newPage(cdp, profile);
  try {
    await page.s("Page.navigate", { url: url(route) });
    await settle(page, profile.cpu > 1 ? 4500 : 2500, 0);
    const cold = await snapshot(page);
    // TBT: long task time over 50 ms from FCP to the end of the settle window.
    const tbt = cold.longTasks.filter((t) => t.t >= cold.fcp).reduce((a, t) => a + Math.max(0, t.d - 50), 0);
    const bytes = { ...page.bytes };
    if (PROBE) console.log(`    probe ${PROBE} (${profile.width}px):`, JSON.stringify(await page.evalIn(`[...document.querySelectorAll(${JSON.stringify(PROBE)})].map((e) => Math.round(e.getBoundingClientRect().height))`)));
    // INP proxy: click the first matching control with a real input event, then time to the next two frames.
    const hit = await page.evalIn(`(() => {
      const list = ${JSON.stringify(INTERACT)}; const main = document.querySelector("main") || document.body;
      for (const q of list) { for (const el of main.querySelectorAll(q)) { const r = el.getBoundingClientRect(); const cs = getComputedStyle(el);
        if (r.width > 8 && r.height > 8 && r.bottom > 0 && r.top < innerHeight && cs.visibility !== "hidden" && cs.pointerEvents !== "none" && !el.closest("[inert]")) { return { q, x: r.left + r.width / 2, y: r.top + r.height / 2, label: (el.textContent || "").trim().slice(0, 30) }; } } }
      return null; })()`);
    let inp = null;
    if (hit) {
      const before = (await snapshot(page)).longTasks.length;
      await page.evalIn(`window.__clickAt = performance.now(); window.__frames = new Promise((res) => { addEventListener("pointerdown", () => { const t0 = performance.now(); requestAnimationFrame(() => requestAnimationFrame(() => res(performance.now() - t0))); }, { once: true, capture: true }); }); 0`);
      if (profile.mobile) {
        await page.s("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: [{ x: hit.x, y: hit.y }] });
        await page.s("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
      } else {
        await page.s("Input.dispatchMouseEvent", { type: "mouseMoved", x: hit.x, y: hit.y });
        await page.s("Input.dispatchMouseEvent", { type: "mousePressed", x: hit.x, y: hit.y, button: "left", clickCount: 1 });
        await page.s("Input.dispatchMouseEvent", { type: "mouseReleased", x: hit.x, y: hit.y, button: "left", clickCount: 1 });
      }
      await sleep(1500);
      const frames = await page.evalIn(`Promise.race([window.__frames, new Promise((r) => setTimeout(() => r(null), 500))])`);
      const after = await snapshot(page);
      const ev = after.events.filter((e) => e.t >= 0).sort((a, b) => b.d - a.d)[0] ?? null;
      const newTasks = after.longTasks.slice(before);
      inp = { target: hit.label || hit.q, frames: frames == null ? null : Math.round(frames), eventDuration: ev?.d ?? null, longestTask: newTasks.reduce((a, t) => Math.max(a, t.d), 0), cls: +(after.cls - cold.cls).toFixed(4) };
    }
    // Route change: a nav link click is a full page load in this app (navigateWithReload); time click -> new document's LCP with a warm cache.
    let nav = null;
    const target = NAV_TARGET[route] ?? "/this-week";
    const clicked = await page.evalIn(`(() => { const a = [...document.querySelectorAll('a[href^="${target}"]')].find((x) => { const r = x.getBoundingClientRect(); return r.width > 4 && r.height > 4 && r.left >= 0 && r.right <= innerWidth && r.top >= 0 && r.bottom <= innerHeight; }); if (!a) return null; const r = a.getBoundingClientRect(); return { x: r.left + r.width / 2, y: r.top + r.height / 2 }; })()`);
    if (clicked) {
      const t0 = Date.now(); const loadsBefore = page.loadCount();
      await page.s("Input.dispatchMouseEvent", { type: "mousePressed", x: clicked.x, y: clicked.y, button: "left", clickCount: 1 });
      await page.s("Input.dispatchMouseEvent", { type: "mouseReleased", x: clicked.x, y: clicked.y, button: "left", clickCount: 1 });
      await settle(page, profile.cpu > 1 ? 3500 : 1800, loadsBefore);
      const second = await snapshot(page);
      nav = { to: target, lcp: Math.round(second.lcp), cls: +second.cls.toFixed(4), wallMs: Date.now() - t0 - (profile.cpu > 1 ? 3500 : 1800) };
    }
    return {
      series: cold.series, resources: cold.res, route, viewport: profile.label, lcp: Math.round(cold.lcp), lcpEl: cold.lcpEl, fcp: Math.round(cold.fcp), ttfb: Math.round(cold.ttfb), cls: +cold.cls.toFixed(4), shifts: cold.shifts.slice(0, 6),
      tbt: Math.round(tbt), longestTask: cold.longTasks.reduce((a, t) => Math.max(a, t.d), 0), longTasks: cold.longTasks.length,
      jsKB: +(bytes.js / 1024).toFixed(1), cssKB: +(bytes.css / 1024).toFixed(1), otherKB: +(bytes.other / 1024).toFixed(1), requests: bytes.requests, inp, nav,
    };
  } finally { await page.close(); }
}

const median = (list) => { const v = list.filter((x) => x != null).sort((a, b) => a - b); return v.length ? v[Math.floor(v.length / 2)] : null; };

async function main() {
  const browserPath = findBrowser(opt("--browser", null));
  if (!browserPath) { console.error("No Chrome/Edge found (set CHROME_PATH or --browser)."); process.exit(2); }
  try { const r = await fetch(BASE + "/this-week"); if (!r.ok) throw new Error(String(r.status)); } catch (e) { console.error(`Preview not reachable at ${BASE} (${e.message}). Start it: node scripts/ui-preview.mjs --detach --no-build`); process.exit(2); }
  const browser = await launchBrowser(browserPath);
  const cdp = await Cdp.connect(browser.wsUrl);
  const results = [];
  try {
    for (const vp of VIEWPORTS) for (const route of ROUTES) {
      const runs = [];
      for (let i = 0; i < RUNS; i++) runs.push(await measureRoute(cdp, route, PROFILES[vp]));
      const pick = runs.sort((a, b) => a.lcp - b.lcp)[Math.floor(runs.length / 2)];
      results.push({ vp, ...pick, runs: RUNS > 1 ? runs.map((r) => r.lcp) : undefined });
      const r = pick;
      if (args.includes("--series")) for (const x of r.series) console.log(`    ${String(x.t).padStart(5)} ${x.h.slice(0, 400)}`);
      if (WATERFALL) for (const x of r.resources.filter((q) => q.e > 300).sort((a, b) => a.s - b.s)) console.log(`    ${String(x.s).padStart(5)} -> ${String(x.e).padStart(5)}  ${String(x.kb).padStart(7)}KB ${x.t.padEnd(8)} ${x.n}`);
      console.log(`${vp.padEnd(8)} ${route.padEnd(24)} LCP ${String(r.lcp).padStart(5)}ms${vp === "phone" && r.lcp > 2000 ? "!" : " "} FCP ${String(r.fcp).padStart(5)} CLS ${r.cls.toFixed(3)}${r.cls > 0.05 ? "!" : " "} TBT ${String(r.tbt).padStart(4)} maxTask ${String(r.longestTask).padStart(4)} JS ${String(r.jsKB).padStart(6)}KB  inp ${r.inp ? `${r.inp.frames ?? "-"}ms/ev ${r.inp.eventDuration ?? "-"}/task ${r.inp.longestTask}${r.inp.longestTask > 200 ? "!" : ""}` : "n/a"}  nav ${r.nav ? `${r.nav.lcp}ms` : "n/a"}`);
    }
  } finally { cdp.ws.close(); browser.close(); }
  if (OUT) { mkdirSync(dirname(resolve(OUT)), { recursive: true }); writeFileSync(resolve(OUT), JSON.stringify({ at: new Date().toISOString(), base: BASE, results }, null, 2)); console.log(`wrote ${OUT}`); }
  const worstPhone = results.filter((r) => r.vp === "phone");
  console.log(`\nphone median LCP ${median(worstPhone.map((r) => r.lcp))} ms, worst CLS ${Math.max(0, ...worstPhone.map((r) => r.cls)).toFixed(3)}, worst interaction task ${Math.max(0, ...worstPhone.map((r) => r.inp?.longestTask ?? 0))} ms`);
}
await main();
