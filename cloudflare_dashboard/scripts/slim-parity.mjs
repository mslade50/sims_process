// Parity proof for the slim explain document: loads each first-screen route twice (slim as served, then with the ?slim=1 request rewritten to the
// full document) and compares the visible text of <main>. Exit 1 on any difference.
//   node scripts/slim-parity.mjs [--base http://localhost:3111] [--out dist-visual/R4b/slim-parity.json] [--event euro%3A2026142%3A2026-11-05]
import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const args = process.argv.slice(2);
const opt = (n, d) => (args.includes(n) ? args[args.indexOf(n) + 1] : d);
const BASE = opt("--base", "http://localhost:3111").replace(/\/+$/, "");
const OUT = opt("--out", "dist-visual/R4b/slim-parity.json");
const EVENTS = opt("--events", "euro%3A2026142%3A2026-11-05,pga%3A457%3A2026-11-05").split(",");
const ROUTES = ["this-week", "why-priced"];

const browser = [process.env.CHROME_PATH, "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe", "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe", "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe", "/usr/bin/google-chrome", "/usr/bin/chromium"].filter(Boolean).find((p) => existsSync(p));
if (!browser) { console.error("No Chrome/Edge found"); process.exit(2); }
const userDir = mkdtempSync(join(tmpdir(), "slim-parity-"));
const child = spawn(browser, ["--headless=new", "--remote-debugging-port=0", `--user-data-dir=${userDir}`, "--no-first-run", "--no-default-browser-check", "--disable-extensions", "about:blank"], { stdio: "ignore", windowsHide: true });
const cleanup = () => {
  try { if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(child.pid), "/T", "/F"], { stdio: "ignore" }); else child.kill("SIGKILL"); } catch { /* gone */ }
  for (let i = 0; i < 5; i++) { try { rmSync(userDir, { recursive: true, force: true }); break; } catch { Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 200); } }
};
process.on("exit", cleanup);
const portFile = join(userDir, "DevToolsActivePort");
for (let i = 0; i < 150 && !existsSync(portFile); i++) await sleep(100);
if (!existsSync(portFile)) { console.error("Browser did not start"); process.exit(2); }
await sleep(100);
const [port, wsPath] = readFileSync(portFile, "utf8").trim().split(/\r?\n/);
const ws = new WebSocket(`ws://127.0.0.1:${port}${wsPath}`);
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = () => rej(new Error("DevTools WebSocket failed")); });
let nextId = 0; const pending = new Map(); const handlers = new Map();
ws.onmessage = (ev) => {
  const m = JSON.parse(typeof ev.data === "string" ? ev.data : Buffer.from(ev.data).toString());
  if (m.id && pending.has(m.id)) { const p = pending.get(m.id); pending.delete(m.id); if (m.error) p.rej(new Error(`${p.method}: ${m.error.message}`)); else p.res(m.result); }
  else if (m.method) handlers.get(m.sessionId ?? "")?.(m.method, m.params);
};
const send = (method, params = {}, sessionId) => new Promise((res, rej) => { const id = ++nextId; pending.set(id, { method, res, rej }); ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) })); });

async function visibleText(route, event, rewrite) {
  const { browserContextId } = await send("Target.createBrowserContext", { disposeOnDetach: true });
  const { targetId } = await send("Target.createTarget", { url: "about:blank", browserContextId });
  const { sessionId } = await send("Target.attachToTarget", { targetId, flatten: true });
  const s = (m, p) => send(m, p, sessionId);
  await s("Page.enable"); await s("Runtime.enable");
  await s("Emulation.setDeviceMetricsOverride", { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false });
  const seen = { slim: 0, full: 0 };
  if (rewrite) {
    handlers.set(sessionId, (method, params) => {
      if (method !== "Fetch.requestPaused") return;
      seen.full++;
      s("Fetch.continueRequest", { requestId: params.requestId, url: params.request.url.replace(/\?slim=1/, "") }).catch(() => undefined);
    });
    await s("Fetch.enable", { patterns: [{ urlPattern: "*slim=1*", requestStage: "Request" }] });
  } else {
    handlers.set(sessionId, (method, params) => { if (method === "Network.requestWillBeSent" && /slim=1/.test(params.request.url)) seen.slim++; });
    await s("Network.enable");
  }
  await s("Page.navigate", { url: `${BASE}/${route}?e=${event}` });
  let text = "";
  for (let i = 0; i < 60; i++) {
    await sleep(500);
    const r = await s("Runtime.evaluate", { expression: "(document.querySelector('main')?.innerText ?? '')", returnByValue: true });
    if (r.result.value && r.result.value.length > 500 && !/Loading the pricing explanation/.test(r.result.value)) { text = r.result.value; break; }
  }
  await sleep(2500);   // late sections (lazy blocks within reach) settle
  const r = await s("Runtime.evaluate", { expression: "(window.scrollTo(0, 0), document.querySelector('main')?.innerText ?? '')", returnByValue: true });
  text = r.result.value || text;
  await send("Target.closeTarget", { targetId }).catch(() => undefined);
  return { text, seen };
}

const result = { generated: new Date().toISOString(), cases: [], pass: true };
for (const event of EVENTS) for (const route of ROUTES) {
  const a = await visibleText(route, event, false);
  const b = await visibleText(route, event, true);
  const norm = (t) => t.replace(/\d+ (?:min|h) ago/g, "N ago").replace(/\s+/g, " ").trim();
  const same = norm(a.text) === norm(b.text) && a.text.length > 500;
  let firstDiff = null;
  if (!same) { const x = norm(a.text), y = norm(b.text); let i = 0; while (i < x.length && x[i] === y[i]) i++; firstDiff = { at: i, slim: x.slice(Math.max(0, i - 60), i + 80), full: y.slice(Math.max(0, i - 60), i + 80) }; }
  result.cases.push({ route, event, slimRequests: a.seen.slim, rewrittenToFull: b.seen.full, textLength: a.text.length, identical: same, firstDiff });
  if (!same) result.pass = false;
  console.log(`${same ? "SAME" : "DIFF"} ${route} ${event} (${a.text.length} chars, slim requests ${a.seen.slim}, rewritten ${b.seen.full})${firstDiff ? "\n   slim: " + firstDiff.slim + "\n   full: " + firstDiff.full : ""}`);
}
mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, JSON.stringify(result, null, 2));
console.log(result.pass ? "PASS" : "FAIL", "->", OUT);
ws.close();
process.exit(result.pass ? 0 : 1);
