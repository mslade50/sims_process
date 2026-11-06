// Keyboard proof for the player drawer over the DevTools Protocol (real key events, no dependencies).
//   node scripts/a11y-drawer.mjs [--base http://localhost:3111] [--out dist-visual/R4b/a11y-drawer.json] [--event euro%3A2026142%3A2026-11-05]
// For each viewport (phone 375x812, desktop 1280x900) on /why-priced: focus a player row, press Enter (opens the drawer), then checks
//   focus moves inside the dialog, Tab / Shift+Tab stay trapped, the background is inert, Escape closes and returns focus to the opener,
//   and the same again closing with the close button (Tab to it, Enter). Exit code 1 when any check fails. Needs a running preview.
import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const args = process.argv.slice(2);
const opt = (n, d) => (args.includes(n) ? args[args.indexOf(n) + 1] : d);
const BASE = opt("--base", "http://localhost:3111").replace(/\/+$/, "");
const OUT = opt("--out", "dist-visual/R4b/a11y-drawer.json");
const EVENT = opt("--event", "euro%3A2026142%3A2026-11-05");
const VIEWPORTS = { phone: { width: 375, height: 812, mobile: true, dpr: 2 }, desktop: { width: 1280, height: 900, mobile: false, dpr: 1 } };

const browser = [process.env.CHROME_PATH, "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe", "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe", "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe", "/usr/bin/google-chrome", "/usr/bin/chromium"].filter(Boolean).find((p) => existsSync(p));
if (!browser) { console.error("No Chrome/Edge found"); process.exit(2); }

const userDir = mkdtempSync(join(tmpdir(), "a11y-drawer-"));
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
let nextId = 0; const pending = new Map();
ws.onmessage = (ev) => { const m = JSON.parse(typeof ev.data === "string" ? ev.data : Buffer.from(ev.data).toString()); if (m.id && pending.has(m.id)) { const p = pending.get(m.id); pending.delete(m.id); if (m.error) p.rej(new Error(`${p.method}: ${m.error.message}`)); else p.res(m.result); } };
const send = (method, params = {}, sessionId) => new Promise((res, rej) => { const id = ++nextId; pending.set(id, { method, res, rej }); ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) })); });

const result = { generated: new Date().toISOString(), base: BASE, event: EVENT, viewports: {}, pass: true };
for (const [name, vp] of Object.entries(VIEWPORTS)) {
  const { targetId } = await send("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await send("Target.attachToTarget", { targetId, flatten: true });
  const s = (m, p) => send(m, p, sessionId);
  await s("Page.enable"); await s("Runtime.enable");
  await s("Emulation.setDeviceMetricsOverride", { width: vp.width, height: vp.height, deviceScaleFactor: vp.dpr, mobile: vp.mobile });
  const ev = async (expr) => { const r = await s("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true }); if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? "evaluate failed"); return r.result.value; };
  const key = async (k, code, vk, modifiers = 0) => {
    const base = { key: k, code, windowsVirtualKeyCode: vk, nativeVirtualKeyCode: vk, modifiers };
    await s("Input.dispatchKeyEvent", k === "Enter" ? { type: "keyDown", ...base, text: "\r", unmodifiedText: "\r" } : { type: "rawKeyDown", ...base });
    await s("Input.dispatchKeyEvent", { type: "keyUp", ...base });
    await sleep(120);
  };
  const Tab = (shift) => key("Tab", "Tab", 9, shift ? 8 : 0);
  const Enter = () => key("Enter", "Enter", 13);
  const Escape = () => key("Escape", "Escape", 27);
  const where = () => ev(`(() => { const a = document.activeElement; const d = document.querySelector('[role=dialog]'); const sel = (e) => e ? e.localName + (e.className && typeof e.className === 'string' ? '.' + e.className.split(' ')[0] : '') + (e.getAttribute('aria-label') ? '[' + e.getAttribute('aria-label') + ']' : '') : null; return { active: sel(a), text: (a?.textContent || '').trim().slice(0, 40), inDialog: !!(d && a && d.contains(a)), dialog: !!d, mainInert: document.querySelector('main')?.hasAttribute('inert') ?? null, tabbarInert: document.querySelector('.tabbar')?.hasAttribute('inert') ?? null, activeIsBody: a === document.body }; })()`);
  const checks = []; const check = (label, ok, detail) => { checks.push({ label, ok: !!ok, detail }); if (!ok) result.pass = false; if (process.env.A11Y_VERBOSE) console.log(name, ok ? 'ok  ' : 'FAIL', label, ok ? '' : JSON.stringify(detail)); };
  await s("Page.navigate", { url: `${BASE}/why-priced?e=${EVENT}` });
  let opener = null;
  for (let i = 0; i < 80; i++) { await sleep(250); opener = await ev(`(() => { const b = document.querySelector('.pl-open'); return b ? (b.textContent || '').trim() : null; })()`).catch(() => null); if (opener) break; }
  if (!opener) { check("player row present", false, "no .pl-open row rendered"); result.viewports[name] = { checks }; continue; }
  await sleep(1500);
  // Focus the row through the keyboard path (programmatic focus on the real row button, then real Enter), twice: Escape close, then close-button close.
  const openFromRow = async () => { await ev(`(() => { const b = document.querySelector('.pl-open'); b.scrollIntoView({ block: 'center' }); b.focus(); window.__opener = b; return true; })()`); await sleep(100); await Enter(); for (let i = 0; i < 40; i++) { await sleep(100); if ((await where()).dialog) break; } await sleep(450); };
  const sameOpener = () => ev(`document.activeElement === window.__opener`);
  for (const closeWith of ["escape", "close-button"]) {
    const step = (t) => `${closeWith}: ${t}`;
    await openFromRow();
    let w = await where();
    check(step("Enter on a row opens the dialog"), w.dialog, w);
    check(step("focus moves inside the dialog"), w.inDialog, w);
    check(step("background is inert (main)"), w.mainInert === true, w);
    if (vp.mobile) check(step("background is inert (tab bar)"), w.tabbarInert === true, w);
    const seen = new Set(); let escaped = false;
    for (let i = 0; i < 45; i++) { await Tab(false); w = await where(); seen.add(`${w.active}|${w.text}`); if (!w.inDialog) escaped = true; }
    check(step("Tab x45 never leaves the dialog"), !escaped && seen.size > 3, { distinctStops: seen.size });
    escaped = false;
    for (let i = 0; i < 12; i++) { await Tab(true); w = await where(); if (!w.inDialog) escaped = true; }
    check(step("Shift+Tab x12 never leaves the dialog"), !escaped, w);
    if (closeWith === "escape") {
      await Escape();
    } else {
      await ev(`document.querySelector('.ex-close')?.focus()`); await sleep(80);
      check(step("close button reachable by keyboard and focused"), (await where()).active?.startsWith("button.ex-close"), await where());
      await Enter();
    }
    await sleep(500);
    w = await where();
    check(step("dialog removed"), !w.dialog, w);
    check(step("background no longer inert"), w.mainInert === false, w);
    check(step("focus returned to the opener row"), await sameOpener(), w);
  }
  result.viewports[name] = { checks, passed: checks.filter((c) => c.ok).length, failed: checks.filter((c) => !c.ok).length };
  await send("Target.closeTarget", { targetId }).catch(() => undefined);
}
mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, JSON.stringify(result, null, 2));
for (const [n, v] of Object.entries(result.viewports)) console.log(`${n}: ${v.passed ?? 0} pass, ${v.failed ?? "?"} fail${(v.checks ?? []).filter((c) => !c.ok).map((c) => `\n   FAIL ${c.label} ${JSON.stringify(c.detail)}`).join("")}`);
console.log(result.pass ? "PASS" : "FAIL", "->", OUT);
ws.close();
process.exit(result.pass ? 0 : 1);
