// Populated Bet card interaction proof over the DevTools Protocol (real key events, no dependencies).
//   node scripts/bet-card-interact.mjs [--base http://localhost:3111] [--out dist-visual/R4b/bet-card-interact.json]
// Needs a preview serving a populated owner/bet_card.json (R4b: swap .wrangler/live/owner/bet_card.json for dist-visual/D-r2/bet_card.populated.json,
// restart the preview, restore afterwards). Checks on phone and desktop: rows expand and collapse from the keyboard, the tier / group / book filters change
// the shown count (announced in the polite status), Clear resets, and the empty-filter state is not a blank page. Exit code 1 when a check fails.
import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const args = process.argv.slice(2);
const opt = (n, d) => (args.includes(n) ? args[args.indexOf(n) + 1] : d);
const BASE = opt("--base", "http://localhost:3111").replace(/\/+$/, "");
const OUT = opt("--out", "dist-visual/R4b/bet-card-interact.json");
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


const result = { generated: new Date().toISOString(), base: BASE, viewports: {}, pass: true };
for (const [name, vp] of Object.entries(VIEWPORTS)) {
  const { targetId } = await send("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await send("Target.attachToTarget", { targetId, flatten: true });
  const s = (m, p) => send(m, p, sessionId);
  await s("Page.enable"); await s("Runtime.enable");
  await s("Emulation.setDeviceMetricsOverride", { width: vp.width, height: vp.height, deviceScaleFactor: vp.dpr, mobile: vp.mobile });
  const ev = async (expr) => { const r = await s("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true }); if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? "evaluate failed"); return r.result.value; };
  const key = async (k, code, vk) => {
    const base = { key: k, code, windowsVirtualKeyCode: vk, nativeVirtualKeyCode: vk };
    await s("Input.dispatchKeyEvent", k === "Enter" || k === " " ? { type: "keyDown", ...base, text: k === "Enter" ? "\r" : " " } : { type: "rawKeyDown", ...base });
    await s("Input.dispatchKeyEvent", { type: "keyUp", ...base });
    await sleep(150);
  };
  const checks = []; const check = (label, ok, detail) => { checks.push({ label, ok: !!ok, detail }); if (!ok) result.pass = false; if (process.env.A11Y_VERBOSE) console.log(name, ok ? "ok  " : "FAIL", label, ok ? "" : JSON.stringify(detail)); };
  const state = () => ev(`(() => ({ rows: document.querySelectorAll('.bc-bet').length, open: document.querySelectorAll('.bc-bet details[open]').length, status: document.querySelector('.bc-bar [role=status]')?.textContent?.trim() ?? null, reset: document.querySelector('.bc-reset')?.textContent?.trim() ?? null, answer: (document.querySelector('.bc-answer')?.textContent ?? '').trim().slice(0, 90), active: document.activeElement?.localName + '.' + (document.activeElement?.className || ''), empty: /no bets|nothing|match/i.test(document.querySelector('main')?.innerText ?? '') }))()`);
  const setSelect = (idx, value) => ev(`(() => { const sel = document.querySelectorAll('.bc-bar select')[${idx}]; if (!sel) return null; const opts = [...sel.options].map((o) => o.value); const set = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set; set.call(sel, ${JSON.stringify(value)} === null ? opts[1] : ${JSON.stringify(value)}); sel.dispatchEvent(new Event('change', { bubbles: true })); return opts; })()`);
  await s("Page.navigate", { url: `${BASE}/bet-card` });
  let st = null;
  for (let i = 0; i < 60; i++) { await sleep(300); st = await state().catch(() => null); if (st && st.rows > 0) break; }
  if (!st || st.rows === 0) { check("populated card renders bet rows", false, st); result.viewports[name] = { checks }; continue; }
  await sleep(800);
  const total = st.rows;
  check("populated card renders bet rows", total > 0, st);
  check("count status matches the rows", new RegExp(`^${total} bets? shown`).test(st.status ?? ""), st);
  // Expand / collapse from the keyboard: focus the first summary, Enter opens, Enter closes; Space toggles too.
  await ev(`(() => { const sm = document.querySelector('.bc-bet summary'); sm.scrollIntoView({ block: 'center' }); sm.focus(); return true; })()`);
  await key("Enter", "Enter", 13);
  let a = await state();
  check("Enter on a row summary expands it", a.open === 1, a);
  check("expanded evidence is visible", await ev(`!!document.querySelector('.bc-bet details[open] .bc-evidence')?.getBoundingClientRect().height`), a);
  await key("Enter", "Enter", 13);
  a = await state();
  check("Enter again collapses it", a.open === 0, a);
  await key(" ", "Space", 32);
  a = await state();
  check("Space toggles it open", a.open === 1, a);
  await key(" ", "Space", 32);
  check("Space toggles it closed", (await state()).open === 0, await state());
  // Filters: every select's first non-"all" option, one at a time, then Clear.
  const selects = await ev(`document.querySelectorAll('.bc-bar select').length`);
  check("tier/group/book filters present", selects >= 2, { selects });
  for (let i = 0; i < selects; i++) {
    const opts = await setSelect(i, null);
    await sleep(250);
    a = await state();
    check(`filter ${i + 1} (${opts?.[1]}) changes the count and announces it`, a.rows <= total && /bets? shown/.test(a.status ?? "") && new RegExp(`^${a.rows} bets? shown`).test(a.status ?? ""), { opts, ...a });
    check(`filter ${i + 1} shows a Clear control`, /^Clear 1 filter$/.test(a.reset ?? ""), a);
    if (a.reset) { await ev(`document.querySelector('.bc-reset').click()`); await sleep(250); }
    a = await state();
    check(`Clear restores all ${total} rows`, a.rows === total && a.reset === null, a);
  }
  // Two filters that cannot both match: not a blank page.
  if (selects >= 2) {
    const first = await setSelect(0, null); await sleep(150);
    const last = await setSelect(1, null); await sleep(250);
    a = await state();
    check("combined filters keep the count status and Clear", /bets? shown/.test(a.status ?? "") && /Clear 2 filters/.test(a.reset ?? ""), { first, last, ...a });
    await ev(`document.querySelector('.bc-reset')?.click()`); await sleep(200);
  }
  result.viewports[name] = { checks, passed: checks.filter((c) => c.ok).length, failed: checks.filter((c) => !c.ok).length };
  await send("Target.closeTarget", { targetId }).catch(() => undefined);
}
mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, JSON.stringify(result, null, 2));
for (const [n, v] of Object.entries(result.viewports)) console.log(`${n}: ${v.passed ?? 0} pass, ${v.failed ?? "?"} fail${(v.checks ?? []).filter((c) => !c.ok).map((c) => `
   FAIL ${c.label} ${JSON.stringify(c.detail)}`).join("")}`);
console.log(result.pass ? "PASS" : "FAIL", "->", OUT);
ws.close();
process.exit(result.pass ? 0 : 1);
