// Two mounted consumers of one key must agree: a saved-copy fallback seen by one is seen by the other (header badge never "live" while data is saved).
import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { Module, createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";
import ts from "typescript";
import { badgeView } from "../app/ui-rules.ts";

const require = createRequire(import.meta.url);
const APP = fileURLToPath(new URL("../app/", import.meta.url));

/** Minimal hook runtime: enough of React for useDashboardData, with effects run after each render and microtask-batched re-renders. */
function makeFakeReact() {
  const mounted = new Set();
  const fake = {
    useState(init) {
      const c = fake.current;
      const i = c.i++;
      if (!(i in c.slots)) c.slots[i] = { v: typeof init === "function" ? init() : init };
      const slot = c.slots[i];
      return [slot.v, (next) => { const v = typeof next === "function" ? next(slot.v) : next; if (Object.is(v, slot.v)) return; slot.v = v; c.schedule(); }];
    },
    useRef(init) { const c = fake.current; const i = c.i++; return (c.slots[i] ??= { current: init }); },
    useCallback(fn) { return fn; },
    useEffect(fn, deps) {
      const c = fake.current;
      const i = c.i++;
      const prev = c.slots[i];
      if (prev && deps && prev.deps.length === deps.length && prev.deps.every((d, k) => Object.is(d, deps[k]))) return;
      c.effects.push(() => { prev?.cleanup?.(); const cleanup = fn(); c.slots[i] = { deps: deps ?? [], cleanup }; });
      c.slots[i] ??= { deps: null, cleanup: null };
    },
    useSyncExternalStore(_sub, get) { fake.current.i++; return get(); },
    current: null,
    mount(hook) {
      const c = { slots: {}, i: 0, effects: [], result: null, pending: false };
      const render = () => { c.i = 0; c.effects = []; fake.current = c; c.result = hook(); fake.current = null; c.effects.forEach((e) => e()); };
      c.schedule = () => { if (c.pending) return; c.pending = true; queueMicrotask(() => { c.pending = false; render(); }); };
      render();
      mounted.add(c);
      return { get current() { return c.result; }, unmount() { Object.values(c.slots).forEach((s) => s?.cleanup?.()); mounted.delete(c); } };
    },
  };
  return fake;
}

function loadData(fakeReact) {
  const modules = new Map();
  const load = (filename) => {
    if (modules.has(filename)) return modules.get(filename).exports;
    const out = ts.transpileModule(readFileSync(filename, "utf8"), { fileName: filename, compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true } }).outputText;
    const mod = new Module(filename);
    mod.filename = filename;
    modules.set(filename, mod);
    mod.require = (id) => {
      if (id === "react") return fakeReact;
      if (id.startsWith(".")) return load(path.resolve(path.dirname(filename), id.endsWith(".ts") ? id : `${id}.ts`));
      return require(id);
    };
    mod._compile(out, filename);
    return mod.exports;
  };
  return load(path.join(APP, "data.ts"));
}

const settle = (ms = 50) => new Promise((resolve) => setTimeout(resolve, ms));

test("a fallback to a saved copy in one consumer reaches the already-mounted consumer (header badge stops saying live)", async (t) => {
  const realFetch = globalThis.fetch;
  t.after(() => { globalThis.fetch = realFetch; delete globalThis.caches; });
  const KEY = "golfprice/index.json";
  let outage = false;
  const body = JSON.stringify({ events: [] });
  globalThis.fetch = async () => {
    if (outage) return { ok: false, status: 503, headers: new Headers(), text: async () => "{}" };
    return { ok: true, status: 200, headers: new Headers({ etag: "v1" }), text: async () => body };
  };
  const files = new Map();
  globalThis.caches = { open: async () => ({ match: async (u) => (files.has(String(u)) ? new Response(files.get(String(u))) : undefined), put: async (u, r) => { files.set(String(u), await r.text()); } }) };
  t.mock.timers.enable({ apis: ["Date"], now: Date.parse("2026-10-29T12:00:00Z") });

  const fake = makeFakeReact();
  const { useDashboardData } = loadData(fake);
  const header = fake.mount(() => useDashboardData(KEY));
  await settle();
  assert.equal(header.current.stale, false);
  assert.equal(badgeView(3 * 60_000, "Run", "run", header.current.stale).tone, "live");

  outage = true;
  t.mock.timers.tick(3 * 60_000);
  const view = fake.mount(() => useDashboardData(KEY));
  await settle(3000);
  assert.equal(view.current.stale, true, "the second consumer got the saved copy");
  assert.equal(header.current.stale, true, "the mounted header follows the shared cache");
  assert.notEqual(badgeView(3 * 60_000, "Run", "run", header.current.stale).tone, "live");
  assert.match(badgeView(3 * 60_000, "Run", "run", header.current.stale).text, /Saved copy/);
  header.unmount();
  view.unmount();
});
