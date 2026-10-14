// Bundle report for a finished build in dist/. Usage: node scripts/measure-bundle.mjs [--json] [route ...]
// For each route the built Worker renders the real HTML; the initial JS is every script / modulepreload in it plus the static-import
// closure of those chunks (dynamic import() edges are NOT followed: those are the lazy chunks). CSS is every stylesheet link.
// Exported helpers are reused by tests/budgets.test.mjs.
import { readFile, readdir, stat } from "node:fs/promises";
import { existsSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { fileURLToPath, pathToFileURL } from "node:url";
import { join, resolve } from "node:path";

export const ROOT = resolve(fileURLToPath(new URL("..", import.meta.url)));
export const CLIENT = join(ROOT, "dist", "client");
export const DEFAULT_ROUTES = ["/", "/this-week", "/weekly-players", "/why-priced", "/players", "/distributions", "/inputs", "/run", "/performance"];

const kb = (n) => `${(n / 1024).toFixed(1)} KB`;

export async function renderHtml(path) {
  const workerUrl = pathToFileURL(join(ROOT, "dist", "server", "index.js"));
  workerUrl.searchParams.set("measure", `${process.pid}-${Date.now()}-${Math.random()}`);
  const worker = (await import(workerUrl.href)).default;
  const env = { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } };
  const response = await worker.fetch(new Request(`https://golf.example${path}`, { headers: { accept: "text/html" } }), env, { waitUntil() {}, passThroughOnException() {} });
  return response.text();
}

const fileCache = new Map();
async function sizes(urlPath) {
  const clean = urlPath.split("?")[0].replace(/^\//, "");
  if (fileCache.has(clean)) return fileCache.get(clean);
  const file = join(CLIENT, clean);
  const buf = existsSync(file) ? await readFile(file) : null;
  const info = buf ? { path: clean, raw: buf.length, gzip: gzipSync(buf, { level: 9 }).length, text: clean.endsWith(".js") ? buf.toString("utf8") : "" } : { path: clean, raw: 0, gzip: 0, text: "", missing: true };
  fileCache.set(clean, info);
  return info;
}

/** Static imports of a built chunk: `import ... from "./x.js"`, `import "./x.js"` and `export ... from "./x.js"`; not `import("./x.js")`. */
function staticImports(text, fromPath) {
  const dir = fromPath.includes("/") ? fromPath.slice(0, fromPath.lastIndexOf("/")) : "";
  const out = new Set();
  for (const m of text.matchAll(/(?:^|[;}\s])(?:import|export)\s*(?:[^"'()]*?[\s}*,]from\s*)?["'](\.{1,2}\/[^"']+\.js)["']/g)) {
    out.add(resolve("/", dir, m[1]).replace(/\\/g, "/").replace(/^[A-Za-z]:/, "").replace(/^\//, ""));
  }
  return out;
}

/** view key -> source file base name ("this-week" -> "ExplainViews"), read from the lazy() declarations and the views registry. */
export async function viewFiles() {
  const src = await readFile(join(ROOT, "app", "DashboardApp.tsx"), "utf8");
  const fileOf = new Map([...src.matchAll(/const (\w+) = lazy\(\(\) => import\("\.\/([\w-]+)"\)/g)].map((m) => [m[1], m[2]]));
  return new Map([...src.matchAll(/^\s+"?([a-z-]+)"?: ([A-Za-z]+View),$/gm)].map((m) => [m[1], fileOf.get(m[2])]).filter((e) => e[1]));
}

async function closure(seedPaths, seen = new Map()) {
  const queue = [...seedPaths];
  while (queue.length) {
    const next = queue.pop();
    if (seen.has(next)) continue;
    const info = await sizes(next);
    seen.set(next, info);
    if (info.text) for (const dep of staticImports(info.text, next)) if (!seen.has(dep)) queue.push(dep);
  }
  return seen;
}

export async function measureRoute(path) {
  const html = await renderHtml(path);
  const seeds = new Set();
  const css = new Set();
  for (const m of html.matchAll(/<(?:script|link)\b[^>]*>/g)) {
    const tag = m[0];
    const url = /(?:src|href)="([^"]+)"/.exec(tag)?.[1];
    if (!url || !url.startsWith("/_next/")) continue;
    if (/rel="stylesheet"/.test(tag)) css.add(url.replace(/^\//, ""));
    else if (/\.js(\?|$)/.test(url)) seeds.add(url.replace(/^\//, ""));
  }
  const seen = await closure(seeds);
  const js = [...seen.values()];
  // The view itself is a lazy chunk fetched right after hydration: add its static closure (and CSS) for the "with view" figure.
  const key = path === "/" ? "this-week" : path.slice(1);
  const viewFile = (await viewFiles()).get(key);
  const withView = new Map(seen);
  if (viewFile) {
    const chunkDir = join(CLIENT, "_next", "static", "chunks");
    const entry = (await readdir(chunkDir)).find((f) => f.startsWith(`${viewFile}-`) && f.endsWith(".js"));
    if (entry) await closure([`_next/static/chunks/${entry}`], withView);
  }
  const viewJs = [...withView.values()];
  const styles = await Promise.all([...css].map(sizes));
  const sum = (list, key) => list.reduce((a, f) => a + f[key], 0);
  return {
    route: path,
    jsFiles: js.map(({ path: p, raw, gzip }) => ({ path: p, raw, gzip })).sort((a, b) => b.gzip - a.gzip),
    jsRaw: sum(js, "raw"), jsGzip: sum(js, "gzip"),
    withViewRaw: sum(viewJs, "raw"), withViewGzip: sum(viewJs, "gzip"), viewFile: viewFile ?? null,
    viewOnlyFiles: viewJs.filter((f) => !seen.has(f.path)).map(({ path: p, gzip }) => ({ path: p, gzip })).sort((a, b) => b.gzip - a.gzip),
    cssFiles: styles.map(({ path: p, raw, gzip }) => ({ path: p, raw, gzip })),
    cssRaw: sum(styles, "raw"), cssGzip: sum(styles, "gzip"),
    missing: [...js, ...styles].filter((f) => f.missing).map((f) => f.path),
  };
}

/** Every built JS chunk (lazy ones included) with sizes, plus duplicate-name detection (same chunk name, two hashes). */
export async function measureChunks() {
  const dir = join(CLIENT, "_next", "static", "chunks");
  const files = (await readdir(dir)).filter((f) => f.endsWith(".js"));
  const chunks = [];
  for (const f of files) {
    const buf = await readFile(join(dir, f));
    chunks.push({ file: f, raw: buf.length, gzip: gzipSync(buf, { level: 9 }).length });
  }
  const byName = new Map();
  for (const c of chunks) {
    const name = c.file.replace(/[-.][A-Za-z0-9_-]{8}\.js$/, "");
    byName.set(name, [...(byName.get(name) ?? []), c.file]);
  }
  const duplicates = [...byName].filter(([, list]) => list.length > 1).map(([name, list]) => ({ name, files: list }));
  return { chunks: chunks.sort((a, b) => b.gzip - a.gzip), duplicates };
}

export async function measureAllCss() {
  const dir = join(CLIENT, "_next", "static", "css");
  const out = [];
  if (!existsSync(dir)) return out;
  for (const f of await readdir(dir)) {
    if (!f.endsWith(".css")) continue;
    const buf = await readFile(join(dir, f));
    out.push({ file: f, raw: buf.length, gzip: gzipSync(buf, { level: 9 }).length });
  }
  return out;
}

async function main() {
  const args = process.argv.slice(2);
  const json = args.includes("--json");
  const routes = args.filter((a) => a.startsWith("/"));
  try { await stat(join(ROOT, "dist", "server", "index.js")); } catch { console.error("No build in dist/. Run: npx cross-env RESEARCH_LOCAL_PREVIEW=1 npm run build"); process.exit(2); }
  const results = [];
  for (const route of routes.length ? routes : DEFAULT_ROUTES) results.push(await measureRoute(route));
  const { chunks, duplicates } = await measureChunks();
  const css = await measureAllCss();
  if (json) { console.log(JSON.stringify({ results, chunks, duplicates, css }, null, 2)); return; }
  console.log("Initial payload per route (JS = scripts + modulepreloads + static-import closure; CSS = stylesheets in the HTML)");
  console.log("route".padEnd(18) + "JS gzip".padStart(11) + "JS raw".padStart(11) + "+view gz".padStart(11) + "+view raw".padStart(11) + "CSS gzip".padStart(11) + "CSS raw".padStart(11));
  for (const r of results) console.log(r.route.padEnd(18) + kb(r.jsGzip).padStart(11) + kb(r.jsRaw).padStart(11) + kb(r.withViewGzip).padStart(11) + kb(r.withViewRaw).padStart(11) + kb(r.cssGzip).padStart(11) + kb(r.cssRaw).padStart(11));
  console.log("\nLargest chunks (gzip)");
  for (const c of chunks.slice(0, 12)) console.log(`  ${c.file.padEnd(48)} ${kb(c.gzip).padStart(10)} gz ${kb(c.raw).padStart(10)} raw`);
  console.log(`\nAll built JS chunks: ${chunks.length}, total ${kb(chunks.reduce((a, c) => a + c.gzip, 0))} gzip`);
  console.log("\nAll CSS files");
  for (const c of css) console.log(`  ${c.file.padEnd(40)} ${kb(c.gzip).padStart(10)} gz ${kb(c.raw).padStart(10)} raw`);
  console.log(duplicates.length ? `\nDUPLICATE chunk names (stale output?): ${JSON.stringify(duplicates)}` : "\nNo duplicate chunk names.");
  const missing = results.flatMap((r) => r.missing);
  if (missing.length) console.log(`\nReferenced but missing from dist/client: ${[...new Set(missing)].join(", ")}`);
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) await main();
