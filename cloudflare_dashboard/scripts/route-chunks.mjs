// Post-build step: writes dist/client/route-chunks.json = { "<route key>": ["/_next/static/chunks/View-hash.js", ...] }.
// The Worker (worker/route-preload.ts) turns the entry for the requested page into `Link: <...>; rel=modulepreload` headers on the HTML
// response, so the browser fetches the view's code in parallel with the shell instead of only after hydration asks for it.
// Reads dist/client/.vite/manifest.json (Vite's own output) and the `loaders` table in app/DashboardApp.tsx (route key -> view file).
// Shell chunks (everything the browser entry and DashboardApp already import statically) are left out: they are modulepreloaded by the HTML anyway.
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const MANIFEST = join(ROOT, "dist", "client", ".vite", "manifest.json");
const OUT = join(ROOT, "dist", "client", "route-chunks.json");
const MAX_PER_ROUTE = 12;

export function buildRouteChunks(manifest, dashboardSource) {
  const closure = (key, seen = new Set()) => {
    if (!key || seen.has(key) || !manifest[key]) return seen;
    seen.add(key);
    for (const next of manifest[key].imports ?? []) closure(next, seen);
    return seen;
  };
  const shell = new Set([...closure("virtual:vinext-app-browser-entry"), ...closure("app/DashboardApp.tsx")]);
  const block = dashboardSource.match(/const loaders:[^\n]*=\s*\{([\s\S]*?)\n\};/)?.[1] ?? "";
  const routes = {};
  for (const m of block.matchAll(/"?([a-z-]+)"?\s*:\s*\(\)\s*=>\s*import\("\.\/([A-Za-z]+)"\)/g)) {
    const key = `app/${m[2]}.tsx`;
    const entry = manifest[key] ?? manifest[`app/${m[2]}.ts`];
    if (!entry) continue;
    const own = [...closure(manifest[key] ? key : `app/${m[2]}.ts`)].filter((k) => !shell.has(k));
    // view first, then its dependencies, each once
    const files = [...new Set(own.map((k) => manifest[k].file))].filter(Boolean).slice(0, MAX_PER_ROUTE).map((f) => `/${f}`);
    // the stylesheets those chunks bring: without them preloaded code can render before its CSS is applied (a layout shift)
    const css = [...new Set(own.flatMap((k) => manifest[k].css ?? []))].map((f) => `/${f}`);
    routes[m[1]] = [...files, ...css];
  }
  return routes;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (!existsSync(MANIFEST)) { console.log("route-chunks: no dist/client/.vite/manifest.json, skipped"); process.exit(0); }
  const routes = buildRouteChunks(JSON.parse(readFileSync(MANIFEST, "utf8")), readFileSync(join(ROOT, "app", "DashboardApp.tsx"), "utf8"));
  writeFileSync(OUT, JSON.stringify(routes));
  console.log(`route-chunks: ${Object.keys(routes).length} routes -> dist/client/route-chunks.json`);
}
