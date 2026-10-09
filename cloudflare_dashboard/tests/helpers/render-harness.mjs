// Server-render real app TSX against fixture data without a browser or a build.
// Relative .ts/.tsx imports are transpiled on the fly; css is stubbed; `./data` (useDashboardData) is replaced
// by a synchronous lookup so the first render shows loaded, error or empty states exactly as users would see them.
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";
import { existsSync, readFileSync } from "node:fs";
import { createRequire, Module } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const APP = fileURLToPath(new URL("../../app/", import.meta.url));

function resolveFile(from, id) {
  const base = path.resolve(path.dirname(from), id);
  for (const candidate of [base, `${base}.ts`, `${base}.tsx`, path.join(base, "index.ts"), path.join(base, "index.tsx")]) {
    if (existsSync(candidate) && /\.(tsx?)$/.test(candidate)) return candidate;
  }
  return null;
}

/** data: object mapping dashboard data keys to payloads, or a function (key) => payload | undefined. Missing keys return a 404-style error. */
export function loadApp(file, { data = {}, loading = false } = {}) {
  const lookup = typeof data === "function" ? data : (key) => data[key];
  const cache = new Map();
  const dataHook = {
    useDashboardData: (key) => {
      if (loading) return { data: null, loading: true, error: null };
      const payload = lookup(key);
      return payload === undefined ? { data: null, loading: false, error: "Data request failed (404)" } : { data: payload, loading: false, error: null };
    },
  };
  const load = (filename) => {
    if (cache.has(filename)) return cache.get(filename).exports;
    const source = readFileSync(filename, "utf8");
    const compiled = ts.transpileModule(source, { fileName: filename, compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022, esModuleInterop: true } }).outputText;
    const mod = new Module(filename);
    mod.filename = filename;
    cache.set(filename, mod);
    mod.require = (id) => {
      if (id.endsWith(".css")) return {};
      if (id.startsWith(".")) {
        const resolved = resolveFile(filename, id);
        if (!resolved) return require(path.resolve(path.dirname(filename), id));
        if (resolved === path.join(APP, "data.ts")) return dataHook;
        return load(resolved);
      }
      if (id === "next/link") return { __esModule: true, default: ({ href, children, ...rest }) => React.createElement("a", { href: typeof href === "string" ? href : "#", ...rest }, children) };
      if (id === "next/navigation") return { usePathname: () => "/", useRouter: () => ({ push() {}, replace() {} }), useSearchParams: () => new URLSearchParams() };
      return require(id);
    };
    mod._compile(compiled, filename);
    return mod.exports;
  };
  return load(path.join(APP, file));
}

/** Render `exportName` of app/<file> to static HTML. `search` fakes window.location.search for components that read the URL. */
export function renderView(file, exportName, { data, loading, search = "", pathname = "/", props = {} } = {}) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, "window");
  globalThis.window = {
    location: { search, pathname, hash: "" },
    history: { replaceState() {} },
    addEventListener() {},
    removeEventListener() {},
    matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }),
  };
  try {
    const exports = loadApp(file, { data, loading });
    return renderToStaticMarkup(React.createElement(exports[exportName], props));
  } finally {
    if (previous) Object.defineProperty(globalThis, "window", previous);
    else delete globalThis.window;
  }
}
