// One-command LOCAL preview of the private dashboard with this week's LIVE data, acting as the owner.
//
//   node scripts/ui-preview.mjs                 build if needed, serve in the foreground on http://localhost:3111
//   node scripts/ui-preview.mjs --detach        same, but in the background (log: .wrangler/preview/server.log, pid: .wrangler/preview/server.pid)
//   node scripts/ui-preview.mjs --stop          stop a detached preview
//   node scripts/ui-preview.mjs --status        say whether it is running
// Options:
//   --refresh      re-pull the live R2 objects (READ-ONLY `wrangler r2 object get --remote`) into .wrangler/live/ first
//   --all-events   with --refresh, also pull explain/run/weather files of past events (default: current events + every event's history)
//   --no-build     reuse dist/ (default: build when dist/server is missing or older than app/, worker/ or package.json)
//   --build        force `npm run build`
//   --port <n>     default 3111
//   then e.g.:  node scripts/visual-check.mjs --base http://localhost:3111
//
// How it works. `vinext start` has no bindings, and `wrangler dev` cannot be the owner (the Worker only trusts an Access JWT whose RS256
// signature verifies against https://<team>/cdn-cgi/access/certs, which workerd cannot be pointed at a local key). So, like
// scripts/ops-preview.mjs, this loads the built Worker (dist/server/index.js) in Node, signs a throw-away owner identity with a generated key,
// answers the Access cert fetch locally, and gives the Worker a DASHBOARD_DATA bucket shim backed by the files in .wrangler/live/ (key == path).
// Writes (overrides, job queue, derived search index) go to an in-memory overlay and vanish on restart. Remote R2/D1 are never written; the only
// remote calls are `r2 object get --remote` reads under --refresh. Production auth code and config are untouched. No secrets are read or printed.
import { spawn, spawnSync } from "node:child_process";
import { createSign, generateKeyPairSync } from "node:crypto";
import { brotliCompressSync, constants as zc, gzipSync } from "node:zlib";
import { existsSync, mkdirSync, openSync, readFileSync, readdirSync, statSync, writeFileSync, rmSync } from "node:fs";
import http from "node:http";
import { dirname, extname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(fileURLToPath(new URL("..", import.meta.url)));
const LIVE = join(ROOT, ".wrangler", "live");
const PREVIEW = join(ROOT, ".wrangler", "preview");
const PID_FILE = join(PREVIEW, "server.pid");
const LOG_FILE = join(PREVIEW, "server.log");
const BUCKET = "golf-model-dashboard-data";
const WRANGLER = join(ROOT, "node_modules", "wrangler", "bin", "wrangler.js");

const args = process.argv.slice(2);
const flag = (n) => args.includes(n);
const opt = (n, d) => (args.includes(n) ? args[args.indexOf(n) + 1] : d);
const PORT = Number(opt("--port", 3111));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const log = (...a) => console.log("[ui-preview]", ...a);

/* ------------------------------------------------------------------ stop / status / detach */
const alive = (pid) => { try { process.kill(pid, 0); return true; } catch { return false; } };
function runningPid() {
  try { const pid = Number(readFileSync(PID_FILE, "utf8").trim()); process.kill(pid, 0); return pid; } catch { return null; }
}
if (flag("--status")) { const pid = runningPid(); console.log(pid ? `running (pid ${pid}) http://localhost:${PORT}` : "not running"); process.exit(0); }
if (flag("--stop")) {
  const pid = runningPid();
  if (!pid) { console.log("not running"); process.exit(0); }
  if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(pid), "/T", "/F"], { stdio: "ignore" }); else process.kill(pid, "SIGTERM");
  rmSync(PID_FILE, { force: true });
  console.log(`stopped pid ${pid}`); process.exit(0);
}
if (flag("--detach") && !process.env.UI_PREVIEW_CHILD) {
  const existing = runningPid();
  if (existing) { console.log(`already running (pid ${existing}) http://localhost:${PORT}; use --stop first to restart`); process.exit(0); }
  mkdirSync(PREVIEW, { recursive: true });
  const out = openSync(LOG_FILE, "w");
  const child = spawn(process.execPath, [fileURLToPath(import.meta.url), ...args.filter((a) => a !== "--detach")], { cwd: ROOT, detached: true, stdio: ["ignore", out, out], windowsHide: true, env: { ...process.env, UI_PREVIEW_CHILD: "1" } });
  child.unref();
  log(`starting in background (pid ${child.pid}), log ${LOG_FILE}`);
  for (let i = 0; i < 1200; i++) {
    await sleep(500);
    try { if ((await fetch(`http://127.0.0.1:${PORT}/api/whoami`)).ok) { console.log(`ready: http://localhost:${PORT}  stop with: node scripts/ui-preview.mjs --stop`); process.exit(0); } } catch { /* not up yet */ }
    if (!alive(child.pid)) { console.error("preview exited early; see " + LOG_FILE); process.exit(1); }
  }
  console.error("timed out waiting for the preview; see " + LOG_FILE); process.exit(1);
}

/* ------------------------------------------------------------------ refresh: pull live objects (read-only) */
const mapPool = async (items, n, fn) => { let i = 0; await Promise.all(Array.from({ length: n }, async () => { while (i < items.length) await fn(items[i++]); })); };
const localPath = (key) => join(LIVE, ...key.split("/"));
const have = (key) => existsSync(localPath(key));
const readJson = (key) => { try { return JSON.parse(readFileSync(localPath(key), "utf8")); } catch { return null; } };

function fetchKey(key) {
  return new Promise((done) => {
    mkdirSync(dirname(localPath(key)), { recursive: true });
    const tmp = localPath(key) + ".part";
    rmSync(tmp, { force: true });
    const c = spawn(process.execPath, [WRANGLER, "r2", "object", "get", `${BUCKET}/${key}`, "--remote", "--file", tmp], { cwd: ROOT, stdio: "ignore", windowsHide: true });
    c.on("exit", () => {
      if (existsSync(tmp) && statSync(tmp).size > 0) { rmSync(localPath(key), { force: true }); writeFileSync(localPath(key), readFileSync(tmp)); rmSync(tmp, { force: true }); done(true); }
      else { rmSync(tmp, { force: true }); done(false); }
    });
  });
}

async function pull(keys, label) {
  const todo = [...new Set(keys)];
  let ok = 0; const missing = [];
  await mapPool(todo, 6, async (k) => { if (await fetchKey(k)) ok++; else missing.push(k); });
  log(`${label}: ${ok}/${todo.length} pulled${missing.length ? `, ${missing.length} not published (${missing.slice(0, 4).join(", ")}${missing.length > 4 ? ", ..." : ""})` : ""}`);
}

const stringsUnder = (value, prefix, out = new Set()) => {
  if (typeof value === "string") { if (value.startsWith(prefix) && value.endsWith(".json")) out.add(value); }
  else if (Array.isArray(value)) value.forEach((v) => stringsUnder(v, prefix, out));
  else if (value && typeof value === "object") Object.values(value).forEach((v) => stringsUnder(v, prefix, out));
  return out;
};
const dirOf = (uid) => uid.replace(/:/g, "_");

async function refresh() {
  mkdirSync(LIVE, { recursive: true });
  log("refreshing live objects (read-only)...");
  await pull([
    "golfprice/index.json", "golfprice/manifest.json", "golfprice/feature_glossary.json", "golfprice/overrides_schema.json", "golfprice/odds_signals/latest.json",
    "golfprice/scorecard/latest.json", "golfprice/player_profiles/index.json", "golfprice/player_profiles/fields.json", "golfprice/player_profiles/dispersion/index.json",
    "golfprice/course_geometry/index.json", "golfprice/changes.json", "data/manifest.json",
    "owner/bet_card.json", "owner/data_freshness.json", "owner/bet_breakdowns.json", "owner/ops/observations.json", "owner/shot-monitor/collectors-desktop.json", "owner/shot-monitor/publications.json",
    "overrides/active.json", "jobs/queue.json", "jobs/roles.json", "jobs/alerts/state.json", "jobs/heartbeat/mckinley_home.json", "jobs/heartbeat/desktop-2ki41v6.json", "ops/registry.json", "jobs/retention/last_report.json",
  ], "pointers and owner files");

  const index = readJson("golfprice/index.json");
  const manifest = readJson("golfprice/manifest.json");
  if (!index?.events) throw new Error("golfprice/index.json did not download; is wrangler logged in (npx wrangler whoami)?");
  const cutoff = new Date(Date.now() - 4 * 86_400_000).toISOString().slice(0, 10);
  const current = index.events.filter((e) => (e.date_end ?? e.date_start ?? "") >= cutoff);
  log(`current events: ${current.map((e) => `${e.name} (${e.event_uid})`).join("; ") || "none"}`);
  const manifestKeys = Object.keys(manifest?.objects ?? {});
  const keys = [];
  if (index.scorecard_key) keys.push(index.scorecard_key);
  if (index.feature_glossary_key) keys.push(index.feature_glossary_key);
  for (const e of index.events) {
    if (e.history_key) keys.push(e.history_key);
    const isCurrent = current.includes(e) || flag("--all-events");
    if (!isCurrent) continue;
    for (const [k, v] of Object.entries(e)) if (k.endsWith("_key") && typeof v === "string" && v.startsWith("golfprice/")) keys.push(v);
    for (const r of (e.distribution_runs ?? []).slice(-3)) if (r.key) keys.push(r.key);
    const d = dirOf(e.event_uid);
    const lastRuns = [...new Set(manifestKeys.filter((k) => k.startsWith(`golfprice/runs/${d}/`)).map((k) => k.split("/")[3]))].sort().slice(-2);
    for (const k of manifestKeys) {
      if (k.startsWith(`golfprice/weather/${d}/`)) keys.push(k);
      else if (k.startsWith(`golfprice/runs/${d}/`) && lastRuns.includes(k.split("/")[3])) keys.push(k);
      else if (k === `golfprice/explain/${d}/latest.json`) keys.push(k);
    }
  }
  await pull(keys, "event files");

  const extra = new Set();
  for (const k of ["golfprice/course_geometry/index.json"]) { const j = readJson(k); if (j) stringsUnder(j, "golfprice/course_geometry/", extra); }
  const disp = readJson("golfprice/player_profiles/dispersion/index.json"); if (disp) stringsUnder(disp, "golfprice/player_profiles/dispersion/", extra);
  if (extra.size) await pull([...extra], "course geometry and dispersion files");

  const catalog = readJson("golfprice/player_profiles/index.json");
  const ids = new Set(current.flatMap((e) => e.player_ids ?? []));
  for (const f of readJson("golfprice/player_profiles/fields.json")?.fields ?? []) if (current.some((e) => e.event_uid === f.event_uid)) (f.player_ids ?? []).forEach((i) => ids.add(i));
  const batches = new Set(); 
  for (const p of catalog?.players ?? []) if (ids.has(p.dg_id)) { if (p.profile_key) batches.add(p.profile_key); if (p.deep_key) batches.add(p.deep_key); }
  await pull([...batches], `player profile batches for ${ids.size} field players`);
  const deep = new Set();
  for (const b of batches) { const j = readJson(b); if (j) stringsUnder(j, "golfprice/player_profiles/", deep); }
  for (const b of batches) deep.delete(b);
  if (deep.size) await pull([...deep], "referenced player-profile files");
  log(`refresh complete (${new Date().toISOString()})`);
  writeFileSync(join(LIVE, ".refreshed_at"), new Date().toISOString());
}

if (flag("--refresh") || !have("golfprice/index.json")) await refresh();
else log(`using cached live data from ${readFileSync(join(LIVE, ".refreshed_at"), "utf8").trim?.() ?? "an earlier run"} (use --refresh to re-pull)`);

/* ------------------------------------------------------------------ build */
const newest = (dir) => { let m = 0; const walk = (d) => { for (const e of readdirSync(d, { withFileTypes: true })) { const p = join(d, e.name); if (e.isDirectory()) { if (!["node_modules", ".next", "dist"].includes(e.name)) walk(p); } else m = Math.max(m, statSync(p).mtimeMs); } }; walk(dir); return m; };
const built = existsSync(join(ROOT, "dist", "server", "index.js")) ? statSync(join(ROOT, "dist", "server", "index.js")).mtimeMs : 0;
const stale = !built || [join(ROOT, "app"), join(ROOT, "worker"), join(ROOT, "public")].some((d) => existsSync(d) && newest(d) > built) || statSync(join(ROOT, "package.json")).mtimeMs > built;
if (flag("--build") || (stale && !flag("--no-build"))) {
  log("npm run build ...");
  const r = spawnSync("npm", ["run", "build"], { cwd: ROOT, stdio: "inherit", shell: true });
  if (r.status !== 0) { console.error("build failed"); process.exit(1); }
} else log("dist/ is current; skipping build (use --build to force)");

/* ------------------------------------------------------------------ serve: built Worker + live-file bucket + owner identity */
const TEAM = "ui-preview.cloudflareaccess.com", AUD = "ui-preview-aud", OWNER = "owner-preview@example.com";
const { publicKey, privateKey } = generateKeyPairSync("rsa", { modulusLength: 2048 });
const b64 = (v) => Buffer.from(JSON.stringify(v)).toString("base64url");
const ownerHeaders = () => {
  const h = b64({ alg: "RS256", kid: "preview-key" }), p = b64({ email: OWNER, aud: [AUD], iss: `https://${TEAM}`, exp: Math.floor(Date.now() / 1000) + 3600, sub: "preview-owner" });
  return { "cf-access-authenticated-user-email": OWNER, "cf-access-jwt-assertion": `${h}.${p}.${createSign("RSA-SHA256").update(`${h}.${p}`).sign(privateKey).toString("base64url")}` };
};
const realFetch = globalThis.fetch;
globalThis.fetch = async (input, init) => {
  const url = typeof input === "string" ? input : input.url ?? String(input);
  if (url === `https://${TEAM}/cdn-cgi/access/certs`) return Response.json({ keys: [{ ...publicKey.export({ format: "jwk" }), kid: "preview-key" }] });
  if (/^https?:\/\/(127\.0\.0\.1|localhost)/.test(url)) return realFetch(input, init);
  throw new Error("ui-preview refuses external network fetches: " + url);
};

/** Minimal R2 bucket: reads .wrangler/live/<key>; writes live only in memory. */
const overlay = new Map(); const deleted = new Set();
let etagN = 0;
const wrap = (key, buf, etag) => ({
  key, size: buf.length, etag, httpEtag: `"${etag}"`, uploaded: new Date(), httpMetadata: {}, customMetadata: {},
  body: new Response(buf).body, bodyUsed: false,
  text: async () => buf.toString("utf8"), json: async () => JSON.parse(buf.toString("utf8")), arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length),
  writeHttpMetadata(h) { h.set("content-type", "application/json"); },
});
const diskKeys = () => { const out = []; const walk = (d, pre) => { for (const e of readdirSync(d, { withFileTypes: true })) { if (e.name.startsWith(".")) continue; if (e.isDirectory()) walk(join(d, e.name), pre + e.name + "/"); else if (!e.name.endsWith(".part")) out.push(pre + e.name); } }; if (existsSync(LIVE)) walk(LIVE, ""); return out; };
const bucket = {
  async get(key) {
    if (overlay.has(key)) return wrap(key, overlay.get(key).buf, overlay.get(key).etag);
    if (deleted.has(key) || !have(key)) return null;
    const buf = readFileSync(localPath(key));
    return wrap(key, buf, `l${buf.length}-${Math.floor(statSync(localPath(key)).mtimeMs)}`);
  },
  async head(key) { const o = await this.get(key); return o && { ...o, body: undefined }; },
  async put(key, value, options) {
    const old = overlay.get(key);
    if (options?.onlyIf?.etagMatches && old?.etag !== options.onlyIf.etagMatches) return null;
    if (options?.onlyIf?.etagDoesNotMatch === "*" && (old || have(key))) return null;
    const buf = typeof value === "string" ? Buffer.from(value) : Buffer.from(await new Response(value).arrayBuffer());
    const etag = `p${++etagN}`; overlay.set(key, { buf, etag }); deleted.delete(key);
    return { key, etag, httpEtag: `"${etag}"`, size: buf.length };
  },
  async delete(keys) { for (const k of [].concat(keys)) { overlay.delete(k); deleted.add(k); } },
  async list({ prefix = "", limit = 1000, cursor } = {}) {
    const all = [...new Set([...diskKeys(), ...overlay.keys()])].filter((k) => k.startsWith(prefix) && !deleted.has(k)).sort();
    const start = cursor ? Number(cursor) : 0;
    const page = all.slice(start, start + limit);
    const objects = page.map((key) => ({ key, size: overlay.get(key)?.buf.length ?? (have(key) ? statSync(localPath(key)).size : 0), etag: "l", httpEtag: '"l"', uploaded: new Date() }));
    const truncated = start + limit < all.length;
    return { objects, delimitedPrefixes: [], truncated, ...(truncated ? { cursor: String(start + limit) } : {}) };
  },
};

const clientRoot = resolve(ROOT, "dist", "client");
const types = { ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png", ".woff2": "font/woff2", ".woff": "font/woff", ".ico": "image/x-icon", ".jpg": "image/jpeg", ".webp": "image/webp", ".txt": "text/plain" };
const staticFile = (pathname) => {
  let p; try { p = resolve(clientRoot, "." + decodeURIComponent(pathname)); } catch { return null; }
  if (!p.startsWith(clientRoot + sep) || !existsSync(p) || !statSync(p).isFile()) return null;
  return { body: readFileSync(p), type: types[extname(p)] ?? "application/octet-stream" };
};
const assets = { async fetch(request) { const f = staticFile(new URL(request.url).pathname); return f ? new Response(f.body, { headers: { "content-type": f.type } }) : new Response("not found", { status: 404 }); } };
const images = { input: (body) => ({ transform: () => ({ output: async () => ({ response: () => new Response(body) }) }) }) };

const worker = (await import(pathToUrl(join(ROOT, "dist", "server", "index.js")))).default;
function pathToUrl(p) { return new URL("file:///" + p.replace(/\\/g, "/")).href; }
const env = { DASHBOARD_DATA: bucket, ASSETS: assets, IMAGES: images, ACCESS_TEAM_DOMAIN: TEAM, ACCESS_AUD: AUD, OWNER_EMAILS: OWNER, LEGACY_ALERTS_ENABLED: "0", OPS_SHADOW_ENABLED: "0", DATA_FRESHNESS_ENABLED: "0", DATA_FRESHNESS_INBOX: "0" };

/** Like Cloudflare's edge, compress text responses (brotli, else gzip) so perf-check measures realistic transfer sizes. Static compressions are cached. */
const compressed = new Map();
function send(req, res, status, headers, body, cacheKey) {
  const type = String(headers["content-type"] ?? "");
  const accept = String(req.headers["accept-encoding"] ?? "");
  const enc = /(^|[\s,])br(\s|,|;|$)/.test(accept) ? "br" : /(^|[\s,])gzip(\s|,|;|$)/.test(accept) ? "gzip" : null;
  if (!enc || body.length < 1024 || !/text|json|javascript|xml|svg/.test(type) || process.env.UI_PREVIEW_NO_COMPRESS) { res.writeHead(status, headers); res.end(body); return; }
  const key = cacheKey ? `${enc}:${cacheKey}:${body.length}` : null;
  let out = key ? compressed.get(key) : undefined;
  if (!out) { out = enc === "br" ? brotliCompressSync(body, { params: { [zc.BROTLI_PARAM_QUALITY]: 5, [zc.BROTLI_PARAM_SIZE_HINT]: body.length } }) : gzipSync(body, { level: 6 }); if (key) compressed.set(key, out); }
  res.writeHead(status, { ...headers, "content-encoding": enc, vary: "accept-encoding" }); res.end(out);
}

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://${req.headers.host}`);
    const asStatic = url.pathname.startsWith("/_next/") || url.pathname.startsWith("/data/") || url.pathname === "/og.png" ? staticFile(url.pathname) : null;
    if (asStatic) { send(req, res, 200, { "content-type": asStatic.type, "cache-control": "no-cache" }, asStatic.body, url.pathname); return; }
    const chunks = []; for await (const c of req) chunks.push(c);
    const body = Buffer.concat(chunks);
    const headers = new Headers(); for (const [k, v] of Object.entries(req.headers)) if (v !== undefined && !["host", "connection", "content-length"].includes(k)) headers.set(k, Array.isArray(v) ? v.join(",") : v);
    for (const [k, v] of Object.entries(ownerHeaders())) headers.set(k, v);
    const request = new Request(url, { method: req.method, headers, ...(["GET", "HEAD"].includes(req.method) ? {} : { body }) });
    const response = await worker.fetch(request, env, { waitUntil() {}, passThroughOnException() {} });
    const out = {}; response.headers.forEach((v, k) => { out[k] = v; });
    const setCookie = response.headers.getSetCookie?.(); if (setCookie?.length) out["set-cookie"] = setCookie;
    delete out["content-encoding"]; delete out["content-length"];
    send(req, res, response.status, out, Buffer.from(await response.arrayBuffer()), null);
  } catch (error) { res.writeHead(500, { "content-type": "text/plain" }); res.end("ui-preview error: " + (error?.stack ?? error)); }
});
server.listen(PORT, "127.0.0.1", () => {
  mkdirSync(PREVIEW, { recursive: true }); writeFileSync(PID_FILE, String(process.pid));
  log(`LOCAL OWNER PREVIEW on http://localhost:${PORT} (live files from ${LIVE}; writes are in-memory only). Ctrl+C or --stop to end.`);
});
const quit = () => { rmSync(PID_FILE, { force: true }); server.close(); process.exit(0); };
process.on("SIGINT", quit); process.on("SIGTERM", quit);
