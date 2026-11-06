// Serialized build + preview restart + visual-check, for several agents sharing one checkout.
//
//   node scripts/ui-check.mjs [visual-check args, e.g. --routes this-week,why-priced --out dist-visual/agentA-r1]
//   node scripts/ui-check.mjs --build-only
//
// Takes a directory lock (.wrangler/ui-check.lock; stale after 20 min), runs `npm run build`, restarts the
// local preview (scripts/ui-preview.mjs --stop / --detach --no-build) and runs scripts/visual-check.mjs with the
// remaining args, then releases the lock. Exit code is the build's (if it failed) or visual-check's.
import { spawnSync } from "node:child_process";
import { mkdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(fileURLToPath(new URL("..", import.meta.url)));
const LOCK = join(ROOT, ".wrangler", "ui-check.lock");
const STALE_MS = 20 * 60 * 1000;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const run = (cmd, args) => spawnSync(cmd, args, { cwd: ROOT, stdio: "inherit", shell: process.platform === "win32" }).status ?? 1;

async function acquire() {
  mkdirSync(join(ROOT, ".wrangler"), { recursive: true });
  for (let waited = 0; ; waited += 5000) {
    try { mkdirSync(LOCK); writeFileSync(join(LOCK, "owner.txt"), `${process.pid} ${new Date().toISOString()}`); return; }
    catch {
      try { if (Date.now() - statSync(LOCK).mtimeMs > STALE_MS) { rmSync(LOCK, { recursive: true, force: true }); continue; } } catch { continue; }
      if (waited % 60000 === 0) console.log(`[ui-check] waiting for lock (${Math.round(waited / 1000)}s)...`);
      await sleep(5000);
    }
  }
}

const args = process.argv.slice(2);
const buildOnly = args.includes("--build-only");
await acquire();
let code = 0;
try {
  code = run("npm", ["run", "build"]);
  if (code !== 0) { console.error("[ui-check] build FAILED (another agent's file may be mid-edit: check the error's file before fixing)"); }
  else {
    run("node", ["scripts/ui-preview.mjs", "--stop"]);
    code = run("node", ["scripts/ui-preview.mjs", "--detach", "--no-build"]);
    if (code === 0 && !buildOnly) { await sleep(3000); code = run("node", ["scripts/visual-check.mjs", "--base", "http://localhost:3111", ...args.filter((a) => a !== "--build-only")]); }
  }
} finally { rmSync(LOCK, { recursive: true, force: true }); }
process.exit(code);
