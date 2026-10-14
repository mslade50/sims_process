import vinext from "vinext";
import { defineConfig } from "vite";
import { cloudflare, getLocalWorkerdCompatibilityDate } from "@cloudflare/vite-plugin";
import hostingConfig from "./.openai/hosting.json";
import { sites } from "./build/sites-vite-plugin";
import { rmSync } from "node:fs";
import { rm } from "node:fs/promises";
import { resolve } from "node:path";
import type { Plugin } from "vite";

/**
 * public/golfprice/ is an untracked, hand-copied snapshot used only by the no-Worker local preview (RESEARCH_LOCAL_PREVIEW=1), where
 * /api/golfprice/* does not exist (site audit D2). The production Worker serves golfprice data from R2 and never reads these files
 * (nothing in worker/ or the app requests /golfprice/ in production), so a production build drops the copy from dist/client.
 * Keep it with RESEARCH_LOCAL_PREVIEW=1 or PACKAGE_GOLFPRICE_STATIC=1. Only the build OUTPUT is touched; public/golfprice is never deleted.
 */
function excludePackagedGolfprice(): Plugin {
  let root = process.cwd();
  return {
    name: "exclude-packaged-golfprice",
    apply: "build",
    enforce: "post",
    configResolved(config) {
      root = config.root;
    },
    async closeBundle() {
      await rm(resolve(root, "dist", "client", "golfprice"), { recursive: true, force: true });
    },
  };
}

/**
 * RESEARCH_LOCAL_PREVIEW builds keep emptyOutDir off (the preview server may be serving dist/), so content-hashed chunks from earlier
 * builds piled up next to the new ones. Clear only the generated hashed output under dist/ once per build process, before any
 * environment writes. Nothing outside dist/ is touched; public/ and dist/client/data (copied by the build) are re-created by the build.
 */
function cleanStaleBuildOutput(rootDir: string) {
  const guard = globalThis as { __dashboardDistCleaned?: boolean };
  if (guard.__dashboardDistCleaned) return;
  guard.__dashboardDistCleaned = true;
  const dist = resolve(rootDir, "dist");
  for (const sub of [["client", "_next"], ["server", "_next"], ["server", "ssr"]]) {
    rmSync(resolve(dist, ...sub), { recursive: true, force: true });
  }
}

const SITE_CREATOR_PLACEHOLDER_DATABASE_ID =
  "00000000-0000-4000-8000-000000000000";

const { d1 } = hostingConfig;

// macOS Seatbelt blocks FSEvents, so Codex previews need polling for HMR.
const isCodexSeatbeltSandbox = process.env.CODEX_SANDBOX === "seatbelt";

const localBindingConfig = {
  main: "./worker/index.ts",
  workers_dev: true,
  preview_urls: false,
  d1_databases: d1
    ? [
        {
          binding: d1,
          database_name: "site-creator-d1",
          database_id: SITE_CREATOR_PLACEHOLDER_DATABASE_ID,
        },
      ]
    : [],
  // r2_buckets, triggers (cron) and vars live in wrangler.jsonc, the single source of truth shared with CI.
};

export default defineConfig(({ command }) => {
  const localResearchPreview = process.env.RESEARCH_LOCAL_PREVIEW === "1";
  const useNodePreview = localResearchPreview && command === "serve";
  // Keep Wrangler and Miniflare state project-local. These are non-secret tool
  // settings; application environment belongs in ignored `.env*` files.
  process.env.WRANGLER_WRITE_LOGS ??= "false";
  process.env.WRANGLER_LOG_PATH ??= ".wrangler/logs";
  process.env.MINIFLARE_REGISTRY_PATH ??= ".wrangler/registry";

  if (command === "build" && localResearchPreview) cleanStaleBuildOutput(process.cwd());

  return {
    build: localResearchPreview ? { emptyOutDir: false } : undefined,
    server: {
      ...(isCodexSeatbeltSandbox ? { watch: { useFsEvents: false, usePolling: true } } : {}),
      proxy: { "/api/backtest": { target: "http://127.0.0.1:8766", changeOrigin: true } },
    },
    plugins: [
      vinext(),
      ...(!localResearchPreview ? [sites()] : []),
      ...(!localResearchPreview && process.env.PACKAGE_GOLFPRICE_STATIC !== "1" ? [excludePackagedGolfprice()] : []),
      ...(!useNodePreview ? [cloudflare({
        viteEnvironment: { name: "rsc", childEnvironments: ["ssr"] },
        config: localResearchPreview
          ? { ...localBindingConfig, compatibility_date: getLocalWorkerdCompatibilityDate().date }
          : localBindingConfig,
        ...(localResearchPreview ? { remoteBindings: false, persistState: false } : {}),
      })] : []),
    ],
  };
});
