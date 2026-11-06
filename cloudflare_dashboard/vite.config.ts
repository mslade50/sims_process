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

/**
 * Round 7: the browser entry used to pull 30 tiny chunks (one per lucide icon, vinext header/hash helpers, the shell's rule files).
 * On the phone profile (150 ms round trip) the requests alone queue for about a second before hydration, which is the LCP of every route.
 * One "shell" chunk holds exactly the modules the entry and DashboardApp import statically; lazy views still import from it.
 * vinext sets the client code-splitting groups itself, so this adds one group in front of them after the config is resolved. Build-only, client-only.
 */
function mergeShellChunks(): Plugin {
  let done = false;
  return {
    name: "merge-shell-chunks",
    apply: "build",
    enforce: "post",
    configResolved(config) {
      if (done) return;
      const output = (config.environments?.client?.build as { rolldownOptions?: { output?: { codeSplitting?: { groups?: unknown[] } } } } | undefined)?.rolldownOptions?.output;
      const groups = output?.codeSplitting?.groups;
      if (!Array.isArray(groups)) return;
      done = true;
      groups.unshift({
        name: "shell",
        priority: 50,
        // Only what the entry already imports statically (the tiny vinext helpers, five lucide icons, the shell's own modules). Anything a view alone needs stays out, so the shell does not grow.
        test: /(?:node_modules\/(?:lucide-react\/dist\/esm\/(?:Icon|createLucideIcon|defaultAttributes|shared\/.*|icons\/(?:chevron-right|circle-gauge|cloud-sun|info|search|sliders-horizontal|users|x))\.js|vinext\/dist\/(?:server\/(?:app-elements|app-rsc-cache-busting|app-rsc-render-mode|headers)|shims\/(?:headers|layout-segment-context|navigation-errors)|utils\/(?:deployment-id|hash|protocol-headers|record))\.js))|(?:\/app\/(?:ShellParts|shell-rules|freshness-rules|ui-rules|ui|lib|data|event-param|viewer)\.tsx?$)/,
      });
    },
  };
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
      mergeShellChunks(),
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
