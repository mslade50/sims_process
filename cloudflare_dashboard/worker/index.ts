/** Cloudflare Worker entry point for the vinext-starter template. */
import { handleImageOptimization, DEFAULT_DEVICE_SIZES, DEFAULT_IMAGE_SIZES } from "vinext/server/image-optimization";
import handler from "vinext/server/app-router-entry";
import { runAlerts } from "./alerts";
import { handleScheduled } from "./cron";
import { handleRetention } from "./retention";
import { handleJobsApi } from "./jobs-api";
import { handleOpsApi } from "./ops-api";
import { handleOwnerGate, ownerEmails, viewerFor } from "./owner";
import { handleOverridesApi } from "./overrides-api";
import { handleWeatherOverridesApi } from "./weather-overrides-api";
import { cacheControlFor, respondWithObject } from "./data-http";

interface Env {
  ASSETS: Fetcher;
  DASHBOARD_DATA?: R2Bucket;
  /** Optional: set both to verify the Cloudflare Access JWT signature on override writes. */
  ACCESS_TEAM_DOMAIN?: string;
  ACCESS_AUD?: string;
  /** Comma-separated owner emails (worker/owner.ts): only they see Run, weather edits and overrides. Unset = gate off. */
  OWNER_EMAILS?: string;
  /** Explicit Cloudflare Access service-client allowlist; empty means agents are denied. */
  OPS_AGENT_CLIENT_IDS?: string;
  /** Optional Worker secrets (owner action). While either is absent the alert tick only logs what it would send. */
  TELEGRAM_BOT_TOKEN?: string;
  TELEGRAM_CHAT_ID?: string;
  DB: D1Database;
  IMAGES: {
    input(stream: ReadableStream): {
      transform(options: Record<string, unknown>): {
        output(options: { format: string; quality: number }): Promise<{ response(): Response }>;
      };
    };
  };
}

interface ExecutionContext {
  waitUntil(promise: Promise<unknown>): void;
  passThroughOnException(): void;
}

// Image security config. SVG sources with .svg extension auto-skip the
// optimization endpoint on the client side (served directly, no proxy).
// To route SVGs through the optimizer (with security headers), set
// dangerouslyAllowSVG: true in next.config.js and uncomment below:
// const imageConfig: ImageConfig = { dangerouslyAllowSVG: true };

const worker = {
  /** Cron triggers (vite.config.ts localBindingConfig.triggers, schedule in worker/cron-rules.ts): enqueue the jobs due at this New York time. Never reachable over HTTP. */
  async scheduled(controller: { scheduledTime: number; cron?: string }, env: Env, ctx: ExecutionContext): Promise<void> {
    ctx.waitUntil(
      handleScheduled(controller, env)
        .then((outcome) => {
          console.log(`cron ${controller.cron ?? ""}: ${outcome.jobs.length ? `enqueued ${outcome.jobs.map((j) => `${j.type} ${j.id}`).join(", ")}` : `skipped (${outcome.reason})`}${outcome.skipped.length ? `; not queued: ${outcome.skipped.map((x) => `${x.type} (${x.reason})`).join("; ")}` : ""}`);
        })
        .catch((error: unknown) => {
          console.error(`cron ${controller.cron ?? ""} failed: ${error instanceof Error ? error.message : String(error)}`);
        }),
    );
    // Stale-primary / failed-job / digest alerts (worker/alerts.ts). Independent of the enqueue above: a failure in one never blocks the other.
    ctx.waitUntil(
      runAlerts(env, controller.scheduledTime)
        .then((outcome) => {
          console.log(`alerts: ${outcome.enabled ? "telegram on" : "telegram off (dark)"}; ${outcome.evaluated.length} due; sent ${outcome.sent.join(",") || "none"}${outcome.notes.length ? `; ${outcome.notes.join("; ")}` : ""}`);
        })
        .catch((error: unknown) => {
          console.error(`alerts failed: ${error instanceof Error ? error.message : String(error)}`);
        }),
    );
    // Daily retention of old job status/log/claim objects (worker/retention.ts). OFF unless jobs/retention/config.json exists; independent of the two above.
    ctx.waitUntil(
      handleRetention(controller, env)
        .then((outcome) => {
          if (outcome.ran) console.log(`retention (${outcome.mode}): ${outcome.note}`);
        })
        .catch((error: unknown) => {
          console.error(`retention failed: ${error instanceof Error ? error.message : String(error)}`);
        }),
    );
  },

  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    try {
      return await routeRequest(request, env, ctx);
    } catch (error) {
      console.error(`unhandled error for ${request.method} ${new URL(request.url).pathname}: ${error instanceof Error ? error.message : String(error)}`);
      return Response.json({ ok: false, error: "internal error" }, { status: 500, headers: { "cache-control": "no-store" } });
    }
  },
};

async function routeRequest(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
  const url = new URL(request.url);

  // This private page fails closed even when the legacy owner-list gate is unset.
  let decodedPath = url.pathname;
  try { decodedPath = decodeURIComponent(decodedPath); } catch { /* an invalid route is handled below */ }
  if (/^\/ops(?:\.rsc)?\/*$/i.test(decodedPath) && (!ownerEmails(env) || !(await viewerFor(request, env)).owner)) {
    return new Response(null, { status: 302, headers: { location: new URL("/this-week", url).href, "cache-control": "no-store" } });
  }
  // Ops authenticates owners and narrowly authorized service clients itself, fail-closed even without OWNER_EMAILS.
  const opsResponse = await handleOpsApi(request, env);
  if (opsResponse) return opsResponse;

  // Owner-only pages and APIs are refused before any handler sees them (worker/owner.ts).
  const ownerGate = await handleOwnerGate(request, env);
  if (ownerGate) return ownerGate;

  const apiResponse = await handleOverridesApi(request, env);
  if (apiResponse) return apiResponse;
  const weatherEditsResponse = await handleWeatherOverridesApi(request, env);
  if (weatherEditsResponse) return weatherEditsResponse;
  const jobsResponse = await handleJobsApi(request, env);
  if (jobsResponse) return jobsResponse;

  // Owner-only research data (R2 owner/<key>, e.g. owner/bet_breakdowns.json). handleOwnerGate above has already refused anyone else;
  // there is deliberately no packaged-asset fallback, so the file is never public.
  if (url.pathname.startsWith("/api/owner-data/")) {
    // Fail closed: unlike the shared gate (off while OWNER_EMAILS is unset), private data needs a configured owner list AND a verified owner.
    if (!ownerEmails(env) || !(await viewerFor(request, env)).owner) {
      return Response.json({ ok: false, error: "This part of the dashboard is for the owner only" }, { status: 403, headers: { "cache-control": "no-store" } });
    }
    let requested = "";
    try {
      requested = decodeURIComponent(url.pathname.slice("/api/owner-data/".length));
    } catch {
      return Response.json({ error: "Invalid data path" }, { status: 400 });
    }
    if (!requested || requested.includes("..") || !/^[A-Za-z0-9_./-]+\.json$/.test(requested)) {
      return Response.json({ error: "Invalid data path" }, { status: 400 });
    }
    let object: R2ObjectBody | null = null;
    try {
      object = env.DASHBOARD_DATA ? await env.DASHBOARD_DATA.get(`owner/${requested}`) : null;
    } catch {
      object = null;
    }
    if (!object) return Response.json({ error: "Not published yet" }, { status: 404, headers: { "cache-control": "no-store" } });
    return respondWithObject(request, object, { contentType: "application/json; charset=utf-8", cacheControl: "private, no-cache" });
  }

  if (url.pathname.startsWith("/api/data/")) {
    let requested = "";
    try {
      requested = decodeURIComponent(url.pathname.slice("/api/data/".length));
    } catch {
      return Response.json({ error: "Invalid data path" }, { status: 400 });
    }
    if (!requested || requested.includes("..")) {
      return Response.json({ error: "Invalid data path" }, { status: 400 });
    }
    const key = `data/${requested}`;
    let object: R2ObjectBody | null = null;
    try {
      object = env.DASHBOARD_DATA ? await env.DASHBOARD_DATA.get(key) : null;
    } catch {
      // Local preview and a newly provisioned bucket intentionally fall back
      // to the packaged snapshot until the first R2 publish completes.
    }
    if (object) {
      // manifest.json is the publish pointer (short max-age + revalidate); If-None-Match answers 304 so unchanged files cost no body.
      return respondWithObject(request, object, { contentType: "application/json; charset=utf-8", cacheControl: cacheControlFor(requested) });
    }
    const assetUrl = new URL(`/data/${requested}`, request.url);
    return env.ASSETS.fetch(new Request(assetUrl, request));
  }

  if (url.pathname === "/_vinext/image") {
    const allowedWidths = [...DEFAULT_DEVICE_SIZES, ...DEFAULT_IMAGE_SIZES];
    return handleImageOptimization(request, {
      fetchAsset: (path) => env.ASSETS.fetch(new Request(new URL(path, request.url))),
      transformImage: async (body, { width, format, quality }) => {
        const result = await env.IMAGES.input(body).transform(width > 0 ? { width } : {}).output({ format, quality });
        return result.response();
      },
    }, allowedWidths);
  }

  return handler.fetch(request, env, ctx);
}

export default worker;
