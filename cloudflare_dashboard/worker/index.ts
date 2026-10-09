/** Cloudflare Worker entry point for the vinext-starter template. */
import { handleImageOptimization, DEFAULT_DEVICE_SIZES, DEFAULT_IMAGE_SIZES } from "vinext/server/image-optimization";
import handler from "vinext/server/app-router-entry";
import { runAlerts } from "./alerts";
import { handleScheduled } from "./cron";
import { handleJobsApi } from "./jobs-api";
import { handleOverridesApi } from "./overrides-api";

interface Env {
  ASSETS: Fetcher;
  DASHBOARD_DATA?: R2Bucket;
  /** Optional: set both to verify the Cloudflare Access JWT signature on override writes. */
  ACCESS_TEAM_DOMAIN?: string;
  ACCESS_AUD?: string;
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

  const apiResponse = await handleOverridesApi(request, env);
  if (apiResponse) return apiResponse;
  const jobsResponse = await handleJobsApi(request, env);
  if (jobsResponse) return jobsResponse;

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
      const headers = new Headers();
      object.writeHttpMetadata(headers);
      headers.set("content-type", "application/json; charset=utf-8");
      headers.set("cache-control", requested === "manifest.json" ? "no-cache" : "public, max-age=300, stale-while-revalidate=3600");
      headers.set("etag", object.httpEtag);
      return new Response(object.body, { headers });
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
