/** Cloudflare Worker entry point for the vinext-starter template. */
import { handleImageOptimization, DEFAULT_DEVICE_SIZES, DEFAULT_IMAGE_SIZES } from "vinext/server/image-optimization";
import handler from "vinext/server/app-router-entry";

interface Env {
  ASSETS: Fetcher;
  DASHBOARD_DATA?: R2Bucket;
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
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname.startsWith("/api/data/")) {
      const requested = decodeURIComponent(url.pathname.slice("/api/data/".length));
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
  },
};

export default worker;
