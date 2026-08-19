import { PxEnv, px, pxConfigured, pxJson } from "../../../shared/prophetx";

// POST /api/px/cancel  { wager_id } | { external_id }
// Cancels one ProphetX wager. Manual-exec wagers carry the "exm-" external_id
// prefix; cancels by external_id are restricted to that prefix so this surface
// can never cancel another system's resting wagers (mirrors the Kalshi
// cancel.ts scope rule). Session-gated by _middleware (master login).
export const onRequestPost: PagesFunction<PxEnv> = async (ctx) => {
  if (!pxConfigured(ctx.env)) {
    return pxJson({ error: "ProphetX not configured (PROPHETX_ACCESS_KEY/SECRET_KEY secrets)" }, 503);
  }
  const body: any = await ctx.request.json().catch(() => ({}));
  const wagerId = String(body.wager_id || "");
  const externalId = String(body.external_id || "");
  if (!wagerId && !externalId) return pxJson({ error: "wager_id or external_id required" }, 400);
  if (wagerId && !/^[A-Za-z0-9_-]+$/.test(wagerId)) return pxJson({ error: "malformed wager_id" }, 400);
  if (externalId && !externalId.startsWith("exm-")) {
    return pxJson({ error: "external_id cancels are limited to manual-exec (exm-) wagers" }, 400);
  }

  try {
    const payload: Record<string, string> = {};
    if (wagerId) payload.wager_id = wagerId;
    if (externalId) payload.external_id = externalId;
    const r = await px(ctx.env, "POST", "/mm/cancel_wager", { body: payload });
    return pxJson({ ok: r.ok, status: r.status, detail: r.data }, r.ok ? 200 : 502);
  } catch (e: any) {
    return pxJson({ error: String(e?.message || e) }, 502);
  }
};
