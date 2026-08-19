import { PxEnv, px, pxConfigured, pxJson } from "../../../shared/prophetx";

// GET /api/px/positions — ProphetX balance + recent wagers for the portfolio
// panel. Read-only. Session-gated by _middleware (master login).
export const onRequestGet: PagesFunction<PxEnv> = async (ctx) => {
  if (!pxConfigured(ctx.env)) {
    return pxJson({ error: "ProphetX not configured (PROPHETX_ACCESS_KEY/SECRET_KEY secrets)" }, 503);
  }
  try {
    const [bal, wagers] = await Promise.all([
      px(ctx.env, "GET", "/mm/get_balance"),
      // v2 histories: newest first; the UI filters open vs settled client-side.
      px(ctx.env, "GET", "/v2/mm/get_wager_histories", { params: { limit: "100" } }),
    ]);
    return pxJson({
      balance: bal.ok ? (bal.data?.data ?? bal.data) : { error: `HTTP ${bal.status}` },
      wagers: wagers.ok ? (wagers.data?.data ?? wagers.data) : { error: `HTTP ${wagers.status}` },
    });
  } catch (e: any) {
    return pxJson({ error: String(e?.message || e) }, 502);
  }
};
