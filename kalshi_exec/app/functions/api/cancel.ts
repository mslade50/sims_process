import { Env, kalshi, json, isGolfTicker } from "../../shared/kalshi";

// POST /api/cancel  { order_id, ticker }
// Golf-scope enforced at the API boundary (mirrors kalshi_maker.cancel_order).
// The caller-supplied ticker is only a fast pre-check: the cancel is authorized
// against the order's ACTUAL ticker from Kalshi, so a spoofed body ticker can't
// cancel a non-golf (or otherwise out-of-scope) order (2026-08 audit).
export const onRequestPost: PagesFunction<Env> = async (ctx) => {
  const body: any = await ctx.request.json().catch(() => ({}));
  const order_id = String(body.order_id || "");
  const ticker = String(body.ticker || "");
  if (!order_id) return json({ error: "order_id required" }, 400);
  if (!/^[A-Za-z0-9-]+$/.test(order_id)) return json({ error: "malformed order_id" }, 400);
  if (!isGolfTicker(ticker)) return json({ error: "golf tickers only (pass the order's ticker)" }, 400);

  const lookup = await kalshi(ctx.env, "GET", `/portfolio/orders/${order_id}`);
  if (!lookup.ok) {
    return json({ error: "order lookup failed — refusing blind cancel", detail: lookup.data }, 502);
  }
  const realTicker = String(lookup.data?.order?.ticker || "");
  if (!isGolfTicker(realTicker)) {
    return json({ error: `order ${order_id} is on '${realTicker}' (non-golf) — refusing` }, 400);
  }

  const r = await kalshi(ctx.env, "DELETE", `/portfolio/orders/${order_id}`);
  try {
    await ctx.env.DB.prepare("UPDATE pending_kills SET status = 'gone' WHERE order_id = ?")
      .bind(order_id)
      .run();
  } catch {
    /* non-fatal */
  }
  return json({ ok: r.ok, status: r.status, detail: r.data }, r.ok ? 200 : 502);
};
