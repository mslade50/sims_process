import { Env, kalshi, json, isGolfTicker } from "../../shared/kalshi";

// POST /api/cancel  { order_id, ticker }
// Golf-scope enforced at the API boundary (mirrors kalshi_maker.cancel_order).
export const onRequestPost: PagesFunction<Env> = async (ctx) => {
  const body: any = await ctx.request.json().catch(() => ({}));
  const order_id = String(body.order_id || "");
  const ticker = String(body.ticker || "");
  if (!order_id) return json({ error: "order_id required" }, 400);
  if (!isGolfTicker(ticker)) return json({ error: "golf tickers only (pass the order's ticker)" }, 400);

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
