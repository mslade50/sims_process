import { Env, kalshi, json, isGolfTicker, classifyTicker } from "../../shared/kalshi";

// GET /api/orders — resting golf orders, annotated with auto-kill info from D1.
export const onRequestGet: PagesFunction<Env> = async (ctx) => {
  const raw: any[] = [];
  let cursor: string | undefined;
  for (let i = 0; i < 20; i++) {
    const r = await kalshi(ctx.env, "GET", "/portfolio/orders", {
      query: { status: "resting", limit: 200, cursor },
    });
    if (!r.ok) return json({ error: "orders failed", status: r.status, detail: r.data }, 502);
    raw.push(...(r.data.orders || []));
    cursor = r.data.cursor;
    if (!cursor || (r.data.orders || []).length < 200) break;
  }

  const orders = raw
    .filter((o) => isGolfTicker(o.ticker))
    .map((o) => ({
      order_id: o.order_id,
      ticker: o.ticker,
      side: o.side,
      price:
        o.side === "yes" ? (o.yes_price_dollars ?? o.yes_price) : (o.no_price_dollars ?? o.no_price),
      remaining: o.remaining_count_fp ?? o.remaining_count,
      expiration_ts: o.expiration_ts ?? null,
      created_time: o.created_time,
      kill_ts: null as number | null,
      hard_kill_ts: null as number | null,
      ...classifyTicker(o.ticker),
    }));

  // Annotate with pending auto-kill schedule.
  try {
    const kr = await ctx.env.DB.prepare(
      "SELECT order_id, kill_ts, hard_kill_ts FROM pending_kills WHERE status = 'pending'"
    ).all();
    const km = new Map<string, any>();
    for (const row of kr.results || []) km.set((row as any).order_id, row);
    for (const o of orders) {
      const k = km.get(o.order_id);
      if (k) {
        o.kill_ts = k.kill_ts;
        o.hard_kill_ts = k.hard_kill_ts;
      }
    }
  } catch {
    /* D1 unavailable — orders still returned without kill annotations */
  }

  return json({ orders });
};
