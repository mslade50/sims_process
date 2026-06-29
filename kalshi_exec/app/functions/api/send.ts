import { Env, kalshi, json, isGolfTicker } from "../../shared/kalshi";

// Identifiable, unique client_order_id (32 chars). "exm" prefix distinguishes
// manual exec orders from the maker bot's SCRIPT_ORDER_PREFIX orders.
function execClientOrderId(): string {
  return "exm" + crypto.randomUUID().replace(/-/g, "").slice(0, 29);
}

// POST /api/send
// { orders: [ { ticker, side: "yes"|"no", price_cents, count, kill_ts? } ] }
// Places limit buys. Golf-only. kill_ts (unix sec) => native expiry + D1 auto-kill registry.
export const onRequestPost: PagesFunction<Env> = async (ctx) => {
  const body: any = await ctx.request.json().catch(() => ({}));
  const orders = Array.isArray(body.orders) ? body.orders : [];
  if (!orders.length) return json({ error: "no orders submitted" }, 400);

  const now = Math.floor(Date.now() / 1000);
  const details: any[] = [];
  let posted = 0;
  let failed = 0;

  for (const o of orders) {
    const ticker = String(o.ticker || "");
    const side = o.side === "no" ? "no" : "yes";
    const count = parseInt(o.count, 10);
    const cents = Number(o.price_cents);
    const kill_ts = o.kill_ts ? parseInt(o.kill_ts, 10) : null;
    const base = { ticker, side, price_cents: cents, count, kill_ts };

    if (!isGolfTicker(ticker)) {
      failed++;
      details.push({ ...base, status: "failed", message: "non-golf ticker — refusing" });
      continue;
    }
    if (!(count > 0)) {
      failed++;
      details.push({ ...base, status: "failed", message: "count must be > 0" });
      continue;
    }
    if (!(cents > 0 && cents < 100)) {
      failed++;
      details.push({ ...base, status: "failed", message: "price must be between 0 and 100¢" });
      continue;
    }
    if (kill_ts !== null && kill_ts <= now) {
      failed++;
      details.push({ ...base, status: "failed", message: "kill time is in the past" });
      continue;
    }

    const priceField = side === "yes" ? "yes_price_dollars" : "no_price_dollars";
    const orderBody: any = {
      ticker,
      client_order_id: execClientOrderId(),
      type: "limit",
      action: "buy",
      side,
      count,
      [priceField]: (cents / 100).toFixed(4),
    };
    if (kill_ts !== null) {
      orderBody.time_in_force = "good_till_canceled";
      orderBody.expiration_ts = kill_ts;
    }

    const r = await kalshi(ctx.env, "POST", "/portfolio/orders", { body: orderBody });
    if (r.ok) {
      const oid = r.data?.order?.order_id || "?";
      posted++;
      details.push({ ...base, status: "posted", order_id: oid });
      if (kill_ts !== null && oid !== "?") {
        try {
          await ctx.env.DB.prepare(
            "INSERT OR REPLACE INTO pending_kills (order_id, ticker, kill_ts, hard_kill_ts, status, created_ts) VALUES (?,?,?,?,?,?)"
          )
            .bind(oid, ticker, kill_ts, kill_ts + 60, "pending", now)
            .run();
        } catch {
          details[details.length - 1].kill_warning = "order placed but D1 kill-registry write failed";
        }
      }
    } else {
      failed++;
      details.push({ ...base, status: "failed", message: JSON.stringify(r.data).slice(0, 300) });
    }
  }

  return json({ posted, failed, details });
};
