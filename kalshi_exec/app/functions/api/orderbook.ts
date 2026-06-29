import { Env, kalshi, json, isGolfTicker } from "../../shared/kalshi";

// Normalize a Kalshi level array to {price_cents, qty}. Kalshi returns either
// `yes`/`no` (integer cents) or `yes_dollars`/`no_dollars` (decimal-string dollars).
function norm(levels: any[], isDollars: boolean): { price: number; qty: number }[] {
  return (levels || [])
    .map((lvl) => {
      const p = Number(lvl[0]);
      const q = Number(lvl[1]);
      return { price: isDollars ? Math.round(p * 1000) / 10 : p, qty: q };
    })
    .filter((l) => Number.isFinite(l.price) && Number.isFinite(l.qty));
}

// GET /api/orderbook?ticker=KXPGATOP10-...
export const onRequestGet: PagesFunction<Env> = async (ctx) => {
  const ticker = new URL(ctx.request.url).searchParams.get("ticker") || "";
  if (!isGolfTicker(ticker)) return json({ error: "golf tickers only" }, 400);

  const r = await kalshi(ctx.env, "GET", `/markets/${ticker}/orderbook`, { query: { depth: 10 } });
  if (!r.ok) return json({ error: "orderbook failed", status: r.status, detail: r.data }, 502);

  const ob = r.data.orderbook_fp || r.data.orderbook || r.data || {};
  const yes = ob.yes_dollars ? norm(ob.yes_dollars, true) : norm(ob.yes, false);
  const no = ob.no_dollars ? norm(ob.no_dollars, true) : norm(ob.no, false);
  // Levels come worst→best from Kalshi; sort bids high→low so [0] = best bid.
  yes.sort((a, b) => b.price - a.price);
  no.sort((a, b) => b.price - a.price);
  return json({ ticker, yes, no });
};
