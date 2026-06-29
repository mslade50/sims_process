import { Env, kalshi, json, isGolfTicker, classifyTicker } from "../../shared/kalshi";

// GET /api/positions — open golf positions only.
export const onRequestGet: PagesFunction<Env> = async (ctx) => {
  const out: any[] = [];
  let cursor: string | undefined;
  for (let i = 0; i < 20; i++) {
    const r = await kalshi(ctx.env, "GET", "/portfolio/positions", { query: { limit: 200, cursor } });
    if (!r.ok) return json({ error: "positions failed", status: r.status, detail: r.data }, 502);
    out.push(...(r.data.market_positions || []));
    cursor = r.data.cursor;
    if (!cursor || (r.data.market_positions || []).length < 200) break;
  }
  const positions = out
    .filter((p) => isGolfTicker(p.ticker) && Number(p.position) !== 0)
    .map((p) => ({
      ticker: p.ticker,
      position: p.position, // +YES / -NO contracts
      market_exposure: p.market_exposure,
      realized_pnl: p.realized_pnl,
      total_traded: p.total_traded,
      resting_orders_count: p.resting_orders_count,
      ...classifyTicker(p.ticker),
    }));
  return json({ positions });
};
