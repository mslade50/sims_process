import { Env, kalshi, json, isGolfTicker, classifyTicker } from "../../shared/kalshi";

// GET /api/positions — open golf positions only.
// Kalshi /portfolio/positions fields are `position_fp` (signed contracts) and
// `*_dollars` strings — NOT `position` / `market_exposure` (those don't exist,
// which is why the UI showed undefined). classify gives market_type/player_code.
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
    .filter((p) => isGolfTicker(p.ticker) && Math.abs(Number(p.position_fp ?? p.position ?? 0)) > 1e-6)
    .map((p) => {
      const pf = Number(p.position_fp ?? p.position) || 0;
      const contracts = Math.round(Math.abs(pf));
      const exposure = Number(p.market_exposure_dollars ?? p.market_exposure) || 0;
      const cls = classifyTicker(p.ticker);
      return {
        ticker: p.ticker,
        position: Math.round(pf), // signed: +YES / -NO contracts
        side: pf > 0 ? "yes" : "no",
        contracts,
        exposure, // dollars at risk
        avg_cost: contracts > 0 ? exposure / contracts : 0, // dollars/contract
        realized_pnl: Number(p.realized_pnl_dollars ?? p.realized_pnl) || 0,
        fees_paid: Number(p.fees_paid_dollars) || 0,
        resting_orders_count: Number(p.resting_orders_count) || 0,
        market_type: cls.marketType,
        player_code: cls.playerCode,
        event_code: cls.eventCode,
      };
    })
    .sort((a, b) => b.exposure - a.exposure);
  return json({ positions });
};
