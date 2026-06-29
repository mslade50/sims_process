import { Env, kalshi, json, classifyTicker, isGolfTicker } from "../../shared/kalshi";
import { computePnl } from "../../shared/pnl-calc";

// Batch-fetch {ticker: {title, yes_bid, yes_ask}} (dollars) from /markets.
async function fetchMarketSummary(env: Env, tickers: string[]) {
  const out: Record<string, { title: string; yes_bid: number; yes_ask: number }> = {};
  const uniq = [...new Set(tickers.filter(Boolean))];
  for (let i = 0; i < uniq.length; i += 100) {
    const chunk = uniq.slice(i, i + 100);
    const r = await kalshi(env, "GET", "/markets", { query: { tickers: chunk.join(","), limit: 100 } });
    if (!r.ok) continue;
    for (const m of r.data.markets || []) {
      let yb = (m as any).yes_bid_dollars;
      let ya = (m as any).yes_ask_dollars;
      if (yb == null) yb = ((m as any).yes_bid || 0) / 100;
      if (ya == null) ya = ((m as any).yes_ask || 0) / 100;
      out[m.ticker] = { title: m.title || "", yes_bid: Number(yb) || 0, yes_ask: Number(ya) || 0 };
    }
  }
  return out;
}

// GET /api/pnl — all-time golf PnL from the durable D1 fills/settlements stores
// plus live positions + current marks. The accounting lives in shared/pnl-calc.js.
export const onRequestGet: PagesFunction<Env> = async (ctx) => {
  // 1. durable fills (graceful if the table doesn't exist yet)
  const fr = await ctx.env.DB.prepare("SELECT * FROM fills").all().catch(() => ({ results: [] as any[] }));
  const fills = (fr.results || []).map((r: any) => ({
    ticker: r.ticker,
    side: r.side,
    action: r.action,
    count: Number(r.count) || 0,
    yes_price: r.yes_price,
    no_price: r.no_price,
    fee_cost: Number(r.fee_cost) || 0,
    ts: r.ts,
  }));

  // 2. durable settlements
  const sr = await ctx.env.DB.prepare("SELECT * FROM settlements").all().catch(() => ({ results: [] as any[] }));
  const settlements: Record<string, any> = {};
  for (const r of (sr.results || []) as any[]) {
    settlements[r.ticker] = {
      market_result: r.market_result,
      yes_count: Number(r.yes_count) || 0,
      no_count: Number(r.no_count) || 0,
      yes_total_cost: Number(r.yes_total_cost) || 0,
      no_total_cost: Number(r.no_total_cost) || 0,
      fee_cost: Number(r.fee_cost) || 0,
      revenue: r.revenue != null ? Number(r.revenue) : null,
      settled_time: r.settled_time,
    };
  }

  // 3. live open golf positions (signed position_fp)
  const positions: Record<string, number> = {};
  let cursor: string | undefined;
  for (let i = 0; i < 20; i++) {
    const r = await kalshi(ctx.env, "GET", "/portfolio/positions", { query: { limit: 200, cursor } });
    if (!r.ok) return json({ error: "positions failed", status: r.status, detail: r.data }, 502);
    for (const p of r.data.market_positions || []) {
      const pf = Number(p.position_fp ?? p.position) || 0;
      if (Math.abs(pf) < 1e-6 || !isGolfTicker(p.ticker)) continue;
      positions[p.ticker] = pf;
    }
    cursor = r.data.cursor;
    if (!cursor || (r.data.market_positions || []).length < 200) break;
  }

  // 4. marks + titles for the open tickers (for MTM)
  const marks: Record<string, { yes_bid: number; yes_ask: number }> = {};
  const titles: Record<string, string> = {};
  const sm = await fetchMarketSummary(ctx.env, Object.keys(positions));
  for (const tk of Object.keys(sm)) {
    marks[tk] = { yes_bid: sm[tk].yes_bid, yes_ask: sm[tk].yes_ask };
    titles[tk] = sm[tk].title;
  }

  const { rows, totals } = computePnl({
    fills,
    settlements,
    positions,
    marks,
    titles,
    classify: classifyTicker,
    isGolf: isGolfTicker,
  });

  return json({
    rows,
    totals,
    fetched_ts: new Date().toISOString(),
    store: { fills: fills.length, settlements: Object.keys(settlements).length },
  });
};
