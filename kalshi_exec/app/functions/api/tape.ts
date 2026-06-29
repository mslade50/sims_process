import { Env, json } from "../../shared/kalshi";
import { buildTapeWhere, tapeRawSql } from "../../shared/tape-sql";

// GET /api/tape — query the captured golf-outright trade tape with filters.
//   from, to        unix seconds (time)
//   min_price, max_price   cents (filters on yes_price)
//   min_size        contracts
//   event           event_code (e.g. TRAV26) — scope to one tournament
//   market          a market_type (winner|top_5|top_10|top_20) OR a ticker substring
//   limit           max rows (default 200, cap 1000)
export const onRequestGet: PagesFunction<Env> = async (ctx) => {
  const u = new URL(ctx.request.url).searchParams;
  const { whereSql, binds } = buildTapeWhere({
    from: u.get("from"),
    to: u.get("to"),
    min_price: u.get("min_price"),
    max_price: u.get("max_price"),
    min_size: u.get("min_size"),
    event: u.get("event"),
    market: u.get("market"),
  });
  const limit = Math.min(u.get("limit") ? parseInt(u.get("limit")!, 10) : 200, 1000);

  try {
    const rs = await ctx.env.DB.prepare(tapeRawSql(whereSql)).bind(...binds, limit).all();
    return json({ trades: rs.results || [] });
  } catch (e) {
    return json({ error: "tape query failed", detail: e instanceof Error ? e.message : String(e) }, 500);
  }
};
