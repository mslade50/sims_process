import { Env, json, OUTRIGHT_TYPES } from "../../shared/kalshi";

// GET /api/tape — query the captured golf-outright trade tape with filters.
//   from, to        unix seconds (time)
//   min_price, max_price   cents (filters on yes_price)
//   min_size        contracts
//   market          a market_type (winner|top_5|top_10|top_20) OR a ticker substring
//   limit           max rows (default 200, cap 1000)
export const onRequestGet: PagesFunction<Env> = async (ctx) => {
  const u = new URL(ctx.request.url).searchParams;
  const where: string[] = [];
  const binds: any[] = [];

  const num = (k: string) => (u.get(k) ? parseInt(u.get(k)!, 10) : null);
  const from = num("from"),
    to = num("to"),
    minPrice = num("min_price"),
    maxPrice = num("max_price"),
    minSize = num("min_size");
  const market = u.get("market");

  if (from !== null) (where.push("ts >= ?"), binds.push(from));
  if (to !== null) (where.push("ts <= ?"), binds.push(to));
  if (minPrice !== null) (where.push("yes_price >= ?"), binds.push(minPrice));
  if (maxPrice !== null) (where.push("yes_price <= ?"), binds.push(maxPrice));
  if (minSize !== null) (where.push("count >= ?"), binds.push(minSize));
  if (market) {
    if (OUTRIGHT_TYPES.has(market)) (where.push("market_type = ?"), binds.push(market));
    else (where.push("ticker LIKE ?"), binds.push("%" + market + "%"));
  }

  const limit = Math.min(num("limit") || 200, 1000);
  const sql =
    "SELECT trade_id, ticker, market_type, event_code, player_code, ts, yes_price, no_price, count, taker_side FROM trades" +
    (where.length ? " WHERE " + where.join(" AND ") : "") +
    " ORDER BY ts DESC LIMIT ?";
  binds.push(limit);

  try {
    const rs = await ctx.env.DB.prepare(sql).bind(...binds).all();
    return json({ trades: rs.results || [] });
  } catch (e) {
    return json({ error: "tape query failed", detail: e instanceof Error ? e.message : String(e) }, 500);
  }
};
