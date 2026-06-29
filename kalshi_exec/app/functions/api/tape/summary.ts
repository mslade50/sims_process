import { Env, json } from "../../../shared/kalshi";
import { buildTapeWhere, tapeSummarySql, shapeSummaryRow } from "../../../shared/tape-sql";

// GET /api/tape/summary — per-ticker aggregates over the SAME filter set as
// /api/tape, computed in SQL so the watchlist reflects the full matching window
// (not just the most-recent N raw prints the tape table is capped to).
//   from, to        unix seconds
//   min_price, max_price   cents (on yes_price)
//   min_size        contracts
//   market          market_type (winner|top_5|top_10|top_20) OR ticker substring
//   limit           max ticker rows (default 200, cap 500)
//
// Per ticker: prints, volume, vwap, open, close (last), high, low, first/last ts,
// buy_vol (YES-aggressor), sell_vol (NO-aggressor). TWAP is a chart overlay and
// is computed client-side from the focused ticker's prints. The query string and
// binding order live in shared/tape-sql.js (also exercised by the SQLite test).
export const onRequestGet: PagesFunction<Env> = async (ctx) => {
  const u = new URL(ctx.request.url).searchParams;
  const { whereSql, binds } = buildTapeWhere({
    from: u.get("from"),
    to: u.get("to"),
    min_price: u.get("min_price"),
    max_price: u.get("max_price"),
    min_size: u.get("min_size"),
    market: u.get("market"),
  });

  const limit = Math.min(u.get("limit") ? parseInt(u.get("limit")!, 10) : 200, 500);
  const sql = tapeSummarySql(whereSql);

  try {
    const rs = await ctx.env.DB.prepare(sql)
      .bind(...binds, limit)
      .all();
    return json({ rows: (rs.results || []).map(shapeSummaryRow) });
  } catch (e) {
    return json({ error: "tape summary failed", detail: e instanceof Error ? e.message : String(e) }, 500);
  }
};
