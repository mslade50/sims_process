"use strict";
/**
 * Shared SQL builder for the tape summary query. Kept as plain JS (not TS) so a
 * Node test can run the *exact* query against an in-memory SQLite engine, and
 * imported by the TS endpoint via a thin re-export. Single source of truth for
 * the query string and its filter binding order.
 */
const OUTRIGHT_TYPES = new Set(["winner", "top_5", "top_10", "top_20"]);

/** Build the WHERE clause + binds shared by /api/tape and /api/tape/summary. */
function buildTapeWhere(params) {
  const where = [];
  const binds = [];
  const n = (k) => (params[k] != null && params[k] !== "" ? parseInt(params[k], 10) : null);
  const from = n("from"), to = n("to"), minPrice = n("min_price"), maxPrice = n("max_price"), minSize = n("min_size");
  const market = params.market;
  if (from !== null) (where.push("ts >= ?"), binds.push(from));
  if (to !== null) (where.push("ts <= ?"), binds.push(to));
  if (minPrice !== null) (where.push("yes_price >= ?"), binds.push(minPrice));
  if (maxPrice !== null) (where.push("yes_price <= ?"), binds.push(maxPrice));
  if (minSize !== null) (where.push("count >= ?"), binds.push(minSize));
  if (market) {
    if (OUTRIGHT_TYPES.has(market)) (where.push("market_type = ?"), binds.push(market));
    else (where.push("ticker LIKE ?"), binds.push("%" + market + "%"));
  }
  return { whereSql: where.length ? " WHERE " + where.join(" AND ") : "", binds };
}

/** Full per-ticker summary SQL. `limit` is appended to binds by the caller. */
function tapeSummarySql(whereSql) {
  return (
    "WITH f AS (SELECT ticker, market_type, player_code, event_code, ts, yes_price, count, taker_side," +
    " ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY ts ASC, trade_id ASC) AS rn_first," +
    " ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY ts DESC, trade_id DESC) AS rn_last" +
    " FROM trades" + whereSql + ")" +
    " SELECT ticker," +
    " MAX(market_type) AS market_type, MAX(player_code) AS player_code, MAX(event_code) AS event_code," +
    " COUNT(*) AS prints, SUM(count) AS volume, SUM(yes_price * count) AS pv," +
    " MIN(yes_price) AS low, MAX(yes_price) AS high," +
    " MIN(ts) AS first_ts, MAX(ts) AS last_ts," +
    " SUM(CASE WHEN taker_side = 'yes' THEN count ELSE 0 END) AS buy_vol," +
    " SUM(CASE WHEN taker_side = 'no'  THEN count ELSE 0 END) AS sell_vol," +
    " MAX(CASE WHEN rn_first = 1 THEN yes_price END) AS open_price," +
    " MAX(CASE WHEN rn_last  = 1 THEN yes_price END) AS close_price" +
    " FROM f GROUP BY ticker ORDER BY volume DESC LIMIT ?"
  );
}

/** Shape a raw summary row into the API response object. */
function shapeSummaryRow(r) {
  const volume = Number(r.volume) || 0;
  const buyVol = Number(r.buy_vol) || 0;
  const sellVol = Number(r.sell_vol) || 0;
  const signed = buyVol + sellVol;
  const open = r.open_price, close = r.close_price;
  return {
    ticker: r.ticker,
    market_type: r.market_type,
    player_code: r.player_code,
    event_code: r.event_code,
    prints: Number(r.prints) || 0,
    volume,
    vwap: volume > 0 ? Number(r.pv) / volume : close,
    open,
    last: close,
    high: r.high,
    low: r.low,
    change: open != null && close != null ? close - open : 0,
    changePct: open ? ((close - open) / open) * 100 : 0,
    buyVol,
    sellVol,
    delta: buyVol - sellVol,
    imbalance: signed > 0 ? (buyVol - sellVol) / signed : 0,
    first_ts: Number(r.first_ts) || null,
    last_ts: Number(r.last_ts) || null,
  };
}

module.exports = { OUTRIGHT_TYPES, buildTapeWhere, tapeSummarySql, shapeSummaryRow };
