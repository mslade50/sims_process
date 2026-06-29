"use strict";
/**
 * All-time PnL accounting, ported faithfully from maker_dashboard/server.py
 * (the version that survived "two false starts"). Pure and dependency-free so a
 * Node test can verify the math; the TS endpoint injects classify/isGolf and the
 * live position/mark data.
 *
 * Money convention here is DOLLARS (fills prices, fees, costs, marks). The cron
 * stores fills/settlements in dollars in D1 for exactly this reason. (The trades
 * tape table is unrelated and stays in cents.)
 *
 * Inputs:
 *   fills:        [{ ticker, side:"yes"|"no", action:"buy"|"sell", count,
 *                    yes_price, no_price, fee_cost, ts, created_time }]   (dollars)
 *   settlements:  { [ticker]: { market_result:"yes"|"no"|"", yes_count, no_count,
 *                    yes_total_cost, no_total_cost, fee_cost, settled_time } }
 *   positions:    { [ticker]: position_fp }   signed: +YES / -NO contracts (open only)
 *   marks:        { [ticker]: { yes_bid, yes_ask } }   dollars, current book
 *   classify(ticker) -> { marketType, eventCode, playerCode }
 *   isGolf(ticker)   -> boolean
 *   titles:       { [ticker]: title }  (optional, display only)
 *
 * Returns { rows, totals }.
 */
function computePnl(args) {
  const {
    fills = [],
    settlements = {},
    positions = {},
    marks = {},
    titles = {},
    classify = () => ({ marketType: "", eventCode: "", playerCode: "" }),
    isGolf = () => false,
  } = args || {};

  // ── group fills per ticker, chronological ──
  const byTicker = new Map();
  for (const f of fills) {
    const tk = f.ticker || "";
    if (!tk) continue;
    if (!byTicker.has(tk)) byTicker.set(tk, []);
    byTicker.get(tk).push(f);
  }
  for (const arr of byTicker.values()) arr.sort((a, b) => (a.ts || 0) - (b.ts || 0));

  const emptyBucket = () => ({
    cashflow: 0, fees: 0, invested: 0, yes_long: 0, no_long: 0,
    yes_buys: 0, yes_sells: 0, no_buys: 0, no_sells: 0,
    auto_converted: 0, first_ts: null, last_ts: null,
  });

  const buckets = new Map();
  for (const [ticker, fs] of byTicker) {
    const b = emptyBucket();
    for (const f of fs) {
      const side = f.side === "no" ? "no" : "yes";
      const action = f.action || "";
      const count = Number(f.count) || 0;
      if (count <= 0) continue;
      const yesP = Number(f.yes_price) || 0;
      const noP = Number(f.no_price) || 0;
      const price = side === "yes" ? yesP : noP;
      const fee = Number(f.fee_cost) || 0;
      const cntInt = Math.round(count);
      if (side === "yes") {
        b.yes_long += count;
        if (action === "buy") b.yes_buys += cntInt; else b.yes_sells += cntInt;
      } else {
        b.no_long += count;
        if (action === "buy") b.no_buys += cntInt; else b.no_sells += cntInt;
      }
      const cost = price * count;
      b.cashflow -= cost;
      b.invested += cost;
      b.fees += fee;
      // auto-conversion: whenever both inventories > 0, Kalshi nets pairs to $1
      if (b.yes_long > 0 && b.no_long > 0) {
        const pairs = Math.min(b.yes_long, b.no_long);
        b.yes_long -= pairs;
        b.no_long -= pairs;
        b.cashflow += pairs;
        b.auto_converted += pairs;
      }
      const ts = f.ts != null ? f.ts : null;
      if (ts != null) {
        if (b.first_ts == null || ts < b.first_ts) b.first_ts = ts;
        if (b.last_ts == null || ts > b.last_ts) b.last_ts = ts;
      }
    }
    buckets.set(ticker, b);
  }
  // settled tickers with no fills (legacy) — still surface a row
  for (const tk of Object.keys(settlements)) if (!buckets.has(tk)) buckets.set(tk, emptyBucket());

  // ── build rows ──
  const rows = [];
  for (const [ticker, b0] of buckets) {
    const b = b0;
    let cashflow = b.cashflow;
    let fees = b.fees;
    const sett = settlements[ticker] || null;

    let marketResult = "", settledTime = null;
    let heldYes = 0, heldNo = 0, settleCost = 0, settleFee = 0, revenue = null;
    if (sett) {
      marketResult = sett.market_result || "";
      settledTime = sett.settled_time || null;
      heldYes = Number(sett.yes_count) || 0;
      heldNo = Number(sett.no_count) || 0;
      settleCost = (Number(sett.yes_total_cost) || 0) + (Number(sett.no_total_cost) || 0);
      settleFee = Number(sett.fee_cost) || 0;
      revenue = sett.revenue != null ? Number(sett.revenue) : null;
    }
    // Binary win pays the winning side count * $1 (exact, and immune to Kalshi's
    // flaky `revenue` on winners). Any non-yes/no result (scalar / void / tie) is
    // NOT count*$1 — use the actual dollars credited (`revenue`) so a refunded
    // matchup nets ~0 instead of looking like a total loss. Falls back to 0 only
    // if revenue is genuinely missing.
    let payout;
    if (marketResult === "yes") payout = heldYes;
    else if (marketResult === "no") payout = heldNo;
    else payout = revenue != null ? revenue : 0;

    const posFp = positions[ticker];
    const isOpen = posFp != null && Math.abs(Number(posFp)) > 1e-6;

    const cls = classify(ticker);
    const sm = marks[ticker] || {};

    let mark = null, unreal = 0, settlementRev = 0;
    let contracts = 0, side = "", realized = 0, costBasis = 0, invested = 0, avgCost = 0;

    if (isOpen) {
      side = Number(posFp) > 0 ? "yes" : "no";
      contracts = Math.round(Math.abs(Number(posFp)));
      realized = cashflow - fees;
      costBasis = Math.abs(cashflow);
      invested = b.invested;
      avgCost = contracts > 0 ? -cashflow / contracts : 0;
      const yesBid = Number(sm.yes_bid) || 0;
      const yesAsk = Number(sm.yes_ask) || 0;
      if (yesBid > 0 && yesAsk > 0 && yesAsk >= yesBid) {
        const yesMid = (yesBid + yesAsk) / 2;
        unreal = b.yes_long * yesMid + b.no_long * (1 - yesMid);
        mark = contracts ? unreal / contracts : 0;
      } else {
        mark = contracts ? Math.abs(cashflow) / contracts : 0;
        unreal = mark * contracts;
      }
    } else if (sett) {
      settlementRev = payout;
      contracts = Math.round(heldYes + heldNo);
      side = heldYes >= heldNo ? "yes" : "no";
      const fillsComplete =
        (b.yes_buys || b.no_buys) &&
        Math.abs(b.yes_long - heldYes) < 1.0 &&
        Math.abs(b.no_long - heldNo) < 1.0;
      if (fillsComplete) {
        realized = cashflow + payout - fees;
        costBasis = Math.abs(cashflow);
        invested = b.invested;
      } else {
        realized = payout - settleCost - settleFee;
        costBasis = settleCost;
        invested = settleCost;
        fees = settleFee;
        cashflow = -settleCost;
      }
      avgCost = contracts > 0 ? costBasis / contracts : 0;
    } else {
      contracts = 0;
      const netYes = b.yes_buys - b.yes_sells;
      const netNo = b.no_buys - b.no_sells;
      side = netYes || netNo ? (Math.abs(netYes) >= Math.abs(netNo) ? "yes" : "no") : "";
      realized = cashflow - fees;
      costBasis = Math.abs(cashflow);
      invested = b.invested;
      avgCost = 0;
    }

    const net = realized + unreal;
    rows.push({
      ticker,
      title: titles[ticker] || "",
      market_type: cls.marketType,
      event_code: cls.eventCode,
      player_code: cls.playerCode,
      is_golf: isGolf(ticker),
      is_open: isOpen,
      side,
      contracts,
      avg_cost: avgCost,
      exposure: costBasis,
      invested,
      auto_converted: b.auto_converted,
      cashflow,
      settlement_revenue: settlementRev,
      realized,
      unrealized: unreal,
      fees,
      mark,
      net,
      market_result: marketResult,
      settled_time: settledTime,
      first_fill_ts: b.first_ts,
      last_fill_ts: b.last_ts,
      return_pct: isOpen && costBasis > 0 ? (unreal / costBasis) * 100 : null,
    });
  }

  const sum = (f) => rows.reduce((s, r) => s + (Number(r[f]) || 0), 0);
  const totals = {
    realized: sum("realized"),
    unrealized: sum("unrealized"),
    fees: sum("fees"),
    net: sum("net"),
    invested: sum("invested"),
    open_count: rows.filter((r) => r.is_open).length,
    settled_count: rows.filter((r) => !r.is_open).length,
  };
  return { rows, totals };
}

(function (root, factory) {
  const mod = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = mod;
  if (root) root.PnlCalc = mod;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  return { computePnl };
});
