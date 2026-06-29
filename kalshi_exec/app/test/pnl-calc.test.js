"use strict";
// Unit tests for the ported PnL accounting. Run: node app/test/pnl-calc.test.js
const { computePnl } = require("../shared/pnl-calc.js");

let pass = 0, fail = 0;
function eq(name, got, want, tol = 1e-6) {
  const ok = typeof want === "number" ? Math.abs(got - want) <= tol : got === want;
  if (ok) pass++;
  else { fail++; console.error(`FAIL ${name}: got ${JSON.stringify(got)} want ${JSON.stringify(want)}`); }
}

const classify = (t) => ({ marketType: "winner", eventCode: "E", playerCode: t.split("-").pop() });
const isGolf = (t) => t.startsWith("KXPGA");
const F = (ticker, side, action, count, yes, fee = 0, ts = 0) =>
  ({ ticker, side, action, count, yes_price: yes, no_price: 1 - yes, fee_cost: fee, ts });

// ── 1. OPEN position, MTM at mid ─────────────────────────────────────────────
// Bought 100 YES @ $0.40 (cost $40, fee $1). Book 0.50/0.52 -> mid 0.51.
// realized = cashflow - fees = -40 - 1 = -41. unreal = 100 * 0.51 = 51. net = 10.
{
  const r = computePnl({
    fills: [F("KXPGATOUR-E-A", "yes", "buy", 100, 0.4, 1, 10)],
    positions: { "KXPGATOUR-E-A": 100 },
    marks: { "KXPGATOUR-E-A": { yes_bid: 0.5, yes_ask: 0.52 } },
    classify, isGolf,
  });
  const row = r.rows[0];
  eq("open is_open", row.is_open, true);
  eq("open contracts", row.contracts, 100);
  eq("open avg_cost", row.avg_cost, 0.4);
  eq("open realized", row.realized, -41);
  eq("open unrealized", row.unrealized, 51);
  eq("open net", row.net, 10);
  eq("open mark", row.mark, 0.51);
  eq("open return_pct", row.return_pct, (51 / 40) * 100);
  eq("totals net", r.totals.net, 10);
  eq("totals open_count", r.totals.open_count, 1);
}

// ── 2. SETTLED win, held to settlement (fills complete) ──────────────────────
// Bought 200 YES @ $0.30 (cost 60, fee 2), market resolves YES -> payout 200.
// fills_complete -> realized = cashflow + payout - fees = -60 + 200 - 2 = 138.
{
  const r = computePnl({
    fills: [F("KXPGATOUR-E-B", "yes", "buy", 200, 0.3, 2, 5)],
    settlements: { "KXPGATOUR-E-B": { market_result: "yes", yes_count: 200, no_count: 0, settled_time: "2026-06-20" } },
    classify, isGolf,
  });
  const row = r.rows[0];
  eq("win is_open", row.is_open, false);
  eq("win realized", row.realized, 138);
  eq("win unrealized", row.unrealized, 0);
  eq("win net", row.net, 138);
  eq("win settlement_revenue", row.settlement_revenue, 200);
  eq("win settled_count", r.totals.settled_count, 1);
}

// ── 3. SETTLED loss ──────────────────────────────────────────────────────────
// Bought 100 YES @ $0.70 (cost 70, fee 1), resolves NO -> payout 0.
// realized = -70 + 0 - 1 = -71.
{
  const r = computePnl({
    fills: [F("KXPGATOUR-E-C", "yes", "buy", 100, 0.7, 1, 5)],
    settlements: { "KXPGATOUR-E-C": { market_result: "no", yes_count: 100, no_count: 0, settled_time: "2026-06-20" } },
    classify, isGolf,
  });
  eq("loss realized", r.rows[0].realized, -71);
  eq("loss net", r.rows[0].net, -71);
}

// ── 4. SETTLED win, fills aged out — price off settlement record ─────────────
// No fills present; settlement says held 50 YES at total cost $20, resolves YES.
// realized = payout - settle_cost - settle_fee = 50 - 20 - 0.5 = 29.5.
{
  const r = computePnl({
    settlements: {
      "KXPGATOUR-E-D": { market_result: "yes", yes_count: 50, no_count: 0,
        yes_total_cost: 20, no_total_cost: 0, fee_cost: 0.5, settled_time: "2026-01-01" },
    },
    classify, isGolf,
  });
  const row = r.rows[0];
  eq("aged realized", row.realized, 29.5);
  eq("aged invested", row.invested, 20);
  eq("aged contracts", row.contracts, 50);
}

// ── 5. Round-tripped before expiry (closed via fills, no settlement) ─────────
// Buy 100 YES @ 0.40 (cost 40), sell 100 YES @ 0.55 ... "sell yes" reports as
// fill.side=no @ no_price. Model as buy NO 100 @ (1-0.55)=0.45 -> auto-convert
// 100 pairs to $100 cash. cashflow = -40 - 45 + 100 = 15, fees 0 -> realized 15.
{
  const r = computePnl({
    fills: [
      F("KXPGATOUR-E-F", "yes", "buy", 100, 0.4, 0, 1),
      F("KXPGATOUR-E-F", "no", "buy", 100, 0.0, 0, 2), // no_price = 1-0 ... set explicitly below
    ],
    classify, isGolf,
  });
  // override no leg price properly: rebuild with explicit no_price 0.45
  const r2 = computePnl({
    fills: [
      { ticker: "KXPGATOUR-E-F", side: "yes", action: "buy", count: 100, yes_price: 0.4, no_price: 0.6, fee_cost: 0, ts: 1 },
      { ticker: "KXPGATOUR-E-F", side: "no", action: "buy", count: 100, yes_price: 0.55, no_price: 0.45, fee_cost: 0, ts: 2 },
    ],
    classify, isGolf,
  });
  const row = r2.rows[0];
  eq("rt auto_converted", row.auto_converted, 100);
  eq("rt realized", row.realized, 15);
  eq("rt is_open", row.is_open, false);
  eq("rt contracts", row.contracts, 0);
}

// ── 5b. SCALAR / tie settlement (refunded matchup) — the Harris English bug ──
// Bought 600 YES @ $0.50 = $300. Matchup tied → result "scalar", refunded $300
// (revenue=300). realized must be ~0, NOT -300.
{
  const r = computePnl({
    fills: [F("KXPGAH2H-E-HENG", "yes", "buy", 600, 0.5, 0, 5)],
    settlements: {
      "KXPGAH2H-E-HENG": { market_result: "scalar", yes_count: 600, no_count: 0,
        yes_total_cost: 300, no_total_cost: 0, fee_cost: 0, revenue: 300, settled_time: "2026-06-27" },
    },
    classify, isGolf,
  });
  const row = r.rows[0];
  eq("scalar realized ~0", row.realized, 0);
  eq("scalar net ~0", row.net, 0);
  eq("scalar not open", row.is_open, false);
  eq("scalar settlement_revenue", row.settlement_revenue, 300);
}
// 5c. scalar with revenue missing falls back to 0 payout (best effort, no crash)
{
  const r = computePnl({
    fills: [F("KXPGAH2H-E-X", "yes", "buy", 100, 0.5, 0, 5)],
    settlements: { "KXPGAH2H-E-X": { market_result: "scalar", yes_count: 100, no_count: 0,
      yes_total_cost: 50, no_total_cost: 0, fee_cost: 0, revenue: null, settled_time: "2026-06-27" } },
    classify, isGolf,
  });
  eq("scalar no-revenue realized", r.rows[0].realized, -50);
}

// ── 5d. ghost settlement (no fills, held 0, cost 0) is dropped ───────────────
{
  const r = computePnl({
    settlements: {
      "KXPGAH2H-USO26R4ABHARMCI-RMCI": { market_result: "no", yes_count: 0, no_count: 0,
        yes_total_cost: 0, no_total_cost: 0, fee_cost: 0, revenue: null, settled_time: "2026-06-20" },
    },
    classify, isGolf,
  });
  eq("ghost dropped", r.rows.length, 0);
  eq("ghost settled_count", r.totals.settled_count, 0);
}
// 5e. a real held-to-settlement row is NOT dropped (guards over-filtering)
{
  const r = computePnl({
    settlements: { "KXPGATOUR-E-Z": { market_result: "yes", yes_count: 50, no_count: 0,
      yes_total_cost: 20, no_total_cost: 0, fee_cost: 0.5, settled_time: "x" } },
    classify, isGolf,
  });
  eq("real settled kept", r.rows.length, 1);
}

// ── 6. Non-golf flag passes through ──────────────────────────────────────────
{
  const r = computePnl({
    fills: [F("KXNFL-E-X", "yes", "buy", 10, 0.5, 0, 1)],
    positions: { "KXNFL-E-X": 10 },
    marks: { "KXNFL-E-X": { yes_bid: 0.5, yes_ask: 0.5 } },
    classify: () => ({ marketType: "", eventCode: "", playerCode: "" }), isGolf,
  });
  eq("nongolf is_golf", r.rows[0].is_golf, false);
}

// ── 7. Empty input ───────────────────────────────────────────────────────────
{
  const r = computePnl({});
  eq("empty rows", r.rows.length, 0);
  eq("empty net", r.totals.net, 0);
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
