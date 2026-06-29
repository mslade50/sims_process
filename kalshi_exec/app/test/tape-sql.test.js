"use strict";
// Validates the tape-summary SQL against a real (in-memory) SQLite engine —
// the same query string the /api/tape/summary endpoint runs on D1. Catches
// window-function / CTE / bare-column syntax issues before deploy.
// Run: node --experimental-sqlite app/test/tape-sql.test.js   (no flag on Node 24+)
const { DatabaseSync } = require("node:sqlite");
const { buildTapeWhere, tapeSummarySql, shapeSummaryRow } = require("../shared/tape-sql.js");

let pass = 0, fail = 0;
function eq(name, got, want, tol = 1e-9) {
  const ok = typeof want === "number" ? Math.abs(got - want) <= tol : got === want;
  if (ok) pass++;
  else { fail++; console.error(`FAIL ${name}: got ${JSON.stringify(got)} want ${JSON.stringify(want)}`); }
}
function truthy(name, got) { if (got) pass++; else { fail++; console.error(`FAIL ${name}: falsy ${JSON.stringify(got)}`); } }

const db = new DatabaseSync(":memory:");
db.exec(`CREATE TABLE trades (
  trade_id TEXT PRIMARY KEY, ticker TEXT NOT NULL, market_type TEXT NOT NULL,
  event_code TEXT, player_code TEXT, ts INTEGER NOT NULL,
  yes_price INTEGER, no_price INTEGER, count INTEGER NOT NULL, taker_side TEXT);`);

const ins = db.prepare(
  "INSERT INTO trades (trade_id,ticker,market_type,event_code,player_code,ts,yes_price,no_price,count,taker_side) VALUES (?,?,?,?,?,?,?,?,?,?)"
);
let id = 0;
const add = (ticker, mt, player, ts, yes, count, taker) =>
  ins.run("t" + id++, ticker, mt, "PGC", player, ts, yes, 100 - yes, count, taker);

// Player A (winner): opens 40, closes 50, high 60, vol 500, buy 200 / sell 300
add("KXPGATOUR-PGC-A", "winner", "A", 100, 40, 100, "yes");
add("KXPGATOUR-PGC-A", "winner", "A", 200, 60, 300, "no");
add("KXPGATOUR-PGC-A", "winner", "A", 300, 50, 100, "yes");
// Player B (winner): one big print, vol 900
add("KXPGATOUR-PGC-B", "winner", "B", 150, 12, 900, "no");
// Player C (top_5): below size floor when min_size filter applied
add("KXPGATOP5-PGC-C", "top_5", "C", 250, 80, 50, "yes");

function runSummary(params, limit = 200) {
  const { whereSql, binds } = buildTapeWhere(params);
  const stmt = db.prepare(tapeSummarySql(whereSql));
  return stmt.all(...binds, limit).map(shapeSummaryRow);
}

// ── all rows, no filter — ordered by volume desc ─────────────────────────────
const all = runSummary({});
eq("rows count", all.length, 3);
eq("top by volume = B", all[0].ticker, "KXPGATOUR-PGC-B");
eq("B volume", all[0].volume, 900);

const A = all.find((r) => r.ticker === "KXPGATOUR-PGC-A");
eq("A prints", A.prints, 3);
eq("A volume", A.volume, 500);
eq("A open", A.open, 40);
eq("A last(close)", A.last, 50);
eq("A high", A.high, 60);
eq("A low", A.low, 40);
eq("A vwap", A.vwap, (40 * 100 + 60 * 300 + 50 * 100) / 500);
eq("A buyVol", A.buyVol, 200);
eq("A sellVol", A.sellVol, 300);
eq("A delta", A.delta, -100);
eq("A imbalance", A.imbalance, (200 - 300) / 500);
eq("A change", A.change, 10);
eq("A first_ts", A.first_ts, 100);
eq("A last_ts", A.last_ts, 300);

// ── market_type filter ───────────────────────────────────────────────────────
const win = runSummary({ market: "winner" });
eq("winner rows", win.length, 2);
truthy("winner only", win.every((r) => r.market_type === "winner"));

// ── min_size filter drops C (50 < 100) ───────────────────────────────────────
const big = runSummary({ min_size: 100 });
truthy("min_size drops C", !big.some((r) => r.ticker === "KXPGATOP5-PGC-C"));

// ── price band filter ────────────────────────────────────────────────────────
const band = runSummary({ min_price: 45, max_price: 70 });
// Only A's 60 and 50 prints qualify; B's 12 and C's 80 excluded
const Aband = band.find((r) => r.ticker === "KXPGATOUR-PGC-A");
eq("band A prints", Aband.prints, 2);
eq("band A volume", Aband.volume, 400);
truthy("band excludes B", !band.some((r) => r.ticker === "KXPGATOUR-PGC-B"));

// ── ticker substring filter ──────────────────────────────────────────────────
const sub = runSummary({ market: "PGC-B" });
eq("substring rows", sub.length, 1);
eq("substring is B", sub[0].ticker, "KXPGATOUR-PGC-B");

// ── time window filter ───────────────────────────────────────────────────────
const win2 = runSummary({ from: 200 });
const A2 = win2.find((r) => r.ticker === "KXPGATOUR-PGC-A");
eq("from=200 A prints", A2.prints, 2);
eq("from=200 A open", A2.open, 60); // first in-window print

console.log(`\n${pass} passed, ${fail} failed`);
db.close();
process.exit(fail ? 1 : 0);
