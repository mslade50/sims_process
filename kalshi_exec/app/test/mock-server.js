"use strict";
/**
 * Local, no-credentials mock harness for the Kalshi-exec frontend.
 *
 *   node app/test/mock-server.js   ->   http://127.0.0.1:8099/
 *
 * Serves the real public/ assets and stubs the /api/* surface with synthetic
 * golf-outright data so the Tape brokerage view (chart / VWAP / TWAP / order
 * flow / watchlist / time & sales) can be exercised without Kalshi, D1, secrets
 * or the password gate.
 *
 * The /api/tape and /api/tape/summary stubs run the PRODUCTION SQL (shared
 * tape-sql.js) against an in-memory SQLite seeded with the synthetic tape, so
 * this also smoke-tests the real query path. The order endpoints (/api/send,
 * /api/cancel) are hard-disabled — this harness can never place an order.
 */
const http = require("http");
const fs = require("fs");
const path = require("path");
const { DatabaseSync } = require("node:sqlite");
const { buildTapeWhere, tapeSummarySql, shapeSummaryRow } = require("../shared/tape-sql.js");
const { computePnl } = require("../shared/pnl-calc.js");

// ── ticker classification (mirrors shared/kalshi.ts) ──
const PREFIX_MT = [["KXPGATOUR", "winner"], ["KXPGATOP5", "top_5"], ["KXPGATOP10", "top_10"], ["KXPGATOP20", "top_20"]];
function classify(t) {
  for (const [p, mt] of PREFIX_MT)
    if (t.startsWith(p + "-")) {
      const seg = t.slice(p.length + 1).split("-");
      return { marketType: mt, eventCode: seg[0] || "", playerCode: seg.slice(1).join("-") };
    }
  return { marketType: "", eventCode: "", playerCode: "" };
}
const isGolf = (t) => PREFIX_MT.some(([p]) => t.startsWith(p + "-"));

const PUBLIC = path.join(__dirname, "..", "public");
const PORT = Number(process.env.MOCK_PORT || 8099);

// ── deterministic PRNG so the harness renders the same tape every run ─────────
function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rnd = mulberry32(20260628);
const pick = (arr) => arr[Math.floor(rnd() * arr.length)];

// ── synthetic golf-outright tape ─────────────────────────────────────────────
const EVENT = "PGATRAV"; // Travelers
const SERIES = [
  ["KXPGATOUR", "winner"],
  ["KXPGATOP5", "top_5"],
  ["KXPGATOP10", "top_10"],
  ["KXPGATOP20", "top_20"],
];
const PLAYERS = [
  ["SCHE", 0.22], ["MCIL", 0.12], ["RAHM", 0.10], ["SCHA", 0.08], ["FLEE", 0.07],
  ["HOVL", 0.06], ["FOWL", 0.05], ["THOM", 0.05], ["CANT", 0.05], ["FITZ", 0.04],
  ["SPIE", 0.04], ["MORI", 0.03],
];

function genTrades() {
  const now = Math.floor(Date.now() / 1000);
  const trades = [];
  let tid = 0;
  for (const [code, baseWin] of PLAYERS) {
    for (const [prefix, mt] of SERIES) {
      // base probability scales up for the easier "top N" markets
      const mult = mt === "winner" ? 1 : mt === "top_5" ? 2.4 : mt === "top_10" ? 3.6 : 5.0;
      let price = Math.min(96, Math.max(2, Math.round(baseWin * 100 * mult)));
      const ticker = `${prefix}-${EVENT}-${code}`;
      // number of prints over the last 24h, busier for headline names/markets
      const n = 6 + Math.floor(rnd() * (baseWin * 220 + (mt === "winner" ? 30 : 10)));
      let ts = now - 24 * 3600 + Math.floor(rnd() * 1800);
      let drift = (rnd() - 0.5) * 0.4; // gentle session drift
      for (let i = 0; i < n; i++) {
        ts += 30 + Math.floor(rnd() * 2400);
        if (ts > now) break;
        const step = Math.round((rnd() - 0.5) * 3 + drift);
        price = Math.min(98, Math.max(1, price + step));
        // occasional deci-cent print to exercise formatting
        const yes = rnd() < 0.06 ? price + 0.5 : price;
        // size: log-ish, with rare blocks
        let count = 100 + Math.floor(Math.pow(rnd(), 3) * 1500);
        if (rnd() < 0.04) count = 1500 + Math.floor(rnd() * 9000); // block
        const taker = step >= 0 ? (rnd() < 0.66 ? "yes" : "no") : rnd() < 0.66 ? "no" : "yes";
        trades.push({
          trade_id: "mk" + tid++,
          ticker,
          market_type: mt,
          event_code: EVENT,
          player_code: code,
          ts,
          yes_price: yes,
          no_price: 100 - yes,
          count,
          taker_side: taker,
        });
      }
    }
  }
  return trades;
}

const TRADES = genTrades();

// load the synthetic tape into in-memory SQLite (mirrors the D1 schema)
const db = new DatabaseSync(":memory:");
db.exec(`CREATE TABLE trades (
  trade_id TEXT PRIMARY KEY, ticker TEXT NOT NULL, market_type TEXT NOT NULL,
  event_code TEXT, player_code TEXT, ts INTEGER NOT NULL,
  yes_price INTEGER, no_price INTEGER, count INTEGER NOT NULL, taker_side TEXT);`);
const ins = db.prepare(
  "INSERT INTO trades (trade_id,ticker,market_type,event_code,player_code,ts,yes_price,no_price,count,taker_side) VALUES (?,?,?,?,?,?,?,?,?,?)"
);
for (const t of TRADES) {
  ins.run(t.trade_id, t.ticker, t.market_type, t.event_code, t.player_code, t.ts,
    t.yes_price, t.no_price, t.count, t.taker_side);
}
console.log(`[mock] seeded ${TRADES.length} synthetic trades across ${PLAYERS.length * SERIES.length} markets`);

// ── synthetic PnL book (open + settled, wins + losses) ────────────────────────
function genPnl() {
  const fills = [], settlements = {}, positions = {}, marks = {};
  const tk = (prefix, ev, code) => `${prefix}-${ev}-${code}`;
  const buy = (ticker, side, count, px, ts, fee = 0.5) =>
    fills.push({
      ticker, side, action: "buy", count,
      yes_price: side === "yes" ? px : 1 - px,
      no_price: side === "yes" ? 1 - px : px,
      fee_cost: fee, ts,
    });
  const T0 = Math.floor(Date.now() / 1000) - 5 * 86400;

  // OPEN — YES winner, in the money (mark above cost)
  let t = tk("KXPGATOUR", "PGATRAV", "SCHE");
  buy(t, "yes", 500, 0.18, T0); positions[t] = 500; marks[t] = { yes_bid: 0.21, yes_ask: 0.23 };
  // OPEN — YES top_5, underwater
  t = tk("KXPGATOP5", "PGATRAV", "MCIL");
  buy(t, "yes", 300, 0.4, T0 + 100); positions[t] = 300; marks[t] = { yes_bid: 0.35, yes_ask: 0.37 };
  // OPEN — NO top_20 (fade), slightly up
  t = tk("KXPGATOP20", "PGATRAV", "FLEE");
  buy(t, "no", 400, 0.3, T0 + 200); positions[t] = -400; marks[t] = { yes_bid: 0.66, yes_ask: 0.68 };
  // OPEN — YES winner, flat
  t = tk("KXPGATOUR", "PGATRAV", "FOWL");
  buy(t, "yes", 250, 0.25, T0 + 300); positions[t] = 250; marks[t] = { yes_bid: 0.24, yes_ask: 0.26 };

  // SETTLED win (held to settlement)
  t = tk("KXPGATOUR", "PGCMEM", "RAHM");
  buy(t, "yes", 200, 0.12, T0 - 3 * 86400);
  settlements[t] = { market_result: "yes", yes_count: 200, no_count: 0, settled_time: "2026-06-22" };
  // SETTLED loss
  t = tk("KXPGATOP10", "PGCMEM", "SCHA");
  buy(t, "yes", 150, 0.55, T0 - 3 * 86400 + 50);
  settlements[t] = { market_result: "no", yes_count: 150, no_count: 0, settled_time: "2026-06-22" };
  // SETTLED win on the NO side (player missed the cut → NO top_20 paid)
  t = tk("KXPGATOP20", "PGCMEM", "MORI");
  buy(t, "no", 300, 0.45, T0 - 3 * 86400 + 80);
  settlements[t] = { market_result: "no", yes_count: 0, no_count: 300, settled_time: "2026-06-22" };

  return { fills, settlements, positions, marks };
}
const PNL = genPnl();

// ── synthetic model-edge proposals (overlay on the order ticket) ──────────────
function genProposals() {
  const rows = [];
  const scan_ts = new Date().toISOString();
  for (const [code, baseWin] of PLAYERS) {
    for (const [prefix, mt] of [["KXPGATOUR", "winner"], ["KXPGATOP5", "top_5"]]) {
      const mult = mt === "winner" ? 1 : 2.4;
      const simProb = Math.min(0.95, Math.max(0.02, baseWin * mult));
      const midC = Math.round(simProb * 100) + Math.round((rnd() - 0.5) * 8); // market ¢
      const edgePp = simProb * 100 - midC;
      rows.push({
        ticker: `${prefix}-${EVENT}-${code}`,
        market_type: mt, player_code: code, event_code: EVENT,
        side: edgePp >= 0 ? "yes" : "no",
        sim_prob: simProb,
        edge_pp: Math.round(edgePp * 10) / 10,
        best_bid: (midC - 1) / 100, best_ask: (midC + 1) / 100,
        post_price: Math.round(simProb * 100) / 100,
        kelly_f: Math.max(0, edgePp) / 100,
        scan_ts,
      });
    }
  }
  return { rows, scan_ts };
}
const PROPOSALS = genProposals();

// ── helpers ───────────────────────────────────────────────────────────────────
const MIME = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml", ".ico": "image/x-icon" };
function sendJson(res, obj, status = 200) {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(obj));
}
function paramObj(u) {
  return {
    from: u.searchParams.get("from"), to: u.searchParams.get("to"),
    min_price: u.searchParams.get("min_price"), max_price: u.searchParams.get("max_price"),
    min_size: u.searchParams.get("min_size"), market: u.searchParams.get("market"),
  };
}

function apiTape(u) {
  const { whereSql, binds } = buildTapeWhere(paramObj(u));
  const limit = Math.min(parseInt(u.searchParams.get("limit") || "200", 10), 1000);
  const sql =
    "SELECT trade_id, ticker, market_type, event_code, player_code, ts, yes_price, no_price, count, taker_side FROM trades" +
    whereSql + " ORDER BY ts DESC LIMIT ?";
  return { trades: db.prepare(sql).all(...binds, limit) };
}
function apiTapeSummary(u) {
  const { whereSql, binds } = buildTapeWhere(paramObj(u));
  const limit = Math.min(parseInt(u.searchParams.get("limit") || "200", 10), 500);
  return { rows: db.prepare(tapeSummarySql(whereSql)).all(...binds, limit).map(shapeSummaryRow) };
}

const server = http.createServer((req, res) => {
  const u = new URL(req.url, "http://localhost");
  const p = u.pathname;

  // ── API stubs ──
  if (p === "/api/balance") return sendJson(res, { balance: 421536, portfolio_value: 588120 });
  if (p === "/api/tape") return sendJson(res, apiTape(u));
  if (p === "/api/tape/summary") return sendJson(res, apiTapeSummary(u));
  if (p === "/api/positions") return sendJson(res, { positions: [] });
  if (p === "/api/orders") return sendJson(res, { orders: [] });
  if (p === "/api/orderbook") {
    const ticker = u.searchParams.get("ticker") || "";
    const cls = classify(ticker);
    const pr = PROPOSALS.rows.find((r) => r.ticker === ticker);
    const mid = pr ? Math.round((pr.best_bid + pr.best_ask) * 50) : 12; // cents
    const yes = [], no = [];
    for (let i = 0; i < 8; i++) {
      yes.push({ price: Math.max(1, mid - i), qty: 100 + Math.floor(rnd() * 1200) });
      no.push({ price: Math.max(1, 100 - mid - i), qty: 100 + Math.floor(rnd() * 1200) });
    }
    return sendJson(res, { ticker, yes, no, market_type: cls.marketType });
  }
  if (p === "/api/proposals") {
    if (req.method === "POST") return sendJson(res, { error: "mock harness: proposals push disabled" }, 403);
    return sendJson(res, { rows: PROPOSALS.rows, scan_ts: PROPOSALS.scan_ts, count: PROPOSALS.rows.length });
  }
  if (p === "/api/pnl") {
    const { rows, totals } = computePnl({ ...PNL, classify, isGolf });
    return sendJson(res, { rows, totals, fetched_ts: new Date().toISOString(),
      store: { fills: PNL.fills.length, settlements: Object.keys(PNL.settlements).length } });
  }
  if (p === "/api/markets") {
    const type = u.searchParams.get("type") || "winner";
    const prefix = SERIES.find(([, mt]) => mt === type)?.[0] || "KXPGATOUR";
    return sendJson(res, {
      markets: PLAYERS.map(([code]) => ({
        ticker: `${prefix}-${EVENT}-${code}`, subtitle: code, title: code,
        yes_bid: 10, yes_ask: 12, no_bid: 88, no_ask: 90,
      })),
    });
  }
  // order paths are hard-disabled in the harness
  if (p === "/api/send" || p === "/api/cancel") {
    return sendJson(res, { error: "mock harness: order endpoints are disabled" }, 403);
  }

  // ── static ──
  let file = p === "/" ? "index.html" : p.replace(/^\/+/, "");
  const full = path.join(PUBLIC, file);
  if (!full.startsWith(PUBLIC) || !fs.existsSync(full)) {
    res.writeHead(404); return res.end("not found");
  }
  res.writeHead(200, { "Content-Type": MIME[path.extname(full)] || "application/octet-stream" });
  fs.createReadStream(full).pipe(res);
});

server.listen(PORT, "127.0.0.1", () => console.log(`[mock] http://127.0.0.1:${PORT}/`));
