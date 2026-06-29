"use strict";
// Dependency-free unit tests for tape-analytics. Run: node app/test/tape-analytics.test.js
const A = require("../public/tape-analytics.js");

let pass = 0, fail = 0;
function eq(name, got, want, tol = 1e-9) {
  const ok = typeof want === "number" ? Math.abs(got - want) <= tol : got === want;
  if (ok) { pass++; }
  else { fail++; console.error(`FAIL ${name}: got ${JSON.stringify(got)} want ${JSON.stringify(want)}`); }
}
function truthy(name, got) {
  if (got) pass++; else { fail++; console.error(`FAIL ${name}: expected truthy, got ${JSON.stringify(got)}`); }
}

const T = (ts, price, count, taker_side, ticker = "KXPGATOUR-X-A") =>
  ({ ts, yes_price: price, no_price: 100 - price, count, taker_side, ticker, player_code: "A", market_type: "winner" });

// ── vwap ─────────────────────────────────────────────────────────────────────
// (10*100 + 20*200 + 30*300) / 600 = 14000/600
eq("vwap weighted", A.vwap([T(1, 10, 100), T(2, 20, 200), T(3, 30, 300)]), 14000 / 600);
eq("vwap single", A.vwap([T(1, 42, 7)]), 42);

// ── twap (step function over observed time) ──────────────────────────────────
// prices 10 (held 1s, ts1->ts2) then 20 (held 4s, ts2->ts6 via toTs=6). last=20
// num = 10*1 + 20*4 = 90, den = 5 -> 18
eq("twap step", A.twap([T(1, 10, 1), T(2, 20, 1)], 6), 18);
// no toTs -> integrate only to last ts: only the first segment counts -> 10
eq("twap no-end", A.twap([T(1, 10, 1), T(2, 20, 1)]), 10);
// all same ts -> simple mean
eq("twap degenerate", A.twap([T(5, 10, 1), T(5, 30, 1)]), 20);
eq("twap single", A.twap([T(1, 55, 9)]), 55);

// ── signedVolume ─────────────────────────────────────────────────────────────
eq("signed yes", A.signedVolume(T(1, 50, 7, "yes")), 7);
eq("signed no", A.signedVolume(T(1, 50, 7, "no")), -7);
eq("signed null", A.signedVolume(T(1, 50, 7, null)), 0);

// ── summarize ────────────────────────────────────────────────────────────────
const set = [T(1, 40, 100, "yes"), T(2, 60, 300, "no"), T(3, 50, 100, "yes")];
const s = A.summarize(set, 3);
eq("sum prints", s.prints, 3);
eq("sum volume", s.volume, 500);
eq("sum open", s.open, 40);
eq("sum close/last", s.last, 50);
eq("sum high", s.high, 60);
eq("sum low", s.low, 40);
eq("sum buyVol", s.buyVol, 200); // 100 + 100 yes-aggressor
eq("sum sellVol", s.sellVol, 300); // 300 no-aggressor
eq("sum delta", s.delta, -100);
eq("sum imbalance", s.imbalance, (200 - 300) / 500);
eq("sum vwap", s.vwap, (40 * 100 + 60 * 300 + 50 * 100) / 500);
eq("summarize empty", A.summarize([]), null);

// ── cumulativeSeries ─────────────────────────────────────────────────────────
const cs = A.cumulativeSeries(set);
eq("cs len", cs.ts.length, 3);
eq("cs vwap[0]", cs.vwap[0], 40);
eq("cs vwap[2]", cs.vwap[2], (40 * 100 + 60 * 300 + 50 * 100) / 500);
eq("cs cumDelta", cs.cumDelta[2], 100 - 300 + 100); // -100
eq("cs price last", cs.price[2], 50);
truthy("cs twap monotone-defined", cs.twap.every((x) => typeof x === "number"));

// ── bucketOHLCV ──────────────────────────────────────────────────────────────
const bk = A.bucketOHLCV(
  [T(0, 10, 5, "yes"), T(30, 20, 5, "no"), T(70, 15, 5, "yes")],
  60
);
eq("bucket count", bk.length, 2);
eq("bucket0 open", bk[0].open, 10);
eq("bucket0 close", bk[0].close, 20);
eq("bucket0 high", bk[0].high, 20);
eq("bucket0 vol", bk[0].volume, 10);
eq("bucket0 buyVol", bk[0].buyVol, 5);
eq("bucket0 sellVol", bk[0].sellVol, 5);
eq("bucket1 open", bk[1].open, 15);

// ── watchlist (group + sort by volume) ───────────────────────────────────────
const multi = [
  T(1, 50, 100, "yes", "KXPGATOUR-X-A"),
  T(2, 51, 900, "no", "KXPGATOUR-X-B"),
  T(3, 52, 200, "yes", "KXPGATOUR-X-A"),
];
const wl = A.watchlist(multi, 3);
eq("wl rows", wl.length, 2);
eq("wl top by volume is B", wl[0].ticker, "KXPGATOUR-X-B");
eq("wl A volume", wl.find((r) => r.ticker === "KXPGATOUR-X-A").volume, 300);
truthy("wl spark present", Array.isArray(wl[0].spark));

// ── downsample / sparkPoints ─────────────────────────────────────────────────
eq("downsample passthrough", A.downsample([1, 2, 3], 5).length, 3);
eq("downsample shrinks", A.downsample([1, 2, 3, 4, 5, 6, 7, 8], 4).length, 4);
truthy("sparkPoints nonempty", A.sparkPoints([1, 2, 3], 60, 16).length > 0);
eq("sparkPoints empty", A.sparkPoints([], 60, 16), "");

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
