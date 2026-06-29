"use strict";
/**
 * Pure, dependency-free tape analytics. Loaded as a classic <script> in the
 * browser (attaches to window.TapeAnalytics) and require()-able from Node for
 * unit tests (see app/test/tape-analytics.test.js).
 *
 * A "trade" is a row from /api/tape:
 *   { trade_id, ticker, market_type, event_code, player_code,
 *     ts (unix sec), yes_price (cents), no_price (cents),
 *     count (contracts), taker_side ("yes"|"no"|null) }
 *
 * Price convention throughout: yes_price in cents. A YES-aggressor print
 * (taker_side === "yes") is treated as BUY pressure (+), a NO-aggressor print
 * as SELL pressure (-) — the standard order-flow sign for the YES leg.
 */
(function (root, factory) {
  const mod = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = mod;
  if (root) root.TapeAnalytics = mod;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  const asc = (a, b) => a.ts - b.ts;
  const sortedAsc = (trades) => trades.slice().sort(asc);

  /** Signed contribution to order-flow delta: +size for a YES lift, -size for a NO hit. */
  function signedVolume(t) {
    if (t.taker_side === "yes") return t.count;
    if (t.taker_side === "no") return -t.count;
    return 0;
  }

  /** Volume-weighted average price (cents). Falls back to last print if no volume. */
  function vwap(trades) {
    let pv = 0, vol = 0, last = null;
    for (const t of trades) {
      pv += t.yes_price * t.count;
      vol += t.count;
      last = t.yes_price;
    }
    return vol > 0 ? pv / vol : last;
  }

  /**
   * Time-weighted average price (cents) as a step function: each print's price
   * is "in effect" until the next print; the final print holds until `toTs`
   * (default: the last print's ts, i.e. integrate only over observed time).
   * All-same-timestamp data degrades to a simple mean.
   */
  function twap(trades, toTs) {
    const ts = sortedAsc(trades);
    if (!ts.length) return null;
    if (ts.length === 1) return ts[0].yes_price;
    const end = toTs != null ? toTs : ts[ts.length - 1].ts;
    let num = 0, den = 0;
    for (let i = 0; i < ts.length; i++) {
      const next = i < ts.length - 1 ? ts[i + 1].ts : end;
      const dt = Math.max(0, next - ts[i].ts);
      num += ts[i].yes_price * dt;
      den += dt;
    }
    if (den <= 0) return ts.reduce((s, t) => s + t.yes_price, 0) / ts.length;
    return num / den;
  }

  /** One-shot summary of a print set (a market, a player, or the whole window). */
  function summarize(trades, toTs) {
    if (!trades || !trades.length) return null;
    const ts = sortedAsc(trades);
    let vol = 0, pv = 0, buyVol = 0, sellVol = 0, prints = 0;
    let high = -Infinity, low = Infinity;
    for (const t of ts) {
      vol += t.count;
      pv += t.yes_price * t.count;
      prints++;
      const sv = signedVolume(t);
      if (sv > 0) buyVol += t.count;
      else if (sv < 0) sellVol += t.count;
      if (t.yes_price > high) high = t.yes_price;
      if (t.yes_price < low) low = t.yes_price;
    }
    const open = ts[0].yes_price;
    const close = ts[ts.length - 1].yes_price;
    const signed = buyVol + sellVol;
    return {
      prints,
      volume: vol,
      vwap: vol > 0 ? pv / vol : close,
      twap: twap(ts, toTs),
      last: close,
      open,
      high,
      low,
      change: close - open,
      changePct: open ? ((close - open) / open) * 100 : 0,
      buyVol,
      sellVol,
      delta: buyVol - sellVol,
      imbalance: signed > 0 ? (buyVol - sellVol) / signed : 0, // -1..+1
      firstTs: ts[0].ts,
      lastTs: ts[ts.length - 1].ts,
      ticker: ts[ts.length - 1].ticker,
      player_code: ts[ts.length - 1].player_code,
      market_type: ts[ts.length - 1].market_type,
    };
  }

  /**
   * Running per-print series for charting. Returns column arrays aligned by
   * index (uPlot-friendly): ts, price, vwap (running), twap (running),
   * cumDelta (running signed volume), size.
   */
  function cumulativeSeries(trades) {
    const ts = sortedAsc(trades);
    const out = { ts: [], price: [], vwap: [], twap: [], cumDelta: [], size: [] };
    let pv = 0, vol = 0, twNum = 0, twDen = 0, delta = 0;
    let prevTs = null, prevPrice = null;
    for (const t of ts) {
      if (prevTs != null) {
        const dt = Math.max(0, t.ts - prevTs);
        twNum += prevPrice * dt;
        twDen += dt;
      }
      pv += t.yes_price * t.count;
      vol += t.count;
      delta += signedVolume(t);
      out.ts.push(t.ts);
      out.price.push(t.yes_price);
      out.vwap.push(vol > 0 ? pv / vol : t.yes_price);
      out.twap.push(twDen > 0 ? twNum / twDen : t.yes_price);
      out.cumDelta.push(delta);
      out.size.push(t.count);
      prevTs = t.ts;
      prevPrice = t.yes_price;
    }
    return out;
  }

  /** Time-bucketed OHLCV + per-bucket VWAP and buy/sell split. */
  function bucketOHLCV(trades, bucketSec) {
    const b = Math.max(1, bucketSec | 0);
    const buckets = new Map();
    for (const t of sortedAsc(trades)) {
      const key = Math.floor(t.ts / b) * b;
      let o = buckets.get(key);
      if (!o) {
        o = { ts: key, open: t.yes_price, high: t.yes_price, low: t.yes_price,
              close: t.yes_price, volume: 0, pv: 0, buyVol: 0, sellVol: 0 };
        buckets.set(key, o);
      }
      o.high = Math.max(o.high, t.yes_price);
      o.low = Math.min(o.low, t.yes_price);
      o.close = t.yes_price;
      o.volume += t.count;
      o.pv += t.yes_price * t.count;
      const sv = signedVolume(t);
      if (sv > 0) o.buyVol += t.count;
      else if (sv < 0) o.sellVol += t.count;
    }
    return Array.from(buckets.values())
      .sort(asc)
      .map((o) => ({ ts: o.ts, open: o.open, high: o.high, low: o.low, close: o.close,
                     volume: o.volume, vwap: o.volume > 0 ? o.pv / o.volume : o.close,
                     buyVol: o.buyVol, sellVol: o.sellVol }));
  }

  /** Map ticker -> trades[] (insertion order preserved). */
  function groupBy(trades, keyFn) {
    const m = new Map();
    for (const t of trades) {
      const k = keyFn(t);
      let a = m.get(k);
      if (!a) (a = [], m.set(k, a));
      a.push(t);
    }
    return m;
  }

  /**
   * Per-ticker summary rows (the watchlist), sorted by volume desc. Each row is
   * a summarize() result plus a downsampled price series for an inline sparkline.
   */
  function watchlist(trades, toTs, sparkPoints = 24) {
    const byTicker = groupBy(trades, (t) => t.ticker);
    const rows = [];
    for (const [, group] of byTicker) {
      const s = summarize(group, toTs);
      if (!s) continue;
      s.spark = downsample(sortedAsc(group).map((t) => t.yes_price), sparkPoints);
      rows.push(s);
    }
    return rows.sort((a, b) => b.volume - a.volume);
  }

  /** Nearest-bin downsample of a numeric series to at most n points. */
  function downsample(values, n) {
    if (values.length <= n) return values.slice();
    const out = [];
    for (let i = 0; i < n; i++) {
      out.push(values[Math.round((i * (values.length - 1)) / (n - 1))]);
    }
    return out;
  }

  /**
   * SVG polyline "points" string mapping a numeric series into a w×h box.
   * Pure (no DOM) so it is unit-testable and usable inline.
   */
  function sparkPoints(values, w, h, pad = 1) {
    if (!values || values.length === 0) return "";
    let min = Infinity, max = -Infinity;
    for (const v of values) { if (v < min) min = v; if (v > max) max = v; }
    const span = max - min || 1;
    const innerW = w - pad * 2, innerH = h - pad * 2;
    const n = values.length;
    return values
      .map((v, i) => {
        const x = pad + (n === 1 ? innerW / 2 : (i * innerW) / (n - 1));
        const y = pad + innerH - ((v - min) / span) * innerH;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  }

  return {
    signedVolume,
    vwap,
    twap,
    summarize,
    cumulativeSeries,
    bucketOHLCV,
    groupBy,
    watchlist,
    downsample,
    sparkPoints,
  };
});
