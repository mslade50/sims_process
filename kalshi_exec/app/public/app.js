"use strict";
const $ = (id) => document.getElementById(id);
const fmtUSD = (cents) =>
  cents == null ? "—" : "$" + (cents / 100).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const fmtTime = (ts) => new Date(ts * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (r.status === 401) {
    window.location = "/__login";
    throw new Error("session expired");
  }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || data.detail || ("HTTP " + r.status));
  return data;
}

// ── Reference (human-readable names) ─────────────────────────────────────────
const ref = { players: {}, events: {}, eventsList: [] };
let currentEvent = ""; // event_code filter for the tape ("" = all)

const MARKET_LABELS = { winner: "Winner", top_5: "Top 5", top_10: "Top 10", top_20: "Top 20", h2h: "Matchup" };
const marketLabel = (t) => MARKET_LABELS[t] || t || "";
const playerName = (code) => ref.players[code] || code || "";
const eventName = (code) => ref.events[code] || code || "";

async function loadReference() {
  try {
    const d = await api("/api/reference");
    ref.players = d.players || {};
    ref.events = d.events || {};
    ref.eventsList = d.eventsList || [];
    populateEventSelect();
    // re-render whatever is showing now that names are available
    if (allMarkets.length) renderMarketList();
    if (positionsData.length) renderPositions();
    if (selected) loadOrderbook(selected.ticker);
    if ($("tape").classList.contains("active")) loadTape();
    if ($("pnl").classList.contains("active")) renderPnL();
  } catch (e) {
    /* names are best-effort; UI falls back to codes */
  }
}

function populateEventSelect() {
  const sel = $("eventSelect");
  if (!sel) return;
  sel.innerHTML =
    `<option value="">all events</option>` +
    ref.eventsList.map((e) => `<option value="${e.code}">${e.name}</option>`).join("");
  // Default to the busiest active event the first time we learn the list.
  if (!currentEvent && ref.eventsList.length) currentEvent = ref.eventsList[0].code;
  sel.value = currentEvent;
}

// ── Tabs ────────────────────────────────────────────────────────────────────
function activateTab(name) {
  const btn = document.querySelector(`.tab[data-tab="${name}"]`);
  if (!btn) return;
  document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
  document.querySelectorAll(".tabpane").forEach((p) => p.classList.remove("active"));
  btn.classList.add("active");
  $(name).classList.add("active");
  if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
  if (name === "tape") loadTape();
  if (name === "pnl") loadPnL();
}
document.querySelectorAll(".tab").forEach((btn) =>
  btn.addEventListener("click", () => activateTab(btn.dataset.tab))
);
// Deep-link: open the tab named in the URL hash (e.g. .../#tape).
if (location.hash && document.querySelector(`.tab[data-tab="${location.hash.slice(1)}"]`)) {
  addEventListener("DOMContentLoaded", () => activateTab(location.hash.slice(1)));
}

// ── Balance ──────────────────────────────────────────────────────────────────
async function loadBalance() {
  try {
    const b = await api("/api/balance");
    $("balance").textContent = `bal ${fmtUSD(b.balance)} · port ${fmtUSD(b.portfolio_value)}`;
  } catch (e) {
    $("balance").textContent = "balance error";
  }
}

// ── Order ticket ─────────────────────────────────────────────────────────────
let allMarkets = [];
let selected = null;
let side = "yes";
let proposalsByTicker = {}; // model-edge overlay keyed by ticker

// Pull model-edge proposals (pushed from the local sim). Optional — the ticket
// works fine without them; this just overlays model fair / edge per market.
async function loadProposals() {
  try {
    const d = await api("/api/proposals");
    proposalsByTicker = {};
    for (const r of d.rows || []) proposalsByTicker[r.ticker] = r;
    if (allMarkets.length) renderMarketList();
    if (selected) renderModelEdge(selected.ticker);
  } catch (e) {
    /* proposals are optional */
  }
}

function edgeTag(ticker) {
  const pr = proposalsByTicker[ticker];
  if (!pr || pr.edge_pp == null) return "";
  return `  [${pr.edge_pp >= 0 ? "+" : ""}${Number(pr.edge_pp).toFixed(1)}pp]`;
}

function renderModelEdge(ticker) {
  const el = $("modelEdge");
  if (!el) return;
  const pr = proposalsByTicker[ticker];
  if (!pr) { el.style.display = "none"; el.innerHTML = ""; return; }
  el.style.display = "block";
  const fair = pr.sim_prob != null ? (pr.sim_prob * 100).toFixed(1) + "¢" : "—";
  const edge = pr.edge_pp != null ? (pr.edge_pp >= 0 ? "+" : "") + Number(pr.edge_pp).toFixed(1) + "pp" : "—";
  const post = pr.post_price != null ? (pr.post_price * 100).toFixed(1) + "¢" : null;
  const cls = (pr.edge_pp || 0) >= 0 ? "up" : "down";
  el.innerHTML =
    `<span class="me-label">MODEL</span> fair <b>${fair}</b> · edge <b class="${cls}">${edge}</b>` +
    (pr.side ? ` <span class="pill ${pr.side}">${String(pr.side).toUpperCase()}</span>` : "") +
    (post ? ` <span class="muted small">suggest ${post}</span>` : "");
}

async function loadMarkets() {
  const type = $("series").value;
  $("market").innerHTML = `<option>loading…</option>`;
  try {
    const d = await api(`/api/markets?type=${type}`);
    allMarkets = d.markets || [];
    renderMarketList();
    // Default-select the first (most-liquid) market so the book + model edge
    // are visible immediately, without forcing a click.
    if (!selected && allMarkets.length) selectMarket(allMarkets[0].ticker);
  } catch (e) {
    $("market").innerHTML = `<option>error: ${e.message}</option>`;
  }
}

function renderMarketList() {
  const q = $("marketSearch").value.trim().toLowerCase();
  const sel = $("market");
  const items = allMarkets
    .filter((m) => !q || (m.subtitle + " " + m.ticker + " " + (m.title || "")).toLowerCase().includes(q))
    .slice(0, 300);
  sel.innerHTML = items
    .map((m) => `<option value="${m.ticker}">${m.subtitle || m.ticker} — ${m.yes_bid ?? "·"}/${m.yes_ask ?? "·"}¢${edgeTag(m.ticker)}</option>`)
    .join("");
}

function selectMarket(ticker) {
  selected = allMarkets.find((m) => m.ticker === ticker);
  if (!selected) return;
  $("ticketBody").hidden = false;
  $("selectedMarket").innerHTML = `<b>${selected.subtitle || selected.ticker}</b><br><span class="muted small">${selected.ticker}</span>`;
  renderModelEdge(ticker);
  prefillPrice();
  loadOrderbook(ticker);
}

function prefillPrice() {
  if (!selected) return;
  const v = side === "yes" ? selected.yes_bid : selected.no_bid;
  if (v != null) $("price").value = v;
}

// Top-of-book for a binary market: best YES bid = highest YES bid; best YES ask
// = 100 - highest NO bid (a NO buyer at P is a YES seller at 100-P). NO mirrors.
function renderTopOfBook(ob) {
  const yes = ob.yes || [];
  const no = ob.no || [];
  const c1d = (v) => Math.round(v * 10) / 10;
  const yesBid = yes[0] || null;
  const yesAsk = no[0] ? { price: c1d(100 - no[0].price), qty: no[0].qty } : null;
  const noBid = no[0] || null;
  const noAsk = yes[0] ? { price: c1d(100 - yes[0].price), qty: yes[0].qty } : null;
  const cell = (label, lvl, side) =>
    lvl
      ? `<span class="tob-fill" data-side="${side}" data-price="${lvl.price}"><span class="tl">${label}</span><b>${lvl.price}¢</b><span class="tq">×${Math.round(lvl.qty).toLocaleString()}</span></span>`
      : `<span class="tob-none"><span class="tl">${label}</span>—</span>`;
  const spread = yesBid && yesAsk ? `<span class="tob-spread">${(yesAsk.price - yesBid.price).toFixed(1)}¢ wide</span>` : "";
  $("topOfBook").innerHTML =
    `<div class="tob-row"><span class="tob-side yes">YES</span>${cell("bid", yesBid, "yes")}${cell("ask", yesAsk, "yes")}${spread}</div>` +
    `<div class="tob-row"><span class="tob-side no">NO</span>${cell("bid", noBid, "no")}${cell("ask", noAsk, "no")}</div>`;
}

async function loadOrderbook(ticker) {
  let name = ticker;
  if (selected && selected.ticker === ticker) {
    const lbl = marketLabel(selected.marketType);
    name = (selected.subtitle || ticker) + (lbl ? ` · ${lbl}` : "");
  }
  $("bookTicker").textContent = name;
  $("yesBook").innerHTML = $("noBook").innerHTML = "…";
  $("topOfBook").innerHTML = "";
  try {
    const ob = await api(`/api/orderbook?ticker=${encodeURIComponent(ticker)}`);
    renderTopOfBook(ob);
    const render = (levels, s) =>
      (levels || [])
        .slice(0, 10)
        .map((l) => `<div class="lvl" data-side="${s}" data-price="${l.price}"><span class="p">${l.price}¢</span><span class="q">${Math.round(l.qty).toLocaleString()}</span></div>`)
        .join("") || `<div class="muted small">empty</div>`;
    $("yesBook").innerHTML = render(ob.yes, "yes");
    $("noBook").innerHTML = render(ob.no, "no");
    document.querySelectorAll(".lvl, .tob-fill").forEach((el) =>
      el.addEventListener("click", () => {
        setSide(el.dataset.side);
        $("price").value = el.dataset.price;
      })
    );
  } catch (e) {
    $("yesBook").innerHTML = $("noBook").innerHTML = `<div class="muted small">${e.message}</div>`;
  }
}

function setSide(s) {
  side = s;
  document.querySelectorAll(".side").forEach((b) => b.classList.toggle("active", b.dataset.side === s));
  prefillPrice();
}

function toLocalInput(d) {
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}`;
}

async function placeOrder() {
  if (!selected) return;
  const cents = Number($("price").value);
  const count = parseInt($("qty").value, 10);
  let kill_ts = null;
  if ($("killOn").checked && $("killTime").value) {
    kill_ts = Math.floor(new Date($("killTime").value).getTime() / 1000);
  }
  const order = { ticker: selected.ticker, side, price_cents: cents, count };
  if (kill_ts) order.kill_ts = kill_ts;

  const killTxt = kill_ts ? `, kill ${fmtTime(kill_ts)} (+60s force)` : ", GTC";
  if (!confirm(`Place ${count} × ${side.toUpperCase()} @ ${cents}¢ on ${selected.subtitle || selected.ticker}${killTxt}?`)) return;

  const msg = $("ticketMsg");
  msg.className = "msg";
  msg.textContent = "placing…";
  try {
    const res = await api("/api/send", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ orders: [order] }) });
    const d = res.details[0] || {};
    if (d.status === "posted") {
      msg.className = "msg ok";
      msg.textContent = `✓ posted — order_id ${d.order_id}${d.kill_warning ? "\n⚠ " + d.kill_warning : ""}`;
      loadOrders();
      loadPositions();
      loadBalance();
    } else {
      msg.className = "msg err";
      msg.textContent = `✗ ${d.status}: ${d.message || "rejected"}`;
    }
  } catch (e) {
    msg.className = "msg err";
    msg.textContent = "✗ " + e.message;
  }
}

// ── Positions & orders ───────────────────────────────────────────────────────
let positionsData = [];
async function loadPositions() {
  try {
    const d = await api("/api/positions");
    positionsData = d.positions || [];
    renderPositions();
  } catch (e) {
    $("positions").innerHTML = `<div class="msg err">${e.message}</div>`;
  }
}

function renderPositions() {
  if (!positionsData.length) {
    $("positions").innerHTML = `<div class="muted small">no open golf positions</div>`;
    return;
  }
  $("positions").innerHTML =
    `<table><thead><tr><th>Player</th><th>Market</th><th>Side</th><th class="num">Pos</th><th class="num">Avg</th><th class="num">Exposure</th></tr></thead><tbody>` +
    positionsData
      .map(
        (p) =>
          `<tr><td>${playerName(p.player_code)}</td><td class="small">${marketLabel(p.market_type)}</td>` +
          `<td><span class="pill ${p.side || ""}">${(p.side || "·").toUpperCase()}</span></td>` +
          `<td class="num">${(p.contracts || 0).toLocaleString()}</td>` +
          `<td class="num">${p.avg_cost ? (p.avg_cost * 100).toFixed(1) + "¢" : "—"}</td>` +
          `<td class="num">${usd(p.exposure)}</td></tr>`
      )
      .join("") +
    `</tbody></table>`;
}

async function loadOrders() {
  try {
    const d = await api("/api/orders");
    if (!d.orders.length) return ($("orders").innerHTML = `<div class="muted small">no resting golf orders</div>`);
    $("orders").innerHTML =
      `<table><thead><tr><th>Ticker</th><th>Side</th><th class="num">Px</th><th class="num">Rem</th><th>Kill</th><th></th></tr></thead><tbody>` +
      d.orders
        .map((o) => {
          const px = typeof o.price === "string" ? (parseFloat(o.price) * 100).toFixed(1) + "¢" : o.price + "¢";
          const kill = o.hard_kill_ts ? fmtTime(o.kill_ts) : o.expiration_ts ? fmtTime(o.expiration_ts) : "—";
          return `<tr><td>${o.ticker}</td><td><span class="pill ${o.side}">${o.side}</span></td><td class="num">${px}</td><td class="num">${o.remaining}</td><td class="small">${kill}</td><td><button class="btn tiny cancel" data-id="${o.order_id}" data-ticker="${o.ticker}">✕</button></td></tr>`;
        })
        .join("") +
      `</tbody></table>`;
    document.querySelectorAll(".cancel").forEach((b) =>
      b.addEventListener("click", () => cancelOrder(b.dataset.id, b.dataset.ticker))
    );
  } catch (e) {
    $("orders").innerHTML = `<div class="msg err">${e.message}</div>`;
  }
}

async function cancelOrder(order_id, ticker) {
  if (!confirm(`Cancel order on ${ticker}?`)) return;
  try {
    await api("/api/cancel", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ order_id, ticker }) });
    loadOrders();
  } catch (e) {
    alert("cancel failed: " + e.message);
  }
}

// ── Tape (institutional brokerage view) ──────────────────────────────────────
const A = window.TapeAnalytics;
const PRICE_H = 240,
  FLOW_H = 120;

const tapeState = {
  trades: [], // raw prints (time & sales), DESC
  summary: [], // per-ticker summary rows (full window)
  sparkByTicker: {}, // ticker -> recent price series for sparklines
  focus: null, // focused ticker
  focusPrints: [], // focused ticker prints ASC
  priceChart: null,
  flowChart: null,
  builtFocus: null, // ticker the charts were built for
  seen: new Set(), // trade_ids already shown (new-print flash)
  primed: false, // suppress flash on the first paint / after a filter change
  showPrice: false, // raw price line off by default (VWAP/TWAP only)
};

// cents formatter — handles deci-cent marks (96.0 -> "96", 96.5 -> "96.5")
function c1(v) {
  if (v == null) return "—";
  const r = Math.round(v * 10) / 10;
  return Number.isInteger(r) ? String(r) : r.toFixed(1);
}
function kfmt(v) {
  const a = Math.abs(v);
  if (a >= 1000) return (v / 1000).toFixed(v % 1000 ? 1 : 0) + "k";
  return String(Math.round(v));
}

// Shared filter params (window/price/size/market) used by tape + summary + focus.
function tapeFilterParams() {
  const p = new URLSearchParams();
  const market = $("tMarket").value;
  const ticker = $("tTicker").value.trim();
  if (market) p.set("market", market);
  else if (ticker) p.set("market", ticker);
  if ($("tMinP").value) p.set("min_price", $("tMinP").value);
  if ($("tMaxP").value) p.set("max_price", $("tMaxP").value);
  if ($("tMinSize").value) p.set("min_size", $("tMinSize").value);
  if (currentEvent) p.set("event", currentEvent);
  const win = Number($("tWindow").value);
  if (win > 0) p.set("from", String(Math.floor(Date.now() / 1000) - win));
  return p;
}

async function loadTape() {
  const base = tapeFilterParams();
  const rawP = new URLSearchParams(base);
  rawP.set("limit", "500");
  const sumP = new URLSearchParams(base);
  sumP.set("limit", "300");
  try {
    const [raw, sum] = await Promise.all([
      api("/api/tape?" + rawP.toString()),
      api("/api/tape/summary?" + sumP.toString()).catch(() => ({ rows: [] })),
    ]);
    tapeState.trades = raw.trades || [];
    tapeState.summary = sum.rows || [];
    // Sparkline series from whatever recent prints we have per ticker.
    tapeState.sparkByTicker = {};
    for (const [tk, arr] of A.groupBy(tapeState.trades, (t) => t.ticker)) {
      tapeState.sparkByTicker[tk] = arr.slice().sort((a, b) => a.ts - b.ts).map((t) => t.yes_price);
    }
    const totalVol = tapeState.summary.reduce((s, r) => s + (r.volume || 0), 0);
    $("tapeStatus").textContent =
      `${tapeState.trades.length} prints shown · ${tapeState.summary.length} markets · ${totalVol.toLocaleString()} contracts in window`;
    renderWatchlist();
    renderTimeSales();
    // First time the tape loads, surface the most-active market automatically.
    if (!tapeState.focus && tapeState.summary.length) focusMarket(tapeState.summary[0].ticker);
    else if (tapeState.focus) refreshFocus();
  } catch (e) {
    $("tape-table").innerHTML = `<div class="msg err">${e.message}</div>`;
  }
}

// User changed a filter — re-prime so we don't flash a whole new result set.
function applyTape() {
  tapeState.primed = false;
  tapeState.seen.clear();
  loadTape();
}

// ── Watchlist ────────────────────────────────────────────────────────────────
function sparkSvg(ticker) {
  const vals = tapeState.sparkByTicker[ticker] || [];
  if (vals.length < 2) return `<svg class="spark" width="64" height="18"></svg>`;
  const pts = A.sparkPoints(A.downsample(vals, 24), 64, 18, 2);
  const col = vals[vals.length - 1] >= vals[0] ? "var(--yes)" : "var(--no)";
  return `<svg class="spark" width="64" height="18" viewBox="0 0 64 18"><polyline points="${pts}" fill="none" stroke="${col}" stroke-width="1.3"/></svg>`;
}

function renderWatchlist() {
  const q = ($("wlSearch").value || "").trim().toLowerCase();
  const rows = tapeState.summary.filter(
    (r) => !q || (playerName(r.player_code) + " " + r.ticker + " " + marketLabel(r.market_type)).toLowerCase().includes(q)
  );
  const el = $("watchlist");
  if (!rows.length) {
    el.innerHTML = `<div class="muted small pad">no markets in window</div>`;
    return;
  }
  el.innerHTML = rows
    .map((r) => {
      const signed = r.buyVol + r.sellVol;
      const buyPct = signed > 0 ? Math.round((r.buyVol / signed) * 100) : 50;
      const up = (r.change || 0) >= 0;
      const focused = r.ticker === tapeState.focus;
      return (
        `<div class="wl-row${focused ? " focused" : ""}" data-ticker="${r.ticker}">` +
        `<div class="wl-main"><div class="wl-name">${playerName(r.player_code)}</div>` +
        `<div class="wl-sub muted small">${marketLabel(r.market_type)} · vol ${kfmt(r.volume)}</div></div>` +
        sparkSvg(r.ticker) +
        `<div class="wl-px"><div class="wl-last">${c1(r.last)}¢</div>` +
        `<div class="wl-chg ${up ? "up" : "down"}">${up ? "+" : ""}${c1(r.change)}</div></div>` +
        `<div class="wl-imb" title="${buyPct}% buy / ${100 - buyPct}% sell">` +
        `<div class="imb-bar"><span class="buy" style="width:${buyPct}%"></span><span class="sell" style="width:${100 - buyPct}%"></span></div></div>` +
        `</div>`
      );
    })
    .join("");
  el.querySelectorAll(".wl-row").forEach((row) =>
    row.addEventListener("click", () => focusMarket(row.dataset.ticker))
  );
}

// ── Focused market ───────────────────────────────────────────────────────────
async function focusMarket(ticker) {
  tapeState.focus = ticker;
  const row = tapeState.summary.find((r) => r.ticker === ticker);
  $("focusName").textContent = row ? `${playerName(row.player_code)} · ${marketLabel(row.market_type)}` : ticker;
  $("focusName").classList.remove("muted");
  $("focusTicker").textContent = ticker + (row && eventName(row.event_code) ? ` · ${eventName(row.event_code)}` : "");
  renderWatchlist();
  if ($("tsFocusOnly").checked) renderTimeSales();
  await refreshFocus();
}

async function refreshFocus() {
  const ticker = tapeState.focus;
  if (!ticker) return;
  const p = tapeFilterParams();
  p.set("market", ticker); // full ticker → LIKE match; we hard-filter below
  p.set("limit", "1000");
  try {
    const d = await api("/api/tape?" + p.toString());
    tapeState.focusPrints = (d.trades || [])
      .filter((t) => t.ticker === ticker)
      .sort((a, b) => a.ts - b.ts);
  } catch (e) {
    /* keep prior prints */
  }
  renderFocusStats();
  renderFocusCharts();
}

function statCell(label, val, cls) {
  return `<div class="stat"><div class="sl">${label}</div><div class="sv ${cls || ""}">${val}</div></div>`;
}

function renderFocusStats() {
  const prints = tapeState.focusPrints;
  const s = A.summarize(prints, Math.floor(Date.now() / 1000));
  if (!s) {
    $("focusStats").innerHTML = "";
    $("focusLast").textContent = "";
    return;
  }
  const up = s.change >= 0;
  $("focusLast").innerHTML =
    `<span class="fl-px">${c1(s.last)}¢</span> <span class="chg ${up ? "up" : "down"}">${up ? "+" : ""}${c1(s.change)} (${up ? "+" : ""}${s.changePct.toFixed(1)}%)</span>`;
  const signed = s.buyVol + s.sellVol;
  const buyPct = signed > 0 ? Math.round((s.buyVol / signed) * 100) : 50;
  $("focusStats").innerHTML =
    statCell("VWAP", c1(s.vwap) + "¢") +
    statCell("TWAP", c1(s.twap) + "¢") +
    statCell("Open", c1(s.open) + "¢") +
    statCell("Range", c1(s.low) + "–" + c1(s.high) + "¢") +
    statCell("Volume", s.volume.toLocaleString()) +
    statCell("Prints", s.prints.toLocaleString()) +
    statCell("Flow", `<b class="buy">${buyPct}%</b> / <b class="sell">${100 - buyPct}%</b>`) +
    statCell("Net Δ", (s.delta >= 0 ? "+" : "") + s.delta.toLocaleString(), s.delta >= 0 ? "up" : "down");
}

function chooseBucket(win) {
  const raw = win / 80; // ~80 bars across the window
  const steps = [30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400, 43200, 86400];
  for (const st of steps) if (raw <= st) return st;
  return 86400;
}

const axisStroke = "#8b949e";
const gridStroke = "rgba(110,118,129,.13)";
function axisX() {
  return { stroke: axisStroke, grid: { stroke: gridStroke }, ticks: { stroke: gridStroke } };
}

function destroyCharts() {
  for (const k of ["priceChart", "flowChart"]) {
    if (tapeState[k]) {
      try { tapeState[k].destroy(); } catch (e) {}
      tapeState[k] = null;
    }
  }
  tapeState.builtFocus = null;
}

function makePriceChart(el, width, data) {
  const stepped = uPlot.paths.stepped ? uPlot.paths.stepped({ align: 1 }) : undefined;
  const opts = {
    width,
    height: PRICE_H,
    cursor: { sync: { key: "tapeSync" }, points: { size: 6 } },
    legend: { show: true, live: true },
    scales: { x: { time: true } },
    axes: [
      axisX(),
      { stroke: axisStroke, grid: { stroke: gridStroke }, size: 50, values: (u, vs) => vs.map((v) => v + "¢") },
    ],
    series: [
      {},
      // Raw price is hidden by default (stepped print-by-print line is noisy);
      // VWAP/TWAP tell the story. Toggle via the "price line" checkbox.
      { label: "Price", stroke: "#e6edf3", width: 1.4, paths: stepped, points: { show: false }, show: !!tapeState.showPrice },
      { label: "VWAP", stroke: "#388bfd", width: 1.8, points: { show: false } },
      { label: "TWAP", stroke: "#d29922", width: 1.5, dash: [5, 3], points: { show: false } },
    ],
  };
  return new uPlot(opts, data, el);
}

function makeFlowChart(el, width, data) {
  const bars = uPlot.paths.bars ? uPlot.paths.bars({ size: [0.7, 60] }) : undefined;
  const opts = {
    width,
    height: FLOW_H,
    cursor: { sync: { key: "tapeSync" } },
    legend: { show: false },
    scales: { x: { time: true }, vol: { range: (u, mn, mx) => [0, (mx || 1) * 1.15] }, delta: {} },
    axes: [
      axisX(),
      { scale: "vol", side: 3, stroke: "#6e7681", grid: { show: false }, size: 46, values: (u, vs) => vs.map(kfmt) },
      { scale: "delta", side: 1, stroke: "#388bfd", grid: { show: false }, size: 46, values: (u, vs) => vs.map(kfmt) },
    ],
    series: [
      {},
      { label: "Vol", stroke: "#484f58", fill: "rgba(110,118,129,.30)", paths: bars, scale: "vol", points: { show: false } },
      { label: "Δflow", stroke: "#388bfd", width: 1.4, scale: "delta", points: { show: false } },
    ],
  };
  return new uPlot(opts, data, el);
}

function renderFocusCharts() {
  const prints = tapeState.focusPrints;
  $("focusEmpty").style.display = prints.length ? "none" : "block";
  if (typeof uPlot === "undefined") return; // chart lib unavailable — stats/tape still work
  if (!prints.length) {
    destroyCharts();
    return;
  }
  const series = A.cumulativeSeries(prints);
  const span = series.ts[series.ts.length - 1] - series.ts[0];
  const win = Number($("tWindow").value) || span || 3600;
  const buckets = A.bucketOHLCV(prints, chooseBucket(win));
  const pdata = [series.ts, series.price, series.vwap, series.twap];
  const bts = [], vol = [], delta = [];
  let run = 0;
  for (const b of buckets) {
    bts.push(b.ts);
    vol.push(b.volume);
    run += b.buyVol - b.sellVol;
    delta.push(run);
  }
  const fdata = [bts, vol, delta];
  const w = $("priceChart").clientWidth || 600;
  if (tapeState.builtFocus !== tapeState.focus || !tapeState.priceChart) {
    destroyCharts();
    tapeState.priceChart = makePriceChart($("priceChart"), w, pdata);
    tapeState.flowChart = makeFlowChart($("flowChart"), w, fdata);
    tapeState.builtFocus = tapeState.focus;
  } else {
    tapeState.priceChart.setData(pdata);
    tapeState.flowChart.setData(fdata);
    tapeState.priceChart.setSize({ width: w, height: PRICE_H });
    tapeState.flowChart.setSize({ width: w, height: FLOW_H });
  }
}

// ── Time & Sales ─────────────────────────────────────────────────────────────
function renderTimeSales() {
  let rows = tapeState.trades;
  if ($("tsFocusOnly").checked && tapeState.focus) rows = rows.filter((t) => t.ticker === tapeState.focus);
  $("tsCount").textContent = `${rows.length} prints`;
  if (!rows.length) {
    $("tape-table").innerHTML = `<div class="muted small">no trades captured yet for this filter</div>`;
  } else {
    const block = Number($("tBlock").value) || 1000;
    const maxSize = Math.max(...rows.map((t) => t.count), 1);
    $("tape-table").innerHTML =
      `<table class="ts-table"><thead><tr><th>Time</th><th>Type</th><th>Player</th><th>Taker</th><th class="num">Yes ¢</th><th class="num">Size</th><th class="flowcol">Flow</th></tr></thead><tbody>` +
      rows
        .map((t) => {
          const isNew = tapeState.primed && !tapeState.seen.has(t.trade_id);
          const isBlock = t.count >= block;
          // sqrt scaling keeps ordinary prints visible alongside the occasional block
          const w = Math.max(4, Math.round(Math.sqrt(t.count / maxSize) * 100));
          const sideCls = t.taker_side === "yes" ? "buy" : t.taker_side === "no" ? "sell" : "";
          return (
            `<tr class="ts-row ${sideCls}${isBlock ? " block" : ""}${isNew ? " flash" : ""}" data-ticker="${t.ticker}">` +
            `<td class="small mono">${fmtTime(t.ts)}</td>` +
            `<td class="small">${marketLabel(t.market_type)}</td>` +
            `<td>${playerName(t.player_code)}</td>` +
            `<td><span class="pill ${t.taker_side || ""}">${t.taker_side || "·"}</span></td>` +
            `<td class="num">${c1(t.yes_price)}</td>` +
            `<td class="num bold">${Math.round(t.count).toLocaleString()}${isBlock ? ' <span class="blocktag">BLK</span>' : ""}</td>` +
            `<td class="flowcol"><div class="sizebar ${sideCls}" style="width:${w}%"></div></td>` +
            `</tr>`
          );
        })
        .join("") +
      `</tbody></table>`;
    $("tape-table")
      .querySelectorAll(".ts-row")
      .forEach((el) => el.addEventListener("click", () => focusMarket(el.dataset.ticker)));
  }
  for (const t of tapeState.trades) tapeState.seen.add(t.trade_id);
  if (tapeState.seen.size > 8000) tapeState.seen = new Set(tapeState.trades.map((t) => t.trade_id));
  tapeState.primed = true;
}

// ── PnL ──────────────────────────────────────────────────────────────────────
const pnlState = { rows: [], totals: {}, groupBy: "none", status: "all", fetchedTs: null, store: null };

function usd(v) {
  const n = Number(v) || 0;
  return (n < 0 ? "-$" : "$") + Math.abs(n).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
function signColor(v) {
  return (Number(v) || 0) > 0 ? "up" : (Number(v) || 0) < 0 ? "down" : "muted";
}

async function loadPnL(refresh = false) {
  try {
    const d = await api("/api/pnl");
    pnlState.rows = d.rows || [];
    pnlState.totals = d.totals || {};
    pnlState.fetchedTs = d.fetched_ts || null;
    pnlState.store = d.store || null;
    renderPnL();
  } catch (e) {
    $("pnl-table").innerHTML = `<div class="msg err">${e.message}</div>`;
  }
}

function visiblePnL() {
  let rows = pnlState.rows;
  if (pnlState.status === "open") rows = rows.filter((r) => r.is_open);
  else if (pnlState.status === "settled") rows = rows.filter((r) => !r.is_open);
  return rows;
}

function pnlGroupedRows() {
  const gb = pnlState.groupBy;
  const rows = visiblePnL();
  if (!gb || gb === "none") return rows;
  const key = (r) => (gb === "is_open" ? (r.is_open ? "open" : "settled") : r[gb] || "(blank)");
  const buckets = new Map();
  for (const r of rows) {
    const k = key(r);
    if (!buckets.has(k))
      buckets.set(k, { _group: k, n: 0, contracts: 0, realized: 0, unrealized: 0, fees: 0, net: 0, exposure: 0, invested: 0, n_open: 0, n_settled: 0 });
    const b = buckets.get(k);
    b.n += 1;
    b.contracts += Number(r.contracts) || 0;
    b.realized += Number(r.realized) || 0;
    b.unrealized += Number(r.unrealized) || 0;
    b.fees += Number(r.fees) || 0;
    b.net += Number(r.net) || 0;
    b.exposure += Number(r.exposure) || 0;
    b.invested += Number(r.invested) || 0;
    if (r.is_open) b.n_open += 1; else b.n_settled += 1;
  }
  return Array.from(buckets.values()).sort((a, b) => b.net - a.net);
}

function renderPnLTotals() {
  const rows = visiblePnL();
  const sum = (f) => rows.reduce((s, r) => s + (Number(r[f]) || 0), 0);
  const net = sum("net"), realized = sum("realized"), unreal = sum("unrealized"),
    fees = sum("fees"), invested = sum("invested");
  const roi = invested > 0 ? (net / invested) * 100 : 0;
  const nOpen = rows.filter((r) => r.is_open).length;
  const nSettled = rows.length - nOpen;
  const cell = (label, val, cls) => `<div class="stat"><div class="sl">${label}</div><div class="sv ${cls || ""}">${val}</div></div>`;
  $("pnlTotals").innerHTML =
    cell("Net PnL", usd(net), signColor(net)) +
    cell("Realized", usd(realized), signColor(realized)) +
    cell("Unrealized", usd(unreal), signColor(unreal)) +
    cell("Fees", usd(fees), "muted") +
    cell("Invested", usd(invested)) +
    cell("ROI", roi.toFixed(1) + "%", signColor(roi)) +
    cell("Positions", `${nOpen} open / ${nSettled} settled`);
  $("pnlMeta").textContent = pnlState.store ? `${pnlState.store.fills} fills · ${pnlState.store.settlements} settlements` : "";
}

// Resolve a grouped-row key to a human label based on the active group-by.
function groupLabel(code) {
  if (pnlState.groupBy === "player_code") return playerName(code);
  if (pnlState.groupBy === "market_type") return marketLabel(code);
  if (pnlState.groupBy === "event_code") return eventName(code);
  return code;
}

function renderPnL() {
  renderPnLTotals();
  const grouped = pnlState.groupBy !== "none";
  const rows = pnlGroupedRows();
  if (!rows.length) {
    $("pnl-table").innerHTML = `<div class="muted small">no positions — the cron sync seeds fills/settlements into D1 (first sync may take a few minutes)</div>`;
    return;
  }
  const dCell = (v) => `<td class="num ${signColor(v)}">${usd(v)}</td>`;
  let html;
  if (grouped) {
    const gh = { player_code: "golfer", market_type: "market", event_code: "tournament", is_open: "status" }[pnlState.groupBy] || "group";
    html =
      `<table><thead><tr><th>${gh}</th><th class="num">#</th><th>open/settled</th><th class="num">contracts</th><th class="num">exposure</th><th class="num">realized</th><th class="num">unrealized</th><th class="num">fees</th><th class="num">net</th></tr></thead><tbody>` +
      rows
        .map(
          (r) =>
            `<tr><td class="bold">${groupLabel(r._group)}</td><td class="num muted">${r.n}</td><td class="small">${r.n_open} / ${r.n_settled}</td>` +
            `<td class="num">${r.contracts.toLocaleString()}</td><td class="num">${usd(r.exposure)}</td>` +
            dCell(r.realized) + dCell(r.unrealized) + `<td class="num muted">${usd(r.fees)}</td>` + dCell(r.net) + `</tr>`
        )
        .join("") +
      `</tbody></table>`;
  } else {
    html =
      `<table><thead><tr><th>Golfer</th><th>Mkt</th><th>Event</th><th>Side</th><th>Status</th><th class="num">Qty</th><th class="num">Avg</th><th class="num">Mark</th><th class="num">Realized</th><th class="num">Unreal</th><th class="num">Fees</th><th class="num">Net</th><th class="num">Ret%</th></tr></thead><tbody>` +
      rows
        .map((r) => {
          const result = r.market_result ? ` <span class="muted small">(${r.market_result})</span>` : "";
          return (
            `<tr>` +
            `<td>${playerName(r.player_code) || r.ticker}</td>` +
            `<td class="small">${marketLabel(r.market_type)}</td>` +
            `<td class="small">${eventName(r.event_code)}</td>` +
            `<td><span class="pill ${r.side || ""}">${r.side || "·"}</span></td>` +
            `<td class="small ${r.is_open ? "" : "muted"}">${r.is_open ? "open" : "settled"}${result}</td>` +
            `<td class="num">${(r.contracts || 0).toLocaleString()}</td>` +
            `<td class="num">${r.avg_cost ? (r.avg_cost * 100).toFixed(1) + "¢" : "—"}</td>` +
            `<td class="num muted">${r.mark != null ? (r.mark * 100).toFixed(1) + "¢" : "—"}</td>` +
            dCell(r.realized) +
            `<td class="num ${signColor(r.unrealized)}">${r.is_open ? usd(r.unrealized) : "—"}</td>` +
            `<td class="num muted">${usd(r.fees)}</td>` +
            dCell(r.net) +
            `<td class="num ${signColor(r.return_pct)}">${r.return_pct != null ? r.return_pct.toFixed(1) + "%" : "—"}</td>` +
            `</tr>`
          );
        })
        .join("") +
      `</tbody></table>`;
  }
  $("pnl-table").innerHTML = html;
}

// ── Wiring ───────────────────────────────────────────────────────────────────
$("loadMarkets").addEventListener("click", loadMarkets);
$("marketSearch").addEventListener("input", renderMarketList);
$("market").addEventListener("change", (e) => selectMarket(e.target.value));
document.querySelectorAll(".side").forEach((b) => b.addEventListener("click", () => setSide(b.dataset.side)));
$("killOn").addEventListener("change", (e) => ($("killTime").disabled = !e.target.checked));
document.querySelectorAll(".quick .chip").forEach((c) =>
  c.addEventListener("click", () => {
    const d = new Date(Date.now() + Number(c.dataset.min) * 60000);
    $("killTime").value = toLocalInput(d);
    $("killOn").checked = true;
    $("killTime").disabled = false;
  })
);
$("place").addEventListener("click", placeOrder);
$("refreshPos").addEventListener("click", loadPositions);
$("refreshOrders").addEventListener("click", loadOrders);
$("tApply").addEventListener("click", applyTape);
$("tMarket").addEventListener("change", applyTape);
$("tWindow").addEventListener("change", applyTape);
$("wlSearch").addEventListener("input", renderWatchlist);
$("tsFocusOnly").addEventListener("change", renderTimeSales);
$("tBlock").addEventListener("input", renderTimeSales);
$("eventSelect").addEventListener("change", (e) => {
  currentEvent = e.target.value;
  tapeState.focus = null;
  destroyCharts();
  applyTape();
});
$("showPrice").addEventListener("change", (e) => {
  tapeState.showPrice = e.target.checked;
  if (tapeState.priceChart) tapeState.priceChart.setSeries(1, { show: tapeState.showPrice });
});
$("pnlGroup").addEventListener("change", (e) => { pnlState.groupBy = e.target.value; renderPnL(); });
$("pnlStatus").addEventListener("change", (e) => { pnlState.status = e.target.value; renderPnL(); });
$("pnlRefresh").addEventListener("click", () => loadPnL(true));

// Resize the focused charts to the container when the window changes width.
let _resizeRaf = null;
window.addEventListener("resize", () => {
  if (_resizeRaf) cancelAnimationFrame(_resizeRaf);
  _resizeRaf = requestAnimationFrame(() => {
    if (!tapeState.priceChart || !$("tape").classList.contains("active")) return;
    const w = $("priceChart").clientWidth || 600;
    tapeState.priceChart.setSize({ width: w, height: PRICE_H });
    tapeState.flowChart.setSize({ width: w, height: FLOW_H });
  });
});

// Auto-refresh
setInterval(() => {
  if ($("trade").classList.contains("active")) {
    loadOrders();
  }
  if ($("tape").classList.contains("active") && $("tAuto").checked) {
    loadTape();
  }
}, 8000);

// Init
loadReference();
loadBalance();
loadMarkets();
loadPositions();
loadOrders();
loadProposals();
