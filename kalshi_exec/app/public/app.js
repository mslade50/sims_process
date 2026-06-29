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

// ── Tabs ────────────────────────────────────────────────────────────────────
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tabpane").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $(btn.dataset.tab).classList.add("active");
    if (btn.dataset.tab === "tape") loadTape();
  });
});

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

async function loadMarkets() {
  const type = $("series").value;
  $("market").innerHTML = `<option>loading…</option>`;
  try {
    const d = await api(`/api/markets?type=${type}`);
    allMarkets = d.markets || [];
    renderMarketList();
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
    .map((m) => `<option value="${m.ticker}">${m.subtitle || m.ticker} — ${m.yes_bid ?? "·"}/${m.yes_ask ?? "·"}¢</option>`)
    .join("");
}

function selectMarket(ticker) {
  selected = allMarkets.find((m) => m.ticker === ticker);
  if (!selected) return;
  $("ticketBody").hidden = false;
  $("selectedMarket").innerHTML = `<b>${selected.subtitle || selected.ticker}</b><br><span class="muted small">${selected.ticker}</span>`;
  prefillPrice();
  loadOrderbook(ticker);
}

function prefillPrice() {
  if (!selected) return;
  const v = side === "yes" ? selected.yes_bid : selected.no_bid;
  if (v != null) $("price").value = v;
}

async function loadOrderbook(ticker) {
  $("bookTicker").textContent = ticker;
  $("yesBook").innerHTML = $("noBook").innerHTML = "…";
  try {
    const ob = await api(`/api/orderbook?ticker=${encodeURIComponent(ticker)}`);
    const render = (levels, s) =>
      (levels || [])
        .slice(0, 10)
        .map((l) => `<div class="lvl" data-side="${s}" data-price="${l.price}"><span class="p">${l.price}¢</span><span class="q">${l.qty}</span></div>`)
        .join("") || `<div class="muted small">empty</div>`;
    $("yesBook").innerHTML = render(ob.yes, "yes");
    $("noBook").innerHTML = render(ob.no, "no");
    document.querySelectorAll(".lvl").forEach((el) =>
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
async function loadPositions() {
  try {
    const d = await api("/api/positions");
    if (!d.positions.length) return ($("positions").innerHTML = `<div class="muted small">no open golf positions</div>`);
    $("positions").innerHTML =
      `<table><thead><tr><th>Market</th><th>Ticker</th><th class="num">Pos</th><th class="num">Exposure</th></tr></thead><tbody>` +
      d.positions
        .map(
          (p) =>
            `<tr><td>${p.market_type}</td><td>${p.ticker}</td><td class="num">${p.position}</td><td class="num">${fmtUSD(p.market_exposure)}</td></tr>`
        )
        .join("") +
      `</tbody></table>`;
  } catch (e) {
    $("positions").innerHTML = `<div class="msg err">${e.message}</div>`;
  }
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

// ── Tape ─────────────────────────────────────────────────────────────────────
async function loadTape() {
  const p = new URLSearchParams();
  const market = $("tMarket").value;
  const ticker = $("tTicker").value.trim();
  if (market) p.set("market", market);
  else if (ticker) p.set("market", ticker);
  if ($("tMinP").value) p.set("min_price", $("tMinP").value);
  if ($("tMaxP").value) p.set("max_price", $("tMaxP").value);
  if ($("tMinSize").value) p.set("min_size", $("tMinSize").value);
  const win = Number($("tWindow").value);
  if (win > 0) p.set("from", String(Math.floor(Date.now() / 1000) - win));
  p.set("limit", "300");
  try {
    const d = await api("/api/tape?" + p.toString());
    $("tapeStatus").textContent = `${d.trades.length} trades`;
    if (!d.trades.length) return ($("tape-table").innerHTML = `<div class="muted small">no trades captured yet for this filter</div>`);
    $("tape-table").innerHTML =
      `<table><thead><tr><th>Time</th><th>Type</th><th>Player / Ticker</th><th>Taker</th><th class="num">Yes ¢</th><th class="num">Size</th></tr></thead><tbody>` +
      d.trades
        .map(
          (t) =>
            `<tr><td class="small">${fmtTime(t.ts)}</td><td>${t.market_type}</td><td>${t.player_code || t.ticker}</td><td><span class="pill ${t.taker_side || ""}">${t.taker_side || "·"}</span></td><td class="num">${t.yes_price}¢</td><td class="num">${Math.round(t.count)}</td></tr>`
        )
        .join("") +
      `</tbody></table>`;
  } catch (e) {
    $("tape-table").innerHTML = `<div class="msg err">${e.message}</div>`;
  }
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
$("tApply").addEventListener("click", loadTape);

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
loadBalance();
loadMarkets();
loadPositions();
loadOrders();
