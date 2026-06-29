/**
 * kalshi-exec-cron — internal Worker, runs every minute (see wrangler.toml).
 *
 *   1. auto-kill sweep: force-cancel any order past kill_ts + 60s (belt-and-
 *      suspenders to Kalshi's native expiration_ts).
 *   2. trade poll: pull /markets/trades since last cursor, keep golf OUTRIGHT
 *      trades, append to the D1 tape.
 *
 * No public functionality — the fetch() handler is read-only status for debugging.
 */
import { Env, kalshi, classifyTicker, OUTRIGHT_TYPES, isGolfTicker, json } from "../../app/shared/kalshi";

const MAX_PAGES = 30; // safety cap on /markets/trades pagination per run
const MIN_TRADE_SIZE = 100; // collector floor — don't store golf-outright trades smaller than this
const RETENTION_DAYS = 7; // weekly sweep keeps this many days of the full >=100 tape
const ARCHIVE_THRESHOLD = 5000; // trades >= this are kept FOREVER (the unusual-block archive)
const WEEKLY_CRON = "0 9 * * 1"; // Monday 09:00 UTC — retention sweep

function nowSec(): number {
  return Math.floor(Date.now() / 1000);
}

function tradeTs(t: any): number {
  if (typeof t.ts === "number") return t.ts;
  const p = Date.parse(t.created_time);
  return Number.isFinite(p) ? Math.floor(p / 1000) : nowSec();
}

// Kalshi trades report price as decimal-dollar strings ("0.9600"). Convert to
// cents, keeping 0.1¢ precision for deci-cent markets.
function priceCents(dollars: any): number | null {
  if (dollars == null) return null;
  const n = Number(dollars);
  return Number.isFinite(n) ? Math.round(n * 1000) / 10 : null;
}

async function autoKill(env: Env): Promise<{ due: number; killed: number }> {
  const now = nowSec();
  const due = await env.DB.prepare(
    "SELECT order_id, ticker FROM pending_kills WHERE status = 'pending' AND hard_kill_ts <= ?"
  )
    .bind(now)
    .all();
  let killed = 0;
  for (const row of (due.results || []) as any[]) {
    const { order_id, ticker } = row;
    if (!isGolfTicker(ticker)) {
      await env.DB.prepare("UPDATE pending_kills SET status='error', note='non-golf' WHERE order_id=?")
        .bind(order_id)
        .run();
      continue;
    }
    const r = await kalshi(env, "DELETE", `/portfolio/orders/${order_id}`);
    // 2xx = we cancelled it; 404 = already gone (native expiry fired) — both fine.
    const status = r.ok ? "killed" : r.status === 404 ? "gone" : "error";
    await env.DB.prepare("UPDATE pending_kills SET status=?, note=? WHERE order_id=?")
      .bind(status, `cron ${r.status}`, order_id)
      .run();
    if (r.ok) killed++;
  }
  return { due: (due.results || []).length, killed };
}

async function pollTrades(env: Env): Promise<{ pages: number; scanned: number; inserted: number; capped: boolean }> {
  const stateRow = (await env.DB.prepare("SELECT value FROM collector_state WHERE key='last_ts'").first()) as any;
  // First run: start from now (no giant backfill). Backfill historical golf
  // trades separately if ever needed.
  let lastTs = stateRow ? parseInt(stateRow.value, 10) : nowSec();

  let cursor: string | undefined;
  let pages = 0;
  let scanned = 0;
  let inserted = 0;
  let maxTs = lastTs;
  let capped = false;

  while (pages < MAX_PAGES) {
    const r = await kalshi(env, "GET", "/markets/trades", { query: { min_ts: lastTs, limit: 1000, cursor } });
    if (!r.ok) break;
    const trades: any[] = r.data.trades || [];

    const stmts: D1PreparedStatement[] = [];
    for (const t of trades) {
      scanned++;
      const ts = tradeTs(t);
      if (ts > maxTs) maxTs = ts; // advance cursor on ALL scanned trades (so it moves even when golf is sparse)
      const cls = classifyTicker(t.ticker || "");
      if (!OUTRIGHT_TYPES.has(cls.marketType)) continue;
      const cnt = Number(t.count_fp ?? t.count ?? 0) || 0;
      if (cnt < MIN_TRADE_SIZE) continue; // collector-side floor — small trades never hit D1
      const yesC = t.yes_price_dollars != null ? priceCents(t.yes_price_dollars) : t.yes_price ?? null;
      const noC = t.no_price_dollars != null ? priceCents(t.no_price_dollars) : t.no_price ?? null;
      stmts.push(
        env.DB.prepare(
          "INSERT OR IGNORE INTO trades (trade_id,ticker,market_type,event_code,player_code,ts,yes_price,no_price,count,taker_side) VALUES (?,?,?,?,?,?,?,?,?,?)"
        ).bind(
          t.trade_id,
          t.ticker,
          cls.marketType,
          cls.eventCode,
          cls.playerCode,
          ts,
          yesC,
          noC,
          cnt,
          t.taker_side ?? null
        )
      );
    }
    if (stmts.length) {
      await env.DB.batch(stmts);
      inserted += stmts.length;
    }

    cursor = r.data.cursor;
    pages++;
    if (!cursor || trades.length < 1000) break;
    if (pages >= MAX_PAGES) capped = true;
  }

  const newLast = Math.max(lastTs, maxTs);
  await env.DB.prepare("INSERT OR REPLACE INTO collector_state (key,value) VALUES ('last_ts',?)")
    .bind(String(newLast))
    .run();
  return { pages, scanned, inserted, capped };
}

// ── PnL store sync ───────────────────────────────────────────────────────────
// Mirror maker_dashboard/server.py's durable approach: Kalshi /portfolio/fills
// only retains ~2 months, so persist every golf fill ever seen. Settlements are
// authoritative for held-to-settlement cost basis and never age out.
const PNL_MAX_PAGES = 20; // per-run pagination cap (first sync catches up over runs)
const PNL_SYNC_INTERVAL = 270; // seconds between syncs (~ every 5 minutes)

function fillTs(f: any): number {
  if (typeof f.ts === "number") return f.ts;
  const p = Date.parse(f.created_time);
  return Number.isFinite(p) ? Math.floor(p / 1000) : nowSec();
}
function dnum(v: any): number | null {
  if (v == null) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

async function syncPnl(env: Env): Promise<{ fills: number; setts: number }> {
  // fills — incremental, golf only, stop when a page adds nothing new
  const knownFills = new Set<string>();
  for (const r of ((await env.DB.prepare("SELECT fill_id FROM fills").all()).results || []) as any[])
    knownFills.add(r.fill_id);

  let cursor: string | undefined;
  let pages = 0;
  let newFills = 0;
  while (pages < PNL_MAX_PAGES) {
    const r = await kalshi(env, "GET", "/portfolio/fills", { query: { limit: 200, cursor } });
    if (!r.ok) break;
    const fills: any[] = r.data.fills || [];
    const stmts: D1PreparedStatement[] = [];
    let pageNew = 0;
    for (const f of fills) {
      const ticker = f.ticker || f.market_ticker || "";
      const id = f.fill_id;
      if (!id || !isGolfTicker(ticker) || knownFills.has(id)) continue;
      knownFills.add(id);
      pageNew++;
      stmts.push(
        env.DB.prepare(
          "INSERT OR IGNORE INTO fills (fill_id,order_id,ticker,side,action,count,yes_price,no_price,is_taker,fee_cost,ts,created_time) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"
        ).bind(
          id, f.order_id ?? null, ticker, f.side ?? null, f.action ?? null,
          Number(f.count_fp ?? f.count ?? 0) || 0,
          dnum(f.yes_price_dollars), dnum(f.no_price_dollars),
          f.is_taker ? 1 : 0, dnum(f.fee_cost) ?? 0, fillTs(f), f.created_time ?? null
        )
      );
    }
    if (stmts.length) { await env.DB.batch(stmts); newFills += stmts.length; }
    cursor = r.data.cursor;
    pages++;
    if (!cursor || pageNew === 0) break;
  }

  // settlements — keyed by ticker, golf only
  const knownSetts = new Set<string>();
  for (const r of ((await env.DB.prepare("SELECT ticker FROM settlements").all()).results || []) as any[])
    knownSetts.add(r.ticker);

  let scursor: string | undefined;
  let spages = 0;
  let newSetts = 0;
  while (spages < PNL_MAX_PAGES) {
    const r = await kalshi(env, "GET", "/portfolio/settlements", { query: { limit: 200, cursor: scursor } });
    if (!r.ok) break;
    const setts: any[] = r.data.settlements || [];
    const stmts: D1PreparedStatement[] = [];
    let pageNew = 0;
    for (const s of setts) {
      const ticker = s.ticker || "";
      if (!ticker || !isGolfTicker(ticker) || knownSetts.has(ticker)) continue;
      knownSetts.add(ticker);
      pageNew++;
      stmts.push(
        env.DB.prepare(
          "INSERT OR IGNORE INTO settlements (ticker,market_result,yes_count,no_count,yes_total_cost,no_total_cost,fee_cost,settled_time,ts) VALUES (?,?,?,?,?,?,?,?,?)"
        ).bind(
          ticker, s.market_result ?? "", Number(s.yes_count_fp ?? s.yes_count ?? 0) || 0,
          Number(s.no_count_fp ?? s.no_count ?? 0) || 0,
          dnum(s.yes_total_cost_dollars) ?? 0, dnum(s.no_total_cost_dollars) ?? 0,
          dnum(s.fee_cost) ?? 0, s.settled_time ?? null, fillTs(s)
        )
      );
    }
    if (stmts.length) { await env.DB.batch(stmts); newSetts += stmts.length; }
    scursor = r.data.cursor;
    spages++;
    if (!scursor || pageNew === 0) break;
  }

  return { fills: newFills, setts: newSetts };
}

// Weekly retention: drop everything older than RETENTION_DAYS EXCEPT all-time
// large/unusual trades (>= ARCHIVE_THRESHOLD), which are kept forever.
async function retentionSweep(env: Env): Promise<number> {
  const cutoff = nowSec() - RETENTION_DAYS * 86400;
  const res = await env.DB.prepare("DELETE FROM trades WHERE ts < ? AND count < ?")
    .bind(cutoff, ARCHIVE_THRESHOLD)
    .run();
  const deleted = (res.meta as any)?.changes ?? -1;
  await env.DB.prepare("INSERT OR REPLACE INTO collector_state (key,value) VALUES ('last_purge',?)")
    .bind(JSON.stringify({ ts: nowSec(), deleted, cutoff, archive_threshold: ARCHIVE_THRESHOLD }))
    .run()
    .catch(() => {});
  return deleted;
}

export default {
  async scheduled(event: ScheduledEvent, env: Env, _ctx: ExecutionContext): Promise<void> {
    // Weekly retention sweep runs on its own schedule; everything else is the per-minute job.
    if (event.cron === WEEKLY_CRON) {
      const deleted = await retentionSweep(env).catch(() => -1);
      console.log("[cron] weekly retention sweep deleted", deleted, "rows");
      return;
    }
    const kill = await autoKill(env).catch((e) => ({ due: -1, killed: -1, error: String(e) }));
    const poll = await pollTrades(env).catch((e) => ({ pages: -1, scanned: -1, inserted: -1, capped: false, error: String(e) }));
    // PnL store sync, throttled to ~5 min so we don't re-read the fills table every minute.
    let pnl: any = { skipped: true };
    const lastSyncRow = (await env.DB.prepare("SELECT value FROM collector_state WHERE key='last_pnl_sync'").first()) as any;
    const lastSync = lastSyncRow ? parseInt(lastSyncRow.value, 10) : 0;
    if (nowSec() - lastSync >= PNL_SYNC_INTERVAL) {
      pnl = await syncPnl(env).catch((e) => ({ fills: -1, setts: -1, error: String(e) }));
      await env.DB.prepare("INSERT OR REPLACE INTO collector_state (key,value) VALUES ('last_pnl_sync',?)")
        .bind(String(nowSec())).run().catch(() => {});
    }
    const run = { ts: nowSec(), kill, poll, pnl };
    await env.DB.prepare("INSERT OR REPLACE INTO collector_state (key,value) VALUES ('last_run',?)")
      .bind(JSON.stringify(run))
      .run()
      .catch(() => {});
    if ((poll as any).capped) console.log("[cron] trade poll hit MAX_PAGES — possible dropped trades", run);
  },

  // Read-only status (no secrets) for eyeballing the collector.
  async fetch(_req: Request, env: Env): Promise<Response> {
    const lastTs = (await env.DB.prepare("SELECT value FROM collector_state WHERE key='last_ts'").first()) as any;
    const lastRun = (await env.DB.prepare("SELECT value FROM collector_state WHERE key='last_run'").first()) as any;
    const lastPurge = (await env.DB.prepare("SELECT value FROM collector_state WHERE key='last_purge'").first()) as any;
    const tc = (await env.DB.prepare("SELECT COUNT(*) AS n FROM trades").first()) as any;
    const arch = (await env.DB.prepare("SELECT COUNT(*) AS n FROM trades WHERE count >= ?").bind(ARCHIVE_THRESHOLD).first()) as any;
    const pk = await env.DB.prepare("SELECT status, COUNT(*) AS n FROM pending_kills GROUP BY status").all();
    const fc = (await env.DB.prepare("SELECT COUNT(*) AS n FROM fills").first().catch(() => null)) as any;
    const sc = (await env.DB.prepare("SELECT COUNT(*) AS n FROM settlements").first().catch(() => null)) as any;
    return json({
      worker: "kalshi-exec-cron",
      last_ts: lastTs?.value ?? null,
      last_run: lastRun?.value ? JSON.parse(lastRun.value) : null,
      last_purge: lastPurge?.value ? JSON.parse(lastPurge.value) : null,
      trades: tc?.n ?? 0,
      archive_rows: arch?.n ?? 0,
      fills: fc?.n ?? 0,
      settlements: sc?.n ?? 0,
      pending_kills: pk.results || [],
    });
  },
};
