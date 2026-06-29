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
    const run = { ts: nowSec(), kill, poll };
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
    return json({
      worker: "kalshi-exec-cron",
      last_ts: lastTs?.value ?? null,
      last_run: lastRun?.value ? JSON.parse(lastRun.value) : null,
      last_purge: lastPurge?.value ? JSON.parse(lastPurge.value) : null,
      trades: tc?.n ?? 0,
      archive_rows: arch?.n ?? 0,
      pending_kills: pk.results || [],
    });
  },
};
