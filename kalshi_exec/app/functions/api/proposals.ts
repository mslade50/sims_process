import { Env, json, classifyTicker } from "../../shared/kalshi";

interface PEnv extends Env {
  PROPOSALS_TOKEN?: string;
}

function num(v: any): number | null {
  if (v == null || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function constEq(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let d = 0;
  for (let i = 0; i < a.length; i++) d |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return d === 0;
}

// GET /api/proposals — current model-edge proposals (session-gated by the
// middleware like every other read). Sorted by |edge| so the strongest edges
// surface first.
export const onRequestGet: PagesFunction<PEnv> = async (ctx) => {
  const r = await ctx.env.DB.prepare("SELECT * FROM proposals ORDER BY ABS(edge_pp) DESC")
    .all()
    .catch(() => ({ results: [] as any[] }));
  const rows = (r.results || []) as any[];
  return json({ rows, scan_ts: rows.length ? rows[0].scan_ts : null, count: rows.length });
};

// POST /api/proposals — headless push from the local sim. Token-gated (the
// middleware lets this one route through without a session cookie; auth is the
// X-Proposals-Token header matching the PROPOSALS_TOKEN secret). Replaces the
// whole proposals table atomically. Display only — never triggers an order.
export const onRequestPost: PagesFunction<PEnv> = async (ctx) => {
  const token = ctx.env.PROPOSALS_TOKEN;
  if (!token) return json({ error: "proposals push not configured (PROPOSALS_TOKEN unset)" }, 503);
  const provided = ctx.request.headers.get("X-Proposals-Token") || "";
  if (!constEq(provided, token)) return json({ error: "unauthorized" }, 401);

  const body: any = await ctx.request.json().catch(() => ({}));
  const rows = Array.isArray(body.rows) ? body.rows : [];
  if (!rows.length) return json({ error: "no rows" }, 400);
  const scanTs = String(body.scan_ts || new Date().toISOString());
  const now = Math.floor(Date.now() / 1000);

  const stmts: D1PreparedStatement[] = [ctx.env.DB.prepare("DELETE FROM proposals")];
  for (const row of rows) {
    const ticker = String(row.ticker || "");
    if (!ticker) continue;
    const cls = classifyTicker(ticker);
    stmts.push(
      ctx.env.DB.prepare(
        "INSERT OR REPLACE INTO proposals (ticker,market_type,player_code,event_code,side,sim_prob,edge_pp,best_bid,best_ask,post_price,kelly_f,scan_ts,updated_ts) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"
      ).bind(
        ticker,
        cls.marketType || row.market || null,
        cls.playerCode || row.player || null,
        cls.eventCode || null,
        row.side ?? null,
        num(row.sim_prob),
        num(row.edge_pp),
        num(row.best_bid),
        num(row.best_ask),
        num(row.post_price),
        num(row.kelly_f),
        scanTs,
        now
      )
    );
  }
  await ctx.env.DB.batch(stmts);
  return json({ ok: true, stored: stmts.length - 1, scan_ts: scanTs });
};
