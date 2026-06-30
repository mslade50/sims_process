import { Env, json } from "../../shared/kalshi";

interface MEnv extends Env {
  MAKER_STATE_TOKEN?: string;
  PROPOSALS_TOKEN?: string;
}

function constEq(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let d = 0;
  for (let i = 0; i < a.length; i++) d |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return d === 0;
}

// GET /api/maker — latest maker cockpit snapshot (session-gated by middleware).
export const onRequestGet: PagesFunction<MEnv> = async (ctx) => {
  const r = await ctx.env.DB.prepare("SELECT payload, updated_ts FROM maker_state WHERE id = 1")
    .first()
    .catch(() => null) as any;
  if (!r || !r.payload) return json({ state: null });
  let state: any = null;
  try {
    state = JSON.parse(r.payload);
  } catch {
    state = null;
  }
  return json({ state, updated_ts: r.updated_ts ?? null });
};

// POST /api/maker — headless push from the local maker run. Token-gated (the
// middleware lets this route through without a session cookie; auth is the
// X-Maker-Token header against MAKER_STATE_TOKEN, falling back to PROPOSALS_TOKEN).
// Stores the raw JSON snapshot. Display only — never triggers an order.
export const onRequestPost: PagesFunction<MEnv> = async (ctx) => {
  const token = ctx.env.MAKER_STATE_TOKEN || ctx.env.PROPOSALS_TOKEN;
  if (!token) return json({ error: "maker push not configured (set MAKER_STATE_TOKEN or PROPOSALS_TOKEN)" }, 503);
  const provided = ctx.request.headers.get("X-Maker-Token") || "";
  if (!constEq(provided, token)) return json({ error: "unauthorized" }, 401);

  const body = await ctx.request.text();
  if (!body) return json({ error: "empty body" }, 400);
  // Validate it parses (we store the raw text but reject garbage).
  try {
    JSON.parse(body);
  } catch {
    return json({ error: "body is not valid JSON" }, 400);
  }
  const now = Math.floor(Date.now() / 1000);
  await ctx.env.DB.prepare("INSERT OR REPLACE INTO maker_state (id, payload, updated_ts) VALUES (1, ?, ?)")
    .bind(body, now)
    .run();
  return json({ ok: true, updated_ts: now });
};
