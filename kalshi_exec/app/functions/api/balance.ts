import { Env, kalshi, json } from "../../shared/kalshi";

// GET /api/balance — account balance + portfolio value (cents).
export const onRequestGet: PagesFunction<Env> = async (ctx) => {
  const r = await kalshi(ctx.env, "GET", "/portfolio/balance");
  if (!r.ok) return json({ error: "balance failed", status: r.status, detail: r.data }, 502);
  return json({
    balance: r.data.balance ?? null,
    portfolio_value: r.data.portfolio_value ?? null,
  });
};
