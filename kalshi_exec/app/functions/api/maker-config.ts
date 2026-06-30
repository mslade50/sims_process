import { Env, json } from "../../shared/kalshi";

// Maker toggle config. GET is public (the headless maker reads it — middleware
// carves out GET /api/maker-config); POST is session-gated (the dashboard button,
// which carries the login cookie). Only known boolean toggles are accepted.
const DEFAULTS: Record<string, boolean> = { block_yes_outrights_pre_wed: true };

async function readConfig(env: Env): Promise<Record<string, boolean>> {
  const r = (await env.DB.prepare("SELECT config FROM maker_config WHERE id = 1")
    .first()
    .catch(() => null)) as any;
  let cfg = { ...DEFAULTS };
  if (r && r.config) {
    try {
      cfg = { ...DEFAULTS, ...JSON.parse(r.config) };
    } catch {
      /* keep defaults */
    }
  }
  return cfg;
}

export const onRequestGet: PagesFunction<Env> = async (ctx) => {
  return json({ config: await readConfig(ctx.env) });
};

export const onRequestPost: PagesFunction<Env> = async (ctx) => {
  const body: any = await ctx.request.json().catch(() => ({}));
  const next = await readConfig(ctx.env);
  // Only accept known boolean toggles.
  for (const k of Object.keys(DEFAULTS)) {
    if (typeof body[k] === "boolean") next[k] = body[k];
  }
  const now = Math.floor(Date.now() / 1000);
  await ctx.env.DB.prepare("INSERT OR REPLACE INTO maker_config (id, config, updated_ts) VALUES (1, ?, ?)")
    .bind(JSON.stringify(next), now)
    .run();
  return json({ ok: true, config: next });
};
