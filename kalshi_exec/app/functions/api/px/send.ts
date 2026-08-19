import {
  PxEnv, px, pxConfigured, pxJson, golfSnapshot, eventMarkets, marketLineIds,
} from "../../../shared/prophetx";

// Identifiable, unique external_id — "exm" prefix mirrors the Kalshi exec
// orders so manual exec wagers are recognizable in histories.
function execExternalId(): string {
  return "exm-" + crypto.randomUUID();
}

// POST /api/px/send
// { event_id, line_id, odds, stake }
// Places ONE limit wager on ProphetX. Golf-only: the line must belong to an
// active golf event (verified server-side against the live snapshot, not the
// caller's word). Odds must be on the exchange's price ladder. Stake capped
// server-side (PX_SEND_MAX_STAKE_DOLLARS, default 500) — same principle as
// /api/send's Kalshi caps: the UI is never the only guard on a real-order
// surface. Session-gated by _middleware (master login).
export const onRequestPost: PagesFunction<PxEnv> = async (ctx) => {
  if (!pxConfigured(ctx.env)) {
    return pxJson({ error: "ProphetX not configured (PROPHETX_ACCESS_KEY/SECRET_KEY secrets)" }, 503);
  }
  const body: any = await ctx.request.json().catch(() => ({}));
  const eventId = String(body.event_id || "");
  const lineId = String(body.line_id || "");
  const odds = Number(body.odds);
  const stake = Number(body.stake);
  const maxStake = Number(ctx.env.PX_SEND_MAX_STAKE_DOLLARS || "500");

  if (!eventId || !lineId) return pxJson({ error: "event_id and line_id required" }, 400);
  if (!Number.isFinite(odds)) return pxJson({ error: "odds required (numeric)" }, 400);
  if (!(stake > 0)) return pxJson({ error: "stake must be > 0" }, 400);
  if (stake > maxStake) {
    return pxJson({ error: `stake $${stake} exceeds server cap $${maxStake} (PX_SEND_MAX_STAKE_DOLLARS)` }, 400);
  }

  try {
    // Golf scope: event must be an active golf event AND the line must exist
    // in that event's markets — a spoofed line_id for some other sport fails here.
    const snap = await golfSnapshot(ctx.env);
    if (!snap.eventIds.has(eventId)) {
      return pxJson({ error: `event ${eventId} is not an active golf event — refusing` }, 400);
    }
    const markets = await eventMarkets(ctx.env, eventId);
    const known = new Set<string>();
    for (const m of markets) for (const id of marketLineIds(m)) known.add(id);
    if (!known.has(lineId)) {
      return pxJson({ error: `line ${lineId} not found in golf event ${eventId} — refusing` }, 400);
    }

    // Only ladder odds are accepted by the exchange; validate up front so a
    // fat-fingered price is a clean 400, not an opaque exchange error.
    const ladder = await px(ctx.env, "GET", "/mm/get_price_ladder");
    if (ladder.ok) {
      const values: number[] = [];
      const walk = (n: any) => {
        if (typeof n === "number") values.push(n);
        else if (Array.isArray(n)) n.forEach(walk);
        else if (n && typeof n === "object") Object.values(n).forEach(walk);
      };
      walk(ladder.data?.data ?? ladder.data);
      if (values.length && !values.includes(odds)) {
        return pxJson({ error: `odds ${odds} not on the ProphetX price ladder` }, 400);
      }
    }

    const external_id = execExternalId();
    const r = await px(ctx.env, "POST", "/mm/place_wager", {
      body: { external_id, line_id: lineId, odds, stake },
    });
    return pxJson(
      { ok: r.ok, status: r.status, external_id, detail: r.data },
      r.ok ? 200 : 502
    );
  } catch (e: any) {
    return pxJson({ error: String(e?.message || e) }, 502);
  }
};
