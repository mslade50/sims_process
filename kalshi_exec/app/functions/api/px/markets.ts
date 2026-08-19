import { PxEnv, pxConfigured, pxJson, golfSnapshot, eventMarkets } from "../../../shared/prophetx";

// GET /api/px/markets           -> active golf tournaments + events
// GET /api/px/markets?event=ID  -> markets for one golf event
// Session-gated by _middleware (master login), like every /api route.
export const onRequestGet: PagesFunction<PxEnv> = async (ctx) => {
  if (!pxConfigured(ctx.env)) {
    return pxJson({ error: "ProphetX not configured (PROPHETX_ACCESS_KEY/SECRET_KEY secrets)" }, 503);
  }
  const url = new URL(ctx.request.url);
  const eventId = url.searchParams.get("event") || "";
  try {
    const snap = await golfSnapshot(ctx.env);
    if (!eventId) {
      return pxJson({ tournaments: snap.tournaments, events: snap.events });
    }
    if (!snap.eventIds.has(eventId)) {
      return pxJson({ error: `event ${eventId} is not an active golf event` }, 400);
    }
    const markets = await eventMarkets(ctx.env, eventId);
    return pxJson({ event: eventId, markets });
  } catch (e: any) {
    return pxJson({ error: String(e?.message || e) }, 502);
  }
};
