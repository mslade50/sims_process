import { Env, kalshi, json, classifyTicker } from "../../shared/kalshi";

// GET /api/reference — human-readable names for the UI:
//   players:    { player_code -> "Full Name" }   (from market yes_sub_title)
//   events:     { event_code  -> "Tournament" }  (from /events sub_title/title)
//   eventsList: [{ code, name, markets }]         (active events, busiest first)
//
// Cached ~10 min in the warm isolate — names change slowly and this fans out to
// several Kalshi calls. ?refresh=1 forces a refresh.
const SERIES = ["KXPGATOUR", "KXPGATOP5", "KXPGATOP10", "KXPGATOP20"];
const TTL_MS = 10 * 60 * 1000;
let _cache: { data: any; at: number } = { data: null, at: 0 };

function cleanEventName(subTitle?: string, title?: string): string {
  let n = (subTitle || "").trim().replace(/^20\d\d\s+/, ""); // "2026 Travelers Championship" -> "Travelers Championship"
  if (!n) n = (title || "").replace(/\s+Winner$/i, "").trim(); // "Travelers Championship Winner" -> "..."
  return n;
}

export const onRequestGet: PagesFunction<Env> = async (ctx) => {
  const force = new URL(ctx.request.url).searchParams.get("refresh") === "1";
  const now = Date.now();
  if (!force && _cache.data && now - _cache.at < TTL_MS) {
    return json({ ..._cache.data, cached: true });
  }

  // Fan out per series in parallel; each gathers event names + player names + counts.
  const perSeries = await Promise.all(
    SERIES.map(async (series) => {
      const players: Record<string, string> = {};
      const events: Record<string, string> = {};
      const counts: Record<string, number> = {};

      // Event names: OPEN (current) + SETTLED (past). PnL is all-time, so settled
      // tournaments must resolve too — they aren't in the open list. (Kalshi
      // /events requires an explicit status; with none it returns nothing.)
      for (const status of ["open", "settled"]) {
        let ecursor: string | undefined;
        for (let i = 0; i < 12; i++) {
          const ev = await kalshi(ctx.env, "GET", "/events", {
            query: { series_ticker: series, status, limit: 200, cursor: ecursor },
          });
          if (!ev.ok) break;
          const evs = ev.data.events || [];
          for (const e of evs) {
            const code = String(e.event_ticker || "").split("-").slice(1).join("-");
            if (code && !events[code]) events[code] = cleanEventName(e.sub_title, e.title);
          }
          ecursor = ev.data.cursor;
          if (!ecursor || evs.length < 200) break;
        }
      }

      let cursor: string | undefined;
      for (let i = 0; i < 10; i++) {
        const r = await kalshi(ctx.env, "GET", "/markets", {
          query: { series_ticker: series, status: "open", limit: 200, cursor },
        });
        if (!r.ok) break;
        const mkts = r.data.markets || [];
        for (const m of mkts) {
          const cls = classifyTicker(m.ticker || "");
          if (cls.playerCode && m.yes_sub_title) players[cls.playerCode] = m.yes_sub_title;
          if (cls.eventCode) counts[cls.eventCode] = (counts[cls.eventCode] || 0) + 1;
        }
        cursor = r.data.cursor;
        if (!cursor || mkts.length < 200) break;
      }
      return { players, events, counts };
    })
  );

  const players: Record<string, string> = {};
  const events: Record<string, string> = {};
  const counts: Record<string, number> = {};
  for (const s of perSeries) {
    Object.assign(players, s.players);
    Object.assign(events, s.events);
    for (const [k, v] of Object.entries(s.counts)) counts[k] = (counts[k] || 0) + v;
  }

  const eventsList = Object.keys(counts)
    .map((code) => ({ code, name: events[code] || code, markets: counts[code] }))
    .sort((a, b) => b.markets - a.markets);

  const data = { players, events, eventsList, fetched_ts: new Date().toISOString() };
  _cache = { data, at: now };
  return json({ ...data, cached: false });
};
