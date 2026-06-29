import { Env, kalshi, json, classifyTicker, SERIES_BY_TYPE } from "../../shared/kalshi";

// GET /api/markets?type=winner|top_5|top_10|top_20
// Lists open golf outright markets for one series (for the order-ticket search).
export const onRequestGet: PagesFunction<Env> = async (ctx) => {
  const url = new URL(ctx.request.url);
  const type = url.searchParams.get("type") || "winner";
  const series = SERIES_BY_TYPE[type];
  if (!series) return json({ error: `unknown type ${type}` }, 400);

  const markets: any[] = [];
  let cursor: string | undefined;
  for (let i = 0; i < 20; i++) {
    const r = await kalshi(ctx.env, "GET", "/markets", {
      query: { series_ticker: series, status: "open", limit: 200, cursor },
    });
    if (!r.ok) return json({ error: "kalshi /markets failed", status: r.status, detail: r.data }, 502);
    const mkts = r.data.markets || [];
    for (const m of mkts) {
      markets.push({
        ticker: m.ticker,
        title: m.title,
        subtitle: m.yes_sub_title ?? m.subtitle ?? "",
        yes_bid: m.yes_bid,
        yes_ask: m.yes_ask,
        no_bid: m.no_bid,
        no_ask: m.no_ask,
        last_price: m.last_price,
        volume: m.volume,
        ...classifyTicker(m.ticker),
      });
    }
    cursor = r.data.cursor;
    if (!cursor || mkts.length < 200) break;
  }
  return json({ type, series, count: markets.length, markets });
};
