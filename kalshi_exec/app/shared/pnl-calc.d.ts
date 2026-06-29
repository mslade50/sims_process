export interface PnlFill {
  ticker: string;
  side: "yes" | "no";
  action: "buy" | "sell" | string;
  count: number;
  yes_price: number;
  no_price: number;
  fee_cost: number;
  ts?: number | null;
  created_time?: string;
}
export interface PnlSettlement {
  market_result?: string;
  yes_count?: number;
  no_count?: number;
  yes_total_cost?: number;
  no_total_cost?: number;
  fee_cost?: number;
  revenue?: number | null;
  settled_time?: string | null;
}
export interface PnlArgs {
  fills?: PnlFill[];
  settlements?: Record<string, PnlSettlement>;
  positions?: Record<string, number>;
  marks?: Record<string, { yes_bid?: number; yes_ask?: number }>;
  titles?: Record<string, string>;
  classify?: (ticker: string) => { marketType: string; eventCode: string; playerCode: string };
  isGolf?: (ticker: string) => boolean;
}
export interface PnlRow {
  ticker: string;
  title: string;
  market_type: string;
  event_code: string;
  player_code: string;
  is_golf: boolean;
  is_open: boolean;
  side: string;
  contracts: number;
  avg_cost: number;
  exposure: number;
  invested: number;
  auto_converted: number;
  cashflow: number;
  settlement_revenue: number;
  realized: number;
  unrealized: number;
  fees: number;
  mark: number | null;
  net: number;
  market_result: string;
  settled_time: string | null;
  first_fill_ts: number | null;
  last_fill_ts: number | null;
  return_pct: number | null;
}
export interface PnlTotals {
  realized: number;
  unrealized: number;
  fees: number;
  net: number;
  invested: number;
  open_count: number;
  settled_count: number;
}
export function computePnl(args: PnlArgs): { rows: PnlRow[]; totals: PnlTotals };
