export const OUTRIGHT_TYPES: Set<string>;

export interface TapeFilterParams {
  from?: string | number | null;
  to?: string | number | null;
  min_price?: string | number | null;
  max_price?: string | number | null;
  min_size?: string | number | null;
  market?: string | null;
  event?: string | null;
}

export function buildTapeWhere(params: TapeFilterParams): { whereSql: string; binds: any[] };
export function tapeSummarySql(whereSql: string): string;
export function tapeRawSql(whereSql: string): string;
export function shapeSummaryRow(r: any): {
  ticker: string;
  market_type: string;
  player_code: string;
  event_code: string;
  prints: number;
  volume: number;
  vwap: number | null;
  open: number | null;
  last: number | null;
  high: number | null;
  low: number | null;
  change: number;
  changePct: number;
  buyVol: number;
  sellVol: number;
  delta: number;
  imbalance: number;
  first_ts: number | null;
  last_ts: number | null;
};
