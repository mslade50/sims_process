-- Kalshi execution platform — shared D1 schema (database: kalshi-exec)

-- Orders we placed with a kill time. Belt-and-suspenders to Kalshi's native
-- expiration_ts: the cron Worker force-cancels any remainder at kill_ts + 60s.
CREATE TABLE IF NOT EXISTS pending_kills (
  order_id      TEXT PRIMARY KEY,
  ticker        TEXT NOT NULL,
  kill_ts       INTEGER NOT NULL,   -- unix seconds; native expiry time
  hard_kill_ts  INTEGER NOT NULL,   -- kill_ts + 60; when cron force-cancels
  status        TEXT NOT NULL DEFAULT 'pending',  -- pending | killed | gone | error
  note          TEXT,
  created_ts    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pending_kills_due
  ON pending_kills (hard_kill_ts) WHERE status = 'pending';

-- Golf OUTRIGHT trade tape (winner / top_5 / top_10 / top_20). Populated by the
-- cron Worker polling /markets/trades. H2H is intentionally excluded.
CREATE TABLE IF NOT EXISTS trades (
  trade_id     TEXT PRIMARY KEY,
  ticker       TEXT NOT NULL,
  market_type  TEXT NOT NULL,       -- winner | top_5 | top_10 | top_20
  event_code   TEXT,
  player_code  TEXT,
  ts           INTEGER NOT NULL,    -- created_time as unix seconds
  yes_price    INTEGER,             -- cents
  no_price     INTEGER,             -- cents
  count        INTEGER NOT NULL,    -- size (contracts)
  taker_side   TEXT                 -- yes | no
);
CREATE INDEX IF NOT EXISTS idx_trades_ts          ON trades (ts);
CREATE INDEX IF NOT EXISTS idx_trades_ticker      ON trades (ticker);
CREATE INDEX IF NOT EXISTS idx_trades_market_type ON trades (market_type);

-- Small key/value for collector cursor + run telemetry.
CREATE TABLE IF NOT EXISTS collector_state (
  key   TEXT PRIMARY KEY,
  value TEXT
);
