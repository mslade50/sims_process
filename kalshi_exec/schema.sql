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

-- Durable fills store for the PnL page. Kalshi /portfolio/fills only retains
-- ~2 months, so we persist every golf fill ever seen — otherwise historical
-- cost basis vanishes as fills age out. Money columns are in DOLLARS (the
-- pnl-calc convention); the trades tape table above is unrelated and in cents.
CREATE TABLE IF NOT EXISTS fills (
  fill_id      TEXT PRIMARY KEY,
  order_id     TEXT,
  ticker       TEXT NOT NULL,
  side         TEXT,                -- yes | no
  action       TEXT,                -- buy | sell
  count        REAL NOT NULL,       -- contracts (count_fp)
  yes_price    REAL,                -- dollars
  no_price     REAL,                -- dollars
  is_taker     INTEGER,
  fee_cost     REAL,                -- dollars
  ts           INTEGER,             -- unix seconds
  created_time TEXT
);
CREATE INDEX IF NOT EXISTS idx_fills_ticker ON fills (ticker);

-- Durable settlements store — authoritative for held-to-settlement cost basis
-- (never ages out, unlike fills). Keyed by ticker (a market settles once).
CREATE TABLE IF NOT EXISTS settlements (
  ticker         TEXT PRIMARY KEY,
  market_result  TEXT,              -- yes | no | (void)
  yes_count      REAL,
  no_count       REAL,
  yes_total_cost REAL,              -- dollars
  no_total_cost  REAL,              -- dollars
  fee_cost       REAL,              -- dollars
  revenue        REAL,              -- dollars actually credited at settlement
                                    -- (used for scalar/void/tie results where the
                                    --  payout isn't count*$1 — e.g. refunded matchups)
  settled_time   TEXT,
  ts             INTEGER
);

-- Maker-bot cockpit state, pushed from the local maker shadow/live run
-- (push_maker_state.py). One row holding the latest JSON snapshot: status
-- (TRADE/HALT + reason), mode (shadow/live), and the working quotes it would
-- post. Display only — for watching what the bot wants to do.
CREATE TABLE IF NOT EXISTS maker_state (
  id          INTEGER PRIMARY KEY,   -- always 1 (single-row latest snapshot)
  payload     TEXT,                  -- JSON blob written by the maker
  updated_ts  INTEGER
);

-- Model-edge proposals pushed from the local sim (push_proposals.py). The order
-- ticket overlays these (model fair / edge) per market — display only, never
-- auto-traded. Whole table is replaced on each push.
CREATE TABLE IF NOT EXISTS proposals (
  ticker       TEXT PRIMARY KEY,
  market_type  TEXT,
  player_code  TEXT,
  event_code   TEXT,
  side         TEXT,                -- recommended side (yes|no)
  sim_prob     REAL,                -- model YES probability (0..1)
  edge_pp      REAL,                -- model edge in percentage points
  best_bid     REAL,                -- market yes bid at scan (dollars)
  best_ask     REAL,                -- market yes ask at scan (dollars)
  post_price   REAL,                -- suggested limit price (dollars)
  kelly_f      REAL,
  scan_ts      TEXT,
  updated_ts   INTEGER
);
