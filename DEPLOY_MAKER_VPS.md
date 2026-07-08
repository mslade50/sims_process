# Running the Kalshi maker live on an always-on VPS

Goal: move live quoting (`kalshi_maker.py --live`) off the desktop/Task-Scheduler
onto a cheap always-on Linux box so the maker has real uptime — and pull it off
the desktop's fragility (sleep, OneDrive truncation, reboots). The serverless
Cloudflare side (UI, auto-kill sweep, tape, PnL) stays as-is; this only relocates
the Python quoting brain.

> **Safety model.** Every live quote is posted with a native expiration pinned
> to the schedule (`maker_guard.quote_expiry`): it rests until just before the
> next round's first tee (minus `MAKER_TEE_BUFFER_SEC`), capped at
> `MAKER_QUOTE_TTL_MAX_SEC`, so untouched quotes keep queue priority through
> the whole quiet window. If this box dies, every resting quote self-cancels
> on its own — and because expiry is pinned to tee-off, a dead box's quotes
> can never survive into live play. When tee times are unreadable the leash
> shortens to `MAKER_QUOTE_TTL_FALLBACK_SEC` (bounds overnight-news risk
> while schedule-blind). See `reconcile_and_post` in `kalshi_maker.py` and
> `test_maker_ttl.py`.
>
> **Cadence invariant:** the run interval MUST be shorter than
> `MAKER_QUOTE_REFRESH_SEC` (default 300s) or a schedule-blind quote can lapse
> between runs. The systemd timer below fires every 2 min — comfortably under
> 300s.

---

## 0. Provider go/no-go — US region + auth smoke (do this FIRST)

Kalshi is a US-regulated exchange: trade from a **US-region** VPS. Before any
further setup, verify the box qualifies — a provider that fails any check here
is a no-go, pick another region/provider before investing in steps 1–4.

```bash
# 1. The box's egress IP must be US:
curl -s ipinfo.io/country          # must print: US
# 2. Kalshi API reachable from this network (no block/challenge):
curl -s -o /dev/null -w '%{http_code}\n' \
  https://api.elections.kalshi.com/trade-api/v2/exchange/status   # must be 200
# 3. Authenticated smoke test — pure GET, no orders touched (after steps 1-2
#    below put the key + env on the box):
python kalshi_maker.py --list-orders
```

`--list-orders` lists resting orders with a golf vs non-golf breakdown and
exits — no DELETE, no POST. If it returns your account's orders, auth + region
+ network are all proven and the provider is a GO.

## 1. Box + one-time setup

A small **US-region** Linux VPS (1 vCPU / 1 GB is plenty — this is I/O bound,
not compute). Region matters (step 0); size doesn't.

```bash
sudo apt update && sudo apt install -y python3-venv git
sudo useradd -r -m -d /opt/sims_process -s /usr/sbin/nologin maker   # service user
sudo -u maker git clone https://github.com/mslade50/sims_process.git /opt/sims_process
cd /opt/sims_process
sudo -u maker git config core.hooksPath .githooks   # OneDrive-truncation guard (harmless here)
sudo -u maker python3 -m venv .venv
sudo -u maker .venv/bin/pip install -r requirements.txt
```

## 2. Secrets (never in the repo)

Put the Kalshi RSA `.pem` **outside** the tree and lock it down:

```bash
sudo install -d -m 700 -o maker -g maker /opt/secrets
sudo -u maker cp kalshi_private.pem /opt/secrets/kalshi_private.pem   # copy your key over
sudo chmod 600 /opt/secrets/kalshi_private.pem
# Google service account for the phone kill-switch / round_config sheet:
sudo -u maker cp credentials.json /opt/sims_process/credentials.json
```

Create `/etc/kalshi-maker.env` (root-owned, `chmod 640`, group `maker`):

```ini
# ── Kalshi auth (see kalshi_auth.py) ─────────────────────────────
KALSHI_ACCESS_KEY=<your access key id>
KALSHI_PRIVATE_KEY_PATH=/opt/secrets/kalshi_private.pem
# ── Data ─────────────────────────────────────────────────────────
DATAGOLF_API_KEY=<datagolf key>
# sim_fairs.json comes from the repo (git pull in the wrapper). If you skip the
# pull and read origin over the API instead, set a PAT for the private repo:
# SIMS_PROCESS_PAT=<github pat with repo:read>
# ── Dashboard Maker tab mirror (push_maker_state.py) ─────────────
MAKER_STATE_TOKEN=<same value as the Cloudflare Pages secret>
# ── Schedule-pinned quote expiry / dead-man's switch (defaults shown) ─
MAKER_QUOTE_TTL_MAX_SEC=43200      # max rest: 12h (quiet-window cap)
MAKER_QUOTE_TTL_FALLBACK_SEC=3600  # tee times unreadable: 1h leash
MAKER_TEE_BUFFER_SEC=900           # expire 15 min before first tee
MAKER_QUOTE_TTL_SEC=600            # tee imminent/past: 10 min leash
MAKER_QUOTE_REFRESH_SEC=300        # re-arm eligible below 5 min left
# ── Kelly pricing floors (min Kelly fraction a fill must keep) ────
MAKER_KELLY_MIN_WINNER=0.25
MAKER_KELLY_MIN_TOP5=0.25
MAKER_KELLY_MIN_TOP10=0.25
MAKER_KELLY_MIN_TOP20=0.15
MAKER_KELLY_MIN_H2H=0.05
# ── Exposure governor (tune to taste; these are the code defaults) ─
MAKER_CAP_MARKET_USD=50
MAKER_CAP_EVENT_USD=400
MAKER_CAP_TOTAL_USD=1000
MAKER_MAX_NEW_USD_PER_RUN=300
MAKER_MAX_ORDERS_PER_RUN=40
# ── Risk limits (defaults shown; every check fail-closes) ────────
# Daily realized-loss circuit breaker (golf settlements, ET day; 0 disables):
MAKER_MAX_DAILY_LOSS_USD=200
# Balance-vs-caps sanity: halts when cash can't fund what the caps allow this
# run. Lower the caps to bankroll rather than setting the override:
# MAKER_ALLOW_CAPS_OVER_BALANCE=1
# Fairs freshness: pre-event cap / intra-event (post-R1) cap in hours:
MAKER_FAIRS_MAX_AGE_HRS=48
MAKER_FAIRS_MAX_AGE_LIVE_HRS=8
# ── Alerting (dead-man + Telegram; strongly recommended for --live) ─
# healthchecks.io check URL — run_maker.sh pings it after every run:
HEALTHCHECKS_URL=<https://hc-ping.com/...>
# Telegram on HALT<->TRADE transitions, post-fail streaks, new fills:
TELEGRAM_BOT_TOKEN=<bot token>
TELEGRAM_CHAT_ID=<chat id>
```

```bash
sudo chown root:maker /etc/kalshi-maker.env && sudo chmod 640 /etc/kalshi-maker.env
```

## 3. Run wrapper — `deploy/run_maker.sh` (in the repo)

The wrapper is committed at `deploy/run_maker.sh` — the `git clone` in step 1
already put it on the box. It: pulls `--ff-only` (fresh code + sim_fairs.json),
sources the venv + `/etc/kalshi-maker.env`, runs
`python kalshi_maker.py --live --no-matchups`, mirrors the cockpit snapshot via
`push_maker_state.py`, then pings `HEALTHCHECKS_URL` (or `…/fail` when the run
errored) as the dead-man heartbeat.

```bash
sudo chmod 755 /opt/sims_process/deploy/run_maker.sh
```

## 4. systemd oneshot + timer

`/etc/systemd/system/kalshi-maker.service`:

```ini
[Unit]
Description=Kalshi maker — one live reconcile pass
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=maker
Group=maker
WorkingDirectory=/opt/sims_process
ExecStart=/opt/sims_process/deploy/run_maker.sh
TimeoutStartSec=60           # From 142 shadow runs (2026-06-29..07-02): median 6.5s,
                             # p95 8.8s, max 12.3s. 60s ≈ 7x shadow p95 — headroom for
                             # the live reconcile path shadow never exercises (0.12s
                             # auth throttle x up to 40 orders + POST/DELETE round
                             # trips ≈ +20s worst case). Stays under the 2-min cadence.
                             # Revisit against `done in Xs` after the supervised live
                             # test — SIGTERM mid-reconcile is worse than a long run.
Nice=5
```

`/etc/systemd/system/kalshi-maker.timer`:

```ini
[Unit]
Description=Run the Kalshi maker every 2 minutes

[Timer]
OnBootSec=1min
OnUnitActiveSec=2min         # < MAKER_QUOTE_REFRESH_SEC (300s): quotes never lapse
AccuracySec=10s
Persistent=false

[Install]
WantedBy=timers.target
```

A oneshot won't overlap itself — if a run runs long, the next tick waits. Enable:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now kalshi-maker.timer
```

## 5. Kill switches (panic buttons)

Any one halts `--live` AND makes the next run pull the bot's own resting quotes;
the rolling TTL is the final backstop if the box itself is unreachable.

```bash
# local file (fastest, on-box):
sudo -u maker touch /opt/sims_process/permanent_data/MAKER_HALT
# env (edit /etc/kalshi-maker.env): MAKER_KILL=1
# phone: set round_config `maker_enabled` = no/0/false in the Google Sheet
# stop scheduling entirely (quotes then lapse on their native expiry —
# at the latest by first tee, or within the fallback leash if schedule-blind):
sudo systemctl disable --now kalshi-maker.timer
```

## 6. Observe

```bash
journalctl -u kalshi-maker.service -f          # live run logs
systemctl list-timers kalshi-maker.timer       # next/last fire
```

The dashboard **Maker tab** populates once `push_maker_state.py` runs with
`MAKER_STATE_TOKEN` set (step 2). The Cloudflare cron status endpoint continues
to show the tape/PnL/auto-kill collector independently.

**Push alerting** (set up in step 2's env; see `maker_alerts.py`):

- **healthchecks.io** — create a check with a ~5 min grace period and put its
  ping URL in `HEALTHCHECKS_URL`. Silence = the box/timer/loop is dead; the TTL
  has already pulled the quotes, this tells you to go fix it.
- **Telegram** — `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` (same bot as the
  reprice alerts) gets: HALT↔TRADE transitions with the guard reason, ≥3
  consecutive runs with failed POSTs, and every new fill.

---

## Go-live sequence (do NOT skip to unattended)

You have never posted a live quote, so stage it:

1. **Shadow on the box first.** Run `MAKER_SHADOW=1 python maker_shadow.py`
   manually — hard no-order dry run against live books. Confirm it generates
   sane outright quotes and the guards behave (fairs fresh, not-live, kill).
2. **One supervised live post.** Run `run_maker.sh` by hand once, watching
   `journalctl`. Confirm: (a) a quote appears on Kalshi with an `expires=…` tag,
   and (b) **kill the process and watch the quote auto-cancel within the TTL** —
   this is the real end-to-end test of the dead-man's switch.
3. **Watch for re-arm post-fails.** On the 2nd/3rd supervised run, confirm the
   `re-armed` count in the summary line matches successful `[post]` lines (not
   `[post-fail]`). If re-arms fail, Kalshi is rejecting the re-post — tell me and
   we'll adjust (the re-arm already uses a fresh `client_order_id`, so this
   should be fine, but verify it live).
4. **Then enable the timer** for unattended operation.

## Open items / caveats

- **Matchups.** Wrapper starts with `--no-matchups`. Confirm what live inputs the
  matchup scan needs on the VPS (e.g. `final_scores*.npy`) before dropping the flag.
- **Queue priority vs crash window.** Resolved by the schedule-pinned expiry:
  an untouched quote is never cancel/re-posted during the quiet window (it keeps
  its queue spot until tee-off), while a dead box's quotes still die before
  play. The residual trade-off lives in `MAKER_QUOTE_TTL_FALLBACK_SEC` (leash
  while tee times are unreadable) and `MAKER_QUOTE_TTL_MAX_SEC` (overnight-news
  bound on a dead box).
- **Clock.** The TTL is wall-clock; keep the box on NTP (`timedatectl` → NTP=yes).
```
