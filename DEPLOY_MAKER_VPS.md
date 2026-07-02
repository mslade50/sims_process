# Running the Kalshi maker live on an always-on VPS

Goal: move live quoting (`kalshi_maker.py --live`) off the desktop/Task-Scheduler
onto a cheap always-on Linux box so the maker has real uptime — and pull it off
the desktop's fragility (sleep, OneDrive truncation, reboots). The serverless
Cloudflare side (UI, auto-kill sweep, tape, PnL) stays as-is; this only relocates
the Python quoting brain.

> **Safety model.** Every live quote is now posted with a rolling native
> expiration (`MAKER_QUOTE_TTL_SEC`, default 600s) and re-armed each cycle. If
> this box dies, every resting quote self-cancels within the TTL with no
> dependence on anything else staying up. See `reconcile_and_post` in
> `kalshi_maker.py` and `test_maker_ttl.py`.
>
> **Cadence invariant:** the run interval MUST be shorter than
> `MAKER_QUOTE_REFRESH_SEC` (default 300s) or quotes lapse between runs. The
> systemd timer below fires every 2 min — comfortably under 300s.

---

## 1. Box + one-time setup

Any small Linux VPS (1 vCPU / 1 GB is plenty — this is I/O bound, not compute).

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
# ── Rolling TTL dead-man's switch (defaults shown) ───────────────
MAKER_QUOTE_TTL_SEC=600
MAKER_QUOTE_REFRESH_SEC=300
# ── Exposure governor (tune to taste; these are the code defaults) ─
MAKER_CAP_MARKET_USD=50
MAKER_CAP_EVENT_USD=400
MAKER_CAP_TOTAL_USD=1000
MAKER_MAX_NEW_USD_PER_RUN=300
MAKER_MAX_ORDERS_PER_RUN=40
```

```bash
sudo chown root:maker /etc/kalshi-maker.env && sudo chmod 640 /etc/kalshi-maker.env
```

## 3. Run wrapper — `/opt/sims_process/deploy/run_maker.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /opt/sims_process
# Fast-forward only: pull fresh sim_fairs.json + code, never clobber local state.
git pull --ff-only origin main || echo "git pull failed — using existing tree"
# shellcheck disable=SC1091
source .venv/bin/activate
set -a; source /etc/kalshi-maker.env; set +a
# START with --no-matchups: outrights price off sim_fairs.json (pulled above);
# matchups may need live inputs this box doesn't have yet. Drop the flag once
# you've confirmed matchup inputs are present on the VPS.
python kalshi_maker.py --live --no-matchups
# Mirror the cockpit snapshot to the dashboard Maker tab (never posts orders).
python push_maker_state.py || true
```

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
TimeoutStartSec=100          # a run must finish well under the 2-min cadence
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
# stop scheduling entirely (quotes then expire within MAKER_QUOTE_TTL_SEC):
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
- **Queue priority vs crash window.** Re-arming every ~5 min resets queue priority
  on refreshed quotes. Raising `MAKER_QUOTE_TTL_SEC`/`_REFRESH_SEC` re-arms less
  often (better queue priority) at the cost of a longer stranded-exposure window
  on a crash. 600/300 is a conservative start.
- **Clock.** The TTL is wall-clock; keep the box on NTP (`timedatectl` → NTP=yes).
```
