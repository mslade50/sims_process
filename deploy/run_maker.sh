#!/usr/bin/env bash
# Kalshi maker live wrapper — invoked by kalshi-maker.service (systemd oneshot).
# See DEPLOY_MAKER_VPS.md. NOT `set -e`: the maker's exit status is captured so
# the dead-man ping still fires (with /fail) when a run errors.
set -uo pipefail
cd /opt/sims_process

# Fast-forward only: pull fresh sim_fairs.json + code, never clobber local state.
git pull --ff-only origin main || echo "git pull failed — using existing tree"

# shellcheck disable=SC1091
source .venv/bin/activate
set -a
# shellcheck disable=SC1091
source /etc/kalshi-maker.env
set +a

# START with --no-matchups: outrights price off sim_fairs.json (pulled above);
# matchups need live inputs this box may not have yet. Drop the flag once
# matchup inputs are confirmed present on the VPS.
python kalshi_maker.py --live --no-matchups
status=$?

# Mirror the cockpit snapshot to the dashboard Maker tab (never posts orders).
python push_maker_state.py || true

# Dead-man ping (healthchecks.io or compatible; set HEALTHCHECKS_URL in
# /etc/kalshi-maker.env). A missed ping means the box/timer/loop is dead — you
# get paged even though the rolling TTL has already pulled the quotes. A run
# that errored pings the /fail endpoint so the check flips red immediately
# instead of waiting out the grace period.
if [ -n "${HEALTHCHECKS_URL:-}" ]; then
  if [ "$status" -eq 0 ]; then
    curl -fsS -m 10 --retry 3 "$HEALTHCHECKS_URL" >/dev/null || true
  else
    curl -fsS -m 10 --retry 3 "$HEALTHCHECKS_URL/fail" >/dev/null || true
  fi
fi

exit "$status"
