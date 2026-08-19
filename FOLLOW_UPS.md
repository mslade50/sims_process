# Follow-ups — 2026-08-19 session

Quick-reference for everything shipped across the deep audit + remediation +
execution-platform work (2026-08-17 → 08-19), and what's still open.
Full detail: the "Yardage Book" / "Pin Sheet" artifacts (claude.ai) and
`permanent_data/regrades_2026_08/`.

## Shipped (all pushed)

| What | Where | Key commits |
|---|---|---|
| Week-latent + 72h skew top-up (live in prod) | sims_process | `ad59133`, kernel 0.3.0 |
| mkt_regress retune (UP_DAMPEN 0.35→0.75) | sims_process | `43b7fda` |
| Monday-grading silent-loss fix + St. Jude backfill | sims_process | `3d2c426` |
| Score Edges grading restored (+65u family, graded weekly now) | sims_process | `118bd1f` |
| Book-of-record regrade: 88 NO rows +29.60u, 13 ties −13.84u | Sheets tabs | archive: `permanent_data/regrades_2026_08/` |
| Kalshi money bugs (maker H2H fairs, 429, trader NO/depth) | sims_process | `de5ec49` |
| Nightly stale-pred fallbacks gated on sync manifest | sims_process | `9b40398` |
| kalshi-exec: Cloudflare Access + hardening + ProphetX tab | sims_process | `49a0303`, `5688f84` (deployed) |
| Odds board: ProphetX orders on Optimal Portfolio page | golf_scraping | `5102006` (worker deployed, board rebuilt) |
| ETR week-latent spec (handoff, no code) | etr-golf-sims | `b035a15` → `WEEK_LATENT_OVERHAUL.html` |

## Open follow-ups

1. **ProphetX first live test (you, next portfolio use):** on the board's
   Optimal Portfolio page, click "Create live order preview" — read-only, but
   it's the client's first real API exchange. If it errors, capture the message.
   Confirmation phrase to send: `SEND LIVE PROPHETX ORDERS`. Caps: $1k/order,
   $5k/ticket (worker vars `PROPHETX_MAX_*`).
2. **Week-latent monitoring (monthly):** re-score the variance-ratio anchor
   (1.15–1.20, band [1.09, 1.30]) against closing lines; watch matchup
   favorite-side realization + finish sharp-tier ROI improve. Harness scripts
   still in session scratchpad — promote into the repo when first needed.
3. **Variance-opinion calibration (~Nov 2026):** bucket graded bets by
   dists-implied round-sd; do Poston-type high-variance win edges realize?
   Lever if not: EB-shrink category stds in sim_prep `cat_dists_player.py`
   (NOT the pred-file std_dev — it's validation-only).
4. **ETR sim week-latent:** other agent implements from
   `etr-golf-sims/WEEK_LATENT_OVERHAUL.html` (skewed latent, NO totals top-up
   there; off-state must be byte-identical).
5. **Audit leftovers (lower queue):** rebuild `bet_ledger.parquet` as a Sheets
   mirror (currently not the book of record); revive the dead kalshi outrights
   scraper (dead since 2026-06-09) for exchange CLV; UP_DAMPEN → 1.0 only after
   re-measuring mu_adj/c_adj asymmetry.

## Handy commands

- **Kernel rebuild (other machines, min 0.3.0):** see `rust/README.md` banner.
- **Parity fixture capture:** `SIMS_DUMP_FIXTURE=1 python new_sim.py --sim-only
  --use-python --no-week-latent`; pin sha from ROOT `final_scores_{t}.npy`.
- **Rotate ProphetX secrets everywhere:** update the two GitHub secrets on
  golf_scraping, then `gh workflow run sync-prophetx-secrets.yml` (pushes to
  both the kalshi-exec Pages app and the golf-odds-board Worker).
- **Revoke all kalshi-exec sessions:** `wrangler pages secret put
  SESSION_GENERATION` (any new value) from `kalshi_exec/app`.
- **Kill switches:** `--no-week-latent` / `--no-skew-cal` on new_sim, or
  `WEEK_LATENT_SD = 0.0` in sim_inputs (both repos — sim_prep is master).
