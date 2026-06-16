# Pipeline Audit — Follow-ups to Investigate

From a multi-agent adversarial audit (June 2026): 8 dimensions, 41 agents, 32 findings,
**31 confirmed** real against the code after refute-by-default verification. This file is
the backlog of items **not yet addressed**. Each is mechanically confirmed unless noted.

---

## Already fixed (this session) — for context
- **`weekly-cleanup.yml`** — now tournament-aware: reads `tourney` from `sim_inputs.py`,
  preserves the active week, only cleans past weeks, aborts if the slug is unreadable.
- **`new_sim.py`** — Top-5/10/20 finish probs routed through the Rust
  `sims_kernel.aggregate_round` (the same path `round_sim` uses): ~418s → 0.18s, output
  FP-identical to pandas. pandas loop kept as `--use-python` fallback.
- **`round_sim.py`** — email banner shows matchup lines priced per sharp book
  (Pinnacle/BetOnline/BetCris); red/orange when a book is offline / sparse.
- **`push_dashboard_data.git_push`** — the deploy commit is now **scoped** to
  `dashboard_data/` + `sim_inputs.py` + prep files (explicit pathspec). A stray staged
  file or mid-edit change can no longer be swept into the auto-deploy commit.
- **`sheets_storage.py`** — Sheet **write-dedup** on all 4 bet tabs (Tournament/Round
  Matchups, Finish Positions, Score Edges) keyed on player(s)/market/event/book/odds,
  keeping the FIRST write. Stops the second sim pass and mid-week re-runs from writing
  identical bet rows.

## NOT a bug — by design (do NOT "fix")
- **Storing the pre-`final_predictions` (first-pass, pre-market-regression) bets is
  intended.** The audit flagged the ledger's first-write-wins keeping the un-regressed
  first pass as "Critical"; that is the desired behavior — the first pass is the raw model
  edge before it is tempered toward the market. The new Sheet write-dedup reinforces this
  (keeps the first write).

---

## High priority

1. **Stale-field backstop does not cover the matchup-pricing path or R1**
   (`round_sim.py` ~614, 2367-2386, 4285-4316). The 90% field-match `RuntimeError` only
   fires inside `_filter_to_active_players` (finish-position sim, rounds 2-3). The
   bet-generating round-matchup path and all of R1 (round=0) price with **no** event/field
   check — a stale `model_predictions_r{N}.csv` can silently price bets stored under the
   *current* `event_id`. **Fix:** intersect `model_predictions` players with the current
   DataGolf field before matchup pricing; abort/alert if overlap < ~90%. Share across the
   round=0 and matchup paths.

2. **Hardcoded DataGolf API key fallback committed in source**
   (`new_sim.py:143`; worse at `live_stats_engine.py:83`, which has no env var at all).
   `api_utils.py`/`mkt_regress.py`/`round_sim.py` correctly use no fallback. **Fix:** drop
   both fallbacks, fail-fast if unset, rotate the key, purge from history.

3. **Kalshi outright/H2H pricers use a fragile substring slug matcher**
   (`new_sim.py:1811-1841, 2407-2435`; `round_sim.py:1333-1365`). The inline matcher strips
   to `{'us'}` for `us_open`; Kalshi's "U.S. Open" stylization → no match → **silent
   fallback to the most-common event by market count** → edges vs the wrong event's prices.
   Relevant *now* (a major). The robust `kalshi_match.match_markets` exists and is already
   used by `kalshi_ancillary`/`kalshi_preflight`. **Fix:** wire it into all three pricers.

4. **Exchange (Kalshi/NoVig) storage bypasses the Monday-3pm time-gate**
   (`new_sim.py:3282-3312`, inside `send_tournament_email`). No `is_valid_run_time()` /
   `SKIP_STORAGE` / `--reprice` guard, so exchange bets store even on an early-Monday
   pre-field run the gate exists to suppress. (The duplicate-row half is now handled by the
   Sheet dedup; the gate bypass remains.) **Fix:** wrap in the same guard as the main
   storage block.

## Medium priority

5. **`sync_event_files.py` reads the tourney slug positionally from cell B20** (`:50`).
   Every other consumer reads `tourney` by label via `sheet_config`. A single Sheet row
   insert (phone edit) silently breaks the event-change pred-file guard (blank/non-slug B20
   → guard dead or spurious wipe). **Fix:** read by label (`load_config()['tourney']`).

6. **R4 skill update undoes the R2 (not R3) adjustment** (`live_stats_engine.py:633-663`).
   `_totals_r3r4` never writes fresh `tot_sg_adj/tot_resid_adj`, so the R4 run subtracts the
   stale R2 value and leaves the R3 SG adjustment baked in — breaks
   `Post = Pre + total_adjustment` for `updated_pred_final` (bounded: R4 is terminal, no
   downstream consumer). **Fix:** persist `tot_sg_adj=fresh_adj`, `tot_resid_adj=0.0` in
   `_totals_r3r4`.

7. **Regression-flips-side creates a contradictory pair in the parquet LEDGER**
   (ledger dedup key includes `bet_on`/`opponent`). The Sheet now dedups this via a
   line-based key, but the ledger key still differs across a flipped second-pass bet, so
   both sides can land as separate ledger rows. **Fix:** side-independent ledger matchup key
   (sorted player pair) — or skip second-pass storage entirely.

8. **`monday-grading.yml` has no `permissions:` block** — every other main-pushing workflow
   grants `contents: write`; this one doesn't. If the default token is read-only the Monday
   dashboard `git push` is rejected and swallowed (Render serves stale data, green run).
   **Fix:** add `permissions: contents: write`; make `git_push` exit non-zero + alert on
   final failure.

9. **Documented DG-override (`pred<0.5` / `dg_override_players`) is implemented nowhere**
   (`mkt_regress.py:19-22` imports the list, never uses it; `pre_sim_skill.py` doesn't
   exist; `live_stats_engine.py:976-990` hardcodes the mask to False). `WEEKLY_PROCESS.md`
   tells the operator it runs. **Fix:** re-implement in `mkt_regress`, or update the docs and
   remove the dead import.

10. **Live distributions dashboard mislabels raw `prob_u` as dead-heat**
    (`dashboard/pages/distributions.py:58-72`). `rank_probs_live_{tourney}.parquet` (from
    `round_sim`) carries only `prob_u` = raw min-rank (semantically new_sim's `prob_ndh`),
    so the page's "dead-heat" Win/T5/T10/T20 are actually raw min-rank. **Fix:** make
    `_compute_stats` source-aware, or have `round_sim` also emit `prob_ndh`.

11. **`reprice.yml` window gate uses UTC day-of-week** (`:22-31`) — `date -u +%u` drops an
    early-Thursday-EST scraper dispatch (every other time surface is US/Eastern).
    **Fix:** `TZ=America/New_York date +%u`, or gate on fair-table existence.

## Low / hygiene

- **weekly-cleanup `*${TOURNEY}*` glob is unanchored** → `rbc` would match `rbc_canada`
  (latent; calendar order saves it). Anchor to `*_${TOURNEY}.<ext>` or key on `event_id`.
- **`git pull --rebase --autostash` in `git_push` can wedge a dirty local tree** on a race.
  Adopt the `publish_sim_fairs._git_push` temp-index pattern (also retires the autostash).
- **Sim R2 residual omits the live engine's `clip(lower=-0.5)`** (`round_sim.py:853` /
  `new_sim.py:825` vs `live_stats_engine.py:587` + `rust/src/round_cascade.rs:319`).
  Divergence only for summed R2 residuals < -0.5 SG; fix in both Python sims AND the Rust
  kernel.
- **Kalshi inline name normalizer mishandles multi-word surnames** (`rsplit(' ',1)`) —
  van/de/von players dropped unless hand-mapped in `name_replacements`. Use `km.norm_name`.
- **Third-party CI actions pinned to mutable tags** (`build-wheels.yml`,
  `deploy-odds-screen.yml`) + rolling wheel pip-installed with no hash check. Pin to SHAs.
- **`run-sim.yml` interpolates dispatch inputs into a `run:` line** (template-injection
  shape; gated by write access). Pass via `env:` and reference `"$VAR"`.
- **Empty `cleanup_backup/`** (`weekly-cleanup.yml:24`) gives false safety — nothing is
  copied in. Populate + upload as an artifact, or remove the misleading `mkdir`.
- **Legacy `fetch_matchup_odds()` lacks the `isinstance(list)` guard** (`round_sim.py:2308`,
  dead code, zero call sites). Delete it or add the guard.

---

## Systemic patterns (root causes worth a structural pass)
1. **Storage/dedup key semantics differ between the Sheet and the ledger** — unify them
   (especially the matchup side question: line-based vs `bet_on`-inclusive).
2. **Two correct utilities exist but aren't wired into hot paths** — `kalshi_match.match_markets`
   and `publish_sim_fairs._git_push` (temp-index). Consolidation, not invention.
3. **`prob_u` vs `prob_ndh` schema inversion handled inconsistently** — `kalshi_maker`/
   `publish_sim_fairs` are source-aware; the dashboard and `novig_pricer` aren't. Have
   `round_sim` emit both columns so every consumer can trust column names.
4. **Identity from mutable / positional / UTC sources** — B20 positional read, UTC DOW gate,
   unanchored slug globs. Prefer label-keyed / EST-scoped / `event_id`-based identity.
5. **Silent failure as house style** — bare `except: print(...)`, `git push` that
   prints-but-never-raises, matchers that fall back to "most-common" rather than aborting.
   Add loud failures + alerts on the storage and push paths.

## Coverage gaps (the audit did not fully examine these — worth a dedicated pass)
- `reprice.py` repricing math + the `round_h2h` fair-table contract (only the guard boundary).
- `mkt_regress.py` regression math (mkt/mu/c_adj caps + asymmetric dampening) — only the
  missing DG-override was checked.
- Full `validate_rust_python.py` parity-gate review beyond the two cascade items.
- `monday_grading.py` / `grade_bets.py` settlement correctness + `units_wagered`/`kelly_stake`
  derivation + write-back to source tabs.
- `live_stats_engine.py` weather/dewpoint centering + wind-blend invariants (CLAUDE.md flags
  these as past bug sources; only the skill cascade was audited).
