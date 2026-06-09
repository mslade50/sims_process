# Finish the Rust migration — resume plan

Written 2026-06-08 EOD, right after flipping `new_sim` to the Rust kernel in
production. Self-contained: assume no memory of that session. Full design doc is
`RUST_MIGRATION_PLAN.md` (on branch `rust-sim-kernel-port`); deep context in
Claude memory `rust-port-sim-kernels`.

---

## Where we are tonight

- **`new_sim.py` on `main` (commit `51bca38`, pushed): Rust is the production default.**
  `sims_kernel.run_pretournament` (Rust, seed 456) replaces the Python draw output.
  The Python draw STILL RUNS as a shadow and Rust overwrites its `final_scores`.
  `--use-python` keeps the Python output; a `try/except` falls back to Python (with
  a warning) if the wheel is missing/errors. Same commit fixed a pre-existing
  `decimal_to_american` ZeroDivisionError on `decimal_odds <= 1.0` placeholder lines.
- **Verified live:** full `rbc_canada` run (147 players, 100k sims) on the Rust kernel
  sent all 3 emails + stored bets (ledger 1077→1130). Win probs within **0.32% max**
  of the Python baseline; aggregation/h2h **bit-exact** vs the 15:47 prod run.
- **`scoring_baseline.py` on `main` (`4c1a116`, pushed):** course-specific SG variance
  filter (Osprey 935) + left-tail winsorization. New numbers pushed to `round_config`.
- **The Rust crate `sims_kernel` lives ONLY on branch `rust-sim-kernel-port`** (`f58a4d8`,
  pushed to origin). It is **gitignored on `main`** (`.gitignore:180` = `/*/`). The wheel
  (`cp310-abi3`) is installed on THIS machine only.
- **`round_sim` is NOT flipped.** `run_remaining_rounds` is ported + stat-equiv vs the
  reference, but Phase 5 (single-round score card, seed 789) is not started and round_sim
  has no production verification.

## First 10 minutes — sanity check (do before anything)

OneDrive can truncate source files overnight; confirm the tree + kernel are intact.
```bash
git status && git log --oneline -5
git switch rust-sim-kernel-port            # crate only lives here until Task 1
cd rust && cargo test --lib && maturin build --release && cd ..
pip install --force-reinstall --no-deps rust/target/wheels/sims_kernel-0.1.0-cp310-abi3-win_amd64.whl
python -c "import sims_kernel as k; print(k.version(), k.selftest(), hasattr(k,'run_pretournament'))"
python rust/fixtures/test_ops_parity.py && python rust/fixtures/test_agg_parity.py \
  && python rust/fixtures/test_cascade_parity.py && python rust/fixtures/test_round_cascade_parity.py \
  && python rust/fixtures/verify_cascade_against_prod.py
git switch main
```
All green → proceed.

---

## Task 1 (DO FIRST) — put the `rust/` crate on `main`

**Why:** `main` imports `sims_kernel` but doesn't contain it, so on any machine/clone
without the locally-installed wheel it silently falls back to the Python shadow.

Do **not** full-merge the branch (its base is stale old-`main` → would drag back deleted
memorial files etc.). Bring just the paths:
```bash
git switch main
git checkout rust-sim-kernel-port -- rust/ RUST_MIGRATION_PLAN.md
# Edit .gitignore: add `!rust/` under the whitelist block (after `!maker_dashboard/`),
# with the same comment as on the branch.
git add .gitignore rust/ RUST_MIGRATION_PLAN.md
git status   # CONFIRM: no rust/target, no *.pyd/.so, and NO 28MB rust/fixtures/prod_final_scores_memorial.npy
git commit -m "Bring sims_kernel Rust crate onto main (un-ignore rust/)"
git push origin main
```
**Acceptance:** `git ls-files rust/` ~28 files; the 28 MB `.npy` NOT listed; `import sims_kernel` works.
- Optional, low priority: re-add the inert `SIMS_DUMP_FIXTURE` dump hook to `main`'s
  `new_sim.py` (it currently lives only on the branch). Fixtures are already captured,
  so this is just for future re-capture. **Do NOT `git checkout <branch> -- new_sim.py`**
  — it would clobber the Rust integration + odds guard now on `main`.
- After this, `rust-sim-kernel-port` is redundant; keep as backup or delete once `main` is confirmed.

## Task 2 — speedup cutover — ✅ DONE 2026-06-09

**Done:** extended the Rust kernel (`run_pretournament` now returns `cat_means_r1..r4`,
each `(n,4)`) so `avg_expected_cat_sg` can be built from Rust; restructured `new_sim.py`
to compute Rust `final_scores` + means up front and run the Python draw + its validation
only behind `if _python_drew:` (set on `--use-python` or Rust-failure fallback). Win-prob +
aggregation always run. Verified: cargo + 5 gates green; `git diff -w` showed the draw/
validation reindent as pure whitespace; default `--sim-only` skips the Python draw and
builds `avg_expected_cat_sg` from Rust means; `--use-python` fallback still runs the Python
draw + validation; Rust-vs-Python `avg_expected_cat_sg` agree within ~0.05 SG (MC noise).
The original blocker analysis is kept below for reference.

**Why (original):** the Python draw ran every time (~21s) and Rust overwrote its `final_scores`,
so the ~8× speedup was not realized.

**BLOCKER found 2026-06-09 (verified in code):** you cannot simply skip the Python draw.
Besides `final_scores`, the Python draw produces a SECOND output the Rust kernel does NOT
return: **`avg_expected_cat_sg_{tourney}.csv`** (root + `permanent_data/`), built at
`new_sim.py` ~L1094–1120 from the per-category arrays `cats_r1..r4` / `sg_r1..r4`. It is a
real consumed output — read by `sg_diagnostic.py` (L55–61, "must exist") and shipped by
`push_dashboard_data.py` (L228). `run_pretournament` returns only `(final_scores, win_prob)`
— no per-category means — so skipping the Python draw silently drops that file. (The
validation prints — checks 1,2,4,5,6 — also use those arrays, but those are only diagnostics.)

**To unblock — extend the Rust kernel (`rust/src/cascade.rs` + `lib.rs`):** have
`run_pretournament` also accumulate per-player per-category means across sims and return
`r1m/r2m/r3m/r4m` (each `(n,4)`) + per-round `sg_rN` means; expose via `lib.rs` (extend the
return tuple or add a parallel entry point); rebuild the wheel + re-verify (add a stat-parity
check on the means). THEN do the cutover:
1. Default: compute Rust `final_scores` **and** category means up front; call the Python draw
   only on `--use-python` / Rust-failure.
2. Wrap the Python draw + the validation that uses internals so they run only when Python drew.
   **3 guards** (line refs as of 2026-06-09): draw `L596–888`; checks 1,2 `L928–946`; checks
   4,5,6 `L962–1042`. **Win-prob `L947–960` STAYS always-run** — it produces `win_counts`
   (→ `sim_win_probs`) used downstream. Build `avg_expected_cat_sg` from the Rust means.
**Acceptance:** default `--sim-only` ~2.5s; `avg_expected_cat_sg_{tourney}.csv` still written and
matches (within noise) the Python version; `--use-python` still works; rank_probs/win prob match.

**Until then — leave as-is (this is a fine production state).** Rust drives ALL betting output
(prices, emails, bets); the Python shadow cheaply produces `avg_expected_cat_sg` AND is exactly
the parity shadow the migration plan wanted before any cutover. The only cost is ~21s/run. Batch
the kernel extension with the round_sim phase (Task 3) rather than rushing it on live code.

---

## Task 3 (LATER — larger phase, likely beyond tomorrow morning) — `round_sim`

a. **Phase 5 (single-round score card).** Port `round_sim.py` `simulate_round_scores_catfirst`
   (~L1884, seed 789 RNG_CF; weather IS split via `WEATHER_CAT_SPLIT=[.35,.35,.15,.15]`;
   scores clipped `[round(avg)±12]`) + `build_score_card` (~L2383) + `build_round_score_probs`
   (~L2421). Add Rust `run_single_round` + `ref_single_round.py` + a stat-parity test.
b. **round_sim production verification.** Add a `SIMS_DUMP_FIXTURE` hook to `round_sim.py`
   (mirror `new_sim.py`) dumping `run_remaining_rounds` inputs; capture a live-round fixture
   (rbc_canada is mid-tournament Thu–Sun this week) → `verify_round_cascade_against_prod.py`.
c. **CI wheel build (plan §8) — REQUIRED before flipping round_sim** (not before Tasks 1/2):
   `nightly-round-sim.yml` and `reprice.yml` run round_sim on ubuntu with no local wheel.
   Build via maturin for `{ubuntu manylinux, windows}`, publish as a release artifact, add
   `pip install <wheel>` to both workflows. (`new_sim` runs manually, not in CI, so this
   was not needed for new_sim.)
d. **Flip round_sim to Rust default** (mirror the new_sim `--use-python` pattern at the
   `run_remaining_rounds` call site) + run for real.

---

## Gotchas (carry-over + new this session)

- **OneDrive truncates** source `.py` to 0 bytes. Recover: `git checkout HEAD -- <file>`.
  Keep `git config core.hooksPath .githooks` (pre-commit guard).
- `.gitignore:180` is `/*/` → ignores `rust/` on `main` until Task 1 adds `!rust/`.
- maturin: **always `maturin build --release`** (debug can ship a stale `.pyd`). No venv →
  `pip install --force-reinstall --no-deps <wheel>`.
- `final_scores` is **int32** on disk; Rust returns **i64** → new_sim casts to `final_scores.dtype`.
- `run_pretournament` (seed 456) signature: `(mu, std_course, eff_skew, l_corr, my_pred_base,
  r2_mu, r3_mu, r4_mu, wx_r1, wx_r2, r1_high/midh/midl/low[4], r2_lt6/6_30/30up[8],
  r3_lt6/6_20/30up[4], cut_line, use_10_shot, sims, seed)` → returns `(final_scores i64, win_prob)`.
  Exact coeff-vector order is in the dump hook / `verify_cascade_against_prod.py`.
- **`new_sim.py` diverges**: `main` has the Rust integration + odds guard (no dump hook);
  the branch has the dump hook (no integration). Never `checkout <branch> -- new_sim.py`.
- `_baseline_1547/` = backup of the 15:47 Python prod run (untracked). Safe to delete.
- `decimal_to_american` now returns NaN for `decimal_odds <= 1.0` (guard committed on `main`).

## "new_sim migration DONE" = 

Task 1 (crate on main) + Task 2 (shadow removed, ~2.5s default, `--use-python` fallback intact)
+ one clean full live run post-cutover. `round_sim` (Task 3) is a separate, larger phase.
