# Rust migration — handoff (as of 2026-06-05 EOD)

Start-here note for resuming the `new_sim`/`round_sim` → Rust port. Full plan in
`../RUST_MIGRATION_PLAN.md`; deep context in Claude memory `rust-port-sim-kernels`.

## TL;DR state

| Piece | Status |
|---|---|
| Scaffold + RNG + op primitives | ✅ done, array-exact vs numpy/pandas |
| `new_sim` cascade (`run_pretournament`) | ✅ ported, **production-verified** |
| post-sim aggregation (`aggregate`/`h2h`) | ✅ ported, **bit-exact vs real prod** |
| `round_sim` tournament cascade (`run_remaining_rounds`) | ✅ ported, stat-equiv vs reference |
| `round_sim` finish aggregation (`aggregate_round`) | ✅ ported (adds `top_nodh`, raw-rank `prob_u`) |
| `round_sim` single-round score card (seed 789) | ⛔ **NOT started** (Phase 5) |
| `round_sim` production verification | ⛔ NOT started |
| `new_sim` shadow-run wiring + cutover | ⛔ NOT started (Phase 3) |

**All green now:** 23 cargo tests + 7 Python gates (run them below).

## Resume checklist (do this first tomorrow)

```bash
# 1. rebuild + install the wheel (RELEASE — debug builds can ship a stale .pyd)
cd rust && cargo test --lib && maturin build --release
cd .. && pip install --force-reinstall --no-deps rust/target/wheels/sims_kernel-0.1.0-cp310-abi3-win_amd64.whl
python -c "import sims_kernel as k; print(k.version(), k.selftest())"

# 2. confirm everything still passes (run from repo root)
python rust/fixtures/test_ops_parity.py
python rust/fixtures/test_agg_parity.py
python rust/fixtures/test_cascade_parity.py
python rust/fixtures/verify_against_prod.py            # new_sim agg vs REAL prod
python rust/fixtures/verify_cascade_against_prod.py    # new_sim cascade vs REAL prod
python rust/fixtures/test_round_cascade_parity.py      # round_sim cascade vs ref
```

## What "verified" means per piece (don't overclaim)

- **new_sim**: verified against GENUINE production output. Aggregation bit-exact
  (cjcup+schwab). Cascade: my Python ref reproduces prod `final_scores` bit-for-bit
  (proves faithful copy + complete input contract); Rust is *statistically* equivalent
  (different RNG by design — NOT bit-identical, expected).
- **round_sim**: verified only against my Python reference (`ref_remaining_rounds.py`),
  statistically. NOT yet against real `round_sim.py` output (needs a dump-hook run).

## Next options (pick one)

**A. Finish `round_sim` Phase 5 — single-round score card (smaller, self-contained).**
- Port `round_sim.py:1884` `simulate_round_scores_catfirst` (seed 789, RNG_CF). KEY DIFF
  from the cascade: weather IS split here and ADDED per category
  (`cat_mu = mu + shift + wx_delta*WEATHER_CAT_SPLIT`); skill = `scores_rN - wx_delta`;
  scores clipped `[round(player_avg)±12]`. Returns `{player: int scores}`.
- Then `build_score_card` (L2383) + `build_round_score_probs` (L2421) aggregation.
- Add a Rust `run_single_round` + a `ref_single_round.py` + stat parity test (same pattern).

**B. `round_sim` production verification.**
- Add a `SIMS_DUMP_FIXTURE` hook to `round_sim.py` (mirror the one already in
  `new_sim.py` ~L585) dumping the `run_remaining_rounds` inputs. The current live event
  (memorial) is mid-tournament, so a real `completed_round≥1` fixture is capturable now.
- Then a `verify_round_cascade_against_prod.py` (mirror `verify_cascade_against_prod.py`).

**C. `new_sim` Phase 3 — shadow-run wiring (the path to prod).**
- Refactor `new_sim.py --sim-only` kernel block (L590–882 + aggregation L1013–1188) to
  optionally call `sims_kernel.run_pretournament` + `aggregate` + `h2h` behind a flag,
  **Python default**. Run both on live events for a few weeks, diff each run.
- Decision gate to add to the harness: **no edge-sign flips, no Kelly-stake-bucket flips**
  (not bit-identity). Then flip default → Rust, delete the Python kernel block.

Recommended order: B (cheap, closes the round_sim prod gap) → A → C.

## Gotchas / things that bit us

- **`.gitignore:180` is `/*/`** — ignores ALL top-level dirs, so `rust/` is invisible to
  git. `git add rust/` silently no-ops. To commit: add `!rust/` to root `.gitignore`.
  Nothing is committed yet; the only tracked change is the inert dump hook in `new_sim.py`.
- **Build trap**: `maturin build` (debug) sometimes finishes in <1s reusing a STALE `.pyd`
  → the wheel lacks new functions. Always `maturin build --release` and verify with
  `hasattr(k, 'run_remaining_rounds')` after reinstall.
- **No venv** — use `maturin build` + `pip install --force-reinstall`, NOT `maturin develop`.
- **Running new_sim/round_sim here needs network** (Google Sheet config + DataGolf +
  Open-Meteo) → use the Bash tool's `dangerouslyDisableSandbox`. `--sim-only` has no
  outward side effects but writes local files — back up + restore (see how the memorial
  fixture was captured; working tree was left untouched).
- **`final_scores` is int32 on disk** — cast to int64 for the Rust boundary; the sha256
  oracle is pinned on the int64 form.
- **prob_u inversion**: round_sim's `prob_u` = RAW min-rank = new_sim's `prob_ndh`.
  `aggregate_round` returns raw-rank as the first array accordingly.

## File map (rust/)

- `src/ops.rs` `rng.rs` — primitives (round_ties_even, skew, min-rank, sum4, Pcg64)
- `src/cascade.rs` — new_sim `run_pretournament` (+ shared `CoeffR1/R2/R3`)
- `src/round_cascade.rs` — round_sim `run_remaining_rounds`
- `src/agg.rs` — `aggregate` (new_sim) + `aggregate_round` (round_sim) + `h2h`
- `src/lib.rs` — PyO3 bindings
- `fixtures/ref_pretournament.py` `ref_remaining_rounds.py` — faithful Python oracles
- `fixtures/test_*_parity.py` `verify_*_against_prod.py` — gates
- `fixtures/prod_input_memorial.npz` + `.sha256` — committable new_sim prod oracle
  (28MB `prod_final_scores_memorial.npy` is gitignored/local-only)
