# `sims_kernel` — Rust Monte Carlo kernel

Statistical-equivalence port of the `new_sim.py` / `round_sim.py` numerical kernels.
See `../RUST_MIGRATION_PLAN.md` for the full plan, parity targets, and rollout phases.

**Status:** Phases 0–2 complete and validated. The full pre-tournament cascade
(`run_pretournament`) + RNG-free aggregation are ported and proven equivalent.
Not yet done: Phase 3 (wire into `new_sim.py --sim-only`, shadow dual-run, cutover)
and Phases 4–6 (`round_sim`).

## What's here

| File | Purpose |
|---|---|
| `src/ops.rs` | Load-bearing primitives: `round_ties_even`, `_apply_skew`, min-rank, `sum4`, single-sim draw. Mirrors specific `new_sim.py` lines. |
| `src/rng.rs` | `Pcg64` + `StandardNormal` deterministic stream (statistical target — not byte-parity with numpy). |
| `src/cascade.rs` | `run_pretournament`: the full R1→R4 seed-456 draw cascade (skill updates + bucket routing + cut + missed-cut penalty + win tiebreak). |
| `src/agg.rs` | RNG-free post-sim aggregation: rank_probs (`prob_u`/`prob_ndh`), top-N dead-heat, O(n²) h2h. rayon-parallel. |
| `src/lib.rs` | PyO3 module. Exposes `version`, `selftest`, the op primitives, `run_pretournament`, `aggregate`, `h2h`. |
| `src/bin/cli.rs` | Optional/debug-only CLI. **Not** a fail-open fallback. |
| `fixtures/capture_ops.py` | Freezes current-Python op outputs to `ops_fixture.npz` (the CI regression oracle). |
| `fixtures/ref_pretournament.py` | Faithful standalone Python copy of the cascade = executable spec of the frozen-input contract + statistical oracle. |
| `fixtures/test_ops_parity.py` | Op primitives vs frozen oracle — **array-exact**. |
| `fixtures/test_agg_parity.py` | Aggregation vs pandas — **integer-exact** (`prob_ndh`/h2h bit-exact). |
| `fixtures/test_cascade_parity.py` | Full cascade vs Python reference — **statistical** (within 5·SE). |
| `fixtures/verify_against_prod.py` | Aggregation vs **real production** parquets (uses on-disk `final_scores.npy`). DONE: cjcup+schwab bit-exact. |
| `fixtures/verify_cascade_against_prod.py` | Cascade vs **real production** — needs a fixture from a live dump-hooked run (Step 2). |

## Verifying against real production

The aggregation half is verified against genuine `new_sim.py` output today
(`verify_against_prod.py` — no code changes needed). The draw cascade needs
production's kernel *inputs*, captured by the env-gated dump hook in `new_sim.py`:

```bash
# On the next live event, capture a real input->output fixture:
SIMS_DUMP_FIXTURE=1 python new_sim.py --sim-only
# Then validate the Rust cascade (and the Python reference's faithfulness):
python rust/fixtures/verify_cascade_against_prod.py
```

## Build & test

```bash
# Rust unit tests (op primitives + RNG)
cargo test --lib

# Build the abi3 wheel (one wheel covers CI 3.10 + local 3.11)
maturin build
pip install --force-reinstall --no-deps target/wheels/sims_kernel-*.whl

# Regenerate the op oracle (only when the Python ops intentionally change)
python fixtures/capture_ops.py

# Validate Rust == Python (gates the wheel publish in CI). Run from repo root:
python rust/fixtures/test_ops_parity.py      # op primitives — array-exact
python rust/fixtures/test_agg_parity.py      # aggregation   — integer-exact
python rust/fixtures/test_cascade_parity.py  # full cascade  — statistical (5 SE)
```

No BLAS / native deps: `L_corr` is injected from Python, so manylinux/Windows
builds are trivial and reproducible.

## Parity invariants (do not regress)

- `np.rint` → `f64::round_ties_even()` (NOT `.round()`, which is half-away-from-zero).
- `rank(method='min')` ported exactly — routes R2/R3 coefficient buckets.
- `_apply_skew` divides by the variance-correction scale (matches Python op order,
  bit-identical — not a multiply-by-reciprocal).
- RNG draw loop stays **serial**; rayon only for RNG-free post-sim.
