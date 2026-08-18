# `sims_kernel` — Rust Monte Carlo kernel

> ## ⚠ REBUILD REQUIRED — kernel 0.3.0 (2026-08-18, week-level latent)
>
> `run_pretournament` grew a trailing `week_latent_sd` argument (the shared
> week-level form draw; see CLAUDE.md "Week-level form latent"). **`git pull`
> does NOT update your installed kernel** — until you rebuild, every
> `new_sim.py` run on this machine prints
> `[rust] WARNING: Rust kernel unavailable/failed (TypeError...)` and falls
> back to the Python cascade. The fallback DOES include the latent, so fairs
> are still correct — it's just ~20s slower and the warning is noise.
>
> **Rebuild (machine with cargo):**
> ```bash
> cd rust
> cargo test --release                                  # expect 27 green
> cargo build --release --features pyo3/extension-module
> python -c "import sims_kernel, pathlib; print(pathlib.Path(sims_kernel.__file__).parent)"
> # back up, then overwrite <that dir>/sims_kernel.pyd with:
> #   rust/target/release/sims_kernel.dll  (rename to sims_kernel.pyd)
> python -c "import sims_kernel; print(sims_kernel.version(), sims_kernel.selftest())"
> # expect: 0.3.0 True
> ```
>
> **No cargo on the machine?** The `.pyd` is an abi3 Windows-x64 binary —
> copy the freshly built `sims_kernel.pyd` from a machine that has rebuilt
> (e.g. the main dev box) straight over your site-packages copy, then run the
> version/selftest check above.
>
> **Verify after installing:** `python rust/fixtures/verify_cascade_against_prod.py bmw`
> (uses the committed BMW fixture + sha sidecar; expect transcription OK +
> all four statistical gates OK). Note for future fixture captures: run with
> `--no-week-latent` and pin the sha from the ROOT `final_scores_{t}.npy`
> (the event-dir cache copy is saved AFTER the totals skew top-up and is not
> a valid bit-for-bit oracle).

Statistical-equivalence port of the `new_sim.py` / `round_sim.py` numerical kernels.
See `../RUST_MIGRATION_PLAN.md` for the full plan, parity targets, and rollout phases.

**Status:** Complete and in production. Both `new_sim` and `round_sim` run the Rust
kernels by default. `new_sim` is fully cut over (Python draw removed); `round_sim`
keeps its Python draws as a `--use-python` fallback pending its first live-round
test. Kernels: `run_pretournament`, `run_remaining_rounds`, `run_single_round`,
`aggregate`/`aggregate_round`, `h2h`. 25 cargo tests + 6 Python parity gates green.

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

## Install on another machine

The kernel ships as an `abi3-py310` wheel — one wheel works on any CPython ≥ 3.10.

**Windows (offline, no Rust toolchain):** a prebuilt wheel is committed at
`rust/prebuilt/`. After a `git pull` / OneDrive sync:

```bash
pip install --force-reinstall --no-deps rust/prebuilt/sims_kernel-0.1.0-cp310-abi3-win_amd64.whl
python -c "import sims_kernel as k; print(k.version(), k.selftest())"
```

**Linux / CI / any platform:** the `build-wheels.yml` GitHub Action builds the
manylinux + Windows wheels on every `rust/` change and attaches them to a rolling
`wheels-latest` prerelease:

```bash
pip install --force-reinstall --no-deps \
  https://github.com/mslade50/sims_process/releases/download/wheels-latest/<wheel-name>
```

**From source (needs Rust):** `maturin build --release -m rust/Cargo.toml`, then
install from `rust/target/wheels/`.

If `sims_kernel` is unimportable, `new_sim.py` (cutover) fails the sim; `round_sim.py`
falls back to the Python draws with a warning. Refresh the committed Windows wheel
whenever the kernel source changes (or pull from the `wheels-latest` release).

## Parity invariants (do not regress)

- `np.rint` → `f64::round_ties_even()` (NOT `.round()`, which is half-away-from-zero).
- `rank(method='min')` ported exactly — routes R2/R3 coefficient buckets.
- `_apply_skew` divides by the variance-correction scale (matches Python op order,
  bit-identical — not a multiply-by-reciprocal).
- RNG draw loop stays **serial**; rayon only for RNG-free post-sim.
