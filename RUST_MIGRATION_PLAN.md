# Rust Migration Plan: `new_sim.py` / `round_sim.py` Monte Carlo Kernels

> Plan to port the numerical Monte Carlo cores of `new_sim.py` (~4,120 lines) and
> `round_sim.py` (~4,419 lines) from numpy/scipy to Rust while preserving outputs.

## Implementation status (2026-06-05)

- **Phase 0 (scaffold) — DONE.** Crate at `rust/` (`sims_kernel`): PyO3 + `rust-numpy`,
  abi3-py310 wheel builds on Win via maturin 1.12.6 / Rust 1.94.1. Optional CLI bin builds.
  `import sims_kernel; sims_kernel.selftest()` → True from system Python.
- **Phase 1 (RNG + op primitives) — DONE & validated array-exact.** `src/ops.rs`
  (`round_ties_even`, `_apply_skew`, min-rank, `sum4`, single-sim `draw_one_sim`) + `src/rng.rs`
  (`Pcg64`+`StandardNormal`). 11 cargo unit tests pass. `fixtures/capture_ops.py` freezes the
  Python oracle to `ops_fixture.npz`; `fixtures/test_ops_parity.py` confirms all four primitives
  **bit-identical** to numpy/pandas (`_apply_skew` matched by dividing by the scale, not a
  reciprocal multiply). This is the load-bearing op layer for Phases 2/4.
- **Phase 2 (new_sim cascade + aggregation) — DONE & validated.** `src/cascade.rs`
  `run_pretournament` ports the full R1→R4 seed-456 cascade (draws, R1→R2/R2→R3/R3→R4 skill
  updates with bucket routing + undo-prior-adjustment, cut + 10-shot rule, missed-cut penalty,
  win-choice tiebreak). `src/agg.rs` ports the RNG-free post-sim layer (rank_probs
  `prob_u`/`prob_ndh`, top-N dead-heat, O(n²) h2h) with rayon. Exposed via PyO3
  (`run_pretournament`, `aggregate`, `h2h`). **Validation:** aggregation integer-exact vs pandas
  (`prob_ndh` + h2h bit-exact; fractional cells <7e-18); full cascade **statistically
  equivalent** to the Python reference (`rust/fixtures/ref_pretournament.py`) — all win/top-N/rank
  cells within 5·SE at n=60, sims=60k. **Perf** (n=147, 100k sims): Rust sim 2.48s + aggregate
  0.18s + h2h 0.11s vs Python-ref ~21s sim → ~8× on the cascade (more vs production's per-sim
  loops). §11 frozen-input contract resolved: `ref_pretournament.py` is the executable spec — the
  kernel is a pure `(inputs, seed) → (final_scores, win_prob)` mapping, all loading/Cholesky/IO
  stays Python.
- **Production verification, aggregation half — DONE (2026-06-05).** `rust/fixtures/verify_against_prod.py`
  loads real production `<event>/final_scores.npy` and diffs Rust `aggregate`/`h2h` against the
  actual `rank_probs_updated_<event>.parquet` / `h2h_matrix_<event>.parquet` from the same run.
  On **cjcup (n=147)** and **schwab (n=132)**, both at 100k sims: `prob_ndh` **bit-exact**, `prob_u`
  within 1.4e-17, and **all 19,377 h2h pairs bit-exact** (`prob_a` + `tie_pct`, zero diff). This is
  verification against genuine `new_sim.py` output, not the Python reference. The RNG-free half of
  the kernel is production-verified. Remaining gap: the draw cascade (needs production *inputs*,
  only saved via the Step-2 dump hook on a live run).
- **NEXT: Phase 3** — wire Rust behind a flag in `new_sim.py --sim-only` (refactor the kernel block
  to call `sims_kernel.run_pretournament` + `aggregate`/`h2h`, Python default), run the multi-week
  shadow dual-run on live events, capture production fixtures, then flip default + delete the
  Python kernel block. This is the invasive new_sim.py edit deferred from this session.

## Decisions locked (2026-06-04)

1. **Parity target = statistical equivalence**, not byte-identical. Validate that
   `rank_probs` / `top-N` / `h2h` / matchup edges agree within Monte Carlo standard
   error, and that **no edge sign or Kelly-stake bucket flips**. We do **not** hand-port
   numpy's ziggurat, Lemire `choice`, or `SeedSequence` to chase bitwise output.
2. **Architecture = PyO3 extension, single delivery.** No permanent fail-open-to-Python
   fallback. A CLI `bin` target is optional/debug-only.
3. **Python is a temporary scaffold.** Keep the Python kernel only through the shadow
   dual-run for each sim, then **delete the kernel block** once the acceptance gate holds.
   numpy/scipy/pandas stay in `requirements.txt` regardless (the ~85% glue still uses them).
4. **Fixtures before deletion.** Freeze the current Python kernel's outputs into `fixtures/`
   **before** removing it, so the CI regression oracle survives without live Python.

---

## 1. Executive summary

Both scripts are **~85% IO/pricing/email/Sheets/Telegram/git glue and ~15% numerical
kernel**. Only the kernel ports to Rust; the glue stays Python untouched.

The kernel's only RNG-dependent product is an **integer** matrix (`final_scores`, e.g.
`(147, 100000)` int). Every downstream file (`rank_probs`, `h2h_matrix`,
`top_finish_probs`) is an exact rational function of integer counts ÷ `SIMULATIONS`. Scores
are quantized via `np.rint(PAR − sg).astype(int)`, so float drift below ~0.5 stroke is
invisible — it only matters when a score sits within ~1e-13 of an exact `*.5` SG boundary.

Because of that quantization, **statistical equivalence is the correct target**: the model's
real uncertainty (odds, weather forecasts, skill preds) dwarfs Monte Carlo noise, and chasing
byte-identical output would cost ~3–5 extra days of hand-porting numpy internals for no
betting-relevant benefit. Two cheap, **mandatory** non-negotiables prevent *structural*
divergence (not bit-drift): `round_ties_even` for `np.rint`, and exact `rank(method='min')`.

---

## 2. What these scripts actually are

The kernel is already physically isolated by a production CLI split: `--sim-only` writes a
cache and exits; `--price-only` / `--reprice` load that cache and skip the kernel entirely.

| File | Total lines | Kernel (port target) | Kernel % | Glue (stays Python) |
|---|---|---|---|---|
| `new_sim.py` | 4,121 | ~700–800 (within L1–1196) | ~15–20% | L1197–4120 |
| `round_sim.py` | 4,419 | ~700–800 | ~12–18% | ~3,600 lines |

**Kernel (port target):** per-player `RNG.standard_normal((SIMS,4))` draw loops, the fixed
4×4 lower-Cholesky transform `Z @ L_corr.T`, Cornish-Fisher skew, scale/clip/sum →
`np.rint(PAR−sg).astype(int)`, the R1→R4 skill-update cascade with position-bucket coefficient
routing, cut logic + 10-shot rule, and post-sim aggregation (`compute_finish_probabilities`,
rank/top-N, dead-heat credit, the O(n²) head-to-head loop).

**Glue (do NOT port):** everything after new_sim L1197 / round_sim ~L1755 — live HTTP to
DataGolf/Kalshi/NoVig/scraped odds, Kelly/edge math, Gmail SMTP, gspread Sheets dual-write +
`bet_ledger.parquet`, Telegram, xlsxwriter, the git push that triggers Render. IO-bound,
non-deterministic (depends on wall-clock market state), zero heavy numerics. Its only kernel
touch is `np.sum(final_scores[i] < final_scores[j])`, which just requires `final_scores` stay
row-aligned with `player_names`.

---

## 3. The parity problem (and why statistical, not bitwise)

Pinned environment: **numpy 1.25.0, scipy 1.11.1, pandas 2.3.3**, Python 3.11 local / 3.10 CI.
All three generators (`default_rng(456)` in new_sim; `default_rng(42)` + `default_rng(789)` in
round_sim) are **PCG64** (not PCG64DXSM).

**Why byte-identical was rejected.** It would require, with zero partial credit (a single
desynced uint64 diverges everything):
- Hand-porting numpy's `SeedSequence` (hash mixer, `0x9E3779B9`) — `rand_pcg::Pcg64` is the
  right family but seeds differently.
- Hand-porting numpy's 256-region ziggurat `standard_normal` tables + exact uint64-consumption
  pattern — `rand_distr` uses a different ziggurat and will desync.
- Porting Lemire-bounded `choice` for the playoff tiebreaks.
- Matching LAPACK Cholesky + BLAS reduction order bit-for-bit.

None of that buys a better betting model. **We skip all four.**

**Mandatory invariants under the statistical target** (these prevent structural divergence
that flips betting decisions, not mere last-bit drift):

- **`round_ties_even`.** Every `np.rint(par − sg)` → Rust `f64::round_ties_even()` (Rust's
  `.round()` is half-away-from-zero, which would systematically shift scores).
- **Exact `rank(method='min')`.** It routes which R2/R3 skill-update **coefficient bucket**
  (`<6` / `6–30` / `>30`) applies — a tie mismatch silently changes the sim math.
- **Inject `L_corr` from Python.** Don't port the 4×4 Cholesky; pass it in as fixed `f64[4][4]`
  (deletes LAPACK/BLAS parity risk and the BLAS build dependency). Keep the ridge fallback
  (`R = 0.95R + 0.05I` on `LinAlgError`) in Python too.
- **Serial draw loop.** Single shared RNG stream — parallelizing it breaks determinism. Use
  `rayon` only for RNG-free post-sim (h2h, aggregation).

**The cascade risk this guards against:** a `*.5` stroke flip → integer score → `rank('min')`
→ position bucket → which coefficients apply in R2/R3/R4 → later draws → cut → top-N/h2h. This
is why `round_ties_even` + exact min-rank are required even though we're not chasing bitwise.

**Out of scope:** round_sim L1876's legacy `np.random.normal` (`--legacy` only) is the
unseeded global MT19937 — non-reproducible run-to-run *even in Python today*. Leave it Python.

---

## 4. Architecture — PyO3 single delivery

**One Rust crate compiled as a PyO3 extension module** (`maturin` + `rust-numpy`). During
validation it lets you call the Python and Rust kernels in one process and diff the returned
arrays directly — the tightest possible parity loop. A thin CLI `bin` target is optional for
standalone debugging only; there is **no permanent fail-open fallback** (Python gets deleted
once Rust is proven, so post-cutover a broken wheel fails the run rather than reverting — an
accepted consequence of the temporary-scaffold decision).

**Kernel boundary** — a pure function of `(fixed inputs, seed)`. `L_corr` is injected;
output player-row order **must** equal input `player_names` order (the glue's name→row-index
map depends on it).

```
run_pretournament(                          # new_sim, seed 456
    player_params:(n,4) f64,  std_course:(n,4) f64,  effective_skew:(n,4) f64,
    l_corr:(4,4) f64,         my_pred_base:(n,) f64,
    weather_delta_r1/r2:(n,) f64,
    coeffs_r1{high,midh,midl,low}, coeffs_r2{base,_6_30,_30_up}, coeffs_r3{base,_mid,_high},
    par, cut_line, use_10_shot_rule, simulations, seed=456,
) -> { final_scores:(n,sims) i64,  rank_probs[idx,rank,prob_u,prob_ndh],
       top_finish[idx,top5,top10,top20],  sim_win_prob[idx,p],  h2h[a,b,prob_a,tie_pct] }
```

The live-round entry (`run_remaining_rounds`, seed 42) is the same shape plus
`completed_round`, `known_strokes`, `known_categories`, and NaN-aware `course_score_adj`. Note
the tournament `_catfirst_draw` does **not** split weather (skill shift `/4` evenly); the
single-round sim (seed 789) **does** split via `WEATHER_CAT_SPLIT = [.35, .35, .15, .15]`.

**Stays Python:** all config/prediction loading, `name_replacements` joins, weather/tee-time
API, the 4×4 Cholesky + ridge fallback, and **all file IO** — return arrays to Python and let
pandas/pyarrow write every `.npy`/`.parquet`/`.csv`. This deletes the entire arrow/polars
schema-parity surface, preserves the `prob_u`/`prob_ndh` naming (and the new_sim/round_sim
inversion), and leaves the bet-ledger dual-write untouched.

---

## 5. Rust crate mapping

| Python construct | Rust approach | Parity notes |
|---|---|---|
| `default_rng(seed)` PCG64 | `rand_pcg::Pcg64` (XSL-RR, **not** `Pcg64Mcg`) | Statistical target: any good stream is fine; numpy `SeedSequence` not replicated |
| `standard_normal((SIMS,4))` | `rand_distr::StandardNormal` | Different ziggurat than numpy — fine under statistical target |
| `choice(tied)` tiebreak | `rand` `gen_range` | Statistical; no Lemire port |
| `np.ndarray` / `.sum(axis=1)` / `.clip` | `ndarray` (`Array2<f64>`, `sum_axis`, `clamp`) | Replicate skew→scale→clip→sum op order |
| `Z @ L_corr.T` | hand-written 4-wide loop | No BLAS |
| `cholesky(R)` + ridge | **do not port** — inject `L_corr` from Python | Deletes LAPACK/BLAS parity + build dep |
| Cornish-Fisher skew | plain `f64` | Match `|γ|<0.01` / `<0.2` branch thresholds |
| `np.rint(par-sg).astype(int)` | `f64::round_ties_even()` then `as i64` (Rust ≥1.77) | **MANDATORY** — Rust `.round()` is wrong |
| `Series.rank(method='min')` | hand-coded competition ranking | **MANDATORY / load-bearing** — routes coefficient buckets; unit-test vs pandas |
| `value_counts` / dead-heat | `HashMap` counting + f64 division | Preserve `prob_u` (dead-heat) vs `prob_ndh` (raw min-rank) + naming inversion |
| O(n²) H2H loop | `rayon` `par_iter` over pairs | RNG-free, integer-exact — biggest safe speedup |
| `.npy` / `.parquet` writes | **keep in Python** (`rust-numpy` returns arrays) | Eliminates arrow schema parity; preserves ledger dual-write |
| Py↔Rust boundary | **PyO3 + `rust-numpy` + `maturin`**, `abi3-py310` | One stable-ABI wheel per platform covers CI 3.10 + local 3.11 |
| draw-loop parallelism | **serial** (rayon only for RNG-free post-sim) | Parallel draws break single-stream order |

---

## 6. Component-by-component port plan

### 6.1 `new_sim.py` kernel — port FIRST (single seed-456 stream)

1. **RNG + op primitives** — `Pcg64` + `StandardNormal`; `round_ties_even`; min-rank;
   pairwise n=4 sum. Unit-test the ops before anything else.
2. **Param setup** (~L201–233, ~L509–572): `_apply_skew`, `_cf_calibration_multiplier`,
   re-center `shift = (my_pred − cat_sum)/4`, `std_course`, `effective_skew` blend — much can
   be precomputed in Python and passed in.
3. **R1 draw loop** (~L594–605).
4. **R1→R2 skill update** (~L627–655) incl. residual caps (`resid<0 & adj>0.2 → 0.2`; hard
   cap `0.5`).
5. **R2 draw + cut logic** (~L667–701): `np.sort(sc)[CUT_LINE-1]`, optional 10-shot rule.
6. **R2→R3, R3→R4 bucketed updates** (~L707–854) — **highest cascade-risk slice; unit-test
   bucket assignment vs pandas.**
7. **R3/R4 draws** + missed-cut 200-stroke penalty (~L785–882).
8. **Aggregation** (~L1013–1188): min-rank, `prob_u`/`prob_ndh`, top-N, rayon H2H.
9. **Win column** (~L908–912): per-sim argmin + tiebreak — statistical, no `choice` port.

### 6.2 `round_sim.py` kernel — port SECOND (dual streams 42/789)

1. **`simulate_remaining_rounds`** (~L508–793) on seed 42 with the R3/R4 **undo-prior-
   adjustment** invariant: subtract `tot_sg_adj_r2` / `tot_resid_adj_r2` **exactly**; never
   wildcard `*_adj` (it catches `sg_total_adj`, which is raw data, not an adjustment). Strokes
   clipped `[par−12, par+12]`.
2. **Known-round seeding** (~L417–499): tile actual strokes/categories for completed rounds vs
   simulate remaining; match pandas-NaN propagation for `course_score_adj`.
3. **`compute_finish_probabilities`** (~L796–852) incl. the L810 playoff tiebreak on the same
   seed-42 stream (all draws, *then* per-sim choice) — statistical.
4. **Single-round `simulate_round_scores_catfirst`** on seed 789 (~L1884–1949): weather **is**
   split here; feeds score-card pricing.
5. **`build_round_score_probs` / `build_score_card`** (~L2383–2469).
6. **Skip** the `--legacy` MT19937 path (L1876).

---

## 7. Verification & golden-test strategy

**Fixture reality (confirmed on disk):** root `final_scores_*.npy` / `rank_probs_*.parquet` /
`h2h_*.parquet` are **untracked, regenerated weekly** — don't rely on them as a long-term
oracle. The git-tracked snapshots in `dashboard_data/` and `permanent_data/historical_dists/`
came from older code/config — use as a Python-stability regression baseline, not the cutover
oracle.

**Capture-then-validate (the load-bearing rule):**
1. Freeze a representative input snapshot per sim.
2. Run the **current Python kernel** on it and capture outputs into `fixtures/`.
3. Validate Rust against those frozen fixtures.
4. **Only after fixtures are captured and the gate holds** do you delete the Python kernel
   (§9). The frozen fixtures remain the permanent CI regression oracle without live Python.

| Layer | What | Gate |
|---|---|---|
| Op unit | `round_ties_even` on `*.5`; `rank('min')` on tie-heavy fixtures; skew thresholds; pairwise n=4 sum | Array-exact |
| Bucket routing | R2/R3 bucket assignment vs pandas on tie-heavy leaderboards | Array-exact (load-bearing) |
| Probability | per-cell `|p_rust − p_py| < k·se`, `se = √(p(1−p)/N)`, `k=5` (Bonferroni); h2h at N=100k, p≈0.5 → tol ≈0.008 | Statistical |
| Decision | **no edge-sign flips, no Kelly-stake-bucket flips** across the fixture set | Hard gate |
| Seed-sweep | 20 seeds; compare per-cell distributions across impls | Statistical |

**Shadow dual-run before cutover:** run both kernels on the same frozen input every live event
for a multi-week window, diffing outputs each run; Python stays default, Rust opt-in (flag)
until the decision gate holds. **CI gating:** a golden-fixture parity job gates the wheel
publish; pin Rust ≥1.77 (for `round_ties_even`) and IEEE-strict floats (no fast-math).

---

## 8. Build, CI & deployment

- **Toolchain:** `maturin` + PyO3 + `rust-numpy`, `abi3-py310` → one stable-ABI wheel per
  platform covering CI 3.10 + local 3.11 (matrix = `{windows, linux}`). Fallback to per-minor
  if an `abi3` API gap appears.
- **No BLAS:** `L_corr` injected + 4×4 work hand-coded ⇒ no native non-Rust deps ⇒ trivial
  manylinux builds, no `apt-get` BLAS in `nightly-round-sim.yml` / `reprice.yml`.
- **CI distribution:** publish the wheel as a GitHub release artifact; the two workflows add a
  `pip install <wheel>` step (ubuntu manylinux for CI, Windows wheel for local).
- **OneDrive quirk (mild plus):** a compiled `.pyd`/`.so` is immune to the 0-byte truncation
  that has hit `new_sim.py`. Keep the `.githooks/pre-commit` guard intact.
- **Cache-path trap:** kernel output paths must exactly match what `--price-only` loads
  (`./{tourney}/final_scores.npy`) and the `actions/cache` path-identity invariant.

---

## 9. Phased rollout & milestones

Effort is rough person-weeks for one engineer, at the statistical target.

| Phase | Deliverable | Effort |
|---|---|---|
| **0. Scaffolding** | Cargo crate (lib, optional bin), maturin build, PyO3 hello-world wheel on Win+Linux CI, fixture-capture script freezing current-Python outputs | 0.5–1 pw |
| **1. RNG + op primitives** | `Pcg64` + `StandardNormal`; `round_ties_even`; min-rank; pairwise sum — unit-tested | 1–1.5 pw |
| **2. new_sim kernel** | `run_pretournament` returning final_scores + rank/top-N/h2h; `--sim-only` calls Rust behind flag; rayon H2H | 1.5–2.5 pw |
| **3. new_sim shadow + cutover + decommission** | Multi-week dual-run; capture fixtures; flip default to Rust; **delete the Python new_sim kernel block** | 0.5 pw + shadow weeks |
| **4. round_sim kernel** | `run_remaining_rounds` (seed 42, cascade, undo-adjustment, known-round, multi-course NaN) + finish probs | 2–3 pw |
| **5. round_sim single-round sim** | seed-789 sim (weather split) + score-card aggregation | 0.5–1 pw |
| **6. round_sim shadow + cutover + decommission** | Dual-run; capture fixtures; flip default; **delete the Python round_sim kernel block** | 0.5 pw + shadow weeks |

**Per-sim lifecycle:** port → shadow dual-run → capture fixtures → flip default to Rust →
delete that sim's Python kernel. `new_sim`'s kernel is removed (Phase 3) before `round_sim`'s
port (Phase 4) begins.

**Smallest valuable first slice:** Phase 2 — `new_sim` only (single stream, no `choice`
dependency for the saved parquets). Proves the RNG/op layer, the build/CI machinery, and the
H2H/draw speedup on the simpler of the two sims.

---

## 10. Risks & mitigations

| Risk | Sev | Mitigation |
|---|---|---|
| `np.rint` banker's rounding vs Rust half-away → off-by-one cascade | High (cheap) | `f64::round_ties_even()` everywhere |
| `rank('min')` tie semantics route coefficient buckets; silent math change | High | Hand-code competition ranking; unit-test buckets vs pandas |
| Deleting Python kernel before fixtures captured → no regression oracle | High | **Capture `fixtures/` first** (§7); gate deletion on captured fixtures + held shadow gate |
| LAPACK Cholesky last-ULP drift through `z²` skew | Med | Inject `L_corr` from Python; never port |
| FP reduction order flips a stroke at `*.5` | Med | Replicate pairwise grouping `(c0+c1)+(c2+c3)` + explicit 4×4 matmul; IEEE-strict |
| round_sim undo-adjustment fragility; wildcard `*_adj` catches `sg_total_adj` | Med | Explicitly enumerate columns; golden-test `Post = Pre + total_adjustment` |
| Parallel draw loop breaks single-stream determinism | Med | Draws **serial**; rayon only post-sim |
| No fallback post-cutover — broken wheel fails the run | Med (accepted) | Pure-Rust (no BLAS) ⇒ trivial reproducible builds; keep wheel build green in CI; tagged release artifacts allow pinning a known-good wheel |
| Cache-path mismatch breaks `--price-only` / reprice | Med | Match exact paths + actions/cache identity |

---

## 11. Open decisions still outstanding

- **Port post-sim aggregations (rank_probs/top-N/h2h) to Rust too, or leave in pandas?**
  Porting the O(n²) h2h + 100k-sim groupbys is the biggest non-draw speedup. *Leaning: port.*
- **Input-bundle serialization** (if/when the optional CLI path is exercised): `.npz` / Arrow
  IPC are safer than parquet for lossless float round-trip of `L_corr` / `std_course`.
- **Wheel distribution mechanism** for the two workflows: GitHub release artifact + `pip
  install`, private index, or `maturin develop` in a CI build step.
- **Measured current wall-clock at 100k sims** (the h2h O(n²) + per-sim finish-prob loops) —
  sizes the payoff; worth confirming before committing to Phase 2+.
- **Frozen input contract:** confirm nothing currently computed *inside* the kernel (e.g.
  `course_score_adj` NaN handling, multi-course routing) must be hoisted to Python so the Rust
  function is a pure `(inputs, seed) → outputs` mapping. Prerequisite for Phase 1.
  **RESOLVED 2026-06-05** — the `SIMS_DUMP_FIXTURE` hook + memorial verification proved it: feeding
  `ref_pretournament` only the frozen npz reproduces production `final_scores` bit-for-bit, so the
  kernel uses nothing not in the captured input bundle.

---

## Production verification log (2026-06-05)

Both halves of the `new_sim` kernel are verified against **genuine `new_sim.py` output** (not just
the Python reference):

- **Aggregation half — bit-exact.** `rust/fixtures/verify_against_prod.py` on cjcup (147) + schwab
  (132): `prob_ndh` + all 19,377 h2h pairs bit-exact; `prob_u` within 1.4e-17.
- **Draw cascade — verified.** Captured a real memorial fixture via the dump hook
  (`SIMS_DUMP_FIXTURE=1 python new_sim.py --sim-only`, run sequestered, working tree restored).
  `verify_cascade_against_prod.py`: (A) `ref_pretournament` == prod `final_scores` **bit-for-bit**
  (72×100k); (B) Rust cascade statistically equivalent to prod (all top-N/rank cells <5·SE).
  Committable oracle = `prod_input_memorial.npz` (18KB) + `prod_final_scores_memorial.sha256`; the
  28MB matrix is gitignored/local-only.

**Remaining before cutover:** Phase 3 shadow-run wiring (call Rust behind a flag in `--sim-only`,
Python default, multi-week dual-run, decision gate on no edge-sign/Kelly-bucket flips), then flip +
delete the Python kernel.

- **Phase 4 (round_sim tournament cascade) — DONE & validated (2026-06-05).** `src/round_cascade.rs`
  `run_remaining_rounds` ports `simulate_remaining_rounds` (seed 42): known-round seeding (tile actual
  strokes/cats for rounds ≤ completed_round), stroke clip `[expected±12]`, per-player multi-course R2
  expected, no weather split, no `r_mu` offsets, R1 resid from rounded strokes, R3/R4 undo-adjustment.
  `aggregate_round` adds the `top_N_nodh` (no-dead-heat) finish variant and returns raw min-rank as
  `prob_u` (the round_sim naming inversion). Validated **statistically equivalent** to
  `rust/fixtures/ref_remaining_rounds.py` across completed_round 0/1/2 (all win/top-N/rank/made-cut
  cells <5·SE at n=60, 60k). 23 cargo tests pass.
- **Phase 5 (round_sim single-round score card) — NOT STARTED.** Seed-789 `simulate_round_scores_catfirst`
  (weather IS split here via WEATHER_CAT_SPLIT) + `build_score_card`/`build_round_score_probs`.
- **round_sim production verification — NOT STARTED.** Add the analogous `SIMS_DUMP_FIXTURE` hook to
  `round_sim.py` and validate against a live round fixture (as done for new_sim/memorial).
