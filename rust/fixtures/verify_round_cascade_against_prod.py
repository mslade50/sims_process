"""round_sim production verification (HANDOFF step 2).

Compares the Rust run_remaining_rounds kernel against round_sim.py's own legacy
Python cascade on IDENTICAL real-tournament inputs, captured via the
SIMS_DUMP_FIXTURE hook (fires inside simulate_remaining_rounds during any
in-tournament run, i.e. completed_round >= 1 — capture after R1 Thursday):

    SIMS_DUMP_FIXTURE=rust/fixtures/<event>_rN_fixture.npz \
        python round_sim.py --cli --sim-round 2 --dry-run --no-store

Then:  python rust/fixtures/verify_round_cascade_against_prod.py <fixture.npz>

Comparison is statistical (different RNG streams by design): win, rank, and
top-N dead-heat tables via k.aggregate_round, all cells within K standard errors.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import sims_kernel as k
import round_sim

K_SE = 5.0


def load_fixture(path):
    z = np.load(path, allow_pickle=True)
    known_strokes, known_cats = {}, {}
    for r in (1, 2, 3):
        if f"known_strokes_r{r}" in z:
            known_strokes[r] = z[f"known_strokes_r{r}"]
            known_cats[r] = z[f"known_cats_r{r}"]
    return z, known_strokes, known_cats


def run_path(z, known_strokes, known_cats, use_python):
    round_sim._USE_PYTHON = use_python
    names = list(z["player_names"])
    preds_base = dict(zip(names, z["my_pred_base"]))
    mu, std = z["mu"], z["std"]
    params = [(mu[i], std[i]) for i in range(len(names))]
    out = round_sim.simulate_remaining_rounds(
        completed_round=int(z["completed_round"]),
        player_names=names,
        known_strokes=known_strokes,
        known_categories=known_cats,
        model_preds=None,  # expected_r2 collapses to default_par; single-course events only
        player_cf_params=params,
        effective_skew=z["eff_skew"],
        L_corr=z["l_corr"],
        tournament_config={"default_par": float(z["default_par"])},
        player_preds_base=preds_base,
        num_sims=int(z["num_sims"]),
    )
    return np.ascontiguousarray(out[0]).astype(np.int64), np.ascontiguousarray(out[1])


def worst_z(p_a, p_b, sims, label, failures):
    p = np.clip(0.5 * (p_a + p_b), 1e-9, 1 - 1e-9)
    se = np.sqrt(p * (1 - p) * (2.0 / sims))
    z = np.abs(p_a - p_b) / np.maximum(se, 1e-12)
    bad = int(np.sum(z > K_SE))
    print(f"  {'OK ' if bad == 0 else 'BAD'} {label:14s} worst z={float(z.max()):5.2f}  cells>{K_SE:.0f}se: {bad}")
    if bad:
        failures.append(label)


def main():
    fixture = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name("the_open_r1_fixture.npz"))
    z, ks, kc = load_fixture(fixture)
    sims = int(z["num_sims"])
    print(f"fixture: {fixture} (completed_round={int(z['completed_round'])}, "
          f"{len(z['player_names'])} players, {sims:,} sims)")

    print("running Rust kernel path ...")
    fs_rust, mc_rust = run_path(z, ks, kc, use_python=False)
    print("running legacy Python cascade ...")
    fs_py, mc_py = run_path(z, ks, kc, use_python=True)

    praw_r, tdh_r, tnodh_r = k.aggregate_round(np.ascontiguousarray(fs_rust))
    praw_p, tdh_p, tnodh_p = k.aggregate_round(np.ascontiguousarray(fs_py))

    failures = []
    worst_z(np.asarray(praw_r), np.asarray(praw_p), sims, "rank_probs", failures)
    worst_z(np.asarray(tdh_r), np.asarray(tdh_p), sims, "top_dh", failures)
    worst_z(np.asarray(tnodh_r), np.asarray(tnodh_p), sims, "top_nodh", failures)
    worst_z(mc_rust.mean(axis=1), mc_py.mean(axis=1), sims, "made_cut", failures)

    if failures:
        raise SystemExit(f"PROD VERIFICATION FAILED: {failures}")
    print("\nROUND CASCADE PRODUCTION-VERIFIED: Rust == Python cascade on real inputs (all < 5 SE).")


if __name__ == "__main__":
    main()
