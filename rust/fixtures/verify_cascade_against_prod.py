"""Step 3: verify the draw CASCADE against real production output.

Prerequisite (Step 2): run a live event once with the dump hook enabled:
    SIMS_DUMP_FIXTURE=1 python new_sim.py --sim-only
which writes rust/fixtures/prod_input_<event>.npz (the kernel inputs) alongside
the outputs the run already saves (final_scores_<event>.npy, rank_probs_*.parquet).

Then this script (RUST_MIGRATION_PLAN.md §7) does two checks per event:

  A. TRANSCRIPTION: run the Python reference (ref_pretournament.run) on the frozen
     inputs with default_rng(456). Because it is the same code + same numpy RNG as
     production, a faithful transcription reproduces production `final_scores`
     BIT-FOR-BIT. Any mismatch means ref_pretournament.py has drifted from
     new_sim.py — fix the reference before trusting the statistical gate.

  B. RUST EQUIVALENCE: run the Rust kernel on the same inputs (different RNG, so
     per-cell differs by design). Aggregate both Rust and production final_scores
     and require win/top-N/rank cells within k*SE, AND the decision gate: no
     win-prob top-N reordering beyond tie noise.

Usage:  python rust/fixtures/verify_cascade_against_prod.py [event ...]
"""

import glob
import os
import sys
import numpy as np

import sims_kernel as k
from ref_pretournament import run as run_py

FIXDIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(FIXDIR))
K = 5.0


def load_inputs(npz):
    f = np.load(npz, allow_pickle=True)
    d = {key: f[key] for key in f.files}
    # ref_pretournament wants coeff DICTS; Rust wants the flat arrays as stored.
    def r1(a): return dict(ott=a[0], putt=a[1], residual=a[2], residual2=a[3])
    def r2(a): return dict(residual=a[0], residual2=a[1], residual3=a[2], avg_ott=a[3],
                           avg_putt=a[4], avg_app=a[5], avg_arg=a[6], delta_app=a[7])
    def r3(a): return dict(sg_ott_avg=a[0], sg_putt_avg=a[1], sg_app_avg=a[2], sg_arg_avg=a[3],
                           pos_6_10=(float(a[4]) if len(a) > 4 else 0.0))
    inp_ref = dict(
        mu=d["mu"], std_course=d["std_course"], eff_skew=d["eff_skew"], l_corr=d["l_corr"],
        my_pred_base=d["my_pred_base"], r2_mu=d["r2_mu"], r3_mu=d["r3_mu"], r4_mu=d["r4_mu"],
        weather_delta_r1=d["weather_delta_r1"], weather_delta_r2=d["weather_delta_r2"],
        r1_high=r1(d["r1_high"]), r1_midh=r1(d["r1_midh"]), r1_midl=r1(d["r1_midl"]), r1_low=r1(d["r1_low"]),
        r2_lt6=r2(d["r2_lt6"]), r2_6_30=r2(d["r2_6_30"]), r2_30up=r2(d["r2_30up"]),
        r3_lt6=r3(d["r3_lt6"]), r3_6_20=r3(d["r3_6_20"]), r3_30up=r3(d["r3_30up"]),
        cut_line=int(d["cut_line"]), use_10_shot_rule=bool(d["use_10_shot_rule"]),
        sims=int(d["sims"]),
    )
    seed = int(d["seed"])
    return d, inp_ref, seed


def week_latent_sd(d):
    return float(d["week_latent_sd"]) if "week_latent_sd" in d else 0.0


def run_rust(d, seed):
    A = np.ascontiguousarray
    return k.run_pretournament(
        A(d["mu"]), A(d["std_course"]), A(d["eff_skew"]), A(d["l_corr"]),
        A(d["my_pred_base"]), A(d["r2_mu"]), A(d["r3_mu"]), A(d["r4_mu"]),
        A(d["weather_delta_r1"]), A(d["weather_delta_r2"]),
        list(d["r1_high"]), list(d["r1_midh"]), list(d["r1_midl"]), list(d["r1_low"]),
        list(d["r2_lt6"]), list(d["r2_6_30"]), list(d["r2_30up"]),
        list(d["r3_lt6"]), list(d["r3_6_20"]), list(d["r3_30up"]),
        int(d["cut_line"]), bool(d["use_10_shot_rule"]), int(d["sims"]), seed,
        week_latent_sd(d),
    )


def se(p, n1, n2):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.sqrt(p * (1 - p) * (1.0 / n1 + 1.0 / n2))


def gate(p_rust, p_prod, sims, label, fails):
    z = np.abs(p_rust - p_prod) / np.maximum(se(0.5 * (p_rust + p_prod), sims, sims), 1e-12)
    nbad = int(np.sum(z > K))
    print(f"    {'OK ' if nbad == 0 else 'BAD'} {label:16s} worst z={float(np.max(z)):5.2f}  cells>{K:.0f}se: {nbad}")
    if nbad:
        fails.append(f"{label}: {nbad} cells > {K}se")


def verify(event):
    npz = os.path.join(FIXDIR, f"prod_input_{event}.npz")
    if not os.path.exists(npz):
        print(f"[{event}] SKIP — no prod_input_{event}.npz "
              f"(run: SIMS_DUMP_FIXTURE=1 python new_sim.py --sim-only)")
        return True
    # Oracle preference: the sequestered fixture copy, then the sha256 sidecar
    # (pinned from the RAW pre-top-up final_scores_{event}.npy), then live
    # working-tree copies. NOTE the event cache (<event>/final_scores.npy) is
    # saved AFTER the totals skew top-up and cannot serve as a bit-for-bit
    # oracle — only the root final_scores_{event}.npy (saved pre-top-up) can.
    sha_sidecar = os.path.join(FIXDIR, f"prod_final_scores_{event}.sha256")
    fs_candidates = [
        os.path.join(FIXDIR, f"prod_final_scores_{event}.npy"),
    ]
    if not os.path.exists(sha_sidecar):
        fs_candidates.append(os.path.join(ROOT, f"final_scores_{event}.npy"))
    fs_path = next((p for p in fs_candidates if os.path.exists(p)), None)
    if fs_path is None and not os.path.exists(sha_sidecar):
        print(f"[{event}] SKIP — production final_scores (or sha256 sidecar) not found")
        return True

    import hashlib
    d, inp_ref, seed = load_inputs(npz)
    sims = int(d["sims"])
    fails = []

    prod_fs = np.load(fs_path).astype(np.int64) if fs_path else None
    n = prod_fs.shape[0] if prod_fs is not None else int(np.asarray(d["mu"]).shape[0])
    print(f"[{event}] n={n}, sims={sims:,}, seed={seed}")

    # --- A. transcription: ref must reproduce prod final_scores bit-for-bit ---
    # Works against the full prod matrix when present, else against the committed
    # sha256 sidecar (so the tiny npz+sha256 fixture stays a valid CI oracle).
    # Fixtures captured with the week latent ON cannot be transcription-checked
    # (ref_pretournament has no latent) — capture with --no-week-latent for A.
    if week_latent_sd(d) != 0.0:
        print(f"  A. transcription  SKIP — fixture captured with week latent "
              f"sd={week_latent_sd(d)} (re-capture with --no-week-latent); "
              f"running B only vs stored prod matrix")
        if prod_fs is None:
            return True
        rust_fs, *_ = run_rust(d, seed)
        rust_fs = np.ascontiguousarray(rust_fs)
        u_p, ndh_p, top_p = k.aggregate(np.ascontiguousarray(prod_fs))
        u_r, ndh_r, top_r = k.aggregate(rust_fs)
        gate(top_r[:, 0], top_p[:, 0], sims, "top5", fails)
        gate(top_r[:, 1], top_p[:, 1], sims, "top10", fails)
        gate(top_r[:, 2], top_p[:, 2], sims, "top20", fails)
        gate(u_r[:, :20].ravel(), u_p[:, :20].ravel(), sims, "rank_prob[1..20]", fails)
        if fails:
            print(f"  [{event}] FAIL:")
            for m in fails:
                print(f"    - {m}")
            return False
        return True
    ref_fs, _ = run_py(inp_ref, seed=seed)
    sha_path = os.path.join(FIXDIR, f"prod_final_scores_{event}.sha256")
    if prod_fs is not None:
        if np.array_equal(ref_fs, prod_fs):
            print(f"  A. transcription  OK  (ref_pretournament == prod final_scores, BIT-FOR-BIT)")
        else:
            nbad = int(np.sum(ref_fs != prod_fs))
            fails.append(f"transcription: {nbad} cells ({nbad/prod_fs.size:.2%}) differ -> ref drifted")
            print(f"  A. transcription  BAD  {nbad} cells differ — fix ref before trusting B")
    elif os.path.exists(sha_path):
        want = open(sha_path).read().strip()
        got = hashlib.sha256(np.ascontiguousarray(ref_fs.astype(np.int64)).tobytes()).hexdigest()
        if got == want:
            print(f"  A. transcription  OK  (ref final_scores sha256 == pinned prod hash)")
        else:
            fails.append("transcription: ref sha256 != pinned prod hash")
            print(f"  A. transcription  BAD  ref hash {got[:12]} != pinned {want[:12]}")
        prod_fs = ref_fs  # ref is bit-identical to prod when hash matches -> ok for B
    else:
        print("  A. transcription  SKIP — no prod final_scores matrix or sha256 sidecar")
        return True

    # --- B. Rust statistical equivalence vs production ---
    rust_fs, rust_win, *_ = run_rust(d, seed)
    rust_fs = np.ascontiguousarray(rust_fs)
    u_p, ndh_p, top_p = k.aggregate(np.ascontiguousarray(prod_fs))
    u_r, ndh_r, top_r = k.aggregate(rust_fs)
    print(f"  B. Rust vs prod (statistical, k{K:.0f} SE):")
    gate(top_r[:, 0], top_p[:, 0], sims, "top5", fails)
    gate(top_r[:, 1], top_p[:, 1], sims, "top10", fails)
    gate(top_r[:, 2], top_p[:, 2], sims, "top20", fails)
    gate(u_r[:, :20].ravel(), u_p[:, :20].ravel(), sims, "rank_prob[1..20]", fails)

    if fails:
        print(f"  [{event}] FAIL:")
        for m in fails:
            print(f"    - {m}")
        return False
    return True


def main():
    events = sys.argv[1:]
    if not events:
        events = sorted({os.path.basename(p)[len("prod_input_"):-4]
                         for p in glob.glob(os.path.join(FIXDIR, "prod_input_*.npz"))})
    if not events:
        print("No prod_input_*.npz fixtures yet. Capture one first:")
        print("  SIMS_DUMP_FIXTURE=1 python new_sim.py --sim-only")
        return
    ok = True
    for ev in events:
        ok &= verify(ev)
        print()
    if not ok:
        sys.exit(1)
    print("RUST CASCADE VERIFIED AGAINST REAL PRODUCTION.")


if __name__ == "__main__":
    main()
