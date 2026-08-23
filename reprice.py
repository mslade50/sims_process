"""reprice.py — cache-free round-matchup repricer.

Triggered by the golf_scraping sentinel (repository_dispatch) whenever fresh odds
land. Instead of restoring the 60 MB sim cache and re-running round_sim's full
pricing pipeline (Rust kernel, Excel export, predictions, weather — all of which
have failed in CI), this:

    1. Loads the committed round-H2H fair table  (round_h2h_r{N}.parquet, ~100 KB)
    2. Fetches fresh round-matchup odds          (odds_loader, GitHub -> DataGolf)
    3. Prices every offered matchup from the table  (reprice_core.price_from_h2h)
    4. Computes edges + the combined/sharp filter
    5. Dedups against the Round Matchups sheet
    6. Delivers new sharp-book Telegram alerts, then stores those rows
       (non-alert rows may store independently)

The fair table is produced by the nightly sim (publish_sim_fairs.py
--round-h2h-only) and by live full round_sim runs, and is conclusive for H2H
pricing: (P(a<b), P(tie)) reconstructs round_sim.price_matchups EXACTLY.

Dependencies: pandas numpy requests gspread google-auth pyarrow python-dotenv.
NO Rust kernel, NO xlsxwriter, NO sim cache.

Usage:
    python reprice.py              # price + dedup + store + Telegram
    python reprice.py --dry-run    # price + print summary, no store/Telegram
"""

import argparse
import json
import os
import sys


def _setup_env():
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    return root


def _deliver_then_store_matchups(
    new_mu,
    seen_alert_keys,
    *,
    sim_round,
    tourney,
    event_id,
    dg_id_lookup,
    spreadsheet,
    rc_module,
    store_func,
    alert_func=None,
    health_check=None,
):
    """Enforce alert-before-storage for newly qualifying sharp-book rows.

    Rows that do not require a new Telegram alert (soft books, or a price move
    for an already-alerted pairing+side) remain independently storable.  If a
    required delivery fails, only that non-alert subset is stored and the
    exception propagates so GitHub Actions fails and retries the sharp rows.

    Returns ``(submitted_to_storage, telegram_rows_delivered)``.
    """
    if new_mu is None or new_mu.empty:
        print("  [reprice] No new matchup rows to store.")
        return 0, 0

    alert_rows, no_alert_rows = rc_module.partition_matchup_alert_rows(
        new_mu, seen_alert_keys=seen_alert_keys
    )

    def require_current_health():
        if health_check is not None:
            health_check()

    if alert_rows is not None and not alert_rows.empty:
        # Re-hash the bound fair bundle at the actual delivery boundary, not
        # only earlier in main() before Sheet reads and dedup work.  Keep this
        # outside the delivery exception handler: a failed health gate can never
        # fall through to the independently storable-row recovery path.
        require_current_health()
        try:
            deliver_alert = alert_func or rc_module.send_matchup_alert
            delivered = deliver_alert(
                alert_rows, sim_round, tourney, seen_alert_keys=None
            )
            if delivered != len(alert_rows):
                raise rc_module.TelegramDeliveryError(
                    f"Telegram accepted {delivered} of {len(alert_rows)} required alert rows"
                )
        except Exception:
            # These rows never depended on an alert.  Storing them cannot suppress
            # the sharp alert on retry because dedup_round_matchups only derives
            # seen_alert_keys from Telegram-eligible books.
            if no_alert_rows is not None and not no_alert_rows.empty:
                require_current_health()
                store_func(
                    no_alert_rows, sim_round, tourney, event_id,
                    dg_id_lookup=dg_id_lookup, spreadsheet=spreadsheet,
                )
                print(f"  [reprice] Stored {len(no_alert_rows)} non-alert row(s); "
                      "alert-required rows were not stored.")
            raise

    # A successful alert does not authorize storage from an artifact that was
    # replaced while Telegram was in flight.  Revalidate immediately before the
    # concrete Sheet/ledger mutation (also covers soft-only batches).
    require_current_health()
    store_func(
        new_mu, sim_round, tourney, event_id,
        dg_id_lookup=dg_id_lookup, spreadsheet=spreadsheet,
    )
    return len(new_mu), len(alert_rows) if alert_rows is not None else 0


def main():
    ap = argparse.ArgumentParser(description="Cache-free round-matchup repricer")
    ap.add_argument("--dry-run", action="store_true",
                    help="Price + print summary; skip dedup/store/Telegram")
    args = ap.parse_args()

    root = _setup_env()
    import pandas as pd
    import reprice_core as rc
    from sheet_config import load_config

    try:
        from sim_inputs import name_replacements as NAME_REPL
    except Exception:
        NAME_REPL = {}

    cfg = load_config()
    tourney = cfg["tourney"]
    event_id = cfg["event_id"]
    round_num = cfg["round_num"]
    sim_round = round_num + 1 if round_num < 4 else 4

    print(f"\n{'='*60}")
    print(f"  REPRICE (cache-free) — {tourney} R{sim_round}")
    print(f"{'='*60}")

    # ── 1. Load the committed round-H2H fair table ───────────────────────
    pq = os.path.join(root, f"round_h2h_r{sim_round}.parquet")
    mj = os.path.join(root, f"round_h2h_r{sim_round}_meta.json")
    hj = os.path.join(root, f"round_h2h_r{sim_round}_health.json")
    if not os.path.exists(pq) or not os.path.exists(mj) or not os.path.exists(hj):
        msg = (f"Reprice: no round-H2H fairs for R{sim_round} {tourney}. "
               f"The exact tape/meta health bundle is incomplete; run the sim first.")
        print(f"  {msg}")
        if not args.dry_run:
            rc.send_telegram(msg)
        return 1

    with open(mj) as f:
        meta = json.load(f)
    with open(hj) as f:
        health = json.load(f)
    if str(meta.get("round")) != str(sim_round) or meta.get("tourney") != tourney:
        msg = (f"Reprice: round-H2H artifact is R{meta.get('round')} "
               f"{meta.get('tourney')}, expected R{sim_round} {tourney}. "
               f"Run the sim first.")
        print(f"  {msg}")
        if not args.dry_run:
            rc.send_telegram(msg)
        return 1

    h2h_df = pd.read_parquet(pq)
    from sim_health_gate import (
        collect_overlay_provenance,
        configured_round_expected_avg,
        configured_round_scoring_baselines,
        require_bound_artifact,
        require_h2h_probability_table,
    )
    configured_avg = configured_round_expected_avg(cfg)
    configured_course_averages = configured_round_scoring_baselines(cfg)
    simulation_health = health.get("simulation_manifest") or {}
    health_model = simulation_health.get("model") or {}
    overlay = collect_overlay_provenance(
        tourney=tourney,
        event_id=event_id,
        dists_path=(health_model.get("shot_dispersion_overlay") or {}).get("distribution_file"),
        selected_model=health_model.get("selected", "category_first"),
    )

    def require_pricing_health():
        report = require_bound_artifact(
            health,
            kind="published_round_h2h",
            files={"h2h_parquet": pq, "h2h_meta": mj},
            tourney=tourney,
            event_id=event_id,
            sim_round=sim_round,
            configured_expected_avg=configured_avg,
            configured_course_averages=configured_course_averages,
            current_overlay=overlay,
        )
        require_h2h_probability_table(h2h_df, simulation_health)
        if meta.get("source_manifest_sha256") != report.facts.get("simulation_manifest_id"):
            from sim_health_gate import SimulationHealthError
            raise SimulationHealthError(
                "BLOCKED — H2H metadata references a different simulation manifest"
            )
        sim_event = simulation_health.get("event") or {}
        sim_source = simulation_health.get("source") or {}
        sim_shape = simulation_health.get("simulation") or {}
        sim_scoring = simulation_health.get("scoring") or {}
        semantic_pairs = (
            (meta.get("tourney"), str(meta.get("event_id")), str(meta.get("round"))),
            (sim_event.get("tourney"), str(sim_event.get("event_id")), str(sim_event.get("round"))),
        )
        if semantic_pairs[0] != semantic_pairs[1]:
            from sim_health_gate import SimulationHealthError
            raise SimulationHealthError("BLOCKED — H2H metadata event/round is not source-aligned")
        if (
            int(meta.get("num_players", -1)) != int(sim_shape.get("num_players", -2))
            or int(meta.get("num_sims", -1)) != int(sim_shape.get("num_sims", -2))
            or float(meta.get("expected_avg", float("nan")))
            != float(sim_scoring.get("expected_avg", float("nan")))
            or meta.get("sim_run_at") != sim_source.get("generated_at")
        ):
            from sim_health_gate import SimulationHealthError
            raise SimulationHealthError("BLOCKED — H2H metadata shape/scoring/timestamp is not source-aligned")
        return report

    # Diagnose bad artifacts before spending time/network on current odds.
    require_pricing_health()
    lookup, known = rc.build_h2h_lookup(h2h_df)
    pred_lookup = {k: float(v) for k, v in (meta.get("pred") or {}).items()}
    sample_lookup = {k: float(v) for k, v in (meta.get("sample") or {}).items()}
    wx_lookup = {k: float(v) for k, v in (meta.get("wx") or {}).items()}
    print(f"  Fairs: {len(h2h_df)} pairs, {len(known)} players "
          f"(sim @ {meta.get('generated_at', '?')})")

    # ── 2. Fetch fresh round-matchup odds ────────────────────────────────
    from odds_loader import load_matchup_odds
    matchup_df = load_matchup_odds("round_matchups", api_key=os.getenv("DATAGOLF_API_KEY"), round=sim_round)
    if matchup_df is None or matchup_df.empty:
        print("  No round-matchup odds available — nothing to price.")
        return 0
    print(f"  Loaded {len(matchup_df)} matchup lines")

    # ── 3-4. Price from the table + edges + combined/sharp filter ─────────
    matchup_df = rc.price_from_h2h(matchup_df, lookup, known, repl=NAME_REPL)
    matchup_df = rc.calculate_edges(matchup_df)
    combined, sharp = rc.build_matchup_outputs(
        matchup_df, sim_round, pred_lookup, sample_lookup, wx_lookup=wx_lookup
    )

    if args.dry_run:
        print(f"\n  [dry-run] {len(combined)} combined / {len(sharp)} sharp edges")
        if not sharp.empty:
            show = [c for c in ["bet_on", "Player 1", "Player 2", "Bookmaker",
                                "edge_on", "Fair_p1", "Fair_p2", "P1 Odds", "P2 Odds"]
                    if c in sharp.columns]
            print(sharp[show].head(25).to_string(index=False))
        return 0

    # ── 5. Dedup against Sheets + store only new rows ─────────────────────
    # Re-hash the exact parquet/meta immediately before the first side effect;
    # a stale/mixed file can never store or alert even if it changed mid-run.
    require_pricing_health()
    from sheets_storage import get_spreadsheet, store_round_matchups, load_dg_id_lookup
    spreadsheet = get_spreadsheet()
    new_mu, seen_alert_keys = rc.dedup_round_matchups(combined, spreadsheet, event_id, sim_round)

    # Keep the concrete mutation behind the immediately preceding health gate;
    # the delivery helper receives this adapter so its alert-before-store policy
    # remains dependency-injectable and offline-testable.
    def store_health_checked_round_matchups(*store_args, **store_kwargs):
        return store_round_matchups(*store_args, **store_kwargs)

    def send_health_checked_matchup_alert(*alert_args, **alert_kwargs):
        return rc.send_matchup_alert(*alert_args, **alert_kwargs)

    try:
        dg_id_lookup = load_dg_id_lookup(tourney, NAME_REPL)
    except Exception:
        dg_id_lookup = {}

    # ── 6. Required sharp alerts are accepted by Telegram before their rows
    # are stored. Re-hash once more after the Sheet reads so a long-running or
    # shared runner cannot swap the pricing bundle between dedup and delivery.
    # A delivery exception escapes main(), failing the workflow. ──
    require_pricing_health()
    n_new, n_alerted = _deliver_then_store_matchups(
        new_mu,
        seen_alert_keys,
        sim_round=sim_round,
        tourney=tourney,
        event_id=event_id,
        dg_id_lookup=dg_id_lookup,
        spreadsheet=spreadsheet,
        rc_module=rc,
        store_func=store_health_checked_round_matchups,
        alert_func=send_health_checked_matchup_alert,
        health_check=require_pricing_health,
    )

    alert_status = (
        f"Telegram delivered {n_alerted} new sharp edge(s)."
        if n_alerted
        else "No Telegram alert required."
    )
    print(f"\n  [reprice] Done. Submitted {n_new} new matchups to storage. "
          f"{alert_status}")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
