"""Read-only recovery inputs for small, event-validated locked-field additions."""

from pathlib import Path
import os
import sqlite3

import numpy as np
import pandas as pd

from category_distribution_guard import require_complete_category_distributions


def historical_database_path():
    """Reuse existing snapshots; never download, prune, or mutate history."""
    override = os.getenv("DG_HISTORICAL_DB")
    if override:
        return Path(override)
    homes = [Path.home()]
    if os.name == "nt":
        homes += list(Path("C:/Users").glob("*"))
    snapshots = []
    for home in homes:
        cache = home / "AppData/Local/etr-golf/cache/dg_historical"
        snapshots += [p for p in cache.glob("*.db") if p.with_suffix(".sha256").is_file()]
    if snapshots:
        return max(snapshots, key=lambda p: p.name)
    for home in homes:
        candidate = home / "OneDrive/dg_historical.db"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("No historical SG snapshot available for late-player EMA20")


def replacement_ema20(player, cutoff, *, db_path=None, aliases=None):
    """EMA with span=20 over chronological, pre-event adjusted total SG rounds."""
    cutoff = pd.Timestamp(cutoff).date().isoformat()
    canonical = str(player).strip().lower()
    aliases = aliases or {}
    names = {canonical} | {
        str(name).strip().lower() for name, target in aliases.items()
        if str(target).strip().lower() == canonical
    }
    path = Path(db_path) if db_path is not None else historical_database_path()
    placeholders = ",".join("?" for _ in names)
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        history = pd.read_sql_query(
            f"SELECT tour, year, event_id, round_num, round_date, sg_total_adj "
            f"FROM player_rounds WHERE lower(trim(player_name)) IN ({placeholders}) "
            "AND round_date < ? ORDER BY round_date, tour, year, event_id, round_num",
            connection, params=[*sorted(names), cutoff],
        )
    history = history.drop_duplicates(["tour", "year", "event_id", "round_num"])
    history["sg_total_adj"] = pd.to_numeric(history["sg_total_adj"], errors="coerce")
    history = history[np.isfinite(history["sg_total_adj"])]
    if history.empty:
        raise ValueError(f"No pre-event adjusted SG history for late replacement {player}")
    return {
        "my_pred": float(history["sg_total_adj"].ewm(span=20, adjust=False).mean().iloc[-1]),
        "fallback_source": "historical_sg_ema20",
        "fallback_history_rounds": len(history),
        "fallback_last_round": str(history["round_date"].iloc[-1]),
        "fallback_cutoff": cutoff,
        "fallback_history_file": path.name,
    }


def replacement_category_distributions(player, cutoff, categories, frozen, *, db_path=None):
    """Use pre-event category history, with frozen-field variance for sparse categories."""
    path = Path(db_path) if db_path is not None else historical_database_path()
    columns = [f"{cat}_adj" for cat in categories]
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        history = pd.read_sql_query(
            f"SELECT {', '.join(columns)} FROM player_rounds "
            "WHERE lower(trim(player_name)) = ? AND round_date < ? "
            "ORDER BY round_date, tour, year, event_id, round_num",
            connection, params=[player, pd.Timestamp(cutoff).date().isoformat()],
        )
    rows = []
    for cat, column in zip(categories, columns):
        values = pd.to_numeric(history[column], errors="coerce")
        values = values[np.isfinite(values)].to_numpy()
        if len(values) >= 2:
            weights = (1 - 2 / 51) ** np.arange(len(values) - 1, -1, -1)
            weights /= weights.sum()
            mean50 = float(weights @ values)
            variance = float(weights @ (values - mean50) ** 2)
            std = float(np.sqrt(variance))
            skew = float(weights @ (values - mean50) ** 3 / std ** 3) if std > 0 else 0.0
            neff = float(1 / (weights @ weights))
            mean = float(pd.Series(values).ewm(span=20, adjust=False).mean().iloc[-1])
        else:
            std, mean, skew, neff = 0.0, 0.0, 0.0, 0.0
        if std <= 0:
            prior = frozen.loc[frozen["category_clean"].eq(cat), "std"]
            std = float(np.sqrt(np.square(pd.to_numeric(prior)).mean()))
            print(f"[late-field] {player}/{cat}: sparse history; frozen-field variance prior")
        rows.append({"player_name": player, "category_clean": cat, "mean": mean,
                     "std": std, "skew": skew, "n_eff": neff})
    return pd.DataFrame(rows)


def validated_late_players(event_id, tourney, *, root="."):
    """Only live-model rows marked by the guarded R1 recovery may use extra dists."""
    for relative in ("r1_live_model.csv", "dashboard_data/r1_live_model.csv"):
        path = Path(root) / relative
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        required = {"player_name", "fallback_source", "fallback_event_id", "fallback_tourney", "fallback_cutoff"}
        if not required.issubset(frame.columns):
            continue
        mask = (
            frame["fallback_source"].eq("historical_sg_ema20")
            & pd.to_numeric(frame["fallback_event_id"], errors="coerce").eq(int(event_id))
            & frame["fallback_tourney"].eq(str(tourney).lower())
        )
        rows = frame.loc[mask]
        return dict(zip(rows["player_name"].str.strip().str.lower(), rows["fallback_cutoff"]))
    return {}


def extend_category_distributions(frozen, catalog, active_players, categories, late_players, *, replacements=None):
    """Append validated catalog rows only for explicitly recovered late players."""
    frozen, frozen_players = require_complete_category_distributions(
        frozen, frozen["player_name"].drop_duplicates().tolist(), categories,
        name_replacements=replacements, source_label="frozen weekly distributions",
    )
    names = pd.Series(list(active_players)).str.strip().str.lower().replace(replacements or {})
    missing = sorted(set(names) - set(frozen_players))
    if not missing:
        return frozen, frozen_players, []
    if set(missing) - set(late_players):
        raise ValueError(f"Missing category players are not validated late replacements: {missing}")
    catalog, _ = require_complete_category_distributions(
        catalog, missing, categories, name_replacements=replacements,
        source_label="late-player full SG catalog",
    )
    extra = catalog[catalog["player_name"].isin(missing)]
    return pd.concat([frozen, extra], ignore_index=True), frozen_players, missing
