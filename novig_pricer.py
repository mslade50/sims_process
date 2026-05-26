"""Live NoVig outright pricing for the dashboard.

Fetches current NoVig orderbook via the public GraphQL endpoint (no auth) and
computes edges vs the latest persisted sim probabilities.

Mirrors `price_novig_outrights_tourney` in new_sim.py but loads its inputs
directly from disk so it can be called from the dashboard process.
"""
from __future__ import annotations

import os

import httpx
import numpy as np
import pandas as pd

GRAPHQL_URL = "https://api.novig.us/v1/graphql"
GQL_HEADERS = {
    "Content-Type": "application/json",
    "Origin": "https://novig.com",
    "Referer": "https://novig.com/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

NOVIG_TYPE_MAP = {
    "TOP_FIVE_FINISH": "top_5",
    "TOP_TEN_FINISH": "top_10",
    "TOP_TWENTY_FINISH": "top_20",
    "WINNER": "winner",
}


def _prob_to_american(p):
    if p is None or pd.isna(p) or p <= 0:
        return None
    if p >= 1:
        return -100
    return int(round(-100 * p / (1 - p))) if p > 0.5 else int(round(100 * (1 - p) / p))


def _implied_prob_to_american_odds(prob):
    if prob is None or prob <= 0 or prob >= 1:
        return None
    if prob >= 0.5:
        return int(np.floor(-100 * prob / (1 - prob)))
    return int(np.floor(100 * (1 - prob) / prob))


def _load_finish_probs(tourney: str, project_root: str) -> pd.DataFrame:
    """Build a finish_probs frame with simulated_win_prob and *_nodh cols.

    Pulls win/top-N (DH-resolved) from simulated_probs_live.csv or the pre-event
    simulated_probs.csv, and *_nodh from rank_probs_live_{tourney}.parquet or
    rank_probs_updated_{tourney}.parquet.
    """
    # Live first, pre-event fallback
    sp_candidates = [
        os.path.join(project_root, "simulated_probs_live.csv"),
        os.path.join(project_root, tourney, "simulated_probs_live.csv"),
        os.path.join(project_root, tourney, "v2", "simulated_probs.csv"),
        os.path.join(project_root, "simulated_probs.csv"),
        os.path.join(project_root, "dashboard_data", "simulated_probs_live.csv"),
        os.path.join(project_root, "dashboard_data", "simulated_probs.csv"),
    ]
    sp_df = pd.DataFrame()
    for p in sp_candidates:
        if os.path.exists(p):
            try:
                sp_df = pd.read_csv(p)
                break
            except Exception:
                continue
    if sp_df.empty:
        return pd.DataFrame()
    sp_df["player_name"] = sp_df["player_name"].astype(str).str.lower()

    # rank_probs for *_nodh
    rp_candidates = [
        os.path.join(project_root, f"rank_probs_live_{tourney}.parquet"),
        os.path.join(project_root, f"rank_probs_updated_{tourney}.parquet"),
        os.path.join(project_root, tourney, "v2", f"rank_probs_updated_{tourney}.parquet"),
        os.path.join(project_root, "dashboard_data", "rank_probs_live.parquet"),
        os.path.join(project_root, "dashboard_data", "rank_probs_pre.parquet"),
    ]
    rp = pd.DataFrame()
    for p in rp_candidates:
        if os.path.exists(p):
            try:
                rp = pd.read_parquet(p)
                break
            except Exception:
                continue

    if not rp.empty and {"player_name", "rank", "prob_ndh"}.issubset(rp.columns):
        rp = rp.copy()
        rp["player_name"] = rp["player_name"].astype(str).str.lower()
        nodh = rp.groupby("player_name").apply(
            lambda g: pd.Series({
                "top_5_nodh": g.loc[g["rank"] <= 5, "prob_ndh"].sum(),
                "top_10_nodh": g.loc[g["rank"] <= 10, "prob_ndh"].sum(),
                "top_20_nodh": g.loc[g["rank"] <= 20, "prob_ndh"].sum(),
                "win_nodh": g.loc[g["rank"] == 1, "prob_ndh"].sum(),
            }),
            include_groups=False,
        ).reset_index()
        sp_df = sp_df.merge(nodh, on="player_name", how="left")

    return sp_df


def price_novig_outrights(tourney: str, project_root: str | None = None) -> pd.DataFrame:
    """Fetch live NoVig outrights, compute edges vs latest sim probs.

    Returns DataFrame with columns: player_name, market_type, bookmaker, side,
    eff_price, american_odds, sim_prob, edge (pp), my_fair, volume, ticker.
    Empty DataFrame if no open NoVig tournament or no sim probs found.
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.abspath(__file__))

    finish_probs = _load_finish_probs(tourney, project_root)
    if finish_probs.empty:
        return pd.DataFrame()

    # Map market_type → which prob column to use
    type_to_col = {
        "top_5": "top_5_nodh", "top_10": "top_10_nodh",
        "top_20": "top_20_nodh", "winner": "simulated_win_prob",
    }
    # Fall back to DH-adjusted cols if nodh missing
    fallback = {"top_5_nodh": "top_5", "top_10_nodh": "top_10",
                "top_20_nodh": "top_20", "win_nodh": "simulated_win_prob"}
    for mtype, col in list(type_to_col.items()):
        if col not in finish_probs.columns:
            type_to_col[mtype] = fallback.get(col, col)
            if type_to_col[mtype] not in finish_probs.columns:
                type_to_col.pop(mtype, None)
    if not type_to_col:
        return pd.DataFrame()

    # Name normalization
    try:
        from sim_inputs import name_replacements
    except Exception:
        name_replacements = {}

    def norm(s):
        x = str(s).strip().lower()
        if "," not in x:
            parts = x.rsplit(" ", 1)
            if len(parts) == 2:
                x = f"{parts[1]}, {parts[0]}"
        return name_replacements.get(x, x)

    client = httpx.Client(timeout=15.0)

    def gql(query, variables=None):
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        resp = client.post(GRAPHQL_URL, headers=GQL_HEADERS, json=payload)
        resp.raise_for_status()
        return resp.json()

    try:
        data = gql("""query {
  event(where: {league: {_eq: "PGA"}, type: {_eq: "Tournament"}, status: {_in: ["OPEN_PREGAME", "OPEN_INGAME"]}}
        order_by: {scheduled_start: asc}) {
    id description
    child_events_aggregate: events_aggregate(where: {status: {_in: ["OPEN_PREGAME", "OPEN_INGAME"]}}) {
      aggregate { count }
    }
  }
}""")
        events = data.get("data", {}).get("event", [])
        if not events:
            client.close()
            return pd.DataFrame()
        best = max(events, key=lambda e: e.get("child_events_aggregate", {})
                   .get("aggregate", {}).get("count", 0))
        tourn_id = best["id"]
        tourn_name = best.get("description", "")
    except Exception:
        client.close()
        return pd.DataFrame()

    rows = []
    for novig_type, our_type in NOVIG_TYPE_MAP.items():
        if our_type not in type_to_col:
            continue
        try:
            data = gql("""query($tournId: uuid!, $mtype: String!) {
  event(where: {parent_event: {id: {_eq: $tournId}}, type: {_eq: "Future"},
                status: {_in: ["OPEN_PREGAME", "OPEN_INGAME"]},
                markets: {type: {_eq: $mtype}, status: {_eq: "OPEN"}}}) {
    id description
    markets(where: {type: {_eq: $mtype}, status: {_eq: "OPEN"}}) {
      id type volume description
      outcomes { id index description available }
    }
  }
}""", {"tournId": tourn_id, "mtype": novig_type})
            events = data.get("data", {}).get("event", [])
            target = None
            for e in events:
                desc = e.get("description", "")
                if "With Ties" in desc or "End Of Round" in desc:
                    continue
                if e.get("markets"):
                    target = e
                    break
            if not target:
                continue

            prob_col = type_to_col[our_type]

            for m in target.get("markets", []):
                outcomes = m.get("outcomes", [])
                yes_price = None
                no_price = None
                for o in outcomes:
                    if o["index"] == 0:
                        yes_price = o.get("available")
                    elif o["index"] == 1:
                        no_price = o.get("available")
                if not yes_price or yes_price <= 0:
                    continue

                desc = m.get("description", "")
                player_raw = desc.replace(f" {novig_type}", "").strip()
                if not player_raw:
                    continue
                player = norm(player_raw)

                match = finish_probs[finish_probs["player_name"] == player]
                if match.empty:
                    continue
                sim_prob = float(match.iloc[0][prob_col])
                if pd.isna(sim_prob) or sim_prob <= 0:
                    continue

                volume = float(m.get("volume", 0) or 0)

                yes_edge = (sim_prob - yes_price) * 100
                yes_american = _implied_prob_to_american_odds(yes_price)
                if yes_american is not None:
                    rows.append({
                        "player_name": player,
                        "market_type": our_type,
                        "bookmaker": "novig",
                        "side": "yes",
                        "eff_price": yes_price,
                        "american_odds": yes_american,
                        "sim_prob": sim_prob,
                        "edge": round(yes_edge, 1),
                        "my_fair": _prob_to_american(sim_prob),
                        "volume": volume,
                        "ticker": m.get("id", ""),
                    })

                if no_price and no_price > 0:
                    sim_no = 1.0 - sim_prob
                    no_edge = (sim_no - no_price) * 100
                    no_american = _implied_prob_to_american_odds(no_price)
                    if no_american is not None:
                        rows.append({
                            "player_name": player,
                            "market_type": our_type,
                            "bookmaker": "novig",
                            "side": "no",
                            "eff_price": no_price,
                            "american_odds": no_american,
                            "sim_prob": sim_no,
                            "edge": round(no_edge, 1),
                            "my_fair": _prob_to_american(sim_no),
                            "volume": volume,
                            "ticker": m.get("id", ""),
                        })
        except Exception:
            continue

    client.close()

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("edge", ascending=False).reset_index(drop=True)
        df.attrs["tournament"] = tourn_name
    return df


if __name__ == "__main__":
    from sim_inputs import tourney
    df = price_novig_outrights(tourney)
    if df.empty:
        print("No NoVig edges (or no open tournament).")
    else:
        print(f"NoVig tournament: {df.attrs.get('tournament', '?')}")
        print(f"{len(df)} lines, {(df['edge'] > 0).sum()} positive edges")
        print(df.head(20).to_string(index=False))
