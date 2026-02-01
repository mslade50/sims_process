"""
round_sim.py — Unified Round Simulation for Matchups + Score Line Pricing

Replaces: round_mu_sim.py + round_scores.py (without HTML scraping)

Reads model_predictions_rN.csv (created by live_stats_engine.py).
Simulates round scores, prices matchups vs DataGolf API odds,
generates fair score-line pricing cards.

Usage:
    python round_sim.py                        (reads config from Google Sheet)
    python round_sim.py --cli --sim-round 2 --expected-avg 72.2

Outputs (saved to {tourney}/ folder):
    round_{N}_sim_{timestamp}.xlsx    — Matchup tabs + Score Card tab
    round_{N}_sim_scores.csv          — Raw simulated score distributions
"""

import os
import argparse
import numpy as np
import pandas as pd
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from datetime import datetime

from sim_inputs import (
    tourney, STD_DEV, name_replacements,
)

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("DATAGOLF_API_KEY")
MATCHUPS_URL = "https://feeds.datagolf.com/betting-tools/matchups"

NUM_SIMULATIONS = 100_000
SHARP_BOOKS = ["pinnacle", "betonline", "betcris"]
HALF_SHOT_ADJ = {"betonline": 25, "betcris": 30}

# Score card: generate fair UNDER prices at these offsets from expected avg
SCORE_CARD_RANGE = 3.0        # ±3 strokes from expected
SCORE_CARD_STEP = 0.5         # half-stroke intervals
MIN_PRED_FOR_CARD = -0.5      # exclude players with pred below this

# Email
EMAIL_FROM = os.getenv("EMAIL_USER")
EMAIL_TO = os.getenv("EMAIL_RECIPIENTS", "").split(",")

# Matchup email filter thresholds
EMAIL_MIN_PRED = 0.75
EMAIL_MIN_SAMPLE = 30


# ══════════════════════════════════════════════════════════════════════════════
# Odds Conversion Helpers
# ══════════════════════════════════════════════════════════════════════════════

def american_to_implied(odds):
    """American odds → implied probability (0–1)."""
    if pd.isna(odds) or odds == 0:
        return None
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def implied_to_american(prob):
    """Implied probability (0–1) → American odds (int)."""
    if prob is None or pd.isna(prob) or prob <= 0 or prob >= 1:
        return None
    if prob >= 0.5:
        return int(round(-100 * prob / (1 - prob)))
    return int(round(100 * (1 - prob) / prob))


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Score Simulation (shared by matchups + score card)
# ══════════════════════════════════════════════════════════════════════════════

def simulate_round_scores(model_preds, sim_round, expected_avg, num_sims=NUM_SIMULATIONS):
    """
    Simulate integer round scores for every player.

    Formula per player:
        actual_score = round( expected_avg − Normal(scores_rN, STD_DEV) )

    For multi-course events, each player's expected_avg comes from the
    'course_score_adj' column if present; otherwise uses the global expected_avg.

    Returns
    -------
    sim_dict : dict
        player_name → np.ndarray of simulated integer scores (shape: num_sims)
    """
    scores_col = f"scores_r{sim_round}"
    if scores_col not in model_preds.columns:
        raise ValueError(f"Column '{scores_col}' not found in predictions file. "
                         f"Available: {list(model_preds.columns)}")

    has_course_adj = "course_score_adj" in model_preds.columns

    sim_dict = {}
    for _, row in model_preds.iterrows():
        player = row["player_name"]
        skill = row[scores_col]

        # Skip players with missing predictions
        if pd.isna(skill):
            continue

        # Per-player expected avg (multi-course) or global
        if has_course_adj and pd.notna(row.get("course_score_adj")):
            player_avg = row["course_score_adj"]
        else:
            player_avg = expected_avg

        raw = np.random.normal(loc=skill, scale=STD_DEV, size=num_sims)
        sim_dict[player] = np.round(player_avg - raw).astype(int)

    print(f"  Simulated {len(sim_dict)} players × {num_sims:,} iterations")
    return sim_dict


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Matchup Pricing
# ══════════════════════════════════════════════════════════════════════════════

def fetch_matchup_odds():
    """Fetch round matchup odds from DataGolf API."""
    params = {
        "tour": "pga",
        "market": "round_matchups",
        "odds_format": "american",
        "file_format": "json",
        "key": API_KEY,
    }
    resp = requests.get(MATCHUPS_URL, params=params, timeout=30)
    if resp.status_code != 200:
        raise Exception(f"Matchup API failed ({resp.status_code}): {resp.text[:200]}")

    data = resp.json()
    rows = []
    for match in data.get("match_list", []):
        p1 = match["p1_player_name"].lower()
        p2 = match["p2_player_name"].lower()
        ties = match.get("ties", "unknown")

        for book, odds in match.get("odds", {}).items():
            if book == "datagolf":
                continue
            rows.append({
                "Player 1": p1,
                "Player 2": p2,
                "Bookmaker": book,
                "P1 Odds": odds.get("p1"),
                "P2 Odds": odds.get("p2"),
                "DG_p1": match["odds"].get("datagolf", {}).get("p1"),
                "DG_p2": match["odds"].get("datagolf", {}).get("p2"),
                "Ties": ties,
            })

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["Player 1", "Player 2", "Bookmaker"], keep="first")
    df["P1 Odds"] = pd.to_numeric(df["P1 Odds"], errors="coerce")
    df["P2 Odds"] = pd.to_numeric(df["P2 Odds"], errors="coerce")
    print(f"  Fetched {len(df)} matchup lines across {df['Bookmaker'].nunique()} books")
    return df


def price_matchups(matchup_df, sim_dict):
    """
    Attach fair win probabilities to each matchup row.

    Two probability modes per side:
        my_odds_pN       — ties are a push (excluded from total)
        my_odds_pN_tl    — ties count as losses
    """
    cols = {"fair_p1": [], "fair_p2": [], "tl_p1": [], "tl_p2": []}

    for _, row in matchup_df.iterrows():
        p1, p2 = row["Player 1"], row["Player 2"]

        if p1 not in sim_dict or p2 not in sim_dict:
            for k in cols:
                cols[k].append(None)
            continue

        s1, s2 = sim_dict[p1], sim_dict[p2]
        w1 = (s1 < s2).sum()
        w2 = (s1 > s2).sum()
        ties = (s1 == s2).sum()
        total = len(s1)
        non_tie = w1 + w2

        cols["fair_p1"].append(w1 / non_tie if non_tie else 0.5)
        cols["fair_p2"].append(w2 / non_tie if non_tie else 0.5)
        cols["tl_p1"].append(w1 / total)
        cols["tl_p2"].append(w2 / total)

    matchup_df["my_odds_p1"] = cols["fair_p1"]
    matchup_df["my_odds_p2"] = cols["fair_p2"]
    matchup_df["my_odds_p1_tl"] = cols["tl_p1"]
    matchup_df["my_odds_p2_tl"] = cols["tl_p2"]
    return matchup_df


def calculate_edges(df):
    """
    Calculate edges, fair odds, half-shot spreads for all matchup rows.
    Operates on the combined DataFrame (all bookmakers).
    """
    df = df.dropna(subset=["my_odds_p1", "my_odds_p2"]).copy()

    # Decimal odds from American
    df["p1_dec"] = np.where(
        df["P1 Odds"] > 0,
        df["P1 Odds"] / 100 + 1,
        100 / df["P1 Odds"].abs() + 1,
    )
    df["p2_dec"] = np.where(
        df["P2 Odds"] > 0,
        df["P2 Odds"] / 100 + 1,
        100 / df["P2 Odds"].abs() + 1,
    )

    # Which probability to use for edge: ties-loss when "separate bet offered"
    use_tl = df["Ties"] == "separate bet offered"

    prob_p1 = np.where(use_tl, df["my_odds_p1_tl"], df["my_odds_p1"])
    prob_p2 = np.where(use_tl, df["my_odds_p2_tl"], df["my_odds_p2"])

    # Edge = (prob × (decimal − 1) − (1 − prob)) × 100
    df["edge_p1"] = (prob_p1 * (df["p1_dec"] - 1) - (1 - prob_p1)) * 100
    df["edge_p2"] = (prob_p2 * (df["p2_dec"] - 1) - (1 - prob_p2)) * 100

    # Fair American odds (ties push)
    df["Fair_p1"] = df["my_odds_p1"].apply(
        lambda p: implied_to_american(p) if pd.notna(p) else None
    )
    df["Fair_p2"] = df["my_odds_p2"].apply(
        lambda p: implied_to_american(p) if pd.notna(p) else None
    )

    # Book implied probabilities (%)
    df["p1_implied"] = df["P1 Odds"].apply(
        lambda o: round(american_to_implied(o) * 100, 1) if pd.notna(o) else None
    )
    df["p2_implied"] = df["P2 Odds"].apply(
        lambda o: round(american_to_implied(o) * 100, 1) if pd.notna(o) else None
    )

    # Half-shot values: the value of a half-shot of spread
    df["half_shot_p1"] = (df["my_odds_p1"] - df["my_odds_p1_tl"]) * 400
    df["half_shot_p2"] = (df["my_odds_p2"] - df["my_odds_p2_tl"]) * 400

    # Push-wins: P(win or tie)
    df["p1_pushwins"] = (1 - df["my_odds_p2_tl"]) * 100
    df["p2_pushwins"] = (1 - df["my_odds_p1_tl"]) * 100

    # No-push: P(win, no tie) = ties-loss prob
    df["p1_nopush"] = df["my_odds_p1_tl"] * 100
    df["p2_nopush"] = df["my_odds_p2_tl"] * 100

    # ±0.5 spread edges for betonline / betcris
    for book, adj in HALF_SHOT_ADJ.items():
        mask = df["Bookmaker"].str.lower() == book
        if not mask.any():
            continue
        for side, odds_col in [("p1", "P1 Odds"), ("p2", "P2 Odds")]:
            pw_imp = (df.loc[mask, odds_col] - adj).apply(
                lambda o: round(american_to_implied(o) * 100, 1) if pd.notna(o) else None
            )
            np_imp = (df.loc[mask, odds_col] + adj).apply(
                lambda o: round(american_to_implied(o) * 100, 1) if pd.notna(o) else None
            )
            df.loc[mask, f"{side}_pushwins_imp"] = pw_imp
            df.loc[mask, f"{side}_nopush_imp"] = np_imp
            df.loc[mask, f"{side}_+0.5"] = df.loc[mask, f"{side}_pushwins"] - pw_imp
            df.loc[mask, f"{side}_-0.5"] = df.loc[mask, f"{side}_nopush"] - np_imp

    return df


def build_matchup_outputs(df, sim_round, pred_lookup, sample_lookup):
    """
    Filter, annotate, and split matchup DataFrame into combined + sharp outputs.

    Returns (combined_df, sharp_df).
    """
    # Merge predictions and sample sizes
    df["p1_pred"] = df["Player 1"].map(pred_lookup)
    df["p2_pred"] = df["Player 2"].map(pred_lookup)
    df["Sample_P1"] = df["Player 1"].map(sample_lookup)
    df["Sample_P2"] = df["Player 2"].map(sample_lookup)
    df["Round"] = f"r{sim_round}"

    # Derived columns
    df["edge_on"] = df[["edge_p1", "edge_p2"]].max(axis=1).round(1)
    df["bet_on"] = df.apply(
        lambda r: r["Player 1"] if r["edge_p1"] > r["edge_p2"] else r["Player 2"],
        axis=1,
    )
    df["pred_on"] = df.apply(
        lambda r: r["p1_pred"] if r["edge_p1"] > r["edge_p2"] else r["p2_pred"],
        axis=1,
    )
    df["pred_against"] = df.apply(
        lambda r: r["p2_pred"] if r["edge_p1"] > r["edge_p2"] else r["p1_pred"],
        axis=1,
    )
    df["sample_on"] = df.apply(
        lambda r: r["Sample_P1"] if r["edge_p1"] > r["edge_p2"] else r["Sample_P2"],
        axis=1,
    )

    # --- Combined: basic filters ---
    combined = df[df["edge_on"] > 3].copy()
    combined = combined[combined["sample_on"].fillna(0) >= 30]
    combined = combined[
        ((combined["pred_on"] > 0) & (combined["edge_on"] > 7))
        | (combined["pred_on"] > 1)
    ]
    combined = combined[
        ~((combined["edge_on"] < 5) & (combined["pred_on"] < 1))
    ]

    # --- Sharp: pinnacle / betonline / betcris, deduplicate by highest edge ---
    sharp = combined[combined["Bookmaker"].str.lower().isin(SHARP_BOOKS)].copy()
    sharp["matchup_key"] = sharp.apply(
        lambda r: "-".join(sorted([r["Player 1"], r["Player 2"]])), axis=1
    )
    sharp = sharp.sort_values("edge_on", ascending=False).drop_duplicates(
        "matchup_key", keep="first"
    )
    sharp = sharp.drop(columns="matchup_key")

    # --- Clean up display columns ---
    for out in [combined, sharp]:
        out["p1_pred"] = out["p1_pred"].round(2)
        out["p2_pred"] = out["p2_pred"].round(2)
        out["edge_p1"] = out["edge_p1"].round(1)
        out["edge_p2"] = out["edge_p2"].round(1)

    # Column ordering for output
    display_cols = [
        "Player 1", "Player 2", "Round", "Bookmaker", "Ties",
        "P1 Odds", "P2 Odds", "Fair_p1", "Fair_p2",
        "edge_p1", "edge_p2", "edge_on", "bet_on",
        "p1_pred", "p2_pred", "pred_on",
        "half_shot_p1", "half_shot_p2",
    ]
    # Add spread columns if they exist
    for col in ["p1_+0.5", "p2_+0.5", "p1_-0.5", "p2_-0.5"]:
        if col in combined.columns:
            display_cols.append(col)

    combined = combined[[c for c in display_cols if c in combined.columns]]
    sharp = sharp[[c for c in display_cols if c in sharp.columns]]

    print(f"  Combined matchups: {len(combined)} rows")
    print(f"  Sharp filtered:    {len(sharp)} rows")

    return combined, sharp


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Score Line Fair Card
# ══════════════════════════════════════════════════════════════════════════════

def build_score_card(sim_dict, expected_avg, pred_lookup):
    """
    Generate fair UNDER prices at half-stroke intervals around expected_avg.

    For each player and each line (e.g. 69.5, 70.5, ...):
        P(under) = P(score <= floor(line))  [no push at .5 lines]
        Fair UNDER = implied_to_american(P(under))

    Players with pred < MIN_PRED_FOR_CARD are excluded.

    Returns DataFrame with columns: Player, Pred, line_1, line_2, ...
    """
    # Generate standard .5 lines sportsbooks use (e.g. 68.5, 69.5, 70.5...)
    low = int(expected_avg - SCORE_CARD_RANGE)      # e.g. 69
    high = int(expected_avg + SCORE_CARD_RANGE) + 1  # e.g. 76
    lines = [x + 0.5 for x in range(low, high)]     # [69.5, 70.5, ..., 75.5]

    rows = []
    for player, scores in sim_dict.items():
        pred = pred_lookup.get(player)
        if pred is None or pred < MIN_PRED_FOR_CARD:
            continue

        row = {"Player": player, "Pred": round(pred, 2)}
        for line in lines:
            threshold = int(line)  # e.g. 70.5 → count scores ≤ 70
            under_pct = (scores <= threshold).mean()
            fair_under = implied_to_american(under_pct)
            row[str(line)] = fair_under

        rows.append(row)

    card = pd.DataFrame(rows)
    card = card.sort_values("Pred", ascending=False)
    print(f"  Score card: {len(card)} players × {len(lines)} lines ({lines[0]}–{lines[-1]})")
    return card


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Export
# ══════════════════════════════════════════════════════════════════════════════

def export_results(combined, sharp, score_card, sim_round):
    """Save all outputs to an Excel workbook + CSV backup."""
    timestamp = datetime.now().strftime("%H%M")
    out_dir = f"./{tourney}"
    os.makedirs(out_dir, exist_ok=True)

    excel_path = os.path.join(out_dir, f"round_{sim_round}_sim_{timestamp}.xlsx")

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        workbook = writer.book

        # --- Matchups: Combined ---
        if not combined.empty:
            combined.to_excel(writer, sheet_name="matchups_all", index=False)
            _format_matchup_sheet(writer, workbook, "matchups_all", combined)

        # --- Matchups: Sharp ---
        if not sharp.empty:
            sharp.to_excel(writer, sheet_name="matchups_sharp", index=False)
            _format_matchup_sheet(writer, workbook, "matchups_sharp", sharp)

        # --- Score Card ---
        if not score_card.empty:
            score_card.to_excel(writer, sheet_name="score_card", index=False)
            ws = writer.sheets["score_card"]

            # Format: highlight cells where fair odds are favorable (negative = under favorite)
            green = workbook.add_format({"bg_color": "#d4edda"})
            red = workbook.add_format({"bg_color": "#f8d7da"})
            num_fmt = workbook.add_format({"num_format": "0"})

            # Apply number format and conditional coloring to score columns
            for col_idx in range(2, len(score_card.columns)):
                col_letter = chr(65 + col_idx) if col_idx < 26 else f"A{chr(65 + col_idx - 26)}"
                ws.conditional_format(
                    1, col_idx, len(score_card), col_idx,
                    {"type": "cell", "criteria": "<", "value": 0, "format": green},
                )
                ws.conditional_format(
                    1, col_idx, len(score_card), col_idx,
                    {"type": "cell", "criteria": ">", "value": 0, "format": red},
                )

            # Auto-width
            for i, col in enumerate(score_card.columns):
                ws.set_column(i, i, max(len(str(col)) + 2, 8))

    print(f"\n  ✓ Saved {excel_path}")

    # Also save score card as standalone CSV for easy reference
    card_csv = os.path.join(out_dir, f"fair_card_r{sim_round}.csv")
    score_card.to_csv(card_csv, index=False)
    print(f"  ✓ Saved {card_csv}")

    return excel_path, card_csv


def _format_matchup_sheet(writer, workbook, sheet_name, df):
    """Apply conditional formatting to a matchup sheet."""
    ws = writer.sheets[sheet_name]
    yellow = workbook.add_format({"bg_color": "#FFFF00"})

    # Highlight rows where pred_on > 1 (strong conviction bets)
    if "pred_on" in df.columns:
        pred_col_idx = df.columns.get_loc("pred_on")
        ws.conditional_format(
            1, 0, len(df), len(df.columns) - 1,
            {
                "type": "formula",
                "criteria": f'=${chr(65 + pred_col_idx)}2>1',
                "format": yellow,
            },
        )

    # Auto-width columns
    for i, col in enumerate(df.columns):
        max_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
        ws.set_column(i, i, min(max_len, 20))


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def find_pred_col(model_preds, sim_round):
    """Find the best prediction column for display/filtering."""
    candidates = [
        f"scores_r{sim_round}",              # always exists
        f"my_pred{sim_round}" if sim_round > 1 else "my_pred",
        f"updated_pred_r{sim_round}",
        "updated_pred",
        "pred",
        "my_pred",
    ]
    for col in candidates:
        if col in model_preds.columns:
            return col
    return f"scores_r{sim_round}"


def load_sample_data():
    """Load sample sizes from pre_sim_summary if it exists."""
    path = f"pre_sim_summary_{tourney}.csv"
    if os.path.exists(path):
        sample = pd.read_csv(path)
        sample["player_name"] = sample["player_name"].str.lower().str.strip()
        return dict(zip(sample["player_name"], sample["sample"]))
    print(f"  ⚠️  {path} not found. Sample filter disabled.")
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# Email
# ══════════════════════════════════════════════════════════════════════════════

def build_matchup_email_html(sharp_df, sim_round, sample_lookup):
    """
    Build HTML email body with a table of sharp matchup picks.

    Filters sharp_df to rows where:
        - bet_on player's pred > EMAIL_MIN_PRED
        - bet_on player's sample > EMAIL_MIN_SAMPLE
    """
    if sharp_df.empty:
        return "<p>No sharp matchup picks for this round.</p>"

    # Filter: pred and sample thresholds on the bet_on side
    filtered = sharp_df.copy()
    filtered["sample_on"] = filtered["bet_on"].map(sample_lookup).fillna(0)
    filtered = filtered[
        (filtered["pred_on"] > EMAIL_MIN_PRED)
        & (filtered["sample_on"] >= EMAIL_MIN_SAMPLE)
    ]

    if filtered.empty:
        return "<p>No matchups passed filters (pred &gt; 0.75, sample &gt; 30).</p>"

    # Sort by edge descending
    filtered = filtered.sort_values("edge_on", ascending=False)

    # Build table rows
    rows_html = ""
    for _, row in filtered.iterrows():
        bet_player = row["bet_on"].title()
        opponent = (
            row["Player 2"].title()
            if row["bet_on"] == row["Player 1"]
            else row["Player 1"].title()
        )
        book = row.get("Bookmaker", "")
        ties = row.get("Ties", "")
        book_odds = (
            row["P1 Odds"] if row["bet_on"] == row["Player 1"] else row["P2 Odds"]
        )
        fair_odds = (
            row["Fair_p1"] if row["bet_on"] == row["Player 1"] else row["Fair_p2"]
        )
        edge = row["edge_on"]
        pred = row["pred_on"]
        sample = int(row["sample_on"])
        half_shot = (
            row.get("half_shot_p1", "")
            if row["bet_on"] == row["Player 1"]
            else row.get("half_shot_p2", "")
        )

        # Color coding
        edge_color = "#d4edda" if edge > 8 else "#fff3cd" if edge > 5 else "#ffffff"
        pred_color = "#d4edda" if pred > 1.5 else "#ffffff"

        # Format odds
        book_str = f"{int(book_odds):+d}" if pd.notna(book_odds) else ""
        fair_str = f"{int(fair_odds):+d}" if pd.notna(fair_odds) else ""
        hs_str = f"{half_shot:.1f}" if pd.notna(half_shot) and half_shot != "" else ""

        rows_html += f"""
        <tr>
            <td style="padding:6px 10px; font-weight:600;">{bet_player}</td>
            <td style="padding:6px 10px; color:#666;">vs {opponent}</td>
            <td style="padding:6px 10px; text-align:center;">{book}</td>
            <td style="padding:6px 10px; text-align:center;">{ties}</td>
            <td style="padding:6px 10px; text-align:center;">{book_str}</td>
            <td style="padding:6px 10px; text-align:center; font-weight:500;">{fair_str}</td>
            <td style="padding:6px 10px; text-align:center; font-weight:bold; background:{edge_color};">{edge:.1f}%</td>
            <td style="padding:6px 10px; text-align:center; background:{pred_color};">{pred:.2f}</td>
            <td style="padding:6px 10px; text-align:center;">{sample}</td>
            <td style="padding:6px 10px; text-align:center;">{hs_str}</td>
        </tr>"""

    html = f"""
    <html>
    <body style="font-family:Arial,sans-serif; max-width:960px; margin:0 auto; padding:20px;">
        <h2 style="margin-bottom:4px;">R{sim_round} Round Sim — {tourney.replace('_', ' ').title()}</h2>
        <p style="color:#666; margin-top:0;">{datetime.now().strftime('%B %d, %Y %I:%M %p')}</p>

        <h3 style="color:#2c5282; margin:20px 0 8px 0;">
            Sharp Matchup Picks (pred &gt; {EMAIL_MIN_PRED}, sample &gt; {EMAIL_MIN_SAMPLE})
        </h3>
        <table style="border-collapse:collapse; font-family:Arial,sans-serif; font-size:13px; width:100%;">
            <tr style="background:#343a40; color:white;">
                <th style="padding:6px 10px; text-align:left;">Bet On</th>
                <th style="padding:6px 10px; text-align:left;">Opponent</th>
                <th style="padding:6px 10px; text-align:center;">Book</th>
                <th style="padding:6px 10px; text-align:center;">Ties</th>
                <th style="padding:6px 10px; text-align:center;">Line</th>
                <th style="padding:6px 10px; text-align:center;">Fair</th>
                <th style="padding:6px 10px; text-align:center;">Edge</th>
                <th style="padding:6px 10px; text-align:center;">Pred</th>
                <th style="padding:6px 10px; text-align:center;">Sample</th>
                <th style="padding:6px 10px; text-align:center;">½ Shot</th>
            </tr>
            {rows_html}
        </table>

        <p style="color:#999; font-size:11px; margin-top:30px;">
            Fair = our no-vig price (ties push) | Edge = expected return % |
            Pred = model SG prediction | ½ Shot = value of half-shot spread (in edge pts)
        </p>
        <p style="color:#999; font-size:11px;">
            Attachments: fair score card (CSV), full matchup workbook (XLSX)
        </p>
    </body>
    </html>"""

    return html


def send_round_sim_email(sharp_df, sim_round, sample_lookup,
                         excel_path=None, card_csv_path=None):
    """
    Send round sim email with:
        - HTML body: filtered sharp matchup table
        - Attachment 1: fair score card CSV
        - Attachment 2: full matchup + score card Excel workbook
    
    Non-blocking: prints warning on failure but doesn't crash.
    """
    password = os.getenv("EMAIL_PASSWORD")
    if not password:
        print("  ⚠️  GMAIL_APP_PASSWORD not set. Skipping email.")
        return

    try:
        html = build_matchup_email_html(sharp_df, sim_round, sample_lookup)

        msg = MIMEMultipart("mixed")
        msg["Subject"] = f"R{sim_round} Round Sim — {tourney.replace('_', ' ').title()}"
        msg["From"] = EMAIL_FROM
        msg["To"] = ", ".join(EMAIL_TO)

        # HTML body
        msg.attach(MIMEText(html, "html"))

        # Attach fair card CSV
        if card_csv_path and os.path.exists(card_csv_path):
            with open(card_csv_path, "rb") as f:
                att = MIMEApplication(f.read(), _subtype="csv")
                att.add_header(
                    "Content-Disposition", "attachment",
                    filename=os.path.basename(card_csv_path),
                )
                msg.attach(att)

        # Attach Excel workbook
        if excel_path and os.path.exists(excel_path):
            with open(excel_path, "rb") as f:
                att = MIMEApplication(
                    f.read(),
                    _subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                att.add_header(
                    "Content-Disposition", "attachment",
                    filename=os.path.basename(excel_path),
                )
                msg.attach(att)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_FROM, password)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())

        print("  ✓ Round sim email sent")

    except Exception as e:
        print(f"  ⚠️  Email failed: {e}")
        print("    (Sim outputs still saved — email is non-blocking)")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Entry point. Reads config from Google Sheet or CLI args.

    The Google Sheet provides:
        round_num        → sim_round = round_num + 1
        expected_score_1 → expected scoring average (or first course for multi-course)
    """
    parser = argparse.ArgumentParser(description="Round Simulation — Matchups + Score Cards")
    parser.add_argument("--cli", action="store_true",
                        help="Use CLI args instead of Google Sheet config")
    parser.add_argument("--sim-round", type=int,
                        help="Round to simulate (e.g. 2 = simulate R2 scores)")
    parser.add_argument("--expected-avg", type=float,
                        help="Expected field scoring average for the round")
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────
    if not args.cli:
        try:
            from sheet_config import load_config
            config = load_config()
            round_num = config["round_num"]
            sim_round = round_num + 1 if round_num < 4 else 4
            expected_avg = config.get("expected_score_1")
            if expected_avg is None:
                from sim_inputs import PAR
                expected_avg = PAR
                print(f"  ⚠️  No expected_score_1 in sheet, using PAR={PAR}")
        except Exception as e:
            print(f"\n⚠️  Could not read Google Sheet: {e}")
            if args.sim_round is None:
                parser.error("Sheet unavailable and no --sim-round provided.")
            sim_round = args.sim_round
            expected_avg = args.expected_avg or 72
    else:
        if args.sim_round is None:
            parser.error("--sim-round is required in CLI mode")
        sim_round = args.sim_round
        expected_avg = args.expected_avg or 72

    # ── Load predictions ─────────────────────────────────────────────────
    pred_file = f"model_predictions_r{sim_round}.csv"
    if not os.path.exists(pred_file):
        raise FileNotFoundError(
            f"{pred_file} not found. Run live_stats_engine.py first."
        )

    model_preds = pd.read_csv(pred_file)
    model_preds["player_name"] = (
        model_preds["player_name"].str.lower().str.strip().replace(name_replacements)
    )

    pred_col = find_pred_col(model_preds, sim_round)
    pred_lookup = dict(zip(model_preds["player_name"], model_preds[pred_col]))
    sample_lookup = load_sample_data()

    print(f"\n{'='*60}")
    print(f"  ROUND {sim_round} SIMULATION — {tourney}")
    print(f"{'='*60}")
    print(f"  Predictions:  {pred_file} ({len(model_preds)} players)")
    print(f"  Expected avg: {expected_avg}")
    print(f"  Std dev:      {STD_DEV}")
    print(f"  Simulations:  {NUM_SIMULATIONS:,}")
    print(f"  Pred column:  {pred_col}")

    # ── Step 1: Simulate scores ──────────────────────────────────────────
    print(f"\n  Simulating R{sim_round} scores...")
    sim_dict = simulate_round_scores(model_preds, sim_round, expected_avg)

    # ── Step 2: Matchup pricing ──────────────────────────────────────────
    print(f"\n  Fetching matchup odds from DataGolf...")
    try:
        matchup_df = fetch_matchup_odds()
        matchup_df = price_matchups(matchup_df, sim_dict)
        matchup_df = calculate_edges(matchup_df)
        combined, sharp = build_matchup_outputs(
            matchup_df, sim_round, pred_lookup, sample_lookup
        )
    except Exception as e:
        print(f"  ⚠️  Matchup pricing failed: {e}")
        combined = pd.DataFrame()
        sharp = pd.DataFrame()

    # ── Step 3: Score card ───────────────────────────────────────────────
    print(f"\n  Building fair score card (expected avg = {expected_avg})...")
    score_card = build_score_card(sim_dict, expected_avg, pred_lookup)

    # ── Step 4: Export ───────────────────────────────────────────────────
    excel_path, card_csv = export_results(combined, sharp, score_card, sim_round)

    # ── Step 5: Email ────────────────────────────────────────────────────
    print(f"\n  Sending email...")
    send_round_sim_email(
        sharp_df=sharp,
        sim_round=sim_round,
        sample_lookup=sample_lookup,
        excel_path=excel_path,
        card_csv_path=card_csv,
    )

    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()