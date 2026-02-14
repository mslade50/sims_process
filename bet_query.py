"""
bet_query.py — CLI for querying the local Parquet bet ledger.

Modes:
  Terminal summary (default):
    python bet_query.py                          # All graded bets this year
    python bet_query.py --event farmers          # Filter by event
    python bet_query.py --type round_matchup     # By bet type
    python bet_query.py --book pinnacle          # By bookmaker
    python bet_query.py --min-edge 5             # Edge >= 5%
    python bet_query.py --graded                 # Only graded bets
    python bet_query.py --summary --by-event     # Summary grouped by event
    python bet_query.py --summary --by-book      # Summary grouped by book

  CSV export:
    python bet_query.py --event farmers --export

  Plotly dashboard:
    python bet_query.py --plot
"""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd
import numpy as np

from sheets_storage import LEDGER_PATH, query_ledger


def print_summary_table(df, group_col=None):
    """Print a formatted summary table to terminal."""
    if df.empty:
        print("  No bets found matching filters.")
        return

    graded = df[df["result"].astype(str).str.strip() != ""].copy()
    resolved = graded[~graded["result"].isin(["no_data", "unknown", "duplicate"])]

    print(f"\n  {'='*60}")
    print(f"  BET LEDGER QUERY RESULTS")
    print(f"  {'='*60}")
    print(f"  Total rows: {len(df)}   Graded: {len(graded)}   Resolved: {len(resolved)}")

    if resolved.empty:
        print("  No resolved bets to summarize.")
        return

    if group_col:
        _print_grouped(resolved, group_col)
    else:
        _print_overall(resolved)


def _print_overall(df):
    """Print overall performance summary."""
    wins = len(df[df["result"].isin(["win", "win_dh"])])
    losses = len(df[df["result"] == "loss"])
    pushes = len(df[df["result"] == "push"])
    wagered = df["units_wagered"].sum()
    won = df["units_won"].sum()
    roi = won / wagered * 100 if wagered > 0 else 0
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    print(f"\n  Record:    {wins}W - {losses}L - {pushes}P")
    print(f"  Win rate:  {win_rate:.1f}%")
    print(f"  Wagered:   {wagered:.2f} units")
    print(f"  Won:       {won:+.2f} units")
    print(f"  ROI:       {roi:+.1f}%")

    # By bet type
    print(f"\n  {'─'*50}")
    print(f"  {'Bet Type':<22} {'Bets':>5} {'Won':>8} {'ROI':>7}")
    print(f"  {'─'*50}")
    for bt in ["tournament_matchup", "round_matchup", "finish_position"]:
        sub = df[df["bet_type"] == bt]
        if sub.empty:
            continue
        n = len(sub)
        w = sub["units_won"].sum()
        wag = sub["units_wagered"].sum()
        r = w / wag * 100 if wag > 0 else 0
        print(f"  {bt:<22} {n:>5} {w:>+8.2f} {r:>+6.1f}%")

    # By edge bucket
    print(f"\n  {'─'*50}")
    print(f"  {'Edge Bucket':<22} {'Bets':>5} {'Won':>8} {'ROI':>7}")
    print(f"  {'─'*50}")
    for lo, hi, label in [(3, 5, "3-5%"), (5, 8, "5-8%"), (8, 100, "8%+")]:
        sub = df[(df["edge"] >= lo) & (df["edge"] < hi)]
        if sub.empty:
            continue
        n = len(sub)
        w = sub["units_won"].sum()
        wag = sub["units_wagered"].sum()
        r = w / wag * 100 if wag > 0 else 0
        print(f"  {label:<22} {n:>5} {w:>+8.2f} {r:>+6.1f}%")


def _print_grouped(df, group_col):
    """Print performance summary grouped by a column."""
    print(f"\n  {'─'*60}")
    print(f"  {group_col.title():<25} {'Bets':>5} {'W':>4} {'L':>4} {'Won':>8} {'ROI':>7}")
    print(f"  {'─'*60}")

    groups = df.groupby(group_col)
    rows = []
    for name, grp in groups:
        n = len(grp)
        w = len(grp[grp["result"].isin(["win", "win_dh"])])
        l = len(grp[grp["result"] == "loss"])
        won = grp["units_won"].sum()
        wag = grp["units_wagered"].sum()
        roi = won / wag * 100 if wag > 0 else 0
        rows.append((name, n, w, l, won, roi))

    # Sort by units won descending
    rows.sort(key=lambda x: x[4], reverse=True)
    for name, n, w, l, won, roi in rows:
        display = str(name)[:25]
        print(f"  {display:<25} {n:>5} {w:>4} {l:>4} {won:>+8.2f} {roi:>+6.1f}%")


def export_csv(df):
    """Export filtered DataFrame to CSV."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"filtered_bets_{ts}.csv"
    df.to_csv(filename, index=False)
    print(f"  Exported {len(df)} rows to {filename}")


def generate_dashboard(df):
    """Generate an interactive Plotly HTML dashboard."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  Error: plotly not installed. Run: pip install plotly")
        sys.exit(1)

    graded = df[df["result"].astype(str).str.strip() != ""].copy()
    resolved = graded[~graded["result"].isin(["no_data", "unknown", "duplicate"])].copy()

    if resolved.empty:
        print("  No resolved bets for dashboard.")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Cumulative P&L by Event",
            "ROI by Bookmaker",
            "Win Rate & ROI by Edge Bucket",
            "Results Scatter (Edge vs Units Won)",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    # ── Panel 1: Cumulative P&L by event ─────────────────────────────────
    for bt, name, color in [
        ("tournament_matchup", "Tournament MU", "#1f77b4"),
        ("round_matchup", "Round MU", "#ff7f0e"),
        ("finish_position", "Finish Position", "#2ca02c"),
    ]:
        sub = resolved[resolved["bet_type"] == bt].copy()
        if sub.empty:
            continue
        # Order by run_timestamp
        sub = sub.sort_values("run_timestamp")
        event_pnl = sub.groupby("event_name", sort=False)["units_won"].sum()
        cum = event_pnl.cumsum()
        fig.add_trace(
            go.Scatter(
                x=list(cum.index),
                y=cum.values,
                mode="lines+markers",
                name=name,
                line=dict(color=color),
            ),
            row=1, col=1,
        )

    # ── Panel 2: ROI by bookmaker ────────────────────────────────────────
    book_stats = (
        resolved.groupby("bookmaker")
        .agg(wagered=("units_wagered", "sum"), won=("units_won", "sum"), count=("bet_id", "count"))
        .reset_index()
    )
    book_stats["roi"] = book_stats["won"] / book_stats["wagered"] * 100
    book_stats = book_stats[book_stats["count"] >= 3].sort_values("roi")

    colors = []
    for _, row in book_stats.iterrows():
        b = str(row["bookmaker"]).lower()
        if any(s in b for s in ["pinnacle", "betonline", "betcris", "bookmaker"]):
            colors.append("#1f77b4")  # sharp = blue
        elif any(r in b for r in ["fanduel", "draftkings", "caesars", "betmgm"]):
            colors.append("#ff7f0e")  # retail = orange
        else:
            colors.append("#7f7f7f")  # other = gray

    fig.add_trace(
        go.Bar(
            y=book_stats["bookmaker"],
            x=book_stats["roi"],
            orientation="h",
            marker_color=colors,
            text=[f"{r:.1f}%" for r in book_stats["roi"]],
            textposition="auto",
            showlegend=False,
        ),
        row=1, col=2,
    )

    # ── Panel 3: W/L by edge bucket ──────────────────────────────────────
    buckets = [(3, 5, "3-5%"), (5, 8, "5-8%"), (8, 100, "8%+")]
    bucket_labels = []
    win_rates = []
    rois = []
    for lo, hi, label in buckets:
        sub = resolved[(resolved["edge"] >= lo) & (resolved["edge"] < hi)]
        if sub.empty:
            bucket_labels.append(label)
            win_rates.append(0)
            rois.append(0)
            continue
        w = len(sub[sub["result"].isin(["win", "win_dh"])])
        l = len(sub[sub["result"] == "loss"])
        wr = w / (w + l) * 100 if (w + l) > 0 else 0
        wag = sub["units_wagered"].sum()
        roi = sub["units_won"].sum() / wag * 100 if wag > 0 else 0
        bucket_labels.append(label)
        win_rates.append(wr)
        rois.append(roi)

    fig.add_trace(
        go.Bar(x=bucket_labels, y=win_rates, name="Win Rate %", marker_color="#2ca02c"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Bar(x=bucket_labels, y=rois, name="ROI %", marker_color="#d62728"),
        row=2, col=1,
    )

    # ── Panel 4: Results scatter ─────────────────────────────────────────
    color_map = {"win": "#2ca02c", "win_dh": "#98df8a", "loss": "#d62728", "push": "#7f7f7f"}
    for result_val, color in color_map.items():
        sub = resolved[resolved["result"] == result_val]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["edge"],
                y=sub["units_won"],
                mode="markers",
                name=result_val,
                marker=dict(color=color, size=6, opacity=0.7),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Event: %{customdata[1]}<br>"
                    "Book: %{customdata[2]}<br>"
                    "Edge: %{x:.1f}%<br>"
                    "Units: %{y:+.2f}<extra></extra>"
                ),
                customdata=sub[["bet_on", "event_name", "bookmaker"]].values,
            ),
            row=2, col=2,
        )

    fig.update_layout(
        height=800,
        title_text="Bet Ledger Dashboard",
        showlegend=True,
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Edge %", row=2, col=2)
    fig.update_yaxes(title_text="Units Won", row=2, col=2)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bet_dashboard_{ts}.html"
    fig.write_html(filename, auto_open=True)
    print(f"  Dashboard saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Query the local Parquet bet ledger")
    parser.add_argument("--event", type=str, help="Filter by event name (substring)")
    parser.add_argument("--type", type=str, dest="bet_type", help="Filter by bet type")
    parser.add_argument("--book", type=str, help="Filter by bookmaker (substring)")
    parser.add_argument("--min-edge", type=float, help="Minimum edge %%")
    parser.add_argument("--graded", action="store_true", help="Only graded bets")
    parser.add_argument("--year", type=int, default=datetime.now().year, help="Year (default: current)")
    parser.add_argument("--all-years", action="store_true", help="Include all years")

    parser.add_argument("--summary", action="store_true", help="Show grouped summary")
    parser.add_argument("--by-event", action="store_true", help="Group by event (with --summary)")
    parser.add_argument("--by-book", action="store_true", help="Group by bookmaker (with --summary)")

    parser.add_argument("--export", action="store_true", help="Export to CSV")
    parser.add_argument("--plot", action="store_true", help="Generate Plotly dashboard")

    args = parser.parse_args()

    if not os.path.exists(LEDGER_PATH):
        print(f"  Ledger not found at {LEDGER_PATH}")
        print("  Run new_sim.py or round_sim.py to create it.")
        sys.exit(1)

    filters = {
        "event": args.event,
        "bet_type": args.bet_type,
        "book": args.book,
        "min_edge": args.min_edge,
        "graded": args.graded,
    }
    if not args.all_years:
        filters["year"] = args.year

    df = query_ledger(**filters)

    if df.empty:
        print("  No bets found matching filters.")
        sys.exit(0)

    if args.export:
        export_csv(df)
    elif args.plot:
        generate_dashboard(df)
    elif args.summary:
        group_col = None
        if args.by_event:
            group_col = "event_name"
        elif args.by_book:
            group_col = "bookmaker"
        else:
            group_col = "event_name"
        print_summary_table(df, group_col=group_col)
    else:
        print_summary_table(df)


if __name__ == "__main__":
    main()
