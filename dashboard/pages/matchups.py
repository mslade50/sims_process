"""Matchups page — Interactive matchup edge table with filters."""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd

from dashboard.data_layer import get_matchups, get_available_matchup_rounds, get_tournament_config
from dashboard.components.filters import (
    round_selector, sportsbook_filter, edge_slider, pred_slider, sample_slider, sharp_toggle,
)
from dashboard.components.tables import make_grid
from dashboard.config import SHARP_BOOKS

dash.register_page(__name__, path="/matchups", title="Matchups", order=2)


def _get_rounds():
    return get_available_matchup_rounds()


def _default_round(rounds):
    """Pick the highest numeric round as default."""
    numeric = [r for r in rounds if isinstance(r, int)]
    return max(numeric) if numeric else (rounds[-1] if rounds else None)


layout = dbc.Container([
    html.H4("Matchup Edges", className="page-header"),

    # Filters
    dbc.Row([
        round_selector("mu", available_rounds=_get_rounds() or [2, 3, 4],
                        default=_default_round(_get_rounds() or [2, 3, 4])),
        sportsbook_filter("mu"),
        edge_slider("mu", default=0),
        pred_slider("mu", default=0.75),
        sample_slider("mu", default=20),
    ], className="mb-3"),

    # Summary stats
    html.Div(id="mu-summary"),

    # Table
    html.Div(id="mu-table"),
], fluid=True)


@callback(
    Output("mu-summary", "children"),
    Output("mu-table", "children"),
    Input("mu-round-filter", "value"),
    Input("mu-book-filter", "value"),
    Input("mu-edge-slider", "value"),
    Input("mu-pred-slider", "value"),
    Input("mu-sample-slider", "value"),
)
def update_matchups(round_num, books, min_edge, min_pred, min_sample):
    if not round_num:
        return dbc.Alert("Select a round.", color="info"), ""

    df = get_matchups(round_num)
    if df.empty:
        return dbc.Alert(f"No matchup data for Round {round_num}.", color="warning"), ""

    # Normalize Bookmaker column name
    if "Bookmaker" not in df.columns and "bookmaker" in df.columns:
        df = df.rename(columns={"bookmaker": "Bookmaker"})

    # Apply filters
    if books and "Bookmaker" in df.columns:
        books_lower = [b.lower() for b in books]
        df = df[df["Bookmaker"].str.lower().isin(books_lower)]

    if min_edge is not None and min_edge > 0 and "edge_on" in df.columns:
        df = df[df["edge_on"] >= min_edge]

    if min_pred is not None and min_pred > 0 and "pred_on" in df.columns:
        df = df[df["pred_on"] >= min_pred]

    if min_sample is not None and min_sample > 0 and "sample_on" in df.columns:
        df = df[df["sample_on"] >= min_sample]

    if df.empty:
        return dbc.Alert("No matchups pass the current filters.", color="info"), ""

    # Summary
    count = len(df)
    avg_edge = df["edge_on"].mean() if "edge_on" in df.columns else 0
    summary = dbc.Row([
        dbc.Col(html.Span(f"{count} matchups", className="fw-bold"), width="auto"),
        dbc.Col(html.Span(f"Avg Edge: {avg_edge:.1f}%"), width="auto"),
    ], className="mb-2 g-3")

    # Display columns
    display_cols = [
        "bet_on", "bet_against", "Bookmaker", "Ties",
        "fair", "bet_on_odds", "bet_against_odds",
        "edge_on", "pred_on", "sample_on",
    ]

    available = [c for c in display_cols if c in df.columns]
    show_df = df[available].copy()

    # Sort by edge descending
    if "edge_on" in show_df.columns:
        show_df = show_df.sort_values("edge_on", ascending=False)

    # Custom column defs
    col_defs = []
    for col in available:
        d = {
            "field": col,
            "headerName": col.replace("_", " ").title(),
            "sortable": True,
            "filter": True,
            "resizable": True,
        }
        if col == "bet_on":
            d["headerName"] = "Bet On"
        elif col == "bet_against":
            d["headerName"] = "Bet Against"
        elif col == "Bookmaker":
            d["headerName"] = "Book"
        elif col == "fair":
            d["headerName"] = "Fair"
            d["valueFormatter"] = {"function": "params.value > 0 ? '+' + params.value : params.value"}
        elif col == "bet_on_odds":
            d["headerName"] = "Bet On Odds"
            d["valueFormatter"] = {"function": "params.value > 0 ? '+' + params.value : params.value"}
        elif col == "bet_against_odds":
            d["headerName"] = "Bet Against Odds"
            d["valueFormatter"] = {"function": "params.value > 0 ? '+' + params.value : params.value"}
        elif col == "edge_on":
            d["headerName"] = "Edge"
            d["valueFormatter"] = {"function": "d3.format('.1f')(params.value) + '%'"}
            d["cellStyle"] = {
                "styleConditions": [
                    {"condition": "params.value >= 10", "style": {"backgroundColor": "#1b5e20", "color": "white"}},
                    {"condition": "params.value >= 5 && params.value < 10", "style": {"backgroundColor": "#2e7d32", "color": "white"}},
                    {"condition": "params.value >= 3 && params.value < 5", "style": {"backgroundColor": "#4a6741", "color": "white"}},
                    {"condition": "params.value < 3", "style": {"backgroundColor": "#6d4c41", "color": "white"}},
                ]
            }
        elif col == "pred_on":
            d["headerName"] = "Pred"
            d["valueFormatter"] = {"function": "d3.format('.2f')(params.value)"}
        col_defs.append(d)

    grid = make_grid(show_df, column_defs=col_defs, id_suffix="matchups", height=650, page_size=30)

    return summary, grid
