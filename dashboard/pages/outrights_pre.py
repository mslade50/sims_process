"""Outrights (Pre-R1) — Pre-tournament finish positions and probability heatmap from new_sim.py."""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dashboard.data_layer import (
    get_finish_equity_pre, get_simulated_probs_pre, get_tournament_config,
)
from dashboard.components.tables import make_grid
from dashboard.components.filters import sportsbook_filter

dash.register_page(__name__, path="/outrights-pre", title="Outrights (Pre-R1)", order=3)

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,33,62,0.8)",
    font=dict(color="#e0e0e0"),
    margin=dict(l=50, r=30, t=40, b=40),
)


layout = dbc.Container([
    html.H4("Outrights & Finish Positions (Pre-R1)", className="page-header"),

    dbc.Row([
        sportsbook_filter("outpre"),
    ], className="mb-3"),

    dbc.Tabs([
        dbc.Tab(label="Finish Positions", tab_id="outpre-finish-tab", children=[
            html.Div(id="outpre-finish-content", className="mt-3"),
        ]),
        dbc.Tab(label="Probability Heatmap", tab_id="outpre-heatmap-tab", children=[
            html.Div(id="outpre-heatmap-content", className="mt-3"),
        ]),
    ], id="outpre-tabs", active_tab="outpre-finish-tab"),
], fluid=True)


@callback(
    Output("outpre-finish-content", "children"),
    Input("outpre-tabs", "active_tab"),
    Input("outpre-book-filter", "value"),
)
def update_finish_tab_pre(active_tab, books):
    if active_tab != "outpre-finish-tab":
        return dash.no_update

    config = get_tournament_config()
    tourney = config.get("tourney", "")
    eq_df = get_finish_equity_pre(tourney)

    if eq_df.empty:
        return dbc.Alert("No pre-tournament finish equity data available.", color="warning")

    if books:
        books_lower = [b.lower() for b in books]
        book_col = "bookmaker" if "bookmaker" in eq_df.columns else "sportsbook" if "sportsbook" in eq_df.columns else None
        if book_col:
            eq_df = eq_df[eq_df[book_col].str.lower().isin(books_lower)]

    if eq_df.empty:
        return dbc.Alert("No finish positions pass the book filter.", color="info")

    # Group by market type
    sections = []
    market_col = "market_type" if "market_type" in eq_df.columns else "market"
    if market_col in eq_df.columns:
        for market in ["win", "top_5", "top_10", "top_20"]:
            sub = eq_df[eq_df[market_col] == market]
            if sub.empty:
                continue
            if "edge" in sub.columns:
                sub = sub.sort_values("edge", ascending=False)
            sections.append(html.H5(f"{market.replace('_', ' ').title()} Market", className="mt-3 mb-2"))
            sections.append(make_grid(sub, id_suffix=f"pre-finish-{market}", height=350))
    else:
        if "edge" in eq_df.columns:
            eq_df = eq_df.sort_values("edge", ascending=False)
        sections.append(make_grid(eq_df, id_suffix="pre-finish-all", height=500))

    return sections


@callback(
    Output("outpre-heatmap-content", "children"),
    Input("outpre-tabs", "active_tab"),
)
def update_heatmap_pre(active_tab):
    if active_tab != "outpre-heatmap-tab":
        return dash.no_update

    probs_df = get_simulated_probs_pre()
    if probs_df.empty:
        return dbc.Alert("No pre-tournament simulated probability data available.", color="warning")

    # Build heatmap
    prob_cols = [c for c in ["simulated_win_prob", "top_5", "top_10", "top_20"] if c in probs_df.columns]
    if not prob_cols:
        return dbc.Alert("Probability columns not found.", color="warning")

    # Sort by win prob descending, take top 40
    sort_col = "simulated_win_prob" if "simulated_win_prob" in probs_df.columns else prob_cols[0]
    probs_df = probs_df.sort_values(sort_col, ascending=False).head(40)

    players = probs_df["player_name"].apply(lambda x: x.title() if isinstance(x, str) else x).tolist()
    z = probs_df[prob_cols].values * 100  # convert to percentage

    col_labels = [c.replace("simulated_win_prob", "Win %").replace("top_", "Top ").replace("_", " ").title()
                  for c in prob_cols]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=col_labels,
        y=players,
        colorscale="Greens",
        text=np.round(z, 1),
        texttemplate="%{text}%",
        textfont={"size": 10},
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        **PLOT_LAYOUT,
        title="Simulated Finish Probabilities -- Pre-R1 (Top 40)",
        height=max(600, len(players) * 22),
        yaxis=dict(autorange="reversed"),
    )

    return dcc.Graph(figure=fig)
