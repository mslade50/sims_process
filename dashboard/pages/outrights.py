"""Outrights page — Win market edges, finish positions, probability heatmap."""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dashboard.data_layer import (
    get_outright_edges, get_devig_fades, get_finish_equity,
    get_simulated_probs, get_tournament_config,
)
from dashboard.components.tables import make_grid
from dashboard.components.filters import sportsbook_filter

dash.register_page(__name__, path="/outrights", title="Outrights", order=3)

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,33,62,0.8)",
    font=dict(color="#e0e0e0"),
    margin=dict(l=50, r=30, t=40, b=40),
)


layout = dbc.Container([
    html.H4("Outrights & Finish Positions", className="page-header"),

    dbc.Row([
        sportsbook_filter("out"),
    ], className="mb-3"),

    dbc.Tabs([
        dbc.Tab(label="Win Market", tab_id="win-tab", children=[
            html.Div(id="out-win-content", className="mt-3"),
        ]),
        dbc.Tab(label="Finish Positions", tab_id="finish-tab", children=[
            html.Div(id="out-finish-content", className="mt-3"),
        ]),
        dbc.Tab(label="Probability Heatmap", tab_id="heatmap-tab", children=[
            html.Div(id="out-heatmap-content", className="mt-3"),
        ]),
    ], id="out-tabs", active_tab="win-tab"),
], fluid=True)


@callback(
    Output("out-win-content", "children"),
    Input("out-tabs", "active_tab"),
    Input("out-book-filter", "value"),
)
def update_win_tab(active_tab, books):
    if active_tab != "win-tab":
        return dash.no_update

    config = get_tournament_config()
    tourney = config.get("tourney", "")

    # Positive edges
    edges_df = get_outright_edges(tourney)
    # Fades
    fades_df = get_devig_fades(tourney)

    # Filter by books
    if books:
        books_lower = [b.lower() for b in books]
        if not edges_df.empty and "bookmaker" in edges_df.columns:
            edges_df = edges_df[edges_df["bookmaker"].str.lower().isin(books_lower)]
        if not fades_df.empty and "bookmaker" in fades_df.columns:
            fades_df = fades_df[fades_df["bookmaker"].str.lower().isin(books_lower)]

    sections = []

    # Positive edges table
    sections.append(html.H5("Positive Edge — Win Market", className="mt-3 mb-2"))
    if edges_df.empty:
        sections.append(dbc.Alert("No outright win edge data available.", color="info"))
    else:
        if "edge" in edges_df.columns:
            edges_df = edges_df.sort_values("edge", ascending=False)
        sections.append(make_grid(edges_df, id_suffix="win-edges", height=400))

    # Fades table
    sections.append(html.H5("Fades (Negative Edge)", className="mt-4 mb-2"))
    if fades_df.empty:
        sections.append(dbc.Alert("No fade data available.", color="info"))
    else:
        if "edge_vs_devig" in fades_df.columns:
            fades_df = fades_df.sort_values("edge_vs_devig", ascending=True)
        sections.append(make_grid(fades_df, id_suffix="fades", height=400))

    return sections


@callback(
    Output("out-finish-content", "children"),
    Input("out-tabs", "active_tab"),
    Input("out-book-filter", "value"),
)
def update_finish_tab(active_tab, books):
    if active_tab != "finish-tab":
        return dash.no_update

    config = get_tournament_config()
    tourney = config.get("tourney", "")
    eq_df = get_finish_equity(tourney)

    if eq_df.empty:
        return dbc.Alert("No finish equity data available.", color="warning")

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
            sections.append(make_grid(sub, id_suffix=f"finish-{market}", height=350))
    else:
        if "edge" in eq_df.columns:
            eq_df = eq_df.sort_values("edge", ascending=False)
        sections.append(make_grid(eq_df, id_suffix="finish-all", height=500))

    return sections


@callback(
    Output("out-heatmap-content", "children"),
    Input("out-tabs", "active_tab"),
)
def update_heatmap(active_tab):
    if active_tab != "heatmap-tab":
        return dash.no_update

    probs_df = get_simulated_probs()
    if probs_df.empty:
        return dbc.Alert("No simulated probability data available.", color="warning")

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
        title="Simulated Finish Probabilities (Top 40)",
        height=max(600, len(players) * 22),
        yaxis=dict(autorange="reversed"),
    )

    return dcc.Graph(figure=fig)
