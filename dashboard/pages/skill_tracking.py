"""Skill Tracking page — Live model prediction changes between rounds."""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dashboard.data_layer import get_live_model, get_available_live_model_rounds
from dashboard.components.filters import round_selector
from dashboard.components.tables import make_grid

dash.register_page(__name__, path="/skill", title="Skill Tracking", order=7)

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,33,62,0.8)",
    font=dict(color="#e0e0e0"),
    margin=dict(l=50, r=30, t=40, b=40),
)

# Columns to look for in the live model
PRED_COLS = {
    1: ("pred", "updated_pred"),
    2: ("updated_pred", "updated_pred_r3"),
    3: ("updated_pred_r3", "updated_pred_r4"),
    4: ("updated_pred_r4", "updated_pred_final"),
}

ADJUSTMENT_COLS = [
    "residual_adj", "residual2_adj", "residual3_adj",
    "avg_ott_adj", "avg_putt_adj", "avg_app_adj", "avg_arg_adj",
    "delta_app_adj", "tot_resid_adj", "tot_sg_adj", "total_adjustment",
]


layout = dbc.Container([
    html.H4("Skill Tracking", className="page-header"),

    dbc.Row([
        round_selector("skill", available_rounds=get_available_live_model_rounds() or [1, 2]),
        dbc.Col([
            html.Label("Player (for waterfall)", className="form-label small text-muted"),
            dcc.Dropdown(id="skill-player-dropdown", placeholder="Select player...", className="dash-dropdown-dark"),
        ], md=4),
    ], className="mb-3"),

    # Prediction table
    html.H5("Model Predictions", className="mb-2"),
    html.Div(id="skill-table"),

    # Charts
    dbc.Row([
        dbc.Col(dcc.Graph(id="skill-waterfall"), md=6),
        dbc.Col(dcc.Graph(id="skill-weather-scatter"), md=6),
    ], className="mt-3"),
], fluid=True)


@callback(
    Output("skill-player-dropdown", "options"),
    Output("skill-table", "children"),
    Output("skill-weather-scatter", "figure"),
    Input("skill-round-filter", "value"),
)
def update_skill_table(round_num):
    empty_fig = go.Figure(layout=PLOT_LAYOUT)

    if not round_num:
        return [], dbc.Alert("Select a round.", color="info"), empty_fig

    df = get_live_model(round_num)
    if df.empty:
        return [], dbc.Alert(f"No live model data for Round {round_num}.", color="warning"), empty_fig

    # Player options
    players = sorted(df["player_name"].dropna().unique().tolist())
    options = [{"label": p.title(), "value": p} for p in players]

    # Determine which columns to display
    pre_col, post_col = PRED_COLS.get(round_num, ("pred", "updated_pred"))
    display_cols = ["player_name"]
    if pre_col in df.columns:
        display_cols.append(pre_col)
    if post_col in df.columns:
        display_cols.append(post_col)
    if "total_adjustment" in df.columns:
        display_cols.append("total_adjustment")
    if "Score" in df.columns:
        display_cols.append("Score")
    if "position" in df.columns:
        display_cols.append("position")

    # Add adjustment columns that exist
    for col in ADJUSTMENT_COLS:
        if col in df.columns and col not in display_cols:
            display_cols.append(col)

    # Wind/dew columns
    for prefix in [f"wind_adj{round_num}", f"dew_adj{round_num}", f"wind_r{round_num}", f"dew_r{round_num}"]:
        if prefix in df.columns and prefix not in display_cols:
            display_cols.append(prefix)

    available = [c for c in display_cols if c in df.columns]
    show_df = df[available].copy()

    # Sort by post-prediction if available
    if post_col in show_df.columns:
        show_df = show_df.sort_values(post_col, ascending=False)
    elif pre_col in show_df.columns:
        show_df = show_df.sort_values(pre_col, ascending=False)

    grid = make_grid(show_df, id_suffix="skill", height=500, page_size=30)

    # Weather impact scatter
    wind_col = f"wind_adj{round_num}"
    adj_col = "total_adjustment"
    scatter_fig = go.Figure(layout={**PLOT_LAYOUT, "title": "Wind Adj vs Total Adjustment"})

    if wind_col in df.columns and adj_col in df.columns:
        scatter_fig.add_trace(go.Scatter(
            x=df[wind_col], y=df[adj_col],
            mode="markers",
            marker=dict(color="#4ecca3", size=6, opacity=0.7),
            text=df["player_name"].apply(lambda x: x.title() if isinstance(x, str) else x),
            hovertemplate="<b>%{text}</b><br>Wind Adj: %{x:.3f}<br>Total Adj: %{y:.3f}<extra></extra>",
        ))
        scatter_fig.update_xaxes(title_text=f"Wind Adj (R{round_num})")
        scatter_fig.update_yaxes(title_text="Total Adjustment")

    return options, grid, scatter_fig


@callback(
    Output("skill-waterfall", "figure"),
    Input("skill-round-filter", "value"),
    Input("skill-player-dropdown", "value"),
)
def update_waterfall(round_num, player):
    fig = go.Figure(layout={**PLOT_LAYOUT, "title": "Prediction Breakdown"})

    if not round_num or not player:
        return fig

    df = get_live_model(round_num)
    if df.empty:
        return fig

    row = df[df["player_name"] == player]
    if row.empty:
        return fig
    row = row.iloc[0]

    pre_col, post_col = PRED_COLS.get(round_num, ("pred", "updated_pred"))

    # Build waterfall steps
    measures = ["absolute"]
    x_labels = [f"Pre ({pre_col})"]
    y_values = [float(row.get(pre_col, 0))]

    # Add available adjustment components
    adj_display = [
        ("residual_adj", "Residual"),
        ("residual2_adj", "Residual 2"),
        ("residual3_adj", "Residual 3"),
        ("avg_ott_adj", "OTT Adj"),
        ("avg_putt_adj", "Putt Adj"),
        ("avg_app_adj", "APP Adj"),
        ("avg_arg_adj", "ARG Adj"),
        ("delta_app_adj", "Delta APP"),
        ("tot_resid_adj", "Tot Resid"),
        ("tot_sg_adj", "Tot SG"),
    ]

    for col, label in adj_display:
        if col in row.index and pd.notna(row[col]) and abs(float(row[col])) > 0.001:
            measures.append("relative")
            x_labels.append(label)
            y_values.append(float(row[col]))

    # Final post value
    if post_col in row.index and pd.notna(row[post_col]):
        measures.append("total")
        x_labels.append(f"Post ({post_col})")
        y_values.append(float(row[post_col]))

    fig.add_trace(go.Waterfall(
        x=x_labels,
        y=y_values,
        measure=measures,
        connector=dict(line=dict(color="#4ecca3", width=1)),
        increasing=dict(marker=dict(color="#2e7d32")),
        decreasing=dict(marker=dict(color="#b71c1c")),
        totals=dict(marker=dict(color="#1f77b4")),
        textposition="outside",
        texttemplate="%{y:.3f}",
    ))

    fig.update_layout(
        title=f"{player.title()} — R{round_num} Prediction Waterfall",
        showlegend=False,
    )

    return fig
