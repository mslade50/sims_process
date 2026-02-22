"""Home page — Tournament overview and navigation."""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import os

from dashboard.data_layer import (
    get_tournament_config, get_bet_ledger, get_available_rounds,
    get_available_matchup_rounds, get_model_predictions, get_file_mtime,
)
from dashboard.components.stat_cards import stat_card_row
from dashboard.config import PROJECT_ROOT

dash.register_page(__name__, path="/", title="Home", order=1)

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,33,62,0.8)",
    font=dict(color="#e0e0e0"),
    margin=dict(l=40, r=20, t=30, b=30),
)

NAV_CARDS = [
    {"title": "Matchups", "href": "/matchups", "icon": "vs", "desc": "Round matchup edges by book"},
    {"title": "Outrights", "href": "/outrights", "icon": "1st", "desc": "Win & finish position edges"},
    {"title": "Pricer", "href": "/pricer", "icon": "O/U", "desc": "Interactive over/under calculator"},
    {"title": "Bets", "href": "/bets", "icon": "$", "desc": "Active bets this tournament"},
    {"title": "Performance", "href": "/performance", "icon": "P&L", "desc": "Historical ROI & results"},
    {"title": "Skill", "href": "/skill", "icon": "SG", "desc": "Live prediction adjustments"},
    {"title": "Diagnostics", "href": "/diagnostics", "icon": "Dx", "desc": "SG prediction analysis"},
]


layout = dbc.Container([
    # Header
    html.Div(id="home-header"),

    # KPI row
    html.Div(id="home-kpi-row", className="mb-3"),

    # Weather sparklines
    dbc.Row([
        dbc.Col(dcc.Graph(id="home-wind-spark", config={"displayModeBar": False}), md=6),
        dbc.Col(dcc.Graph(id="home-dew-spark", config={"displayModeBar": False}), md=6),
    ], className="mb-3"),

    # Data freshness
    html.Div(id="home-freshness", className="mb-4"),

    # Navigation cards
    html.H5("Pages", className="mb-3"),
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.Div(
                        card["icon"],
                        className="fw-bold text-center",
                        style={"fontSize": "1.5rem", "color": "#4ecca3"},
                    ),
                    html.H6(card["title"], className="card-title text-center mt-2"),
                    html.P(card["desc"], className="card-text small text-muted text-center"),
                ]),
            ], color="dark", outline=True, className="h-100",
               style={"cursor": "pointer"},
            ),
            md=3, sm=6, className="mb-3",
            # Wrap in a link
        ) for card in NAV_CARDS
    ]),

    # Wrap cards in links via dcc.Link
    dcc.Location(id="home-nav-url"),

    # Interval for auto-refresh
    dcc.Interval(id="home-refresh", interval=300_000, n_intervals=0),  # 5 min
], fluid=True)


# Wrap nav cards in links
layout.children[6] = dbc.Row([
    dbc.Col(
        dcc.Link(
            dbc.Card([
                dbc.CardBody([
                    html.Div(
                        card["icon"],
                        className="fw-bold text-center",
                        style={"fontSize": "1.5rem", "color": "#4ecca3"},
                    ),
                    html.H6(card["title"], className="card-title text-center mt-2"),
                    html.P(card["desc"], className="card-text small text-muted text-center"),
                ]),
            ], color="dark", outline=True, className="h-100",
               style={"cursor": "pointer"}),
            href=card["href"],
            style={"textDecoration": "none"},
        ),
        md=3, sm=6, className="mb-3",
    ) for card in NAV_CARDS
])


@callback(
    Output("home-header", "children"),
    Output("home-kpi-row", "children"),
    Output("home-wind-spark", "figure"),
    Output("home-dew-spark", "figure"),
    Output("home-freshness", "children"),
    Input("home-refresh", "n_intervals"),
)
def update_home(_):
    config = get_tournament_config()
    tourney = config.get("tourney", "Unknown")
    course_par = config.get("course_par", config.get("PAR", 72))

    # Header
    avail_rounds = get_available_rounds()
    current_round = max(avail_rounds) if avail_rounds else 0
    round_indicators = []
    for r in range(1, 5):
        color = "success" if r <= current_round else "secondary"
        round_indicators.append(dbc.Badge(f"R{r}", color=color, className="me-1"))

    header = html.Div([
        html.H3(f"{tourney.title()}", className="mb-1"),
        html.Div([
            html.Span(f"Par {course_par}", className="text-muted me-3"),
            *round_indicators,
        ]),
    ], className="page-header")

    # KPI cards
    df = get_bet_ledger(event=tourney)
    total_bets = len(df) if not df.empty else 0
    avg_edge = df["edge"].mean() if not df.empty and "edge" in df.columns else 0

    resolved = pd.DataFrame()
    if not df.empty:
        resolved = df[df["result"].astype(str).str.strip() != ""]
        resolved = resolved[~resolved["result"].isin(["no_data", "unknown", "duplicate"])]

    pnl = resolved["units_won"].sum() if not resolved.empty else 0

    # Field size from latest model predictions, fall back to unique players in bets
    field_size = None
    if avail_rounds:
        pred_df = get_model_predictions(avail_rounds[-1])
        if not pred_df.empty:
            field_size = len(pred_df)
    if field_size is None and not df.empty and "bet_on" in df.columns:
        field_size = df["bet_on"].nunique()
    field_str = str(field_size) if field_size else "—"

    kpi = stat_card_row([
        {"title": "Bets This Week", "value": str(total_bets)},
        {"title": "Avg Edge", "value": f"{avg_edge:.1f}%"},
        {"title": "P&L", "value": f"{pnl:+.2f}u", "color": "success" if pnl >= 0 else "danger"},
        {"title": "Field Size", "value": field_str},
    ])

    # Wind sparklines
    wind_fig = go.Figure(layout={**PLOT_LAYOUT, "title": "Wind Forecast (mph)", "height": 200})
    dew_fig = go.Figure(layout={**PLOT_LAYOUT, "title": "Dewpoint Forecast", "height": 200})

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for r, color in zip(range(1, 5), colors):
        wind_key = f"wind_{r}"
        dew_key = f"dewpoint_{r}"
        if wind_key in config:
            hours = list(range(6, 6 + len(config[wind_key])))
            wind_fig.add_trace(go.Scatter(
                x=hours, y=config[wind_key],
                mode="lines", name=f"R{r}", line=dict(color=color, width=2),
            ))
        if dew_key in config:
            hours = list(range(6, 6 + len(config[dew_key])))
            dew_fig.add_trace(go.Scatter(
                x=hours, y=config[dew_key],
                mode="lines", name=f"R{r}", line=dict(color=color, width=2),
            ))

    wind_fig.update_xaxes(title_text="Hour", dtick=2)
    dew_fig.update_xaxes(title_text="Hour", dtick=2)

    # Data freshness
    files_to_check = [
        ("Model Predictions R1", "model_predictions_r1.csv"),
        ("Model Predictions R2", "model_predictions_r2.csv"),
        ("Model Predictions R3", "model_predictions_r3.csv"),
        ("Live Model R1", "r1_live_model.csv"),
        ("Live Model R2", "r2_live_model.csv"),
        ("Bet Ledger", "permanent_data/bet_ledger.parquet"),
        ("Simulated Probs", "simulated_probs_live.csv"),
    ]

    freshness_items = []
    for label, filename in files_to_check:
        mtime = get_file_mtime(filename)
        if mtime:
            dt = datetime.fromtimestamp(mtime)
            age_min = (datetime.now() - dt).total_seconds() / 60
            badge_color = "success" if age_min < 60 else "warning" if age_min < 360 else "secondary"
            freshness_items.append(
                dbc.Badge(
                    f"{label}: {dt.strftime('%m/%d %I:%M%p')}",
                    color=badge_color, className="me-2 mb-1",
                )
            )

    freshness = html.Div([
        html.H6("Data Freshness", className="text-muted mb-1"),
        html.Div(freshness_items),
    ]) if freshness_items else ""

    return header, kpi, wind_fig, dew_fig, freshness
