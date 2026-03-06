"""Score Pricer page — Interactive over/under calculator."""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.stats import norm

from dashboard.data_layer import get_model_predictions, get_available_rounds, get_tournament_config
from dashboard.components.filters import round_selector

dash.register_page(__name__, path="/pricer", title="Score Pricer", order=4)

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,33,62,0.8)",
    font=dict(color="#e0e0e0"),
    margin=dict(l=50, r=30, t=40, b=40),
)


def _prob_to_american(prob):
    """Convert probability to American odds."""
    if prob <= 0 or prob >= 1:
        return 0
    if prob >= 0.5:
        return int(-prob / (1 - prob) * 100)
    else:
        return int((1 - prob) / prob * 100)


layout = dbc.Container([
    html.H4("Score Pricer", className="page-header"),

    dbc.Row([
        round_selector("pricer", available_rounds=get_available_rounds() or [1, 2, 3, 4]),
        dbc.Col([
            html.Label("Player", className="form-label small text-muted"),
            dcc.Dropdown(id="pricer-player-dropdown", placeholder="Select player...", className="dash-dropdown-dark"),
        ], md=4),
        dbc.Col([
            html.Label("Score Line", className="form-label small text-muted"),
            dbc.Input(id="pricer-score-input", type="number", value=70, step=0.5,
                      className="bg-dark text-light border-secondary"),
        ], md=2),
    ], className="mb-4"),

    # Results
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("UNDER", className="text-muted"),
                    html.H2(id="pricer-under-odds", className="text-success fw-bold"),
                    html.Small(id="pricer-under-prob", className="text-muted"),
                ])
            ], color="dark", outline=True),
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("OVER", className="text-muted"),
                    html.H2(id="pricer-over-odds", className="text-danger fw-bold"),
                    html.Small(id="pricer-over-prob", className="text-muted"),
                ])
            ], color="dark", outline=True),
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Expected Score", className="text-muted"),
                    html.H2(id="pricer-expected", className="fw-bold"),
                    html.Small(id="pricer-std", className="text-muted"),
                ])
            ], color="dark", outline=True),
        ], md=3),
    ], className="mb-4 justify-content-center"),

    # Distribution chart
    dbc.Row([
        dbc.Col(dcc.Graph(id="pricer-dist-chart"), md=8),
        dbc.Col([
            html.H6("Score Line Prices", className="mb-2"),
            html.Div(id="pricer-card-table"),
        ], md=4),
    ]),
], fluid=True)


@callback(
    Output("pricer-player-dropdown", "options"),
    Input("pricer-round-filter", "value"),
)
def update_player_list(round_num):
    if not round_num:
        return []
    df = get_model_predictions(round_num)
    if df.empty:
        return []

    # Find prediction column
    pred_col = f"my_pred{round_num}" if f"my_pred{round_num}" in df.columns else "my_pred"
    if pred_col in df.columns:
        df = df[df[pred_col] > 0.5]  # filter to meaningful players

    players = sorted(df["player_name"].dropna().unique().tolist())
    return [{"label": p.title(), "value": p} for p in players]


@callback(
    Output("pricer-under-odds", "children"),
    Output("pricer-under-prob", "children"),
    Output("pricer-over-odds", "children"),
    Output("pricer-over-prob", "children"),
    Output("pricer-expected", "children"),
    Output("pricer-std", "children"),
    Output("pricer-dist-chart", "figure"),
    Output("pricer-card-table", "children"),
    Input("pricer-round-filter", "value"),
    Input("pricer-player-dropdown", "value"),
    Input("pricer-score-input", "value"),
)
def update_pricer(round_num, player, score_line):
    empty = ("--", "", "--", "", "--", "", go.Figure(layout=PLOT_LAYOUT), "")

    if not round_num or not player or score_line is None:
        return empty

    df = get_model_predictions(round_num)
    if df.empty:
        return empty

    player_row = df[df["player_name"] == player]
    if player_row.empty:
        return empty

    row = player_row.iloc[0]

    # Get expected score: scores_r{N} is strokes gained (relative),
    # so expected_score = course_par - scores_r{N}
    score_col = f"scores_r{round_num}"
    if score_col not in row.index:
        return empty
    sg_value = float(row[score_col])

    config = get_tournament_config()
    course_par = config.get("course_par", config.get("PAR", 72))
    expected = course_par - sg_value

    # Get std_dev — per-player if available, else global
    if "std_dev" in row.index and pd.notna(row["std_dev"]):
        std_dev = float(row["std_dev"])
    else:
        std_dev = config.get("STD_DEV", 2.85)

    score_line = float(score_line)

    # Compute probabilities
    p_under = norm.cdf(score_line, loc=expected, scale=std_dev)
    p_over = 1 - p_under

    under_odds = _prob_to_american(p_under)
    over_odds = _prob_to_american(p_over)

    under_str = f"+{under_odds}" if under_odds > 0 else str(under_odds)
    over_str = f"+{over_odds}" if over_odds > 0 else str(over_odds)

    # Distribution chart
    x = np.linspace(expected - 4 * std_dev, expected + 4 * std_dev, 200)
    y = norm.pdf(x, loc=expected, scale=std_dev)

    fig = go.Figure(layout={**PLOT_LAYOUT, "title": f"{player.title()} — R{round_num} Score Distribution"})

    # Under area (guard against empty slice)
    x_under = x[x <= score_line]
    if len(x_under) > 0:
        y_under = norm.pdf(x_under, loc=expected, scale=std_dev)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_under, [score_line, x_under[0]]]),
            y=np.concatenate([y_under, [0, 0]]),
            fill="toself", fillcolor="rgba(46,125,50,0.3)", line=dict(width=0),
            name=f"Under ({p_under:.1%})", hoverinfo="skip",
        ))

    # Over area (guard against empty slice)
    x_over = x[x >= score_line]
    if len(x_over) > 0:
        y_over = norm.pdf(x_over, loc=expected, scale=std_dev)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_over, [x_over[-1], score_line]]),
            y=np.concatenate([y_over, [0, 0]]),
            fill="toself", fillcolor="rgba(183,28,28,0.3)", line=dict(width=0),
            name=f"Over ({p_over:.1%})", hoverinfo="skip",
        ))

    # Full curve
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", line=dict(color="#4ecca3", width=2),
        name="Distribution", hoverinfo="skip",
    ))

    # Score line
    fig.add_vline(x=score_line, line_dash="dash", line_color="white", annotation_text=f"{score_line}")
    fig.add_vline(x=expected, line_dash="dot", line_color="#4ecca3",
                  annotation_text=f"Exp: {expected:.1f}", annotation_position="top right")

    fig.update_xaxes(title_text="Score")
    fig.update_yaxes(title_text="Density", showticklabels=False)

    # Score card table — standard .5 lines like sportsbooks use
    # P(under 70.5) = P(score <= 70) since scores are integers
    low = int(expected - 3)
    high = int(expected + 3) + 1
    lines = [x + 0.5 for x in range(low, high)]

    card_rows = []
    for line in lines:
        # Discrete: under 70.5 means score <= 70, so use CDF at the .5 boundary
        p_u = norm.cdf(line, loc=expected, scale=std_dev)
        p_o = 1 - p_u
        u_odds = _prob_to_american(p_u)
        o_odds = _prob_to_american(p_o)
        u_str = f"+{u_odds}" if u_odds > 0 else str(u_odds)
        o_str = f"+{o_odds}" if o_odds > 0 else str(o_odds)
        highlight = "table-active" if abs(line - score_line) < 0.01 else ""
        card_rows.append(html.Tr([
            html.Td(f"{line:.1f}"),
            html.Td(u_str),
            html.Td(o_str),
        ], className=highlight))

    card_table = dbc.Table([
        html.Thead(html.Tr([html.Th("Line"), html.Th("Under"), html.Th("Over")])),
        html.Tbody(card_rows),
    ], bordered=True, color="dark", hover=True, size="sm", striped=True)

    return (
        under_str, f"({p_under:.1%})",
        over_str, f"({p_over:.1%})",
        f"{expected:.1f}", f"Std Dev: {std_dev:.2f}",
        fig, card_table,
    )
