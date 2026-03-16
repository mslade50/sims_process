"""Historical Distributions — Browse archived finish position distributions from past events."""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dashboard.data_layer import get_historical_events, get_historical_rank_probs

dash.register_page(__name__, path="/historical-distributions", title="Historical Distributions", order=7)

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,33,62,0.8)",
    font=dict(color="#e0e0e0"),
    margin=dict(l=50, r=30, t=60, b=40),
)

THRESHOLDS = [
    (1.5, "Win", "gold"),
    (5.5, "T5", "#2ca02c"),
    (10.5, "T10", "#ff7f0e"),
    (20.5, "T20", "#d62728"),
]

COMPARE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def _prob_to_american(prob):
    """Convert probability to American odds string."""
    if prob <= 0 or prob >= 1:
        return "N/A"
    if prob >= 0.5:
        return f"{int(-prob / (1 - prob) * 100):+d}"
    else:
        return f"+{int((1 - prob) / prob * 100)}"


def _compute_stats(df):
    """Compute Win%, T5%, T10%, T20% from rank_probs for a single player."""
    win = df.loc[df["rank"] == 1, "prob_u"].sum() * 100
    t5 = df.loc[df["rank"] <= 5, "prob_u"].sum() * 100
    t10 = df.loc[df["rank"] <= 10, "prob_u"].sum() * 100
    t20 = df.loc[df["rank"] <= 20, "prob_u"].sum() * 100
    return win, t5, t10, t20


def _make_figure(rank_df, players, max_rank):
    """Build a Plotly bar chart for one or more players."""
    fig = go.Figure()
    multi = len(players) > 1
    opacity = 0.5 if multi else 0.75

    for i, player in enumerate(players):
        pdf = rank_df[rank_df["player_name"] == player].copy()
        if pdf.empty:
            continue

        full_ranks = pd.DataFrame({"rank": np.arange(1, max_rank + 1)})
        pdf = full_ranks.merge(pdf[["rank", "prob_u"]], on="rank", how="left").fillna(0)

        color = COMPARE_COLORS[i % len(COMPARE_COLORS)] if multi else "steelblue"

        fig.add_trace(go.Bar(
            x=pdf["rank"],
            y=pdf["prob_u"] * 100,
            name=player.title(),
            marker_color=color,
            opacity=opacity,
        ))

    for x_pos, label, color in THRESHOLDS:
        if x_pos <= max_rank:
            fig.add_vline(x=x_pos, line_dash="dash", line_color=color, line_width=1.5)
            fig.add_annotation(
                x=x_pos, y=1.02, yref="paper", text=label,
                showarrow=False, font=dict(color=color, size=11),
            )

    if len(players) == 1 and not rank_df[rank_df["player_name"] == players[0]].empty:
        pdf = rank_df[rank_df["player_name"] == players[0]]
        win, t5, t10, t20 = _compute_stats(pdf)
        title_text = (
            f"<b>{players[0].title()}</b><br>"
            f"<sup>Win: {win:.2f}% | T5: {t5:.1f}% | T10: {t10:.1f}% | T20: {t20:.1f}%</sup>"
        )
    elif len(players) > 1:
        title_text = " vs ".join(p.title() for p in players)
    else:
        title_text = "Select a player"

    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text=title_text, x=0.5, xanchor="center"),
        xaxis_title="Finish Position",
        yaxis_title="Probability (%)",
        xaxis=dict(range=[0, min(max_rank + 1, 80)], dtick=5),
        bargap=0.1,
        barmode="overlay" if multi else "relative",
        showlegend=multi,
        legend=dict(x=0.85, y=0.95, bgcolor="rgba(0,0,0,0.5)"),
    )

    return fig


# ── Layout ───────────────────────────────────────────────────────────────────

def _build_event_options():
    events = get_historical_events()
    return [{"label": f"{tourney.replace('_', ' ').title()} ({eid})", "value": f"{eid}_{tourney}"} for eid, tourney in events]


layout = dbc.Container([
    html.H4("Historical Finish Distributions", className="page-header"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Event"),
            dcc.Dropdown(id="hdist-event", options=_build_event_options(), placeholder="Select event...", className="mb-2"),
        ], md=4),
        dbc.Col([
            dbc.Label("Data Source"),
            dbc.RadioItems(
                id="hdist-mode",
                options=[
                    {"label": "Pre-Tournament", "value": "pre"},
                    {"label": "Live", "value": "live"},
                ],
                value="pre",
                inline=True,
                className="mb-2",
            ),
        ], md=2),
        dbc.Col([
            dbc.Label("Player"),
            dcc.Dropdown(id="hdist-player", placeholder="Select player...", className="mb-2"),
        ], md=3),
        dbc.Col([
            dbc.Label("Compare (optional)"),
            dcc.Dropdown(
                id="hdist-compare", placeholder="Add players to compare...",
                multi=True, className="mb-2",
            ),
        ], md=3),
    ], className="mb-3"),

    dcc.Graph(id="hdist-chart", style={"height": "550px"}),

    html.Hr(),

    # ── Finish Position Pricer ───────────────────────────────────────────
    html.H5("Finish Position Pricer", className="mt-3 mb-2"),
    dbc.Row([
        dbc.Col([
            dbc.Label("Position"),
            dbc.Input(id="hdist-pos", type="number", min=1, max=160, step=1, value=10, className="mb-2"),
        ], md=2),
        dbc.Col([
            dbc.Label("Side"),
            dbc.RadioItems(
                id="hdist-side",
                options=[
                    {"label": "Better (\u2264)", "value": "better"},
                    {"label": "Worse (\u2265)", "value": "worse"},
                ],
                value="better",
                inline=True,
                className="mb-2",
            ),
        ], md=3),
        dbc.Col([
            html.Div(id="hdist-pricer-output", className="mt-4"),
        ], md=7),
    ], className="mb-4"),
], fluid=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_event_value(value):
    """Parse '{event_id}_{tourney}' dropdown value into (event_id, tourney)."""
    if not value:
        return None, None
    parts = value.split("_", 1)
    if len(parts) != 2:
        return None, None
    try:
        return int(parts[0]), parts[1]
    except ValueError:
        return None, None


def _load_hist_data(event_value, mode):
    """Load historical rank probs for the selected event and mode."""
    event_id, tourney = _parse_event_value(event_value)
    if event_id is None:
        return pd.DataFrame()
    return get_historical_rank_probs(event_id, tourney, mode)


# ── Callbacks ────────────────────────────────────────────────────────────────

@callback(
    Output("hdist-player", "options"),
    Output("hdist-player", "value"),
    Output("hdist-compare", "options"),
    Input("hdist-event", "value"),
    Input("hdist-mode", "value"),
)
def populate_players(event_value, mode):
    df = _load_hist_data(event_value, mode)
    if df.empty:
        return [], None, []

    win_probs = df[df["rank"] == 1].groupby("player_name")["prob_u"].sum().sort_values(ascending=False)
    sorted_players = win_probs.index.tolist()

    all_players = df["player_name"].unique().tolist()
    remaining = [p for p in sorted(all_players) if p not in sorted_players]
    sorted_players.extend(remaining)

    options = [{"label": p.title(), "value": p} for p in sorted_players]
    default = sorted_players[0] if sorted_players else None

    return options, default, options


@callback(
    Output("hdist-chart", "figure"),
    Input("hdist-player", "value"),
    Input("hdist-compare", "value"),
    Input("hdist-event", "value"),
    Input("hdist-mode", "value"),
)
def update_chart(player, compare_players, event_value, mode):
    if not player or not event_value:
        fig = go.Figure()
        fig.update_layout(**PLOT_LAYOUT, title="Select an event and player")
        return fig

    df = _load_hist_data(event_value, mode)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**PLOT_LAYOUT, title="No distribution data available")
        return fig

    max_rank = int(df["rank"].max())

    players = [player]
    if compare_players:
        players.extend([p for p in compare_players if p != player])

    return _make_figure(df, players, max_rank)


@callback(
    Output("hdist-pricer-output", "children"),
    Input("hdist-player", "value"),
    Input("hdist-pos", "value"),
    Input("hdist-side", "value"),
    Input("hdist-event", "value"),
    Input("hdist-mode", "value"),
)
def update_pricer(player, pos, side, event_value, mode):
    if not player or not pos or not event_value:
        return ""

    df = _load_hist_data(event_value, mode)
    if df.empty:
        return ""

    pdf = df[df["player_name"] == player]
    if pdf.empty:
        return ""

    pos = int(pos)
    if side == "better":
        prob = pdf.loc[pdf["rank"] <= pos, "prob_u"].sum()
        label = f"{pos}th or better (\u2264 {pos})"
    else:
        prob = pdf.loc[pdf["rank"] >= pos, "prob_u"].sum()
        label = f"{pos}th or worse (\u2265 {pos})"

    pct = prob * 100
    american = _prob_to_american(prob) if prob > 0 else "N/A"
    decimal_odds = f"{1 / prob:.2f}" if prob > 0 else "N/A"

    return dbc.Card(
        dbc.CardBody([
            html.Span(f"{player.title()}: ", style={"fontWeight": "bold"}),
            html.Span(f"{label}  "),
            html.Span(f"{pct:.1f}%", style={"color": "gold", "fontWeight": "bold", "fontSize": "1.1em"}),
            html.Span(f"  |  {american}  |  {decimal_odds} dec", style={"color": "#aaa"}),
        ]),
        color="dark",
        className="p-2",
    )
