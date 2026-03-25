"""Round Score Distributions — Visualize simulated round score distributions."""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dashboard.data_layer import get_round_score_probs
from dashboard.components.tables import make_grid

dash.register_page(__name__, path="/round-distributions", title="Round Distributions", order=7)

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,33,62,0.8)",
    font=dict(color="#e0e0e0"),
    margin=dict(l=50, r=30, t=60, b=40),
)

COMPARE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]

CAT_LABELS = {"sg_ott": "OTT", "sg_app": "APP", "sg_arg": "ARG", "sg_putt": "PUTT"}


def _prob_to_american(prob):
    """Convert implied probability to American odds string."""
    if prob <= 0 or prob >= 1:
        return "—"
    if prob >= 0.5:
        return f"{int(round(-prob / (1 - prob) * 100))}"
    else:
        return f"+{int(round((1 - prob) / prob * 100))}"


def _available_rounds():
    """Return list of round numbers that have score prob data."""
    rounds = []
    for r in range(1, 5):
        df = get_round_score_probs(r)
        if not df.empty:
            rounds.append(r)
    return rounds


def _load_players(round_num):
    """Return sorted player list from score probs (by expected score, ascending)."""
    df = get_round_score_probs(round_num)
    if df.empty:
        return []

    # Compute expected score per player for sorting
    df["ev"] = df["score"] * df["prob"]
    ev_map = df.groupby("player_name")["ev"].sum()

    players = list(df["player_name"].unique())
    players.sort(key=lambda p: (ev_map.get(p, 99), p))  # lowest expected score first
    return players


# ── Layout ───────────────────────────────────────────────────────────────────

layout = dbc.Container([
    html.H4("Round Score Distributions", className="page-header"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Round"),
            dcc.Dropdown(id="rddist-round", placeholder="Select round...", className="mb-2"),
        ], md=2),
        dbc.Col([
            dbc.Label("Players"),
            dcc.Dropdown(id="rddist-players", placeholder="Select players...", multi=True, className="mb-2"),
        ], md=6),
    ], className="mb-3"),

    html.Div(id="rddist-warning"),

    dcc.Graph(id="rddist-chart", style={"height": "550px"}),

    # Matchup probability (shown when exactly 2 players selected)
    html.Div(id="rddist-matchup"),

    html.H5("Summary Statistics", className="mt-3 mb-2"),
    html.Div(id="rddist-stats-table"),
], fluid=True)


# ── Callbacks ────────────────────────────────────────────────────────────────

@callback(
    Output("rddist-round", "options"),
    Output("rddist-round", "value"),
    Input("rddist-round", "id"),  # fires once on load
)
def populate_rounds(_):
    rounds = _available_rounds()
    if not rounds:
        return [], None
    options = [{"label": f"Round {r}", "value": r} for r in rounds]
    return options, rounds[0]


@callback(
    Output("rddist-players", "options"),
    Output("rddist-players", "value"),
    Input("rddist-round", "value"),
)
def populate_players(round_num):
    if not round_num:
        return [], []
    players = _load_players(round_num)
    if not players:
        return [], []
    options = [{"label": p.title(), "value": p} for p in players]
    default = players[:2] if len(players) >= 2 else players[:1]
    return options, default


@callback(
    Output("rddist-chart", "figure"),
    Output("rddist-stats-table", "children"),
    Output("rddist-matchup", "children"),
    Output("rddist-warning", "children"),
    Input("rddist-round", "value"),
    Input("rddist-players", "value"),
)
def update_chart(round_num, selected_players):
    empty_fig = go.Figure()
    empty_fig.update_layout(**PLOT_LAYOUT, title="Select a round and players")

    if not round_num:
        return empty_fig, None, None, None

    df = get_round_score_probs(round_num)
    if df.empty:
        return empty_fig, None, None, dbc.Alert(
            f"No score distribution data for round {round_num}.", color="warning",
        )

    if not selected_players:
        return empty_fig, None, None, None

    expected_avg = df["expected_avg"].iloc[0] if "expected_avg" in df.columns else None
    fig = go.Figure()
    stats_rows = []
    player_probs = {}  # player -> (scores array, probs array) for matchup calc

    for i, player in enumerate(selected_players):
        pdf = df[df["player_name"] == player]
        if pdf.empty:
            continue

        scores = pdf["score"].values
        probs = pdf["prob"].values
        player_probs[player] = (scores, probs)

        color = COMPARE_COLORS[i % len(COMPARE_COLORS)]
        mean_score = float(np.sum(scores * probs))

        # Bar chart of score probabilities (width=1 removes gaps between integer scores)
        fig.add_trace(go.Bar(
            x=scores,
            y=probs,
            name=player.title(),
            marker_color=color,
            opacity=0.55,
            width=1.0,
            hovertemplate="Score: %{x}<br>Prob: %{y:.3f}<extra></extra>",
        ))

        # Mean line (no annotation — overlapping labels are messy)
        fig.add_vline(
            x=mean_score, line_dash="dash", line_color=color, line_width=2,
        )

        # Stats row
        var = float(np.sum(probs * (scores - mean_score) ** 2))
        std = var ** 0.5

        row_data = {
            "Player": player.title(),
            "Mean": round(mean_score, 2),
            "Std": round(std, 2),
        }

        # Add category SG means if available
        cat_row = pdf.iloc[0]
        for cat in ["sg_ott", "sg_app", "sg_arg", "sg_putt"]:
            val = cat_row.get(cat)
            if val is not None and not pd.isna(val):
                row_data[CAT_LABELS[cat]] = round(float(val), 3)

        stats_rows.append(row_data)

    title = f"R{round_num} Simulated Scores — " + ", ".join(p.title() for p in selected_players)
    fig.update_layout(
        **PLOT_LAYOUT,
        barmode="overlay",
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title="Score",
        yaxis_title="Probability",
        showlegend=True,
        legend=dict(x=0.85, y=0.98, bgcolor="rgba(0,0,0,0.5)"),
        height=550,
    )

    # Stats table
    stats_df = pd.DataFrame(stats_rows) if stats_rows else pd.DataFrame()
    stats_table = make_grid(stats_df, id_suffix="rddist-stats", height=200, page_size=10) if not stats_df.empty else None

    # Matchup probability (exactly 2 players)
    matchup_div = None
    if len(player_probs) == 2:
        players_list = list(player_probs.keys())
        a, b = players_list[0], players_list[1]
        sa, pa = player_probs[a]
        sb, pb = player_probs[b]

        # Compute P(A < B), P(A == B), P(A > B) from joint distribution
        a_wins = 0.0
        ties = 0.0
        for si, pi in zip(sa, pa):
            for sj, pj in zip(sb, pb):
                joint = pi * pj
                if si < sj:
                    a_wins += joint
                elif si == sj:
                    ties += joint
        b_wins = 1.0 - a_wins - ties

        # No-tie probabilities for American odds (dead-heat removed)
        a_no_tie = a_wins / (a_wins + b_wins) if (a_wins + b_wins) > 0 else 0.5
        b_no_tie = 1.0 - a_no_tie

        matchup_div = dbc.Card(
            dbc.CardBody([
                html.H5("Head-to-Head Matchup", className="card-title"),
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Strong(a.title(), style={"color": COMPARE_COLORS[0]}),
                        html.Span(f"  {a_wins * 100:.1f}%", className="ms-2"),
                        html.Span(f"  ({_prob_to_american(a_no_tie)})", className="ms-1",
                                  style={"color": "#aaa"}),
                    ]), md=5),
                    dbc.Col(html.Div([
                        html.Strong("Tie"),
                        html.Span(f"  {ties * 100:.1f}%", className="ms-2"),
                    ]), md=2, className="text-center"),
                    dbc.Col(html.Div([
                        html.Strong(b.title(), style={"color": COMPARE_COLORS[1]}),
                        html.Span(f"  {b_wins * 100:.1f}%", className="ms-2"),
                        html.Span(f"  ({_prob_to_american(b_no_tie)})", className="ms-1",
                                  style={"color": "#aaa"}),
                    ]), md=5, className="text-end"),
                ]),
            ]),
            className="mt-3 mb-2",
            color="dark",
        )

    return fig, stats_table, matchup_div, None
