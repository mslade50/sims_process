"""Distributions — V2 SG category distributions with per-player skew."""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from dashboard.data_layer import get_v2_dists

dash.register_page(__name__, path="/distributions", title="Distributions", order=5)

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,33,62,0.8)",
    font=dict(color="#e0e0e0"),
    margin=dict(l=50, r=30, t=60, b=50),
)

CAT_ORDER = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
CAT_LABELS = {"sg_ott": "OTT", "sg_app": "APP", "sg_arg": "ARG", "sg_putt": "PUTT"}
CAT_COLORS = {"sg_ott": "#1f77b4", "sg_app": "#2ca02c", "sg_arg": "#ff7f0e", "sg_putt": "#d62728"}

COMPARE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def _cf_skew_pdf(x, mu, sigma, gamma):
    """Approximate PDF under Cornish-Fisher skew transform of a normal."""
    if sigma <= 0:
        return np.zeros_like(x)
    z = (x - mu) / sigma
    # Standard normal PDF
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi) / sigma
    if abs(gamma) < 0.01:
        return phi
    # Gram-Charlier type A expansion (skewness term)
    he3 = z ** 3 - 3 * z
    return phi * (1 + (gamma / 6.0) * he3)


def _make_category_figure(dists_df, players):
    """Build 2x2 subplot with SG category PDFs for one or more players."""
    multi = len(players) > 1
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[CAT_LABELS[c] for c in CAT_ORDER],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    for pi, player in enumerate(players):
        pdf = dists_df[dists_df["player_name"] == player]
        if pdf.empty:
            continue
        color = COMPARE_COLORS[pi % len(COMPARE_COLORS)] if multi else None

        for ci, cat in enumerate(CAT_ORDER):
            row, col = divmod(ci, 2)
            row += 1
            col += 1
            cat_row = pdf[pdf["category_clean"] == cat]
            if cat_row.empty:
                continue
            r = cat_row.iloc[0]
            mu, sigma, skew_val = float(r["mean"]), float(r["std"]), float(r["skew"])

            x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
            y = _cf_skew_pdf(x, mu, sigma, skew_val)
            y = np.clip(y, 0, None)  # ensure non-negative

            c = color or CAT_COLORS[cat]
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines",
                name=f"{player.title()}" if ci == 0 else None,
                legendgroup=player,
                showlegend=(ci == 0),
                line=dict(color=c, width=2),
                opacity=0.7 if multi else 0.9,
                fill="tozeroy" if not multi else None,
                fillcolor=f"rgba{tuple(list(int(c.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + [0.15])}" if not multi else None,
            ), row=row, col=col)

            # Mean line
            fig.add_vline(
                x=mu, line_dash="dash", line_color=c, line_width=1,
                row=row, col=col,
            )

            # Annotation with stats (single player only)
            if not multi:
                n_eff = float(r["n_eff"]) if pd.notna(r["n_eff"]) else 0
                fig.add_annotation(
                    x=0.98, y=0.95, xref=f"x{ci+1 if ci > 0 else ''} domain",
                    yref=f"y{ci+1 if ci > 0 else ''} domain",
                    text=f"<b>{mu:+.2f}</b> | std {sigma:.2f} | skew {skew_val:+.2f} | n_eff {n_eff:.0f}",
                    showarrow=False, font=dict(size=10, color="#aaa"),
                    xanchor="right", yanchor="top",
                    bgcolor="rgba(0,0,0,0.5)",
                )

    title = (
        f"<b>{players[0].title()}</b> — SG Category Distributions"
        if len(players) == 1
        else " vs ".join(p.title() for p in players)
    )

    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text=title, x=0.5, xanchor="center"),
        showlegend=multi,
        legend=dict(x=0.85, y=1.05, bgcolor="rgba(0,0,0,0.5)"),
    )
    for ci in range(4):
        ax = f"xaxis{ci+1 if ci > 0 else ''}"
        fig.update_layout(**{ax: dict(title="Strokes Gained")})

    return fig


def _make_stats_table(dists_df, player):
    """Build a stats summary table for a single player."""
    pdf = dists_df[dists_df["player_name"] == player]
    if pdf.empty:
        return dbc.Alert("No distribution data for this player.", color="warning")

    rows = []
    for cat in CAT_ORDER:
        cat_row = pdf[pdf["category_clean"] == cat]
        if cat_row.empty:
            rows.append([CAT_LABELS[cat], "-", "-", "-", "-", "-", "-"])
            continue
        r = cat_row.iloc[0]
        rows.append([
            CAT_LABELS[cat],
            f"{r['mean']:+.3f}",
            f"{r['std']:.3f}",
            f"{r['skew']:+.3f}",
            f"{r['excess_kurtosis']:+.3f}",
            f"{int(r['n'])}",
            f"{r['n_eff']:.1f}",
        ])

    # Total row
    total_mu = sum(float(pdf[pdf["category_clean"] == c].iloc[0]["mean"])
                   for c in CAT_ORDER if not pdf[pdf["category_clean"] == c].empty)
    rows.append(["TOTAL", f"{total_mu:+.3f}", "", "", "", "", ""])

    return dbc.Table(
        [html.Thead(html.Tr([html.Th(h) for h in
            ["Category", "Mean", "Std", "Skew", "Ex. Kurt", "N", "N_eff"]]))] +
        [html.Tbody([html.Tr([html.Td(c) for c in row]) for row in rows])],
        bordered=True, hover=True, size="sm", color="dark",
    )


# -- Layout ---------------------------------------------------------------

layout = dbc.Container([
    html.H4("V2 SG Category Distributions", className="page-header"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Player"),
            dcc.Dropdown(id="dist-player", placeholder="Select player...", className="mb-2"),
        ], md=4),
        dbc.Col([
            dbc.Label("Compare (optional)"),
            dcc.Dropdown(
                id="dist-compare", placeholder="Add players to compare...",
                multi=True, className="mb-2",
            ),
        ], md=5),
    ], className="mb-3"),

    dcc.Graph(id="dist-chart", style={"height": "550px"}),

    html.Div(id="dist-stats-table", className="mt-3"),
], fluid=True)


# -- Callbacks -------------------------------------------------------------

@callback(
    Output("dist-player", "options"),
    Output("dist-player", "value"),
    Output("dist-compare", "options"),
    Input("dist-player", "id"),  # fires once on page load
)
def populate_players(_):
    df = get_v2_dists()
    if df.empty:
        return [], None, []

    # Sort by total mean (sum of 4 categories) descending
    totals = df.groupby("player_name")["mean"].sum().sort_values(ascending=False)
    sorted_players = totals.index.tolist()

    options = [{"label": p.title(), "value": p} for p in sorted_players]
    default = sorted_players[0] if sorted_players else None

    return options, default, options


@callback(
    Output("dist-chart", "figure"),
    Output("dist-stats-table", "children"),
    Input("dist-player", "value"),
    Input("dist-compare", "value"),
)
def update_chart(player, compare_players):
    if not player:
        fig = go.Figure()
        fig.update_layout(**PLOT_LAYOUT, title="Select a player to view distribution")
        return fig, ""

    df = get_v2_dists()
    if df.empty:
        fig = go.Figure()
        fig.update_layout(**PLOT_LAYOUT, title="No V2 distribution data available")
        return fig, dbc.Alert("this_week_dists_v2.csv not found. Run cat_dists_player.py first.", color="danger")

    players = [player]
    if compare_players:
        players.extend([p for p in compare_players if p != player])

    fig = _make_category_figure(df, players)
    table = _make_stats_table(df, player) if len(players) == 1 else ""

    return fig, table
