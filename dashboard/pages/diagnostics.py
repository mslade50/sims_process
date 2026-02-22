"""Diagnostics page — SG prediction analysis and archetypes."""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dashboard.data_layer import get_sg_diagnostics
from dashboard.components.tables import make_grid

dash.register_page(__name__, path="/diagnostics", title="Diagnostics", order=8)

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,33,62,0.8)",
    font=dict(color="#e0e0e0"),
    margin=dict(l=50, r=30, t=40, b=40),
)

SG_CATEGORIES = ["ott", "app", "arg", "putt"]


layout = dbc.Container([
    html.H4("SG Diagnostics", className="page-header"),

    dbc.Row([
        dbc.Col([
            html.Label("Event", className="form-label small text-muted"),
            dcc.Dropdown(id="diag-event-filter", placeholder="All events", className="dash-dropdown-dark"),
        ], md=3),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="diag-bias-chart"), md=6),
        dbc.Col(dcc.Graph(id="diag-archetype-heatmap"), md=6),
    ]),

    html.H5("Biggest Misses", className="mt-4 mb-2"),
    html.Div(id="diag-misses-table"),

    html.H5("Recurring Misses (Cross-Event)", className="mt-4 mb-2"),
    html.Div(id="diag-recurring-table"),

    html.H5("Player Deep Dive", className="mt-4 mb-2"),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(id="diag-player-dropdown", placeholder="Select player...", className="dash-dropdown-dark"),
        ], md=3),
    ], className="mb-2"),
    html.Div(id="diag-player-detail"),
], fluid=True)


@callback(
    Output("diag-event-filter", "options"),
    Input("diag-event-filter", "id"),
)
def populate_events(_):
    df = get_sg_diagnostics()
    if df.empty or "event_id" not in df.columns:
        return []
    events = df["event_id"].dropna().unique().tolist()
    # Try to get event names if available
    if "event_name" in df.columns:
        event_map = df.drop_duplicates("event_id").set_index("event_id")["event_name"].to_dict()
        return [{"label": str(event_map.get(e, e)), "value": e} for e in sorted(events, key=str)]
    return [{"label": str(e), "value": e} for e in sorted(events, key=str)]


@callback(
    Output("diag-bias-chart", "figure"),
    Output("diag-archetype-heatmap", "figure"),
    Output("diag-misses-table", "children"),
    Output("diag-recurring-table", "children"),
    Output("diag-player-dropdown", "options"),
    Input("diag-event-filter", "value"),
)
def update_diagnostics(event_id):
    empty_fig = go.Figure(layout=PLOT_LAYOUT)

    df = get_sg_diagnostics()
    if df.empty:
        alert = dbc.Alert("No SG diagnostic data. Run sg_diagnostic.py after a ShotLink event.", color="warning")
        return empty_fig, empty_fig, alert, alert, []

    if event_id:
        df = df[df["event_id"] == event_id]

    if df.empty:
        alert = dbc.Alert("No data for this event.", color="info")
        return empty_fig, empty_fig, alert, alert, []

    # Player options
    players = sorted(df["player_name"].dropna().unique().tolist())
    player_options = [{"label": p.title(), "value": p} for p in players]

    # ── Bias chart: avg miss per category ──
    miss_col = "miss" if "miss" in df.columns else None
    pred_col = next((c for c in ["predicted_sg", "predicted"] if c in df.columns), None)
    actual_col = next((c for c in ["actual_sg", "actual"] if c in df.columns), None)

    bias_fig = go.Figure(layout={**PLOT_LAYOUT, "title": "Average Prediction Bias by Category"})

    if miss_col and "category" in df.columns:
        bias = df.groupby("category")[miss_col].mean()
        cats = [c for c in SG_CATEGORIES if c in bias.index]
        if cats:
            values = [bias[c] for c in cats]
            colors = ["#2e7d32" if v >= 0 else "#b71c1c" for v in values]
            bias_fig.add_trace(go.Bar(
                x=[c.upper() for c in cats],
                y=values,
                marker_color=colors,
                text=[f"{v:+.3f}" for v in values],
                textposition="outside",
            ))
            bias_fig.update_yaxes(title_text="Avg Miss (Predicted - Actual)")
    elif pred_col and actual_col and "category" in df.columns:
        df["_miss"] = df[pred_col] - df[actual_col]
        bias = df.groupby("category")["_miss"].mean()
        cats = [c for c in SG_CATEGORIES if c in bias.index]
        if cats:
            values = [bias[c] for c in cats]
            colors = ["#2e7d32" if v >= 0 else "#b71c1c" for v in values]
            bias_fig.add_trace(go.Bar(
                x=[c.upper() for c in cats],
                y=values,
                marker_color=colors,
                text=[f"{v:+.3f}" for v in values],
                textposition="outside",
            ))
            bias_fig.update_yaxes(title_text="Avg Miss (Predicted - Actual)")

    # ── Archetype heatmap ──
    arch_fig = go.Figure(layout={**PLOT_LAYOUT, "title": "Miss Magnitude by Archetype"})
    if "archetype" in df.columns and "category" in df.columns:
        miss_source = miss_col or "_miss"
        if miss_source in df.columns:
            arch_pivot = df.pivot_table(
                values=miss_source, index="archetype", columns="category",
                aggfunc=lambda x: np.mean(np.abs(x)),
            )
            cats_available = [c for c in SG_CATEGORIES if c in arch_pivot.columns]
            if cats_available and not arch_pivot.empty:
                arch_fig = go.Figure(data=go.Heatmap(
                    z=arch_pivot[cats_available].values,
                    x=[c.upper() for c in cats_available],
                    y=arch_pivot.index.tolist(),
                    colorscale="YlOrRd",
                    text=np.round(arch_pivot[cats_available].values, 3),
                    texttemplate="%{text}",
                    hovertemplate="<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>",
                ))
                arch_fig.update_layout(**PLOT_LAYOUT, title="Avg |Miss| by Archetype & Category")

    # ── Biggest misses table ──
    miss_source = miss_col or ("_miss" if "_miss" in df.columns else None)
    if miss_source and miss_source in df.columns:
        df["abs_miss"] = df[miss_source].abs()
        top_misses = df.nlargest(15, "abs_miss")
        miss_display_cols = ["player_name", "category", miss_source, "abs_miss"]
        if "round" in top_misses.columns:
            miss_display_cols.insert(1, "round")
        if "archetype" in top_misses.columns:
            miss_display_cols.append("archetype")
        available = [c for c in miss_display_cols if c in top_misses.columns]
        misses_content = make_grid(top_misses[available], id_suffix="misses", height=400)
    else:
        misses_content = dbc.Alert("Miss column not found in diagnostic data.", color="info")

    # ── Recurring misses ──
    if miss_source and miss_source in df.columns and "event_id" in df.columns:
        all_diag = get_sg_diagnostics()  # unfiltered
        if miss_source not in all_diag.columns and pred_col and actual_col:
            all_diag["_miss"] = all_diag[pred_col] - all_diag[actual_col]
            miss_source = "_miss"

        if miss_source in all_diag.columns:
            player_event = (
                all_diag.groupby(["player_name", "event_id"])[miss_source]
                .mean().reset_index()
            )
            # Players with consistent direction across 2+ events
            player_events = player_event.groupby("player_name").agg(
                n_events=("event_id", "nunique"),
                avg_miss=(miss_source, "mean"),
                all_positive=(miss_source, lambda x: (x > 0).all()),
                all_negative=(miss_source, lambda x: (x < 0).all()),
            ).reset_index()
            recurring = player_events[
                (player_events["n_events"] >= 2) &
                (player_events["all_positive"] | player_events["all_negative"])
            ].sort_values("avg_miss", key=abs, ascending=False)

            if recurring.empty:
                recurring_content = dbc.Alert("No recurring directional misses found.", color="info")
            else:
                recurring_content = make_grid(
                    recurring[["player_name", "n_events", "avg_miss"]],
                    id_suffix="recurring", height=300,
                )
        else:
            recurring_content = dbc.Alert("Cannot compute recurring misses.", color="info")
    else:
        recurring_content = dbc.Alert("Insufficient data for recurring miss analysis.", color="info")

    return bias_fig, arch_fig, misses_content, recurring_content, player_options


@callback(
    Output("diag-player-detail", "children"),
    Input("diag-player-dropdown", "value"),
    Input("diag-event-filter", "value"),
)
def update_player_detail(player, event_id):
    if not player:
        return dbc.Alert("Select a player above to see details.", color="secondary")

    df = get_sg_diagnostics()
    if df.empty:
        return dbc.Alert("No data.", color="warning")

    if event_id:
        df = df[df["event_id"] == event_id]

    player_df = df[df["player_name"] == player]
    if player_df.empty:
        return dbc.Alert(f"No data for {player}.", color="info")

    # Show per-round, per-category predicted vs actual
    pred_col = next((c for c in ["predicted_sg", "predicted"] if c in player_df.columns), None)
    actual_col = next((c for c in ["actual_sg", "actual"] if c in player_df.columns), None)

    display_cols = ["player_name", "round", "category"]
    if pred_col:
        display_cols.append(pred_col)
    if actual_col:
        display_cols.append(actual_col)
    if "miss" in player_df.columns:
        display_cols.append("miss")
    if "archetype" in player_df.columns:
        display_cols.append("archetype")

    available = [c for c in display_cols if c in player_df.columns]
    return make_grid(player_df[available], id_suffix="player-detail", height=300)
