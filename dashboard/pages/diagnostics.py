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
        dbc.Col(dcc.Graph(id="diag-bias-chart"), md=4),
        dbc.Col(dcc.Graph(id="diag-directional-heatmap"), md=4),
        dbc.Col(dcc.Graph(id="diag-archetype-heatmap"), md=4),
    ]),

    html.H5("Player Explorer", className="mt-4 mb-2"),
    dbc.Row([
        dbc.Col([
            html.Label("View", className="form-label small text-muted"),
            dcc.Dropdown(
                id="diag-archetype-filter",
                placeholder="Biggest Misses (default)",
                className="dash-dropdown-dark",
            ),
        ], md=3),
    ], className="mb-2"),
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
    Output("diag-directional-heatmap", "figure"),
    Output("diag-archetype-heatmap", "figure"),
    Output("diag-archetype-filter", "options"),
    Output("diag-recurring-table", "children"),
    Output("diag-player-dropdown", "options"),
    Input("diag-event-filter", "value"),
)
def update_diagnostics(event_id):
    empty_fig = go.Figure(layout=PLOT_LAYOUT)

    df = get_sg_diagnostics()
    if df.empty:
        alert = dbc.Alert("No SG diagnostic data. Run sg_diagnostic.py after a ShotLink event.", color="warning")
        return empty_fig, empty_fig, empty_fig, [], alert, []

    if event_id:
        df = df[df["event_id"] == event_id]

    if df.empty:
        alert = dbc.Alert("No data for this event.", color="info")
        return empty_fig, empty_fig, empty_fig, [], alert, []

    # Field-center miss to remove systematic field-strength bias
    if "miss_centered" not in df.columns and "miss" in df.columns:
        df["miss_centered"] = df.groupby(["round", "category"])["miss"].transform(
            lambda x: x - x.mean()
        )

    # Player options
    players = sorted(df["player_name"].dropna().unique().tolist())
    player_options = [{"label": p.title(), "value": p} for p in players]

    miss_col = "miss_centered" if "miss_centered" in df.columns else ("miss" if "miss" in df.columns else None)

    # ── Archetype bias bar chart (field-centered total miss by archetype) ──
    bias_fig = go.Figure(layout={**PLOT_LAYOUT, "title": "Avg Bias by Archetype", "height": 350})

    if miss_col and "archetype" in df.columns and "category" in df.columns:
        total_df = df[df["category"] == "total"]
        if not total_df.empty:
            arch_bias = total_df.groupby("archetype")[miss_col].mean().sort_values()
            archetypes = arch_bias.index.tolist()
            values = arch_bias.values.tolist()
            colors = ["#2e7d32" if v >= 0 else "#b71c1c" for v in values]
            bias_fig.add_trace(go.Bar(
                x=archetypes,
                y=values,
                marker_color=colors,
                text=[f"{v:+.3f}" for v in values],
                textposition="outside",
            ))
            bias_fig.update_yaxes(title_text="Avg Centered Miss (Total SG)")
            bias_fig.update_xaxes(tickangle=-30)

    # ── Directional heatmap (avg miss — shows systematic over/under by archetype x category) ──
    dir_fig = go.Figure(layout={**PLOT_LAYOUT, "title": "Directional Bias by Archetype", "height": 350})
    if miss_col and "archetype" in df.columns and "category" in df.columns:
        dir_pivot = df.pivot_table(
            values=miss_col, index="archetype", columns="category",
            aggfunc="mean",
        )
        cats_available = [c for c in SG_CATEGORIES if c in dir_pivot.columns]
        if cats_available and not dir_pivot.empty:
            z_vals = dir_pivot[cats_available].values
            dir_fig = go.Figure(data=go.Heatmap(
                z=z_vals,
                x=[c.upper() for c in cats_available],
                y=dir_pivot.index.tolist(),
                colorscale="RdBu",
                zmid=0,
                text=np.round(z_vals, 3),
                texttemplate="%{text}",
                hovertemplate="<b>%{y}</b><br>%{x}: %{z:+.3f}<extra></extra>",
            ))
            dir_fig.update_layout(**PLOT_LAYOUT, title="Avg Miss by Archetype (+ = under, - = over)")

    # ── Absolute heatmap (avg |miss| — shows magnitude regardless of direction) ──
    abs_fig = go.Figure(layout={**PLOT_LAYOUT, "title": "Miss Magnitude by Archetype", "height": 350})
    if miss_col and "archetype" in df.columns and "category" in df.columns:
        abs_pivot = df.pivot_table(
            values=miss_col, index="archetype", columns="category",
            aggfunc=lambda x: np.mean(np.abs(x)),
        )
        cats_available = [c for c in SG_CATEGORIES if c in abs_pivot.columns]
        if cats_available and not abs_pivot.empty:
            abs_fig = go.Figure(data=go.Heatmap(
                z=abs_pivot[cats_available].values,
                x=[c.upper() for c in cats_available],
                y=abs_pivot.index.tolist(),
                colorscale="YlOrRd",
                text=np.round(abs_pivot[cats_available].values, 3),
                texttemplate="%{text}",
                hovertemplate="<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>",
            ))
            abs_fig.update_layout(**PLOT_LAYOUT, title="Avg |Miss| by Archetype & Category")

    # ── Archetype filter options for player explorer ──
    archetype_options = [{"label": "Biggest Misses", "value": "__biggest__"}]
    if "archetype" in df.columns:
        archs = sorted(df["archetype"].dropna().unique().tolist())
        for a in archs:
            count = df[df["archetype"] == a]["player_name"].nunique()
            archetype_options.append({"label": f"{a} ({count})", "value": a})

    # ── Recurring misses (field-centered) ──
    if miss_source and miss_source in df.columns and "event_id" in df.columns:
        all_diag = get_sg_diagnostics()  # unfiltered
        # Compute field-centered miss for accumulated data
        if "miss_centered" not in all_diag.columns and "miss" in all_diag.columns:
            all_diag["miss_centered"] = all_diag.groupby(["round", "category"])["miss"].transform(
                lambda x: x - x.mean()
            )
        if miss_source not in all_diag.columns and pred_col and actual_col:
            all_diag["_miss"] = all_diag[pred_col] - all_diag[actual_col]
            miss_source = "_miss"
        # Prefer centered miss for recurring analysis
        if "miss_centered" in all_diag.columns:
            miss_source = "miss_centered"

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

    return bias_fig, dir_fig, abs_fig, archetype_options, recurring_content, player_options


@callback(
    Output("diag-misses-table", "children"),
    Input("diag-archetype-filter", "value"),
    Input("diag-event-filter", "value"),
)
def update_player_explorer(archetype, event_id):
    df = get_sg_diagnostics()
    if df.empty:
        return dbc.Alert("No SG diagnostic data.", color="warning")

    if event_id:
        df = df[df["event_id"] == event_id]
    if df.empty:
        return dbc.Alert("No data for this event.", color="info")

    # Field-center
    if "miss_centered" not in df.columns and "miss" in df.columns:
        df["miss_centered"] = df.groupby(["round", "category"])["miss"].transform(
            lambda x: x - x.mean()
        )

    miss_col = "miss_centered" if "miss_centered" in df.columns else ("miss" if "miss" in df.columns else None)
    if not miss_col or miss_col not in df.columns:
        return dbc.Alert("Miss column not found.", color="info")

    if not archetype or archetype == "__biggest__":
        # Default: biggest misses (top 20 by absolute centered miss)
        df["abs_miss"] = df[miss_col].abs()
        top = df.nlargest(20, "abs_miss")
        display_cols = ["player_name", "round", "category", miss_col, "abs_miss", "archetype"]
    else:
        # Filter to selected archetype — summarize per player
        arch_df = df[df["archetype"] == archetype]
        if arch_df.empty:
            return dbc.Alert(f"No players classified as {archetype}.", color="info")

        # Pivot: one row per player, avg centered miss per category
        total_miss = (
            arch_df[arch_df["category"] == "total"]
            .groupby("player_name")[miss_col].mean()
            .reset_index()
            .rename(columns={miss_col: "total_miss"})
        )
        cat_misses = arch_df[arch_df["category"].isin(SG_CATEGORIES)].pivot_table(
            values=miss_col, index="player_name", columns="category", aggfunc="mean"
        ).reset_index()
        # Rename category columns for clarity
        cat_misses.columns = [f"{c}_miss" if c in SG_CATEGORIES else c for c in cat_misses.columns]

        top = total_miss.merge(cat_misses, on="player_name", how="left")
        top["abs_total"] = top["total_miss"].abs()
        top = top.sort_values("abs_total", ascending=False).drop(columns=["abs_total"])
        display_cols = ["player_name", "total_miss"] + [f"{c}_miss" for c in SG_CATEGORIES if f"{c}_miss" in top.columns]

    available = [c for c in display_cols if c in top.columns]
    return make_grid(top[available], id_suffix="misses", height=400)


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

    # Field-center miss for player detail
    if "miss_centered" not in player_df.columns and "miss" in player_df.columns:
        full_df = get_sg_diagnostics()
        if event_id:
            full_df = full_df[full_df["event_id"] == event_id]
        full_df["miss_centered"] = full_df.groupby(["round", "category"])["miss"].transform(
            lambda x: x - x.mean()
        )
        player_df = full_df[full_df["player_name"] == player]

    # Show per-round, per-category predicted vs actual
    pred_col = next((c for c in ["predicted_sg", "predicted"] if c in player_df.columns), None)
    actual_col = next((c for c in ["actual_sg", "actual"] if c in player_df.columns), None)

    display_cols = ["player_name", "round", "category"]
    if pred_col:
        display_cols.append(pred_col)
    if actual_col:
        display_cols.append(actual_col)
    if "miss_centered" in player_df.columns:
        display_cols.append("miss_centered")
    elif "miss" in player_df.columns:
        display_cols.append("miss")
    if "archetype" in player_df.columns:
        display_cols.append("archetype")

    available = [c for c in display_cols if c in player_df.columns]
    return make_grid(player_df[available], id_suffix="player-detail", height=300)
