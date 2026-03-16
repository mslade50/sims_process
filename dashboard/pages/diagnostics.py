"""Diagnostics page — SG prediction analysis and archetypes."""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dashboard.data_layer import get_sg_diagnostics, get_mkt_regress_diagnostics
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

    html.Hr(className="mt-5"),
    html.H4("Market Regression Analysis", className="page-header mt-3"),
    html.Div(id="diag-mkt-regress-alert"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="diag-mkt-rmse"), md=6),
        dbc.Col(dcc.Graph(id="diag-mkt-direction"), md=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="diag-mkt-scatter"), md=6),
        dbc.Col(dcc.Graph(id="diag-mkt-component"), md=6),
    ], className="mt-2"),
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

    # Always recompute field-centered miss (stored column may have NaN for older events)
    if "miss" in df.columns:
        df["miss_centered"] = df.groupby(["event_id", "round", "category"])["miss"].transform(
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
    if miss_col and miss_col in df.columns and "event_id" in df.columns:
        all_diag = get_sg_diagnostics()  # unfiltered
        # Always recompute field-centered miss
        if "miss" in all_diag.columns:
            all_diag["miss_centered"] = all_diag.groupby(["event_id", "round", "category"])["miss"].transform(
                lambda x: x - x.mean()
            )
        # Prefer centered miss for recurring analysis
        recur_col = "miss_centered" if "miss_centered" in all_diag.columns else miss_col

        if recur_col in all_diag.columns:
            player_event = (
                all_diag.groupby(["player_name", "event_id"])[recur_col]
                .mean().reset_index()
            )
            # Players with consistent direction across 2+ events
            player_events = player_event.groupby("player_name").agg(
                n_events=("event_id", "nunique"),
                avg_miss=(recur_col, "mean"),
                all_positive=(recur_col, lambda x: (x > 0).all()),
                all_negative=(recur_col, lambda x: (x < 0).all()),
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

    # Always recompute field-centered miss
    if "miss" in df.columns:
        df["miss_centered"] = df.groupby(["event_id", "round", "category"])["miss"].transform(
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
        # Filter to selected archetype — one row per player per event
        arch_df = df[df["archetype"] == archetype]
        if arch_df.empty:
            return dbc.Alert(f"No players classified as {archetype}.", color="info")

        # Total miss per player-event
        total_miss = (
            arch_df[arch_df["category"] == "total"]
            .groupby(["player_name", "event_id", "event_name"])[miss_col].mean()
            .reset_index()
            .rename(columns={miss_col: "total_miss"})
        )
        cat_misses = arch_df[arch_df["category"].isin(SG_CATEGORIES)].pivot_table(
            values=miss_col, index=["player_name", "event_id"], columns="category", aggfunc="mean"
        ).reset_index()
        cat_misses.columns = [f"{c}_miss" if c in SG_CATEGORIES else c for c in cat_misses.columns]

        top = total_miss.merge(cat_misses, on=["player_name", "event_id"], how="left")
        top["abs_total"] = top["total_miss"].abs()
        top = top.sort_values("abs_total", ascending=False).drop(columns=["abs_total", "event_id"])
        display_cols = ["player_name", "event_name", "total_miss"] + [f"{c}_miss" for c in SG_CATEGORIES if f"{c}_miss" in top.columns]

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

    # Always recompute field-centered miss for player detail
    full_df = get_sg_diagnostics()
    if event_id:
        full_df = full_df[full_df["event_id"] == event_id]
    if "miss" in full_df.columns:
        full_df["miss_centered"] = full_df.groupby(["event_id", "round", "category"])["miss"].transform(
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


# ══════════════════════════════════════════════════════════════════════════════
# Market Regression Analysis
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("diag-mkt-regress-alert", "children"),
    Output("diag-mkt-rmse", "figure"),
    Output("diag-mkt-direction", "figure"),
    Output("diag-mkt-scatter", "figure"),
    Output("diag-mkt-component", "figure"),
    Input("diag-event-filter", "id"),  # fires once on load
)
def update_mkt_regress(_):
    empty = go.Figure(layout={**PLOT_LAYOUT, "height": 350})

    df = get_mkt_regress_diagnostics()
    if df.empty:
        alert = dbc.Alert(
            "No market regression data. Drop mkt_regress_*.csv files in the project root.",
            color="info",
        )
        return alert, empty, empty, empty, empty

    alert = None
    events = df["event_name"].unique()

    # ── 1. RMSE comparison: raw pred vs regressed pred (per event + overall) ──
    rmse_rows = []
    for ename in events:
        edf = df[df["event_name"] == ename]
        rmse_raw = np.sqrt((edf["error_raw"] ** 2).mean())
        rmse_reg = np.sqrt((edf["error_regressed"] ** 2).mean())
        rmse_rows.append({"event": ename, "Raw Pred": rmse_raw, "Regressed": rmse_reg})
    # Overall
    rmse_raw_all = np.sqrt((df["error_raw"] ** 2).mean())
    rmse_reg_all = np.sqrt((df["error_regressed"] ** 2).mean())
    rmse_rows.append({"event": "OVERALL", "Raw Pred": rmse_raw_all, "Regressed": rmse_reg_all})

    rmse_df = pd.DataFrame(rmse_rows)
    rmse_fig = go.Figure(layout={**PLOT_LAYOUT, "title": "RMSE: Raw vs Regressed", "height": 350})
    rmse_fig.add_trace(go.Bar(
        name="Raw Pred", x=rmse_df["event"], y=rmse_df["Raw Pred"],
        marker_color="#b71c1c", text=rmse_df["Raw Pred"].round(3), textposition="outside",
    ))
    rmse_fig.add_trace(go.Bar(
        name="Regressed", x=rmse_df["event"], y=rmse_df["Regressed"],
        marker_color="#2e7d32", text=rmse_df["Regressed"].round(3), textposition="outside",
    ))
    rmse_fig.update_layout(barmode="group", yaxis_title="RMSE (SG)")
    rmse_fig.update_xaxes(tickangle=-20)

    # ── 2. Directional accuracy by regression bucket ──
    df["rgrs_bucket"] = pd.cut(
        df["tot_rgrs"],
        bins=[-np.inf, -0.08, -0.03, 0.03, 0.08, np.inf],
        labels=["Big Down", "Mod Down", "Neutral", "Mod Up", "Big Up"],
    )
    bucket_stats = df.groupby("rgrs_bucket", observed=True).agg(
        n=("regress_helped", "count"),
        hit_rate=("regress_helped", "mean"),
    ).reset_index()

    dir_fig = go.Figure(layout={**PLOT_LAYOUT, "title": "Regression Hit Rate by Bucket", "height": 350})
    colors = ["#b71c1c", "#e57373", "#9e9e9e", "#81c784", "#2e7d32"]
    dir_fig.add_trace(go.Bar(
        x=bucket_stats["rgrs_bucket"].astype(str),
        y=bucket_stats["hit_rate"] * 100,
        marker_color=colors[:len(bucket_stats)],
        text=[f"{r:.0f}%<br>(n={n})" for r, n in zip(bucket_stats["hit_rate"] * 100, bucket_stats["n"])],
        textposition="outside",
    ))
    dir_fig.add_hline(y=50, line_dash="dot", line_color="#888", annotation_text="50%")
    dir_fig.update_yaxes(title_text="% of Players Where Regression Helped", range=[0, 100])

    # ── 3. Scatter: tot_rgrs vs error improvement ──
    df["error_improvement"] = df["error_raw"].abs() - df["error_regressed"].abs()
    scatter_fig = go.Figure(layout={**PLOT_LAYOUT, "title": "Regression Size vs Error Improvement", "height": 350})
    scatter_fig.add_trace(go.Scatter(
        x=df["tot_rgrs"],
        y=df["error_improvement"],
        mode="markers",
        marker=dict(
            size=5, opacity=0.5,
            color=df["error_improvement"],
            colorscale="RdYlGn", cmid=0,
        ),
        text=df["player_name"].str.title() + " (" + df["event_name"].str.slice(0, 12) + ")",
        hovertemplate="<b>%{text}</b><br>Regression: %{x:+.3f}<br>Error improvement: %{y:+.3f}<extra></extra>",
    ))
    scatter_fig.add_hline(y=0, line_dash="dot", line_color="#888")
    scatter_fig.add_vline(x=0, line_dash="dot", line_color="#888")
    scatter_fig.update_xaxes(title_text="Total Regression (SG)")
    scatter_fig.update_yaxes(title_text="Error Improvement (+ = helped)")

    # ── 4. Component value: mkt_adj vs mu_adj contribution ──
    comp_rows = []
    for comp_name, comp_col in [("mkt_adj", "mkt_adj"), ("mu_adj", "mu_adj")]:
        for ename in events:
            edf = df[df["event_name"] == ename]
            # Correlation between component and actual error reduction
            if edf[comp_col].std() > 0:
                corr = edf[comp_col].corr(edf["actual_sg"] - edf["pred"])
            else:
                corr = 0
            comp_rows.append({"event": ename, "component": comp_name, "corr_with_actual": corr})
        # Overall
        if df[comp_col].std() > 0:
            corr_all = df[comp_col].corr(df["actual_sg"] - df["pred"])
        else:
            corr_all = 0
        comp_rows.append({"event": "OVERALL", "component": comp_name, "corr_with_actual": corr_all})

    comp_df = pd.DataFrame(comp_rows)
    comp_fig = go.Figure(layout={**PLOT_LAYOUT, "title": "Component Signal Strength", "height": 350})
    for comp_name, color in [("mkt_adj", "#1f77b4"), ("mu_adj", "#ff7f0e")]:
        cdf = comp_df[comp_df["component"] == comp_name]
        comp_fig.add_trace(go.Bar(
            name=comp_name, x=cdf["event"], y=cdf["corr_with_actual"],
            marker_color=color,
            text=cdf["corr_with_actual"].round(3), textposition="outside",
        ))
    comp_fig.update_layout(barmode="group", yaxis_title="Corr(component, actual miss)")
    comp_fig.update_xaxes(tickangle=-20)
    comp_fig.add_hline(y=0, line_dash="dot", line_color="#888")

    return alert, rmse_fig, dir_fig, scatter_fig, comp_fig
