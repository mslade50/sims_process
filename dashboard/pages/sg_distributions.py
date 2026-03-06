"""SG Distributions — Per-player input SG category distributions (raw & course-adjusted)."""

import json

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from dashboard.data_layer import (
    get_sg_dist_player,
    get_this_week_dists_adjusted,
    get_model_predictions,
)
from dashboard.components.tables import make_grid

dash.register_page(__name__, path="/sg-distributions", title="SG Distributions", order=6)

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,33,62,0.8)",
    font=dict(color="#e0e0e0"),
    margin=dict(l=50, r=30, t=60, b=40),
)

CATEGORIES = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
CAT_LABELS = {"sg_ott": "OTT", "sg_app": "APP", "sg_arg": "ARG", "sg_putt": "PUTT"}
CAT_COLORS = {
    "sg_ott": "#1f77b4",
    "sg_app": "#2ca02c",
    "sg_arg": "#ff7f0e",
    "sg_putt": "#d62728",
}

BIN_EDGES = np.arange(-10.0, 10.25, 0.25)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2


def _load_field_players():
    """Return sorted player list from this week's adjusted dists, plus my_pred mapping."""
    adj = get_this_week_dists_adjusted()
    if adj.empty:
        return [], {}

    # Get unique players from adjusted file (lowercase names)
    players = adj["player_name"].unique().tolist()

    # Try to load predictions for sorting
    preds = get_model_predictions(1)
    pred_map = {}
    if not preds.empty and "player_name" in preds.columns and "my_pred" in preds.columns:
        for _, row in preds.iterrows():
            pred_map[str(row["player_name"]).lower().strip()] = row["my_pred"]

    # Sort by my_pred descending (best players first), then alpha
    players.sort(key=lambda p: (-pred_map.get(p.lower().strip(), -99), p))
    return players, pred_map


def _get_player_raw(player_name, raw_df, adj_df):
    """Get raw distribution data for a player, matching names across files."""
    # adj_df has lowercase names; raw_df has mixed case
    # Find original name from adj_df if available
    adj_row = adj_df[adj_df["player_name"] == player_name.lower().strip()]
    if not adj_row.empty and "player_name_original" in adj_row.columns:
        original_name = adj_row["player_name_original"].iloc[0]
    else:
        original_name = player_name

    # Try matching raw_df on original name or lowercase
    raw_lower = raw_df.copy()
    raw_lower["_join_key"] = raw_lower["player_name"].str.lower().str.strip()
    match = raw_lower[raw_lower["_join_key"] == player_name.lower().strip()]
    if match.empty:
        match = raw_df[raw_df["player_name"] == original_name]
    return match


def _build_raw_figure(player_raw):
    """Build 2x2 subplot with raw histogram bars."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["OTT", "APP", "ARG", "PUTT"],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for cat, (r, c) in zip(CATEGORIES, positions):
        cat_data = player_raw[player_raw["category_clean"] == cat]
        if cat_data.empty:
            continue

        row = cat_data.iloc[0]
        try:
            counts = json.loads(row["hist_counts_json"])
        except (json.JSONDecodeError, TypeError, KeyError):
            continue

        # Normalize to density
        total = sum(counts)
        if total > 0:
            density = [c_val / (total * 0.25) for c_val in counts]
        else:
            density = counts

        # Trim to visible range
        mask = [abs(bc) <= 5 for bc in BIN_CENTERS[: len(density)]]
        x_vals = [bc for bc, m in zip(BIN_CENTERS, mask) if m]
        y_vals = [d for d, m in zip(density, mask) if m]

        fig.add_trace(
            go.Bar(
                x=x_vals, y=y_vals,
                marker_color=CAT_COLORS[cat],
                opacity=0.75,
                showlegend=False,
                hovertemplate="SG: %{x:.2f}<br>Density: %{y:.3f}<extra></extra>",
            ),
            row=r, col=c,
        )

        # Add mean line
        mean_val = row.get("mean", None)
        if mean_val is not None and not pd.isna(mean_val):
            fig.add_vline(
                x=float(mean_val), line_dash="dash", line_color="white",
                line_width=1.5, row=r, col=c,
            )

    return fig


def _build_adjusted_figure(player_name, adj_df, pred_map):
    """Build 2x2 subplot with course-adjusted PDF curves and raw normal overlay."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["OTT", "APP", "ARG", "PUTT"],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    player_adj = adj_df[adj_df["player_name"] == player_name.lower().strip()]
    if player_adj.empty:
        return fig

    # Compute recentering shift
    target = pred_map.get(player_name.lower().strip())
    shift = 0.0
    if target is not None:
        cat_sum = 0.0
        for cat in CATEGORIES:
            cat_row = player_adj[player_adj["category_clean"] == cat]
            if not cat_row.empty:
                cat_sum += float(cat_row["mean_adj"].iloc[0])
        shift = (target - cat_sum) / 4.0

    x_range = np.linspace(-5, 5, 400)

    for cat, (row_idx, col_idx) in zip(CATEGORIES, positions):
        cat_data = player_adj[player_adj["category_clean"] == cat]
        if cat_data.empty:
            continue

        rec = cat_data.iloc[0]
        mean_adj = float(rec["mean_adj"]) + shift
        std_adj = float(rec["std_adj"])
        dist_type = str(rec.get("distribution", "normal")).lower()

        # Course-adjusted PDF
        if dist_type == "student_t" and "df_t" in rec.index and not pd.isna(rec["df_t"]):
            from scipy.stats import t as t_dist
            df_t = float(rec["df_t"])
            y_adj = t_dist.pdf(x_range, df=df_t, loc=mean_adj, scale=std_adj)
        else:
            from scipy.stats import norm
            y_adj = norm.pdf(x_range, loc=mean_adj, scale=std_adj)

        is_first = (row_idx == 1 and col_idx == 1)
        fig.add_trace(
            go.Scatter(
                x=x_range, y=y_adj,
                mode="lines",
                line=dict(color=CAT_COLORS[cat], width=2.5),
                name=f"{CAT_LABELS[cat]} (adjusted)",
                showlegend=is_first,
                hovertemplate="SG: %{x:.2f}<br>Density: %{y:.3f}<extra></extra>",
            ),
            row=row_idx, col=col_idx,
        )

        # Raw normal overlay (dashed) using original mean/std
        raw_mean = float(rec["mean"]) if "mean" in rec.index and not pd.isna(rec["mean"]) else None
        raw_std = float(rec["std"]) if "std" in rec.index and not pd.isna(rec["std"]) else None
        if raw_mean is not None and raw_std is not None and raw_std > 0:
            from scipy.stats import norm
            y_raw = norm.pdf(x_range, loc=raw_mean, scale=raw_std)
            fig.add_trace(
                go.Scatter(
                    x=x_range, y=y_raw,
                    mode="lines",
                    line=dict(color="gray", width=1.5, dash="dash"),
                    name="Raw" if is_first else None,
                    showlegend=is_first,
                    hovertemplate="SG: %{x:.2f}<br>Density: %{y:.3f}<extra></extra>",
                ),
                row=row_idx, col=col_idx,
            )

        # Mean line
        fig.add_vline(
            x=mean_adj, line_dash="dot", line_color="white",
            line_width=1, row=row_idx, col=col_idx,
        )

    return fig


def _build_moments_table(player_name, mode, raw_df, adj_df, pred_map):
    """Build moments summary DataFrame for the AG Grid."""
    rows = []

    if mode == "adjusted":
        player_data = adj_df[adj_df["player_name"] == player_name.lower().strip()]
        # Compute recentering shift
        target = pred_map.get(player_name.lower().strip())
        shift = 0.0
        if target is not None:
            cat_sum = 0.0
            for cat in CATEGORIES:
                cat_row = player_data[player_data["category_clean"] == cat]
                if not cat_row.empty:
                    cat_sum += float(cat_row["mean_adj"].iloc[0])
            shift = (target - cat_sum) / 4.0

        total_mean = 0.0
        total_var = 0.0
        for cat in CATEGORIES:
            cat_row = player_data[player_data["category_clean"] == cat]
            if cat_row.empty:
                rows.append({"Category": CAT_LABELS[cat]})
                continue
            rec = cat_row.iloc[0]
            mean_val = float(rec["mean_adj"]) + shift
            std_val = float(rec["std_adj"])
            total_mean += mean_val
            total_var += std_val ** 2
            rows.append({
                "Category": CAT_LABELS[cat],
                "Mean": round(mean_val, 3),
                "Std": round(std_val, 3),
                "Skew": "—",
                "Kurtosis": round(float(rec["excess_kurtosis"]), 3) if "excess_kurtosis" in rec.index and not pd.isna(rec["excess_kurtosis"]) else "—",
                "N_eff": int(rec["n"]) if "n" in rec.index and not pd.isna(rec["n"]) else "",
                "Q10": round(float(rec["q10_adj"]), 2) if "q10_adj" in rec.index and not pd.isna(rec["q10_adj"]) else "—",
                "Q25": round(float(rec["q25_adj"]), 2) if "q25_adj" in rec.index and not pd.isna(rec["q25_adj"]) else "—",
                "Q50": round(float(rec["q50_adj"]), 2) if "q50_adj" in rec.index and not pd.isna(rec["q50_adj"]) else "—",
                "Q75": round(float(rec["q75_adj"]), 2) if "q75_adj" in rec.index and not pd.isna(rec["q75_adj"]) else "—",
                "Q90": round(float(rec["q90_adj"]), 2) if "q90_adj" in rec.index and not pd.isna(rec["q90_adj"]) else "—",
            })
        # Total row
        rows.append({
            "Category": "Total",
            "Mean": round(total_mean, 3),
            "Std": round(total_var ** 0.5, 3),
            "Skew": "—",
            "Kurtosis": "—",
            "N_eff": "",
            "Q10": "—", "Q25": "—", "Q50": "—", "Q75": "—", "Q90": "—",
        })
    else:
        # Raw mode — use sg_dist_player.csv
        player_raw = _get_player_raw(player_name, raw_df, adj_df)
        total_mean = 0.0
        total_var = 0.0
        for cat in CATEGORIES:
            cat_row = player_raw[player_raw["category_clean"] == cat]
            if cat_row.empty:
                rows.append({"Category": CAT_LABELS[cat]})
                continue
            rec = cat_row.iloc[0]
            mean_val = float(rec["mean"])
            std_val = float(rec["std"])
            total_mean += mean_val
            total_var += std_val ** 2
            rows.append({
                "Category": CAT_LABELS[cat],
                "Mean": round(mean_val, 3),
                "Std": round(std_val, 3),
                "Skew": round(float(rec["skew"]), 3) if "skew" in rec.index and not pd.isna(rec["skew"]) else "—",
                "Kurtosis": round(float(rec["excess_kurtosis"]), 3) if "excess_kurtosis" in rec.index and not pd.isna(rec["excess_kurtosis"]) else "—",
                "N_eff": round(float(rec["n_eff"]), 1) if "n_eff" in rec.index and not pd.isna(rec["n_eff"]) else "—",
                "Q10": round(float(rec["q10"]), 2) if "q10" in rec.index and not pd.isna(rec["q10"]) else "—",
                "Q25": round(float(rec["q25"]), 2) if "q25" in rec.index and not pd.isna(rec["q25"]) else "—",
                "Q50": round(float(rec["q50"]), 2) if "q50" in rec.index and not pd.isna(rec["q50"]) else "—",
                "Q75": round(float(rec["q75"]), 2) if "q75" in rec.index and not pd.isna(rec["q75"]) else "—",
                "Q90": round(float(rec["q90"]), 2) if "q90" in rec.index and not pd.isna(rec["q90"]) else "—",
            })
        rows.append({
            "Category": "Total",
            "Mean": round(total_mean, 3),
            "Std": round(total_var ** 0.5, 3),
            "Skew": "—",
            "Kurtosis": "—",
            "N_eff": "",
            "Q10": "—", "Q25": "—", "Q50": "—", "Q75": "—", "Q90": "—",
        })

    return pd.DataFrame(rows)


# ── Layout ───────────────────────────────────────────────────────────────────

layout = dbc.Container([
    html.H4("SG Input Distributions", className="page-header"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Mode"),
            dbc.RadioItems(
                id="sgdist-mode",
                options=[
                    {"label": "Raw", "value": "raw"},
                    {"label": "Course-Adjusted", "value": "adjusted"},
                ],
                value="raw",
                inline=True,
                className="mb-2",
            ),
        ], md=3),
        dbc.Col([
            dbc.Label("Player"),
            dcc.Dropdown(id="sgdist-player", placeholder="Select player...", className="mb-2"),
        ], md=5),
    ], className="mb-3"),

    # Warning area
    html.Div(id="sgdist-warning"),

    # Chart
    dcc.Graph(id="sgdist-chart", style={"height": "600px"}),

    # Moments table
    html.H5("Distribution Moments", className="mt-3 mb-2"),
    html.Div(id="sgdist-moments-table"),
], fluid=True)


# ── Callbacks ────────────────────────────────────────────────────────────────

@callback(
    Output("sgdist-player", "options"),
    Output("sgdist-player", "value"),
    Input("sgdist-mode", "value"),
)
def populate_players(mode):
    players, _ = _load_field_players()
    if not players:
        return [], None
    options = [{"label": p.title(), "value": p} for p in players]
    return options, players[0] if players else None


@callback(
    Output("sgdist-chart", "figure"),
    Output("sgdist-moments-table", "children"),
    Output("sgdist-warning", "children"),
    Input("sgdist-player", "value"),
    Input("sgdist-mode", "value"),
)
def update_distributions(player, mode):
    empty_fig = go.Figure()
    empty_fig.update_layout(**PLOT_LAYOUT, title="Select a player to view SG distributions")

    if not player:
        return empty_fig, None, None

    raw_df = get_sg_dist_player()
    adj_df = get_this_week_dists_adjusted()
    _, pred_map = _load_field_players()

    warning = None

    if raw_df.empty and adj_df.empty:
        return empty_fig, dbc.Alert("No SG distribution data available.", color="warning"), None

    # Build figure
    if mode == "adjusted":
        if adj_df.empty:
            return empty_fig, None, dbc.Alert("No course-adjusted data available.", color="warning")
        fig = _build_adjusted_figure(player, adj_df, pred_map)
        title_suffix = " (Course-Adjusted)"

        # Warn if no predictions for recentering
        if not pred_map or player.lower().strip() not in pred_map:
            warning = dbc.Alert(
                "model_predictions_r1.csv not found — showing un-recentered means.",
                color="info", dismissable=True,
            )
    else:
        if raw_df.empty:
            return empty_fig, None, dbc.Alert("No raw distribution data available.", color="warning")
        player_raw = _get_player_raw(player, raw_df, adj_df)
        if player_raw.empty:
            return empty_fig, None, dbc.Alert(f"No data found for {player.title()}.", color="warning")
        fig = _build_raw_figure(player_raw)
        title_suffix = " (Raw)"

    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text=f"<b>{player.title()}</b>{title_suffix}", x=0.5, xanchor="center"),
        showlegend=True,
        legend=dict(x=0.85, y=0.98, bgcolor="rgba(0,0,0,0.5)"),
        height=600,
    )

    # Build moments table
    moments_df = _build_moments_table(player, mode, raw_df, adj_df, pred_map)
    if moments_df.empty:
        table = dbc.Alert("No moment data available.", color="info")
    else:
        table = make_grid(moments_df, id_suffix="sgdist-moments", height=250, page_size=10)

    return fig, table, warning
