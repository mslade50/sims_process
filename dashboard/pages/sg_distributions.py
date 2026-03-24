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
    get_v2_dists,
    get_model_predictions,
)
from dashboard.components.tables import make_grid
from sheet_config import load_config as _load_sheet_config

dash.register_page(__name__, path="/sg-distributions", title="SG Distributions", order=6)

_cfg = _load_sheet_config()
_COURSE_MULTS = {cat: _cfg.get("course_cat_mults", {}).get(cat, 1.0) for cat in ["sg_ott", "sg_app", "sg_arg", "sg_putt"]}

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

COMPARE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def _load_field_players():
    """Return sorted player list from this week's adjusted dists, plus my_pred mapping."""
    adj = get_v2_dists()
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


def _get_player_raw(player_name, raw_df):
    """Get raw distribution data for a player by lowercase name match."""
    raw_lower = raw_df.copy()
    raw_lower["_join_key"] = raw_lower["player_name"].str.lower().str.strip()
    return raw_lower[raw_lower["_join_key"] == player_name.lower().strip()]


def _build_raw_figure(player_raws, player_names):
    """Build 2x2 subplot with raw histogram bars, supporting multi-player overlay."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["OTT", "APP", "ARG", "PUTT"],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    multi = len(player_names) > 1
    axis_nums = [1, 2, 3, 4]  # subplot axis numbering

    for cat_idx, (cat, (r, c)) in enumerate(zip(CATEGORIES, positions)):
        primary_stats = {}  # for annotation

        for pi, (p_raw, p_name) in enumerate(zip(player_raws, player_names)):
            cat_data = p_raw[p_raw["category_clean"] == cat]
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

            color = COMPARE_COLORS[pi % len(COMPARE_COLORS)] if multi else CAT_COLORS[cat]
            opacity = 0.5 if multi else 0.75
            is_first_subplot = (r == 1 and c == 1)

            fig.add_trace(
                go.Bar(
                    x=x_vals, y=y_vals,
                    marker_color=color,
                    opacity=opacity,
                    name=p_name.title() if is_first_subplot else None,
                    showlegend=(multi and is_first_subplot),
                    legendgroup=p_name,
                    hovertemplate="SG: %{x:.2f}<br>Density: %{y:.3f}<extra></extra>",
                ),
                row=r, col=c,
            )

            # Mean line for primary player only
            if pi == 0:
                mean_val = row.get("mean", None)
                std_val = row.get("std", None)
                skew_val = row.get("skew", None)
                if mean_val is not None and not pd.isna(mean_val):
                    fig.add_vline(
                        x=float(mean_val), line_dash="dash", line_color="white",
                        line_width=1.5, row=r, col=c,
                    )
                    primary_stats["mean"] = float(mean_val)
                if std_val is not None and not pd.isna(std_val):
                    primary_stats["std"] = float(std_val)
                if skew_val is not None and not pd.isna(skew_val):
                    primary_stats["skew"] = float(skew_val)

        # Annotation for primary player stats
        if primary_stats:
            ax_num = axis_nums[cat_idx]
            ax_suffix = "" if ax_num == 1 else str(ax_num)
            parts = []
            if "mean" in primary_stats:
                parts.append(f"\u03bc={primary_stats['mean']:+.2f}")
            if "std" in primary_stats:
                parts.append(f"\u03c3={primary_stats['std']:.2f}")
            if "skew" in primary_stats:
                parts.append(f"skew={primary_stats['skew']:.2f}")
            if parts:
                fig.add_annotation(
                    text="  ".join(parts),
                    xref=f"x{ax_suffix} domain", yref=f"y{ax_suffix} domain",
                    x=0.98, y=0.95, showarrow=False,
                    font=dict(size=10, color="#aaaaaa"),
                    xanchor="right", yanchor="top",
                    bgcolor="rgba(0,0,0,0.4)",
                )

    if multi:
        fig.update_layout(barmode="overlay")

    return fig


def _build_adjusted_figure(player_names, adj_df, pred_map):
    """Build 2x2 subplot with course-adjusted PDF curves, field avg overlay, and annotations."""
    from scipy.stats import norm

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["OTT", "APP", "ARG", "PUTT"],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    axis_nums = [1, 2, 3, 4]
    multi = len(player_names) > 1

    # Pre-compute field averages per category (V2: mean, std * course mult)
    field_avgs = {}
    for cat in CATEGORIES:
        cat_all = adj_df[adj_df["category_clean"] == cat]
        if not cat_all.empty:
            field_avgs[cat] = {
                "mean": cat_all["mean"].mean(),
                "std": (cat_all["std"] * _COURSE_MULTS[cat]).mean(),
            }

    x_range = np.linspace(-5, 5, 400)

    for cat_idx, (cat, (row_idx, col_idx)) in enumerate(zip(CATEGORIES, positions)):
        is_first = (row_idx == 1 and col_idx == 1)
        primary_stats = {}

        for pi, player_name in enumerate(player_names):
            player_adj = adj_df[adj_df["player_name"] == player_name.lower().strip()]
            if player_adj.empty:
                continue

            cat_data = player_adj[player_adj["category_clean"] == cat]
            if cat_data.empty:
                continue

            # Compute per-player recentering shift
            target = pred_map.get(player_name.lower().strip())
            shift = 0.0
            if target is not None:
                cat_sum = sum(
                    float(player_adj[player_adj["category_clean"] == c]["mean"].iloc[0])
                    for c in CATEGORIES
                    if not player_adj[player_adj["category_clean"] == c].empty
                )
                shift = (target - cat_sum) / 4.0

            rec = cat_data.iloc[0]
            mean_adj = float(rec["mean"]) + shift
            std_adj = float(rec["std"]) * _COURSE_MULTS[cat]

            # Course-adjusted PDF (always normal — V2 dropped Student-t)
            y_adj = norm.pdf(x_range, loc=mean_adj, scale=std_adj)

            if multi:
                color = COMPARE_COLORS[pi % len(COMPARE_COLORS)]
                fig.add_trace(
                    go.Scatter(
                        x=x_range, y=y_adj,
                        mode="lines",
                        line=dict(color=color, width=2.5),
                        name=player_name.title() if is_first else None,
                        showlegend=is_first,
                        legendgroup=player_name,
                        hovertemplate="SG: %{x:.2f}<br>Density: %{y:.3f}<extra></extra>",
                    ),
                    row=row_idx, col=col_idx,
                )
            else:
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

            # Collect primary player stats for annotation
            if pi == 0:
                primary_stats["mean"] = mean_adj
                primary_stats["std"] = std_adj
                skew_val = rec.get("skew")
                if skew_val is not None and not pd.isna(skew_val):
                    primary_stats["skew"] = float(skew_val)

                # Mean line for primary player
                fig.add_vline(
                    x=mean_adj, line_dash="dot", line_color="white",
                    line_width=1, row=row_idx, col=col_idx,
                )

        # Field average overlay (gray dashed)
        if cat in field_avgs:
            fa = field_avgs[cat]
            y_field = norm.pdf(x_range, loc=fa["mean"], scale=fa["std"])
            fig.add_trace(
                go.Scatter(
                    x=x_range, y=y_field,
                    mode="lines",
                    line=dict(color="gray", width=1.5, dash="dash"),
                    name="Field Avg" if is_first else None,
                    showlegend=is_first,
                    legendgroup="field_avg",
                    hovertemplate="SG: %{x:.2f}<br>Density: %{y:.3f}<extra></extra>",
                ),
                row=row_idx, col=col_idx,
            )

        # Annotation for primary player stats
        if primary_stats:
            ax_num = axis_nums[cat_idx]
            ax_suffix = "" if ax_num == 1 else str(ax_num)
            parts = [f"\u03bc={primary_stats['mean']:+.2f}", f"\u03c3={primary_stats['std']:.2f}"]
            if "skew" in primary_stats:
                parts.append(f"skew={primary_stats['skew']:.2f}")
            fig.add_annotation(
                text="  ".join(parts),
                xref=f"x{ax_suffix} domain", yref=f"y{ax_suffix} domain",
                x=0.98, y=0.95, showarrow=False,
                font=dict(size=10, color="#aaaaaa"),
                xanchor="right", yanchor="top",
                bgcolor="rgba(0,0,0,0.4)",
            )

    return fig


def _build_moments_table(player_name, mode, raw_df, adj_df, pred_map):
    """Build moments summary DataFrame for the AG Grid."""
    from scipy.stats import norm

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
                    cat_sum += float(cat_row["mean"].iloc[0])
            shift = (target - cat_sum) / 4.0

        total_mean = 0.0
        total_var = 0.0
        for cat in CATEGORIES:
            cat_row = player_data[player_data["category_clean"] == cat]
            if cat_row.empty:
                rows.append({"Category": CAT_LABELS[cat]})
                continue
            rec = cat_row.iloc[0]
            mean_val = float(rec["mean"]) + shift
            std_val = float(rec["std"]) * _COURSE_MULTS[cat]
            total_mean += mean_val
            total_var += std_val ** 2
            skew_val = round(float(rec["skew"]), 3) if "skew" in rec.index and not pd.isna(rec.get("skew")) else "—"
            n_eff_val = round(float(rec["n_eff"]), 1) if "n_eff" in rec.index and not pd.isna(rec.get("n_eff")) else ""
            # Compute quantiles from normal distribution
            qs = norm.ppf([0.1, 0.25, 0.5, 0.75, 0.9], loc=mean_val, scale=std_val)
            rows.append({
                "Category": CAT_LABELS[cat],
                "Mean": round(mean_val, 3),
                "Std": round(std_val, 3),
                "Skew": skew_val,
                "Kurtosis": round(float(rec["excess_kurtosis"]), 3) if "excess_kurtosis" in rec.index and not pd.isna(rec.get("excess_kurtosis")) else "—",
                "N_eff": n_eff_val,
                "Q10": round(float(qs[0]), 2),
                "Q25": round(float(qs[1]), 2),
                "Q50": round(float(qs[2]), 2),
                "Q75": round(float(qs[3]), 2),
                "Q90": round(float(qs[4]), 2),
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
        player_raw = _get_player_raw(player_name, raw_df)
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
            dcc.Dropdown(id="sgdist-player", placeholder="Select player...", multi=True, className="mb-2"),
        ], md=4),
        dbc.Col([
            dbc.Label("Compare (optional)"),
            dcc.Dropdown(id="sgdist-compare", placeholder="Add players...", multi=True, className="mb-2"),
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
    Output("sgdist-compare", "options"),
    Input("sgdist-mode", "value"),
)
def populate_players(mode):
    players, _ = _load_field_players()
    if not players:
        return [], None, []
    options = [{"label": p.title(), "value": p} for p in players]
    return options, [players[0]] if players else [], options


@callback(
    Output("sgdist-chart", "figure"),
    Output("sgdist-moments-table", "children"),
    Output("sgdist-warning", "children"),
    Input("sgdist-player", "value"),
    Input("sgdist-compare", "value"),
    Input("sgdist-mode", "value"),
)
def update_distributions(player, compare_players, mode):
    empty_fig = go.Figure()
    empty_fig.update_layout(**PLOT_LAYOUT, title="Select a player to view SG distributions")

    if not player:
        return empty_fig, None, None

    raw_df = get_sg_dist_player()
    adj_df = get_v2_dists()
    _, pred_map = _load_field_players()

    # Normalize player to list (multi=True dropdown)
    primary = player if isinstance(player, list) else ([player] if player else [])
    if not primary:
        return empty_fig, None, None
    primary_player = primary[0]  # first selected player for moments table / annotations
    all_players = list(primary)
    if compare_players:
        all_players.extend([p for p in compare_players if p not in all_players])

    warning = None

    if raw_df.empty and adj_df.empty:
        return empty_fig, dbc.Alert("No SG distribution data available.", color="warning"), None

    # Build figure
    if mode == "adjusted":
        if adj_df.empty:
            return empty_fig, None, dbc.Alert("No course-adjusted data available.", color="warning")
        fig = _build_adjusted_figure(all_players, adj_df, pred_map)
        title_suffix = " (Course-Adjusted)"

        # Warn if no predictions for recentering
        if not pred_map or primary_player.lower().strip() not in pred_map:
            warning = dbc.Alert(
                "model_predictions_r1.csv not found — showing un-recentered means.",
                color="info", dismissable=True,
            )
    else:
        if raw_df.empty:
            return empty_fig, None, dbc.Alert("No raw distribution data available.", color="warning")
        # Build raw data for each player
        player_raws = []
        valid_names = []
        for p in all_players:
            p_raw = _get_player_raw(p, raw_df)
            if not p_raw.empty:
                player_raws.append(p_raw)
                valid_names.append(p)
        if not player_raws:
            return empty_fig, None, dbc.Alert(f"No data found for {primary_player.title()}.", color="warning")
        fig = _build_raw_figure(player_raws, valid_names)
        title_suffix = " (Raw)"

    # Title
    if len(all_players) > 1:
        title_text = " vs ".join(p.title() for p in all_players) + title_suffix
    else:
        title_text = f"<b>{primary_player.title()}</b>{title_suffix}"

    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text=title_text, x=0.5, xanchor="center"),
        showlegend=True,
        legend=dict(x=0.85, y=0.98, bgcolor="rgba(0,0,0,0.5)"),
        height=600,
    )

    # Build moments table (primary player only)
    moments_df = _build_moments_table(primary_player, mode, raw_df, adj_df, pred_map)
    if moments_df.empty:
        table = dbc.Alert("No moment data available.", color="info")
    else:
        table = make_grid(moments_df, id_suffix="sgdist-moments", height=250, page_size=10)

    return fig, table, warning
