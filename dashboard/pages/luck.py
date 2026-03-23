"""Luck & Attribution page — Margin analysis and SG category breakdown."""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from dashboard.data_layer import get_luck_attribution
from dashboard.components.stat_cards import stat_card_row
from dashboard.components.tables import make_grid

dash.register_page(__name__, path="/luck", title="Luck & Attribution", order=10)

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,33,62,0.8)",
    font=dict(color="#e0e0e0"),
    margin=dict(l=50, r=30, t=40, b=40),
)

SG_CATS = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
SG_LABELS = {"sg_ott": "OTT", "sg_app": "APP", "sg_arg": "ARG", "sg_putt": "PUTT"}

RESULT_COLORS = {"win": "#2ca02c", "win_dh": "#2ca02c", "loss": "#d62728", "push": "#7f7f7f"}

SHARP_BOOKS = ["pinnacle", "betonline", "betcris", "bet online", "bookmaker"]


def _sg_waterfall(sg_df, title):
    """Build SG category delta bar chart from a filtered DataFrame."""
    if sg_df.empty:
        return go.Figure().update_layout(**PLOT_LAYOUT, title=f"{title} — No data")
    cat_means = []
    for cat in SG_CATS:
        opp_col, bet_col = f"opp_{cat}", f"bet_on_{cat}"
        if opp_col in sg_df.columns and bet_col in sg_df.columns:
            delta = (sg_df[opp_col] - sg_df[bet_col]).mean()
            cat_means.append({"category": SG_LABELS[cat], "delta": delta if pd.notna(delta) else 0})
    if not cat_means:
        return go.Figure().update_layout(**PLOT_LAYOUT, title=f"{title} — No SG data")
    wf_df = pd.DataFrame(cat_means)
    colors = ["#d62728" if d > 0 else "#2ca02c" for d in wf_df["delta"]]
    fig = go.Figure(go.Bar(
        x=wf_df["category"], y=wf_df["delta"], marker_color=colors,
        text=[f"{d:+.2f}" for d in wf_df["delta"]], textposition="outside",
    ))
    fig.update_layout(**PLOT_LAYOUT, title=title, xaxis_title="Category", yaxis_title="Avg SG Delta")
    return fig


def _sg_heatmap(sg_df, title):
    """Build SG category delta heatmap by event from a filtered DataFrame."""
    if sg_df.empty or "event_name" not in sg_df.columns:
        return go.Figure().update_layout(**PLOT_LAYOUT, title=f"{title} — No data")
    events = sg_df["event_name"].unique()
    heatmap_data = []
    for ev in events:
        ev_df = sg_df[sg_df["event_name"] == ev]
        row = {"event": ev}
        for cat in SG_CATS:
            opp_col, bet_col = f"opp_{cat}", f"bet_on_{cat}"
            if opp_col in ev_df.columns and bet_col in ev_df.columns:
                row[SG_LABELS[cat]] = (ev_df[opp_col] - ev_df[bet_col]).mean()
            else:
                row[SG_LABELS[cat]] = 0
        heatmap_data.append(row)
    hm_df = pd.DataFrame(heatmap_data).set_index("event")
    if hm_df.empty:
        return go.Figure().update_layout(**PLOT_LAYOUT, title=f"{title} — No data")
    fig = go.Figure(go.Heatmap(
        z=hm_df.values, x=hm_df.columns.tolist(), y=hm_df.index.tolist(),
        colorscale="RdYlGn_r", text=np.round(hm_df.values, 2), texttemplate="%{text}",
        hovertemplate="Event: %{y}<br>Category: %{x}<br>SG Delta: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(**PLOT_LAYOUT, title=title, xaxis_title="Category", yaxis_title="Event")
    return fig


layout = dbc.Container([
    html.H4("Luck & Attribution", className="page-header"),

    # Filters
    dbc.Row([
        dbc.Col([
            html.Label("Event", className="form-label small text-muted"),
            dcc.Dropdown(
                id="luck-event-filter",
                options=[],
                value=None,
                multi=True,
                placeholder="All events",
                className="dash-dropdown-dark",
            ),
        ], md=3),
        dbc.Col([
            html.Label("Bet Type", className="form-label small text-muted"),
            dcc.Dropdown(
                id="luck-type-filter",
                options=[
                    {"label": "Matchups Only", "value": "matchups"},
                    {"label": "Tournament Matchup", "value": "tournament_matchup"},
                    {"label": "Round Matchup", "value": "round_matchup"},
                    {"label": "Finish Position", "value": "finish_position"},
                ],
                multi=True,
                placeholder="All types",
                className="dash-dropdown-dark",
            ),
        ], md=2),
        dbc.Col([
            html.Label("Round", className="form-label small text-muted"),
            dcc.Dropdown(
                id="luck-round-filter",
                options=[
                    {"label": "R1", "value": "1"},
                    {"label": "R2", "value": "2"},
                    {"label": "R3", "value": "3"},
                    {"label": "R4", "value": "4"},
                    {"label": "Tournament", "value": "tournament"},
                ],
                multi=True,
                placeholder="All rounds",
                className="dash-dropdown-dark",
            ),
        ], md=2),
        dbc.Col([
            html.Label("SG Charts", className="form-label small text-muted"),
            dcc.Dropdown(
                id="luck-sg-result-filter",
                options=[
                    {"label": "Losses", "value": "loss"},
                    {"label": "Wins", "value": "win"},
                ],
                multi=True,
                placeholder="All results",
                className="dash-dropdown-dark",
            ),
        ], md=2),
    ], className="mb-3"),

    # KPI cards
    html.Div(id="luck-kpis"),

    # Charts row 1
    dbc.Row([
        dbc.Col(dcc.Graph(id="luck-margin-hist"), md=3),
        dbc.Col(dcc.Graph(id="luck-round-mu-hist"), md=3),
        dbc.Col(dcc.Graph(id="luck-close-by-event"), md=3),
        dbc.Col(dcc.Graph(id="luck-margin-odds"), md=3),
    ], className="mb-3"),

    # Charts row 2 — SG breakdown
    dbc.Row([
        dbc.Col(dcc.Graph(id="luck-sg-waterfall"), md=6),
        dbc.Col(dcc.Graph(id="luck-sg-heatmap"), md=6),
    ], className="mb-3"),

    # Charts row 3 — Best Sharp Price R2-R4
    html.H5("Best Sharp Price Only (R2-R4 Matchups)", className="mt-3 mb-2"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="luck-sg-waterfall-sharp"), md=6),
        dbc.Col(dcc.Graph(id="luck-sg-heatmap-sharp"), md=6),
    ], className="mb-3"),

    # Detail table
    html.H5("Detail", className="mt-3 mb-2"),
    html.Div(id="luck-table"),

], fluid=True)


@callback(
    Output("luck-event-filter", "options"),
    Input("luck-type-filter", "value"),  # trigger on page load
)
def populate_events(_):
    df = get_luck_attribution()
    if df.empty or "event_name" not in df.columns:
        return []
    events = sorted(df["event_name"].dropna().unique())
    return [{"label": e.title(), "value": e} for e in events]


@callback(
    Output("luck-kpis", "children"),
    Output("luck-margin-hist", "figure"),
    Output("luck-round-mu-hist", "figure"),
    Output("luck-close-by-event", "figure"),
    Output("luck-margin-odds", "figure"),
    Output("luck-sg-waterfall", "figure"),
    Output("luck-sg-heatmap", "figure"),
    Output("luck-sg-waterfall-sharp", "figure"),
    Output("luck-sg-heatmap-sharp", "figure"),
    Output("luck-table", "children"),
    Input("luck-event-filter", "value"),
    Input("luck-type-filter", "value"),
    Input("luck-round-filter", "value"),
    Input("luck-sg-result-filter", "value"),
)
def update_luck(event, bet_type, round_filter, sg_result_filter):
    empty_fig = go.Figure().update_layout(**PLOT_LAYOUT, title="No data")
    empty = (
        dbc.Alert("No graded bets with attribution data found.", color="warning"),
        empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
        empty_fig, empty_fig,
        dbc.Alert("No data", color="secondary"),
    )

    df = get_luck_attribution()
    if df.empty:
        return empty

    # Apply filters
    if event:
        events = event if isinstance(event, list) else [event]
        df = df[df["event_name"].isin(events)]
    if bet_type:
        bts = bet_type if isinstance(bet_type, list) else [bet_type]
        # Expand "matchups" shorthand
        expanded = []
        for bt in bts:
            if bt == "matchups":
                expanded.extend(["tournament_matchup", "round_matchup"])
            else:
                expanded.append(bt)
        df = df[df["bet_type"].isin(expanded)]
    if round_filter:
        rounds = [str(r) for r in (round_filter if isinstance(round_filter, list) else [round_filter])]
        df = df[df["round"].astype(str).isin(rounds)]

    if df.empty:
        return empty

    # ── KPI Cards ─────────────────────────────────────────────────────────
    has_margin = df["margin"].notna() if "margin" in df.columns else pd.Series(False, index=df.index)
    margin_df = df[has_margin].copy() if has_margin.any() else pd.DataFrame()

    close_wins = len(margin_df[(margin_df["result"].isin(["win", "win_dh"])) & (margin_df["margin"].abs() <= 1)]) if not margin_df.empty else 0
    close_losses = margin_df[(margin_df["result"] == "loss") & (margin_df["margin"].abs() <= 1)] if not margin_df.empty else pd.DataFrame()
    close_loss_count = len(close_losses)
    close_loss_pnl = close_losses["units_won"].sum() if not close_losses.empty else 0

    unlucky_pushes = df[df.get("is_unlucky_push", pd.Series(False, index=df.index)) == True] if "is_unlucky_push" in df.columns else pd.DataFrame()
    unlucky_count = len(unlucky_pushes)

    wins_with_margin = margin_df[margin_df["result"].isin(["win", "win_dh"])] if not margin_df.empty else pd.DataFrame()
    avg_win_margin = wins_with_margin["margin"].mean() if not wins_with_margin.empty else 0

    # Worst SG category in losses
    worst_cat = "N/A"
    losses = df[df["result"] == "loss"]
    if not losses.empty:
        cat_deltas = {}
        for cat in SG_CATS:
            opp_col = f"opp_{cat}"
            bet_col = f"bet_on_{cat}"
            if opp_col in losses.columns and bet_col in losses.columns:
                delta = (losses[opp_col] - losses[bet_col]).mean()
                if pd.notna(delta):
                    cat_deltas[cat] = delta
        if cat_deltas:
            worst = max(cat_deltas, key=cat_deltas.get)
            worst_cat = f"{SG_LABELS[worst]} ({cat_deltas[worst]:+.2f})"

    kpis = stat_card_row([
        {"title": "Close Wins (1 stroke)", "value": str(close_wins), "color": "success"},
        {"title": "Close Losses (1 stroke)", "value": str(close_loss_count),
         "subtitle": f"P&L: {close_loss_pnl:+.1f}u", "color": "danger"},
        {"title": "Unlucky Pushes (+130+)", "value": str(unlucky_count), "color": "warning"},
        {"title": "Avg Win Margin", "value": f"{avg_win_margin:+.1f}", "color": "info"},
        {"title": "Worst Category (Losses)", "value": worst_cat, "color": "secondary"},
    ])

    # ── Chart 1: Margin Distribution Histogram ────────────────────────────
    if not margin_df.empty:
        fig_hist = go.Figure()
        for result, color in [("win", "#2ca02c"), ("win_dh", "#2ca02c"), ("loss", "#d62728"), ("push", "#7f7f7f")]:
            subset = margin_df[margin_df["result"] == result]
            if not subset.empty:
                fig_hist.add_trace(go.Histogram(
                    x=subset["margin"],
                    name=result.replace("_", " ").title(),
                    marker_color=color,
                    opacity=0.7,
                    xbins=dict(size=1),
                ))
        fig_hist.update_layout(
            **PLOT_LAYOUT,
            title="Margin Distribution (strokes)",
            xaxis_title="Margin (+ = our player better)",
            yaxis_title="Count",
            barmode="stack",
            showlegend=True,
        )
    else:
        fig_hist = go.Figure().update_layout(**PLOT_LAYOUT, title="Margin Distribution — No data")

    # ── Chart 1b: Round Matchup Margin Distribution ─────────────────────
    round_mu = margin_df[margin_df["bet_type"] == "round_matchup"] if not margin_df.empty and "bet_type" in margin_df.columns else pd.DataFrame()
    if not round_mu.empty:
        fig_round_hist = go.Figure()
        for result, color in [("win", "#2ca02c"), ("loss", "#d62728"), ("push", "#7f7f7f")]:
            subset = round_mu[round_mu["result"] == result]
            if not subset.empty:
                fig_round_hist.add_trace(go.Histogram(
                    x=subset["margin"],
                    name=result.title(),
                    marker_color=color,
                    opacity=0.7,
                    xbins=dict(size=1),
                ))
        fig_round_hist.update_layout(
            **PLOT_LAYOUT,
            title="Round Matchup Margins",
            xaxis_title="Margin (strokes)",
            yaxis_title="Count",
            barmode="stack",
            showlegend=True,
        )
    else:
        fig_round_hist = go.Figure().update_layout(**PLOT_LAYOUT, title="Round Matchup Margins — No data")

    # ── Chart 2: Close Outcomes by Event ──────────────────────────────────
    if not margin_df.empty and "event_name" in margin_df.columns:
        margin_df = margin_df.copy()
        abs_margin = margin_df["margin"].abs()
        margin_df["closeness"] = np.where(abs_margin <= 1, "1 stroke",
                                  np.where(abs_margin <= 2, "2 strokes", "3+ strokes"))
        event_close = margin_df.groupby(["event_name", "closeness"]).size().reset_index(name="count")
        fig_close = px.bar(
            event_close,
            x="event_name",
            y="count",
            color="closeness",
            color_discrete_map={"1 stroke": "#ff7f0e", "2 strokes": "#1f77b4", "3+ strokes": "#2ca02c"},
            barmode="stack",
        )
        fig_close.update_layout(
            **PLOT_LAYOUT,
            title="Close Outcomes by Event",
            xaxis_title="Event",
            yaxis_title="Bets",
        )
    else:
        fig_close = go.Figure().update_layout(**PLOT_LAYOUT, title="Close Outcomes — No data")

    # ── Chart 3: Margin vs Odds Scatter ───────────────────────────────────
    if not margin_df.empty and "book_odds" in margin_df.columns:
        scatter_df = margin_df.dropna(subset=["book_odds"])
        if not scatter_df.empty:
            colors = scatter_df["result"].map(RESULT_COLORS).fillna("#7f7f7f")
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=scatter_df["book_odds"],
                y=scatter_df["margin"],
                mode="markers",
                marker=dict(color=colors, size=8, opacity=0.7),
                text=scatter_df.apply(
                    lambda r: f"{r.get('bet_on', '')} vs {r.get('opponent', '')}<br>{r['result']}", axis=1
                ),
                hovertemplate="%{text}<br>Odds: %{x}<br>Margin: %{y}<extra></extra>",
            ))
            # Highlight unlucky pushes
            if "is_unlucky_push" in scatter_df.columns:
                unlucky = scatter_df[scatter_df["is_unlucky_push"] == True]
                if not unlucky.empty:
                    fig_scatter.add_trace(go.Scatter(
                        x=unlucky["book_odds"],
                        y=unlucky["margin"],
                        mode="markers",
                        marker=dict(color="yellow", size=12, symbol="diamond-open", line=dict(width=2)),
                        name="Unlucky Push",
                    ))
            fig_scatter.update_layout(
                **PLOT_LAYOUT,
                title="Margin vs Book Odds",
                xaxis_title="Book Odds (American)",
                yaxis_title="Margin (strokes)",
                showlegend=True,
            )
        else:
            fig_scatter = go.Figure().update_layout(**PLOT_LAYOUT, title="Margin vs Odds — No data")
    else:
        fig_scatter = go.Figure().update_layout(**PLOT_LAYOUT, title="Margin vs Odds — No data")

    # ── SG result filter ────────────────────────────────────────────────
    sg_filters = sg_result_filter if isinstance(sg_result_filter, list) else ([sg_result_filter] if sg_result_filter else [])
    if not sg_filters:
        sg_df = df[df["result"].isin(["win", "win_dh", "loss"])]
        sg_label = "All"
    else:
        result_vals = []
        if "loss" in sg_filters:
            result_vals.append("loss")
        if "win" in sg_filters:
            result_vals.extend(["win", "win_dh"])
        sg_df = df[df["result"].isin(result_vals)]
        sg_label = " & ".join(sg_filters).title()

    fig_waterfall = _sg_waterfall(sg_df, f"SG Delta in {sg_label} (Opp - Us)")
    fig_heatmap = _sg_heatmap(sg_df, f"SG Delta in {sg_label} by Event (Opp - Us)")

    # ── Best Sharp Price R2-R4 ───────────────────────────────────────────
    sharp_df = df[
        (df["bet_type"] == "round_matchup") &
        (df["round"].astype(str).isin(["2", "3", "4"])) &
        (df["bookmaker"].str.lower().isin(SHARP_BOOKS))
    ].copy() if "bookmaker" in df.columns else pd.DataFrame()

    if not sharp_df.empty:
        # Keep best odds per unique matchup
        sharp_df["book_odds_num"] = pd.to_numeric(sharp_df["book_odds"], errors="coerce")
        sharp_df = sharp_df.sort_values("book_odds_num", ascending=False)
        sharp_df = sharp_df.drop_duplicates(
            subset=["event_id", "round", "bet_on", "opponent"], keep="first"
        )
        sharp_df = sharp_df.drop(columns=["book_odds_num"])

        # Apply same result filter
        if not sg_filters:
            sharp_sg = sharp_df[sharp_df["result"].isin(["win", "win_dh", "loss"])]
        else:
            sharp_sg = sharp_df[sharp_df["result"].isin(result_vals)]

        n_sharp = len(sharp_df)
        fig_wf_sharp = _sg_waterfall(sharp_sg, f"Best Sharp R2-R4 — {sg_label} (n={n_sharp})")
        fig_hm_sharp = _sg_heatmap(sharp_sg, f"Best Sharp R2-R4 — {sg_label} by Event")
    else:
        fig_wf_sharp = go.Figure().update_layout(**PLOT_LAYOUT, title="Best Sharp R2-R4 — No data")
        fig_hm_sharp = go.Figure().update_layout(**PLOT_LAYOUT, title="Best Sharp R2-R4 — No data")

    # ── Detail Table ──────────────────────────────────────────────────────
    table_cols = ["event_name", "bet_on", "opponent", "round", "result", "margin",
                  "book_odds", "bookmaker", "edge", "units_won"]
    for cat in SG_CATS:
        table_cols.extend([f"bet_on_{cat}", f"opp_{cat}"])

    existing_cols = [c for c in table_cols if c in df.columns]
    table_df = df[existing_cols].copy()

    # Add SG diff total
    if all(f"bet_on_{c}" in table_df.columns and f"opp_{c}" in table_df.columns for c in SG_CATS):
        table_df["sg_diff_total"] = sum(
            table_df[f"opp_{c}"] - table_df[f"bet_on_{c}"] for c in SG_CATS
        )

    table = make_grid(table_df, id_suffix="luck", height=500, page_size=20)

    return (kpis, fig_hist, fig_round_hist, fig_close, fig_scatter,
            fig_waterfall, fig_heatmap, fig_wf_sharp, fig_hm_sharp, table)
