"""Performance page — Historical P&L, ROI breakdowns."""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from dashboard.data_layer import get_bet_ledger
from dashboard.components.stat_cards import stat_card_row
from dashboard.components.filters import sportsbook_filter, bet_type_selector, edge_slider, event_selector, round_selector
from dashboard.components.tables import make_grid
from dashboard.config import SHARP_BOOKS, COLOR_WIN, COLOR_LOSS, COLOR_PUSH, COLOR_SHARP, COLOR_RETAIL

dash.register_page(__name__, path="/performance", title="Performance", order=6)

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,33,62,0.8)",
    font=dict(color="#e0e0e0"),
    margin=dict(l=50, r=30, t=40, b=40),
)

# 1 unit = $200 — finish position bets store dollar amounts, convert to units
UNIT_SIZE = 200.0


def _american_to_prob(odds):
    """Convert American odds Series to implied probabilities."""
    prob = pd.Series(np.nan, index=odds.index)
    pos = odds >= 0
    prob[pos] = 100.0 / (odds[pos] + 100.0)
    neg = odds < 0
    prob[neg] = odds[neg].abs() / (odds[neg].abs() + 100.0)
    return prob


def _american_to_decimal(odds):
    """Convert American odds Series to decimal odds."""
    dec = pd.Series(np.nan, index=odds.index)
    pos = odds >= 0
    dec[pos] = odds[pos] / 100.0 + 1.0
    neg = odds < 0
    dec[neg] = 100.0 / odds[neg].abs() + 1.0
    return dec


layout = dbc.Container([
    html.H4("Performance", className="page-header"),

    # Filters — Row 1 (event dropdown populated by callback on page load)
    dbc.Row([
        event_selector("perf", events=[]),
        bet_type_selector("perf"),
        sportsbook_filter("perf"),
        edge_slider("perf", default=0),
    ], className="mb-2"),

    # Filters — Row 2: Round, Sample & Pred ranges
    dbc.Row([
        dbc.Col([
            html.Label("Round", className="form-label small text-muted"),
            dcc.Dropdown(
                id="perf-round-filter",
                options=[
                    {"label": "All Rounds", "value": "all"},
                    {"label": "R1", "value": "1"},
                    {"label": "R2", "value": "2"},
                    {"label": "R3", "value": "3"},
                    {"label": "R4", "value": "4"},
                ],
                value="all",
                clearable=False,
                className="dash-dropdown-dark",
            ),
        ], md=2),
        dbc.Col([
            html.Label("Sample Size", className="form-label small text-muted"),
            dbc.InputGroup([
                dbc.InputGroupText("Min", className="bg-dark text-light border-secondary"),
                dbc.Input(id="perf-sample-min", type="number", placeholder="0",
                          value=None, min=0, step=1,
                          className="bg-dark text-light border-secondary"),
                dbc.InputGroupText("Max", className="bg-dark text-light border-secondary"),
                dbc.Input(id="perf-sample-max", type="number", placeholder="any",
                          value=None, min=0, step=1,
                          className="bg-dark text-light border-secondary"),
            ], size="sm"),
        ], md=4),
        dbc.Col([
            html.Label("Pred (Skill Estimate)", className="form-label small text-muted"),
            dbc.InputGroup([
                dbc.InputGroupText("Min", className="bg-dark text-light border-secondary"),
                dbc.Input(id="perf-pred-min", type="number", placeholder="0",
                          value=None, min=0, step=0.1,
                          className="bg-dark text-light border-secondary"),
                dbc.InputGroupText("Max", className="bg-dark text-light border-secondary"),
                dbc.Input(id="perf-pred-max", type="number", placeholder="any",
                          value=None, min=0, step=0.1,
                          className="bg-dark text-light border-secondary"),
            ], size="sm"),
        ], md=4),
    ], className="mb-2"),

    # Filters — Row 3: Raw Edge, Decimal Odds, Player
    dbc.Row([
        dbc.Col([
            html.Label("Raw % Edge", className="form-label small text-muted"),
            dbc.InputGroup([
                dbc.InputGroupText("Min", className="bg-dark text-light border-secondary"),
                dbc.Input(id="perf-raw-edge-min", type="number", placeholder="0",
                          value=None, step=0.5,
                          className="bg-dark text-light border-secondary"),
                dbc.InputGroupText("Max", className="bg-dark text-light border-secondary"),
                dbc.Input(id="perf-raw-edge-max", type="number", placeholder="any",
                          value=None, step=0.5,
                          className="bg-dark text-light border-secondary"),
            ], size="sm"),
        ], md=3),
        dbc.Col([
            html.Label("Decimal Odds", className="form-label small text-muted"),
            dbc.InputGroup([
                dbc.InputGroupText("Min", className="bg-dark text-light border-secondary"),
                dbc.Input(id="perf-dec-odds-min", type="number", placeholder="1.0",
                          value=None, min=1.0, step=0.1,
                          className="bg-dark text-light border-secondary"),
                dbc.InputGroupText("Max", className="bg-dark text-light border-secondary"),
                dbc.Input(id="perf-dec-odds-max", type="number", placeholder="any",
                          value=None, min=1.0, step=0.1,
                          className="bg-dark text-light border-secondary"),
            ], size="sm"),
        ], md=3),
        dbc.Col([
            html.Label("Player", className="form-label small text-muted"),
            dcc.Dropdown(
                id="perf-player-filter",
                options=[],
                value=None,
                multi=True,
                placeholder="All players",
                className="dash-dropdown-dark",
            ),
        ], md=4),
    ], className="mb-3"),

    # KPI cards
    html.Div(id="perf-kpi-row"),

    # Charts
    dbc.Row([
        dbc.Col(dcc.Graph(id="perf-cumulative-chart"), md=4),
        dbc.Col(dcc.Graph(id="perf-book-roi-chart"), md=4),
        dbc.Col(dcc.Graph(id="perf-pnl-chart"), md=4),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="perf-raw-edge-bucket-chart"), md=4),
        dbc.Col(dcc.Graph(id="perf-edge-bucket-chart"), md=4),
        dbc.Col(dcc.Graph(id="perf-scatter-chart"), md=4),
    ]),

    # Summary table
    html.H5("Event Summary", className="mt-4 mb-2"),
    html.Div(id="perf-summary-table"),

    # Filtered bets detail table
    html.H5("Filtered Bets", className="mt-4 mb-2"),
    html.Div(id="perf-remaining-table"),
], fluid=True)


@callback(
    Output("perf-event-filter", "options"),
    Input("perf-event-filter", "id"),  # on load trigger
)
def populate_events(_):
    df = get_bet_ledger()
    if df.empty:
        return []
    events = sorted(df["event_name"].dropna().unique().tolist())
    return [{"label": e.title(), "value": e} for e in events]


def _convert_to_units(df):
    """Convert finish position dollar amounts to units ($200 = 1 unit).

    Matchup bets are already in unit terms (flat 1-unit wagers).
    Finish position bets store raw dollar amounts from kelly-stake sizing.
    """
    df = df.copy()

    is_finish = df["bet_type"] == "finish_position"

    if "units_won" in df.columns:
        df.loc[is_finish, "units_won"] = df.loc[is_finish, "units_won"] / UNIT_SIZE

    if "units_wagered" in df.columns:
        df.loc[is_finish, "units_wagered"] = df.loc[is_finish, "units_wagered"] / UNIT_SIZE

    return df


@callback(
    Output("perf-kpi-row", "children"),
    Output("perf-cumulative-chart", "figure"),
    Output("perf-book-roi-chart", "figure"),
    Output("perf-pnl-chart", "figure"),
    Output("perf-raw-edge-bucket-chart", "figure"),
    Output("perf-edge-bucket-chart", "figure"),
    Output("perf-scatter-chart", "figure"),
    Output("perf-summary-table", "children"),
    Output("perf-remaining-table", "children"),
    Output("perf-player-filter", "options"),
    Input("perf-event-filter", "value"),
    Input("perf-type-filter", "value"),
    Input("perf-book-filter", "value"),
    Input("perf-edge-slider", "value"),
    Input("perf-round-filter", "value"),
    Input("perf-sample-min", "value"),
    Input("perf-sample-max", "value"),
    Input("perf-pred-min", "value"),
    Input("perf-pred-max", "value"),
    Input("perf-raw-edge-min", "value"),
    Input("perf-raw-edge-max", "value"),
    Input("perf-dec-odds-min", "value"),
    Input("perf-dec-odds-max", "value"),
    Input("perf-player-filter", "value"),
)
def update_performance(event, bet_type, books, min_edge, round_filter,
                       sample_min, sample_max, pred_min, pred_max,
                       raw_edge_min, raw_edge_max, dec_odds_min, dec_odds_max,
                       player_filter):
    empty_fig = go.Figure(layout={**PLOT_LAYOUT, "title": "No data"})
    alert = dbc.Alert("No bet data found. Run the simulation pipeline first.", color="warning")
    empty_return = (alert, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
                    empty_fig, alert, "", [])

    filters = {}
    if event:
        filters["event"] = event
    if bet_type and bet_type != "all":
        filters["bet_type"] = bet_type
    if min_edge:
        filters["min_edge"] = min_edge
    if books:
        filters["books"] = books

    df = get_bet_ledger(**filters)

    # Apply round filter
    if round_filter and round_filter != "all" and "round" in df.columns:
        df = df[df["round"].astype(str).str.strip() == str(round_filter)]

    # Apply sample_on range filter
    if (sample_min is not None or sample_max is not None) and "sample_on" in df.columns:
        if sample_min is not None:
            df = df[df["sample_on"].fillna(0) >= sample_min]
        if sample_max is not None:
            df = df[df["sample_on"].fillna(0) <= sample_max]

    # Apply pred_on range filter
    if (pred_min is not None or pred_max is not None) and "pred_on" in df.columns:
        if pred_min is not None:
            df = df[df["pred_on"].fillna(0) >= pred_min]
        if pred_max is not None:
            df = df[df["pred_on"].fillna(0) <= pred_max]

    if df.empty:
        return empty_return

    # Derive units_wagered from kelly_stake for finish pos, default 1.0 for matchups
    if "units_wagered" not in df.columns:
        if "kelly_stake" in df.columns:
            df["units_wagered"] = df["kelly_stake"].fillna(1.0)
        else:
            df["units_wagered"] = 1.0
    if "units_won" not in df.columns:
        df["units_won"] = np.nan

    # Convert finish position dollar amounts to units ($200 = 1 unit)
    df = _convert_to_units(df)

    # Pre-compute raw_edge and dec_odds on full df (before new filters)
    if "book_odds" in df.columns and "fair_odds" in df.columns:
        market_prob = _american_to_prob(df["book_odds"])
        my_prob = _american_to_prob(df["fair_odds"])
        df["raw_edge"] = (my_prob - market_prob) * 100
        df["dec_odds"] = _american_to_decimal(df["book_odds"])
    else:
        df["raw_edge"] = np.nan
        df["dec_odds"] = np.nan

    # Build player options BEFORE player filter is applied
    players = sorted(df["bet_on"].dropna().unique().tolist())
    player_options = [{"label": p.title(), "value": p} for p in players]

    # Apply new filters (raw_edge, dec_odds, player)
    if raw_edge_min is not None:
        df = df[df["raw_edge"].fillna(0) >= raw_edge_min]
    if raw_edge_max is not None:
        df = df[df["raw_edge"].fillna(0) <= raw_edge_max]
    if dec_odds_min is not None:
        df = df[df["dec_odds"].fillna(0) >= dec_odds_min]
    if dec_odds_max is not None:
        df = df[df["dec_odds"].fillna(999) <= dec_odds_max]
    if player_filter:
        df = df[df["bet_on"].isin(player_filter)]

    if df.empty:
        return empty_return[:-1] + (player_options,)

    # Separate graded vs all
    graded = df[df["result"].astype(str).str.strip() != ""].copy()
    resolved = graded[~graded["result"].isin(["no_data", "unknown", "duplicate"])].copy()

    # KPI cards
    total_bets = len(df)
    if not resolved.empty:
        wins = len(resolved[resolved["result"].isin(["win", "win_dh"])])
        losses = len(resolved[resolved["result"] == "loss"])
        pushes = len(resolved[resolved["result"] == "push"])
        wagered = resolved["units_wagered"].sum()
        won = resolved["units_won"].sum()
        roi = won / wagered * 100 if wagered > 0 else 0
        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    else:
        wins = losses = pushes = 0
        wagered = won = roi = win_rate = 0

    kpi = stat_card_row([
        {"title": "Total Bets", "value": str(total_bets), "color": "primary"},
        {"title": "Record", "value": f"{wins}W-{losses}L-{pushes}P", "color": "info"},
        {"title": "Win Rate", "value": f"{win_rate:.1f}%", "color": "info"},
        {"title": "Units Won", "value": f"{won:+.2f}u", "color": "success" if won >= 0 else "danger"},
        {"title": "ROI", "value": f"{roi:+.1f}%", "color": "success" if roi >= 0 else "danger"},
    ])

    # ── Chart 1: P&L by Event (dot chart) ──
    fig1 = go.Figure(layout={**PLOT_LAYOUT, "title": "P&L by Event"})
    if not resolved.empty:
        bet_types = [
            ("tournament_matchup", "Tourn MU", "#1f77b4", "circle"),
            ("round_matchup", "Round MU", "#ff7f0e", "diamond"),
            ("finish_position", "Finish Pos", "#2ca02c", "square"),
        ]
        events_order = (
            resolved.groupby("event_name")["run_timestamp"]
            .min().sort_values().index.tolist()
        )

        for bt, name, color, symbol in bet_types:
            sub = resolved[resolved["bet_type"] == bt]
            if sub.empty:
                continue
            event_pnl = sub.groupby("event_name")["units_won"].sum()
            events_with_data = [e for e in events_order if e in event_pnl.index]
            values = [event_pnl[e] for e in events_with_data]
            fig1.add_trace(go.Scatter(
                x=events_with_data, y=values,
                mode="markers", name=name,
                marker=dict(color=color, size=12, symbol=symbol, line=dict(width=1, color="white")),
            ))

        # Total dot per event
        total_pnl = resolved.groupby("event_name")["units_won"].sum()
        total_vals = [total_pnl.get(e, 0) for e in events_order]
        fig1.add_trace(go.Scatter(
            x=events_order, y=total_vals,
            mode="markers", name="Total",
            marker=dict(color="white", size=14, symbol="star", line=dict(width=1, color="#4ecca3")),
        ))

        fig1.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig1.update_yaxes(title_text="Units")

    # ── Chart 2: ROI by bookmaker ──
    fig2 = go.Figure(layout={**PLOT_LAYOUT, "title": "ROI by Bookmaker"})
    if not resolved.empty:
        book_stats = (
            resolved.groupby("bookmaker")
            .agg(wagered=("units_wagered", "sum"), won=("units_won", "sum"), count=("bet_on", "count"))
            .reset_index()
        )
        book_stats["roi"] = book_stats["won"] / book_stats["wagered"] * 100
        book_stats = book_stats[book_stats["count"] >= 3].sort_values("roi")

        colors = []
        for _, row in book_stats.iterrows():
            b = str(row["bookmaker"]).lower()
            if any(s in b for s in ["pinnacle", "betonline", "betcris", "bookmaker"]):
                colors.append(COLOR_SHARP)
            elif any(r in b for r in ["fanduel", "draftkings", "caesars", "betmgm"]):
                colors.append(COLOR_RETAIL)
            else:
                colors.append("#7f7f7f")

        fig2.add_trace(go.Bar(
            y=book_stats["bookmaker"], x=book_stats["roi"],
            orientation="h", marker_color=colors,
            text=[f"{r:.1f}%" for r in book_stats["roi"]], textposition="auto",
        ))

    # ── Chart 3: Win rate + ROI by edge bucket ──
    fig3 = go.Figure(layout={**PLOT_LAYOUT, "title": "Performance by Edge Bucket"})
    if not resolved.empty:
        buckets = [(3, 5, "3-5%"), (5, 8, "5-8%"), (8, 100, "8%+")]
        labels, wrs, rois_list = [], [], []
        for lo, hi, label in buckets:
            sub = resolved[(resolved["edge"] >= lo) & (resolved["edge"] < hi)]
            w = len(sub[sub["result"].isin(["win", "win_dh"])]) if not sub.empty else 0
            l = len(sub[sub["result"] == "loss"]) if not sub.empty else 0
            wr = w / (w + l) * 100 if (w + l) > 0 else 0
            wag = sub["units_wagered"].sum() if not sub.empty else 0
            r = sub["units_won"].sum() / wag * 100 if wag > 0 else 0
            labels.append(label)
            wrs.append(wr)
            rois_list.append(r)

        fig3.add_trace(go.Bar(x=labels, y=wrs, name="Win Rate %", marker_color=COLOR_WIN))
        fig3.add_trace(go.Bar(x=labels, y=rois_list, name="ROI %", marker_color="#d62728"))
        fig3.update_layout(barmode="group")

    # ── Chart 3b: Performance by Raw % Edge Bucket ──
    fig3b = go.Figure(layout={**PLOT_LAYOUT, "title": "Performance by Raw % Edge"})
    if not resolved.empty and "raw_edge" in resolved.columns:
        raw_valid = resolved.dropna(subset=["raw_edge"])
        if not raw_valid.empty:
            buckets = [(0, 2, "0-2%"), (2, 4, "2-4%"), (4, 6, "4-6%"), (6, 100, "6%+")]
            labels, wrs, rois_list = [], [], []
            for lo, hi, label in buckets:
                sub = raw_valid[(raw_valid["raw_edge"] >= lo) & (raw_valid["raw_edge"] < hi)]
                w = len(sub[sub["result"].isin(["win", "win_dh"])]) if not sub.empty else 0
                l = len(sub[sub["result"] == "loss"]) if not sub.empty else 0
                wr = w / (w + l) * 100 if (w + l) > 0 else 0
                wag = sub["units_wagered"].sum() if not sub.empty else 0
                r = sub["units_won"].sum() / wag * 100 if wag > 0 else 0
                labels.append(label)
                wrs.append(wr)
                rois_list.append(r)

            fig3b.add_trace(go.Bar(x=labels, y=wrs, name="Win Rate %", marker_color=COLOR_WIN))
            fig3b.add_trace(go.Bar(x=labels, y=rois_list, name="ROI %", marker_color="#d62728"))
            fig3b.update_layout(barmode="group")
            fig3b.update_xaxes(title_text="Raw Prob Edge (pp)")
            fig3b.update_yaxes(title_text="%")

    # ── Chart 4: Performance by Odds Bucket ──
    fig4 = go.Figure(layout={**PLOT_LAYOUT, "title": "Performance by Odds Bucket"})
    if not resolved.empty and "dec_odds" in resolved.columns:
        odds_valid = resolved.dropna(subset=["dec_odds"])
        if not odds_valid.empty:
            buckets = [
                (0, 2.0, "< 2.0"),
                (2.0, 2.5, "2.0–2.5"),
                (2.5, 3.5, "2.5–3.5"),
                (3.5, 8.0, "3.5–8.0"),
                (8.0, 999, "8.0+"),
            ]
            labels, wrs, rois_list, counts = [], [], [], []
            for lo, hi, label in buckets:
                sub = odds_valid[(odds_valid["dec_odds"] >= lo) & (odds_valid["dec_odds"] < hi)]
                w = len(sub[sub["result"].isin(["win", "win_dh"])]) if not sub.empty else 0
                l = len(sub[sub["result"] == "loss"]) if not sub.empty else 0
                wr = w / (w + l) * 100 if (w + l) > 0 else 0
                wag = sub["units_wagered"].sum() if not sub.empty else 0
                r = sub["units_won"].sum() / wag * 100 if wag > 0 else 0
                labels.append(label)
                wrs.append(wr)
                rois_list.append(r)
                counts.append(w + l)

            fig4.add_trace(go.Bar(
                x=labels, y=wrs, name="Win Rate %", marker_color=COLOR_WIN,
                text=[f"{v:.0f}%" for v in wrs], textposition="auto",
            ))
            fig4.add_trace(go.Bar(
                x=labels, y=rois_list, name="ROI %", marker_color="#d62728",
                text=[f"{v:+.0f}%" for v in rois_list], textposition="auto",
            ))
            fig4.update_layout(barmode="group")
            fig4.update_xaxes(title_text="Decimal Odds")
            fig4.update_yaxes(title_text="%")

    # ── Chart 5: Cumulative P&L ──
    fig5 = go.Figure(layout={**PLOT_LAYOUT, "title": "Cumulative P&L"})
    if not resolved.empty:
        sorted_df = resolved.sort_values("run_timestamp").reset_index(drop=True)
        cum_pnl = sorted_df["units_won"].cumsum()
        fig5.add_trace(go.Scatter(
            x=list(range(1, len(cum_pnl) + 1)), y=cum_pnl.values,
            mode="lines", name="Cumulative P&L",
            line=dict(color="#4ecca3", width=2),
            fill="tozeroy",
            fillcolor="rgba(78,204,163,0.15)",
        ))
        fig5.add_hline(y=0, line_dash="dash", line_color="gray")
        fig5.update_xaxes(title_text="Bet #")
        fig5.update_yaxes(title_text="Units")

    # ── Summary table ──
    summary_content = dbc.Alert("No resolved bets.", color="secondary")
    if not resolved.empty:
        event_summary = (
            resolved.groupby("event_name")
            .agg(
                bets=("bet_on", "count"),
                wins=("result", lambda x: (x.isin(["win", "win_dh"])).sum()),
                losses=("result", lambda x: (x == "loss").sum()),
                wagered=("units_wagered", "sum"),
                won=("units_won", "sum"),
            )
            .reset_index()
        )
        event_summary["roi"] = (event_summary["won"] / event_summary["wagered"] * 100).round(1)
        event_summary["wagered"] = event_summary["wagered"].round(2)
        event_summary["won"] = event_summary["won"].round(2)

        summary_content = dbc.Table.from_dataframe(
            event_summary.sort_values("won", ascending=False),
            striped=True, bordered=True, hover=True, color="dark", size="sm",
        )

    # ── Filtered bets detail table ──
    detail_cols = ["bet_on", "opponent", "bookmaker", "bet_type", "round",
                   "edge", "raw_edge", "dec_odds", "pred_on", "result", "units_wagered", "units_won"]
    available_cols = [c for c in detail_cols if c in df.columns]
    detail_df = df[available_cols].copy()
    for col in ["edge", "raw_edge", "dec_odds", "pred_on", "units_wagered", "units_won"]:
        if col in detail_df.columns:
            detail_df[col] = detail_df[col].round(2)
    remaining_table = make_grid(detail_df, id_suffix="perf-remaining", height=500)

    return kpi, fig1, fig2, fig5, fig3b, fig3, fig4, summary_content, remaining_table, player_options
