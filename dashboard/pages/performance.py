"""Performance page — Historical P&L, ROI breakdowns."""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from dashboard.data_layer import get_bet_ledger, get_archetype_lookup
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
        sportsbook_filter("perf", default_all=False),
        edge_slider("perf", default=0),
    ], className="mb-2"),

    # Filters — Row 2: Round, Sample & Pred ranges
    dbc.Row([
        dbc.Col([
            html.Label("Round", className="form-label small text-muted"),
            dcc.Dropdown(
                id="perf-round-filter",
                options=[
                    {"label": "R1", "value": "1"},
                    {"label": "R2", "value": "2"},
                    {"label": "R3", "value": "3"},
                    {"label": "R4", "value": "4"},
                ],
                multi=True,
                placeholder="All rounds",
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

    # Filters — Row 3: Analysis mode, Raw Edge, Decimal Odds, Archetype, Player
    dbc.Row([
        dbc.Col([
            html.Label("Analysis", className="form-label small text-muted"),
            dcc.Dropdown(
                id="perf-analysis-mode",
                options=[
                    {"label": "All Bets", "value": "all"},
                    {"label": "Best Price", "value": "best_price"},
                    {"label": "Sharp Only", "value": "sharp_only"},
                    {"label": "Sharp Only, Best Price", "value": "sharp_best"},
                ],
                value="all",
                clearable=False,
                className="dash-dropdown-dark",
            ),
        ], md=2),
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
        ], md=2),
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
        ], md=2),
        dbc.Col([
            html.Label("Archetype", className="form-label small text-muted"),
            dcc.Dropdown(
                id="perf-archetype-filter",
                options=[],
                value=None,
                multi=True,
                placeholder="All archetypes",
                className="dash-dropdown-dark",
            ),
        ], md=2),
        dbc.Col([
            html.Label("Arch. Against", className="form-label small text-muted"),
            dcc.Dropdown(
                id="perf-archetype-against-filter",
                options=[],
                value=None,
                multi=True,
                placeholder="All archetypes",
                className="dash-dropdown-dark",
            ),
        ], md=2),
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
        ], md=2),
    ], className="mb-3"),

    # KPI cards
    html.Div(id="perf-kpi-row"),

    # Charts — Row 1: Core metrics
    dbc.Row([
        dbc.Col(dcc.Graph(id="perf-cumulative-chart"), md=4),
        dbc.Col(dcc.Graph(id="perf-pnl-chart"), md=4),
        dbc.Col(dcc.Graph(id="perf-book-roi-chart"), md=4),
    ]),
    # Charts — Row 2: Bucket breakdowns
    dbc.Row([
        dbc.Col(dcc.Graph(id="perf-bucket-chart"), md=12),
    ]),
    # Charts — Row 3: Archetype P&L
    dbc.Row([
        dbc.Col(dcc.Graph(id="perf-archetype-cumulative-chart"), md=6),
        dbc.Col(dcc.Graph(id="perf-archetype-bar-chart"), md=6),
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
    """Convert finish position kelly_stake dollars to units ($200 = 1 unit).

    Matchup bets are already in unit terms (flat 1-unit wagers).
    units_wagered is derived from kelly_stake (raw dollars) so needs conversion.
    units_won is recomputed for finish positions because older grading wrote
    flat 0/1 values instead of proper kelly-based amounts.
    """
    df = df.copy()

    is_finish = df["bet_type"].str.startswith("finish_position")

    if "units_wagered" in df.columns:
        df.loc[is_finish, "units_wagered"] = df.loc[is_finish, "units_wagered"] / UNIT_SIZE

    # Recompute units_won for finish positions from units_wagered + result + odds
    if "units_won" in df.columns and "units_wagered" in df.columns and "result" in df.columns:
        is_loss = is_finish & (df["result"] == "loss")
        df.loc[is_loss, "units_won"] = -df.loc[is_loss, "units_wagered"]

        is_win = is_finish & (df["result"].isin(["win", "win_dh"]))
        if "book_odds" in df.columns and is_win.any():
            book_odds = pd.to_numeric(df.loc[is_win, "book_odds"], errors="coerce")
            dec = pd.Series(np.nan, index=book_odds.index)
            pos = book_odds >= 0
            dec[pos] = book_odds[pos] / 100.0 + 1.0
            neg = book_odds < 0
            dec[neg] = 100.0 / book_odds[neg].abs() + 1.0
            df.loc[is_win, "units_won"] = df.loc[is_win, "units_wagered"] * (dec - 1)

    return df


@callback(
    Output("perf-kpi-row", "children"),
    Output("perf-cumulative-chart", "figure"),
    Output("perf-pnl-chart", "figure"),
    Output("perf-book-roi-chart", "figure"),
    Output("perf-bucket-chart", "figure"),
    Output("perf-archetype-cumulative-chart", "figure"),
    Output("perf-archetype-bar-chart", "figure"),
    Output("perf-summary-table", "children"),
    Output("perf-remaining-table", "children"),
    Output("perf-player-filter", "options"),
    Output("perf-archetype-filter", "options"),
    Output("perf-archetype-against-filter", "options"),
    Input("perf-event-filter", "value"),
    Input("perf-type-filter", "value"),
    Input("perf-book-filter", "value"),
    Input("perf-edge-slider", "value"),
    Input("perf-round-filter", "value"),
    Input("perf-sample-min", "value"),
    Input("perf-sample-max", "value"),
    Input("perf-pred-min", "value"),
    Input("perf-pred-max", "value"),
    Input("perf-analysis-mode", "value"),
    Input("perf-raw-edge-min", "value"),
    Input("perf-raw-edge-max", "value"),
    Input("perf-dec-odds-min", "value"),
    Input("perf-dec-odds-max", "value"),
    Input("perf-archetype-filter", "value"),
    Input("perf-archetype-against-filter", "value"),
    Input("perf-player-filter", "value"),
)
def update_performance(event, bet_type, books, min_edge, round_filter,
                       sample_min, sample_max, pred_min, pred_max,
                       analysis_mode,
                       raw_edge_min, raw_edge_max, dec_odds_min, dec_odds_max,
                       archetype_filter, archetype_against_filter, player_filter):
    empty_fig = go.Figure(layout={**PLOT_LAYOUT, "title": "No data"})
    alert = dbc.Alert("No bet data found. Run the simulation pipeline first.", color="warning")
    empty_return = (alert, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, alert, "", [], [], [])

    filters = {}
    if event:
        filters["event"] = event
    if bet_type:
        filters["bet_type"] = bet_type
    if min_edge:
        filters["min_edge"] = min_edge
    if books:
        filters["books"] = books

    df = get_bet_ledger(**filters)

    # Apply round filter
    if round_filter and "round" in df.columns:
        round_strs = [str(r) for r in round_filter] if isinstance(round_filter, list) else [str(round_filter)]
        df = df[df["round"].astype(str).str.strip().isin(round_strs)]

    # Apply analysis mode: sharp filtering + best-price dedup
    if analysis_mode in ("sharp_only", "sharp_best") and "bookmaker" in df.columns:
        sharp_pattern = "|".join(SHARP_BOOKS)
        df = df[df["bookmaker"].str.lower().str.contains(sharp_pattern, na=False)]
    if analysis_mode in ("best_price", "sharp_best"):
        # For each unique matchup side, keep only bets from the FIRST run,
        # then pick the best edge among those. Later runs are duplicates.
        dedup_cols = ["event_id", "bet_type", "round", "bet_on", "opponent"]
        existing = [c for c in dedup_cols if c in df.columns]
        if existing and "run_timestamp" in df.columns:
            # Find earliest run_timestamp per matchup side
            first_run = df.groupby(existing)["run_timestamp"].transform("min")
            df = df[df["run_timestamp"] == first_run]
            # Among first-run bets, keep the one with the best edge
            if "edge" in df.columns:
                df = df.sort_values("edge", ascending=False, na_position="last")
            df = df.drop_duplicates(subset=existing, keep="first")
        elif existing:
            df = df.drop_duplicates(subset=existing, keep="first")

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

    # Join archetype from sg_diagnostic.parquet (bet_on + opponent)
    arch_lookup = get_archetype_lookup()
    if not arch_lookup.empty and "event_id" in df.columns:
        df = df.merge(arch_lookup, left_on=["event_id", "bet_on"],
                      right_on=["event_id", "player_name"],
                      how="left", suffixes=("", "_arch"))
        df.drop(columns=["player_name_arch"], errors="ignore", inplace=True)
        df["archetype"] = df["archetype"].fillna("Unknown")

        # Join opponent archetype for matchup bets
        opp_lookup = arch_lookup.rename(columns={"player_name": "opp_name", "archetype": "archetype_against"})
        df = df.merge(opp_lookup, left_on=["event_id", "opponent"],
                      right_on=["event_id", "opp_name"],
                      how="left", suffixes=("", "_opp"))
        df.drop(columns=["opp_name"], errors="ignore", inplace=True)
        df["archetype_against"] = df["archetype_against"].fillna("")
    else:
        df["archetype"] = "Unknown"
        df["archetype_against"] = ""

    # Pre-compute raw_edge and dec_odds on full df (before new filters)
    if "book_odds" in df.columns and "fair_odds" in df.columns:
        market_prob = _american_to_prob(df["book_odds"])
        my_prob = _american_to_prob(df["fair_odds"])
        df["raw_edge"] = (my_prob - market_prob) * 100
        df["dec_odds"] = _american_to_decimal(df["book_odds"])
    else:
        df["raw_edge"] = np.nan
        df["dec_odds"] = np.nan

    # Build player + archetype options BEFORE filters are applied
    players = sorted(df["bet_on"].dropna().unique().tolist())
    player_options = [{"label": p.title(), "value": p} for p in players]
    archetypes = sorted(df["archetype"].dropna().unique().tolist())
    archetype_options = [{"label": a, "value": a} for a in archetypes]
    archetypes_against = sorted(df["archetype_against"].dropna().unique().tolist())
    archetypes_against = [a for a in archetypes_against if a]  # exclude empty
    archetype_against_options = [{"label": a, "value": a} for a in archetypes_against]

    # Apply new filters (raw_edge, dec_odds, archetype, archetype_against, player)
    if raw_edge_min is not None:
        df = df[df["raw_edge"].fillna(0) >= raw_edge_min]
    if raw_edge_max is not None:
        df = df[df["raw_edge"].fillna(0) <= raw_edge_max]
    if dec_odds_min is not None:
        df = df[df["dec_odds"].fillna(0) >= dec_odds_min]
    if dec_odds_max is not None:
        df = df[df["dec_odds"].fillna(999) <= dec_odds_max]
    if archetype_filter:
        df = df[df["archetype"].isin(archetype_filter)]
    if archetype_against_filter:
        df = df[df["archetype_against"].isin(archetype_against_filter)]
    if player_filter:
        df = df[df["bet_on"].isin(player_filter)]

    if df.empty:
        return empty_return[:-3] + (player_options, archetype_options, archetype_against_options)

    # Separate graded vs all — exclude duplicates entirely
    df = df[~df["result"].astype(str).str.strip().isin(["duplicate"])].copy()
    graded = df[df["result"].astype(str).str.strip() != ""].copy()
    resolved = graded[~graded["result"].isin(["no_data", "unknown"])].copy()

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

    # ── Chart 1: Cumulative P&L ──
    fig_cum = go.Figure(layout={**PLOT_LAYOUT, "title": "Cumulative P&L"})
    if not resolved.empty:
        sorted_df = resolved.sort_values("run_timestamp").reset_index(drop=True)
        cum_pnl = sorted_df["units_won"].cumsum()
        fig_cum.add_trace(go.Scatter(
            x=list(range(1, len(cum_pnl) + 1)), y=cum_pnl.values,
            mode="lines", name="Cumulative P&L",
            line=dict(color="#4ecca3", width=2),
            fill="tozeroy", fillcolor="rgba(78,204,163,0.15)",
        ))
        fig_cum.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_cum.update_xaxes(title_text="Bet #")
        fig_cum.update_yaxes(title_text="Units")

    # ── Chart 2: P&L by Event (dot chart) ──
    fig_pnl = go.Figure(layout={**PLOT_LAYOUT, "title": "P&L by Event"})
    if not resolved.empty:
        bet_types = [
            ("tournament_matchup", "Tourn MU", "#1f77b4", "circle"),
            ("round_matchup", "Round MU", "#ff7f0e", "diamond"),
            ("finish_position", "Finish Pos", "#2ca02c", "square"),
            ("finish_position_live", "Finish Live", "#17becf", "star"),
            ("score_bet", "Score Bet", "#9467bd", "triangle-up"),
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
            fig_pnl.add_trace(go.Scatter(
                x=events_with_data, y=values,
                mode="markers", name=name,
                marker=dict(color=color, size=12, symbol=symbol, line=dict(width=1, color="white")),
            ))
        total_pnl = resolved.groupby("event_name")["units_won"].sum()
        total_vals = [total_pnl.get(e, 0) for e in events_order]
        fig_pnl.add_trace(go.Scatter(
            x=events_order, y=total_vals,
            mode="markers", name="Total",
            marker=dict(color="white", size=14, symbol="star", line=dict(width=1, color="#4ecca3")),
        ))
        fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_pnl.update_yaxes(title_text="Units")

    # ── Chart 3: ROI by Bookmaker ──
    fig_book = go.Figure(layout={**PLOT_LAYOUT, "title": "ROI by Bookmaker"})
    if not resolved.empty:
        book_stats = (
            resolved.groupby("bookmaker")
            .agg(wagered=("units_wagered", "sum"), won=("units_won", "sum"), count=("bet_on", "count"))
            .reset_index()
        )
        book_stats["roi"] = book_stats["won"] / book_stats["wagered"] * 100
        book_stats = book_stats[book_stats["count"] >= 3].sort_values("roi")

        bk_colors = []
        for _, row in book_stats.iterrows():
            b = str(row["bookmaker"]).lower()
            if any(s in b for s in ["pinnacle", "betonline", "betcris", "bookmaker"]):
                bk_colors.append(COLOR_SHARP)
            elif any(r in b for r in ["fanduel", "draftkings", "caesars", "betmgm"]):
                bk_colors.append(COLOR_RETAIL)
            else:
                bk_colors.append("#7f7f7f")

        fig_book.add_trace(go.Bar(
            y=book_stats["bookmaker"], x=book_stats["roi"],
            orientation="h", marker_color=bk_colors,
            text=[f"{r:.1f}%" for r in book_stats["roi"]], textposition="auto",
        ))

    # ── ROI by Bucket (single bar chart, color = category) ──
    fig_buckets = go.Figure(layout={**PLOT_LAYOUT, "title": "ROI by Bucket"})

    def _bucket_roi(data, col, buckets):
        labels, rois, counts = [], [], []
        valid = data.dropna(subset=[col])
        for lo, hi, label in buckets:
            sub = valid[(valid[col] >= lo) & (valid[col] < hi)]
            w = len(sub[sub["result"].isin(["win", "win_dh"])]) if not sub.empty else 0
            l = len(sub[sub["result"] == "loss"]) if not sub.empty else 0
            wag = sub["units_wagered"].sum() if not sub.empty else 0
            roi = sub["units_won"].sum() / wag * 100 if wag > 0 else 0
            labels.append(label)
            rois.append(roi)
            counts.append(w + l)
        return labels, rois, counts

    if not resolved.empty:
        bucket_groups = []
        if "raw_edge" in resolved.columns:
            labs, rois, cnts = _bucket_roi(resolved, "raw_edge",
                [(0, 2, "0-2%"), (2, 4, "2-4%"), (4, 6, "4-6%"), (6, 100, "6%+")])
            bucket_groups.append(("Raw % Edge", "#1f77b4", labs, rois, cnts))

        labs, rois, cnts = _bucket_roi(resolved, "edge",
            [(3, 5, "3-5%"), (5, 8, "5-8%"), (8, 100, "8%+")])
        bucket_groups.append(("Edge Bucket", "#ff7f0e", labs, rois, cnts))

        if "dec_odds" in resolved.columns:
            labs, rois, cnts = _bucket_roi(resolved, "dec_odds",
                [(0, 2.0, "<2.0"), (2.0, 2.5, "2.0-2.5"), (2.5, 3.5, "2.5-3.5"),
                 (3.5, 8.0, "3.5-8.0"), (8.0, 999, "8.0+")])
            bucket_groups.append(("Odds Bucket", "#2ca02c", labs, rois, cnts))

        for group_name, color, labs, rois, cnts in bucket_groups:
            fig_buckets.add_trace(go.Bar(
                x=labs, y=rois, name=group_name,
                marker_color=color,
                text=[f"{r:+.0f}% n={n}" for r, n in zip(rois, cnts)],
                textposition="outside", textfont=dict(size=10),
            ))

    fig_buckets.update_yaxes(title_text="ROI %")
    fig_buckets.update_layout(barmode="group")

    # ── Archetype charts ──
    ARCHETYPE_COLORS = {
        "Stud": "#FFD700", "Ball Striker": "#1f77b4", "Long Accurate": "#2ca02c",
        "Long Wild": "#ff7f0e", "Short Accurate": "#9467bd", "Elite Putter": "#e377c2",
        "Short Game Specialist": "#17becf", "Balanced": "#7f7f7f", "Low Skill": "#d62728",
        "Unknown": "#555555",
    }

    # Cumulative P&L by Archetype (line chart)
    fig_arch_cum = go.Figure(layout={**PLOT_LAYOUT, "title": "Cumulative P&L by Archetype"})
    if not resolved.empty and "archetype" in resolved.columns:
        for arch in sorted(resolved["archetype"].unique()):
            sub = resolved[resolved["archetype"] == arch].sort_values("run_timestamp").reset_index(drop=True)
            if sub.empty:
                continue
            cum = sub["units_won"].cumsum()
            fig_arch_cum.add_trace(go.Scatter(
                x=list(range(1, len(cum) + 1)), y=cum.values,
                mode="lines", name=arch,
                line=dict(color=ARCHETYPE_COLORS.get(arch, "#aaaaaa"), width=2),
            ))
        fig_arch_cum.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_arch_cum.update_xaxes(title_text="Bet #")
        fig_arch_cum.update_yaxes(title_text="Units")
        fig_arch_cum.update_layout(legend=dict(font=dict(size=10)))

    # Total P&L by Archetype (bar chart)
    fig_arch_bar = go.Figure(layout={**PLOT_LAYOUT, "title": "Total P&L by Archetype"})
    if not resolved.empty and "archetype" in resolved.columns:
        arch_pnl = resolved.groupby("archetype")["units_won"].sum().sort_values()
        bar_colors = [ARCHETYPE_COLORS.get(a, "#aaaaaa") for a in arch_pnl.index]
        fig_arch_bar.add_trace(go.Bar(
            y=arch_pnl.index, x=arch_pnl.values,
            orientation="h", marker_color=bar_colors,
            text=[f"{v:+.1f}u" for v in arch_pnl.values], textposition="auto",
        ))
        fig_arch_bar.update_xaxes(title_text="Units")
        fig_arch_bar.update_layout(showlegend=False)

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

    # ── Filtered bets detail table (duplicates already removed from df above) ──
    detail_cols = ["bet_on", "opponent", "bookmaker", "bet_type", "round", "event_name",
                   "archetype", "archetype_against", "edge", "raw_edge", "dec_odds", "pred_on",
                   "result", "units_wagered", "units_won"]
    available_cols = [c for c in detail_cols if c in df.columns]
    detail_df = df[available_cols].copy()
    for col in ["edge", "raw_edge", "dec_odds", "pred_on", "units_wagered", "units_won"]:
        if col in detail_df.columns:
            detail_df[col] = detail_df[col].round(2)

    _edge_style = {
        "function": "params.value > 0 ? {'color': '#4ecca3'} : params.value < 0 ? {'color': '#e74c3c'} : {}"
    }
    _result_style = {
        "function": "params.value === 'win' || params.value === 'win_dh' ? {'color': '#4ecca3'} "
                    ": params.value === 'loss' ? {'color': '#e74c3c'} : {}"
    }
    col_headers = {
        "bet_on": "Bet On", "opponent": "Opponent", "bookmaker": "Book",
        "bet_type": "Type", "round": "R", "event_name": "Event",
        "archetype": "Archetype", "archetype_against": "Arch. Against",
        "edge": "Edge", "raw_edge": "Raw Edge",
        "dec_odds": "Odds", "pred_on": "Pred", "result": "Result",
        "units_wagered": "Wagered", "units_won": "Won",
    }
    detail_col_defs = []
    for col in available_cols:
        cd = {
            "field": col,
            "headerName": col_headers.get(col, col.replace("_", " ").title()),
            "sortable": True, "filter": True, "resizable": True,
        }
        if detail_df[col].dtype in ("float64", "float32"):
            cd["valueFormatter"] = {"function": "d3.format('.2f')(params.value)"}
        if "edge" in col:
            cd["cellStyle"] = _edge_style
        if col == "result":
            cd["cellStyle"] = _result_style
        detail_col_defs.append(cd)

    remaining_table = make_grid(detail_df, column_defs=detail_col_defs,
                                id_suffix="perf-remaining", height=500)

    return kpi, fig_cum, fig_pnl, fig_book, fig_buckets, fig_arch_cum, fig_arch_bar, summary_content, remaining_table, player_options, archetype_options, archetype_against_options
