"""Active Bets page — Current tournament bets by type."""

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

from dashboard.data_layer import get_bet_ledger, get_tournament_config
from dashboard.components.stat_cards import stat_card_row
from dashboard.components.filters import sportsbook_filter
from dashboard.components.tables import make_grid

dash.register_page(__name__, path="/bets", title="Active Bets", order=5)

# 1 unit = $200 — finish position bets store dollar amounts, convert to units
UNIT_SIZE = 200.0


def _convert_to_units(df):
    """Convert finish position kelly_stake dollars to units ($200 = 1 unit).

    Matchup bets are already in unit terms (flat 1-unit wagers).
    units_wagered is derived from kelly_stake (raw dollars) so needs conversion.
    units_won is already in units from grade_bets.py.
    """
    df = df.copy()
    is_finish = df["bet_type"] == "finish_position"
    if "units_wagered" in df.columns:
        df.loc[is_finish, "units_wagered"] = df.loc[is_finish, "units_wagered"] / UNIT_SIZE
    return df


def _result_badge(result):
    if not result or str(result).strip() == "":
        return "Pending"
    r = str(result).lower().strip()
    if r in ("win", "win_dh"):
        return "Win"
    elif r == "loss":
        return "Loss"
    elif r == "push":
        return "Push"
    return result.title()


def _section_summary(df):
    """Build a summary row for a bet section."""
    if df.empty:
        return html.P("No bets", className="text-muted")

    count = len(df)
    avg_edge = df["edge"].mean() if "edge" in df.columns else 0
    wagered = df["units_wagered"].sum() if "units_wagered" in df.columns else 0
    won = df["units_won"].sum() if "units_won" in df.columns else 0

    resolved = df[df["result"].astype(str).str.strip() != ""]
    resolved = resolved[~resolved["result"].isin(["no_data", "unknown", "duplicate"])]
    wins = len(resolved[resolved["result"].isin(["win", "win_dh"])]) if not resolved.empty else 0
    losses = len(resolved[resolved["result"] == "loss"]) if not resolved.empty else 0
    pending = count - len(resolved)

    return dbc.Row([
        dbc.Col(html.Span(f"{count} bets", className="fw-bold"), width="auto"),
        dbc.Col(html.Span(f"Avg Edge: {avg_edge:.1f}%"), width="auto"),
        dbc.Col(html.Span(f"{wins}W-{losses}L, {pending} pending"), width="auto"),
        dbc.Col(html.Span(
            f"P&L: {won:+.2f}u",
            className="fw-bold text-success" if won >= 0 else "fw-bold text-danger",
        ), width="auto"),
    ], className="mb-2 g-3")


layout = dbc.Container([
    html.H4("Active Bets", className="page-header"),

    dbc.Row([
        sportsbook_filter("bets"),
        dbc.Col([
            html.Label("Event", className="form-label small text-muted"),
            dcc.Dropdown(id="bets-event-filter", placeholder="Current tournament", className="dash-dropdown-dark"),
        ], md=3),
        dbc.Col([
            dbc.Button("Export CSV", id="bets-export-btn", color="secondary", size="sm", className="mt-4"),
            dcc.Download(id="bets-csv-download"),
        ], md=2),
    ], className="mb-3"),

    html.Div(id="bets-kpi-row", className="mb-3"),

    dbc.Accordion([
        dbc.AccordionItem([
            html.Div(id="bets-tourney-mu-summary"),
            html.Div(id="bets-tourney-mu-table"),
        ], title="Tournament Matchups", item_id="tourney-mu"),
        dbc.AccordionItem([
            html.Div(id="bets-round-mu-summary"),
            html.Div(id="bets-round-mu-table"),
        ], title="Round Matchups", item_id="round-mu"),
        dbc.AccordionItem([
            html.Div(id="bets-finish-summary"),
            html.Div(id="bets-finish-table"),
        ], title="Finish Positions", item_id="finish-pos"),
    ], always_open=True, active_item=["tourney-mu", "round-mu", "finish-pos"]),
], fluid=True)


@callback(
    Output("bets-event-filter", "options"),
    Output("bets-event-filter", "value"),
    Input("bets-event-filter", "id"),  # on load trigger
)
def populate_events(_):
    df = get_bet_ledger()
    if df.empty:
        return [], None
    events = sorted(df["event_name"].dropna().unique().tolist())
    config = get_tournament_config()
    current = config.get("tourney", events[-1] if events else None)
    options = [{"label": e.title(), "value": e} for e in events]
    value = current if current in events else (events[-1] if events else None)
    return options, value


@callback(
    Output("bets-kpi-row", "children"),
    Output("bets-tourney-mu-summary", "children"),
    Output("bets-tourney-mu-table", "children"),
    Output("bets-round-mu-summary", "children"),
    Output("bets-round-mu-table", "children"),
    Output("bets-finish-summary", "children"),
    Output("bets-finish-table", "children"),
    Input("bets-event-filter", "value"),
    Input("bets-book-filter", "value"),
)
def update_bets(event, books):
    filters = {}
    if event:
        filters["event"] = event
    if books:
        filters["books"] = books

    df = get_bet_ledger(**filters)

    if df.empty:
        alert = dbc.Alert("No bets found for this event.", color="warning")
        return alert, alert, "", alert, "", alert, ""

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

    # KPI
    total = len(df)
    avg_edge = df["edge"].mean() if "edge" in df.columns else 0
    resolved = df[df["result"].astype(str).str.strip() != ""]
    resolved = resolved[~resolved["result"].isin(["no_data", "unknown", "duplicate"])]
    won = resolved["units_won"].sum() if not resolved.empty else 0

    kpi = stat_card_row([
        {"title": "Total Bets", "value": str(total)},
        {"title": "Avg Edge", "value": f"{avg_edge:.1f}%"},
        {"title": "P&L", "value": f"{won:+.2f}u", "color": "success" if won >= 0 else "danger"},
    ])

    # Split by type
    display_cols = ["bet_on", "opponent", "bookmaker", "edge", "pred_on", "book_odds", "fair_odds", "units_wagered", "result", "units_won"]

    tourney_mu = df[df["bet_type"] == "tournament_matchup"]
    round_mu = df[df["bet_type"] == "round_matchup"]
    finish = df[df["bet_type"] == "finish_position"]

    def _make_section(section_df, suffix):
        available = [c for c in display_cols if c in section_df.columns]
        show_df = section_df[available].copy() if not section_df.empty else pd.DataFrame()
        return _section_summary(section_df), make_grid(show_df, id_suffix=suffix, height=400)

    tm_sum, tm_tbl = _make_section(tourney_mu, "tourney-mu")
    rm_sum, rm_tbl = _make_section(round_mu, "round-mu")
    fp_sum, fp_tbl = _make_section(finish, "finish-pos")

    return kpi, tm_sum, tm_tbl, rm_sum, rm_tbl, fp_sum, fp_tbl


@callback(
    Output("bets-csv-download", "data"),
    Input("bets-export-btn", "n_clicks"),
    State("bets-event-filter", "value"),
    State("bets-book-filter", "value"),
    prevent_initial_call=True,
)
def export_csv(n_clicks, event, books):
    filters = {}
    if event:
        filters["event"] = event
    if books:
        filters["books"] = books
    df = get_bet_ledger(**filters)
    if df.empty:
        return None
    return dcc.send_data_frame(df.to_csv, f"bets_{event or 'all'}.csv", index=False)
