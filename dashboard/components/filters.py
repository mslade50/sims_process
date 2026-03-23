"""Reusable filter components for dashboard pages."""

import dash_bootstrap_components as dbc
from dash import html, dcc

from dashboard.config import BOOKS_TO_USE


def sportsbook_filter(id_prefix, multi=True, default_all=True):
    """Sportsbook multi-select dropdown."""
    options = [{"label": b.title(), "value": b} for b in BOOKS_TO_USE]
    value = BOOKS_TO_USE if default_all else []

    return dbc.Col([
        html.Label("Sportsbook", className="form-label small text-muted"),
        dcc.Dropdown(
            id=f"{id_prefix}-book-filter",
            options=options,
            value=value,
            multi=multi,
            placeholder="All books",
            className="dash-dropdown-dark",
        ),
    ], md=3)


def round_selector(id_prefix, available_rounds=None, default=None):
    """Round number selector. Supports int rounds and 'tourn' for tournament matchups."""
    if available_rounds is None:
        available_rounds = [1, 2, 3, 4]
    options = []
    for r in available_rounds:
        if r == "tourn":
            options.append({"label": "Tournament", "value": "tourn"})
        else:
            options.append({"label": f"Round {r}", "value": r})

    return dbc.Col([
        html.Label("Round", className="form-label small text-muted"),
        dcc.Dropdown(
            id=f"{id_prefix}-round-filter",
            options=options,
            multi=True,
            placeholder="All rounds",
            className="dash-dropdown-dark",
        ),
    ], md=2)


def edge_slider(id_prefix, min_val=0, max_val=25, default=3):
    """Min edge slider."""
    return dbc.Col([
        html.Label(f"Min Edge: {default}%", id=f"{id_prefix}-edge-label",
                    className="form-label small text-muted"),
        dcc.Slider(
            id=f"{id_prefix}-edge-slider",
            min=min_val,
            max=max_val,
            step=1,
            value=default,
            marks={i: f"{i}" for i in range(0, max_val + 1, 5)},
            tooltip={"placement": "bottom"},
        ),
    ], md=3)


def pred_slider(id_prefix, default=0.75):
    """Min prediction slider."""
    return dbc.Col([
        html.Label(f"Min Pred: {default}", id=f"{id_prefix}-pred-label",
                    className="form-label small text-muted"),
        dcc.Slider(
            id=f"{id_prefix}-pred-slider",
            min=0,
            max=3,
            step=0.25,
            value=default,
            marks={i: f"{i}" for i in range(0, 4)},
            tooltip={"placement": "bottom"},
        ),
    ], md=2)


def sample_slider(id_prefix, default=20):
    """Min sample size slider."""
    return dbc.Col([
        html.Label(f"Min Sample: {default}", id=f"{id_prefix}-sample-label",
                    className="form-label small text-muted"),
        dcc.Slider(
            id=f"{id_prefix}-sample-slider",
            min=0,
            max=50,
            step=5,
            value=default,
            marks={0: "0", 10: "10", 20: "20", 30: "30", 40: "40", 50: "50"},
            tooltip={"placement": "bottom"},
        ),
    ], md=2)


def sharp_toggle(id_prefix):
    """Sharp books only toggle."""
    return dbc.Col([
        html.Label("Sharp Only", className="form-label small text-muted"),
        dbc.Switch(
            id=f"{id_prefix}-sharp-toggle",
            value=False,
            className="mt-1",
        ),
    ], md=1)


def event_selector(id_prefix, events=None):
    """Event name selector (multi-select)."""
    if events is None:
        events = []
    options = [{"label": e.title(), "value": e} for e in events]

    return dbc.Col([
        html.Label("Event", className="form-label small text-muted"),
        dcc.Dropdown(
            id=f"{id_prefix}-event-filter",
            options=options,
            multi=True,
            placeholder="All events",
            className="dash-dropdown-dark",
        ),
    ], md=3)


def bet_type_selector(id_prefix):
    """Bet type selector (multi-select)."""
    options = [
        {"label": "Tournament Matchup", "value": "tournament_matchup"},
        {"label": "Round Matchup", "value": "round_matchup"},
        {"label": "Finish Position", "value": "finish_position"},
        {"label": "Finish Position (Live)", "value": "finish_position_live"},
        {"label": "Score Bet", "value": "score_bet"},
    ]
    return dbc.Col([
        html.Label("Bet Type", className="form-label small text-muted"),
        dcc.Dropdown(
            id=f"{id_prefix}-type-filter",
            options=options,
            multi=True,
            placeholder="All types",
            className="dash-dropdown-dark",
        ),
    ], md=2)
