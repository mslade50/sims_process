"""Fragility page — Flag bets on players with low pred/sample."""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

from dashboard.data_layer import get_bet_ledger, get_tournament_config
from dashboard.components.stat_cards import stat_card_row
from dashboard.components.tables import make_grid
from dashboard.config import EMAIL_MIN_PRED, EMAIL_MIN_SAMPLE

dash.register_page(__name__, path="/fragility", title="Fragility", order=9)


def _get_events():
    df = get_bet_ledger()
    if df.empty:
        return []
    return sorted(df["event_name"].dropna().unique().tolist())


def _default_event():
    config = get_tournament_config()
    tourney = config.get("tourney", "")
    events = _get_events()
    if tourney:
        for e in events:
            if e.lower().strip() == tourney.lower().strip():
                return e
    return events[-1] if events else None


layout = dbc.Container([
    html.H4("Fragility", className="page-header"),

    # Controls
    dbc.Row([
        dbc.Col([
            html.Label("Event", className="form-label small text-muted"),
            dcc.Dropdown(
                id="frag-event-filter",
                options=[{"label": e.title(), "value": e} for e in _get_events()],
                value=_default_event(),
                placeholder="Select event",
                className="dash-dropdown-dark",
            ),
        ], md=4),
        dbc.Col([
            html.Label("Show All Bets", className="form-label small text-muted"),
            dbc.Switch(id="frag-show-all", value=False, className="mt-1"),
        ], md=2),
    ], className="mb-3"),

    # KPI row
    html.Div(id="frag-kpi-row"),

    # Accordion sections
    html.Div(id="frag-content"),
], fluid=True)


def _classify(df):
    """Classify bets by fragility. Returns df with 'flag_reason' column."""
    df = df.copy()

    pred = pd.to_numeric(df.get("pred_on", pd.Series(dtype=float)), errors="coerce")
    sample = pd.to_numeric(df.get("sample_on", pd.Series(dtype=float)), errors="coerce")

    low_pred = pred < EMAIL_MIN_PRED
    low_sample = sample < EMAIL_MIN_SAMPLE

    conditions = [
        low_pred & low_sample,
        low_pred & ~low_sample,
        ~low_pred & low_sample,
    ]
    choices = ["Both", "Low Pred", "Low Sample"]
    df["flag_reason"] = np.select(conditions, choices, default="OK")

    # NaN pred/sample → not flagged
    df.loc[pred.isna() & sample.isna(), "flag_reason"] = "OK"

    return df


def _make_section_grid(section_df, suffix):
    """Build an AG Grid for one bet type section."""
    display_cols = ["bet_on", "opponent", "bookmaker", "edge", "pred_on", "sample_on", "flag_reason"]
    available = [c for c in display_cols if c in section_df.columns]
    show_df = section_df[available].copy()

    col_defs = []
    for col in available:
        d = {
            "field": col,
            "headerName": col.replace("_", " ").title(),
            "sortable": True,
            "filter": True,
            "resizable": True,
        }
        if col == "edge":
            d["headerName"] = "Edge"
            d["valueFormatter"] = {"function": "d3.format('.1f')(params.value) + '%'"}
        elif col == "pred_on":
            d["headerName"] = "Pred"
            d["valueFormatter"] = {"function": "d3.format('.2f')(params.value)"}
        elif col == "sample_on":
            d["headerName"] = "Sample"
        elif col == "flag_reason":
            d["headerName"] = "Flag"
            d["cellStyle"] = {
                "styleConditions": [
                    {"condition": "params.value === 'Both'", "style": {"backgroundColor": "#b71c1c", "color": "white"}},
                    {"condition": "params.value === 'Low Pred'", "style": {"backgroundColor": "#e65100", "color": "white"}},
                    {"condition": "params.value === 'Low Sample'", "style": {"backgroundColor": "#f57f17", "color": "white"}},
                ]
            }
        col_defs.append(d)

    return make_grid(show_df, column_defs=col_defs, id_suffix=f"frag-{suffix}", height=400, page_size=20)


@callback(
    Output("frag-kpi-row", "children"),
    Output("frag-content", "children"),
    Input("frag-event-filter", "value"),
    Input("frag-show-all", "value"),
)
def update_fragility(event, show_all):
    if not event:
        return "", dbc.Alert("Select an event.", color="info")

    df = get_bet_ledger(event=event)
    if df.empty:
        return "", dbc.Alert("No bet data for this event.", color="warning")

    df = _classify(df)

    flagged = df[df["flag_reason"] != "OK"]
    total = len(df)
    n_flagged = len(flagged)
    n_both = len(df[df["flag_reason"] == "Both"])
    n_pred = len(df[df["flag_reason"] == "Low Pred"])
    n_sample = len(df[df["flag_reason"] == "Low Sample"])

    kpi = stat_card_row([
        {"title": "Total Bets", "value": str(total), "color": "primary"},
        {"title": "Flagged", "value": str(n_flagged), "color": "danger" if n_flagged else "success"},
        {"title": "Both", "value": str(n_both), "color": "danger" if n_both else "secondary"},
        {"title": "Low Pred", "value": str(n_pred), "color": "warning" if n_pred else "secondary"},
        {"title": "Low Sample", "value": str(n_sample), "color": "warning" if n_sample else "secondary"},
    ])

    display_df = df if show_all else flagged

    # Finish positions: keep only the highest edge bet per player
    finish_mask = display_df["bet_type"] == "finish_position"
    finish_df = display_df[finish_mask].copy()
    other_df = display_df[~finish_mask]
    if not finish_df.empty:
        finish_df["_edge_num"] = pd.to_numeric(finish_df["edge"], errors="coerce")
        finish_df = finish_df.sort_values("_edge_num", ascending=False)
        finish_df = finish_df.drop_duplicates(subset=["bet_on"], keep="first")
        finish_df = finish_df.drop(columns=["_edge_num"])
    display_df = pd.concat([other_df, finish_df], ignore_index=True)

    sections = [
        ("Tournament MU", "tournament_matchup", "tourn"),
        ("Finish Positions", "finish_position", "finish"),
    ]

    items = []
    for label, bt, suffix in sections:
        sub = display_df[display_df["bet_type"] == bt]
        count = len(sub)
        if sub.empty:
            body = dbc.Alert(f"No {label.lower()} bets to display.", color="secondary")
        else:
            body = _make_section_grid(sub, suffix)
        items.append(
            dbc.AccordionItem(body, title=f"{label} ({count})")
        )

    accordion = dbc.Accordion(items, always_open=True, className="mt-3")

    return kpi, accordion
