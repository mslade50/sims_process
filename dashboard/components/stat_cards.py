"""Reusable KPI stat card components."""

import dash_bootstrap_components as dbc
from dash import html


def stat_card(title, value, subtitle=None, color="primary"):
    """Create a single KPI card."""
    body = [
        html.H6(title, className="card-title text-muted mb-1"),
        html.H3(value, className="card-text fw-bold"),
    ]
    if subtitle:
        body.append(html.Small(subtitle, className="text-muted"))

    return dbc.Card(
        dbc.CardBody(body, className="py-3"),
        color=color,
        outline=True,
        className="h-100",
    )


def stat_card_row(cards_data):
    """Create a row of stat cards.

    Args:
        cards_data: list of dicts with keys: title, value, subtitle (optional), color (optional)
    """
    n = len(cards_data)
    col_width = max(12 // n, 2)
    cols = []
    for card in cards_data:
        cols.append(
            dbc.Col(
                stat_card(
                    card["title"],
                    card["value"],
                    card.get("subtitle"),
                    card.get("color", "primary"),
                ),
                md=col_width,
                className="mb-3",
            )
        )
    return dbc.Row(cols)
