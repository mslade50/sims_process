"""Top navigation bar for the dashboard."""

import dash_bootstrap_components as dbc
from dash import html


def make_navbar():
    return dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("Golf Sim Dashboard", href="/", className="ms-2 fw-bold"),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("Home", href="/")),
                    dbc.NavItem(dbc.NavLink("Matchups", href="/matchups")),
                    dbc.NavItem(dbc.NavLink("Outrights (Pre)", href="/outrights-pre")),
                    dbc.NavItem(dbc.NavLink("Outrights (Live)", href="/outrights-live")),
                    dbc.NavItem(dbc.NavLink("Pricer", href="/pricer")),
                    dbc.NavItem(dbc.NavLink("Bets", href="/bets")),
                    dbc.NavItem(dbc.NavLink("Performance", href="/performance")),
                    dbc.NavItem(dbc.NavLink("Skill", href="/skill")),
                    dbc.NavItem(dbc.NavLink("Diagnostics", href="/diagnostics")),
                    dbc.NavItem(dbc.NavLink("Fragility", href="/fragility")),
                ], className="ms-auto", navbar=True),
                id="navbar-collapse",
                navbar=True,
            ),
        ], fluid=True),
        color="dark",
        dark=True,
        className="mb-3",
    )
