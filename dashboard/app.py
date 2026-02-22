"""Golf Sim Dashboard — Dash multi-page app with Bootstrap dark theme.

Usage:
    python -m dashboard.app          # Local dev (port 8050)
    gunicorn dashboard.app:server    # Production (Render)
"""

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output

from dashboard.components.nav import make_navbar

app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder="pages",
    external_stylesheets=[dbc.themes.SLATE],
    suppress_callback_exceptions=True,
    title="Golf Sim Dashboard",
    update_title="Loading...",
)

server = app.server  # For gunicorn

app.layout = dbc.Container([
    dcc.Location(id="url", refresh=False),
    make_navbar(),
    dash.page_container,
], fluid=True, className="px-4 pb-4")


# Navbar toggler callback for mobile
@app.callback(
    Output("navbar-collapse", "is_open"),
    Input("navbar-toggler", "n_clicks"),
    dash.State("navbar-collapse", "is_open"),
)
def toggle_navbar(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


if __name__ == "__main__":
    app.run(debug=True, port=8050)
