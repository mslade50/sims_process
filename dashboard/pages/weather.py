"""Weather page — forecasts, player impact, and matchup discrepancies."""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dashboard.data_layer import (
    get_weather_forecast, get_weather_impact_players, get_weather_matchup_edges,
)
from dashboard.components.stat_cards import stat_card_row
from dashboard.components.tables import make_grid

dash.register_page(__name__, path="/weather", title="Weather", order=7)

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,33,62,0.8)",
    font=dict(color="#e0e0e0"),
    margin=dict(l=50, r=30, t=40, b=40),
)

ROUND_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


layout = dbc.Container([
    html.H4("Weather", className="page-header"),

    # KPI cards
    html.Div(id="weather-kpi-row", className="mb-3"),

    # Forecast charts
    dbc.Row([
        dbc.Col(dcc.Graph(id="weather-wind-chart"), md=6),
        dbc.Col(dcc.Graph(id="weather-dew-chart"), md=6),
    ], className="mb-4"),

    # Matchup discrepancies
    html.H5("Matchup Weather Discrepancies (BetOnline)", className="mb-2"),
    html.Div(id="weather-matchup-section", className="mb-4"),

    # Field weather impact
    html.H5("Field Weather Impact", className="mb-2"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="weather-bar-chart"), md=12),
    ], className="mb-3"),
    html.Div(id="weather-field-table"),

    dcc.Interval(id="weather-refresh", interval=300_000, n_intervals=0),
], fluid=True)


@callback(
    Output("weather-kpi-row", "children"),
    Output("weather-wind-chart", "figure"),
    Output("weather-dew-chart", "figure"),
    Output("weather-matchup-section", "children"),
    Output("weather-bar-chart", "figure"),
    Output("weather-field-table", "children"),
    Input("weather-refresh", "n_intervals"),
)
def update_weather(_):
    empty_fig = go.Figure(layout={**PLOT_LAYOUT, "height": 350})

    # ── Forecast data ──
    forecast = get_weather_forecast()
    wind_override = forecast.get("wind_override", 0.0)
    dew_calculation = forecast.get("dew_calculation", 0.0)
    baseline_wind = forecast.get("baseline_wind")
    wind_speed_base = forecast.get("wind_speed_base")

    # Compute avg wind for R1/R2 and AM/PM differential
    wind_r1 = forecast.get("wind_r1", [])
    wind_r2 = forecast.get("wind_r2", [])
    avg_wind_r1 = f"{np.mean(wind_r1):.1f}" if wind_r1 else "—"
    avg_wind_r2 = f"{np.mean(wind_r2):.1f}" if wind_r2 else "—"

    # AM/PM split: first 6 hours (6AM-noon) vs next 6 (noon-6PM)
    if len(wind_r1) >= 12:
        am_wind = np.mean(wind_r1[:6])
        pm_wind = np.mean(wind_r1[6:12])
        am_pm_diff = f"{pm_wind - am_wind:+.1f}"
    else:
        am_pm_diff = "—"

    wind_coeff_str = f"{wind_override}" if wind_override else (f"{baseline_wind}" if baseline_wind else "—")
    dew_coeff_str = f"{dew_calculation}" if dew_calculation else "—"

    kpi = stat_card_row([
        {"title": "Wind Coeff", "value": wind_coeff_str},
        {"title": "Dew Coeff", "value": dew_coeff_str},
        {"title": "Avg Wind R1", "value": avg_wind_r1},
        {"title": "Avg Wind R2", "value": avg_wind_r2},
        {"title": "AM/PM Diff R1", "value": am_pm_diff},
    ])

    # ── Wind forecast chart ──
    wind_fig = go.Figure(layout={**PLOT_LAYOUT, "title": "Hourly Wind Forecast by Round (mph)", "height": 380})
    for r, color in zip(range(1, 5), ROUND_COLORS):
        arr = forecast.get(f"wind_r{r}", [])
        if arr:
            hours = list(range(6, 6 + len(arr)))
            wind_fig.add_trace(go.Scatter(
                x=hours, y=arr, mode="lines+markers", name=f"R{r}",
                line=dict(color=color, width=2), marker=dict(size=4),
            ))
    if wind_speed_base:
        wind_fig.add_hline(
            y=wind_speed_base, line_dash="dash", line_color="#888",
            annotation_text=f"Historical base ({wind_speed_base})",
            annotation_position="top left",
        )
    wind_fig.update_xaxes(title_text="Hour", dtick=2)
    wind_fig.update_yaxes(title_text="Wind (mph)")

    # ── Dew forecast chart ──
    dew_fig = go.Figure(layout={**PLOT_LAYOUT, "title": "Hourly Dewpoint Forecast by Round", "height": 380})
    for r, color in zip(range(1, 5), ROUND_COLORS):
        arr = forecast.get(f"dew_r{r}", [])
        if arr:
            hours = list(range(6, 6 + len(arr)))
            dew_fig.add_trace(go.Scatter(
                x=hours, y=arr, mode="lines+markers", name=f"R{r}",
                line=dict(color=color, width=2), marker=dict(size=4),
            ))
    dew_fig.update_xaxes(title_text="Hour", dtick=2)
    dew_fig.update_yaxes(title_text="Dewpoint")

    # ── Matchup discrepancies ──
    matchups = get_weather_matchup_edges()
    if matchups.empty:
        matchup_section = dbc.Alert("No BetOnline tournament matchups with weather data.", color="info")
    else:
        # Color-code by magnitude
        def _diff_style():
            return {
                "styleConditions": [
                    {"condition": "Math.abs(params.value) >= 0.15", "style": {"backgroundColor": "#b71c1c", "color": "white"}},
                    {"condition": "Math.abs(params.value) >= 0.08 && Math.abs(params.value) < 0.15", "style": {"backgroundColor": "#e65100", "color": "white"}},
                    {"condition": "Math.abs(params.value) >= 0.04 && Math.abs(params.value) < 0.08", "style": {"backgroundColor": "#4a6741", "color": "white"}},
                ]
            }

        col_defs = [
            {"field": "player_1", "headerName": "Player 1", "sortable": True, "filter": True,
             "valueFormatter": {"function": "params.value ? params.value.split(', ').reverse().join(' ').replace(/^\\w/, c => c.toUpperCase()) : ''"}},
            {"field": "player_2", "headerName": "Player 2", "sortable": True, "filter": True,
             "valueFormatter": {"function": "params.value ? params.value.split(', ').reverse().join(' ').replace(/^\\w/, c => c.toUpperCase()) : ''"}},
            {"field": "p1_weather_adv", "headerName": "P1 Weather Adv", "sortable": True,
             "valueFormatter": {"function": "d3.format('+.3f')(params.value)"}},
            {"field": "p2_weather_adv", "headerName": "P2 Weather Adv", "sortable": True,
             "valueFormatter": {"function": "d3.format('+.3f')(params.value)"}},
            {"field": "differential", "headerName": "Differential", "sortable": True,
             "valueFormatter": {"function": "d3.format('+.3f')(params.value)"}, "cellStyle": _diff_style()},
            {"field": "favored", "headerName": "Favored", "sortable": True,
             "valueFormatter": {"function": "params.value ? params.value.split(', ').reverse().join(' ').replace(/^\\w/, c => c.toUpperCase()) : ''"}},
        ]
        matchup_section = make_grid(matchups, column_defs=col_defs, id_suffix="weather-mu", height=400)

    # ── Field weather impact ──
    players_df = get_weather_impact_players()
    if players_df.empty:
        bar_fig = empty_fig
        field_table = dbc.Alert("No weather_impact_players.csv found.", color="info")
    else:
        # Compute total advantage
        adv_cols = [c for c in ["wind_adv_r1_2", "dew_adv_r1_2"] if c in players_df.columns]
        if adv_cols:
            players_df["total_adv"] = players_df[adv_cols].sum(axis=1)
        else:
            players_df["total_adv"] = 0.0

        sorted_df = players_df.sort_values("total_adv", ascending=True)

        # Top/bottom 20
        n_show = min(20, len(sorted_df))
        top = sorted_df.tail(n_show)
        bottom = sorted_df.head(n_show)
        display = pd.concat([bottom, top]).drop_duplicates(subset="player_name")

        # Color gradient
        vals = display["total_adv"].values
        max_abs = max(abs(vals.min()), abs(vals.max()), 0.01)
        colors = []
        for v in vals:
            ratio = v / max_abs
            if ratio > 0:
                g = int(100 + 155 * ratio)
                colors.append(f"rgb(30, {g}, 30)")
            else:
                r = int(100 + 155 * abs(ratio))
                colors.append(f"rgb({r}, 30, 30)")

        bar_fig = go.Figure(layout={
            **PLOT_LAYOUT,
            "title": f"Total Weather Advantage (Top/Bottom {n_show})",
            "height": max(400, n_show * 22),
        })
        bar_fig.add_trace(go.Bar(
            y=display["player_name"].str.title(),
            x=display["total_adv"],
            orientation="h",
            marker_color=colors,
            text=display["total_adv"].round(3),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Total advantage: %{x:+.3f}<extra></extra>",
        ))
        bar_fig.update_xaxes(title_text="Total Weather Advantage (SG)")
        bar_fig.update_yaxes(tickfont=dict(size=10))

        # Full table
        field_table = make_grid(
            players_df.sort_values("total_adv", ascending=False),
            id_suffix="weather-field", height=500,
        )

    return kpi, wind_fig, dew_fig, matchup_section, bar_fig, field_table
