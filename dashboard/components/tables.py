"""AG Grid table wrappers with dark mode styling."""

import dash_ag_grid as dag


def _edge_cell_style():
    """JS expression for coloring edge cells green/yellow/red."""
    return {
        "styleConditions": [
            {"condition": "params.value >= 10", "style": {"backgroundColor": "#1b5e20", "color": "white"}},
            {"condition": "params.value >= 5 && params.value < 10", "style": {"backgroundColor": "#2e7d32", "color": "white"}},
            {"condition": "params.value >= 0 && params.value < 5", "style": {"backgroundColor": "#4a6741", "color": "white"}},
            {"condition": "params.value < 0 && params.value >= -5", "style": {"backgroundColor": "#6d4c41", "color": "white"}},
            {"condition": "params.value < -5", "style": {"backgroundColor": "#b71c1c", "color": "white"}},
        ]
    }


def _result_cell_style():
    """JS expression for coloring result cells."""
    return {
        "styleConditions": [
            {"condition": "params.value === 'win' || params.value === 'win_dh'", "style": {"backgroundColor": "#1b5e20", "color": "white"}},
            {"condition": "params.value === 'loss'", "style": {"backgroundColor": "#b71c1c", "color": "white"}},
            {"condition": "params.value === 'push'", "style": {"backgroundColor": "#616161", "color": "white"}},
        ]
    }


def make_grid(df, column_defs=None, id_suffix="", height=600, page_size=25):
    """Create a styled AG Grid from a DataFrame.

    Args:
        df: pandas DataFrame
        column_defs: list of column definitions (auto-generated if None)
        id_suffix: unique suffix for the grid id
        height: grid height in pixels
        page_size: rows per page
    """
    if df.empty:
        return dag.AgGrid(
            id=f"grid-{id_suffix}",
            rowData=[],
            columnDefs=[],
            style={"height": f"{height}px"},
        )

    if column_defs is None:
        column_defs = []
        for col in df.columns:
            col_def = {
                "field": col,
                "headerName": col.replace("_", " ").title(),
                "sortable": True,
                "filter": True,
                "resizable": True,
            }

            # Auto-format numeric columns
            if df[col].dtype in ("float64", "float32"):
                col_def["valueFormatter"] = {"function": "d3.format('.2f')(params.value)"}

            # Apply edge coloring
            if "edge" in col.lower():
                col_def["cellStyle"] = _edge_cell_style()

            # Apply result coloring
            if col == "result":
                col_def["cellStyle"] = _result_cell_style()

            column_defs.append(col_def)

    return dag.AgGrid(
        id=f"grid-{id_suffix}",
        rowData=df.to_dict("records"),
        columnDefs=column_defs,
        defaultColDef={
            "sortable": True,
            "filter": True,
            "resizable": True,
            "minWidth": 70,
            "flex": 1,
        },
        dashGridOptions={
            "pagination": True,
            "paginationPageSize": page_size,
            "animateRows": True,
            "domLayout": "normal",
        },
        className="ag-theme-alpine-dark",
        style={"height": f"{height}px"},
    )
