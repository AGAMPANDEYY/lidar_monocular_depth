# calibration/visualise/lidar_select_save.py
import numpy as np
import pandas as pd
import open3d as o3d
from pathlib import Path

import dash
from dash import html, dcc, Output, Input, State
import plotly.graph_objects as go

# ----------------- config -----------------
PCD_PATH = Path("calibration/accum_2745s.pcd")
AX_X = (-2.0, 1.0)   # meters
AX_Y = ( 1.0, 6.0)
AX_Z = (-1.0, 1.0)
POINT_SIZE = 2
SELECTED_SIZE = 4
CSV_OUT = PCD_PATH.parent / "selected_points.csv"
# -----------------------------------------

# Load point cloud
pcd = o3d.io.read_point_cloud(str(PCD_PATH))
pts = np.asarray(pcd.points)  # (N,3)
if pts.size == 0:
    raise SystemExit(f"{PCD_PATH} has 0 points")

# Base figure with two traces:
#  - trace 0: all points
#  - trace 1: selected points (initially empty, will be updated by callback)
fig = go.Figure(data=[
    go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers",
        marker=dict(size=POINT_SIZE, color=pts[:, 2], colorscale="Viridis", opacity=0.8),
        name="all"
    ),
    go.Scatter3d(
        x=[], y=[], z=[],
        mode="markers",
        marker=dict(size=SELECTED_SIZE, color="red", opacity=0.95),
        name="selected"
    ),
])

fig.update_layout(
    scene=dict(
        xaxis=dict(title="X (m)", range=list(AX_X)),
        yaxis=dict(title="Y (m)", range=list(AX_Y)),
        zaxis=dict(title="Z (m)", range=list(AX_Z)),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1.5, z=0.8),
    ),
    dragmode="lasso",  # start in selection mode
    margin=dict(l=0, r=0, t=30, b=0),
    title=str(PCD_PATH.name),
)

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H4("LiDAR points — use lasso/box to select; rotate with left-drag, then switch back to lasso"),
    dcc.Graph(
        id="pcd-graph",
        figure=fig,
        style={"height": "88vh"},
        config={
            "scrollZoom": True,
            "displaylogo": False,
            # make sure the 3D selection tools are visible
            "modeBarButtonsToAdd": ["lasso3d", "select3d"]
        },
    ),

    # store selected indices so you can accumulate multiple selections
    dcc.Store(id="sel-store", data=[]),

    html.Div([
        html.Button("Save selected to CSV", id="save-btn", n_clicks=0),
        html.Button("Clear selection", id="clear-btn", n_clicks=0,
                    style={"marginLeft": "10px"}),
        html.Span(id="selected-count", style={"marginLeft": "16px"}),
    ], style={"margin": "8px 0"}),

    html.Div(id="save-status", style={"margin": "6px 0", "fontStyle": "italic"})
])

# --- helper: read indices from selectedData ---
def _indices_from_selectedData(selectedData):
    """Extract point indices from Plotly selectedData payload."""
    if not selectedData or "points" not in selectedData:
        return []
    # In Scatter3d, the key is 'pointNumber'
    idxs = [p.get("pointNumber") for p in selectedData["points"] if p.get("pointNumber") is not None]
    # Deduplicate and keep valid ones only
    idxs = [i for i in set(idxs) if 0 <= i < len(pts)]
    return idxs

# Update stored selection (accumulate) and highlight trace
@app.callback(
    Output("pcd-graph", "figure"),
    Output("sel-store", "data"),
    Output("selected-count", "children"),
    Input("pcd-graph", "selectedData"),
    Input("clear-btn", "n_clicks"),
    State("sel-store", "data"),
    State("pcd-graph", "figure"),
    prevent_initial_call=True
)
def update_selection(selectedData, clear_clicks, stored_idxs, fig_json):
    # Clear selection if clear button pressed
    trigger = dash.callback_context.triggered[0]["prop_id"]
    if trigger.startswith("clear-btn"):
        stored_idxs = []
    else:
        # accumulate indices from this selection
        new_idxs = _indices_from_selectedData(selectedData)
        if new_idxs:
            stored_idxs = sorted(set(stored_idxs).union(new_idxs))

    # Update the "selected" trace (index 1) with the chosen points
    if stored_idxs:
        sel_pts = pts[np.array(stored_idxs)]
        fig_json["data"][1]["x"] = sel_pts[:, 0].tolist()
        fig_json["data"][1]["y"] = sel_pts[:, 1].tolist()
        fig_json["data"][1]["z"] = sel_pts[:, 2].tolist()
        count_text = f"{len(stored_idxs)} points selected"
    else:
        fig_json["data"][1]["x"] = []
        fig_json["data"][1]["y"] = []
        fig_json["data"][1]["z"] = []
        count_text = "No points selected"

    return fig_json, stored_idxs, count_text

# Save to CSV
@app.callback(
    Output("save-status", "children"),
    Input("save-btn", "n_clicks"),
    State("sel-store", "data"),
    prevent_initial_call=True
)
def save_selected(n_clicks, stored_idxs):
    if not stored_idxs:
        return "No points selected to save."
    sel = pts[np.array(stored_idxs)]
    df = pd.DataFrame(sel, columns=["x", "y", "z"])
    df.to_csv(CSV_OUT, index=False)
    return f"Saved {len(df)} points to {CSV_OUT}"

if __name__ == "__main__":
    app.run(debug=True)   # Dash ≥2.15
