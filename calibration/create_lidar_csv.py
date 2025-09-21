import numpy as np
import open3d as o3d
import plotly.graph_objects as go

# Load point cloud
pcd = o3d.io.read_point_cloud("data/lidar/input (Frame 2745).pcd")
points = np.asarray(pcd.points)

# Downsample if very dense
# points = points[::10]  # take every 10th point

# Create 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=points[:,0],
    y=points[:,1],
    z=points[:,2],
    mode='markers',
    marker=dict(
        size=2,
        color=points[:,2],       # color by z for example
        colorscale='Turbo',
        opacity=0.7
    )
)])

fig.update_layout(scene=dict(
    xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
    aspectmode='data'
))

fig.show()
