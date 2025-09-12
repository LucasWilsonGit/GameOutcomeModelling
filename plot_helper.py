import numpy as np
import plotly.graph_objects as go

def make_bar_mesh(x, y, z0, dz, dx, dy):
    """
    Returns vertices, i,j,k (triangles) of a box with base at z0, height dz,
    corner at (x, y), size dx in x, dy in y direction.
    """
    x0, x1 = x, x + dx
    y0, y1 = y, y + dy
    z1 = z0
    z2 = z0 + dz

    verts = np.array([
        [x0, y0, z1],  # 0
        [x1, y0, z1],  # 1
        [x1, y1, z1],  # 2
        [x0, y1, z1],  # 3
        [x0, y0, z2],  # 4
        [x1, y0, z2],  # 5
        [x1, y1, z2],  # 6
        [x0, y1, z2],  # 7
    ])

    faces = []
    faces += [[0,1,2], [0,2,3]]
    faces += [[4,6,5], [4,7,6]]
    faces += [[0,4,5], [0,5,1]]
    faces += [[1,5,6], [1,6,2]]
    faces += [[2,6,7], [2,7,3]]
    faces += [[3,7,4], [3,4,0]]

    i, j, k = np.array(faces).T.tolist()
    return verts[:,0], verts[:,1], verts[:,2], i, j, k


def plot_with_mesh(X, Y, q25s, q75s, q05s, q95s, bar_sx=0.1, bar_sy=0.005):
    """
    3D candlestick-style plot using Plotly Mesh3D.

    Parameters
    ----------
    X, Y : array-like
        Meshgrid coordinates of bars (flattened).
    q25s, q75s : array-like
        25th and 75th percentiles (bar body).
    q05s, q95s : array-like
        5th and 95th percentiles (wicks).
    bar_sx, bar_sy : float
        Width/depth of each bar in x and y directions.
    """
    fig = go.Figure()

    # global min/max for world-space Z
    z_min = q05s.min()
    z_max = q95s.max()

    for x, y, q25, q75, q05, q95 in zip(X, Y, q25s, q75s, q05s, q95s):
        z0 = q25
        height = q75 - q25
        if height <= 0:
            continue

        # bar mesh
        x_v, y_v, z_v, i, j, k = make_bar_mesh(x, y, z0, height, bar_sx, bar_sy)

        intensity = (np.array(z_v) - z_min) / (z_max - z_min)

        fig.add_trace(go.Mesh3d(
            x=x_v, y=y_v, z=z_v,
            i=i, j=j, k=k,
            intensity=intensity,
            colorscale="Viridis",
            cmin=0, cmax=1,
            showscale=True,
            opacity=1,
            flatshading=True,
            name=f"{x:.2f},{y:.3f}"
        ))

        # wick gradient
        wick_intensity = (np.array([q05, q95]) - z_min) / (z_max - z_min)

        # black outline
        fig.add_trace(go.Scatter3d(
            x=[x+bar_sx/2, x+bar_sx/2],
            y=[y+bar_sy/2, y+bar_sy/2],
            z=[q05, q95],
            mode="lines",
            line=dict(width=8, color="black"),
            showlegend=False
        ))

        # gradient overlay
        fig.add_trace(go.Scatter3d(
            x=[x+bar_sx/2, x+bar_sx/2],
            y=[y+bar_sy/2, y+bar_sy/2],
            z=[q05, q95],
            mode="lines",
            line=dict(width=6, color=wick_intensity, colorscale="Viridis", cmin=0, cmax=1),
            showlegend=False
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Troll probability",
            yaxis_title="Loss Δ (WR penalty)",
            zaxis_title="Final Rating Distribution"
        ),
        title="25–75% quantile bodies with 5–95% wicks (Plotly Mesh3D, global Z shading)",
        template="plotly_white"
    )

    fig.show()