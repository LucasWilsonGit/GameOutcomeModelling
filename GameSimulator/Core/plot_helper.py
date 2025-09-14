import numpy as np
import plotly.graph_objects as go

class Plotter:
    def plot(self, results_matrix, x_param, y_param):
        raise NotImplementedError("Subclasses must implement this.")

class Candlestick3DPlotter(Plotter):
    """
    Base class for 3D candlestick plots using Plotly Mesh3D.
    """

    def plot(self, results_matrix, x_param, y_param, bar_sx=None, bar_sy=None, opacity=1):
        X, Y, q25s, q75s, q05s, q95s = self._flatten_results(results_matrix, x_param, y_param)

        bar_sx = bar_sx or getattr(x_param, "step_scale", 0.1)
        bar_sy = bar_sy or getattr(y_param, "step_scale", 0.1)

        self._render_candlesticks(X, Y, q25s, q75s, q05s, q95s, bar_sx, bar_sy, x_param, y_param, opacity)

    def _flatten_results(self, results_matrix, x_param, y_param):
        """
        Flatten results_matrix into arrays for plotting, preserving x_param major, y_param minor order.
        """
        q25s = np.array([results_matrix[i, j]["q25"] for i in range(len(x_param)) for j in range(len(y_param))])
        q75s = np.array([results_matrix[i, j]["q75"] for i in range(len(x_param)) for j in range(len(y_param))])
        q05s = np.array([results_matrix[i, j]["q05"] for i in range(len(x_param)) for j in range(len(y_param))])
        q95s = np.array([results_matrix[i, j]["q95"] for i in range(len(x_param)) for j in range(len(y_param))])
        X, Y = np.meshgrid(x_param.values, y_param.values, indexing="ij")
        return X.flatten(), Y.flatten(), q25s, q75s, q05s, q95s

    def _make_bar_vertices(self, x, y, z0, dz, dx, dy):
        """
        Create the vertices and faces of a candlestick bar.
        """
        x0, x1 = x, x + dx
        y0, y1 = y, y + dy
        z1 = z0
        z2 = z0 + dz

        verts = np.array([
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
            [x0, y0, z2],
            [x1, y0, z2],
            [x1, y1, z2],
            [x0, y1, z2],
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

    def _render_candlesticks(self, X, Y, q25s, q75s, q05s, q95s, bar_sx, bar_sy, x_param, y_param, opacity):
        fig = go.Figure()
        z_min, z_max = q05s.min(), q95s.max()

        for x, y, q25, q75, q05, q95 in zip(X, Y, q25s, q75s, q05s, q95s):
            height = q75 - q25
            if height <= 0:
                continue

            # Bar vertices
            x_v, y_v, z_v, i, j, k = self._make_bar_vertices(x, y, q25, height, bar_sx, bar_sy)
            intensity = (np.array(z_v) - z_min) / (z_max - z_min)
            fig.add_trace(go.Mesh3d(
                x=x_v, y=y_v, z=z_v,
                i=i, j=j, k=k,
                intensity=intensity,
                colorscale="Viridis",
                cmin=0, cmax=1,
                opacity=opacity,
                flatshading=True,
                showscale=False
            ))

            # Wick with black outline + gradient
            wick_intensity = (np.array([q05, q95]) - z_min) / (z_max - z_min)
            fig.add_trace(go.Scatter3d(
                x=[x + bar_sx/2, x + bar_sx/2],
                y=[y + bar_sy/2, y + bar_sy/2],
                z=[q05, q95],
                mode="lines",
                line=dict(width=8, color="black"),
                showlegend=False
            ))
            fig.add_trace(go.Scatter3d(
                x=[x + bar_sx/2, x + bar_sx/2],
                y=[y + bar_sy/2, y + bar_sy/2],
                z=[q05, q95],
                mode="lines",
                line=dict(width=6, color=wick_intensity, colorscale="Viridis", cmin=0, cmax=1),
                showlegend=False
            ))

        fig.update_layout(
            scene=dict(
                xaxis=dict(title=x_param.name),
                yaxis=dict(title=y_param.name),
                zaxis_title="Final Rating Distribution"
            ),
            title="3D Candlestick Plot",
            template="plotly_white"
        )
        fig.show()