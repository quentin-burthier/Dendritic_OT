"""Python and NumPy roots."""

from typing import List, Tuple
import os
import sys

import numpy as np
import plotly.graph_objects as go

sys.path.append(os.environ["PLANTBOX_PATH"])
from plantbox import RootSystem, SegmentAnalyser


class Root:
    """Root.

    Wrapper for CPlantBox Rootsystem.
    """

    def __init__(self, root_system: RootSystem, rescale=False):
        self.lines = [to_array(s) for s in root_system.getPolylines()]
        self.nodes, unique_nodes_idx = np.unique(np.concatenate(self.lines),
                                                 axis=0, return_index=True)
        self.radii = np.array(root_system.getParameter("radius"))
        nodes_weights = np.repeat(self.radii, [len(line) for line in self.lines])
        self.nodes_weights = nodes_weights[unique_nodes_idx]

        self.analyser = SegmentAnalyser(root_system)
        self.plot_rescale_factor = 5 / self.radii.max()

        # Translates root so that surface level corresponds to z = 0
        z_surface = self.nodes[:, -1].max()
        self.nodes[:, -1] -= z_surface
        for line in self.lines:
            line[:, -1] -= z_surface

        self.depth = - self.nodes[:, -1].min()

        self._rescaled = False
        if rescale:
            self.rescale()

    def rescale(self):
        """Rescales vertical position between 0 and -1."""
        if not self._rescaled:
            self.nodes[:, -1] /= self.depth
            for line in self.lines:
                line[:, -1] /= self.depth
            self._rescaled = True

    def true_scale(self):
        """Scale the root as it was before rescaling"""
        if self._rescaled:
            self.nodes[:, -1] *= self.depth
            for line in self.lines:
                line[:, -1] *= self.depth
            self._rescaled = False

    def layers(self, n_layers) -> List[np.ndarray]:
        """Computes layers decomposition."""
        top_z = 0
        bot_z = -1 if self._rescaled else -self.depth

        z_grid = np.linspace(top_z, bot_z, n_layers, False)
        # layer[k]: z_k \in [z_grid[k+1], .z_grid[k]]
        # (defining z_grid[n_layers] := bot_z)
        dz = z_grid[0] - z_grid[1]

        layers: List[List[Tuple]] = [[] for _ in range(n_layers)]
        for line, radius in zip(self.lines, self.radii):
            for n_i, n_ip1 in zip(line[:-1], line[1:]):
                if n_i[2] < n_ip1[2]:
                    # n_i always higher than n_ip1 this way,
                    # we are only intersted in the segment
                    n_i, n_ip1 = n_ip1, n_i

                k_i = int((top_z - n_i[2]) // dz)
                k_ip1 = int((top_z - n_ip1[2]) // dz)

                if k_i == k_ip1: # FIXME
                    # there is a k s.t. k*dz > n_i.z > n_ip1.z > (k+1)*de
                    # the segment is extended to z_k
                    # (this prevents to have empty layers)
                    layers[k_i].append((n_i[:2], radius))

                for k, z_k in enumerate(z_grid[k_i+1:k_ip1+1]):
                    pos = (n_i[:2]
                           + (z_k - n_i[2]) / (n_i[2] - n_ip1[2])
                           * (n_i[:2] - n_ip1[:2]))

                    layers[k_i+k].append((pos, radius))

        np_layers = [None] * len(layers)
        for i, layer in enumerate(layers):
            if layer:
                positions, radii = zip(*layer)
                np_layers[i] = np.concatenate((np.vstack(positions),
                                               np.vstack(radii)),
                                              axis=1)

        return np_layers

    def vertical_distribution(self, top_z: float, bot_z: float, n_points: int):
        """Distribution of mass along vertical axis."""
        layers = self.layers(n_points)
        layers_mass = np.array([layer[:, -1].sum() if layer else 0
                                for layer in layers])
        # distribution = self.analyser.distribution("volume", top_z, bot_z, n_points, False)
        # return np.array(distribution)
        return layers_mass

    @classmethod
    def from_file(cls, path: str, age: int, rescale=False):
        root_system = RootSystem()
        root_system.readParameters(path)
        root_system.initialize()
        root_system.simulate(age)
        return cls(root_system, rescale)

    def plot(self, draw_nodes=False):
        """Computes plotly figure of the root."""
        trace = [go.Scatter3d(x=line[:, 0],
                              y=line[:, 1],
                              z=line[:, 2],
                              mode='lines',
                              line=dict(width=self.plot_rescale_factor * radius),
                              showlegend=False)
                 for line, radius in zip(self.lines, self.radii)]

        if draw_nodes:
            nodes = go.Scatter3d(
                x=self.nodes[:, 0],
                y=self.nodes[:, 1],
                z=self.nodes[:, 2],
                mode='markers',
                marker=dict(
                    color="black",
                    size=1.2,
                ),
                name="plant"
            )
            trace.append(nodes)

        fig = go.Figure(data=trace)
        fig.update_layout(scene_aspectmode='data')

        return fig

    @property
    def n_nodes(self):
        return len(self.nodes)

    @property
    def n_lines(self):
        return len(self.lines)


def to_array(root_system_iter):
    return np.vstack([np.array(elem) for elem in root_system_iter])
