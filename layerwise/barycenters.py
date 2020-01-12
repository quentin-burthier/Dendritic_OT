"""Balanced Optimal Transport methods for roots."""
from typing import List
import warnings

import ot
import numpy as np
from scipy.spatial import distance_matrix

import plotly.graph_objects as go

from root import Root


def lw_barycenter(roots: List[Root], n_layers: int, reg=1e-3):
    """Layerwise barycenter distance between two roots."""
    if len(roots) > 2:
        raise ValueError("Barycenter of more than 2 roots is not implemented")

    layers = [root.layers(n_layers) for root in roots]

    distributions = [layer_mass(root_layers) for root_layers in layers]
    A = np.vstack(distributions).T
    A /= A.sum(0)

    vertical_bar = vertical_barycenter(A, n_layers, reg)
    layers_bar = layers_barycenters(layers)
    # for i, layer in enumerate(vertical_bar):
    #     if layer is None:
    #         print(i)

    nodes = np.vstack([np.hstack((layer[:, :2],
                                  z*np.ones((len(layer), 1)),
                                  dmuV * layer[:, -1:]))
                    #    if isinstance(layer, np.ndarray) else np.array([[0, 0, z, 0]])
                       for dmuV, z, layer in zip(vertical_bar,
                                                 np.linspace(0, -1, n_layers),
                                                 layers_bar)])

    return vertical_bar, layers_bar, nodes


def layers_barycenters(layers) -> List[np.ndarray]:
    """Layerwise barycenter."""
    barycenters = [barycenter(layer_1, layer_2)
                   for layer_1, layer_2 in zip(*layers)]
    return barycenters


def barycenter(layer_1, layer_2) -> np.ndarray:
    """Barycenter of two layers."""
    if isinstance(layer_1, np.ndarray) and isinstance(layer_2, np.ndarray):
        M = distance_matrix(layer_1[:, :2], layer_2[:, :2])**2
        marginal_1 = layer_1[:, -1].copy(order='C')
        marginal_1 /= np.sum(marginal_1)
        marginal_2 = layer_2[:, -1].copy(order='C')
        marginal_2 /= np.sum(marginal_2)
        # McCann interpolant with lambda = 1/2
        P = ot.emd(marginal_1, marginal_2, M)
        bar = [np.hstack(((layer_1[i, :2] + layer_2[j, :2])/2, P[i, j]))
               for i, j in zip(*np.nonzero(P))]
        return np.vstack(bar)

    if isinstance(layer_1, np.ndarray):
        return layer_1

    if isinstance(layer_2, np.ndarray):
        return layer_2

    warnings.warn("Empty layer")
    return None

def vertical_barycenter(A, n_points: int, reg: float):
    M = ot.utils.dist0(n_points) / n_points**2
    return ot.barycenter(A, M, reg, method="sinkhorn_stabilized")


def layer_mass(layers):
    return np.array([layer[:, -1].sum()
                     if isinstance(layer, np.ndarray) else 0
                     for layer in layers])


def plot_barycenter_nodes(nodes: np.ndarray, size_factor=2.):
    """Plots a point cloud."""
    fig = go.Figure(go.Scatter3d(
        x=nodes[:, 0],
        y=nodes[:, 1],
        z=nodes[:, 2],
        mode='markers',
        marker=dict(
            color="darkblue",
            size=size_factor * nodes[:, -1]/np.mean(nodes[:, -1]),
        ),
        name="Barycenter"
    ))
    return fig
