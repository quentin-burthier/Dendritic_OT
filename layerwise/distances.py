"""Balanced Optimal Transport methods for roots."""
import warnings

import ot
import numpy as np
from scipy.spatial import distance_matrix

from root import Root


def lw_wasserstein(root_1: Root, root_2: Root, n_points: int):
    """Wasserstein distance between two roots."""
    roots = (root_1, root_2)

    layers = [root.layers(n_points) for root in roots]

    distributions = [layer_mass(root_layers) for root_layers in layers]
    A = np.vstack(distributions)
    A[0] /= A[0].sum()
    A[1] /= A[1].sum()

    layers_distance = layers_distances(layers, A, n_points)

    vertical_distance = vertical_wasserstein(A, n_points)

    return np.sqrt(vertical_distance + np.sum(layers_distance))


def layers_distances(layers, A, n_points):
    layers_distance = np.zeros(n_points)
    for i, (layer_1, layer_2) in enumerate(zip(*layers)):
        if isinstance(layer_1, np.ndarray) and isinstance(layer_2, np.ndarray):
            M = distance_matrix(layer_1[:, :2], layer_2[:, :2])**2
            marginal_1 = layer_1[:, -1].copy(order='C')
            marginal_1 /= np.sum(marginal_1)
            marginal_2 = layer_2[:, -1].copy(order='C')
            marginal_2 /= np.sum(marginal_2)
            cost = ot.emd2(a=marginal_1, b=marginal_2,
                           M=M, log=False)
            layers_distance[i] = cost * A[:, i].mean()
        else:
            warnings.warn("Empty layer")

    return layers_distance


def vertical_wasserstein(A, n_points):
    M = ot.utils.dist0(n_points) / n_points**2
    return ot.emd2(A[0], A[1], M)


def layer_mass(layers):
    return np.array([layer[:, -1].sum()
                     if isinstance(layer, np.ndarray) else 0
                     for layer in layers])
