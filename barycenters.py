"""Roots barycenters."""
from typing import List

import numpy as np

from root import Root
from balanced_ot import wasserstein


def point_cloud_barycenter(roots: List[Root]) -> np.ndarray:
    """Points cloud barycenter distance between two roots."""
    if len(roots) > 2:
        raise ValueError("Barycenter of more than 2 roots is not implemented")

    root_1, root_2 = roots

    _, P = wasserstein(root_1, root_2)
    barycenter_nodes = [np.hstack(((root_1.nodes[i] + root_2.nodes[j])/2, P[i, j]))
                        for i, j in zip(*np.nonzero(P))]
    return np.vstack(barycenter_nodes)
