"""Roots barycenters."""

import numpy as np

from root import Root
from balanced_ot import wasserstein

def point_cloud_barycenter(root_1: Root, root_2: Root):
    """Points cloud barycenter distance between two roots."""
    _, P = wasserstein(root_1, root_2)
    bar = [np.hstack(((root_1.nodes[i] + root_2.nodes[j])/2, P[i, j]))
           for i, j in zip(*np.nonzero(P))]
    return np.vstack(bar)
