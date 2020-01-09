"""Unbalanced Optimal Transport methods for roots."""

from ot.unbalanced import sinkhorn_unbalanced, sinkhorn_unbalanced2
import numpy as np
from scipy.spatial import distance_matrix

from root import Root


def unb_reg_wasserstein(root_1: Root, root_2: Root, reg=0.5, reg_m=0.5,
                        method="sinkhorn_stabilized", numItermax=1000):
    """Unbalanced Regularized Wasserstein distance."""
    sq_dist_matrix = distance_matrix(root_1.nodes, root_2.nodes)**2
    nodes_1 = root_1.nodes_weights
    nodes_2 = root_2.nodes_weights

    coupling = sinkhorn_unbalanced(
        a=nodes_1, b=nodes_2, M=sq_dist_matrix,
        reg=reg, reg_m=reg_m,
        numItermax=numItermax,
        method=method)

    cost = sinkhorn_unbalanced2(
        a=nodes_1, b=nodes_2, M=sq_dist_matrix,
        reg=reg, reg_m=reg_m,
        numItermax=numItermax,
        method=method)

    return np.sqrt(cost), coupling
