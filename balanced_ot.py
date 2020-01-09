"""Balanced Optimal Transport methods for roots."""

import ot
import numpy as np
from scipy.spatial import distance_matrix

from root import Root


def wasserstein(root_1: Root, root_2: Root):
    """Wasserstein distance between two roots."""
    sq_dist_matrix = distance_matrix(root_1.nodes, root_2.nodes)**2
    rescaled_nodes_1 = root_1.nodes_weights / root_1.nodes_weights.sum()
    rescaled_nodes_2 = root_2.nodes_weights / root_2.nodes_weights.sum()

    coupling, log = ot.emd(a=rescaled_nodes_1, b=rescaled_nodes_2,
                           M=sq_dist_matrix, log=True)
    distance = np.sqrt(log["cost"])

    return distance, coupling

def reg_wasserstein(root_1: Root, root_2: Root, reg=0.5,
                    method="sinkhorn_stabilized", numItermax=1000):
    """Regularized Wasserstein distance, computed with Sinkhorn algorithm."""
    sq_dist_matrix = distance_matrix(root_1.nodes, root_2.nodes)**2
    rescaled_nodes_1 = root_1.nodes_weights / root_1.nodes_weights.sum()
    rescaled_nodes_2 = root_2.nodes_weights / root_2.nodes_weights.sum()

    coupling = ot.sinkhorn(a=rescaled_nodes_1, b=rescaled_nodes_2, M=sq_dist_matrix,
                           reg=reg, method=method, numItermax=numItermax)
    cost = ot.sinkhorn2(a=rescaled_nodes_1, b=rescaled_nodes_2, M=sq_dist_matrix,
                        reg=reg, method=method, numItermax=numItermax)

    return np.sqrt(cost), coupling
