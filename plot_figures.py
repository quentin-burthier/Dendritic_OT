"""Script to save plotly figure."""
from typing import List

import os
from os.path import join

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import sys
plantbox_path = os.environ["PLANTBOX_PATH"]
sys.path.append(plantbox_path)

from plantbox import RootSystem

from root import Root
from barycenters import point_cloud_barycenter
from layerwise.barycenters import lw_barycenter
from layerwise.distances import lw_wasserstein

def plot_root(param_dir: str, name: str, age: float):
    rs = RootSystem()
    rs.readParameters(join(param_dir,  f"{name}.xml"))
    rs.initialize()
    rs.simulate(age)
    pyroot = Root(rs)
    fig = pyroot.plot(draw_nodes=False)
    fig.update_layout(margin={"t": 0, "l": 0, "b": 0, "r": 0})
    fig.write_image(join(os.environ["HOME"], "reportOTDendrit",
                         "images", f"{name}_age_{age}.pdf"))


def plot_barycenters(param_dir: str, names: List[str], ages: List[float]):
    roots = []
    for name, age in zip(names, ages):
        rs = RootSystem()
        rs.readParameters(join(param_dir,  f"{name}.xml"))
        rs.initialize()
        rs.simulate(age)
        pyroot = Root(rs)
        pyroot.rescale()
        roots.append(pyroot)
    for root in roots:
        print(root.n_nodes)

    pt_bar = point_cloud_barycenter(roots)
    fig = plot_barycenter_nodes(pt_bar)
    fig.write_image(join(os.environ["HOME"], "reportOTDendrit",
                         "images", f"ptbar_{names[0]}_{names[1]}.pdf"))

    _, _, bar_nodes = lw_barycenter(roots, n_layers=100, reg=1e-1)
    fig = plot_barycenter_nodes(bar_nodes)
    fig.write_image(join(os.environ["HOME"], "reportOTDendrit",
                         "images", f"lwbar_{names[0]}_{names[1]}.pdf"))


def plot_barycenter_nodes(nodes: np.ndarray, size_factor=2.):
    """Plots a point cloud."""
    assert nodes.shape[1] == 4, nodes.shape
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
    fig.update_layout(margin={"t": 0, "l": 0, "b": 0, "r": 0})

    return fig

def plot_distance_matrix(param_dir: str, names: List[str], ages: List[float],
                         n_layers: int):
    assert len(names) == len(ages)
    roots = []
    for name, age in zip(names, ages):
        rs = RootSystem()
        rs.readParameters(join(param_dir,  f"{name}.xml"))
        rs.initialize()
        rs.simulate(age)
        pyroot = Root(rs)
        pyroot.rescale()
        roots.append(pyroot)

    n = len(roots)
    dist_matrix = np.zeros((n, n))
    for i, root_i in enumerate(roots):
        for j, root_j in enumerate(roots[:i+1]):
            dist_matrix[i, j] = lw_wasserstein(root_i, root_j, n_layers)

    cleaned_names = [" ".join(name.split("_")[:-2]) for name in names]
    #cleaned_names = ages

    heatmap = pd.DataFrame(dist_matrix, columns=cleaned_names, index=cleaned_names)
    ax = sns.heatmap(heatmap, square=True, cmap="Greens")
    ax.tick_params(axis='both', which='both', length=0)
    ax.xaxis.set_ticks_position('bottom')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.savefig(join(os.environ["HOME"], "reportOTDendrit",
                     "images", f"distance_matrix_Anagallis.pdf"),
                bbox_inches="tight")



if __name__ == "__main__":
    PARAM_DIR = join(plantbox_path, "modelparameter", "rootsystem")
    # NAME = "Zea_mays_1_Leitner_2010"
    # plot_root(PARAM_DIR, NAME, 7)

    NAMES = [
        "Anagallis_femina_Leitner_2010",
        "Anagallis_femina_Leitner_2010",
        "Anagallis_femina_Leitner_2010",
        "Anagallis_femina_Leitner_2010",
        "Anagallis_femina_Leitner_2010"

    ]
    AGES = list(range(2, 12, 2))
    # plot_barycenters(PARAM_DIR, NAMES, AGES)
    plot_distance_matrix(PARAM_DIR, NAMES, AGES, 100)
