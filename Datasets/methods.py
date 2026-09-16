import time

import igraph as ig
import leidenalg
import networkx as nx
import numpy as np


def _communities_to_labels(communities, n):
    labels = np.full(n, -1, dtype=int)
    for i, comm in enumerate(communities):
        for node in comm:
            labels[node] = i
    assert (labels >= 0).all(), "some node was left unassigned"
    return labels


def _to_igraph(G):
    return ig.Graph(n=G.number_of_nodes(), edges=list(G.edges()), directed=False)


def louvain(G, seed=0, resolution=1.0, **kw):
    """
    Louvain. Greedy modularity optimisation: repeatedly
    move nodes to the neighbouring community that most improves modularity,
    then collapse communities into super-nodes and repeat.

    Node visit order depends on the seed, and different orders give different
    partitions - so this is stochastic, and a single run tells you little.
    """
    comms = nx.community.louvain_communities(G, seed=seed, resolution=resolution)
    return _communities_to_labels(comms, G.number_of_nodes())


def leiden(G, seed=0, resolution=1.0, n_iterations=-1, **kw):
    """
    Leiden. Same greedy idea plus a refinement phase that
    guarantees every returned community is internally connected -- the defect
    Louvain has no protection against.

    RBConfigurationVertexPartition is used rather than ModularityVertexPartition
    because it exposes a resolution parameter; at resolution=1.0 the two are
    equivalent, so this costs nothing and enables the gamma sweep.

    n_iterations=-1 runs to convergence; 1 stops after a single pass, which
    serves as a crude ablation of "keep refining".
    """
    part = leidenalg.find_partition(
        _to_igraph(G),
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        n_iterations=n_iterations,
        seed=seed,
    )
    return _communities_to_labels([set(c) for c in part], G.number_of_nodes())


METHODS = {
    "louvain": louvain,
    "leiden": leiden,
}


def run_method(name, G, seed=0, **params):
    fn = METHODS[name]
    t0 = time.perf_counter()
    labels = fn(G, seed=seed, **params)
    return labels, time.perf_counter() - t0
