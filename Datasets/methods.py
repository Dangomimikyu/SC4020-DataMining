import random
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


def _weights(G):
    # None keeps unweighted graphs unchanged. The edge order matches igraph.
    if not nx.is_weighted(G):
        return None
    return [d["weight"] for _, _, d in G.edges(data=True)]


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
        weights=_weights(G),
    )
    return _communities_to_labels([set(c) for c in part], G.number_of_nodes())


def _leiden_steps(G, seed=0, resolution=1.0, refine=True, max_iterations=100):
    """
    Run Leiden level by level, with refinement optionally disabled.

    The steps are written out so the refinement stage can be removed without
    changing the rest of the algorithm. With refinement enabled, communities
    are split into connected sub-communities before aggregation. Without it,
    they are aggregated directly, giving a Louvain-style comparison.
    """
    g, w = _to_igraph(G), _weights(G)
    n = G.number_of_nodes()

    def blank():
        return leidenalg.RBConfigurationVertexPartition(
            g, weights=w, resolution_parameter=resolution
        )

    opt = leidenalg.Optimiser()
    opt.set_rng_seed(seed)
    labels = np.arange(n)
    for _ in range(max_iterations):
        level = blank()
        level.set_membership(labels.tolist())
        single = blank()  # singleton partition on the same graph
        node_at = np.arange(n)  # level node for each original node
        gain = 0.0
        while True:
            gain += opt.move_nodes(level)
            if len(level) == level.graph.vcount():
                break
            if refine:
                opt.merge_nodes_constrained(single, level)
                if len(single) == level.graph.vcount():
                    break
                nxt = single.aggregate_partition(level)
                node_at = np.asarray(single.membership)[node_at]
                single = single.aggregate_partition()
            else:
                nxt = level.aggregate_partition()
                node_at = np.asarray(level.membership)[node_at]
            level = nxt
        labels = np.asarray(level.membership)[node_at]
        if gain <= 1e-12:
            break
    return np.unique(labels, return_inverse=True)[1]


def leiden_steps(G, seed=0, resolution=1.0, **kw):
    """
    Run the refinement-enabled version of the level-by-level Leiden loop.
    This is used as the refined arm of the ablation.
    """
    return _leiden_steps(G, seed, resolution, refine=True)


def leiden_norefine(G, seed=0, resolution=1.0, **kw):
    """
    Run the same level-by-level loop without refinement.
    This provides a Louvain-style ablation with the same implementation.
    """
    return _leiden_steps(G, seed, resolution, refine=False)


def louvain_split(G, seed=0, resolution=1.0, **kw):
    """
    Run Louvain and split disconnected communities into components.

    This fixes disconnected communities without changing the underlying
    Louvain run. Since the components have no edges between them, splitting
    them does not reduce the modularity score.
    """
    comms = nx.community.louvain_communities(G, seed=seed, resolution=resolution)
    pieces = [cc for c in comms for cc in nx.connected_components(G.subgraph(c))]
    return _communities_to_labels(pieces, G.number_of_nodes())


def louvain_igraph(G, seed=0, resolution=1.0, **kw):
    """
    Run igraph's Louvain implementation for timing comparisons.
    python-igraph uses Python's random module, so the seed is set explicitly.
    """
    random.seed(seed)
    comms = _to_igraph(G).community_multilevel(
        weights=_weights(G), resolution=resolution
    )
    return _communities_to_labels([set(c) for c in comms], G.number_of_nodes())


def _consensus(G, base, seed=0, n_p=10, tau=0.5, max_rounds=10, resolution=1.0):
    """
    Repeat clustering on an edge co-assignment graph until it stabilizes.

    The base method is run several times. Each edge is weighted by the
    fraction of runs that place its endpoints together, weak edges are removed,
    and the method is run again on the resulting graph. The process stops when
    all runs agree or max_rounds is reached.
    """
    n = G.number_of_nodes()
    edges = np.array(G.edges(), dtype=int)
    H = G
    for rnd in range(1, max_rounds + 1):
        runs = [
            base(H, seed=seed * 10_000 + rnd * n_p + i, resolution=resolution)
            for i in range(n_p)
        ]
        together = np.mean(
            [lab[edges[:, 0]] == lab[edges[:, 1]] for lab in runs], axis=0
        )
        if np.all((together == 0) | (together == 1)):
            break

        keep = together >= tau
        H = nx.Graph()
        H.add_nodes_from(range(n))
        H.add_weighted_edges_from(
            (int(u), int(v), float(w)) for (u, v), w in zip(edges[keep], together[keep])
        )

        # Keep the strongest edge for nodes whose edges all fell below tau.
        weight = {(int(u), int(v)): float(w) for (u, v), w in zip(edges, together)}
        for node in list(nx.isolates(H)):
            nbrs = list(G[node])
            if not nbrs:
                continue
            w_of = lambda m: weight.get((node, m), weight.get((m, node)))
            best = max(nbrs, key=w_of)
            H.add_edge(node, best, weight=max(w_of(best), 1e-6))
    return runs[0], rnd


def louvain_consensus(G, seed=0, resolution=1.0, n_p=10, tau=0.5, **kw):
    return _consensus(G, louvain, seed, n_p, tau, resolution=resolution)[0]


def leiden_consensus(G, seed=0, resolution=1.0, n_p=10, tau=0.5, **kw):
    return _consensus(G, leiden, seed, n_p, tau, resolution=resolution)[0]


METHODS = {
    "louvain": louvain,
    "leiden": leiden,
    "leiden_steps": leiden_steps,
    "leiden_norefine": leiden_norefine,
    "louvain_split": louvain_split,
    "louvain_igraph": louvain_igraph,
    "louvain_consensus": louvain_consensus,
    "leiden_consensus": leiden_consensus,
}


def run_method(name, G, seed=0, **params):
    fn = METHODS[name]
    t0 = time.perf_counter()
    labels = fn(G, seed=seed, **params)
    return labels, time.perf_counter() - t0
