import os
import pickle

import networkx as nx
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")


# helpers
def _relabel(G, y):
    """Relabel nodes to a contiguous 0..n-1 range, keeping y aligned."""
    nodes = sorted(G.nodes())
    mapping = {old: new for new, old in enumerate(nodes)}
    G2 = nx.relabel_nodes(G, mapping, copy=True)
    y2 = np.asarray([y[old] for old in nodes])
    return G2, y2


def largest_component(G, y):
    """Restrict to the largest connected component. Returns (G, y) relabelled."""
    keep = max(nx.connected_components(G), key=len)
    H = G.subgraph(keep).copy()
    y_sub = {n: y[n] for n in H.nodes()}
    return _relabel(H, y_sub)


def summarise(name, G, y):
    known = y[y >= 0]
    return (
        f"{name:14s} n={G.number_of_nodes():5d}  m={G.number_of_edges():6d}  "
        f"communities={len(np.unique(known)):3d}  "
        f"unlabelled={int((y < 0).sum()):4d}  "
        f"components={nx.number_connected_components(G):3d}"
    )


# 1. Cora citation network (Planetoid format)
def load_cora(raw_dir=RAW_DIR):
    #Load Cora from the Planetoid pickle format: 2708 papers, 7 topic classes.
    objs = {}
    for key in ["x", "y", "tx", "ty", "allx", "ally", "graph"]:
        with open(os.path.join(raw_dir, f"ind.cora.{key}"), "rb") as f:
            objs[key] = pickle.load(f, encoding="latin1")

    with open(os.path.join(raw_dir, "ind.cora.test.index")) as f:
        test_idx_reorder = [int(line.strip()) for line in f]
    test_idx_range = np.sort(test_idx_reorder)

    labels = np.vstack((objs["ally"], objs["ty"]))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    y = labels.argmax(axis=1)
    y[labels.sum(axis=1) == 0] = -1   # safety net, Cora has no unlabelled nodes

    G = nx.from_dict_of_lists(objs["graph"])
    G = nx.Graph(G)                   # drop direction and parallel edges
    G.remove_edges_from(nx.selfloop_edges(G))

    y_dict = {n: int(y[n]) for n in G.nodes()}
    return _relabel(G, y_dict)


# 2. email-Eu-core
def load_email_eu_core(raw_dir=RAW_DIR):
    # SNAP email-Eu-core. Directed in the original; symmetrised here, because
    # every method under comparison assumes an undirected graph. 
    edges = np.loadtxt(os.path.join(raw_dir, "email-Eu-core.txt"), dtype=int)
    lab = np.loadtxt(
        os.path.join(raw_dir, "email-Eu-core-department-labels.txt"), dtype=int
    )

    G = nx.Graph()
    G.add_nodes_from(lab[:, 0].tolist())
    G.add_edges_from(map(tuple, edges))
    G.remove_edges_from(nx.selfloop_edges(G))

    y_dict = {int(n): int(d) for n, d in lab}
    for n in G.nodes():
        y_dict.setdefault(int(n), -1)
    return _relabel(G, y_dict)


# 3. LFR synthetic benchmark
def load_lfr(mu, n=1000, tau1=3.0, tau2=1.5, average_degree=15, min_community=30, seed=0, max_tries=12):
    """
    LFR benchmark with mixing parameter mu (fraction of each node's edges that go outside its own community). 
    Low mu = easy, high mu = hard.
    Generator fails to converge for some parameter/seed combinations, most often at high mu, so retry with a bumped seed 
    rather than letting a sweep die halfway through.
    """
    last = None
    for attempt in range(max_tries):
        try:
            G = nx.generators.community.LFR_benchmark_graph(
                n=n, tau1=tau1, tau2=tau2, mu=mu,
                average_degree=average_degree,
                min_community=min_community,
                seed=seed + attempt * 1000,
            )
        except Exception as exc:         
            last = exc
            continue

        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))

        comms = {}
        y_dict = {}
        for node, data in G.nodes(data=True):
            key = frozenset(data["community"])
            if key not in comms:
                comms[key] = len(comms)
            y_dict[node] = comms[key]

        for _, data in G.nodes(data=True):
            data.pop("community", None)   

        return _relabel(G, y_dict)

    raise RuntimeError(f"LFR failed for mu={mu} after {max_tries} tries: {last}")


LOADERS = {
    "cora": load_cora,
    "email-eu-core": load_email_eu_core,
    "lfr-easy": lambda: load_lfr(mu=0.1, seed=42),
    "lfr-hard": lambda: load_lfr(mu=0.6, seed=42),
}


if __name__ == "__main__":
    for key, fn in LOADERS.items():
        G, y = fn()
        print(summarise(key, G, y))
        Gc, yc = largest_component(G, y)
        print(summarise(key + " (LCC)", Gc, yc))
        print()
