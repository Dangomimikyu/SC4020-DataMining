import networkx as nx
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def labels_to_communities(labels):
    out = {}
    for node, lab in enumerate(labels):
        out.setdefault(int(lab), set()).add(node)
    return list(out.values())


def average_conductance(G, communities):
    total_vol = 2 * G.number_of_edges()
    vals = []
    for comm in communities:
        vol = sum(dict(G.degree(comm)).values())
        if vol == 0 or vol >= total_vol:
            continue
        cut = nx.cut_size(G, comm)
        vals.append(cut / min(vol, total_vol - vol))
    return float(np.mean(vals)) if vals else float("nan")


def disconnected_fraction(G, communities):
    """
    Fraction of communities whose induced subgraph is NOT connected.

    This is the Louvain defect Leiden was built to fix. Louvain can move a node
    out of a community and disconnect what it leaves behind, with nothing in
    the algorithm to catch it. Leiden's refinement phase guarantees connected
    communities, so this should be exactly 0 for Leiden and > 0 for Louvain.

    Not a quality score - it is a correctness check, and the cleanest
    evidence that the refinement phase does something real.
    """
    if not communities:
        return float("nan")
    bad = sum(
        1 for c in communities
        if len(c) > 1 and not nx.is_connected(G.subgraph(c))
    )
    return bad / len(communities)


def evaluate(G, y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    comms = labels_to_communities(y_pred)

    mask = y_true >= 0
    if mask.sum() > 0:
        nmi = normalized_mutual_info_score(y_true[mask], y_pred[mask])
        ari = adjusted_rand_score(y_true[mask], y_pred[mask])
    else:
        nmi = ari = float("nan")

    sizes = np.array([len(c) for c in comms])

    return {
        "nmi": float(nmi),
        "ari": float(ari),
        "modularity": float(nx.community.modularity(G, comms)),
        "conductance": average_conductance(G, comms),
        "n_communities": len(comms),
        "n_true_communities": int(len(np.unique(y_true[mask]))) if mask.sum() else -1,
        "disconnected_frac": disconnected_fraction(G, comms),
        "largest_community_frac": float(sizes.max() / sizes.sum()),
        "singleton_communities": int((sizes == 1).sum()),
    }
