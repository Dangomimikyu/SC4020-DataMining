import itertools
import os

import networkx as nx
import numpy as np
import pandas as pd

from graph_datasets import LOADERS
from run_experiments import LFR_MUS, RESULTS_DIR, TEST_GRAPH_SEED, get_dataset, get_lfr

STATS_CSV = os.path.join(RESULTS_DIR, "dataset_stats.csv")


def edge_homophily(G, y):
    """Fraction of edges (between labelled nodes) whose endpoints share a class."""
    return float(np.mean([y[u] == y[v] for u, v in G.edges()
                          if y[u] >= 0 and y[v] >= 0]))


def label_mixing(G, y):
    """
    Measured mu: for each labelled node, the fraction of its labelled
    neighbours in a different class, averaged over nodes.

    This is how LFR defines its mixing parameter, so it puts a real dataset on
    the same difficulty scale as the LFR sweep (fig 1). On the LFR graphs it
    should come back close to the mu they were generated with.
    """
    vals = []
    for u in G:
        if y[u] < 0:
            continue
        nbrs = [y[v] for v in G[u] if y[v] >= 0]
        if nbrs:
            vals.append(np.mean([c != y[u] for c in nbrs]))
    return float(np.mean(vals))


def describe(name, G, y):
    n, m = G.number_of_nodes(), G.number_of_edges()
    known = y[y >= 0]
    return {
        "dataset": name,
        "n_nodes": n,
        "n_edges": m,
        "avg_degree": 2 * m / n,
        "max_degree": max(d for _, d in G.degree()),
        "density": nx.density(G),
        "avg_clustering": nx.average_clustering(G),
        "n_classes": len(np.unique(known)),
        "largest_class_frac": np.bincount(known).max() / len(known),
        "edge_homophily": edge_homophily(G, y),
        "label_mixing_mu": label_mixing(G, y),
    }


def main():
    rows = [describe(name, *get_dataset(name)) for name in LOADERS]
    for mu, g in itertools.product(LFR_MUS, range(3)):
        row = describe(f"lfr-mu{mu}-g{g}", *get_lfr(mu, TEST_GRAPH_SEED + g))
        row["mu"] = mu
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(STATS_CSV, index=False)

    real = df[df.mu.isna()].drop(columns="mu")
    print(real.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    lfr = df[df.mu.notna()]
    gap = (lfr.label_mixing_mu - lfr.mu).abs()
    print(f"\nLFR check: measured vs generating mu, max gap {gap.max():.3f} "
          f"(mean {gap.mean():.3f}) over {len(lfr)} graphs")
    print(f"wrote {os.path.abspath(STATS_CSV)}")


if __name__ == "__main__":
    main()
