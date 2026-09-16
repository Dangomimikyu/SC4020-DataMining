import itertools
import os
import sys
import time

import pandas as pd

from graph_datasets import LOADERS, largest_component, load_lfr
from methods import METHODS, run_method
from metrics import evaluate

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
RAW_CSV = os.path.join(RESULTS_DIR, "raw.csv")

# Use the largest connected component everywhere. Both choices are defensible,
# but it has to be the same choice in every experiment or nothing is comparable.
USE_LCC = True

_cache = {}


def get_dataset(name):
    if name not in _cache:
        G, y = LOADERS[name]()
        _cache[name] = largest_component(G, y) if USE_LCC else (G, y)
    return _cache[name]


def record(rows, experiment, dataset, method, G, y, seed, **params):
    labels, secs = run_method(method, G, seed=seed, **params)
    row = {
        "experiment": experiment,
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "seconds": round(secs, 4),
        "resolution": params.get("resolution", 1.0),
        "n_iterations": params.get("n_iterations", -1),
        "mu": params.get("mu", float("nan")),
    }
    row.update(evaluate(G, y, labels))
    rows.append(row)
    return row



def exp_baseline(rows, n_seeds=20):
    """
    Default settings, many seeds. 
    Answers two questions at once: how well does each method do, and how much does the answer move between runs?

    Louvain's variance is the graph analogue of K-means' sensitivity to initialisation 
    """
    for dataset in ["cora", "email-eu-core", "lfr-easy", "lfr-hard"]:
        G, y = get_dataset(dataset)
        for method, seed in itertools.product(METHODS, range(n_seeds)):
            record(rows, "baseline", dataset, method, G, y, seed)


def exp_resolution(rows, n_seeds=3):
    # Resolution parameter sweep. Modularity has a built-in scale, and gamma shifts it: low gamma finds few large communities, high gamma finds many small ones. Shows neither method is parameter-free.
    gammas = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    for dataset in ["cora", "email-eu-core"]:
        G, y = get_dataset(dataset)
        for method, gamma, seed in itertools.product(METHODS, gammas, range(n_seeds)):
            record(rows, "resolution", dataset, method, G, y, seed, resolution=gamma)


def exp_lfr_mu(rows, mus=None, graph_seeds=3, n_seeds=2):
    if mus is None:
        mus = [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
    for mu in mus:
        for gs in range(graph_seeds):
            G, y = largest_component(*load_lfr(mu=mu, seed=100 + gs))
            for method, seed in itertools.product(METHODS, range(n_seeds)):
                record(rows, "lfr_mu", f"lfr-mu{mu}-g{gs}", method,
                       G, y, seed, mu=mu)


def exp_refinement(rows, n_seeds=10):
    """
    Ablation on Leiden's iteration count. leidenalg does not expose a switch to disable the refinement phase itself, so this is the closest honest proxy: n_iterations=1 stops after one pass, -1 runs to convergence. 
    Louvain in the baseline experiment serves as the genuine "no refinement guarantee" arm.
    """
    for dataset in ["cora", "email-eu-core"]:
        G, y = get_dataset(dataset)
        for n_iter, seed in itertools.product([1, 2, -1], range(n_seeds)):
            record(rows, "refinement", dataset, "leiden", G, y, seed,
                   n_iterations=n_iter)


EXPERIMENTS = {
    "baseline": exp_baseline,
    "resolution": exp_resolution,
    "lfr_mu": exp_lfr_mu,
    "refinement": exp_refinement,
}


def main():
    wanted = sys.argv[1:] or list(EXPERIMENTS)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []
    for name in wanted:
        t0 = time.perf_counter()
        print(f"running {name} ...", flush=True)
        EXPERIMENTS[name](rows)
        print(f"  {len(rows)} rows so far  ({time.perf_counter() - t0:.1f}s)")

    df = pd.DataFrame(rows)
    header = not os.path.exists(RAW_CSV)
    df.to_csv(RAW_CSV, mode="a", header=header, index=False)
    print(f"\nwrote {len(df)} rows to {os.path.abspath(RAW_CSV)}")


if __name__ == "__main__":
    main()
