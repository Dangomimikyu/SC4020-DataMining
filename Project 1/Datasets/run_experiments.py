import itertools
import json
import os
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from graph_datasets import LOADERS, largest_component, load_lfr
from methods import METHODS, run_method
from metrics import evaluate, labels_to_communities

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
RAW_CSV = os.path.join(RESULTS_DIR, "raw.csv")
CONSENSUS_CONFIG = os.path.join(RESULTS_DIR, "consensus_config.json")
STABILITY_CSV = os.path.join(RESULTS_DIR, "stability.csv")

# Use the largest connected component everywhere. Both choices are defensible,
# but it has to be the same choice in every experiment or nothing is comparable.
USE_LCC = True

# Shared grids, so every experiment that sweeps gamma or mu uses the same points.
GAMMAS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
LFR_MUS = [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]

_cache = {}


def get_dataset(name):
    if name not in _cache:
        G, y = LOADERS[name]()
        _cache[name] = largest_component(G, y) if USE_LCC else (G, y)
    return _cache[name]


def get_lfr(mu, graph_seed, n=1000):
    key = ("lfr", mu, graph_seed, n)
    if key not in _cache:
        _cache[key] = largest_component(*load_lfr(mu=mu, n=n, seed=graph_seed))
    return _cache[key]


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
        "tau": params.get("tau", float("nan")),
        "n_p": params.get("n_p", float("nan")),
    }
    row.update(evaluate(G, y, labels))
    # evaluate() scores modularity at gamma = 1; this is the quantity the method
    # was actually optimising, which is what the ablation compares.
    row["objective"] = nx.community.modularity(
        G, labels_to_communities(labels), resolution=row["resolution"])
    rows.append(row)
    return row, labels



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
    for dataset in ["cora", "email-eu-core"]:
        G, y = get_dataset(dataset)
        for method, gamma, seed in itertools.product(METHODS, GAMMAS, range(n_seeds)):
            record(rows, "resolution", dataset, method, G, y, seed, resolution=gamma)


def exp_lfr_mu(rows, mus=None, graph_seeds=3, n_seeds=2):
    if mus is None:
        mus = LFR_MUS
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


def exp_ablation(rows, n_seeds=10):
    """
    The real ablation of Leiden's refinement: leiden_steps vs leiden_norefine
    is the same code with one switch. louvain vs louvain_split isolates
    connectivity alone (paired: same Louvain run per seed), and leiden (C++)
    is the reference leiden_steps must stay close to. Swept over gamma because
    that is where Louvain's disconnected communities showed up.
    """
    arms = ["leiden", "leiden_steps", "leiden_norefine", "louvain", "louvain_split"]
    for dataset in ["cora", "email-eu-core"]:
        G, y = get_dataset(dataset)
        for method, gamma, seed in itertools.product(arms, GAMMAS, range(n_seeds)):
            record(rows, "ablation", dataset, method, G, y, seed, resolution=gamma)


# Consensus is tuned on development graphs and reported on test graphs.
# The test graphs are exp_lfr_mu's (graph seed 100 + g), so results line up.
DEV_GRAPH_SEED, TEST_GRAPH_SEED = 300, 100
CONSENSUS_ARMS = ["louvain", "leiden", "louvain_consensus", "leiden_consensus"]


def exp_consensus_tune(rows, n_seeds=2):
    """
    Choose tau and n_p for consensus clustering on development LFR graphs only
    (never reported), around the collapse at mu = 0.4-0.5, then freeze them in
    consensus_config.json. Tuning on the graphs or labels we report would
    flatter the method.
    """
    taus, n_ps = [0.2, 0.4, 0.6, 0.8], [5, 10, 20]
    start = len(rows)
    for mu, g in itertools.product([0.4, 0.45, 0.5], range(3)):
        G, y = get_lfr(mu, DEV_GRAPH_SEED + g)
        for tau, n_p, seed in itertools.product(taus, n_ps, range(n_seeds)):
            record(rows, "consensus_tune", f"lfr-mu{mu}-dev{g}", "louvain_consensus",
                   G, y, seed, mu=mu, tau=tau, n_p=n_p)
    choose_consensus_config(pd.DataFrame(rows[start:]))


def choose_consensus_config(tune):
    """
    Rule: best mean ARI; among settings within 0.01 of it, the smallest n_p,
    since cost grows linearly with it.

    ARI, not NMI. The first version picked by NMI and chose tau = 0.8, which
    shatters the dev graphs into ~130 communities against ~10 true ones: NMI
    rewards over-splitting (it is not corrected for chance), while ARI ranked
    that setting last. Switched on the development graphs, before any test
    result existed; the NMI choice is kept in the config for the record.
    """
    def pick(metric):
        score = tune.groupby(["tau", "n_p"])[metric].mean()
        near = score[score >= score.max() - 0.01].reset_index()
        best = near.sort_values(["n_p", metric], ascending=[True, False]).iloc[0]
        return best, score.max()

    chosen, best_ari = pick("ari")
    old, _ = pick("nmi")
    config = {"tau": float(chosen.tau), "n_p": int(chosen.n_p),
              "dev_ari": round(float(chosen.ari), 4),
              "best_dev_ari": round(float(best_ari), 4),
              "rule": "best mean dev ARI; within 0.01 of it, smallest n_p",
              "nmi_rule_picked": {"tau": float(old.tau), "n_p": int(old.n_p)}}
    with open(CONSENSUS_CONFIG, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  frozen consensus config: {config}")


def exp_consensus(rows, n_seeds_lfr=3, n_seeds_real=20):
    """
    Consensus vs its base methods with the frozen config. On the LFR test
    graphs: does it move the collapse to higher mu? On Cora/email-Eu-core over
    20 seeds: does it make runs agree with each other? Agreement is the mean
    ARI between every pair of seeds' partitions (written to stability.csv); it
    needs no labels, so it measures instability directly.
    """
    with open(CONSENSUS_CONFIG) as f:
        cfg = json.load(f)

    def params(method):
        if method.endswith("consensus"):
            return {"tau": cfg["tau"], "n_p": cfg["n_p"]}
        return {}

    for mu, g in itertools.product(LFR_MUS, range(3)):
        G, y = get_lfr(mu, TEST_GRAPH_SEED + g)
        for method, seed in itertools.product(CONSENSUS_ARMS, range(n_seeds_lfr)):
            record(rows, "consensus", f"lfr-mu{mu}-g{g}", method, G, y, seed,
                   mu=mu, **params(method))

    agreement = []
    for dataset in ["cora", "email-eu-core"]:
        G, y = get_dataset(dataset)
        for method in CONSENSUS_ARMS:
            runs = [record(rows, "consensus", dataset, method, G, y, seed,
                           **params(method))[1] for seed in range(n_seeds_real)]
            aris = [adjusted_rand_score(a, b) for a, b in itertools.combinations(runs, 2)]
            agreement.append({"dataset": dataset, "method": method,
                              "seed_agreement_ari": float(np.mean(aris)),
                              "std": float(np.std(aris)), "n_pairs": len(aris)})
    pd.DataFrame(agreement).to_csv(STABILITY_CSV, index=False)


def exp_scaling(rows, sizes=(1000, 2000, 5000, 10000, 20000, 50000), graph_seeds=2,
                n_seeds=3):
    """
    Runtime and disconnection vs graph size, at fixed mu = 0.3 and average
    degree, so edges grow linearly with n. louvain vs louvain_igraph isolates
    the implementation (same algorithm, Python vs C); louvain_igraph vs leiden
    isolates the algorithm (both compiled). leiden_steps / leiden_norefine are
    Python loops over C calls: here for disconnection vs n, not for timing.
    Each method gets one discarded warm-up run per graph.
    """
    arms = ["louvain", "louvain_igraph", "louvain_split", "leiden",
            "leiden_steps", "leiden_norefine"]
    for n, g in itertools.product(sizes, range(graph_seeds)):
        t0 = time.perf_counter()
        G, y = get_lfr(0.3, 200 + g, n=n)
        print(f"  lfr n={n} g={g}: generated in {time.perf_counter() - t0:.0f}s", flush=True)
        for method in arms:
            run_method(method, G, seed=0)
            for seed in range(n_seeds):
                record(rows, "scaling", f"lfr-n{n}-g{g}", method, G, y, seed, mu=0.3)


EXPERIMENTS = {
    "baseline": exp_baseline,
    "resolution": exp_resolution,
    "lfr_mu": exp_lfr_mu,
    "refinement": exp_refinement,
    "ablation": exp_ablation,
    "consensus_tune": exp_consensus_tune,
    "consensus": exp_consensus,
    "scaling": exp_scaling,
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
    if os.path.exists(RAW_CSV):
        # Replace the rows of the experiments just re-run and keep everything
        # else. Appending blindly duplicated rows on every re-run (inflating the
        # seed counts behind every std and boxplot) and misaligned the file
        # whenever a column was added.
        old = pd.read_csv(RAW_CSV)
        df = pd.concat([old[~old.experiment.isin(wanted)], df], ignore_index=True)
    df.to_csv(RAW_CSV, index=False)
    print(f"\n{len(rows)} new rows; {len(df)} rows in {os.path.abspath(RAW_CSV)}")


if __name__ == "__main__":
    main()
