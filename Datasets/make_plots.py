import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from graph_datasets import LOADERS, largest_component
from methods import run_method

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
FIGDIR = os.path.join(RESULTS, "figures")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

C = {"louvain": "#d1495b", "leiden": "#00798c"}


def save(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {name}")


# Fig 1: the mu cliff
def fig_lfr_mu(df):
    d = df[df.experiment == "lfr_mu"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    for metric, ax, label in [("nmi", axes[0], "NMI vs ground truth"),
                              ("modularity", axes[1], "Modularity Q")]:
        for m in ["louvain", "leiden"]:
            s = d[d.method == m].groupby("mu")[metric].agg(["mean", "std"])
            ax.errorbar(s.index, s["mean"], yerr=s["std"], marker="o", ms=4,
                        capsize=3, lw=1.6, color=C[m], label=m.capitalize())
        ax.set_xlabel(r"mixing parameter $\mu$")
        ax.set_ylabel(label)
        ax.legend(frameon=False)

    axes[0].axvspan(0.35, 0.50, color="grey", alpha=0.15, zorder=0)
    axes[0].text(0.425, 0.75, "transition", ha="center", fontsize=8, color="0.3")
    axes[0].set_title("Recovery collapses between $\\mu=0.35$ and $0.50$")
    axes[1].set_title("Modularity stays high after recovery fails")
    save(fig, "fig1_lfr_mu.png")


# Fig 2: seed variance
def fig_seed_variance(df):
    d = df[df.experiment == "baseline"]
    datasets = ["cora", "email-eu-core", "lfr-easy", "lfr-hard"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    for ax, metric, ylab in [(axes[0], "nmi", "NMI"),
                             (axes[1], "n_communities", "communities found")]:
        data, pos, colors = [], [], []
        for i, ds in enumerate(datasets):
            for j, m in enumerate(["louvain", "leiden"]):
                data.append(d[(d.dataset == ds) & (d.method == m)][metric].values)
                pos.append(i * 3 + j)
                colors.append(C[m])
        bp = ax.boxplot(data, positions=pos, widths=0.7, patch_artist=True,
                        medianprops=dict(color="black", lw=1.2),
                        flierprops=dict(marker="o", ms=3, alpha=0.5))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.65)
        ax.set_xticks([i * 3 + 0.5 for i in range(len(datasets))])
        ax.set_xticklabels(datasets, rotation=15)
        ax.set_ylabel(ylab)

    axes[0].legend(handles=[Line2D([], [], color=C[m], lw=6, alpha=0.65,
                                   label=m.capitalize())
                            for m in ["louvain", "leiden"]], frameon=False)
    fig.suptitle("Spread over 20 random seeds", y=1.02)
    save(fig, "fig2_seed_variance.png")


# Fig 3: resolution sweep
def fig_resolution(df):
    d = df[df.experiment == "resolution"]
    datasets = ["cora", "email-eu-core"]
    true_k = {"cora": 7, "email-eu-core": 42}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)

    for col, ds in enumerate(datasets):
        sub = d[d.dataset == ds]
        for m in ["louvain", "leiden"]:
            s = sub[sub.method == m].groupby("resolution")
            axes[0, col].plot(s.n_communities.mean().index,
                              s.n_communities.mean().values,
                              marker="o", ms=4, color=C[m], label=m.capitalize())
            axes[1, col].plot(s.nmi.mean().index, s.nmi.mean().values,
                              marker="o", ms=4, color=C[m], label=m.capitalize())
        axes[0, col].axhline(true_k[ds], ls="--", c="0.4", lw=1.2,
                             label=f"true k = {true_k[ds]}")
        axes[0, col].set_title(ds)
        axes[0, col].set_ylabel("communities found")
        axes[1, col].set_ylabel("NMI")
        axes[1, col].set_xlabel(r"resolution $\gamma$")
        axes[0, col].legend(frameon=False, fontsize=8)

    fig.suptitle("Cora over-partitions at default $\\gamma$; "
                 "email-Eu-core under-partitions at every $\\gamma$", y=1.0)
    save(fig, "fig3_resolution.png")


# Fig 4: the clusterings
def _layout(G):
    # DRL for larger/sparser graphs, Fruchterman-Reingold for small dense ones.
    import igraph as ig
    g = ig.Graph(n=G.number_of_nodes(), edges=list(G.edges()), directed=False)
    lay = (g.layout_drl() if G.number_of_nodes() > 1500
           else g.layout_fruchterman_reingold(niter=500))
    return {i: tuple(p) for i, p in enumerate(lay)}


def _draw(ax, G, pos, labels, title):
    labs = np.asarray(labels)
    uniq = [u for u in np.unique(labs) if u >= 0]
    cmap = plt.get_cmap("tab20")
    colour = {u: cmap(i % 20) for i, u in enumerate(uniq)}
    node_c = [colour.get(int(l), (0.8, 0.8, 0.8, 1.0)) for l in labs]

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.06, width=0.4)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=7, node_color=node_c,
                           linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()


def fig_clusterings(datasets=("email-eu-core", "lfr-easy", "cora")):
    fig, axes = plt.subplots(len(datasets), 3,
                             figsize=(11, 3.6 * len(datasets)))
    if len(datasets) == 1:
        axes = axes[None, :]

    for r, ds in enumerate(datasets):
        G, y = largest_component(*LOADERS[ds]())
        pos = _layout(G)
        lou, _ = run_method("louvain", G, seed=42)
        lei, _ = run_method("leiden", G, seed=42)

        k_true = len(np.unique(y[y >= 0]))
        _draw(axes[r, 0], G, pos, y, f"{ds} — ground truth ({k_true} communities)")
        _draw(axes[r, 1], G, pos, lou,
              f"Louvain ({len(np.unique(lou))} found)")
        _draw(axes[r, 2], G, pos, lei,
              f"Leiden ({len(np.unique(lei))} found)")

    fig.suptitle("Detected communities vs ground truth "
                 "(colours are arbitrary; compare the grouping, not the hue)",
                 y=1.0, fontsize=10)
    save(fig, "fig4_clusterings.png")


# Fig 5: contingency heatmaps
def fig_contingency(datasets=("cora", "email-eu-core")):
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 4.4))
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        G, y = largest_component(*LOADERS[ds]())
        pred, _ = run_method("louvain", G, seed=42)
        mask = y >= 0
        yt, yp = y[mask], pred[mask]

        tu, pu = np.unique(yt), np.unique(yp)
        M = np.zeros((len(tu), len(pu)), dtype=int)
        for a, b in zip(yt, yp):
            M[np.searchsorted(tu, a), np.searchsorted(pu, b)] += 1

        # order predicted communities by their dominant true class, so any
        # block structure lines up along the diagonal
        M = M[:, np.argsort(M.argmax(axis=0), kind="stable")]

        im = ax.imshow(M, aspect="auto", cmap="magma_r")
        ax.set_xlabel("Louvain community (reordered)")
        ax.set_ylabel("true community")
        ax.set_title(f"{ds}: {len(tu)} true vs {len(pu)} found", fontsize=9)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8, label="nodes")

    fig.suptitle("A clean diagonal means recovery; "
                 "horizontal spread means splitting, vertical means merging",
                 y=1.02, fontsize=10)
    save(fig, "fig5_contingency.png")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    df = pd.read_csv(os.path.join(RESULTS, "raw.csv"))
    print(f"loaded {len(df)} rows")

    fig_lfr_mu(df)
    fig_seed_variance(df)
    fig_resolution(df)
    fig_clusterings()
    fig_contingency()
    print(f"\nfigures in {os.path.abspath(FIGDIR)}")


if __name__ == "__main__":
    main()
