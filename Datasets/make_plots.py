import json
import os
import sys

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

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

C = {"louvain": "#d1495b", "leiden": "#00798c"}

# Figures 6+ add methods.
C.update(
    {
        "leiden_steps": C["leiden"],
        "leiden_consensus": C["leiden"],
        "louvain_split": C["louvain"],
        "louvain_consensus": C["louvain"],
        "leiden_norefine": "#4a3aa7",
        "louvain_igraph": "#c98500",
    }
)
STYLE = {
    "louvain": dict(ls="-", marker="o"),
    "leiden": dict(ls="-", marker="s"),
    "leiden_steps": dict(ls="--", marker="s", mfc="white"),
    "leiden_norefine": dict(ls="-", marker="^"),
    "louvain_split": dict(ls=":", marker="x"),
    "louvain_igraph": dict(ls="-", marker="D"),
    "louvain_consensus": dict(ls="--", marker="o", mfc="white"),
    "leiden_consensus": dict(ls="--", marker="s", mfc="white"),
}
LABEL = {
    "louvain": "Louvain (networkx)",
    "leiden": "Leiden (leidenalg)",
    "leiden_steps": "Leiden, step by step",
    "leiden_norefine": "Leiden, no refinement",
    "louvain_split": "Louvain + split",
    "louvain_igraph": "Louvain (igraph)",
    "louvain_consensus": "Louvain consensus",
    "leiden_consensus": "Leiden consensus",
}


def _line(ax, x, y, m, yerr=None, **kw):
    style = {
        "color": C[m],
        "label": LABEL[m],
        "ms": 5,
        "lw": 1.6,
        "capsize": 3 if yerr is not None else 0,
        **STYLE[m],
        **kw,
    }
    ax.errorbar(x, y, yerr=yerr, **style)


def save(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {name}")


# Fig 1: the mu cliff
def fig_lfr_mu(df):
    d = df[df.experiment == "lfr_mu"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    for metric, ax, label in [
        ("nmi", axes[0], "NMI vs ground truth"),
        ("modularity", axes[1], "Modularity Q"),
    ]:
        for m in ["louvain", "leiden"]:
            s = d[d.method == m].groupby("mu")[metric].agg(["mean", "std"])
            ax.errorbar(
                s.index,
                s["mean"],
                yerr=s["std"],
                marker="o",
                ms=4,
                capsize=3,
                lw=1.6,
                color=C[m],
                label=m.capitalize(),
            )
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

    for ax, metric, ylab in [
        (axes[0], "nmi", "NMI"),
        (axes[1], "n_communities", "communities found"),
    ]:
        data, pos, colors = [], [], []
        for i, ds in enumerate(datasets):
            for j, m in enumerate(["louvain", "leiden"]):
                data.append(d[(d.dataset == ds) & (d.method == m)][metric].values)
                pos.append(i * 3 + j)
                colors.append(C[m])
        bp = ax.boxplot(
            data,
            positions=pos,
            widths=0.7,
            patch_artist=True,
            medianprops=dict(color="black", lw=1.2),
            flierprops=dict(marker="o", ms=3, alpha=0.5),
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.65)
        ax.set_xticks([i * 3 + 0.5 for i in range(len(datasets))])
        ax.set_xticklabels(datasets, rotation=15)
        ax.set_ylabel(ylab)

    axes[0].legend(
        handles=[
            Line2D([], [], color=C[m], lw=6, alpha=0.65, label=m.capitalize())
            for m in ["louvain", "leiden"]
        ],
        frameon=False,
    )
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
            axes[0, col].plot(
                s.n_communities.mean().index,
                s.n_communities.mean().values,
                marker="o",
                ms=4,
                color=C[m],
                label=m.capitalize(),
            )
            axes[1, col].plot(
                s.nmi.mean().index,
                s.nmi.mean().values,
                marker="o",
                ms=4,
                color=C[m],
                label=m.capitalize(),
            )
        axes[0, col].axhline(
            true_k[ds], ls="--", c="0.4", lw=1.2, label=f"true k = {true_k[ds]}"
        )
        axes[0, col].set_title(ds)
        axes[0, col].set_ylabel("communities found")
        axes[1, col].set_ylabel("NMI")
        axes[1, col].set_xlabel(r"resolution $\gamma$")
        axes[0, col].legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Cora over-partitions at default $\\gamma$; "
        "email-Eu-core under-partitions at every $\\gamma$",
        y=1.0,
    )
    save(fig, "fig3_resolution.png")


# Fig 4: the clusterings
def _layout(G):
    # DRL for larger/sparser graphs, Fruchterman-Reingold for small dense ones.
    import igraph as ig

    g = ig.Graph(n=G.number_of_nodes(), edges=list(G.edges()), directed=False)
    lay = (
        g.layout_drl()
        if G.number_of_nodes() > 1500
        else g.layout_fruchterman_reingold(niter=500)
    )
    return {i: tuple(p) for i, p in enumerate(lay)}


def _draw(ax, G, pos, labels, title):
    labs = np.asarray(labels)
    uniq = [u for u in np.unique(labs) if u >= 0]
    cmap = plt.get_cmap("tab20")
    colour = {u: cmap(i % 20) for i, u in enumerate(uniq)}
    node_c = [colour.get(int(l), (0.8, 0.8, 0.8, 1.0)) for l in labs]

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.06, width=0.4)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=7, node_color=node_c, linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()


def fig_clusterings(datasets=("email-eu-core", "lfr-easy", "cora")):
    fig, axes = plt.subplots(len(datasets), 3, figsize=(11, 3.6 * len(datasets)))
    if len(datasets) == 1:
        axes = axes[None, :]

    for r, ds in enumerate(datasets):
        G, y = largest_component(*LOADERS[ds]())
        pos = _layout(G)
        lou, _ = run_method("louvain", G, seed=42)
        lei, _ = run_method("leiden", G, seed=42)

        k_true = len(np.unique(y[y >= 0]))
        _draw(axes[r, 0], G, pos, y, f"{ds} — ground truth ({k_true} communities)")
        _draw(axes[r, 1], G, pos, lou, f"Louvain ({len(np.unique(lou))} found)")
        _draw(axes[r, 2], G, pos, lei, f"Leiden ({len(np.unique(lei))} found)")

    fig.suptitle(
        "Detected communities vs ground truth "
        "(colours are arbitrary; compare the grouping, not the hue)",
        y=1.0,
        fontsize=10,
    )
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

    fig.suptitle(
        "A clean diagonal means recovery; "
        "horizontal spread means splitting, vertical means merging",
        y=1.02,
        fontsize=10,
    )
    save(fig, "fig5_contingency.png")


# Fig 6: ablation of Leiden's refinement
def fig_ablation(df):
    d = df[df.experiment == "ablation"]
    arms = ["leiden_steps", "leiden_norefine", "leiden", "louvain", "louvain_split"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))

    for r, ds in enumerate(["cora", "email-eu-core"]):
        sub = d[d.dataset == ds]
        ref = sub[sub.method == "leiden_steps"].groupby("resolution").objective.mean()

        ax = axes[r, 0]
        ax.axhline(0, color=C["leiden_steps"], ls="--", lw=1.2)
        for m in arms[1:]:
            s = sub[sub.method == m].groupby("resolution").objective
            _line(ax, s.mean().index, s.mean() - ref, m, yerr=s.std())
        ax.set_ylabel(r"objective $Q_\gamma$ minus step-by-step Leiden")
        ax.set_title(
            f"{ds}: quality vs step-by-step Leiden (dashed zero line)", fontsize=9
        )

        ax = axes[r, 1]
        for m in arms:
            s = sub[sub.method == m].groupby("resolution").nmi.mean()
            _line(ax, s.index, s.values, m)
        ax.set_ylabel("NMI")
        ax.set_title(f"{ds}: agreement with ground truth", fontsize=9)

        ax = axes[r, 2]
        runs = sub.groupby("method").disconnected_frac
        hits = [(runs.get_group(m) > 0).sum() for m in arms]
        total = runs.size()[arms[0]]
        ax.barh(range(len(arms)), hits, color=[C[m] for m in arms], height=0.6)
        for i, h in enumerate(hits):
            ax.text(h + total * 0.01, i, str(h), va="center", fontsize=8)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([LABEL[m] for m in arms], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, max(max(hits) * 1.3, 5))
        ax.set_xlabel(f"runs with a disconnected community (of {total})")
        ax.set_title(f"{ds}: disconnected communities", fontsize=9)

    for ax in axes[:, :2].ravel():
        ax.set_xlabel(r"resolution $\gamma$")
    fig.suptitle(
        "Refinement raises modularity on sparse Cora but not on dense "
        "email-Eu-core; only unrefined runs leave disconnected communities",
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(arms), frameon=False, fontsize=9
    )
    save(fig, "fig6_ablation.png")


# Fig 7: degree distributions
def fig_degree(datasets=("cora", "email-eu-core", "lfr-easy")):
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    looks = [
        dict(color="0.1", ls="-"),
        dict(color="0.45", ls="--"),
        dict(color="0.65", ls=":"),
    ]
    for ds, look in zip(datasets, looks):
        G, _ = largest_component(*LOADERS[ds]())
        deg = np.sort([d for _, d in G.degree()])
        ccdf = 1 - np.arange(len(deg)) / len(deg)
        ax.loglog(
            deg,
            ccdf,
            drawstyle="steps-post",
            lw=1.8,
            label=f"{ds}  (mean degree {deg.mean():.1f})",
            **look,
        )
    ax.set_xlabel("degree d")
    ax.set_ylabel(r"fraction of nodes with degree $\geq$ d")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Degree distributions (largest connected component)", fontsize=10)
    save(fig, "fig7_degree_dist.png")


# Fig 8: real datasets on the LFR difficulty scale
def fig_real_on_lfr(df):
    stats = pd.read_csv(os.path.join(RESULTS, "dataset_stats.csv"))
    lfr = stats[stats.mu.notna()]
    real = stats[stats.mu.isna()].set_index("dataset")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    g = lfr.groupby("mu").label_mixing_mu.agg(["mean", "min", "max"])
    ax.plot([0, 1], [0, 1], ls="--", c="0.55", lw=1, label="measured = generating")
    ax.errorbar(
        g.index,
        g["mean"],
        yerr=[g["mean"] - g["min"], g["max"] - g["mean"]],
        color="0.15",
        marker="o",
        ms=4,
        capsize=3,
        lw=1.6,
        label="networkx LFR, 3 graphs per $\\mu$",
    )
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"generating $\mu$ (what fig 1 plots)")
    ax.set_ylabel("measured label mixing")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("networkx's LFR over-shoots its mixing parameter", fontsize=10)

    ax = axes[1]
    d = df[df.experiment == "lfr_mu"].merge(
        lfr[["dataset", "label_mixing_mu"]], on="dataset"
    )
    b = df[df.experiment == "baseline"]
    for m in ["louvain", "leiden"]:
        s = (
            d[d.method == m]
            .groupby("mu")
            .agg(x=("label_mixing_mu", "mean"), y=("nmi", "mean"), e=("nmi", "std"))
        )
        _line(ax, s.x, s.y, m, yerr=s.e, label=f"{LABEL[m]} on LFR")
        for ds, mk in [("cora", "*"), ("email-eu-core", "P")]:
            ax.scatter(
                real.loc[ds, "label_mixing_mu"],
                b[(b.dataset == ds) & (b.method == m)].nmi.mean(),
                marker=mk,
                s=150,
                color=C[m],
                edgecolor="black",
                lw=0.6,
                zorder=5,
            )
    for ds, mk in [("cora", "*"), ("email-eu-core", "P")]:
        ax.scatter(
            [],
            [],
            marker=mk,
            s=110,
            color="white",
            edgecolor="black",
            label=f"{ds} (mean degree {real.loc[ds, 'avg_degree']:.0f})",
        )
    ax.set_xlabel("measured label mixing")
    ax.set_ylabel("NMI")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Real datasets placed at their measured mixing", fontsize=10)
    save(fig, "fig8_real_on_lfr.png")


# Fig 9: scaling
def fig_scaling(df):
    d = df[df.experiment == "scaling"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    for m in ["louvain", "louvain_igraph", "leiden"]:
        t = (
            d[d.method == m]
            .groupby(["n_nodes", "dataset"])
            .seconds.median()
            .groupby("n_nodes")
            .mean()
        )
        slope = np.polyfit(np.log10(t.index), np.log10(t.values), 1)[0]
        _line(ax, t.index, t.values, m, label=f"{LABEL[m]}  (slope {slope:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nodes n (LFR, $\\mu$ = 0.3, mean degree ~16)")
    ax.set_ylabel("seconds per run (median of 3 seeds)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Runtime: implementation vs algorithm", fontsize=10)

    ax = axes[1]
    for m in ["louvain", "louvain_igraph", "leiden_norefine", "leiden_steps", "leiden"]:
        s = d[d.method == m].groupby("n_nodes").disconnected_frac.mean() * 100
        _line(ax, s.index, s.values, m)
    ax.set_xscale("log")
    ax.set_xlabel("nodes n")
    ax.set_ylabel("disconnected communities (% of communities)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Disconnected communities vs graph size", fontsize=10)
    save(fig, "fig9_scaling.png")


# Fig 10: consensus clustering
def fig_consensus(df):
    d = df[df.experiment == "consensus"]
    lfr = d[d.mu.notna()]
    arms = ["louvain", "louvain_consensus", "leiden", "leiden_consensus"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    ax = axes[0, 0]
    for m in arms:
        s = lfr[lfr.method == m].groupby("mu").nmi
        _line(ax, s.mean().index, s.mean(), m, yerr=s.std())
    ax.set_xlabel(r"generating $\mu$")
    ax.set_ylabel("NMI")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("LFR test graphs (3 graphs x 3 seeds per point)", fontsize=10)

    ax = axes[0, 1]
    ax.axhline(0, color="0.3", lw=1)
    ax.axhline(0.03, color="0.5", lw=1, ls=":", label="pre-registered bar (+0.03)")
    for base in ["louvain", "leiden"]:
        key = ["dataset", "seed"]
        a = lfr[lfr.method == base].set_index(key)
        c = lfr[lfr.method == base + "_consensus"].set_index(key)
        gain = (c.nmi - a.nmi).groupby(a.mu).agg(["mean", "std"])
        _line(ax, gain.index, gain["mean"], base + "_consensus", yerr=gain["std"])
    ax.set_xlabel(r"generating $\mu$")
    ax.set_ylabel("NMI gain over the base method")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Paired gain (same graph, same seed)", fontsize=10)

    stab = pd.read_csv(os.path.join(RESULTS, "stability.csv"))
    ax = axes[1, 0]
    for i, ds in enumerate(["cora", "email-eu-core"]):
        for j, m in enumerate(arms):
            row = stab[(stab.dataset == ds) & (stab.method == m)].iloc[0]
            x = i * 5 + j
            ax.bar(
                x,
                row.seed_agreement_ari,
                yerr=row["std"],
                color=C[m],
                hatch="//" if m.endswith("consensus") else None,
                edgecolor="white",
                capsize=3,
                width=0.8,
                label=LABEL[m] if i == 0 else None,
            )
            ax.text(
                x,
                0.02,
                f"{row.seed_agreement_ari:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="white",
            )
    ax.set_xticks([1.5, 6.5])
    ax.set_xticklabels(["cora", "email-eu-core"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("seed agreement (mean pairwise ARI)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    ax.set_title("Do runs with different seeds agree? (20 seeds)", fontsize=10)

    ax = axes[1, 1]
    real = d[d.mu.isna()]
    data, pos, cols = [], [], []
    for i, ds in enumerate(["cora", "email-eu-core"]):
        for j, m in enumerate(arms):
            data.append(real[(real.dataset == ds) & (real.method == m)].nmi.values)
            pos.append(i * 5 + j)
            cols.append(C[m])
    bp = ax.boxplot(
        data,
        positions=pos,
        widths=0.7,
        patch_artist=True,
        medianprops=dict(color="black", lw=1.2),
        flierprops=dict(marker="o", ms=3, alpha=0.5),
    )
    for patch, c, m in zip(bp["boxes"], cols, arms * 2):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
        if m.endswith("consensus"):
            patch.set_hatch("//")
    ax.set_xticks([1.5, 6.5])
    ax.set_xticklabels(["cora", "email-eu-core"])
    ax.set_ylabel("NMI")
    ax.set_title("NMI over 20 seeds on real data", fontsize=10)

    with open(os.path.join(RESULTS, "consensus_config.json")) as f:
        cfg = json.load(f)
    fig.suptitle(
        f"Consensus clustering (frozen: tau = {cfg['tau']}, "
        f"n_p = {cfg['n_p']}, tuned on separate graphs)",
        y=1.0,
    )
    fig.tight_layout()
    save(fig, "fig10_consensus.png")


# Fig 11: consensus parameter sensitivity (development graphs only)
def fig_consensus_params(df):
    d = df[df.experiment == "consensus_tune"]
    with open(os.path.join(RESULTS, "consensus_config.json")) as f:
        cfg = json.load(f)
    old = cfg["nmi_rule_picked"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))

    for ax, col, cmap, label, fmt, title in [
        (
            axes[0, 0],
            "ari",
            "Blues",
            "mean ARI",
            "{:.3f}",
            "ARI (the selection criterion)",
        ),
        (
            axes[0, 1],
            "nmi",
            "Blues",
            "mean NMI",
            "{:.3f}",
            "NMI (first criterion, replaced)",
        ),
        (
            axes[1, 0],
            "n_communities",
            "Purples",
            "mean communities found",
            "{:.0f}",
            "Communities found (true: 7-13)",
        ),
        (axes[1, 1], "seconds", "Greys", "mean seconds per run", "{:.1f}", "Cost"),
    ]:
        M = d.pivot_table(index="tau", columns="n_p", values=col, aggfunc="mean")
        im = ax.imshow(M.values, cmap=cmap, aspect="auto", origin="lower")
        for (i, j), v in np.ndenumerate(M.values):
            dark = im.norm(v) > 0.6
            ax.text(
                j,
                i,
                fmt.format(v),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if dark else "black",
            )
        ax.set_xticks(range(len(M.columns)))
        ax.set_xticklabels([int(c) for c in M.columns])
        ax.set_yticks(range(len(M.index)))
        ax.set_yticklabels(M.index)
        ax.set_xlabel("runs per round n_p")
        ax.set_ylabel("threshold tau")
        ax.set_title(title, fontsize=10)
        ax.grid(False)
        for pick, look in [
            (cfg, dict(ec="#c98500", lw=2.8, ls="-")),
            (old, dict(ec="0.35", lw=1.8, ls="--")),
        ]:
            i = list(M.index).index(pick["tau"])
            j = list(M.columns).index(pick["n_p"])
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, **look))
        fig.colorbar(im, ax=ax, shrink=0.85, label=label)
    fig.suptitle(
        "Consensus settings on development graphs (mu 0.4-0.5). Solid "
        "outline: frozen setting (best ARI).\nDashed: what NMI would have "
        "picked - it wins on NMI by shattering the graph",
        y=1.01,
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, "fig11_consensus_params.png")


FIGURES = {
    "lfr_mu": fig_lfr_mu,
    "seed_variance": fig_seed_variance,
    "resolution": fig_resolution,
    "clusterings": lambda df: fig_clusterings(),
    "contingency": lambda df: fig_contingency(),
    "ablation": fig_ablation,
    "degree": lambda df: fig_degree(),
    "real_on_lfr": fig_real_on_lfr,
    "scaling": fig_scaling,
    "consensus": fig_consensus,
    "consensus_params": fig_consensus_params,
}


def main():
    wanted = sys.argv[1:] or list(FIGURES)
    os.makedirs(FIGDIR, exist_ok=True)
    df = pd.read_csv(os.path.join(RESULTS, "raw.csv"))
    print(f"loaded {len(df)} rows")

    for name in wanted:
        FIGURES[name](df)
    print(f"\nfigures in {os.path.abspath(FIGDIR)}")


if __name__ == "__main__":
    main()
