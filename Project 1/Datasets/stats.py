import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


def unpaired(a, b):
    """
    Mann-Whitney U, two-sided, with the rank-biserial effect size: +1 means
    every value in a beats every value in b, -1 the reverse, 0 no tendency.
    """
    if np.allclose(np.r_[a, b], np.r_[a, b][0]):
        return 1.0, 0.0
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    return p, 2 * u / (len(a) * len(b)) - 1


def paired(diff):
    """Wilcoxon signed-rank on paired differences, two-sided."""
    diff = np.asarray(diff)
    if np.allclose(diff, 0):
        return 1.0
    return wilcoxon(diff).pvalue


def e1_ablation(df):
    print("E1  ablation of Leiden's refinement")
    d = df[df.experiment == "ablation"]
    for ds in ["cora", "email-eu-core"]:
        sub = d[d.dataset == ds]
        print(f"  {ds}")
        for gamma, g in sub.groupby("resolution"):
            on = g[g.method == "leiden_steps"].objective.values
            off = g[g.method == "leiden_norefine"].objective.values
            p, r = unpaired(on, off)
            print(f"    gamma={gamma:<5} objective with - without refinement "
                  f"{on.mean() - off.mean():+.5f}   p={p:.3g}  rank-biserial={r:+.2f}")
        hits = sub.groupby("method").disconnected_frac.apply(lambda s: int((s > 0).sum()))
        print(f"    runs with a disconnected community (of "
              f"{len(sub) // sub.method.nunique()}): {hits.to_dict()}")
        lo = sub[sub.method == "louvain"].set_index(["resolution", "seed"])
        sp = sub[sub.method == "louvain_split"].set_index(["resolution", "seed"])
        dn = (sp.nmi - lo.nmi)
        print(f"    louvain_split - louvain: NMI {dn.mean():+.5f} (max |diff| "
              f"{dn.abs().max():.4f}), objective never lower: "
              f"{bool(((sp.objective - lo.objective) >= -1e-12).all())}")
        st = sub[sub.method == "leiden_steps"].groupby("resolution").objective.mean()
        cp = sub[sub.method == "leiden"].groupby("resolution").objective.mean()
        print(f"    faithfulness, |leiden_steps - leiden| objective: max "
              f"{(st - cp).abs().max():.4f}")


def e2_consensus(df):
    print("\nE2  consensus clustering")
    with open(os.path.join(RESULTS, "consensus_config.json")) as f:
        print(f"  frozen config: {json.load(f)}")
    d = df[df.experiment == "consensus"]
    lfr = d[d.mu.notna()]
    key = ["dataset", "seed"]
    for base in ["louvain", "leiden"]:
        a = lfr[lfr.method == base].set_index(key)
        c = lfr[lfr.method == base + "_consensus"].set_index(key)
        in_band = a.mu.isin([0.4, 0.45, 0.5])
        for metric in ["nmi", "ari"]:
            gain = (c[metric] - a[metric])
            band = gain[in_band]
            print(f"  {base}_consensus - {base}, mu in 0.4-0.5: {metric.upper()} "
                  f"{band.mean():+.4f} (n={len(band)}, Wilcoxon p={paired(band):.3g}, "
                  f"{(band > 0).mean():.0%} of pairs improved)")
            per_mu = gain.groupby(a.mu).mean()
            print("    per mu: " + "  ".join(f"{m}:{v:+.3f}" for m, v in per_mu.items()))
        print(f"    communities found in band: {a[in_band].n_communities.mean():.1f} -> "
              f"{c[in_band].n_communities.mean():.1f} "
              f"(true {a[in_band].n_true_communities.mean():.1f})")

    stab = pd.read_csv(os.path.join(RESULTS, "stability.csv")).set_index(["dataset", "method"])
    real = d[d.mu.isna()]
    for ds in ["cora", "email-eu-core"]:
        for base in ["louvain", "leiden"]:
            s0 = stab.loc[(ds, base), "seed_agreement_ari"]
            s1 = stab.loc[(ds, base + "_consensus"), "seed_agreement_ari"]
            r = real[real.dataset == ds]
            n0, n1 = r[r.method == base].nmi, r[r.method == base + "_consensus"].nmi
            t0, t1 = r[r.method == base].seconds, r[r.method == base + "_consensus"].seconds
            p, _ = unpaired(n1.values, n0.values)
            print(f"  {ds:14s} {base:8s} seed agreement {s0:.3f} -> {s1:.3f} "
                  f"({s1 - s0:+.3f}); NMI {n0.mean():.3f}±{n0.std():.3f} -> "
                  f"{n1.mean():.3f}±{n1.std():.3f} (p={p:.3g}); "
                  f"cost x{t1.mean() / t0.mean():.0f}")


def e3_scaling(df):
    print("\nE3  scaling and fair runtime")
    d = df[df.experiment == "scaling"]
    t = (d.groupby(["method", "n_nodes", "dataset"]).seconds.median()
         .groupby(["method", "n_nodes"]).mean().unstack(0))
    for m in ["louvain", "louvain_igraph", "leiden"]:
        slope = np.polyfit(np.log10(t.index), np.log10(t[m]), 1)[0]
        print(f"  {m:15s} log-log slope {slope:.2f}   seconds: "
              + "  ".join(f"{n}:{v:.3f}" for n, v in t[m].items()))
    print("  ratios per n:  louvain/louvain_igraph "
          + " ".join(f"{v:.0f}" for v in t.louvain / t.louvain_igraph)
          + "   |   leiden/louvain_igraph "
          + " ".join(f"{v:.1f}" for v in t.leiden / t.louvain_igraph))
    disc = d.groupby(["method", "n_nodes"]).disconnected_frac.agg(
        lambda s: f"{(s > 0).sum()}/{len(s)}").unstack(0)
    print("  runs with a disconnected community, per n:")
    print("    " + disc.to_string().replace("\n", "\n    "))


def e4_datasets(df):
    print("\nE4  dataset characterisation")
    stats = pd.read_csv(os.path.join(RESULTS, "dataset_stats.csv"))
    lfr = stats[stats.mu.notna()]
    gap = (lfr.label_mixing_mu - lfr.mu).abs()
    print(f"  LFR measured vs generating mu: max gap {gap.max():.3f}, mean {gap.mean():.3f}")
    curve = df[df.experiment == "lfr_mu"].merge(
        lfr[["dataset", "label_mixing_mu"]], on="dataset")
    b = df[df.experiment == "baseline"]
    for m in ["louvain", "leiden"]:
        c = curve[curve.method == m].groupby("mu")[["label_mixing_mu", "nmi"]].mean()
        c = c.sort_values("label_mixing_mu")
        for ds in ["cora", "email-eu-core"]:
            x = stats[stats.dataset == ds].label_mixing_mu.iloc[0]
            pred = np.interp(x, c.label_mixing_mu, c.nmi)
            actual = b[(b.dataset == ds) & (b.method == m)].nmi.mean()
            print(f"  {m:8s} {ds:14s} mixing {x:.3f}: LFR curve predicts NMI {pred:.3f}, "
                  f"actual {actual:.3f} (gap {actual - pred:+.3f})")


def main():
    df = pd.read_csv(os.path.join(RESULTS, "raw.csv"))
    have = set(df.experiment)
    for name, fn in [("ablation", e1_ablation), ("consensus", e2_consensus),
                     ("scaling", e3_scaling)]:
        if name in have:
            fn(df)
        else:
            print(f"\n({name}: no rows yet)")
    e4_datasets(df)


if __name__ == "__main__":
    main()
